"""On-demand person/object floor-anchor distances for a fixed, calibrated camera."""
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from framegrabber.acquisition import Frame, LatestFrame

LOG = logging.getLogger(__name__)
FLOOR_LABELS = {"chair", "couch", "bench", "bed", "dining table", "refrigerator", "suitcase", "potted plant"}


class FloorCalibration:
    def __init__(self, data):
        if not isinstance(data, dict) or data.get("floor_confirmed") is not True:
            raise ValueError("Confirm fixed camera, flat floor and selected objects resting on the floor")
        width, height = data.get("image_width"), data.get("image_height")
        if any(type(v) is not int or not 16 <= v <= 4096 for v in (width, height)):
            raise ValueError("Invalid calibration image size")
        source = data.get("source_id")
        if not isinstance(source, str) or not 1 <= len(source) <= 200:
            raise ValueError("Calibration requires the camera source ID")
        dimensions = [data.get("width_m"), data.get("depth_m")]
        if any(type(v) not in (float, int) or not math.isfinite(v) or not .05 <= v <= 100 for v in dimensions):
            raise ValueError("Rectangle dimensions must be between 0.05 and 100 metres")
        labels = data.get("floor_labels")
        if not isinstance(labels, list) or not labels or any(not isinstance(v, str) or v not in FLOOR_LABELS for v in labels):
            raise ValueError("Choose supported floor-object classes")
        try:
            points = np.asarray(data.get("points_px"), dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError("Four image points are required") from None
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Four finite image points are required")
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            raise ValueError("Calibration points must be inside the image")
        self.polygon = points.astype(np.float32)
        if not cv2.isContourConvex(self.polygon) or abs(cv2.contourArea(self.polygon)) < width * height * .001:
            raise ValueError("Mark a non-degenerate rectangle in perimeter order")
        w, d = dimensions
        self.transform = cv2.getPerspectiveTransform(self.polygon, np.float32([[0, 0], [w, 0], [w, d], [0, d]]))
        if not np.isfinite(self.transform).all() or np.linalg.cond(self.transform) > 1e9:
            raise ValueError("Unstable floor calibration")
        self.data = dict(image_width=width, image_height=height, source_id=source,
                         width_m=w, depth_m=d, points_px=points.tolist(),
                         floor_labels=sorted(set(labels)), floor_confirmed=True,
                         calibration_id=str(uuid.UUID(str(data["calibration_id"]))) if "calibration_id" in data else str(uuid.uuid4()))

    def matches(self, metadata):
        return all(metadata.get(key) == self.data[key] for key in ("source_id",)) and (
            metadata.get("width") == self.data["image_width"] and metadata.get("height") == self.data["image_height"])

    def project(self, point):
        if cv2.pointPolygonTest(self.polygon, tuple(map(float, point)), False) < 0:
            return None  # No extrapolation outside the measured floor region.
        value = self.transform @ np.array([*point, 1.0])
        if abs(value[2]) < 1e-9:
            return None
        xy = value[:2] / value[2]
        return xy.tolist() if np.isfinite(xy).all() else None


def box_anchor(item, width, height):
    try:
        box = np.asarray(item["bbox"], dtype=float)
        confidence = float(item["confidence"])
    except (ValueError, TypeError, KeyError):
        return None
    if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(confidence) or confidence < .5:
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height - 1):
        return None  # A clipped object base cannot be located reliably.
    return [float((x1 + x2) / 2), float(y2)]


def estimate(people, objects, metadata, calibration):
    width, height = metadata["width"], metadata["height"]
    result = dict(method="floor-anchor-homography-v1", state="uncalibrated", units=None,
                  people=[], objects=[], distances=[], accuracy_m=None,
                  lidar_used_for_geometry=False)
    if calibration is not None:
        result["state"] = "ready" if calibration.matches(metadata) else "calibration_mismatch"
        result["calibration_id"] = calibration.data["calibration_id"]
    usable = calibration if result["state"] == "ready" else None
    labels = set(usable.data["floor_labels"]) if usable else FLOOR_LABELS
    for index, person in enumerate(people[:8]):
        base = box_anchor(person, width, height)
        try:
            points = np.asarray(person.get("keypoints"), dtype=float)
        except (ValueError, TypeError):
            continue
        if base is None or points.shape != (17, 3) or not np.isfinite(points).all():
            continue
        ankles = points[[15, 16]]
        if np.any(ankles[:, 2] < .5) or np.any(ankles[:, :2] < 0):
            continue
        if np.any(ankles[:, 0] >= width) or np.any(ankles[:, 1] >= height - 1):
            continue
        # Ankles are only a floor-contact proxy; reject those far above the bbox base.
        if np.any(ankles[:, 1] < person["bbox"][3] - .2 * (person["bbox"][3] - person["bbox"][1])):
            continue
        anchor = ankles[:, :2].mean(axis=0).tolist()
        result["people"].append(dict(index=index, anchor_px=anchor, anchor_method="visible-ankle-midpoint",
                                     floor_m=usable.project(anchor) if usable else None))
    for index, obj in enumerate(objects[:16]):
        if obj.get("label") not in labels:
            continue
        anchor = box_anchor(obj, width, height)
        if anchor is not None:
            result["objects"].append(dict(index=index, label=obj["label"], anchor_px=anchor,
                                          anchor_method="bbox-base-assumed-on-floor",
                                          floor_m=usable.project(anchor) if usable else None))
    if usable:
        result["units"] = "m"
        for person in result["people"]:
            for obj in result["objects"]:
                if person["floor_m"] is not None and obj["floor_m"] is not None:
                    distance = math.dist(person["floor_m"], obj["floor_m"])
                    result["distances"].append(dict(person_index=person["index"], object_index=obj["index"],
                                                    label=obj["label"], distance_m=distance,
                                                    estimate=True, accuracy_m=None))
    return result


class DistanceModel:
    def __init__(self):
        from .inference import PoseModel
        from ultralytics import YOLO
        checkpoint = Path(os.environ.get("CSI_VIDEO_DISTANCE_MODEL", "models/yolo11n.pt"))
        if not checkpoint.is_file():
            raise ValueError("Local object checkpoint missing: CSI_VIDEO_DISTANCE_MODEL")
        pose_checkpoint = Path(os.environ.get("CSI_VIDEO_DISTANCE_POSE_MODEL", "models/yolo11n-pose.pt"))
        if not pose_checkpoint.is_file():
            raise ValueError("Local pose checkpoint missing: CSI_VIDEO_DISTANCE_POSE_MODEL")
        self.confidence = float(os.environ.get("CSI_VIDEO_DISTANCE_CONFIDENCE", ".5"))
        if not .5 <= self.confidence <= 1:
            raise ValueError("CSI_VIDEO_DISTANCE_CONFIDENCE must be between 0.5 and 1")
        self.pose = PoseModel(str(pose_checkpoint), image_size=416)
        self.objects = YOLO(str(checkpoint), task="detect")
        if self.objects.task != "detect":
            raise ValueError("Distance mode requires an object-detection checkpoint")

    def predict(self, image):
        people = self.pose.predict(image)
        output = self.objects.predict(image, conf=self.confidence, imgsz=416, device="cpu", verbose=False, max_det=32)[0]
        objects = [dict(bbox=box.xyxy[0].tolist(), confidence=float(box.conf[0]),
                        label=output.names[int(box.cls[0])]) for box in output.boxes
                   if output.names[int(box.cls[0])] in FLOOR_LABELS]
        return people, objects


def render(image, people, measurements, calibration):
    from .skeleton import geometry
    canvas = image.copy()
    cv2.putText(canvas, "[experimental] floor estimates", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 200, 240), 1, cv2.LINE_AA)
    if calibration is not None and measurements["state"] == "ready":
        cv2.polylines(canvas, [calibration.polygon.astype(np.int32)], True, (120, 180, 120), 1)
    for person in people[:8]:
        try:
            skeleton = geometry(person, image.shape[1], image.shape[0])
            for bone in skeleton["bones"]:
                a, b = (skeleton["joints"][bone[key]] for key in ("start", "end"))
                cv2.line(canvas, (int(a["x_px"]), int(a["y_px"])),
                         (int(b["x_px"]), int(b["y_px"])), (170, 220, 200), 1)
        except (ValueError, KeyError, TypeError):
            continue
    persons = {p["index"]: p for p in measurements["people"]}
    objects = {o["index"]: o for o in measurements["objects"]}
    for anchor in [*persons.values(), *objects.values()]:
        point = tuple(round(v) for v in anchor["anchor_px"])
        cv2.circle(canvas, point, 4, (60, 200, 240), -1)
        label = f"o{anchor['index']+1} {anchor['label']}" if "label" in anchor else f"p{anchor['index']+1}"
        cv2.putText(canvas, label, (point[0]+6, point[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (60, 200, 240), 1, cv2.LINE_AA)
    for pair in measurements["distances"]:
        a = tuple(round(v) for v in persons[pair["person_index"]]["anchor_px"])
        b = tuple(round(v) for v in objects[pair["object_index"]]["anchor_px"])
        cv2.line(canvas, a, b, (60, 200, 240), 1)
        cv2.putText(canvas, f"~{pair['distance_m']:.2f} m", ((a[0]+b[0])//2, (a[1]+b[1])//2),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 200, 240), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Distance preview encoding failed")
    return jpeg.tobytes()


class DistanceSession:
    """[experimental] best-effort supervisor; never runs a model in the service process."""
    def __init__(self, root, factory=None, frame_provider=None, blocked_by=None):
        self.path = Path(root) / "floor-calibration.json"
        self.factory = factory or DistanceModel
        self.frame_provider, self.blocked_by = frame_provider, blocked_by
        self.lock = threading.RLock()
        self.process_lock = threading.Lock()
        self.latest, self.stop = LatestFrame(), threading.Event()
        self.thread = self.process = self.connection = None
        self.worker_nice = None
        self.deadline = 0
        self.generation = 0
        self.output = None
        self.status, self.error = "disabled", None
        self.calibration = None
        if self.path.exists():
            try:
                self.calibration = FloorCalibration(json.loads(self.path.read_text()))
            except (ValueError, TypeError, OSError):
                LOG.warning("Stored floor calibration is invalid; recalibration required")

    def active(self):
        with self.lock:
            return not self.stop.is_set() and time.monotonic() < self.deadline

    def _blocked(self):
        try:
            return self.blocked_by() if self.blocked_by else None
        except Exception:
            return "priority_check_unavailable"  # Fail closed for experimental work.

    def enable(self, enabled):
        with self.lock:
            if self.stop.is_set():
                raise ValueError("Distance session is closed")
            if not self.active() or not enabled:
                self.generation += 1
                self.output = None
                self.status, self.error = ("waiting" if enabled else "disabled"), None
            self.deadline = time.monotonic() + 5 if enabled else 0
            if enabled and self.thread is None:
                self.thread = threading.Thread(target=self.run, name="experimental-distance", daemon=True)
                self.thread.start()

    def calibrate(self, data):
        calibration = FloorCalibration(data)
        with self.lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(calibration.data, allow_nan=False))
            temporary.replace(self.path)
            self.calibration = calibration
            self.generation += 1
            self.output = None
        return calibration.data

    def clear_calibration(self):
        with self.lock:
            self.path.unlink(missing_ok=True)
            self.calibration = None
            self.generation += 1
            self.output = None

    def submit(self, metadata, payload, image=None):
        # Standalone/test input. The main camera path never calls this method.
        if not payload and image is not None:
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError("Invalid experimental input")
            payload = encoded.tobytes()
        with self.lock:
            if self.active():
                self.latest.put(Frame(bytes(payload), dict(metadata, distance_generation=self.generation,
                                                          distance_received_monotonic_ns=time.monotonic_ns())))

    def snapshot(self):
        with self.lock:
            active = self.active()
            reason = self._blocked() if active else None
            output = self.output if (active and not reason and self.output and
                time.monotonic() - self.output[0] < 2 and
                time.monotonic_ns() - self.output[2]["distance_received_monotonic_ns"] < 2_000_000_000) else None
            status = "paused" if reason else self.status if active else "disabled"
            if status == "ready" and output is None:
                status = "waiting"
            return output, dict(enabled=active, experimental=True, status=status, paused_by=reason,
                                error=self.error, worker_pid=self.process.pid if self.process else None,
                                worker_nice=self.worker_nice, max_fps=1,
                                calibration=self.calibration.data if self.calibration else None)

    def _start_process(self):
        from .distance_process import run_worker
        with self.process_lock:
            if self.stop.is_set():
                raise RuntimeError("Distance session is closed")
            context = multiprocessing.get_context("spawn")
            nice_target = min(19, os.getpriority(os.PRIO_PROCESS, 0) + 10)
            parent, child = context.Pipe()
            process = context.Process(target=run_worker, args=(child, self.factory, nice_target), daemon=True,
                                      name="experimental-distance-inference")
            try:
                process.start()
                # Apply immediately; the child verifies it before model startup.
                os.setpriority(os.PRIO_PROCESS, process.pid, nice_target)
            except Exception:
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                parent.close()
                child.close()
                raise
            child.close()
            with self.lock:
                self.process, self.connection = process, parent
                self.worker_nice = None
            return parent

    def _stop_process(self):
        with self.process_lock:
            with self.lock:
                process, connection = self.process, self.connection
            if process is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=.5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1)
            if connection is not None:
                connection.close()
            with self.lock:
                self.process = self.connection = None
                self.worker_nice = None

    def run(self):
        version, previous, pending, next_run = 0, None, None, 0
        try:
            while not self.stop.is_set():
                reason = self._blocked()
                if not self.active() or reason:
                    self._stop_process()
                    pending = None
                    with self.lock:
                        self.output = None
                        self.status = "paused" if self.active() and reason else "disabled"
                    self.stop.wait(.05)
                    continue
                try:
                    if pending is not None:
                        connection = self.connection
                        if connection is None:
                            raise RuntimeError("Experimental worker was stopped")
                        if connection.poll(.05):
                            message = connection.recv()
                            if message["kind"] == "started":
                                with self.lock:
                                    self.worker_nice = message["nice"]
                            elif message["kind"] == "error":
                                raise RuntimeError(message["error"])
                            elif message["kind"] == "result":
                                generation, _ = pending
                                with self.lock:
                                    if generation == self.generation and self.active() and not self._blocked():
                                        self.output = time.monotonic(), message["jpeg"], message["result"]
                                        self.status, self.error = "ready", None
                                pending, next_run = None, time.monotonic() + 1
                        if pending is not None and time.monotonic() - pending[1] > 15:
                            raise TimeoutError("Experimental inference timeout")
                        continue
                    if time.monotonic() < next_run:
                        self.stop.wait(.05)
                        continue
                    with self.lock:
                        generation, calibration = self.generation, self.calibration
                    if self.frame_provider:
                        raw = self.frame_provider()
                        if raw is None:
                            self.stop.wait(.05)
                            continue
                        received, payload, metadata = raw
                        metadata = dict(metadata, distance_generation=generation,
                                        distance_received_monotonic_ns=int(received * 1e9))
                    else:
                        version, frame = self.latest.get(version, timeout=.05)
                        if frame is None:
                            continue
                        metadata, payload = frame.metadata, frame.image
                    key = (metadata.get("frame_id"), metadata.get("timestamp_ns"), generation)
                    if (key == previous or metadata["distance_generation"] != generation or
                        not 0 <= time.monotonic_ns() - metadata["distance_received_monotonic_ns"] < 2_000_000_000):
                        self.stop.wait(.05)
                        continue
                    previous = key
                    connection = self.connection or self._start_process()
                    with self.lock:
                        self.status = "loading" if self.worker_nice is None else "processing"
                    connection.send((metadata, payload, calibration.data if calibration else None))
                    pending = generation, time.monotonic()
                except Exception as exc:
                    if self.stop.is_set():
                        break
                    self._stop_process()
                    pending, previous = None, None
                    with self.lock:
                        self.output, self.status, self.error = None, "error", str(exc)[:300] or type(exc).__name__
                    self.stop.wait(1)
        finally:
            self._stop_process()

    def close(self):
        self.stop.set()
        self._stop_process()  # Also unblocks a broken or stalled worker pipe.
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                LOG.warning("Experimental distance supervisor did not stop")

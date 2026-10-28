"""Shared pose extraction and three live analysis stages, with one pending frame."""
from datetime import datetime, timezone
from itertools import combinations
import logging
import threading
import time

import cv2
import numpy as np

from framegrabber.acquisition import Frame, LatestFrame
from .fall_detection import FallDetector
from .presence import PresenceClassifier
from .posture_detection import PostureDetector
from .skeleton_generation import SkeletonGenerator

LOG = logging.getLogger(__name__)
OVERLAYS = ("presence", "posture", "skeleton")
PREVIEW_MODES = ("none",) + tuple(
    ",".join(group) for count in (1, 2) for group in combinations(OVERLAYS, count)
) + ("all",)


def normalize_mode(mode):
    if mode in ("all", "none"):
        return mode
    selected = mode.split(",")
    if not selected or any(item not in OVERLAYS for item in selected):
        raise ValueError("Unknown preview mode")
    return "all" if set(selected) == set(OVERLAYS) else ",".join(
        item for item in OVERLAYS if item in selected)


class VisionPipeline:
    def __init__(self, pose_model, monochrome=True):
        self.model, self.monochrome = pose_model, monochrome
        self.presence = PresenceClassifier()
        self.posture = PostureDetector()
        self.skeleton = SkeletonGenerator()

    def predict(self, image):
        if image.ndim == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif self.monochrome:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        people = self.model.predict(image)
        return image, people, dict(
            presence=self.presence.predict(people),
            postures=self.posture.predict(people),
            skeletons=self.skeleton.predict(people, image.shape[1], image.shape[0]),
        )


def annotate(image, people, analysis, metadata, mode="presence"):
    # Annotation is a derived preview; the original recording bytes remain intact.
    image = image.copy()
    mode = normalize_mode(mode)
    selected = OVERLAYS if mode == "all" else mode.split(",")
    if "presence" in selected:
        label = "PRESENT" if analysis["presence"]["state"] == "PRESENT" else "NOT PRESENT"
        scale = max(.45, min(.8, image.shape[1]/800))
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(image, (8, 8), (min(image.shape[1]-1, width+28), height+baseline+24), (12, 16, 18), -1)
        cv2.putText(image, label, (18, height+16), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (169, 235, 206), 2, cv2.LINE_AA)
    if "posture" in selected:
        for person, state in zip(people, analysis["postures"]):
            x1, y1, x2, y2 = map(int, person["bbox"])
            cv2.rectangle(image, (x1, y1), (x2, y2), (169, 235, 206), 2)
            label = state["posture"]
            (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
            left = max(0, min(x1, image.shape[1]-width-8))
            top = max(height+8, y1-8)
            cv2.rectangle(image, (left, top-height-5), (left+width+8, top+baseline), (12, 16, 18), -1)
            cv2.putText(image, label, (left+4, top), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (169, 235, 206), 1, cv2.LINE_AA)
    if "skeleton" in selected:
        for skeleton in analysis["skeletons"]:
            for bone in skeleton["bones"]:
                a, b = skeleton["joints"][bone["start"]], skeleton["joints"][bone["end"]]
                cv2.line(image, (int(a["x_px"]), int(a["y_px"])),
                         (int(b["x_px"]), int(b["y_px"])), (169, 235, 206), 2)
            for joint in skeleton["joints"]:
                if joint["visible"]:
                    cv2.circle(image, (int(joint["x_px"]), int(joint["y_px"])), 3, (240, 240, 240), -1)
    fall = analysis.get("fall_detection", {})
    if fall.get("enabled"):
        suspected = fall.get("state") == "SUSPECTED_FALL"
        labels = {"MONITORING": "Monitoraggio", "UNKNOWN": "Non determinabile",
                  "SUSPECTED_FALL": "POSSIBILE CADUTA"}
        color = (70, 80, 255) if suspected else (120, 212, 245)
        label = "Cadute (Experimental): " + labels.get(fall.get("state"), "In attesa")
        scale = min(.55, max(.25, (image.shape[1]-24) / 1000))
        (_, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, image.shape[0]-height-baseline-42)
        cv2.rectangle(image, (0, top), (image.shape[1], image.shape[0]-25), (12, 16, 18), -1)
        cv2.putText(image, label, (8, top+height+5), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
        if suspected:
            for person in people:
                x1, y1, x2, y2 = map(int, person["bbox"])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    timestamp = int(metadata["timestamp_ns"])
    stamp = datetime.fromtimestamp(timestamp // 1_000_000_000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    stamp += f".{timestamp % 1_000_000_000 // 1_000_000:03d} UTC"
    scale = min(.48, max(.25, image.shape[1] / 1400))
    cv2.rectangle(image, (0, image.shape[0]-24), (image.shape[1], image.shape[0]), (12, 16, 18), -1)
    cv2.putText(image, stamp, (8, image.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (235, 240, 240), 1, cv2.LINE_AA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Preview JPEG encoding failed")
    return jpeg.tobytes()


class PreviewFrames(dict):
    """Render only requested combinations, caching each for this capture."""

    def __init__(self, image, people, analysis, metadata):
        super().__init__()
        self.inputs = image, people, analysis, metadata
        self.lock = threading.Lock()

    def __missing__(self, mode):
        mode = normalize_mode(mode)
        with self.lock:
            if mode not in self:
                self[mode] = annotate(*self.inputs, mode)
            return self[mode]


class VisionWorker:
    def __init__(self, factory):
        self.factory = factory
        self.fall_detector = FallDetector()
        self.fall_enabled = False
        self.fall_generation = 0
        self.fall_lock = threading.Lock()
        self.latest = LatestFrame()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.output = None
        self.status = "loading"
        self.error = None
        self.processed = 0
        self.skipped = 0
        self.thread = threading.Thread(target=self.run, name="live-vision", daemon=True)
        self.thread.start()

    def set_fall_detection(self, enabled):
        with self.fall_lock:
            self.fall_generation += 1
            self.fall_enabled = enabled
            self.fall_detector.reset()
            with self.lock:
                self.output = None

    def submit(self, metadata, payload, image=None):
        received_wall, received_mono = time.time_ns(), time.monotonic_ns()
        self.latest.put(Frame((payload, image), dict(metadata, received_ns=received_wall,
                                                   received_monotonic_ns=received_mono)))

    def run(self):
        try:
            pipeline = self.factory()
        except Exception as exc:
            with self.lock:
                self.status, self.error = "error", str(exc)
            LOG.exception("Live model could not load")
            return
        with self.lock:
            self.status = "waiting"
        version = 0
        while not self.stop.is_set():
            next_version, frame = self.latest.get(version)
            if frame is None:
                continue
            skipped = max(0, next_version-version-1)
            version = next_version
            started = time.monotonic_ns()
            metadata = frame.metadata
            try:
                payload, image = frame.image
                if image is None:
                    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid image")
                image, people, analysis = pipeline.predict(image)
                with self.fall_lock:
                    fall_generation = self.fall_generation
                    analysis["fall_detection"] = (self.fall_detector.predict(people, analysis["postures"], metadata)
                        if self.fall_enabled else dict(enabled=False, experimental=True, state="DISABLED"))
                completed_mono = time.monotonic_ns()
                result = dict(metadata, **analysis, inference_completed_ns=time.time_ns(),
                              inference_ms=(completed_mono-started)/1e6,
                              queue_ms=(started-metadata["received_monotonic_ns"])/1e6,
                              backend_ms=(completed_mono-metadata["received_monotonic_ns"])/1e6)
                previews = PreviewFrames(image, people, analysis, metadata)
                result["backend_ms"] = (time.monotonic_ns()-metadata["received_monotonic_ns"])/1e6
                with self.fall_lock:
                    if fall_generation != self.fall_generation:
                        continue
                    with self.lock:
                        self.output = (time.monotonic(), previews, result)
                        self.processed += 1
                        self.skipped += skipped
                        self.status, self.error = "ready", None
            except Exception as exc:
                with self.lock:
                    self.status, self.error, self.output = "error", str(exc), None
                LOG.exception("Live inference failed")

    def snapshot(self):
        with self.lock:
            return self.output, dict(status=self.status, error=self.error,
                                     processed=self.processed, skipped=self.skipped)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            LOG.warning("Live inference still finishing at shutdown")

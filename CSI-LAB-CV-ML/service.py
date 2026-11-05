"""Camera recording lifecycle and visual analysis."""

import threading
import time
import uuid

from .timing import recording_manifest
from .inference import Policy, PoseModel, run_inference
from framegrabber.protocol import STATES, integer


class Service:
    def __init__(self, store, model_path=None, policy=None, stop_timeout=5):
        self.store, self.model_path = store, model_path
        self.policy, self.stop_timeout = policy or Policy(), stop_timeout
        self.server_id = str(uuid.uuid4())
        self.lifecycle_lock, self.analysis_lock = threading.RLock(), threading.Lock()
        self.active = None
        self.model = None
        self.last_camera_poll_ns = None
        self.closed = False
        self.preview_deadline = 0
        self.preview_frame = None
        self.live_vision = None
        self.distance_session = None
        self.distance_lock = threading.RLock()
        self.preview_lock = threading.Lock()
        self.clock_reference = (time.time_ns(), time.monotonic_ns())
        self.clock_unstable = False

    def observe_clock(self):
        wall, mono = time.time_ns(), time.monotonic_ns()
        if (
            abs((wall - self.clock_reference[0]) - (mono - self.clock_reference[1]))
            > 5_000_000
        ):
            self.clock_unstable = True
            if self.active:
                self.store.update(self.active, pc_clock_unstable=True)
        return self.clock_unstable


    def start(self, camera=True, title=""):
        if type(camera) is not bool:
            raise ValueError("camera_enabled deve essere booleano")
        with self.lifecycle_lock:
            if self.closed or self.active:
                raise ValueError("Una registrazione è già attiva oppure il servizio è in chiusura")
            item = self.store.create(camera, title)
            identifier = item["id"]
            if self.clock_unstable:
                self.store.update(identifier, pc_clock_unstable=True)
            self.store.update(identifier, state="RECORDING")
            self.active = identifier
            return self.store.get(identifier)

    def control(self):
        self.last_camera_poll_ns = time.time_ns()
        active = self.active
        if active:
            item = self.store.get(active)
            if item["camera_enabled"] and item["state"] == "RECORDING":
                return dict(experiment_id=active, server_id=self.server_id)
        return dict(experiment_id=None, server_id=self.server_id, preview_enabled=(
            time.monotonic() < self.preview_deadline or self.distance_requested()))

    def distance_requested(self):
        try:
            return bool(self.distance_session and self.distance_session.active())
        except Exception:
            return False  # An experimental failure must not break camera control.

    def distance_blocked_by(self):
        if self.active:
            return "recording"
        if self.analysis_lock.locked():
            return "analysis_or_export"
        if time.monotonic() < self.preview_deadline:
            return "main_preview"
        return None

    def distance_frame(self):
        # Experimental code only reads the published JPEG; it never owns this lock
        # while copying/decoding data, and skips the read if the main path is busy.
        if not self.preview_lock.acquire(blocking=False):
            return None
        try:
            raw = self.preview_frame
        finally:
            self.preview_lock.release()
        if not raw or time.monotonic() - raw[0] >= 2:
            return None
        return raw

    def set_preview(self, enabled):
        self.preview_deadline = time.monotonic() + 5 if enabled else 0
        if not enabled:
            with self.preview_lock:
                self.preview_frame = None

    def enable_live(self, factory=None, image_size=416, monochrome=True):
        from .vision import VisionPipeline, VisionWorker
        if self.live_vision is not None:
            raise ValueError("Live inference is already running")
        self.live_vision = VisionWorker(factory or (lambda: VisionPipeline(
            PoseModel(self.model_path or "", image_size=image_size), monochrome)))

    def preview_packet(self, mode="presence"):
        with self.preview_lock:
            raw = self.preview_frame
        if not raw or time.monotonic() - raw[0] >= 2:
            return None
        if self.live_vision:
            output, status = self.live_vision.snapshot()
            if output and time.monotonic() - output[0] < 2:
                return (output[0], output[1][mode], output[2]), status
            return raw, status
        return raw, dict(status="disabled", error=None)

    def latest_frame(self, metadata, payload, image=None):
        # One verified JPEG in memory; no dataset or frame history for preview.
        with self.preview_lock:
            self.preview_frame = (time.monotonic(), payload, dict(metadata))
        if self.live_vision:
            self.live_vision.submit(metadata, payload, image)

    def distance(self):
        from .distance import DistanceSession
        with self.distance_lock:
            if self.closed:
                raise ValueError("Service is closed")
            if self.distance_session is None:
                self.distance_session = DistanceSession(
                    self.store.root, frame_provider=self.distance_frame,
                    blocked_by=self.distance_blocked_by)
            return self.distance_session


    def stop(self, identifier):
        with self.lifecycle_lock:
            item = self.store.get(identifier)
            if identifier != self.active:
                if item["state"] in {"STOPPED", "ERROR", "INTERRUPTED"}:
                    return item
                raise ValueError("Esperimento non attivo")
            if item["state"] != "STOPPING":
                self.store.update(identifier, state="STOPPING", end_ns=time.time_ns())
            deadline = time.monotonic() + self.stop_timeout
            while item["camera_enabled"] and time.monotonic() < deadline:
                if self.store.get(identifier)["camera_ack"]:
                    break
                time.sleep(0.05)
            item = self.store.get(identifier)
            warning = (
                "Camera non ha confermato lo svuotamento della coda"
                if item["camera_enabled"] and not item["camera_ack"]
                else None
            )
            self.store.update(
                identifier,
                state="STOPPED",
                video_warning=warning,
            )
            self.active = None
            self.store.artifact(
                identifier, "timing", recording_manifest(self.store, identifier)
            )
            return self.store.get(identifier)

    def camera_stopped(self, identifier, telemetry):
        with self.store.lock:
            item = self.store.get(identifier)
            if item["state"] not in {"STOPPING", "STOPPED", "ERROR"}:
                raise ValueError("Arresto camera non richiesto")
            clean = {
                name: integer(telemetry.get(name, 0), name)
                for name in (
                    "captured",
                    "queue_dropped",
                    "rejected",
                    "network_retries",
                    "capture_errors",
                )
            }
            if item["camera_ack"] and clean != item["camera_telemetry"]:
                raise ValueError("Telemetria finale già registrata")
            self.store.update(identifier, camera_ack=True, camera_telemetry=clean)

    def infer(self, identifier):
        if not self.analysis_lock.acquire(blocking=False):
            raise ValueError("Analisi già in corso")
        try:
            if self.model is None:
                self.model = PoseModel(self.model_path or "")
            return run_inference(self.store, identifier, self.model, self.policy)
        finally:
            self.analysis_lock.release()


    def review(self, identifier, label_id, activity, status, reviewer, note):
        if activity not in STATES or status not in {
            "MANUALLY_VALIDATED",
            "REJECTED",
            "NEEDS_REVIEW",
        }:
            raise ValueError("Etichetta o stato non valido")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 100:
            raise ValueError("Specificare il revisore")
        if not isinstance(note, str) or not note.strip() or len(note) > 2000:
            raise ValueError("Specificare una motivazione")
        integer(label_id, "label_id")
        with self.store.lock, self.store.db:
            labels = self.store.artifact(identifier, "labels")
            if label_id >= len(labels):
                raise KeyError("Etichetta inesistente")
            before = dict(labels[label_id])
            labels[label_id].update(
                activity=activity,
                status=status,
                reviewer=reviewer,
                review_note=note,
                reviewed_ns=time.time_ns(),
            )
            # Direct SQL keeps audit and current state in the same SQLite transaction.
            from .storage import dumps

            self.store.db.execute(
                "INSERT OR REPLACE INTO artifacts VALUES(?,?,?)",
                (identifier, "labels", dumps(labels)),
            )
            self.store.db.execute(
                "INSERT INTO audit(experiment,body) VALUES(?,?)",
                (
                    identifier,
                    dumps(
                        dict(
                            timestamp_ns=time.time_ns(),
                            reviewer=reviewer,
                            note=note,
                            before=before,
                            after=labels[label_id],
                        )
                    ),
                ),
            )
            return labels[label_id]

    def close(self):
        with self.lifecycle_lock:
            if self.active:
                self.stop(self.active)
            self.closed = True
        if self.live_vision:
            self.live_vision.close()
        with self.distance_lock:
            if self.distance_session:
                self.distance_session.close()

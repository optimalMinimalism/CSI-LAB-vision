"""Acquisizione separata dalla rete: coda limitata, retry idempotenti e contatori."""

import logging
import queue
import threading
import time
import urllib.error
from urllib.parse import urlsplit

import cv2

from .protocol import encode_frame, measure_clock, request

LOG = logging.getLogger(__name__)
PREVIEW = "__preview__"


class CameraService:
    def __init__(
        self,
        base,
        token,
        device=0,
        width=640,
        height=480,
        fps=10,
        capacity=8,
        capture_uncertainty_ns=None,
        source_factory=None,
    ):
        if (
            not 1 <= fps <= 120
            or not 1 <= capacity <= 256
            or not 1 <= width <= 4096
            or not 1 <= height <= 2160
        ):
            raise ValueError("Configurazione camera non valida")
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL server non valido")
        self.base, self.token, self.device = base, token, device
        self.width, self.height, self.fps = width, height, fps
        self.capture_uncertainty_ns = capture_uncertainty_ns
        self.factory = source_factory or cv2.VideoCapture
        self.queue = queue.Queue(capacity)
        self.preview_pending = None
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        self.active, self.clock, self.sequence = None, None, 0
        self.telemetry = {}
        self.thread = None
        self.last_error = None
        self.capture_source = None
        self.capture_failure = None

    def capture(self):
        source = None
        try:
            while not self.stop_event.is_set():
                started = time.monotonic()
                with self.lock:
                    active, clock = self.active, self.clock
                if not active:
                    if source is not None:
                        source.release()
                        source = None
                    self.stop_event.wait(0.05)
                    continue
                if source is None:
                    source = self.factory(self.device)
                    self.capture_source = source
                    source.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    source.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    source.set(cv2.CAP_PROP_FPS, self.fps)
                    source.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                native = None
                if hasattr(source, "read_packet"):
                    try:
                        native, payload = source.read_packet()
                        acquired, monotonic = (
                            native["timestamp_ns"],
                            native["monotonic_ns"],
                        )
                        ok = True
                    except (EOFError, OSError, ValueError):
                        ok = False
                else:
                    ok, image = source.read()
                    acquired = (
                        time.time_ns()
                    )  # Immediately after read; NOT claimed as sensor exposure time.
                    monotonic = time.monotonic_ns()
                with self.lock:
                    if active != self.active:
                        continue
                    if not ok:
                        self.telemetry[active]["capture_errors"] += 1
                        source.release()
                        source = None
                        self.stop_event.wait(0.5)
                        continue
                    sequence = self.sequence
                    self.sequence += 1
                    self.telemetry[active]["captured"] += 1
                    if native is not None:
                        metadata = dict(
                            native,
                            frame_id=sequence,
                            clock=clock,
                            capture_uncertainty_ns=self.capture_uncertainty_ns,
                        )
                    else:
                        metadata, payload = encode_frame(
                            image,
                            sequence,
                            acquired,
                            clock,
                            self.capture_uncertainty_ns,
                        )
                        metadata["monotonic_ns"] = monotonic
                        if getattr(source, "synthetic", False):
                            metadata.update(
                                timestamp_source="synthetic-camera-interface",
                                simulated=True,
                            )
                    metadata.setdefault("source_id", str(self.device))
                    if active == PREVIEW:
                        if self.preview_pending is not None:
                            self.telemetry[active]["queue_dropped"] += 1
                        self.preview_pending = (active, metadata, payload)
                    else:
                        try:
                            self.queue.put_nowait((active, metadata, payload))
                        except queue.Full:
                            self.telemetry[active]["queue_dropped"] += 1
                self.stop_event.wait(
                    max(0, 1 / self.fps - (time.monotonic() - started))
                )
        except Exception as exc:
            self.capture_failure = exc
            LOG.exception("Acquisizione interrotta")
            self.stop_event.set()
        finally:
            if source is not None:
                source.release()

    def run(self):
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()
        pending, draining, control_at, clock_at = None, set(), 0, 0
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now >= control_at:
                    try:
                        control = request(self.base, self.token, "/api/camera/control")
                        desired = control.get("experiment_id") or (PREVIEW if control.get("preview_enabled") else None)
                        with self.lock:
                            old = self.active
                        if desired != old:
                            clock = (
                                measure_clock(self.base, self.token)
                                if desired
                                else None
                            )
                            with self.lock:
                                if old and old != PREVIEW:
                                    draining.add(old)
                                if old == PREVIEW:
                                    self.telemetry.pop(PREVIEW, None)
                                    self.preview_pending = None
                                    pending = None
                                    while not self.queue.empty():
                                        try:
                                            self.queue.get_nowait()
                                        except queue.Empty:
                                            break
                                self.active, self.clock, self.sequence = (
                                    desired,
                                    clock,
                                    0,
                                )
                                if desired:
                                    self.telemetry[desired] = dict(
                                        captured=0,
                                        queue_dropped=0,
                                        rejected=0,
                                        network_retries=0,
                                        capture_errors=0,
                                    )
                            clock_at = now + 10
                        elif desired and now >= clock_at:
                            clock = measure_clock(self.base, self.token)
                            with self.lock:
                                self.clock = clock
                            clock_at = now + 10
                        control_at = now + 0.5
                    except Exception as exc:
                        self.last_error = type(exc).__name__
                        LOG.warning(
                            "Controllo camera non disponibile: %s", self.last_error
                        )
                        control_at = now + 1
                if pending is None:
                    try:
                        pending = self.queue.get(timeout=0.01)
                    except queue.Empty:
                        with self.lock:
                            pending = self.preview_pending
                            self.preview_pending = None
                if pending:
                    experiment, metadata, payload = pending
                    if experiment == PREVIEW:
                        try:
                            request(self.base, self.token, "/api/camera/preview", "POST", payload=payload, metadata=metadata)
                        except (OSError, ValueError):
                            pass  # Preview never retries stale frames or blocks saved recording.
                        pending = None
                    else:
                        try:
                            request(
                                self.base,
                                self.token,
                                f"/api/experiments/{experiment}/frames",
                                "POST",
                                payload=payload,
                                metadata=metadata,
                            )
                            pending = None
                        except urllib.error.HTTPError as exc:
                            if exc.code in {400, 404, 409, 413, 422}:
                                with self.lock:
                                    self.telemetry[experiment]["rejected"] += 1
                                pending = None
                            else:
                                with self.lock:
                                    self.telemetry[experiment]["network_retries"] += 1
                                self.stop_event.wait(0.25)
                        except (OSError, ValueError):
                            with self.lock:
                                self.telemetry[experiment]["network_retries"] += 1
                            self.stop_event.wait(0.25)
                for experiment in list(draining):
                    with self.queue.mutex:
                        queued = any(
                            entry[0] == experiment for entry in self.queue.queue
                        )
                    if (pending and pending[0] == experiment) or queued:
                        continue
                    try:
                        with self.lock:
                            telemetry = dict(self.telemetry[experiment])
                        request(
                            self.base,
                            self.token,
                            f"/api/experiments/{experiment}/cameraStopped",
                            "POST",
                            telemetry,
                        )
                        draining.remove(experiment)
                        with self.lock:
                            self.telemetry.pop(experiment, None)
                    except (OSError, ValueError):
                        pass
        finally:
            self.stop_event.set()
            self.thread.join(timeout=3)
            if self.capture_failure:
                raise RuntimeError(
                    "Acquisizione camera interrotta"
                ) from self.capture_failure

    def stop(self):
        self.stop_event.set()
        # Killing the isolated native worker also unblocks a stalled USB read.
        if self.capture_source and hasattr(self.capture_source, "read_packet"):
            self.capture_source.release()

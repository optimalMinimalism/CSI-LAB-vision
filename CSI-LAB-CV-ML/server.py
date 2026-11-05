"""HTTP locale/LAN senza dipendenze web; autenticazione Bearer su ogni dato."""

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .timing import recording_manifest
from .export import export_dataset, export_video
from framegrabber.protocol import MAX_PAYLOAD, integer, verify_frame

LOG = logging.getLogger(__name__)


class Receiver(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service, token):
        if len(token) < 16:
            raise ValueError("CSI_VIDEO_TOKEN deve contenere almeno 16 caratteri")
        self.service, self.token = service, token
        self.slots = threading.BoundedSemaphore(24)
        self.stream_slots = threading.BoundedSemaphore(4)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "CSI-Video/0.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, fmt, *args):
        # No credentials, bodies or query strings in logs.
        LOG.info("HTTP %s", args[1] if len(args) > 1 else "request")

    def send(self, status, body, content_type="application/json", headers=None):
        if (
            content_type == "application/json"
            and self.headers.get("X-Timestamp-Format") == "string"
        ):

            def exact(value):
                if isinstance(value, list):
                    return [exact(v) for v in value]
                if isinstance(value, dict):
                    return {
                        k: str(v) if k.endswith("_ns") and type(v) is int else exact(v)
                        for k, v in value.items()
                    }
                return value

            body = exact(body)
        payload = (
            json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            if content_type == "application/json"
            else body
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(payload)

    def body(self, raw=False):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Transfer-Encoding non supportato")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= (MAX_PAYLOAD if raw else 1_048_576):
            raise ValueError("Richiesta troppo grande")
        expected = "image/jpeg" if raw else "application/json"
        if self.headers.get("Content-Type", "").split(";")[0] != expected:
            raise ValueError("Content-Type non valido")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("Richiesta troncata")
        if raw:
            return payload
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Oggetto JSON richiesto")
        return value

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def dispatch(self, method):
        try:
            self.route(method)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except KeyError as exc:
            self.send(404, dict(error=str(exc)))
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            self.send(400, dict(error=str(exc)[:300]))
        except Exception:
            LOG.exception("Errore interno")
            self.send(500, dict(error="Errore interno; consultare il log del servizio"))

    def stream_preview(self, service, mode):
        if not self.server.stream_slots.acquire(blocking=False):
            self.send(503, dict(error="Too many preview streams"))
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=csi-frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            previous = None
            last_frame = time.monotonic()
            while not service.closed:
                service.set_preview(True)
                packet = service.preview_packet(mode)
                if packet and "presence" in packet[0][2]:
                    frame, status = packet
                    if frame[0] != previous:
                        previous = frame[0]
                        last_frame = time.monotonic()
                        metadata = frame[2]
                        headers = ("--csi-frame\r\nContent-Type: image/jpeg\r\n"
                                   f"Content-Length: {len(frame[1])}\r\n"
                                   f"X-Frame-Id: {metadata['frame_id']}\r\n"
                                   f"X-Timestamp-Ns: {metadata['timestamp_ns']}\r\n\r\n")
                        self.wfile.write(headers.encode("ascii"))
                        self.wfile.write(frame[1])
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                # Latest-only delivery: slow readers skip frames rather than queue them.
                if time.monotonic()-last_frame > 15:
                    break
                time.sleep(.1)
        except (OSError, TimeoutError):
            pass
        finally:
            self.server.stream_slots.release()

    def route(self, method):
        received = time.time_ns()
        path = urlsplit(self.path)
        service, store = self.server.service, self.server.service.store
        if method == "GET" and path.path == "/":
            self.send(
                200,
                Path(__file__).with_name("index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if method == "GET" and path.path == "/distance":
            self.send(200, Path(__file__).with_name("distance.html").read_bytes(), "text/html; charset=utf-8")
            return
        auth = self.headers.get("Authorization", "")
        if not hmac.compare_digest(
            auth.encode(), ("Bearer " + self.server.token).encode()
        ):
            self.send(401, dict(error="Accesso richiesto"))
            return
        unstable = service.observe_clock()
        parts = path.path.strip("/").split("/")
        if parts[:2] == ["api", "recordings"]:
            parts[1] = "experiments"
        if parts[:3] == ["api", "experimental", "distance"]:
            from .distance_api import route
            route(self, method, parts, service)
            return
        if method == "GET" and parts == ["api", "clock"]:
            self.send(
                200,
                dict(
                    receive_ns=received,
                    send_ns=time.time_ns(),
                    server_id=service.server_id,
                    clock_unstable=unstable,
                ),
            )
        elif method == "GET" and parts == ["api", "camera", "control"]:
            self.send(200, service.control())
        elif method == "GET" and parts == ["api", "camera", "stream"]:
            from .vision import normalize_mode
            mode = normalize_mode(parse_qs(path.query).get("mode", ["all"])[0])
            if service.live_vision is None:
                self.send(503, dict(error="Live inference is disabled"))
                return
            self.stream_preview(service, mode)
        elif parts == ["api", "camera", "fall-detection"] and method in ("GET", "POST"):
            worker = service.live_vision
            if method == "POST":
                enabled = self.body().get("enabled")
                if type(enabled) is not bool:
                    raise ValueError("enabled must be boolean")
                if worker is None:
                    raise ValueError("Live inference is disabled")
                worker.set_fall_detection(enabled)
            self.send(200, dict(enabled=bool(worker and worker.fall_enabled), experimental=True))
        elif parts == ["api", "camera", "preview"]:
            if method == "POST" and self.headers.get("Content-Type", "").startswith("image/jpeg"):
                metadata = json.loads(self.headers.get("X-Frame-Metadata", "{}"))
                payload = self.body(raw=True)
                image = verify_frame(metadata, payload)
                if time.monotonic() < service.preview_deadline or service.distance_requested():
                    service.latest_frame(metadata, payload, image)
                self.send(200, dict(accepted=True))
            elif method == "POST":
                enabled = self.body().get("enabled")
                if type(enabled) is not bool:
                    raise ValueError("enabled must be boolean")
                service.set_preview(enabled)
                self.send(200, dict(enabled=enabled))
            elif method == "GET":
                mode = parse_qs(path.query).get("mode", ["presence"])[0]
                from .vision import normalize_mode
                mode = normalize_mode(mode)
                packet = service.preview_packet(mode)
                if packet:
                    frame, status = packet
                    source = frame[2]
                    metadata = {name: source.get(name) for name in (
                        "frame_id", "width", "height", "source_id", "timestamp_source")}
                    for name in ("timestamp_ns", "monotonic_ns", "received_ns", "inference_completed_ns"):
                        if source.get(name) is not None:
                            metadata[name] = str(source[name])
                    summary = dict(status=status["status"], error=status.get("error"))
                    if "presence" in source:
                        summary.update(fall_detection=source.get("fall_detection"),
                                       presence=source["presence"],
                                       postures=source["postures"][:10],
                                       skeleton_count=len(source["skeletons"]),
                                       inference_ms=source["inference_ms"],
                                       backend_ms=source["backend_ms"])
                    elif summary["status"] == "ready":
                        summary["status"] = "waiting"
                    metadata["analysis"] = summary
                    metadata["preview_mode"] = mode
                    self.send(200, frame[1], "image/jpeg", headers={
                        "X-Frame-Metadata": json.dumps(metadata, separators=(",", ":"), ensure_ascii=True)})
                else:
                    self.send(204, b"", "image/jpeg")
            else:
                raise ValueError("Unsupported preview method")
        elif method == "GET" and parts == ["api", "camera", "analysis"]:
            packet = service.preview_packet()
            if service.live_vision:
                _, status = service.live_vision.snapshot()
                result = packet[0][2] if packet and "presence" in packet[0][2] else None
                self.send(200, dict(**status, result=result))
            else:
                self.send(200, dict(status="disabled", result=None))
        elif method == "GET" and parts == ["api", "health"]:
            self.send(
                200,
                dict(
                    adapter="passive",
                    active=service.active,
                    last_camera_poll_ns=service.last_camera_poll_ns,
                    server_id=service.server_id,
                ),
            )
        elif parts == ["api", "experiments"]:
            if method == "GET":
                self.send(200, store.list())
            else:
                body = self.body()
                self.send(
                    201,
                    service.start(
                        body.get("camera_enabled", True), body.get("title", "")
                    ),
                )
        elif len(parts) >= 3 and parts[:2] == ["api", "experiments"]:
            identifier = parts[2]
            item = store.get(identifier)
            action = parts[3] if len(parts) > 3 else ""
            if method == "GET" and len(parts) == 3:
                self.send(
                    200,
                    dict(
                        metadata=item,
                        frames=len(store.frames(identifier)),
                        labels=store.artifact(identifier, "labels"),
                    ),
                )
            elif action == "frames" and method == "GET" and len(parts) == 5:
                sequence = int(parts[4])
                integer(sequence, "frame_id")
                _, payload = store.frame(identifier, sequence)
                self.send(200, payload, "image/jpeg")
            elif action == "frames" and method == "GET" and len(parts) == 4:
                query = parse_qs(path.query)
                offset = integer(int(query.get("offset", ["0"])[0]), "offset")
                limit = integer(int(query.get("limit", ["100"])[0]), "limit", 1, 1000)
                frames = store.frames(identifier)
                self.send(
                    200, dict(total=len(frames), frames=frames[offset : offset + limit])
                )
            elif action == "frames" and method == "POST" and len(parts) == 4:
                header = self.headers.get("X-Frame-Metadata", "")
                if len(header) > 12000:
                    raise ValueError("Metadati troppo grandi")
                metadata = json.loads(header)
                if not isinstance(metadata, dict):
                    raise ValueError("Metadati non validi")
                payload = self.body(raw=True)
                result = store.receive(identifier, metadata, payload, service.server_id)
                service.latest_frame(metadata, payload)
                self.send(200, result)
            elif len(parts) != 4:
                self.send(404, dict(error="Percorso inesistente"))
            elif action == "stop" and method == "POST":
                self.body()
                self.send(200, service.stop(identifier))
            elif action == "cameraStopped" and method == "POST":
                service.camera_stopped(identifier, self.body())
                self.send(200, dict(accepted=True))
            elif action == "inference" and method == "POST":
                self.body()
                self.send(200, service.infer(identifier))
            elif action == "review" and method == "POST":
                body = self.body()
                self.send(
                    200,
                    service.review(
                        identifier,
                        body["label_id"],
                        body["activity"],
                        body["status"],
                        body["reviewer"],
                        body["note"],
                    ),
                )
            elif (
                action in {"skeletons", "predictions", "audit"}
                and method == "GET"
            ):
                if action == "audit":
                    self.send(200, store.audit(identifier))
                else:
                    values = store.artifact(identifier, action)
                    query = parse_qs(path.query)
                    if action == "skeletons" and "frame_id" in query:
                        frame_id = integer(int(query["frame_id"][0]), "frame_id")
                        values = [v for v in values if v["frame_id"] == frame_id]
                    self.send(200, values)
            elif action == "timing" and method == "GET":
                self.send(200, recording_manifest(store, identifier))
            elif action == "video" and method == "GET":
                mode = parse_qs(path.query).get("mode", ["all"])[0]
                if mode != "raw":
                    from .vision import normalize_mode
                    mode = normalize_mode(mode)
                if not service.analysis_lock.acquire(blocking=False):
                    raise ValueError("Esportazione video già in corso")
                try:
                    with tempfile.TemporaryFile() as output:
                        pipeline = None
                        if mode not in {"raw", "none"}:
                            if service.live_vision is not None:
                                pipeline = service.live_vision.factory()
                            else:
                                from .vision import VisionPipeline
                                from .inference import PoseModel
                                pipeline = VisionPipeline(PoseModel(service.model_path or ""))
                        export_video(store, identifier, output, pipeline=pipeline, mode=mode)
                        size = output.tell()
                        output.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", "video/mp4")
                        self.send_header("Content-Length", str(size))
                        self.send_header("Content-Disposition", f'attachment; filename="video-{identifier}.mp4"')
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        while block := output.read(65536):
                            self.wfile.write(block)
                finally:
                    service.analysis_lock.release()
            elif action == "export" and method == "GET":
                # One analysis/export at a time: avoids RAM/CPU amplification and inconsistent review snapshots.
                if not service.analysis_lock.acquire(blocking=False):
                    raise ValueError("Analisi o export già in corso")
                try:
                    with tempfile.TemporaryFile() as output:
                        with store.lock:
                            export_dataset(store, identifier, output)
                        size = output.tell()
                        output.seek(0)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header("Content-Length", str(size))
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header(
                            "Content-Disposition",
                            f'attachment; filename="experiment-{identifier}.zip"',
                        )
                        self.end_headers()
                        while block := output.read(65536):
                            self.wfile.write(block)
                finally:
                    service.analysis_lock.release()
            else:
                self.send(404, dict(error="Percorso inesistente"))
        else:
            self.send(404, dict(error="Percorso inesistente"))

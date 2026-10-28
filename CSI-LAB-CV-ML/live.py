"""Local and HTTP live detection, sharing acquisition and frame representation."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
import threading
import time

import cv2

from framegrabber.acquisition import Acquisition, Frame, LatestFrame
from framegrabber.transport import transmit
from framegrabber.protocol import MAX_PAYLOAD, verify_frame


class Detector:
    def __init__(self, checkpoint, confidence=.25, size=640, device="cpu"):
        from ultralytics import YOLO
        import os
        import torch
        threads = int(os.environ.get("CSI_VIDEO_INFERENCE_THREADS", "2"))
        if not 1 <= threads <= 64:
            raise ValueError("CSI_VIDEO_INFERENCE_THREADS must be between 1 and 64")
        torch.set_num_threads(threads)
        if not Path(checkpoint).is_file():
            raise ValueError(f"Download pretrained weights first: {checkpoint}")
        self.model = YOLO(checkpoint, task="detect")
        if self.model.task != "detect":
            raise ValueError("A detection checkpoint is required")
        self.options = dict(conf=confidence, imgsz=size, device=device, verbose=False)

    def predict(self, image):
        result = self.model.predict(image, **self.options)[0]
        detections = [dict(bbox=box.xyxy[0].tolist(), confidence=float(box.conf[0]),
                           class_id=int(box.cls[0]), label=result.names[int(box.cls[0])])
                      for box in result.boxes]
        return detections, result.plot()


def receiver(address, token, latest):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(401)
                return
            if self.path != "/frame":
                self.send_error(404)
                return
            try:
                received = time.time_ns()
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_PAYLOAD:
                    raise ValueError("Invalid payload length")
                metadata = json.loads(self.headers["X-Frame-Metadata"])
                payload = self.rfile.read(length)
                image = verify_frame(metadata, payload)
                latest.put(Frame(image, dict(metadata, received_ns=received)))
            except (ValueError, KeyError, TypeError, OSError):
                self.send_error(400)
                return
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def log_message(self, *_):
            pass

    return HTTPServer(address, Handler)


def process(latest, model, headless=False, limit=0, output=None):
    version, count = 0, 0
    while not limit or count < limit:
        version, frame = latest.get(version)
        if frame is None:
            continue
        detections, annotated = model.predict(frame.image)
        result = dict(frame.metadata, inference_completed_ns=time.time_ns(), detections=detections)
        if output:
            output.write(json.dumps(result) + "\n")
            output.flush()
        count += 1
        if not headless:
            cv2.imshow("YOLO live (q to quit)", annotated)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["local", "camera", "server"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--backend", choices=["opencv", "picamera2"], default="opencv")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--server", default="http://127.0.0.1:8766")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--results", help="JSONL detections and original metadata")
    args = parser.parse_args()
    import os
    token = os.environ.get("CSI_VIDEO_TOKEN", "")
    if args.mode != "local" and len(token) < 16:
        parser.error("Set CSI_VIDEO_TOKEN to at least 16 characters")
    if not 0 < args.confidence <= 1 or args.size < 32 or args.frames < 0:
        parser.error("Invalid inference settings")
    logging.basicConfig(level=logging.INFO)
    import signal
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    acquisition = server = output = None
    try:
        model = None if args.mode == "camera" else Detector(
            args.model, args.confidence, args.size, args.device)
        if args.mode in {"local", "camera"}:
            source = int(args.source) if args.source.isdecimal() else args.source
            acquisition = Acquisition(source, args.width, args.height, args.fps, args.backend).start()
            latest = acquisition.latest
        else:
            latest = LatestFrame()
            server = receiver((args.host, args.port), token, latest)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            logging.info("Live backend listening on %s", server.server_address)
        if args.mode == "camera":
            transmit(acquisition, args.server, token)
        else:
            if args.results:
                output = open(args.results, "w")
            process(latest, model, args.headless, args.frames, output)
    except KeyboardInterrupt:
        pass
    finally:
        if acquisition:
            acquisition.close()
        if server:
            server.shutdown()
            server.server_close()
        if output:
            output.close()
        if not args.headless and args.mode != "camera":
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

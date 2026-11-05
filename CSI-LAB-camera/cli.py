"""Standalone live camera client."""
import argparse
import logging
import os
import signal

from .acquisition import Acquisition
from .transport import transmit
from .benewake import add_arguments, from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["live", "recording"], default="live",
                        help="live: YOLO backend; recording: browser preview and video recording")
    parser.add_argument("--source", default=os.environ.get("CSI_VIDEO_DEVICE", "0"))
    parser.add_argument("--backend", choices=["opencv", "picamera2"], default=os.environ.get("CSI_VIDEO_CAPTURE_BACKEND", "opencv"))
    parser.add_argument("--width", type=int, default=int(os.environ.get("CSI_VIDEO_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("CSI_VIDEO_HEIGHT", "480")))
    parser.add_argument("--fps", type=float, default=float(os.environ.get("CSI_VIDEO_FPS", "10")))
    parser.add_argument("--server", default=os.environ.get("CSI_VIDEO_SERVER", "http://127.0.0.1:8766"))
    add_arguments(parser)
    args = parser.parse_args()
    token = os.environ.get("CSI_VIDEO_TOKEN", "")
    if len(token) < 16:
        parser.error("Set CSI_VIDEO_TOKEN to at least 16 characters")
    logging.basicConfig(level=logging.INFO)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    source = int(args.source) if args.source.isdecimal() else args.source
    try:
        rangefinder = from_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        if rangefinder is not None:
            rangefinder.start()
        run(args, source, token, rangefinder)
    finally:
        if rangefinder is not None:
            rangefinder.close()


def run(args, source, token, rangefinder):
    if args.mode == "recording":
        from .camera import CameraService
        capture = CameraService(args.server, token, source, args.width, args.height, args.fps,
                                rangefinder=rangefinder)
        if args.backend == "picamera2":
            from .acquisition import PiCapture
            capture.factory = lambda device: PiCapture(device, args.width, args.height, args.fps)
        try:
            capture.run()
        except KeyboardInterrupt:
            pass
        finally:
            capture.stop()
        return
    capture = Acquisition(source, args.width, args.height, args.fps, args.backend,
                          rangefinder=rangefinder)
    try:
        capture.start()
        transmit(capture, args.server, token)
    except KeyboardInterrupt:
        pass
    finally:
        capture.close()

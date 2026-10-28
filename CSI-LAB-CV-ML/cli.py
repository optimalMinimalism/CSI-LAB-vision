import argparse
import logging
import os
from pathlib import Path
import signal
import threading

from framegrabber.camera import CameraService
from .inference import Policy
from .server import Receiver
from .service import Service
from .storage import Store


def load_env(path):
    if path is None:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if (
            not sep
            or not key.replace("_", "").isalnum()
            or not key.startswith("CSI_VIDEO_")
        ):
            raise ValueError("Riga .env non valida")
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser(
        description="Camera frame recording and visual analysis"
    )
    parser.add_argument(
        "--env", help="File di configurazione KEY=value, senza esecuzione shell"
    )
    parser.add_argument("command", choices=["server", "camera"])
    args = parser.parse_args()
    load_env(args.env)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    env = lambda name, default=None: os.environ.get("CSI_VIDEO_" + name, default)
    token = env("TOKEN", "")
    if len(token) < 16:
        parser.error("Impostare CSI_VIDEO_TOKEN con almeno 16 caratteri")
    if args.command == "camera":
        device = env("DEVICE", "0")
        uncertainty = env("CAPTURE_UNCERTAINTY_MS", "")
        camera = CameraService(
            env("SERVER", "http://127.0.0.1:8765"),
            token,
            int(device) if device.isdecimal() else device,
            int(env("WIDTH", "640")),
            int(env("HEIGHT", "480")),
            float(env("FPS", "10")),
            int(env("QUEUE", "8")),
            int(float(uncertainty) * 1e6) if uncertainty else None,
        )
        if env("NATIVE_CAPTURE"):
            from framegrabber.native import NativeCapture

            camera.factory = lambda device: NativeCapture(
                env("NATIVE_CAPTURE"), device, camera.width, camera.height, camera.fps
            )
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: camera.stop())
        camera.run()
    else:
        policy = Policy(
            confidence=float(env("AUTO_CONFIDENCE", ".9")),
            max_sync_ns=int(float(env("MAX_SYNC_MS", "20")) * 1e6),
            min_frames=int(env("MIN_CONSISTENT_FRAMES", "3")),
            calibrated_postures=env("CALIBRATED_POSTURES", "false").lower() == "true",
            stationary_camera=env("STATIONARY_CAMERA", "false").lower() == "true",
        )
        store = Store(env("DATA", "./data"))
        service = Service(
            store, env("MODEL", "./models/yolo11n-pose.pt"), policy
        )
        if env("LIVE_INFERENCE", "true").lower() == "true":
            service.enable_live(image_size=int(env("LIVE_IMAGE_SIZE", "416")),
                                monochrome=env("MONOCHROME", "true").lower() == "true")
        server = Receiver(
            (env("HOST", "127.0.0.1"), int(env("PORT", "8765"))), service, token
        )

        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, stop)
        logging.info("Receiver su %s:%s", *server.server_address)
        try:
            server.serve_forever(poll_interval=0.2)
        finally:
            service.close()
            server.server_close()
            store.close()


if __name__ == "__main__":
    main()

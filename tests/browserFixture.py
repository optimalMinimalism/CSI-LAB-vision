"""Synthetic camera server with deterministic visual inference."""

import json
import os
from pathlib import Path
import signal
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framegrabber.camera import CameraService
from backend.server import Receiver
from backend.service import Service
from backend.storage import Store
from testSystem import DeterministicModel, FakeCamera
import cv2
import numpy as np


class PreviewCamera(FakeCamera):
    synthetic = True

    def read(self):
        image = np.full((240, 320, 3), 230, np.uint8)
        for a, b in [
            ((130, 30), (130, 80)),
            ((130, 80), (130, 165)),
            ((100, 60), (160, 60)),
        ]:
            cv2.line(image, a, b, (40, 40, 40), 6)
        cv2.circle(image, (130, 15), 10, (40, 40, 40), 2)
        return True, image


class PreviewModel(DeterministicModel):
    def predict(self, image):
        people = super().predict(image)
        for person in people:
            person["bbox"] = [
                v + 80 if i % 2 == 0 else v for i, v in enumerate(person["bbox"])
            ]
            for point in person["keypoints"]:
                point[0] += 80
        return people


store = Store(os.environ["CSI_VIDEO_TEST_DATA"])
service = Service(store)
service.model = PreviewModel()
from backend.vision import VisionPipeline
service.enable_live(lambda: VisionPipeline(service.model))
server = Receiver(("127.0.0.1", 0), service, os.environ["CSI_VIDEO_TOKEN"])
camera = CameraService(
    "http://127.0.0.1:" + str(server.server_port),
    os.environ["CSI_VIDEO_TOKEN"],
    source_factory=PreviewCamera,
    capture_uncertainty_ns=1_000_000,
)
thread = threading.Thread(target=camera.run, daemon=True)
thread.start()


def stop(*_):
    threading.Thread(target=server.shutdown, daemon=True).start()


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, stop)
print(json.dumps(dict(port=server.server_port)), flush=True)
try:
    server.serve_forever(poll_interval=0.1)
finally:
    service.close()
    camera.stop()
    thread.join(4)
    server.server_close()
    store.close()

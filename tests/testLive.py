import io
import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from framegrabber.acquisition import Acquisition, Frame, LatestFrame
from backend.live import process, receiver
from framegrabber.transport import transmit
from framegrabber.protocol import encode_frame, request


class LiveTests(unittest.TestCase):
    def test_latest_replaces_backlog(self):
        latest = LatestFrame()
        for sequence in range(100):
            latest.put(Frame(None, {"frame_id": sequence}))
        version, frame = latest.get()
        self.assertEqual(frame.metadata["frame_id"], 99)
        self.assertEqual(latest.get(version, .001), (version, None))

    def test_slow_inference_skips_intermediate_frames(self):
        latest = LatestFrame()
        latest.put(Frame(None, {"frame_id": 0}))
        seen = []
        class Model:
            def predict(self, image):
                if not seen:
                    for sequence in range(1, 101):
                        latest.put(Frame(None, {"frame_id": sequence}))
                seen.append(image)
                return [], image
        output = io.StringIO()
        process(latest, Model(), headless=True, limit=2, output=output)
        self.assertEqual([json.loads(line)["frame_id"] for line in output.getvalue().splitlines()], [0, 100])

    def test_capture_reconnect_and_timestamps(self):
        class Camera:
            def __init__(self, broken):
                self.broken, self.closed = broken, False
            def read(self):
                return (False, None) if self.broken else (True, np.zeros((12, 16, 3), np.uint8))
            def release(self):
                self.closed = True
        cameras = []
        def factory(_):
            camera = Camera(not cameras)
            cameras.append(camera)
            return camera
        capture = Acquisition(fps=120, factory=factory).start()
        try:
            _, frame = capture.latest.get(timeout=2)
            self.assertIsNotNone(frame)
            first = frame.metadata
            time.sleep(.05)
            _, frame = capture.latest.get()
            self.assertGreater(frame.metadata["frame_id"], first["frame_id"])
            self.assertGreater(frame.metadata["monotonic_ns"], first["monotonic_ns"])
            self.assertEqual(frame.metadata["width"], 16)
            self.assertTrue(cameras[0].closed)
        finally:
            capture.close()
        self.assertFalse(capture.thread.is_alive())
        self.assertTrue(cameras[-1].closed)

    def test_upload_recovery_sends_newest_frame(self):
        capture = Acquisition()
        image = np.zeros((12, 16, 3), np.uint8)
        def frame(sequence):
            return Frame(image, dict(frame_id=sequence, timestamp_ns=time.time_ns(),
                                     monotonic_ns=time.monotonic_ns(), source_id="test"))
        capture.latest.put(frame(0))
        sent = []
        def upload(*args, metadata, **kwargs):
            sent.append(metadata["frame_id"])
            if len(sent) == 1:
                for sequence in range(1, 101):
                    capture.latest.put(frame(sequence))
                raise OSError("offline")
            capture.stop.set()
        with patch("framegrabber.transport.request", side_effect=upload):
            transmit(capture, "http://unused", "test")
        self.assertEqual(sent, [0, 100])

    def test_http_to_inference_preserves_metadata(self):
        latest = LatestFrame()
        server = receiver(("127.0.0.1", 0), "test-token-123456", latest)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            image = np.zeros((12, 16, 3), np.uint8)
            metadata, payload = encode_frame(image, 42, time.time_ns(), None)
            metadata["source_id"] = "test-camera"
            with self.assertRaises(OSError):
                request(base, "wrong", "/frame", "POST", payload=payload, metadata=metadata)
            request(base, "test-token-123456", "/frame", "POST", payload=payload, metadata=metadata)
            class Model:
                def predict(self, image):
                    return [{"label": "synthetic", "confidence": .9}], image
            output = io.StringIO()
            process(latest, Model(), headless=True, limit=1, output=output)
            result = json.loads(output.getvalue())
            for key, value in metadata.items():
                self.assertEqual(result[key], value)
            self.assertLessEqual(result["received_ns"], result["inference_completed_ns"])
            bad = dict(metadata, sha256="bad")
            with self.assertRaises(OSError):
                request(base, "test-token-123456", "/frame", "POST", payload=payload, metadata=bad)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

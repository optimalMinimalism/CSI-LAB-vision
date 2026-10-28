import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np

from backend.timing import recording_manifest
from framegrabber.camera import CameraService
from backend.export import export_dataset
from backend.inference import Policy, PoseModel, posture, run_inference, temporal
from framegrabber.protocol import (
    corrected_time,
    encode_frame,
    measure_clock,
    request,
    verify_frame,
)
from backend.server import Receiver
from backend.service import Service
from backend.storage import Store

TOKEN = "test-only-token-at-least-32-characters"


def person(kind="STANDING"):
    points = [[50, 10, 0.99] for _ in range(17)]
    if kind == "STANDING":
        joints = [(50, 30), (50, 80), (50, 120), (50, 165)]
        box = [20, 10, 80, 170]
    elif kind == "SITTING":
        joints = [(50, 30), (50, 80), (90, 80), (90, 130)]
        box = [20, 10, 100, 140]
    else:
        joints = [(30, 100), (80, 100), (110, 100), (140, 100)]
        box = [20, 90, 170, 150]
    for ids, xy in zip([(5, 6), (11, 12), (13, 14), (15, 16)], joints):
        for i in ids:
            points[i] = [*xy, 0.99]
    return dict(confidence=0.99, bbox=box, keypoints=points)


class DeterministicModel:
    provenance = dict(
        name="deterministic-test",
        version="1",
        framework="mock",
        checkpoint_sha256="not-a-checkpoint",
    )

    def predict(self, image):
        return [person()]


class FakeCamera:
    released = False

    def __init__(self, device):
        self.device = device
        self.settings = {}

    def set(self, key, value):
        self.settings[key] = value
        return True

    def read(self):
        return True, np.full((48, 64, 3), 100, np.uint8)

    def release(self):
        self.released = True


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(self.temporary.name)
        self.service = Service(self.store, stop_timeout=0.02)
        self.image = np.full((48, 64, 3), 120, np.uint8)
        self.identifier = self.service.start(True, "Prova")["id"]
        self.start = self.store.get(self.identifier)["start_ns"]
        self.monotonic = time.monotonic_ns()

    def tearDown(self):
        self.service.close()
        self.store.close()
        self.temporary.cleanup()

    def frame(self, sequence=0, delta=1_000_000, capture=1_000_000, offset=50_000_000):
        camera_time = self.start + delta - offset
        clock = dict(
            server_id=self.service.server_id,
            offset_ns=offset,
            uncertainty_ns=100_000,
            measured_camera_ns=camera_time,
            measured_pc_ns=self.start + delta,
            measured_monotonic_ns=self.monotonic + delta,
            source="synthetic-known-offset",
            rtt_ns=200_000,
        )
        meta, payload = encode_frame(self.image, sequence, camera_time, clock, capture)
        meta["monotonic_ns"] = self.monotonic + delta
        return meta, payload

    def insert(self, sequence=0, delta=1_000_000, capture=1_000_000):
        meta, payload = self.frame(sequence, delta, capture)
        self.store.receive(self.identifier, meta, payload, self.service.server_id)
        return meta, payload

    def finish(self):
        self.service.camera_stopped(self.identifier, {}) if self.store.get(
            self.identifier
        )["state"] == "STOPPING" else None
        self.service.stop(self.identifier)
        self.service.camera_stopped(
            self.identifier, dict(captured=len(self.store.frames(self.identifier)))
        )

    def test_jpeg_hash_and_decode(self):
        meta, payload = self.frame()
        self.assertEqual(meta["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(verify_frame(meta, payload).shape, self.image.shape)
        with self.assertRaises(ValueError):
            verify_frame(meta, payload[:-1] + b"x")
        meta["width"] = 4096
        with self.assertRaises(ValueError):
            verify_frame(meta, payload)

    def test_duplicates_and_conflicting_retry(self):
        meta, payload = self.insert()
        self.assertTrue(
            self.store.receive(self.identifier, meta, payload, self.service.server_id)[
                "duplicate"
            ]
        )
        meta["timestamp_ns"] += 1
        with self.assertRaises(ValueError):
            self.store.receive(self.identifier, meta, payload, self.service.server_id)
        self.assertEqual(len(self.store.frames(self.identifier)), 1)

    def test_order_and_gap_manifest(self):
        self.insert(0, 1_000_000)
        self.insert(2, 3_000_000)
        with self.assertRaises(ValueError):
            self.insert(1, 2_000_000)
        manifest = recording_manifest(self.store, self.identifier)
        self.assertEqual(
            manifest["sequence_gaps"], [dict(first_missing_id=1, last_missing_id=1)]
        )


    def test_stale_clock_and_unknown_capture_bound(self):
        meta, _ = self.frame(capture=None)
        timestamp, uncertainty, fresh = corrected_time(meta, self.service.server_id)
        self.assertEqual(timestamp, self.start + 1_000_000)
        self.assertIsNone(uncertainty)
        self.assertTrue(fresh)
        meta["clock"]["measured_camera_ns"] -= 31_000_000_000
        self.assertFalse(corrected_time(meta, self.service.server_id)[2])
        with self.assertRaises(ValueError):
            corrected_time(meta, "other-server")

    def test_clock_step_prevents_validation(self):
        self.insert(0, 1_000_000)
        meta, payload = self.frame(1, 20_000_000)
        meta["monotonic_ns"] -= 10_000_000
        self.store.receive(self.identifier, meta, payload, self.service.server_id)
        self.assertTrue(self.store.frames(self.identifier)[1]["clock_step"])

    def test_lifecycle_shared_id_and_late_frames(self):
        with self.assertRaises(ValueError):
            self.service.start()
        self.insert()
        self.finish()
        self.assertEqual(self.store.get(self.identifier)["state"], "STOPPED")
        with self.assertRaises(ValueError):
            self.insert(1, 2_000_000)


    def test_restart_marks_interrupted(self):
        self.service.active = None  # Simulate process death, not graceful close.
        self.store.close()
        self.store = Store(self.temporary.name)
        self.service.store = self.store
        self.assertEqual(self.store.get(self.identifier)["state"], "INTERRUPTED")

    def test_posture_geometry(self):
        for state in ["STANDING", "SITTING", "LYING_DOWN"]:
            self.assertEqual(posture(person(state))[0], state)
        ambiguous = person()
        ambiguous["keypoints"][5][2] = 0.1
        self.assertEqual(posture(ambiguous)[0], "UNKNOWN")

    def test_temporal_fall_requires_preceding_transition(self):
        predictions = [
            dict(
                frame_id=i, timestamp_ns=i * 100_000_000, people=[person("LYING_DOWN")]
            )
            for i in range(15)
        ]
        temporal(predictions, Policy())
        self.assertTrue(all(p["activity"] == "LYING_DOWN" for p in predictions))
        predictions[0]["people"] = [person()]
        temporal(predictions, Policy())
        self.assertEqual(predictions[1]["activity"], "FALLING")
        self.assertEqual(predictions[-1]["activity"], "FALLEN")
        predictions[1]["people"] = [person(), person()]
        temporal(predictions, Policy())
        self.assertFalse(any(p["activity"] == "FALLEN" for p in predictions))

    def test_temporal_gap_resets_fall(self):
        predictions = [
            dict(frame_id=0, timestamp_ns=0, people=[person()]),
            dict(frame_id=2, timestamp_ns=100_000_000, people=[person("LYING_DOWN")]),
        ]
        temporal(predictions, Policy())
        self.assertEqual(predictions[1]["activity"], "LYING_DOWN")

    def fixture_labels(self, capture=1_000_000, calibrated=True):
        for sequence in range(4):
            delta = (sequence + 1) * 1_000_000
            self.insert(sequence, delta, capture)
        time.sleep(0.01)
        self.finish()
        return run_inference(
            self.store,
            self.identifier,
            DeterministicModel(),
            Policy(calibrated_postures=calibrated),
        )

    def test_auto_validation_gates_and_review_audit(self):
        labels = self.fixture_labels()
        self.assertEqual(labels[0]["status"], "AUTO_VALIDATED")
        self.assertFalse(labels[0]["score_is_probability"])
        self.service.review(
            self.identifier,
            0,
            "SITTING",
            "MANUALLY_VALIDATED",
            "Mario",
            "Controllo dei fotogrammi",
        )
        self.assertEqual(
            self.store.artifact(self.identifier, "labels")[0]["activity"], "SITTING"
        )
        self.assertEqual(
            self.store.artifact(self.identifier, "predictions")[0]["observed_posture"],
            "STANDING",
        )
        self.assertEqual(
            self.store.audit(self.identifier)[0]["before"]["activity"], "STANDING"
        )
        with self.assertRaises(ValueError):
            self.service.review(
                self.identifier, 0, "STANDING", "AUTO_VALIDATED", "Mario", "x"
            )
        with self.assertRaises(ValueError):
            run_inference(self.store, self.identifier, DeterministicModel(), Policy())

    def test_unknown_sync_requires_review(self):
        labels = self.fixture_labels(capture=None)
        self.assertEqual(labels[0]["status"], "NEEDS_REVIEW")

    def test_uncalibrated_default_requires_review(self):
        labels = self.fixture_labels(calibrated=False)
        self.assertEqual(labels[0]["status"], "NEEDS_REVIEW")

    def test_export_structure_integrity_and_video(self):
        self.fixture_labels()
        output = io.BytesIO()
        export_dataset(self.store, self.identifier, output)
        with zipfile.ZipFile(output) as archive:
            prefix = "experiment-" + self.identifier + "/"
            names = set(n.removeprefix(prefix) for n in archive.namelist())
            self.assertTrue(
                {
                    "metadata.json",
                    "frames.jsonl",
                    "predictions.jsonl",
                    "labels.json",
                    "timing.json",
                    "audit.json",
                }
                <= names
            )
            metadata = json.loads(archive.read(prefix + "metadata.json"))
            self.assertEqual(metadata["video"]["status"], "generated")
            self.assertIn("video.mp4", names)
            for line in archive.read(prefix + "frames.jsonl").splitlines():
                frame = json.loads(line)
                jpeg = archive.read(prefix + f"frames/{frame['frame_id']:012d}.jpg")
                self.assertEqual(hashlib.sha256(jpeg).hexdigest(), frame["sha256"])

    def test_path_and_input_boundaries(self):
        for identifier in ["../escape", "not-a-uuid", self.identifier.upper()]:
            with self.assertRaises(ValueError):
                self.store.get(identifier)
        meta, payload = self.frame()
        meta["frame_id"] = -1
        with self.assertRaises(ValueError):
            self.store.receive(self.identifier, meta, payload, self.service.server_id)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.service = Service(self.store, stop_timeout=3)
        self.server = Receiver(("127.0.0.1", 0), self.service, TOKEN)
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.service.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        self.store.close()
        self.temp.cleanup()

    def test_auth_clock_network_and_corruption(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            request(self.base, "wrong", "/api/experiments")
        self.assertEqual(error.exception.code, 401)
        clock = measure_clock(self.base, TOKEN)
        self.assertLess(clock["uncertainty_ns"], 100_000_000)
        item = request(
            self.base,
            TOKEN,
            "/api/experiments",
            "POST",
            dict(camera_enabled=True, title="Rete"),
        )
        control = request(self.base, TOKEN, "/api/camera/control")
        self.assertEqual(control["experiment_id"], item["id"])
        meta, payload = encode_frame(
            np.zeros((48, 64, 3), np.uint8), 0, time.time_ns(), clock
        )
        path = "/api/experiments/" + item["id"] + "/frames"
        self.assertFalse(
            request(self.base, TOKEN, path, "POST", payload=payload, metadata=meta)[
                "duplicate"
            ]
        )
        self.assertTrue(
            request(self.base, TOKEN, path, "POST", payload=payload, metadata=meta)[
                "duplicate"
            ]
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            request(self.base, TOKEN, path, "POST", payload=payload[:-1], metadata=meta)
        self.assertEqual(error.exception.code, 400)
        self.assertEqual(request(self.base, TOKEN, path)["total"], 1)
        with urllib.request.urlopen(self.base) as response:
            self.assertIn("CSI-LAB Vision", response.read().decode())

    def test_camera_interface_actual_http_and_stop_ack(self):
        sources = []

        def factory(device):
            source = FakeCamera(device)
            sources.append(source)
            return source

        camera = CameraService(
            self.base,
            TOKEN,
            fps=30,
            capacity=2,
            source_factory=factory,
            capture_uncertainty_ns=1_000_000,
        )
        thread = threading.Thread(target=camera.run, daemon=True)
        thread.start()
        try:
            item = request(
                self.base, TOKEN, "/api/experiments", "POST", dict(camera_enabled=True)
            )
            deadline = time.monotonic() + 5
            while (
                time.monotonic() < deadline and len(self.store.frames(item["id"])) < 4
            ):
                time.sleep(0.05)
            self.assertGreaterEqual(len(self.store.frames(item["id"])), 4)
            result = request(
                self.base,
                TOKEN,
                "/api/experiments/" + item["id"] + "/stop",
                "POST",
                {},
                timeout=8,
            )
            self.assertTrue(result["camera_ack"])
            self.assertGreater(result["camera_telemetry"]["captured"], 0)
            self.assertEqual(
                [f["frame_id"] for f in self.store.frames(item["id"])],
                list(range(len(self.store.frames(item["id"])))),
            )
            self.assertTrue(sources[0].released)
        finally:
            camera.stop()
            thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_camera_bounded_queue_when_offline(self):
        camera = CameraService(
            self.base, TOKEN, fps=100, capacity=2, source_factory=FakeCamera
        )
        camera.active = "fixture"
        camera.clock = {}
        camera.telemetry["fixture"] = dict(
            captured=0, queue_dropped=0, capture_errors=0
        )
        thread = threading.Thread(target=camera.capture)
        thread.start()
        time.sleep(0.2)
        camera.stop()
        thread.join(2)
        self.assertEqual(camera.queue.qsize(), 2)
        self.assertGreater(camera.telemetry["fixture"]["queue_dropped"], 0)


@unittest.skipUnless(
    os.environ.get("CSI_VIDEO_TEST_MODEL"),
    "Checkpoint reale opzionale: impostare CSI_VIDEO_TEST_MODEL",
)
class RealModelTests(unittest.TestCase):
    def test_actual_checkpoint_execution(self):
        model = PoseModel(os.environ["CSI_VIDEO_TEST_MODEL"])
        image = (
            cv2.imread(os.environ["CSI_VIDEO_TEST_IMAGE"])
            if os.environ.get("CSI_VIDEO_TEST_IMAGE")
            else np.zeros((320, 320, 3), np.uint8)
        )
        people = model.predict(image)
        self.assertIsInstance(people, list)
        self.assertEqual(len(model.provenance["checkpoint_sha256"]), 64)
        if os.environ.get("CSI_VIDEO_TEST_IMAGE"):
            self.assertGreater(len(people), 0)
            self.assertEqual(len(people[0]["keypoints"]), 17)
        print("\nREAL MODEL:", model.provenance, "people:", len(people))


if __name__ == "__main__":
    unittest.main()

import json
import functools
import multiprocessing
import os
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from backend.distance import FloorCalibration, DistanceSession, estimate
from backend.server import Receiver
from backend.service import Service
from backend.storage import Store
from framegrabber.protocol import encode_frame, request

TOKEN = "distance-only-test-token-123456"


def calibration():
    return dict(image_width=320, image_height=240, source_id="0", width_m=4, depth_m=3,
                points_px=[[20, 200], [300, 200], [300, 80], [20, 80]],
                floor_labels=["chair"], floor_confirmed=True)


def inputs():
    points = [[90, 110, .99] for _ in range(17)]
    points[15], points[16] = [80, 180, .99], [100, 180, .99]
    return ([dict(confidence=.99, bbox=[60, 80, 120, 190], keypoints=points)],
            [dict(label="chair", confidence=.99, bbox=[210, 90, 250, 180])],
            dict(width=320, height=240, source_id="0", frame_id=1, timestamp_ns=time.time_ns()))


def wait_until(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for distance result")


class StubDistanceModel:
    def predict(self, image):
        return inputs()[:2]


class BrokenDistanceModel:
    def __init__(self):
        raise ValueError("missing distance weights")


class BlockingDistanceModel:
    def __init__(self, started):
        self.started = started

    def predict(self, image):
        self.started.set()
        time.sleep(20)
        return inputs()[:2]


class CrashingDistanceModel:
    def __init__(self):
        os._exit(17)


class GeometryTests(unittest.TestCase):
    def test_known_floor_distance_and_raw_lidar_not_used(self):
        people, objects, metadata = inputs()
        metadata["rangefinder"] = dict(state="available", distance_m=30)
        result = estimate(people, objects, metadata, FloorCalibration(calibration()))
        self.assertEqual(result["state"], "ready")
        self.assertAlmostEqual(result["distances"][0]["distance_m"], 2)
        self.assertIsNone(result["accuracy_m"])
        self.assertFalse(result["lidar_used_for_geometry"])

    def test_no_metric_without_calibration_or_after_camera_change(self):
        people, objects, metadata = inputs()
        result = estimate(people, objects, metadata, None)
        self.assertEqual(result["state"], "uncalibrated")
        self.assertFalse(result["distances"])
        for key, value in (("source_id", "another"), ("width", 640), ("height", 480)):
            result = estimate(people, objects, dict(metadata, **{key: value}), FloorCalibration(calibration()))
            self.assertEqual(result["state"], "calibration_mismatch")
            self.assertFalse(result["distances"])

    def test_occluded_feet_clipped_bases_and_outside_floor_are_suppressed(self):
        plane = FloorCalibration(calibration())
        for kind in ("hidden_ankle", "raised_ankle", "clipped_base", "outside_region", "unsupported_class"):
            people, objects, metadata = inputs()
            if kind == "hidden_ankle":
                people[0]["keypoints"][15][2] = .1
            elif kind == "raised_ankle":
                people[0]["keypoints"][15][1] = 90
            elif kind == "clipped_base":
                objects[0]["bbox"][3] = 240
            elif kind == "outside_region":
                objects[0]["bbox"] = [301, 90, 318, 180]
            else:
                objects[0]["label"] = "cup"
            with self.subTest(kind=kind):
                self.assertFalse(estimate(people, objects, metadata, plane)["distances"])

    def test_invalid_calibration_is_rejected(self):
        for change in (dict(floor_confirmed=False), dict(width_m=float("nan")), dict(depth_m=0),
                       dict(floor_labels=["cup"]), dict(source_id=""),
                       dict(points_px=[[20, 20]] * 4),
                       dict(points_px=[[20, 200], [300, 80], [300, 200], [20, 80]]),
                       dict(points_px=[[-1, 200], [300, 200], [300, 80], [20, 80]])):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FloorCalibration(dict(calibration(), **change))


class WorkerTests(unittest.TestCase):
    def test_on_demand_worker_persistence_and_disable(self):
        with tempfile.TemporaryDirectory() as root:
            session = DistanceSession(root, StubDistanceModel)
            try:
                _, _, metadata = inputs()
                image = np.zeros((240, 320, 3), np.uint8)
                session.submit(metadata, b"", image)
                self.assertIsNone(session.thread)
                self.assertIsNone(session.process)
                session.calibrate(calibration())
                session.enable(True)
                session.submit(metadata, b"", image)
                wait_until(lambda: session.snapshot()[0] is not None)
                output, status = session.snapshot()
                self.assertEqual(status["status"], "ready")
                self.assertTrue(status["experimental"])
                self.assertNotEqual(status["worker_pid"], os.getpid())
                self.assertGreaterEqual(status["worker_nice"], 10)
                self.assertAlmostEqual(output[2]["floor_distances"]["distances"][0]["distance_m"], 2)
                self.assertEqual(cv2.imdecode(np.frombuffer(output[1], np.uint8), cv2.IMREAD_COLOR).shape, image.shape)
                session.enable(False)
                session.submit(metadata, b"", image)
                time.sleep(.03)
                self.assertIsNone(session.snapshot()[0])
                wait_until(lambda: session.process is None)
            finally:
                session.close()
            loaded = DistanceSession(root, StubDistanceModel)
            self.assertIsNotNone(loaded.calibration)
            self.assertEqual(loaded.calibration.data["calibration_id"], session.calibration.data["calibration_id"])
            self.assertIsNone(loaded.thread)
            loaded.clear_calibration()
            self.assertIsNone(loaded.calibration)
            self.assertFalse(loaded.path.exists())
            loaded.close()

    def test_expired_lease_and_inflight_disabled_results_are_not_published(self):
        context = multiprocessing.get_context("spawn")
        started = context.Event()
        with tempfile.TemporaryDirectory() as root:
            session = DistanceSession(root, functools.partial(BlockingDistanceModel, started))
            try:
                session.enable(True)
                session.submit(inputs()[2], b"", np.zeros((240, 320, 3), np.uint8))
                self.assertTrue(started.wait(8))
                session.enable(False)
                time.sleep(.05)
                self.assertIsNone(session.snapshot()[0])
                session.enable(True)
                with session.lock:
                    session.deadline = time.monotonic() - 1
                self.assertFalse(session.active())
                self.assertEqual(session.snapshot()[1]["status"], "disabled")
            finally:
                session.close()

    def test_model_error_does_not_escape_into_camera(self):
        with tempfile.TemporaryDirectory() as root:
            session = DistanceSession(root, BrokenDistanceModel)
            try:
                session.enable(True)
                session.submit(inputs()[2], b"", np.zeros((240, 320, 3), np.uint8))
                wait_until(lambda: session.snapshot()[1]["status"] == "error")
                self.assertIn("missing distance weights", session.snapshot()[1]["error"])
            finally:
                session.close()

    def test_primary_work_kills_inflight_experimental_process(self):
        context = multiprocessing.get_context("spawn")
        started = context.Event()
        reason = [None]
        parent_nice = os.getpriority(os.PRIO_PROCESS, 0)
        with tempfile.TemporaryDirectory() as root:
            session = DistanceSession(root, functools.partial(BlockingDistanceModel, started),
                                      blocked_by=lambda: reason[0])
            try:
                session.enable(True)
                session.submit(inputs()[2], b"", np.zeros((240, 320, 3), np.uint8))
                self.assertTrue(started.wait(8))
                process = session.process
                self.assertGreaterEqual(os.getpriority(os.PRIO_PROCESS, process.pid), min(19, parent_nice + 10))
                reason[0] = "recording"
                wait_until(lambda: session.process is None)
                self.assertFalse(process.is_alive())
                self.assertEqual(session.snapshot()[1]["status"], "paused")
                self.assertIsNone(session.snapshot()[0])
                self.assertEqual(os.getpriority(os.PRIO_PROCESS, 0), parent_nice)
            finally:
                session.close()

    def test_worker_crash_is_contained_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as root:
            session = DistanceSession(root, CrashingDistanceModel)
            try:
                session.enable(True)
                session.submit(inputs()[2], b"", np.zeros((240, 320, 3), np.uint8))
                wait_until(lambda: session.snapshot()[1]["status"] == "error")
                self.assertIsNone(session.snapshot()[0])
                self.assertIsNone(session.process)
                session.enable(False)
                self.assertEqual(session.snapshot()[1]["status"], "disabled")
            finally:
                session.close()


class APITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(self.temporary.name)
        self.service = Service(self.store)
        self.service.distance().factory = StubDistanceModel
        self.server = Receiver(("127.0.0.1", 0), self.service, TOKEN)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        self.service.close()
        self.store.close()
        self.temporary.cleanup()

    def upload(self):
        image = np.zeros((240, 320, 3), np.uint8)
        metadata, payload = encode_frame(image, 1, time.time_ns(), None)
        metadata["source_id"] = "0"
        request(self.base, TOKEN, "/api/camera/preview", "POST", payload=payload, metadata=metadata)
        return metadata

    def test_authenticated_calibration_inference_and_matching_preview(self):
        import urllib.request
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as failure:
            request(self.base, "wrong", "/api/experimental/distance")
        self.assertEqual(failure.exception.code, 401)
        state = request(self.base, TOKEN, "/api/experimental/distance")
        self.assertFalse(state["enabled"])
        self.assertIsNone(self.service.distance().thread)
        request(self.base, TOKEN, "/api/experimental/distance", "POST", {"enabled": True})
        self.upload()
        request(self.base, TOKEN, "/api/experimental/distance/calibration", "POST", calibration())
        self.upload()
        wait_until(lambda: self.service.distance().snapshot()[0] is not None)
        state = request(self.base, TOKEN, "/api/experimental/distance")
        self.assertEqual(state["result"]["floor_distances"]["state"], "ready")
        req = urllib.request.Request(self.base + "/api/experimental/distance/preview", headers={"Authorization": "Bearer " + TOKEN})
        with urllib.request.urlopen(req) as response:
            meta = json.loads(response.headers["X-Frame-Metadata"])
            self.assertEqual(meta["frame_id"], state["result"]["frame_id"])
            self.assertEqual(meta["timestamp_ns"], str(state["result"]["timestamp_ns"]))
            self.assertAlmostEqual(meta["floor_distances"]["distances"][0]["distance_m"], 2)
            self.assertTrue(response.read().startswith(b"\xff\xd8"))
        request(self.base, TOKEN, "/api/experimental/distance", "POST", {"enabled": False})
        self.assertIsNone(self.service.distance().snapshot()[0])

    def test_distance_never_diverts_normal_preview_or_analysis(self):
        from unittest.mock import Mock, patch
        vision = Mock()
        vision.snapshot.return_value = (None, dict(status="waiting", error=None))
        self.service.live_vision = vision
        session = self.service.distance()
        session.enable(True)
        image = np.zeros((240, 320, 3), np.uint8)
        # An experimental submit failure cannot reach the primary frame path:
        # primary acquisition never invokes the experimental submit method.
        with patch.object(session, "submit", side_effect=RuntimeError("experimental broken")) as submit:
            self.service.latest_frame(inputs()[2], b"original-jpeg", image)
            submit.assert_not_called()
        vision.submit.assert_called_once()
        packet, status = self.service.preview_packet()
        self.assertEqual(packet[1], b"original-jpeg")
        self.assertEqual(status["status"], "waiting")
        self.assertNotIn("experimental", packet[2])

    def test_independent_leases_and_primary_work_pauses_distance(self):
        request(self.base, TOKEN, "/api/experimental/distance", "POST", {"enabled": True})
        self.assertEqual(self.service.preview_deadline, 0)
        self.assertTrue(self.service.control()["preview_enabled"])
        request(self.base, TOKEN, "/api/camera/preview", "POST", {"enabled": True})
        state = request(self.base, TOKEN, "/api/experimental/distance")
        self.assertEqual(state["paused_by"], "main_preview")
        self.assertEqual(state["status"], "paused")
        self.service.preview_deadline = 0
        self.service.start(False, "Primary recording")
        self.assertEqual(self.service.distance().snapshot()[1]["paused_by"], "recording")
        self.service.stop(self.service.active)
        with self.service.analysis_lock:
            self.assertEqual(self.service.distance().snapshot()[1]["paused_by"], "analysis_or_export")
        request(self.base, TOKEN, "/api/camera/preview", "POST", {"enabled": False})
        self.assertTrue(self.service.distance_requested())

    def test_experimental_state_failure_does_not_break_camera_control(self):
        from unittest.mock import Mock
        self.service.distance().active = Mock(side_effect=RuntimeError("experimental state failed"))
        self.assertFalse(self.service.control()["preview_enabled"])
        self.service.set_preview(True)
        self.assertTrue(self.service.control()["preview_enabled"])



if __name__ == "__main__":
    unittest.main()

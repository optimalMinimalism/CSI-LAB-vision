import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch
import urllib.error

import cv2
import numpy as np

import testSystem as fixture
from framegrabber.camera import CameraService
from framegrabber.native import NativeCapture, read_packet
from framegrabber.protocol import request, verify_frame
from backend.skeleton import geometry


class SkeletonTests(unittest.TestCase):
    def setUp(self):
        self.system = fixture.SystemTests()
        self.system.setUp()

    def tearDown(self):
        self.system.tearDown()

    def test_skeleton_geometry_and_common_clock(self):
        system = self.system
        system.fixture_labels()
        records = system.store.artifact(system.identifier, "skeletons")
        self.assertEqual(len(records), 4)
        skeleton = records[0]
        self.assertEqual(skeleton["timestamp_ns"], system.start + 1_000_000)
        person = skeleton["people"][0]
        self.assertEqual(len(person["joints"]), 17)
        self.assertEqual(len(person["bones"]), 16)
        self.assertAlmostEqual(person["joint_angles_degrees"]["leftKnee"], 180)
        self.assertIsNone(person["track_id"])
        self.assertFalse(person["metric_coordinates"])
        self.assertAlmostEqual(person["body_coordinates"][11][1], 0)
        points = fixture.person()
        points["keypoints"][13][2] = 0.1
        hidden = geometry(points, 64, 48)
        self.assertIsNone(hidden["joint_angles_degrees"]["leftKnee"])
        self.assertFalse(
            any(b["start"] == 13 or b["end"] == 13 for b in hidden["bones"])
        )


    def test_export_survives_encoder_timeout(self):
        self.system.fixture_labels()
        from backend.export import export_dataset
        import zipfile

        output = io.BytesIO()
        with patch(
            "backend.export.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 120),
        ):
            export_dataset(self.system.store, self.system.identifier, output)
        with zipfile.ZipFile(output) as archive:
            prefix = "experiment-" + self.system.identifier + "/"
            metadata = json.loads(archive.read(prefix + "metadata.json"))
            self.assertEqual(metadata["video"]["status"], "failed")
            self.assertTrue(archive.read(prefix + "frames.jsonl"))

    def test_first_frame_clock_step_is_not_fresh(self):
        from framegrabber.protocol import corrected_time

        metadata, _ = self.system.frame()
        metadata["monotonic_ns"] += 10_000_000
        self.assertFalse(corrected_time(metadata, self.system.service.server_id)[2])

    def test_video_coverage_edges_and_internal_gaps(self):
        from backend.timing import coverage_gaps

        self.assertEqual(
            coverage_gaps([80, 20], 0, 100, 10),
            [
                dict(start_ns=0, end_ns=9),
                dict(start_ns=31, end_ns=69),
                dict(start_ns=91, end_ns=100),
            ],
        )
        self.assertEqual(coverage_gaps([], 0, 100, 10), [dict(start_ns=0, end_ns=100)])
        self.assertEqual(coverage_gaps([10, 30], 0, 40, 10), [])


    def test_disabled_camera_does_not_claim_missing_video(self):
        from backend.timing import recording_manifest

        system = self.system
        system.store.update(system.identifier, camera_enabled=False)
        manifest = recording_manifest(system.store, system.identifier)
        self.assertFalse(manifest["video_coverage_requested"])
        self.assertEqual(manifest["video_coverage_gaps"], [])

    def test_worker_protocol_limits(self):
        with self.assertRaises(ValueError):
            read_packet(io.BytesIO(b"NOPE"))
        with self.assertRaises(ValueError):
            read_packet(io.BytesIO(b"CVF1" + b"\xff" * 8))
        with self.assertRaises(EOFError):
            read_packet(io.BytesIO(b"CV"))


class RecoveryTests(unittest.TestCase):
    def test_network_retry_and_bounded_recovery(self):
        system = fixture.NetworkTests()
        system.setUp()
        camera = CameraService(
            system.base,
            fixture.TOKEN,
            fps=100,
            capacity=2,
            source_factory=fixture.FakeCamera,
        )
        real_request = request
        failures = [0]

        def unreliable(*args, **kwargs):
            if args[2].endswith("/frames") and failures[0] < 3:
                failures[0] += 1
                raise urllib.error.URLError("simulated LAN interruption")
            return real_request(*args, **kwargs)

        thread = threading.Thread(target=camera.run, daemon=True)
        try:
            with patch("framegrabber.camera.request", side_effect=unreliable):
                thread.start()
                item = request(
                    system.base,
                    fixture.TOKEN,
                    "/api/experiments",
                    "POST",
                    dict(camera_enabled=True),
                )
                deadline = time.monotonic() + 6
                while (
                    time.monotonic() < deadline
                    and len(system.store.frames(item["id"])) < 5
                ):
                    time.sleep(0.05)
                result = request(
                    system.base,
                    fixture.TOKEN,
                    "/api/experiments/" + item["id"] + "/stop",
                    "POST",
                    {},
                    timeout=8,
                )
                self.assertTrue(result["camera_ack"])
                self.assertEqual(result["camera_telemetry"]["network_retries"], 3)
                self.assertGreater(result["camera_telemetry"]["queue_dropped"], 0)
                self.assertGreater(len(system.store.frames(item["id"])), 3)
                manifest = request(
                    system.base,
                    fixture.TOKEN,
                    "/api/experiments/" + item["id"] + "/timing",
                )
                self.assertTrue(manifest["sequence_gaps"])
                self.assertLessEqual(camera.queue.qsize(), 2)
        finally:
            camera.stop()
            thread.join(5)
            system.tearDown()

    def test_review_and_export_through_api(self):
        system = fixture.NetworkTests()
        system.setUp()
        camera = CameraService(
            system.base,
            fixture.TOKEN,
            fps=30,
            source_factory=fixture.FakeCamera,
            capture_uncertainty_ns=1_000_000,
        )
        thread = threading.Thread(target=camera.run, daemon=True)
        thread.start()
        try:
            item = request(
                system.base,
                fixture.TOKEN,
                "/api/experiments",
                "POST",
                dict(camera_enabled=True),
            )
            identifier = item["id"]
            base = "/api/experiments/" + identifier
            deadline = time.monotonic() + 5
            while (
                time.monotonic() < deadline and len(system.store.frames(identifier)) < 5
            ):
                time.sleep(0.05)
            request(system.base, fixture.TOKEN, base + "/stop", "POST", {}, timeout=8)
            system.service.model = fixture.DeterministicModel()
            labels = request(
                system.base, fixture.TOKEN, base + "/inference", "POST", {}
            )
            self.assertEqual(labels[0]["status"], "NEEDS_REVIEW")
            self.assertTrue(
                request(system.base, fixture.TOKEN, base + "/skeletons?frame_id=0")
            )
            review = request(
                system.base,
                fixture.TOKEN,
                base + "/review",
                "POST",
                dict(
                    label_id=0,
                    activity="SITTING",
                    status="MANUALLY_VALIDATED",
                    reviewer="Test",
                    note="Correzione controllata",
                ),
            )
            self.assertEqual(review["activity"], "SITTING")
            self.assertEqual(
                len(request(system.base, fixture.TOKEN, base + "/audit")), 1
            )
            import urllib.request, zipfile

            req = urllib.request.Request(
                system.base + base + "/export",
                headers={"Authorization": "Bearer " + fixture.TOKEN},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                    self.assertTrue(
                        any(n.endswith("/skeletons.jsonl") for n in archive.namelist())
                    )
        finally:
            camera.stop()
            thread.join(5)
            system.tearDown()


@unittest.skipUnless(
    os.environ.get("CSI_VIDEO_TEST_NATIVE"),
    "Compilare il worker e impostare CSI_VIDEO_TEST_NATIVE",
)
class NativeTests(unittest.TestCase):
    def test_cpp_encoded_payload_and_hash(self):
        output = subprocess.check_output(
            [os.environ["CSI_VIDEO_TEST_NATIVE"], "--self-test"]
        )
        metadata, payload = read_packet(io.BytesIO(output))
        metadata.update(frame_id=0)
        self.assertEqual(verify_frame(metadata, payload).shape, (48, 64, 3))
        self.assertEqual(metadata["timestamp_source"], "synthetic-native")

    def test_cpp_opencv_video_acquisition(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            for i in range(3):
                writer.write(np.full((48, 64, 3), i * 30, np.uint8))
            writer.release()
            capture = NativeCapture(
                os.environ["CSI_VIDEO_TEST_NATIVE"], video, 64, 48, 30
            )
            try:
                packets = [capture.read_packet() for _ in range(3)]
                self.assertLess(
                    packets[0][0]["timestamp_ns"], packets[-1][0]["timestamp_ns"]
                )
                for meta, payload in packets:
                    meta["frame_id"] = 0
                    verify_frame(meta, payload)
            finally:
                capture.release()

    def test_cpp_worker_full_camera_pipeline(self):
        system = fixture.NetworkTests()
        system.setUp()
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48)
            )
            for i in range(30):
                writer.write(np.full((48, 64, 3), i, np.uint8))
            writer.release()
            camera = CameraService(
                system.base,
                fixture.TOKEN,
                fps=30,
                source_factory=lambda _: NativeCapture(
                    os.environ["CSI_VIDEO_TEST_NATIVE"], video, 64, 48, 30
                ),
            )
            thread = threading.Thread(target=camera.run, daemon=True)
            thread.start()
            try:
                item = request(
                    system.base,
                    fixture.TOKEN,
                    "/api/experiments",
                    "POST",
                    dict(camera_enabled=True),
                )
                deadline = time.monotonic() + 5
                while (
                    time.monotonic() < deadline
                    and len(system.store.frames(item["id"])) < 3
                ):
                    time.sleep(0.05)
                self.assertGreaterEqual(len(system.store.frames(item["id"])), 3)
                result = request(
                    system.base,
                    fixture.TOKEN,
                    "/api/experiments/" + item["id"] + "/stop",
                    "POST",
                    {},
                    timeout=8,
                )
                self.assertTrue(result["camera_ack"])
                self.assertEqual(
                    system.store.frames(item["id"])[0]["timestamp_source"],
                    "opencv-cpp-read-completion",
                )
            finally:
                camera.stop()
                thread.join(5)
                system.tearDown()

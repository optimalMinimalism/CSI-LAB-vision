"""Linux pseudo-terminal tests: real native worker, simulated sensor bytes."""
import argparse
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from framegrabber.benewake import Benewake, add_arguments, from_arguments, from_environment
from framegrabber.acquisition import Acquisition
from framegrabber.camera import CameraService, PREVIEW
from framegrabber.transport import transmit
from backend.live import process

ROOT = Path(__file__).resolve().parents[1]


def packet(distance=123, strength=500, temperature=2248):
    data = bytearray(b"\x59\x59")
    for value in (distance, strength, temperature):
        data.extend(value.to_bytes(2, "little"))
    data.append(sum(data) & 255)
    return bytes(data)


def wait_until(check, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for sensor state")


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if not compiler or not hasattr(os, "openpty"):
            raise unittest.SkipTest("Native serial tests require Linux and g++")
        cls.build = tempfile.TemporaryDirectory()
        cls.executable = Path(cls.build.name) / "benewakeCapture"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                        str(ROOT / "CSI-LAB-camera/native/benewakeCapture.cpp"),
                        "-o", str(cls.executable)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.master, self.slave = os.openpty()
        self.addCleanup(os.close, self.master)
        self.addCleanup(os.close, self.slave)
        self.path = os.ttyname(self.slave)

    def worker(self, model="tfmini-s", path=None):
        process = subprocess.Popen([str(self.executable), path or self.path, "115200", model],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        events = queue.Queue()

        def read():
            for line in process.stdout:
                events.put(json.loads(line))
        reader = threading.Thread(target=read, daemon=True)
        reader.start()

        def close():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
            reader.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
        self.addCleanup(close)
        return process, events

    def event(self, events, state):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            item = events.get(timeout=max(.01, deadline - time.monotonic()))
            if item["state"] == state:
                return item
        self.fail("Missing state: " + state)

    def test_fragmented_noisy_and_corrupt_packets_resynchronize(self):
        _, events = self.worker()
        self.event(events, "connected")
        corrupt = bytearray(packet())
        corrupt[-1] ^= 1
        os.write(self.master, b"garbage" + corrupt + packet()[:1])
        time.sleep(.02)
        os.write(self.master, packet()[1:])
        reading = self.event(events, "reading")
        self.assertEqual(reading["distance_m"], 1.23)
        self.assertEqual(reading["signal_strength"], 500)
        self.assertEqual(reading["temperature_c"], 25)
        self.assertEqual(reading["checksum_errors"], 1)
        self.assertEqual(reading["sequence"], 1)
        self.assertLess(abs(reading["received_monotonic_ns"] - time.monotonic_ns()), 1_000_000_000)

    def test_invalid_signal_range_and_model_profile(self):
        _, events = self.worker("tf-luna")
        self.event(events, "connected")
        for bad in (packet(strength=99), packet(strength=65535), packet(distance=900), packet(distance=0)):
            os.write(self.master, bad)
            self.event(events, "invalid")
        os.write(self.master, packet(distance=800))
        reading = self.event(events, "reading")
        self.assertEqual(reading["invalid_measurements"], 4)
        self.assertEqual(reading["distance_m"], 8)

    def test_device_appears_after_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "sensor")
            process, events = self.worker(path=path)
            self.event(events, "disconnected")
            os.symlink(self.path, path)
            self.event(events, "connected")
            os.write(self.master, packet())
            self.event(events, "reading")
            process.terminate()
            process.wait(timeout=2)
            self.assertEqual(process.returncode, 0)

    def test_tf02_pro_range_signal_limits_and_error_sentinels(self):
        _, events = self.worker("tf02-pro")
        self.event(events, "connected")
        for distance in (10, 1234, 4000):
            os.write(self.master, packet(distance=distance, strength=61))
            reading = self.event(events, "reading")
            self.assertEqual(reading["distance_m"], distance / 100)
            self.assertEqual(reading["temperature_c"], 25)
        bad = [(9, 500), (4001, 500), (4500, 500), (65534, 500), (65535, 500),
               (1234, 0), (1234, 59), (1234, 60), (1234, 65535)]
        for index, (distance, strength) in enumerate(bad, 1):
            os.write(self.master, packet(distance=distance, strength=strength))
            self.assertEqual(self.event(events, "invalid")["invalid_measurements"], index)
        os.write(self.master, packet(distance=2345, strength=99))
        reading = self.event(events, "reading")
        self.assertEqual(reading["distance_m"], 23.45)
        self.assertEqual(reading["sequence"], 4)

    def test_disconnect_reopen_discards_partial_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor"
            master, slave = os.openpty()
            try:
                os.symlink(os.ttyname(slave), path)
                _, events = self.worker(path=str(path))
                self.event(events, "connected")
                os.write(master, packet()[:4])
                time.sleep(.02)
                os.close(master)
                master = None
                self.event(events, "disconnected")
                path.unlink()
                os.symlink(self.path, path)
                self.event(events, "connected")
                os.write(self.master, packet(distance=234))
                self.assertEqual(self.event(events, "reading")["distance_m"], 2.34)
            finally:
                if master is not None:
                    os.close(master)
                os.close(slave)

    def test_bridge_worker_restart_and_shutdown(self):
        sensor = Benewake(self.path, executable=self.executable).start()
        self.addCleanup(sensor.close)
        wait_until(lambda: sensor.snapshot(time.monotonic_ns())["state"] == "connected")
        os.write(self.master, packet())
        wait_until(lambda: sensor.snapshot(time.monotonic_ns())["state"] == "available")
        first = sensor.snapshot(time.monotonic_ns())
        self.assertEqual(first["distance_m"], 1.23)
        sensor.process.kill()
        wait_until(lambda: sensor.generation > first["worker_generation"] and
                   sensor.snapshot(time.monotonic_ns())["state"] == "connected")
        os.write(self.master, packet(distance=2345, strength=61))
        wait_until(lambda: sensor.snapshot(time.monotonic_ns())["state"] == "available")
        self.assertEqual(sensor.snapshot(time.monotonic_ns())["distance_m"], 23.45)
        sensor.close()
        self.assertFalse(sensor.thread.is_alive())

    def test_argument_validation(self):
        for args in (("/dev/null", "115200junk", "tfmini-s"),
                     ("/dev/null", "115200", "unknown")):
            result = subprocess.run([str(self.executable), *args], capture_output=True, timeout=2)
            self.assertEqual(result.returncode, 1)


class BridgeTests(unittest.TestCase):
    def event(self, mono=1_000_000_000):
        return dict(state="reading", sequence=1, distance_m=1.23, signal_strength=500,
                    temperature_c=25, received_monotonic_ns=mono, received_timestamp_ns=2_000_000_000,
                    checksum_errors=2, invalid_measurements=3)

    def test_timing_history_staleness_and_invalidation(self):
        sensor = Benewake("/dev/example")
        sensor._accept(self.event())
        sensor._accept(dict(self.event(1_050_000_000), sequence=2))
        matched = sensor.snapshot(1_000_000_000, now_ns=1_060_000_000)
        self.assertEqual(matched["sequence"], 1)
        self.assertEqual(matched["checksum_errors"], 2)
        self.assertEqual(matched["frame_delta_ns"], 0)
        self.assertIsNone(matched["measurement_uncertainty_ns"])
        self.assertEqual(sensor.snapshot(1_000_000_000, now_ns=2_000_000_000)["state"], "stale")
        self.assertIsNone(sensor.snapshot(1_000_000_000, now_ns=2_000_000_000)["distance_m"])
        self.assertEqual(sensor.snapshot(500_000_000, now_ns=1_060_000_000)["state"], "unmatched")
        sensor._accept(dict(state="invalid", invalid_measurements=4))
        self.assertIsNone(sensor.snapshot(1_000_000_000, now_ns=1_060_000_000)["distance_m"])

    def test_bounded_history_and_future_reception(self):
        sensor = Benewake("/dev/example")
        for index in range(1000):
            sensor._accept(dict(self.event(1_000_000_000 + index), sequence=index))
        self.assertEqual(len(sensor.samples), 64)
        self.assertEqual(sensor.snapshot(1_000_000_999, now_ns=0)["state"], "stale")

    def test_absent_worker_is_isolated(self):
        sensor = Benewake("/dev/example", executable="/no/such/worker").start()
        try:
            time.sleep(.03)
            self.assertTrue(sensor.thread.is_alive())
            self.assertEqual(sensor.snapshot(time.monotonic_ns())["state"], "disconnected")
        finally:
            sensor.close()
        self.assertFalse(sensor.thread.is_alive())

    def test_disabled_by_default_and_bad_settings(self):
        parser = argparse.ArgumentParser()
        with patch.dict(os.environ, {}, clear=True):
            add_arguments(parser)
        self.assertIsNone(from_arguments(parser.parse_args([])))
        self.assertEqual(parser.parse_args([]).benewake_model, "tf02-pro")
        for kwargs in (dict(baud=0), dict(model="other"), dict(tolerance_ms=float("nan")),
                       dict(stale_ms=0)):
            with self.assertRaises(ValueError):
                Benewake("/dev/example", **kwargs)

    def test_tf02_pro_bridge_validation_and_legacy_profiles(self):
        sensor = Benewake("/dev/example")
        self.assertEqual(sensor.model, "tf02-pro")
        for sequence, distance in enumerate((0.1, 12.34, 40), 1):
            # Batched UART packets share reception time; newest must win ties.
            sensor._accept(dict(self.event(), sequence=sequence, distance_m=distance, signal_strength=61))
            self.assertEqual(sensor.snapshot(1_000_000_000, now_ns=1_000_000_000)["distance_m"], distance)
        for distance, strength in ((0.09, 500), (40.01, 500), (45, 500), (655.34, 500),
                                   (655.35, 500), (1.23, 60), (1.23, 65535)):
            with self.subTest(distance=distance, strength=strength), self.assertRaises(ValueError):
                sensor._accept(dict(self.event(), distance_m=distance, signal_strength=strength))
        for model in ("tfmini-s", "tf-luna"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                Benewake("/dev/example", model=model)._accept(dict(self.event(), distance_m=12.34))

    def test_environment_configuration_and_allowlist(self):
        from backend.cli import load_env
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(from_environment())
            os.environ["CSI_BENEWAKE_DEVICE"] = "/dev/example"
            self.assertEqual(from_environment().model, "tf02-pro")
            path = Path(directory) / "camera.env"
            path.write_text("CSI_BENEWAKE_DEVICE=/dev/example\nCSI_BENEWAKE_MODEL=tf-luna\n")
            load_env(path)
            sensor = from_environment()
            self.assertEqual(sensor.command, ["benewakeCapture", "/dev/example", "115200", "tf-luna"])
            path.write_text("UNRELATED_SETTING=1\n")
            with self.assertRaises(ValueError):
                load_env(path)

    def test_recorded_frames_preserve_range_and_exact_retry(self):
        from testSystem import SystemTests
        fixture = SystemTests()
        fixture.setUp()
        try:
            sensor = Benewake("/dev/example")
            metadata, payload = fixture.frame()
            sensor._accept(self.event(metadata["monotonic_ns"]))
            metadata["rangefinder"] = sensor.snapshot(metadata["monotonic_ns"],
                                                       now_ns=metadata["monotonic_ns"])
            fixture.store.receive(fixture.identifier, metadata, payload, fixture.service.server_id)
            saved = fixture.store.frames(fixture.identifier)[0]
            self.assertEqual(saved["rangefinder"], metadata["rangefinder"])
            self.assertTrue(fixture.store.receive(fixture.identifier, metadata, payload,
                                                  fixture.service.server_id)["duplicate"])
            metadata["rangefinder"]["distance_m"] = 2
            with self.assertRaises(ValueError):
                fixture.store.receive(fixture.identifier, metadata, payload, fixture.service.server_id)
        finally:
            fixture.tearDown()

    def test_camera_continues_with_disconnected_sensor(self):
        sensor = Benewake("/dev/example")

        class Camera:
            def read(self):
                return True, np.zeros((12, 16, 3), np.uint8)
            def release(self):
                pass

        acquisition = Acquisition(factory=lambda _: Camera(), rangefinder=sensor).start()
        try:
            version, frame = acquisition.latest.get(timeout=1)
            self.assertEqual(frame.metadata["rangefinder"]["state"], "disconnected")
            self.assertIsNone(frame.metadata["rangefinder"]["distance_m"])
            _, newer = acquisition.latest.get(version, timeout=1)
            self.assertGreater(newer.metadata["frame_id"], frame.metadata["frame_id"])
        finally:
            acquisition.close()

    def test_live_capture_transport_and_results_preserve_measurement(self):
        sensor = Benewake("/dev/example")
        sensor._accept(self.event(time.monotonic_ns()))

        class Camera:
            def read(self):
                return True, np.zeros((12, 16, 3), np.uint8)
            def release(self):
                pass

        acquisition = Acquisition(factory=lambda _: Camera(), rangefinder=sensor).start()
        try:
            _, frame = acquisition.latest.get(timeout=1)
            self.assertEqual(frame.metadata["rangefinder"]["distance_m"], 1.23)
            sent = []
            def upload(*args, **kwargs):
                sent.append(kwargs["metadata"])
                acquisition.stop.set()
            with patch("framegrabber.transport.request", side_effect=upload):
                transmit(acquisition, "http://example", "token")
            self.assertEqual(sent[0]["rangefinder"]["distance_m"], 1.23)

            class Model:
                def predict(self, image):
                    return [], image
            output = io.StringIO()
            process(acquisition.latest, Model(), headless=True, limit=1, output=output)
            self.assertEqual(json.loads(output.getvalue())["rangefinder"]["distance_m"], 1.23)
        finally:
            acquisition.close()

    def test_recording_metadata_is_immutable_after_capture(self):
        sensor = Benewake("/dev/example")
        sensor._accept(self.event(time.monotonic_ns()))

        class Camera:
            def set(self, *_):
                pass
            def read(self):
                return True, np.zeros((12, 16, 3), np.uint8)
            def release(self):
                pass

        service = CameraService("http://example", "token", source_factory=lambda _: Camera(),
                                rangefinder=sensor)
        service.active = PREVIEW
        service.telemetry[PREVIEW] = dict(captured=0, capture_errors=0, queue_dropped=0)
        thread = threading.Thread(target=service.capture)
        thread.start()
        try:
            wait_until(lambda: service.preview_pending is not None)
            with service.lock:
                metadata = service.preview_pending[1]
            sensor._accept(dict(state="disconnected"))
            self.assertEqual(metadata["rangefinder"]["distance_m"], 1.23)
        finally:
            service.stop()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()

"""Latest-frame acquisition; no inference dependencies on the camera host."""
from dataclasses import dataclass
import logging
import threading
import time

import cv2


@dataclass(frozen=True)
class Frame:
    image: object
    metadata: dict


class LatestFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.version = 0

    def put(self, frame):
        with self.condition:
            self.frame = frame
            self.version += 1
            self.condition.notify_all()

    def get(self, after=0, timeout=0.2):
        with self.condition:
            self.condition.wait_for(lambda: self.version > after, timeout)
            return self.version, self.frame if self.version > after else None


class PiCapture:
    """Optional libcamera adapter; install Picamera2 only on Raspberry Pi OS."""
    def __init__(self, device, width, height, fps):
        from picamera2 import Picamera2
        self.camera = Picamera2(int(device))
        self.camera.configure(self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps}, buffer_count=2, queue=False))
        self.camera.start()

    def set(self, *_):
        return True  # Resolution/FPS were configured when the camera was created.

    def read(self):
        request = self.camera.capture_request()
        try:
            self.capture_metadata = request.get_metadata()
            return True, request.make_array("main")
        finally:
            request.release()

    def release(self):
        self.camera.stop()
        self.camera.close()


class Acquisition:
    def __init__(self, source=0, width=640, height=480, fps=10, backend="opencv", factory=None):
        if not 1 <= fps <= 120 or not 1 <= width <= 4096 or not 1 <= height <= 2160:
            raise ValueError("Invalid capture dimensions/FPS")
        self.source, self.width, self.height, self.fps = source, width, height, fps
        self.backend, self.factory = backend, factory
        self.latest, self.stop = LatestFrame(), threading.Event()
        self.thread = None

    def start(self):
        if self.backend == "picamera2" and self.factory is None:
            import importlib
            importlib.import_module("picamera2")
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        camera, sequence = None, 0
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    if camera is None:
                        if self.factory:
                            camera = self.factory(self.source)
                        elif self.backend == "picamera2":
                            camera = PiCapture(self.source, self.width, self.height, self.fps)
                        else:
                            camera = cv2.VideoCapture(self.source)
                            for prop, value in [(cv2.CAP_PROP_FRAME_WIDTH, self.width),
                                                (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                                                (cv2.CAP_PROP_FPS, self.fps),
                                                (cv2.CAP_PROP_BUFFERSIZE, 1)]:
                                camera.set(prop, value)
                    ok, image = camera.read()
                    wall, mono = time.time_ns(), time.monotonic_ns()
                    if not ok:
                        raise OSError("Camera read failed")
                    evidence = {}
                    sensor = getattr(camera, 'capture_metadata', {}).get('SensorTimestamp')
                    if type(sensor) is int and hasattr(time, 'CLOCK_BOOTTIME'):
                        before = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                        sensor_wall = time.time_ns()
                        after = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                        if 0 <= after-sensor <= 5_000_000_000:
                            evidence = {'sensor_timestamp_ns': sensor, 'sensor_clock': 'CLOCK_BOOTTIME',
                                'sensor_mapping_bracket_ns': after-before, 'read_completed_timestamp_ns': wall}
                            wall = sensor_wall + sensor-(before+after)//2
                            mono -= (before+after)//2-sensor
                    self.latest.put(Frame(image, dict(
                        frame_id=sequence, timestamp_ns=wall, monotonic_ns=mono,
                        width=image.shape[1], height=image.shape[0], source_id=str(self.source),
                        timestamp_source="picamera2-sensor-timestamp" if evidence else f"{self.backend}-read-completion", **evidence)))
                    sequence += 1
                except (OSError, RuntimeError, cv2.error) as exc:
                    logging.warning("Camera reconnect: %s", exc)
                    if camera:
                        camera.release()
                    camera = None
                    self.stop.wait(1)
                self.stop.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        finally:
            if camera:
                camera.release()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                logging.warning("Camera driver did not unblock within 3 seconds")

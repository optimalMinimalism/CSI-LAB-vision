"""Optional native UART telemetry; no dependency on the camera or model workers."""
from collections import deque
import json
import logging
import math
import os
import subprocess
import threading
import time

LOG = logging.getLogger(__name__)
PROFILES = {
    "tf02-pro": (0.1, 40.0, 61),
    "tfmini-s": (0.1, 12.0, 100),
    "tf-luna": (0.2, 8.0, 100),
}


class Benewake:
    def __init__(self, device, model="tf02-pro", baud=115200, executable="benewakeCapture",
                 tolerance_ms=100, stale_ms=250):
        if model not in PROFILES:
            raise ValueError("Unsupported Benewake model")
        if baud not in {9600, 19200, 38400, 57600, 115200, 230400}:
            raise ValueError("Unsupported Benewake baud rate")
        if not device or not executable:
            raise ValueError("Benewake device and executable are required")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 5000
               for v in (tolerance_ms, stale_ms)):
            raise ValueError("Benewake timing limits must be in (0, 5000] ms")
        self.device, self.model = str(device), model
        self.command = [str(executable), self.device, str(baud), model]
        self.tolerance_ns, self.stale_ns = int(tolerance_ms * 1e6), int(stale_ms * 1e6)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = self.process = None
        self.samples = deque(maxlen=64)
        self.state = "disconnected"
        self.counters = dict(checksum_errors=0, invalid_measurements=0)
        self.generation = 0

    def start(self):
        if self.thread is not None:
            return self
        self.thread = threading.Thread(target=self._run, name="benewake", daemon=True)
        self.thread.start()
        return self

    def _accept(self, event):
        if not isinstance(event, dict):
            raise ValueError("Invalid native telemetry object")
        counters = {key: event.get(key) for key in self.counters}
        state = event["state"]
        if state not in {"connected", "disconnected", "reading", "invalid"}:
            raise ValueError("Invalid native sensor state")
        if state == "reading":
            for key in ("sequence", "received_monotonic_ns", "received_timestamp_ns", "signal_strength"):
                if type(event.get(key)) is not int or event[key] < 0:
                    raise ValueError("Invalid native sensor integer")
            minimum, maximum, minimum_strength = PROFILES[self.model]
            distance = event.get("distance_m")
            temperature = event.get("temperature_c")
            if (type(distance) not in (int, float) or not math.isfinite(distance)
                    or not minimum <= distance <= maximum
                    or not minimum_strength <= event["signal_strength"] < 65535
                    or type(temperature) not in (int, float) or not math.isfinite(temperature)):
                raise ValueError("Invalid native sensor reading")
            event = {key: event[key] for key in (
                "sequence", "distance_m", "signal_strength", "temperature_c",
                "received_monotonic_ns", "received_timestamp_ns")}
        with self.lock:
            for key in self.counters:
                value = counters[key]
                if type(value) is int and value >= 0:
                    self.counters[key] = value
            self.state = state
            if state == "reading":
                self.samples.append(dict(event, worker_generation=self.generation))
            else:
                self.samples.clear()

    def _run(self):
        while not self.stop.is_set():
            process = None
            try:
                process = subprocess.Popen(self.command, stdout=subprocess.PIPE)
                with self.lock:
                    self.process = process
                    self.generation += 1
                    self.counters = dict(checksum_errors=0, invalid_measurements=0)
                while not self.stop.is_set():
                    line = process.stdout.readline(4097)
                    if not line:
                        break
                    if len(line) > 4096 or not line.endswith(b"\n"):
                        raise ValueError("Oversized native telemetry record")
                    self._accept(json.loads(line))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                LOG.warning("Benewake unavailable: %s", exc)
            finally:
                if process is not None:
                    self._terminate(process)
                    process.stdout.close()
                with self.lock:
                    self.process = None
                    self.state = "disconnected"
                    self.samples.clear()
            self.stop.wait(1)

    @staticmethod
    def _terminate(process):
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            except ProcessLookupError:
                pass

    def snapshot(self, frame_monotonic_ns, now_ns=None):
        now = time.monotonic_ns() if now_ns is None else now_ns
        with self.lock:
            result = dict(enabled=True, model=self.model, source_id=self.device,
                          state=self.state, distance_m=None, **self.counters)
            if not self.samples:
                return result
            # A single UART read can contain multiple packets with equal timestamps.
            # Prefer the newest packet when reception-time differences are tied.
            sample = min(reversed(self.samples),
                         key=lambda item: abs(item["received_monotonic_ns"] - frame_monotonic_ns))
            age = now - sample["received_monotonic_ns"]
            delta = sample["received_monotonic_ns"] - frame_monotonic_ns
            if not 0 <= age <= self.stale_ns:
                result["state"] = "stale"
            elif abs(delta) > self.tolerance_ns:
                result["state"] = "unmatched"
            else:
                result.update(sample, state="available")
            result.update(age_ns=age, frame_delta_ns=delta,
                          timestamp_source="host-uart-reception",
                          measurement_uncertainty_ns=None, sensor_timestamp_ns=None)
            return result

    def close(self):
        self.stop.set()
        with self.lock:
            process = self.process
        if process is not None:
            self._terminate(process)
        if self.thread is not None:
            self.thread.join(timeout=4)
            if self.thread.is_alive():
                LOG.warning("Benewake worker did not stop within four seconds")


def add_arguments(parser):
    parser.add_argument("--benewake-device", default=os.environ.get("CSI_BENEWAKE_DEVICE"),
                        help="Optional UART path; no serial ports are probed when omitted")
    parser.add_argument("--benewake-model", choices=list(PROFILES),
                        default=os.environ.get("CSI_BENEWAKE_MODEL", "tf02-pro"))
    parser.add_argument("--benewake-baud", type=int, default=int(os.environ.get("CSI_BENEWAKE_BAUD", "115200")))
    parser.add_argument("--benewake-worker", default=os.environ.get("CSI_BENEWAKE_WORKER", "benewakeCapture"))
    parser.add_argument("--benewake-tolerance-ms", type=float, default=100)
    parser.add_argument("--benewake-stale-ms", type=float, default=250)


def from_arguments(args):
    if not args.benewake_device:
        return None
    return Benewake(args.benewake_device, args.benewake_model, args.benewake_baud,
                    args.benewake_worker, args.benewake_tolerance_ms, args.benewake_stale_ms)


def from_environment():
    device = os.environ.get("CSI_BENEWAKE_DEVICE")
    if not device:
        return None
    return Benewake(device, model=os.environ.get("CSI_BENEWAKE_MODEL", "tf02-pro"),
                    baud=int(os.environ.get("CSI_BENEWAKE_BAUD", "115200")),
                    executable=os.environ.get("CSI_BENEWAKE_WORKER", "benewakeCapture"))

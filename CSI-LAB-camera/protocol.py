"""JPEG è il payload canonico: SHA-256 prima e dopo la trasmissione."""

import hashlib
import json
import math
import time
import urllib.request
import uuid

import cv2
import numpy as np

MAX_PAYLOAD = 8 * 1024 * 1024
STATES = {
    "STANDING",
    "SITTING",
    "WALKING",
    "LYING_DOWN",
    "FALLING",
    "FALLEN",
    "UNKNOWN",
}
STATUSES = {"AUTO_VALIDATED", "NEEDS_REVIEW", "MANUALLY_VALIDATED", "REJECTED"}


def experiment_id(value):
    if str(uuid.UUID(value)) != value:
        raise ValueError("Identificatore esperimento non canonico")
    return value


def integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name}: intero fuori intervallo")
    return value


def number(value, name, minimum=0, maximum=1):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name}: numero fuori intervallo")
    return value


def encode_frame(image, sequence, timestamp_ns, clock, capture_uncertainty_ns=None):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Codifica JPEG fallita")
    payload = encoded.tobytes()
    return dict(
        frame_id=sequence,
        timestamp_ns=timestamp_ns,
        monotonic_ns=time.monotonic_ns(),
        sha256=hashlib.sha256(payload).hexdigest(),
        width=image.shape[1],
        height=image.shape[0],
        encoding="JPEG",
        clock=clock,
        timestamp_source="opencv-read-completion",
        capture_uncertainty_ns=capture_uncertainty_ns,
    ), payload


def verify_frame(metadata, payload):
    integer(metadata["frame_id"], "frame_id")
    integer(metadata["timestamp_ns"], "timestamp_ns")
    integer(metadata["monotonic_ns"], "monotonic_ns")
    width = integer(metadata["width"], "width", 1, 4096)
    height = integer(metadata["height"], "height", 1, 2160)
    if metadata["encoding"] != "JPEG" or not 0 < len(payload) <= MAX_PAYLOAD:
        raise ValueError("Payload non valido")
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError("SHA-256 non corrispondente")
    # Check JPEG dimensions before decoding to prevent decompression bombs.
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("JPEG non valido")
    cursor, dimensions = 2, None
    while cursor + 4 <= len(payload):
        if payload[cursor] != 255:
            raise ValueError("Marcatori JPEG non validi")
        while cursor < len(payload) and payload[cursor] == 255:
            cursor += 1
        marker = payload[cursor]
        cursor += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            break
        length = int.from_bytes(payload[cursor : cursor + 2], "big")
        if length < 2 or cursor + length > len(payload):
            raise ValueError("Segmento JPEG troncato")
        if marker in {0xC0, 0xC1, 0xC2}:
            dimensions = (
                int.from_bytes(payload[cursor + 5 : cursor + 7], "big"),
                int.from_bytes(payload[cursor + 3 : cursor + 5], "big"),
            )
            break
        cursor += length
    if dimensions != (width, height):
        raise ValueError("Dimensioni JPEG non corrispondenti")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError("JPEG indecodificabile")
    return image


def request(
    base, token, path, method="GET", data=None, payload=None, metadata=None, timeout=3
):
    headers = {"Authorization": "Bearer " + token}
    if payload is not None:
        body = payload
        headers["Content-Type"] = "image/jpeg"
        headers["X-Frame-Metadata"] = json.dumps(metadata, separators=(",", ":"))
    elif data is not None:
        body = json.dumps(data, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    else:
        body = None
    req = urllib.request.Request(
        base.rstrip("/") + path, data=body, headers=headers, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def measure_clock(base, token):
    t1, m1 = time.time_ns(), time.monotonic_ns()
    reply = request(base, token, "/api/clock")
    t4, m4 = time.time_ns(), time.monotonic_ns()
    if reply.get("clock_unstable"):
        raise ValueError(
            "Orologio PC discontinuo: verificare NTP e riavviare il receiver"
        )
    t2, t3 = reply["receive_ns"], reply["send_ns"]
    rtt = max(0, (m4 - m1) - (t3 - t2))
    if abs((t4 - t1) - (m4 - m1)) > 5_000_000:
        raise ValueError("Orologio locale cambiato durante la misura")
    return dict(
        offset_ns=((t2 - t1) + (t3 - t4)) // 2,
        uncertainty_ns=rtt // 2,
        measured_camera_ns=t4,
        measured_pc_ns=t3,
        measured_monotonic_ns=m4,
        rtt_ns=rtt,
        source="four-timestamp-http",
        server_id=reply["server_id"],
    )


def corrected_time(metadata, server_id):
    clock = metadata["clock"]
    if clock["server_id"] != server_id:
        raise ValueError("Misura temporale di un altro processo server")
    offset = integer(clock["offset_ns"], "offset_ns", -(2**62), 2**62)
    uncertainty = integer(clock["uncertainty_ns"], "uncertainty_ns", 0, 60_000_000_000)
    age = abs(
        metadata["timestamp_ns"] - integer(clock["measured_camera_ns"], "misura clock")
    )
    capture = metadata.get("capture_uncertainty_ns")
    if capture is not None:
        capture = integer(capture, "capture_uncertainty_ns", 0, 60_000_000_000)
    # Conservative drift allowance 100 ppm; stale measurements remain review-only.
    total = None if capture is None else uncertainty + capture + age // 10_000
    measured_mono = clock.get("measured_monotonic_ns")
    stable = False
    if measured_mono is not None:
        integer(measured_mono, "measured_monotonic_ns")
        wall_delta = metadata["timestamp_ns"] - clock["measured_camera_ns"]
        mono_delta = metadata["monotonic_ns"] - measured_mono
        stable = abs(wall_delta - mono_delta) <= 5_000_000
    return metadata["timestamp_ns"] + offset, total, age <= 30_000_000_000 and stable

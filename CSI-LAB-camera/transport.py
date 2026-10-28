"""Latest-frame HTTP upload with reconnection."""
import logging

from .protocol import encode_frame, request


def transmit(acquisition, base, token):
    version = 0
    while not acquisition.stop.is_set():
        version, frame = acquisition.latest.get(version)
        if frame is None:
            continue
        metadata, payload = encode_frame(frame.image, frame.metadata["frame_id"],
                                          frame.metadata["timestamp_ns"], None)
        metadata.update(frame.metadata)
        try:
            request(base, token, "/frame", "POST", payload=payload, metadata=metadata)
        except (OSError, ValueError) as exc:
            logging.warning("Upload failed; next attempt uses latest frame: %s", exc)
            acquisition.stop.wait(.5)


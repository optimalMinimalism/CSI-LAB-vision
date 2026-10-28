"""Bridge al worker C++: framing limitato, nessuna ricodifica del JPEG."""

import json
import struct
import subprocess

from .protocol import MAX_PAYLOAD


def read_exact(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        block = stream.read(length - len(chunks))
        if not block:
            raise EOFError("Worker di acquisizione terminato")
        chunks.extend(block)
    return bytes(chunks)


def read_packet(stream):
    if read_exact(stream, 4) != b"CVF1":
        raise ValueError("Protocollo worker non valido")
    metadata_length, payload_length = struct.unpack("!II", read_exact(stream, 8))
    if not 0 < metadata_length <= 12000 or not 0 < payload_length <= MAX_PAYLOAD:
        raise ValueError("Pacchetto worker oltre il limite")
    metadata = json.loads(read_exact(stream, metadata_length))
    return metadata, read_exact(stream, payload_length)


class NativeCapture:
    def __init__(self, executable, device, width, height, fps):
        self.process = subprocess.Popen(
            [str(executable), str(device), str(width), str(height), str(fps)],
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

    def set(self, *_):
        return True

    def read_packet(self):
        return read_packet(self.process.stdout)

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.process.stdout:
            self.process.stdout.close()

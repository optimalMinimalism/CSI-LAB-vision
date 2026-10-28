"""Export riproducibile: JPEG canonici e MP4 derivato, con manifest separato."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from .timing import recording_manifest


def export_dataset(store, identifier, output):
    item = store.get(identifier)
    if item["state"] not in {"STOPPED", "INTERRUPTED", "ERROR"}:
        raise ValueError("Fermare l'esperimento prima dell'export")
    prefix = "experiment-" + identifier + "/"
    frames = store.frames(identifier)
    metadata = dict(
        item,
        schema_version=1,
        video_role="derived preview; frames.jsonl is authoritative",
        inference=store.artifact(identifier, "inference"),
    )
    with tempfile.TemporaryDirectory(prefix="vision-export-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:

            def save(name, value):
                archive.writestr(
                    prefix + name,
                    json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                )

            for frame in frames:
                _, payload = store.frame(identifier, frame["frame_id"])
                if hashlib.sha256(payload).hexdigest() != frame["sha256"]:
                    raise ValueError("Integrità JPEG su disco non valida")
                name = f"frames/{frame['frame_id']:012d}.jpg"
                archive.writestr(prefix + name, payload)
                (root / f"{frame['frame_id']:012d}.jpg").write_bytes(payload)
            archive.writestr(
                prefix + "frames.jsonl",
                "".join(json.dumps(f, allow_nan=False) + "\n" for f in frames),
            )
            archive.writestr(
                prefix + "predictions.jsonl",
                "".join(
                    json.dumps(p, allow_nan=False) + "\n"
                    for p in store.artifact(identifier, "predictions")
                ),
            )
            save("labels.json", store.artifact(identifier, "labels"))
            save("audit.json", store.audit(identifier))
            for kind in ("skeletons",):
                archive.writestr(
                    prefix + kind + ".jsonl",
                    "".join(
                        json.dumps(p, allow_nan=False) + "\n"
                        for p in store.artifact(identifier, kind)
                    ),
                )
            save("timing.json", recording_manifest(store, identifier))
            metadata["video"] = encode_video(root, frames)
            if metadata["video"]["status"] == "generated":
                archive.write(root / "video.mp4", prefix + "video.mp4")
            save("metadata.json", metadata)
    return output


def encode_video(root, frames):
    if not frames or not shutil.which("ffmpeg"):
        return dict(status="absent", reason="nessun frame o ffmpeg non disponibile")
    lines = []
    for i, frame in enumerate(frames):
        lines.append(f"file '{frame['frame_id']:012d}.jpg'")
        duration = (
            (
                frames[i + 1]["aligned_timestamp_ns"]
                - frame["aligned_timestamp_ns"]
            )
            / 1e9
            if i + 1 < len(frames)
            else 0.1
        )
        # MP4 is only a preview. Gaps remain visible and explicit in manifest.
        lines.append(f"duration {max(0.001, duration):.9f}")
    lines.append(f"file '{frames[-1]['frame_id']:012d}.jpg'")
    (root / "timeline.txt").write_text("\n".join(lines) + "\n")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                "timeline.txt",
                "-fps_mode",
                "vfr",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "video.mp4",
            ],
            cwd=root,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        result = subprocess.CompletedProcess(
            [], 1, stderr=str(exc).encode()
        )
    if result.returncode == 0:
        return dict(status="generated", origin_ns=frames[0]["aligned_timestamp_ns"],
                    variable_frame_rate=True, last_frame_padding_ns=100_000_000)
    return dict(status="failed", error=result.stderr.decode(errors="replace")[:1000])


def export_video(store, identifier, output, pipeline=None, mode="all"):
    from .vision import normalize_mode, annotate

    if mode != "raw":
        mode = normalize_mode(mode)
        if mode != "none" and pipeline is None:
            raise ValueError("Overlay richiesti ma modello di analisi non disponibile")
    item = store.get(identifier)
    if item["state"] not in {"STOPPED", "INTERRUPTED", "ERROR"}:
        raise ValueError("Fermare la registrazione prima di scaricare il video")
    frames = store.frames(identifier)
    with tempfile.TemporaryDirectory(prefix="vision-video-") as directory:
        root = Path(directory)
        for frame in frames:
            _, payload = store.frame(identifier, frame["frame_id"])
            if hashlib.sha256(payload).hexdigest() != frame["sha256"]:
                raise ValueError("Integrità JPEG su disco non valida")
            if mode != "raw":
                import cv2
                import numpy as np

                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Invalid recorded image")
                if pipeline is not None:
                    image, people, analysis = pipeline.predict(image)
                else:
                    people, analysis = [], {}
                payload = annotate(image, people, analysis, frame, mode)
            (root / f"{frame['frame_id']:012d}.jpg").write_bytes(payload)
        result = encode_video(root, frames)
        if result["status"] != "generated":
            raise ValueError("MP4 non disponibile: " + result.get("error", result.get("reason", "errore di codifica")))
        with (root / "video.mp4").open("rb") as video:
            shutil.copyfileobj(video, output)
    return output

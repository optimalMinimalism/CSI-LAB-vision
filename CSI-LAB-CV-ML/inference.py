"""Pose osservata e azione temporale restano due risultati distinti."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import math
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass
class Policy:
    confidence: float = 0.9
    max_sync_ns: int = 20_000_000
    min_frames: int = 3
    max_gap_ns: int = 250_000_000
    calibrated_postures: bool = False
    stationary_camera: bool = False

    def __post_init__(self):
        if (
            not 0 <= self.confidence <= 1
            or self.min_frames < 2
            or self.max_sync_ns < 0
            or self.max_gap_ns <= 0
        ):
            raise ValueError("Soglie di validazione non valide")


def posture(person):
    """COCO 17 punti: regole geometriche conservatrici, score non probabilità."""
    points = np.array(person.get("keypoints", []), dtype=float)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return "UNKNOWN", 0.0
    selected = points[[5, 6, 11, 12, 13, 14, 15, 16]]
    score = min(float(person["confidence"]), float(selected[:, 2].min()))
    if score < 0.5:
        return "UNKNOWN", score
    shoulder = points[[5, 6], :2].mean(axis=0)
    hip = points[[11, 12], :2].mean(axis=0)
    knee = points[[13, 14], :2].mean(axis=0)
    ankle = points[[15, 16], :2].mean(axis=0)
    torso = np.linalg.norm(shoulder - hip)
    if torso < 5:
        return "UNKNOWN", score
    if abs(shoulder[0] - hip[0]) > abs(shoulder[1] - hip[1]) * 1.5:
        return "LYING_DOWN", score
    if shoulder[1] >= hip[1] or hip[1] >= ankle[1]:
        return "UNKNOWN", score
    thigh, shin = hip - knee, ankle - knee
    denom = np.linalg.norm(thigh) * np.linalg.norm(shin)
    angle = (
        math.degrees(math.acos(float(np.clip(np.dot(thigh, shin) / denom, -1, 1))))
        if denom
        else 0
    )
    if angle > 155 and knee[1] > hip[1] and ankle[1] > knee[1]:
        return "STANDING", score
    if 60 <= angle <= 130 and abs(hip[1] - knee[1]) < torso * 0.65:
        return "SITTING", score
    return "UNKNOWN", score


class PoseModel:
    def __init__(self, checkpoint, image_size=640):
        from ultralytics import YOLO
        import os
        import torch

        threads = int(os.environ.get("CSI_VIDEO_INFERENCE_THREADS", "2"))
        if not 1 <= threads <= 64:
            raise ValueError("CSI_VIDEO_INFERENCE_THREADS deve essere tra 1 e 64")
        torch.set_num_threads(threads)
        self.detector_confidence = float(
            os.environ.get("CSI_VIDEO_DETECTOR_CONFIDENCE", ".25")
        )
        if not 0 < self.detector_confidence <= 1:
            raise ValueError("Soglia detector non valida")
        path = Path(checkpoint)
        # No implicit network access or unpinned automatic model download at runtime.
        if not path.is_file():
            raise ValueError("Checkpoint locale assente: configurare CSI_VIDEO_MODEL")
        if not 32 <= image_size <= 1280:
            raise ValueError("Image size must be between 32 and 1280")
        self.image_size = image_size
        self.model = YOLO(str(path), task="pose")
        if self.model.task != "pose":
            raise ValueError("Il checkpoint non è un modello pose")
        self.provenance = dict(
            name=path.name,
            framework="ultralytics",
            version=importlib.metadata.version("ultralytics"),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            method="coco-geometry-temporal-v1",
            heuristic_calibrated=False,
            detector_confidence=self.detector_confidence,
            keypoint_visibility=0.5,
            image_size=self.image_size,
            device="cpu",
            inference_threads=threads,
        )

    def predict(self, image):
        result = self.model.predict(
            image, verbose=False, device="cpu", conf=self.detector_confidence, imgsz=self.image_size
        )[0]
        if result.keypoints is None:
            return []
        return [
            dict(confidence=float(conf), bbox=box.tolist(), keypoints=points.tolist())
            for conf, box, points in zip(
                result.boxes.conf.cpu().numpy(),
                result.boxes.xyxy.cpu().numpy(),
                result.keypoints.data.cpu().numpy(),
            )
        ]


def temporal(predictions, policy):
    previous = None
    fall_at, lying_since = None, None
    for prediction in predictions:
        people, timestamp = prediction["people"], prediction["timestamp_ns"]
        observed, score = posture(people[0]) if len(people) == 1 else ("UNKNOWN", 0.0)
        prediction.update(
            observed_posture=observed,
            activity=observed,
            heuristic_score=score,
            temporal_method="single-person geometry, uncalibrated",
            action_evidence=[],
        )
        continuous = (
            previous is not None
            and prediction["frame_id"] == previous["frame_id"] + 1
            and 0 < timestamp - previous["timestamp_ns"] <= policy.max_gap_ns
        )
        if (
            not continuous
            or len(people) != 1
            or not previous
            or len(previous["people"]) != 1
        ):
            fall_at, lying_since = None, None
        elif observed != "UNKNOWN" and previous["observed_posture"] != "UNKNOWN":
            box, prev = people[0]["bbox"], previous["people"][0]["bbox"]
            scale = max(prev[3] - prev[1], prev[2] - prev[0], 1)
            dx = ((box[0] + box[2]) - (prev[0] + prev[2])) / 2 / scale
            dy = ((box[1] + box[3]) - (prev[1] + prev[3])) / 2 / scale
            # A cut, a different person or a very large jump is not temporal evidence.
            if abs(dx) > 0.5 or abs(dy) > 0.8:
                fall_at, lying_since = None, None
                prediction["activity"] = "UNKNOWN"
            elif observed == "LYING_DOWN":
                lying_since = lying_since or timestamp
                if (
                    previous["observed_posture"] in {"STANDING", "SITTING"}
                    and dy > 0.12
                ):
                    fall_at = timestamp
                    prediction["activity"] = "FALLING"
                    prediction["action_evidence"] = [
                        previous["frame_id"],
                        prediction["frame_id"],
                    ]
                elif fall_at and timestamp - fall_at >= 1_000_000_000:
                    prediction["activity"] = "FALLEN"
                    prediction["action_evidence"] = [
                        p["frame_id"]
                        for p in predictions
                        if fall_at <= p["timestamp_ns"] <= timestamp
                    ]
            else:
                fall_at, lying_since = None, None
                if (
                    observed == previous["observed_posture"] == "STANDING"
                    and policy.stationary_camera
                    and 0.02 < abs(dx) < 0.3
                ):
                    prediction["activity"] = "WALKING"
                    prediction["action_evidence"] = [
                        previous["frame_id"],
                        prediction["frame_id"],
                    ]
        previous = prediction
    return predictions


def make_labels(store, identifier, predictions, model, policy):
    item, frames = store.get(identifier), store.frames(identifier)
    by_id = {f["frame_id"]: f for f in frames}
    groups = []
    for prediction in predictions:
        if (
            not groups
            or groups[-1][-1]["activity"] != prediction["activity"]
            or prediction["timestamp_ns"] - groups[-1][-1]["timestamp_ns"]
            > policy.max_gap_ns
            or prediction["frame_id"] != groups[-1][-1]["frame_id"] + 1
        ):
            groups.append([])
        groups[-1].append(prediction)
    labels = []
    for group in groups:
        first, last = group[0]["timestamp_ns"], group[-1]["timestamp_ns"]
        activity = group[0]["activity"]
        # Include precursor frames used by temporal rules as explicit supporting evidence.
        ids = sorted(
            {p["frame_id"] for p in group}
            | {i for p in group for i in p["action_evidence"]}
        )
        supporting = [by_id[i] for i in ids]
        first = min(first, min(f["aligned_timestamp_ns"] for f in supporting))
        reasons = []
        score = min(p["heuristic_score"] for p in group)
        if activity == "UNKNOWN":
            reasons.append("postura ambigua o numero persone diverso da uno")
        if item.get("pc_clock_unstable"):
            reasons.append("orologio PC discontinuo")
        if len(group) < policy.min_frames:
            reasons.append("durata visiva insufficiente")
        if score < policy.confidence:
            reasons.append("score geometrico/keypoint sotto soglia")
        if any(
            f["sync_uncertainty_ns"] is None
            or f["sync_uncertainty_ns"] > policy.max_sync_ns
            or not f["clock_fresh"]
            or f["clock_step"]
            for f in supporting
        ):
            reasons.append("tempi video incerti, scaduti o discontinui")
        if item["state"] != "STOPPED" or (
            item["camera_enabled"] and not item["camera_ack"]
        ):
            reasons.append("arresto dei dispositivi non confermato")
        if activity in {"WALKING", "FALLING", "FALLEN"}:
            reasons.append("azione euristica: verifica umana obbligatoria")
        if not policy.calibrated_postures:
            reasons.append("regole posturali non calibrate su questo impianto")
        labels.append(
            dict(
                id=len(labels),
                experiment_id=identifier,
                start_ns=first,
                end_ns=last,
                activity=activity,
                observed_postures=sorted({p["observed_posture"] for p in group}),
                frame_ids=ids,
                frame_hashes=[f["sha256"] for f in supporting],
                model=model,
                model_confidence=min(
                    (p["people"][0]["confidence"] if len(p["people"]) == 1 else 0)
                    for p in group
                ),
                heuristic_score=score,
                score_is_probability=False,
                status="NEEDS_REVIEW" if reasons else "AUTO_VALIDATED",
                reasons=reasons,
                automatic_activity=activity,
            )
        )
    return labels


def run_inference(store, identifier, model, policy):
    item = store.get(identifier)
    if item["state"] not in {"STOPPED", "INTERRUPTED", "ERROR"}:
        raise ValueError("Fermare l'esperimento prima dell'analisi")
    if store.artifact(identifier, "predictions"):
        raise ValueError("Analisi già presente; le revisioni non vengono sovrascritte")
    predictions = []
    for frame in store.frames(identifier):
        _, payload = store.frame(identifier, frame["frame_id"])
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        predictions.append(
            dict(
                frame_id=frame["frame_id"],
                sha256=frame["sha256"],
                timestamp_ns=frame["aligned_timestamp_ns"],
                people=model.predict(image),
                model=model.provenance,
            )
        )
    temporal(predictions, policy)
    provenance = dict(model.provenance, heuristic_calibrated=policy.calibrated_postures)
    labels = make_labels(store, identifier, predictions, provenance, policy)
    from .skeleton import build_skeletons

    skeletons = build_skeletons(store, identifier, predictions)
    # One transaction: readers never see labels without their source predictions.
    with store.lock, store.db:
        for kind, value in (
            ("predictions", predictions),
            ("labels", labels),
            ("skeletons", skeletons),
            (
                "inference",
                dict(model=provenance, policy=vars(policy), run_ns=time.time_ns()),
            ),
        ):
            store.db.execute(
                "INSERT OR REPLACE INTO artifacts VALUES(?,?,?)",
                (identifier, kind, __import__("json").dumps(value, allow_nan=False)),
            )
    return labels

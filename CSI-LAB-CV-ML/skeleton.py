"""COCO 2D skeleton geometry and camera-frame provenance."""

import math


JOINTS = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]
BONES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def geometry(person, width, height, threshold=0.5):
    points = person["keypoints"]
    if len(points) != 17 or any(len(p) != 3 for p in points):
        raise ValueError("Sono necessari 17 keypoint COCO con x, y, confidence")
    joints = [
        dict(
            index=i,
            name=name,
            x_px=p[0],
            y_px=p[1],
            x=p[0] / width,
            y=p[1] / height,
            confidence=p[2],
            visible=p[2] >= threshold,
        )
        for i, (name, p) in enumerate(zip(JOINTS, points))
    ]
    bones = []
    for a, b in BONES:
        if not (joints[a]["visible"] and joints[b]["visible"]):
            continue
        dx, dy = points[b][0] - points[a][0], points[b][1] - points[a][1]
        bones.append(
            dict(
                start=a,
                end=b,
                length_px=math.hypot(dx, dy),
                angle_image_degrees=math.degrees(math.atan2(dy, dx)),
                vector_normalized=[dx / width, dy / height],
            )
        )
    angles = {}
    for name, a, b, c in [
        ("leftKnee", 11, 13, 15),
        ("rightKnee", 12, 14, 16),
        ("leftElbow", 5, 7, 9),
        ("rightElbow", 6, 8, 10),
    ]:
        if not all(joints[i]["visible"] for i in [a, b, c]):
            angles[name] = None
            continue
        u = [points[a][i] - points[b][i] for i in range(2)]
        v = [points[c][i] - points[b][i] for i in range(2)]
        denominator = math.hypot(*u) * math.hypot(*v)
        angles[name] = (
            math.degrees(
                math.acos(
                    max(-1, min(1, sum(x * y for x, y in zip(u, v)) / denominator))
                )
            )
            if denominator
            else None
        )
    root = [(points[11][axis] + points[12][axis]) / 2 for axis in (0, 1)]
    shoulders = [(points[5][axis] + points[6][axis]) / 2 for axis in (0, 1)]
    scale = math.hypot(shoulders[0] - root[0], shoulders[1] - root[1])
    body_valid = all(joints[i]["visible"] for i in (5, 6, 11, 12)) and scale > 1
    body_coordinates = [
        [(p[0] - root[0]) / scale, (p[1] - root[1]) / scale]
        if body_valid and joints[i]["visible"]
        else None
        for i, p in enumerate(points)
    ]
    return dict(
        joints=joints,
        bones=bones,
        joint_angles_degrees=angles,
        root_image_px=root if body_valid else None,
        torso_scale_px=scale if body_valid else None,
        body_coordinates=body_coordinates,
        body_coordinate_definition="hip midpoint origin, torso-length scale, image axes",
        coordinates="image-2d: x right, y down; normalized separately by width and height",
        metric_coordinates=False,
        confidence=person["confidence"],
    )


def build_skeletons(store, identifier, predictions):
    frames = {f["frame_id"]: f for f in store.frames(identifier)}
    output = []
    for prediction in predictions:
        frame = frames[prediction["frame_id"]]
        # Person index is local to the frame: no unproven persistent identity.
        people = [
            dict(
                person_index=i,
                track_id=None,
                identity_status="untracked",
                bbox=person["bbox"],
                **geometry(person, frame["width"], frame["height"]),
            )
            for i, person in enumerate(prediction["people"])
        ]
        output.append(
            dict(
                schema_version=1,
                experiment_id=identifier,
                frame_id=frame["frame_id"],
                sha256=frame["sha256"],
                timestamp_ns=frame["aligned_timestamp_ns"],
                timestamp_domain="pc-unix-ns",
                sync_uncertainty_ns=frame["sync_uncertainty_ns"],
                width=frame["width"],
                height=frame["height"],
                people=people,
                model=prediction["model"],
            )
        )
    return output



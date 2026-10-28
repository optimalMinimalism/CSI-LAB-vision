"""Camera recording clock metadata, frame gaps and coverage."""


def coverage_gaps(times, start_ns, end_ns, tolerance_ns):
    """Intervalli inclusivi senza frame entro tolleranza, non stime di frame persi."""
    if tolerance_ns < 0 or end_ns < start_ns:
        raise ValueError("Intervallo o tolleranza non validi")
    cursor, gaps = start_ns, []
    for timestamp in sorted(times):
        left, right = (
            max(start_ns, timestamp - tolerance_ns),
            min(end_ns, timestamp + tolerance_ns),
        )
        if right < cursor or left > end_ns:
            continue
        if left > cursor:
            gaps.append(dict(start_ns=cursor, end_ns=left - 1))
        cursor = max(cursor, right + 1)
    if cursor <= end_ns:
        gaps.append(dict(start_ns=cursor, end_ns=end_ns))
    return gaps


def recording_manifest(store, identifier, max_distance_ns=100_000_000):
    frames = store.frames(identifier)
    times = [f["aligned_timestamp_ns"] for f in frames]
    gaps = []
    previous = None
    for frame in sorted(frames, key=lambda f: f["frame_id"]):
        expected = previous["frame_id"] + 1 if previous else 0
        if frame["frame_id"] != expected:
            gaps.append(
                dict(first_missing_id=expected, last_missing_id=frame["frame_id"] - 1)
            )
        previous = frame
    item = store.get(identifier)
    telemetry = item.get("camera_telemetry", {})
    if not frames and telemetry.get("captured", 0):
        gaps.append(dict(first_missing_id=0, last_missing_id=telemetry["captured"] - 1))
    last_frame_id = max((f["frame_id"] for f in frames), default=-1)
    if frames and telemetry.get("captured", 0) > last_frame_id + 1:
        gaps.append(
            dict(
                first_missing_id=last_frame_id + 1,
                last_missing_id=telemetry["captured"] - 1,
            )
        )
    coverage_end = (
        item["end_ns"]
        if item["end_ns"] is not None
        else max([item["start_ns"], *times])
    )
    return dict(
        experiment_id=identifier,
        timestamp_domain="pc-unix-ns",
        offset_sign="pc minus camera",
        hash_representation="exact transmitted JPEG bytes",
        max_distance_ns=max_distance_ns,
        frames=len(frames),
        sequence_gaps=gaps,
        video_coverage_gaps=coverage_gaps(
            times, item["start_ns"], coverage_end, max_distance_ns
        )
        if item["camera_enabled"]
        else [],
        coverage_start_ns=item["start_ns"],
        coverage_end_ns=coverage_end,
        video_coverage_requested=item["camera_enabled"],
        camera_ack=item.get("camera_ack"),
        camera_telemetry=telemetry,
        camera_clock_measurements=[
            dict(
                frame_id=f["frame_id"],
                clock=f["clock"],
                sync_uncertainty_ns=f["sync_uncertainty_ns"],
                clock_step=f["clock_step"],
                fresh=f["clock_fresh"],
            )
            for f in frames
        ],
        caveat="Le incertezze sono limiti dichiarati/misurati, non una prova dell'istante di esposizione.",
    )

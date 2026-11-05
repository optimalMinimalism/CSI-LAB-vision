"""Authenticated routes for the separate [experimental] distance feature."""
import json


def route(handler, method, parts, service):
    if parts == ["api", "experimental", "distance"]:
        session = service.distance()
        if method == "POST":
            enabled = handler.body().get("enabled")
            if type(enabled) is not bool:
                raise ValueError("enabled must be boolean")
            session.enable(enabled)
        output, status = session.snapshot()
        handler.send(200, dict(**status, result=output[2] if output else None))
    elif parts == ["api", "experimental", "distance", "calibration"]:
        session = service.distance()
        if method == "POST":
            data = handler.body()
            if data.get("clear") is True:
                session.clear_calibration()
                handler.send(200, dict(cleared=True))
                return
            raw = service.distance_frame()
            if not raw:
                raise ValueError("A current camera frame is required for calibration")
            from .distance import FloorCalibration
            candidate = FloorCalibration(data)
            if not candidate.matches(raw[2]):
                raise ValueError("Calibration source/size differs from the current camera")
            handler.send(200, session.calibrate(data))
        else:
            handler.send(200, session.snapshot()[1]["calibration"])
    elif method == "GET" and parts == ["api", "experimental", "distance", "preview"]:
        output, _ = service.distance().snapshot()
        if output:
            result = output[2]
            metadata = {key: result.get(key) for key in ("frame_id", "width", "height", "source_id", "floor_distances", "inference_ms", "experimental")}
            metadata["timestamp_ns"] = str(result["timestamp_ns"])
            lidar = result.get("rangefinder")
            if isinstance(lidar, dict):
                metadata["rangefinder"] = {key: lidar.get(key) for key in ("state", "model", "distance_m")}
            handler.send(200, output[1], "image/jpeg", headers={
                "X-Frame-Metadata": json.dumps(metadata, separators=(",", ":"), allow_nan=False),
                "X-Frame-Id": str(result["frame_id"]),
                "X-Timestamp-Ns": str(result["timestamp_ns"])})
        else:
            handler.send(204, b"", "image/jpeg")
    else:
        handler.send(404, dict(error="Unknown experimental distance route"))

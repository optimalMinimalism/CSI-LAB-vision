"""Disposable, low-priority inference process for [experimental] floor distances."""
import os
import time


def run_worker(connection, factory, nice_target):
    try:
        # Priority/thread settings belong exclusively to this spawned process.
        # Refuse to start if lowering OS priority fails.
        os.setpriority(os.PRIO_PROCESS, 0, nice_target)
        nice = os.getpriority(os.PRIO_PROCESS, 0)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CSI_VIDEO_INFERENCE_THREADS"):
            os.environ[key] = "1"
        os.environ["CSI_VIDEO_DETECTOR_CONFIDENCE"] = os.environ.get("CSI_VIDEO_DISTANCE_CONFIDENCE", ".5")
        import cv2
        import numpy as np
        from .distance import DistanceModel, FloorCalibration, estimate, render
        cv2.setNumThreads(1)
        if factory is DistanceModel:
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        connection.send(dict(kind="started", nice=nice))
        model = None
        while True:
            job = connection.recv()
            if model is None:
                model = factory()
            metadata, payload, calibration_data = job
            started = time.monotonic()
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid experimental distance image")
            calibration = FloorCalibration(calibration_data) if calibration_data else None
            people, objects = model.predict(image)
            measurements = estimate(people, objects, metadata, calibration)
            measurements["experimental"] = True
            jpeg = render(image, people, measurements, calibration)
            result = dict(metadata, experimental=True, floor_distances=measurements,
                          inference_ms=(time.monotonic()-started)*1000,
                          inference_completed_ns=time.time_ns())
            connection.send(dict(kind="result", jpeg=jpeg, result=result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(dict(kind="error", error=str(exc)[:300]))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()

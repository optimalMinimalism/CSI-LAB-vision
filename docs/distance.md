# [experimental] Person/object floor-distance screen

## What this measures

The user confirmed a fixed camera and distances across one flat floor. The separate `/distance` screen, visibly labelled **[experimental]**, provides approximate **floor-anchor separation**, not a full 3D reconstruction or the shortest physical clearance between bodies.

Its dedicated `DistanceModel` combines the existing COCO pose estimator with a local YOLO object detector. The estimator maps the midpoint of two visible ankles and the bottom centre of a supported object's bounding box onto a measured floor plane. Euclidean separation in that plane gives metres. No new neural network was trained and no monocular metric-depth model is claimed.

The object base is a proxy for a ground contact, and ankles are above the soles. Occlusion, box localisation error, shoe geometry and distortion introduce errors. Metric accuracy remains unknown until measured on the installed system. `accuracy_m` is explicitly null; `estimate` is true.

The TF02-Pro reading appears separately as distance along its beam. A sensor mounted a few centimetres below a camera has a different origin; angular misalignment also changes which surface it sees. That single return is not a depth map and is not assigned to every detected object. This implementation does **not** fuse it into floor geometry. Measured camera intrinsics, LiDAR/camera translation and orientation, beam/scene association, and timing uncertainty would be required before using it for that purpose.

## Run and configure

Use the normal backend and camera recording-mode client. Restart the backend after updating source, then open `http://BACKEND:8765/distance` or select **Distanze [experimental]** from the video page. Authenticate with the existing token. Click **Avvia modello distanze** to request frames and start this mode. No model loads merely because the page is opened or authenticated.

The backend needs the existing inference extra and two local checkpoints:

```text
CSI_VIDEO_DISTANCE_POSE_MODEL=./models/yolo11n-pose.pt
CSI_VIDEO_DISTANCE_MODEL=./models/yolo11n.pt
CSI_VIDEO_DISTANCE_CONFIDENCE=0.5
```

These experimental checkpoint/confidence settings are independent of the primary model configuration. The object checkpoint is task `detect`; the other is task `pose`. Both are present in the current accessible development environment. Model loading never downloads weights automatically. Processing uses CPU and 416-pixel model input; original frame coordinates are retained. Per-frame computation is capped to eight people and sixteen supported object anchors for geometry. Additional detections can be omitted. Runtime model throughput has not been benchmarked on Raspberry Pi. The intended deployment keeps inference on the backend computer and capture/UART acquisition on the Pi.

## Calibrate a fixed installation

1. Keep the camera fixed, including lens, focus, zoom, crop and resolution. Use a flat floor. Ideally correct significant lens distortion before measuring; this implementation works on delivered images and does not perform lens undistortion.
2. Mark a rectangle on the floor. Measure its width and depth with a tape measure; enter metres. Choose a region large enough to cover the planned person and object positions.
3. In the live distance screen, select **Calibra il pavimento**. The displayed frame freezes while calibration points are marked.
4. Click four corners in perimeter order: near-left, near-right, far-right, far-left. Enter measured width/depth.
5. Select only classes actually resting on this same floor. Confirm the fixed-camera/flat-floor/object-contact assumptions, then save.
6. Check known separations at several points **inside** the rectangle using a tape measure. This independent check is necessary: the four fitted corner points do not measure out-of-sample accuracy.
7. Recalibrate after camera movement or any change of crop, zoom, resolution or floor geometry. **Rimuovi calibrazione** immediately disables metric publication and removes the saved map.

Calibration uses a planar homography; [OpenCV explains its planar-scene assumptions](https://docs.opencv.org/4.6.0/d9/dab/tutorial_homography.html). Projection is restricted to the marked quadrilateral; there is no extrapolation outside it. Calibration is tied to source ID and exact image width/height. Motion with unchanged source ID/resolution cannot be detected automatically; the operator must invalidate the calibration. Use distinct source IDs if switching physical cameras.

Supported classes are chair, couch, bench, bed, dining table, refrigerator, suitcase and potted plant. Selecting a class is an assumption about actual placement, not model evidence of floor contact. A plant on a table must not be included. The measurements do not account for object footprints, hidden legs, pose identity, room obstacles, sloped floors or occluded contact points. There are no collision alerts or physical clearance guarantees.

## Operation and lifecycle

The browser renews its own authenticated five-second lease. It never renews, disables or replaces the main preview lease. Opening/login alone starts no model. A background supervisor reads the most recent immutable JPEG through a non-blocking snapshot; the main frame handler never submits to experimental inference. The distance screen and `/api/experimental/distance...` endpoints have their own output and calibration state. Main `/api/camera/preview`, analysis, recording and export results receive no experimental overlays or result fields.

The model runs in a **spawned process**, not in the main vision thread or process. Linux niceness is raised by ten (capped at 19) before child model startup. Torch and OpenCV inference are restricted to one thread in that process; the main process's priority, model settings and thread counts are not modified. Inference is limited to at most one job per second, with only one job in flight. Worker startup/crash/errors are contained and reported in the experimental status. A stalled job times out after fifteen seconds.

**Primary work always wins.** An active main preview/stream lease, recording, or analysis/export pauses experimental inference. The supervisor terminates an in-flight experimental process on its next scheduling check (normally within 50 ms, plus process cleanup); the API immediately hides its result. The page shows the reason and resumes best-effort work after the primary activity ends. The normal frame submission, preview selection, posture/fall processing, recording and export implementations remain in place. Experimental inference does not replace or suspend them. Background normal live inference continues receiving frames as before.

Stopping or expiring the distance lease terminates the child and releases its models; a tab that stops renewing expires after five seconds. Resuming requires reloading the child models. Hidden tabs stop renewing. This is shared experimental state; any authenticated client can stop it or replace its calibration. Clearing the main preview lease does not clear the experimental lease, or vice versa.

Process isolation and lower scheduling priority do not provide a hard CPU/RAM quota or a real-time latency guarantee. The secondary process still uses shared machine resources while it is permitted to run. Validate resource use on the intended backend; inference on the Pi 3 B+ has not been validated.

There is one pending frame and one published JPEG/result. Freshness is checked against receipt and publication times. Source timestamps and distance results travel with the same preview JPEG in its metadata header, avoiding image/result mismatches. Stale, disabled or missing-camera output is removed from the screen. No recordings are created by preview or calibration. Floor-distance results are currently live only; they are not added to dataset exports or burned into recorded MP4s.

## API

All `/api/...` routes require the existing Bearer token. `/distance` is the public login shell.

- `GET /api/experimental/distance`: enabled state, `experimental: true`, status, `paused_by`, error, worker PID/niceness, calibration and latest fresh result.
- `POST /api/experimental/distance` with `{"enabled":true}`: start/renew the lease. `false` disables it.
- `GET /api/experimental/distance/preview`: annotated JPEG or 204; `X-Frame-Metadata` contains the matching measurements and frame identity.
- `GET /api/experimental/distance/calibration`: current calibration or null.
- `POST /api/experimental/distance/calibration`: validate and atomically save a calibration to `CSI_VIDEO_DATA/floor-calibration.json`. A current matching camera frame is required. `{"clear":true}` removes calibration.

Calibration input example for a 320×240 image (sample coordinates only, not calibration for your installation):

```json
{
  "source_id": "0", "image_width": 320, "image_height": 240,
  "points_px": [[20,200], [300,200], [300,80], [20,80]],
  "width_m": 4, "depth_m": 3,
  "floor_labels": ["chair"], "floor_confirmed": true
}
```

Four finite, convex, non-degenerate image points, positive measured dimensions and explicit floor confirmation are required. Self-intersecting regions and out-of-image points are rejected. Results use `uncalibrated`, `calibration_mismatch` or `ready` geometry states; missing/hidden feet and invalid/out-of-region bases produce no metric pair. Pose/object indices are local to each frame, not tracked identities. Observed ranges do not supply an accuracy estimate for the geometric model.

## Validation

```bash
.venv312/bin/python -m unittest discover -s tests -p testDistance.py -v
CSI_VIDEO_TEST_PYTHON=.venv312/bin/python node tests/testDistanceBrowser.mjs
```

Thirteen Python tests cover known synthetic floor distance, absent/mismatched calibration, occluded/raised feet, clipped/outside bases, unsupported classes, invalid calibration, lazy model loading, save/reload/remove, lease expiry, discarded in-flight outputs, fault isolation, authenticated APIs, independent leases, unchanged main frame/preview routing, lower child priority, worker-crash containment and interruption when primary work starts. The browser test covers login, opt-in inference, no uncalibrated metres, four-point calibration, a known synthetic distance, mobile layout, calibration removal, stop and logout. It also checks that the main annotated preview retains presence overlays, recordings receive frames while the experiment is paused, and experimental inference resumes afterwards.

Both local checkpoints were smoke-tested on a blank synthetic image. This verifies loading and execution only. It is not a hardware test or a real-scene accuracy benchmark. Physical floor measurements, camera mounting, LiDAR alignment, lens distortion, occlusions and long-running performance still require validation at the installation.

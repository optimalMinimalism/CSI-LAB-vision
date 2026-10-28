# CSI-LAB-CV-ML

CSI-LAB-CV-ML runs on the computer. It receives camera JPEGs, stores recordings, runs YOLO pose inference, serves the dashboard/API and exports videos. Its primary role is to supply timestamped camera frames to CSI-LAB through the HTTP API. Radio CSI belongs entirely to CSI-LAB.

## Processing and configuration

One YOLO pose extraction supplies the following analysis features:

| Stage | Result |
|---|---|
| Presence | `PRESENT` / `ABSENT`, person count and detector confidence. |
| Posture | Per-person `STANDING`, `SITTING`, `LYING_DOWN`, or `UNKNOWN`; geometric heuristic score. |
| Skeleton | 17 COCO joints, visible bone segments, body coordinates and joint angles. |
| Fall detection (Experimental) | Optional single-person temporal heuristic; reports suspected falls. Off by default. |

Live processing retains the latest pending frame and latest result. Person indices identify positions within one result, not tracked identities. Selected overlays always use the image corresponding to that inference result.

Start with `csi-video --env .env server`. Main settings:

| Setting | Default |
|---|---|
| `CSI_VIDEO_HOST`, `CSI_VIDEO_PORT` | `127.0.0.1`, `8765` |
| `CSI_VIDEO_TOKEN` | Required, ≥16 characters |
| `CSI_VIDEO_DATA` | `./data` |
| `CSI_VIDEO_MODEL` | `./models/yolo11n-pose.pt` (local checkpoint required) |
| `CSI_VIDEO_LIVE_INFERENCE` | `true` |
| `CSI_VIDEO_LIVE_IMAGE_SIZE` | `416` |
| `CSI_VIDEO_MONOCHROME` | `true`; grayscale replicated into three channels |
| `CSI_VIDEO_INFERENCE_THREADS` | `2` |
| `CSI_VIDEO_DETECTOR_CONFIDENCE` | `0.25` |

## Dashboard and recording API

Base URL: `http://COMPUTER-IP:8765`. All routes below require `Authorization: Bearer <token>`. POST JSON uses `Content-Type: application/json`. `{id}` is a session UUID.

| Method | Route | Request → response |
|---|---|---|
| GET | `/api/health` | Adapter, active session, camera heartbeat and backend process UUID. |
| GET | `/api/clock` | Clock calibration timestamps and process UUID. |
| GET | `/api/camera/control` | Camera capture instructions; see [camera contract](CSI-LAB-camera.md). |
| POST | `/api/camera/preview` | `{"enabled":true}` renews a five-second lease; `false` disables it. Also accepts camera JPEG uploads. |
| GET | `/api/camera/preview?mode=all` | Latest JPEG plus `X-Frame-Metadata`; `204` if absent/stale (two-second freshness window). Default mode: `presence`. |
| GET | `/api/camera/stream?mode=all` | Annotated MJPEG stream; keeps camera preview capture active. |
| GET | `/api/camera/fall-detection` | Shared setting: `enabled` and `experimental`. |
| POST | `/api/camera/fall-detection` | `{"enabled":true}` or `{"enabled":false}` → updated setting; live inference required. |
| GET | `/api/camera/analysis` | Latest full analysis and worker status. |
| GET | `/api/recordings` | Array of video-only session metadata. |
| POST | `/api/recordings` | `{"title":"Session A"}` → `201`, session metadata including `id`. |
| GET | `/api/recordings/{id}` | `{"metadata":{...},"frames":120,"labels":[]}`. |
| POST | `/api/recordings/{id}/stop` | `{}` → final metadata, camera acknowledgement or warning. |
| GET | `/api/recordings/{id}/frames?offset=0&limit=100` | `{"total":120,"frames":[...]}`; limit 1–1000. |
| GET | `/api/recordings/{id}/frames/{frame_id}` | Original JPEG. |
| POST | `/api/recordings/{id}/frames` | Recorded JPEG upload; same contract as `/api/experiments/{id}/frames`. |
| GET | `/api/recordings/{id}/video?mode=all` | MP4 download after stop. Default mode: `all`. |

Overlay modes: `presence`, `posture`, `skeleton`, comma-separated combinations such as `presence,skeleton`, `all`, or `none` (timestamp only, plus the fall banner when live fall detection is enabled). MP4 also accepts `raw` for original images without annotations. The browser allows independent toggles and an **All** button; download uses the selection at click time.

MP4 generation requires `ffmpeg`. It reanalyzes saved frames, preserves their aligned timing and can take longer than playback. Only one analysis/export job runs at a time; another receives `400`. The dashboard shows elapsed time and prevents duplicate downloads while waiting. Loading/error/disabled live inference falls back to raw preview, so absence of annotations is not an absence classification.

## Live video integration with CSI-LAB

Use **`GET /api/camera/stream?mode=all`** to consume the annotated preview as a continuous MJPEG stream. This is a derived live view: CSI-LAB can display it without repeating YOLO inference or drawing the overlays. For a single JPEG, use `GET /api/camera/preview?mode=all`; the snapshot route requires an active recording or a renewed preview lease.

Set `CSI_VIDEO_HOST` to a network interface reachable from CSI-LAB if the applications run on different computers; the default `127.0.0.1` only accepts local connections. The camera client must also be running and configured with the same token. Use a trusted network or HTTPS.

```bash
# Set the backend address and token in the consuming client's environment.
export BASE='http://COMPUTER-IP:8765'
export CSI_VIDEO_TOKEN='<token-from-backend-.env>'

# Receive annotated live video. Ctrl+C disconnects.
curl --fail --no-buffer \
  -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$BASE/api/camera/stream?mode=all" --output preview.mjpeg
```

The response has content type `multipart/x-mixed-replace; boundary=csi-frame`. Each part contains a JPEG and these headers:

| Part header | Meaning |
|---|---|
| `Content-Type` | `image/jpeg` |
| `Content-Length` | JPEG size in bytes |
| `X-Frame-Id` | Camera frame ID |
| `X-Timestamp-Ns` | Original camera acquisition timestamp in integer nanoseconds |

The stream supports the preview overlay modes listed above; `all` is the default. It renews preview capture automatically, so CSI-LAB does not need to send separate preview lease requests. Live inference must be enabled. Frames are latest-only, up to 10 FPS: missing frames and slow consumers do not create a complete recording history. After 15 seconds without fresh analyzed frames the connection closes; consumers should reconnect. The backend permits four simultaneous stream clients and returns HTTP 503 when full.

The dashboard and stream share inference and the rendered JPEG cache for matching overlay modes. This saves duplicate inference and overlay rendering in CSI-LAB. Every viewer still requires network transfer and JPEG decoding, and MJPEG does not compress changes between frames. A browser `<img>` cannot supply the Bearer header directly: use an authenticated CSI-LAB backend proxy or a client supporting request headers. Keep the token out of URLs.

For a completed recording, download the derived MP4 instead (requires `ffmpeg`):

```bash
# ID is the stopped recording UUID.
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$BASE/api/recordings/$ID/video?mode=all" --output recording.mp4
```

Use the [recorded frame API](#frame-delivery-to-csi-lab) for dataset synchronization and original timestamps. Live video does not replace the original frame records.

## Fall detection (Experimental)

Click the yellow **Rilevamento cadute (Experimental)** button beside the posture and skeleton buttons to enable or disable detection. The button becomes brighter when enabled. This setting is shared by all viewers, starts disabled after a backend restart, and is independent of the **All** overlay selector. Disabling it clears the temporal history and previous live result.

When enabled, both the dashboard preview and MJPEG stream show a yellow status banner. A suspected fall displays a red **POSSIBILE CADUTA** banner and a red person box. The live analysis panel also shows the state. These indicators are transient, not latched alerts.

```bash
# Enable detection; send false to disable it.
curl --fail -X POST \
  -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true}' "$BASE/api/camera/fall-detection"

# Read whether the experimental detector is enabled.
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$BASE/api/camera/fall-detection"

# Read the latest result, including result.fall_detection.
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  -H 'X-Timestamp-Format: string' "$BASE/api/camera/analysis"
```

The setting response is `{"enabled":true,"experimental":true}`. The analysis result contains a `fall_detection` object with those fields plus `state`:

| State | Meaning |
|---|---|
| `DISABLED` | Detection is switched off. |
| `UNKNOWN` | Insufficient or ambiguous evidence, including a lying person without an observed fall transition. |
| `MONITORING` | A single person has an upright posture available as a temporal reference. |
| `SUSPECTED_FALL` | A rapid downward transition to sustained lying met the experimental thresholds. |

The heuristic requires an upright-to-lying transition within 1.5 seconds, a downward bounding-box center movement of at least 20% of the upright box height, and lying observations spanning at least 0.3 seconds. It resets when people are absent or multiple, posture is uncertain, camera source changes, capture time fails to increase or the frame gap exceeds one second. It does not track identities. Camera movement, occlusion, person replacement and slow inference can produce false or missed detections. This is an uncalibrated single-person experiment, not a validated safety alarm.

**Dataset behavior:** recording saves original JPEGs and timestamps, but fall detections are not saved as events or labels. The API exposes only the latest live result; MP4 exports do not include fall detection. A consumer needing persistent fall annotations must store results with their recording and frame identifiers and provide a review process.

## Analysis data format

`GET /api/camera/analysis` returns the following shape. Arrays are abbreviated here; a skeleton contains all 17 joints.

```json
{
  "status": "ready",
  "error": null,
  "processed": 120,
  "skipped": 3,
  "result": {
    "frame_id": 42,
    "timestamp_ns": "1792960000000000000",
    "presence": {"state":"PRESENT","person_count":1,"confidence":0.94},
    "fall_detection": {"enabled":true,"experimental":true,"state":"MONITORING"},
    "postures": [{"person_index":0,"posture":"STANDING","heuristic_score":0.88}],
    "skeletons": [{
      "person_index": 0,
      "joints": [{"index":0,"name":"nose","x_px":120,"y_px":80,"x":0.1875,"y":0.1667,"confidence":0.98,"visible":true}],
      "bones": [{"start":5,"end":7,"length_px":60,"angle_image_degrees":90,"vector_normalized":[0,0.125]}]
    }],
    "inference_ms": 90.0,
    "queue_ms": 1.0,
    "backend_ms": 91.0
  }
}
```

This example assumes `X-Timestamp-Format: string`; without it, JSON nanoseconds are integers. Results also carry input frame metadata, receipt timestamps and `inference_completed_ns`. Worker status is `loading`, `waiting`, `ready`, `error`, or `disabled`; `result` can be `null`. Presence confidence is `null` when no people are detected.

Image coordinates start at the top left; x increases rightward, y downward. Joint `x/y` are pixel coordinates divided by width/height; bone endpoints are joint indices. Joints are marked visible at confidence ≥0.5; bones require both endpoints visible. Geometry is 2D, without depth or persistent identity.

The preview JPEG header contains frame identifiers, dimensions, source and string timestamps, `preview_mode`, plus an `analysis` summary: worker status/error, presence, up to ten postures, `fall_detection`, `skeleton_count`, `inference_ms`, `backend_ms`. Full skeleton geometry is available from `/api/camera/analysis`.

## Frame delivery to CSI-LAB

CSI-LAB retrieves frame metadata from `GET /api/recordings/{id}/frames` and the original JPEG from `GET /api/recordings/{id}/frames/{frame_id}`. Both work during recording and after stop. This is a pull API; the backend does not automatically push frames to CSI-LAB.

The frame-list response is `{"total":120,"frames":[...]}`. Each frame retains the [camera metadata](CSI-LAB-camera.md#frame-wire-format) and adds `aligned_timestamp_ns`, `sync_uncertainty_ns`, `clock_fresh`, `clock_step` and `received_ns`. JPEGs are stored as SQLite blobs with their original bytes and hash. CSI-LAB performs any association with its own radio measurements using these timestamps and uncertainties.

```bash
# BASE is the backend URL; ID and FRAME_ID identify a camera session and frame.
curl -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  -H 'X-Timestamp-Format: string' \
  "$BASE/api/recordings/$ID/frames?offset=0&limit=100"
curl -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$BASE/api/recordings/$ID/frames/$FRAME_ID" -o frame.jpg
```

## Offline visual analysis

The recording namespace also supports `POST /{id}/inference`, `GET /{id}/predictions`, `GET /{id}/skeletons?frame_id=42`, `POST /{id}/review`, `GET /{id}/audit`, `GET /{id}/timing` and `GET /{id}/export`. These paths are relative to `/api/recordings`. `/api/experiments` is an alias used by existing camera clients.

Offline predictions and skeletons reference original frame IDs, JPEG hashes and aligned camera timestamps. Timing reports contain camera clock information and frame gaps. Dataset ZIPs contain original JPEGs, JSONL frame/prediction/skeleton records, labels, audit history, camera timing and an original-view MP4 when encoding succeeds.

The complete route list, review schema and archive layout are in the [API reference](../../api.md).

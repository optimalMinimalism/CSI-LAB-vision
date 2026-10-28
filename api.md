# Camera frame and visual-analysis API

Base URL: `http://COMPUTER-IP:8765`. Every API request requires `Authorization: Bearer <CSI_VIDEO_TOKEN>`. JSON POST bodies use `Content-Type: application/json`. The root `/` serves the dashboard.

This API captures, stores and supplies camera data to CSI-LAB. Radio ingestion, acquisition control, radio processing and frame-to-radio association are outside this application.

## Routes

`{id}` is a camera recording UUID. `/api/experiments` is an alias for `/api/recordings`, retained for camera clients.

| Method | Route | Request / response |
|---|---|---|
| GET | `/api/health` | Active camera session, last camera poll and backend process UUID. |
| GET | `/api/clock` | `receive_ns`, `send_ns`, `server_id`, `clock_unstable`. |
| GET | `/api/camera/control` | `experiment_id`, `server_id`; idle responses include `preview_enabled`. |
| POST | `/api/camera/preview` | `{"enabled":true}` renews a five-second lease; `false` disables it. Camera uploads use JPEG instead. |
| GET | `/api/camera/preview?mode=all` | JPEG with `X-Frame-Metadata`; `204` when no fresh frame exists. Default mode: `presence`. |
| GET | `/api/camera/stream?mode=all` | Annotated MJPEG stream; keeps preview capture active. Bearer authentication required. |
| GET | `/api/camera/fall-detection` | `{"enabled":false,"experimental":true}`; shared backend setting. |
| POST | `/api/camera/fall-detection` | `{"enabled":true}` or `{"enabled":false}`; live inference must be enabled. |
| GET | `/api/camera/analysis` | Latest visual result and worker status. |
| GET | `/api/recordings` | Array of camera session metadata. |
| POST | `/api/recordings` | `{"title":"Session A","camera_enabled":true}` → `201` and session metadata. |
| GET | `/api/recordings/{id}` | `{"metadata":{...},"frames":120,"labels":[]}`. |
| POST | `/api/recordings/{id}/stop` | `{}` → final camera session metadata. |
| GET | `/api/recordings/{id}/frames?offset=0&limit=100` | `{"total":120,"frames":[...]}`; limit 1–1000. |
| GET | `/api/recordings/{id}/frames/{frame_id}` | Original JPEG bytes. |
| POST | `/api/recordings/{id}/frames` | JPEG upload with metadata header; `{"duplicate":false}` or `true`. |
| POST | `/api/recordings/{id}/cameraStopped` | Final camera counters, after upload queue drains. |
| GET | `/api/recordings/{id}/video?mode=all` | Annotated MP4 after stop. Default mode: `all`. |
| POST | `/api/recordings/{id}/inference` | `{}`; saved visual predictions, skeletons and activity labels after stop. |
| GET | `/api/recordings/{id}/predictions` | Saved predictions. |
| GET | `/api/recordings/{id}/skeletons?frame_id=42` | Saved skeletons; filter is optional. |
| POST | `/api/recordings/{id}/review` | Review body below; updates a visual label and audit history. |
| GET | `/api/recordings/{id}/audit` | Visual-label review history. |
| GET | `/api/recordings/{id}/timing` | Camera clock measurements, frame sequence gaps and video coverage gaps. |
| GET | `/api/recordings/{id}/export` | Camera dataset ZIP after stop. |

MP4 modes: `presence`, `posture`, `skeleton`, comma-separated combinations, `all`, `none` (timestamp only), and `raw` (no annotations). Preview accepts the same modes except `raw`. Annotation never modifies stored originals.

Analysis and export are synchronous. Only one job runs at a time; use a timeout suitable for the recording length. MP4 requires `ffmpeg` and recomputes visual analysis on saved frames.

## Consume live video and experimental fall detection

Use the backend address reachable from CSI-LAB. The default bind address is localhost; configure `CSI_VIDEO_HOST` to expose it on the intended network interface. Supply the token from `.env` through an Authorization header on every request. Never put the token in a URL.

```bash
# Set these in the consuming client's environment.
export CSI_VIDEO_SERVER='http://127.0.0.1:8765'
export CSI_VIDEO_TOKEN='<token-from-backend-.env>'

# Receive the annotated live MJPEG stream (Ctrl+C to disconnect).
curl --no-buffer --fail \
  -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$CSI_VIDEO_SERVER/api/camera/stream?mode=all" --output preview.mjpeg

# Enable experimental fall detection (use false to disable).
curl --fail -X POST \
  -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true}' \
  "$CSI_VIDEO_SERVER/api/camera/fall-detection"

# Read current setting or latest analysis.
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$CSI_VIDEO_SERVER/api/camera/fall-detection"
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  -H 'X-Timestamp-Format: string' \
  "$CSI_VIDEO_SERVER/api/camera/analysis"

# Download a completed recording as MP4; replace RECORDING_ID.
curl --fail -H "Authorization: Bearer $CSI_VIDEO_TOKEN" \
  "$CSI_VIDEO_SERVER/api/recordings/RECORDING_ID/video?mode=all" \
  --output recording.mp4
```

The live response uses `multipart/x-mixed-replace; boundary=csi-frame`. Each JPEG part has `Content-Length`, `X-Frame-Id` and `X-Timestamp-Ns` (camera acquisition timestamp). It reuses the dashboard's inference and rendered-frame cache for the same mode. Default mode is `all`; presence, posture, skeleton, combinations and `none` are also supported. Enabled fall overlays appear in every live mode, including `none`.

Streams are latest-only, at most 10 FPS, with four simultaneous clients maximum (HTTP 503 when full). No fresh analyzed frame for 15 seconds closes the stream; consumers must reconnect. Live inference must be running. A browser `<img>` cannot attach the Bearer header: CSI-LAB should use an authenticated backend proxy or a client supporting headers. Network transfer and JPEG decoding remain per viewer.

`GET /api/camera/analysis` returns `result: null` while frames or inference are unavailable. Otherwise `result.fall_detection` looks like:

```json
{"enabled":true,"experimental":true,"state":"SUSPECTED_FALL"}
```

States are `DISABLED`, `UNKNOWN`, `MONITORING`, and `SUSPECTED_FALL`. The yellow live banner shows monitoring or uncertainty; a suspected fall adds a red banner and box. The setting affects every viewer, defaults to off after restart, and disabling it clears temporal history. This single-person heuristic is uncalibrated and is not a validated safety alarm. Detections are transient, not saved events or dataset labels, and are not included in MP4 exports. See [thresholds and limitations](docs/wiki/overview.md#fall-detection-experimental).

## Frame upload and storage format

```http
POST /api/recordings/{id}/frames
Authorization: Bearer <token>
Content-Type: image/jpeg
X-Frame-Metadata: <compact JSON object>

<binary JPEG bytes>
```

The metadata contract is defined in [CSI-LAB-camera](docs/wiki/CSI-LAB-camera.md#frame-wire-format). Required fields include `frame_id`, `timestamp_ns`, `monotonic_ns`, `width`, `height`, `encoding`, `sha256` and a valid `clock` measurement for recorded frames. Preview uploads can use `clock=null`.

Stored metadata adds `aligned_timestamp_ns`, `sync_uncertainty_ns`, `clock_fresh`, `clock_step` and `received_ns`. Hashes identify the exact JPEG bytes. Limits: JPEG ≤8 MiB, dimensions ≤4096 × 2160, recorded-frame metadata header ≤12,000 characters.

Frame IDs are non-negative sequences. Identical retries preserve metadata and bytes and are idempotent. Conflicting IDs, out-of-order new frames and new frames after closure are rejected. Drops leave sequence gaps.

Stop acknowledgement:

```json
{"captured":120,"queue_dropped":2,"rejected":1,"network_retries":3,"capture_errors":0}
```

## Time and analysis formats

`*_ns` fields are integer nanoseconds. Uploaded Unix timestamps belong to the camera clock; aligned timestamps belong to the backend computer's clock. Monotonic timestamps are local to one host. Camera timestamps measure read completion, not exposure. Clock calibration measures an offset and uncertainty; unknown acquisition uncertainty remains unknown.

Send `X-Timestamp-Format: string` to receive JSON nanoseconds as decimal strings without JavaScript precision loss. JSON POST timestamps must remain integers. Preview header timestamps are already strings.

`GET /api/camera/analysis` returns `status`, `error`, `processed`, `skipped` and `result`. Status is `loading`, `waiting`, `ready`, `error` or `disabled`; the result can be null. A successful result contains frame metadata, `presence`, `postures`, `skeletons`, `inference_completed_ns`, `inference_ms`, `queue_ms` and `backend_ms`. See the [analysis schema](docs/wiki/CSI-LAB-CV-ML.md#analysis-data-format).

Skeletons contain 17 COCO joints. Pixel coordinates use top-left origin, x right and y down; normalized coordinates divide by image width/height. Bone endpoints are joint indices. Person indices are local to a frame and do not establish persistent identities.

Live results are latest-only. Offline inference stores predictions and skeletons linked to original frame IDs, hashes and aligned timestamps. Posture rules and temporal activity estimates are geometric heuristics.

Review body:

```json
{"label_id":0,"activity":"SITTING","status":"MANUALLY_VALIDATED","reviewer":"name","note":"Visual verification"}
```

Activities: `STANDING`, `SITTING`, `LYING_DOWN`, `WALKING`, `FALLING`, `FALLEN`, `UNKNOWN`. Manual review accepts `MANUALLY_VALIDATED`, `NEEDS_REVIEW` or `REJECTED`. Automatic labels also use `AUTO_VALIDATED`; calibration and visual evidence govern that status.

## Dataset ZIP

Each archive has an `experiment-{id}/` directory containing original `frames/*.jpg`, `frames.jsonl`, `predictions.jsonl`, `skeletons.jsonl`, `labels.json`, `audit.json`, `timing.json`, `metadata.json`, and `video.mp4` when encoding succeeds. JSONL contains one JSON object per line. The archive's video uses original frames; annotated MP4 is available through `/video`.

MP4 uses variable frame timing from aligned timestamps and 100 ms final-frame padding. Original JPEGs and metadata are the authoritative recording.

## Errors

Errors return `{"error":"message"}`. `400` means invalid input/state or busy analysis/export; `401` means missing/invalid token; `404` means missing route/resource; `500` means internal failure. Backend logs contain internal error details.

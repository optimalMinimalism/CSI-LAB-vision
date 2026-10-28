# CSI-LAB-camera

CSI-LAB-camera is the capture and transport component. It runs on a Raspberry Pi or computer, supports OpenCV/USB and Picamera2 capture, and sends JPEGs to the backend. Its frames are made available to CSI-LAB through the backend API. It runs no YOLO inference and has no role in radio CSI acquisition, processing or control.

## Installation and operation

The `CSI-LAB-camera/` directory can be installed independently. On Raspberry Pi OS with Python 3.11–3.12:

```bash
sudo apt install python3-venv python3-numpy python3-opencv
cd CSI-LAB-camera
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --no-deps .
export CSI_VIDEO_TOKEN='<backend-token>'
csi-framegrabber --mode recording --source 0 --server http://COMPUTER-IP:8765
```

For a Pi CSI camera, install `python3-picamera2` and add `--backend picamera2`. On a desktop, a normal virtual environment and `python -m pip install .` install the camera dependencies.

| Option | Default / meaning |
|---|---|
| `--mode recording` | Dashboard-controlled preview and recording, port 8765. Set this explicitly. |
| `--mode live` | CLI default: standalone detection stream, usually port 8766. |
| `--source` | `CSI_VIDEO_DEVICE`, otherwise `0`; camera index or supported source path. |
| `--backend` | `CSI_VIDEO_CAPTURE_BACKEND`, otherwise `opencv`; also `picamera2`. |
| `--width`, `--height`, `--fps` | Environment `CSI_VIDEO_WIDTH/HEIGHT/FPS`, otherwise 640 × 480 at 10 FPS. |
| `--server` | `CSI_VIDEO_SERVER`, otherwise `http://127.0.0.1:8766`; explicitly use 8765 for recording. |

CLI options override environment defaults. Stop with Ctrl-C. Preview keeps only the newest pending frame. Recording uses a bounded upload queue; dropped frames leave gaps in the frame IDs. Upload retries preserve the original metadata and JPEG bytes.

## Backend API used by recording mode

These routes are hosted by CSI-LAB-CV-ML; the camera exposes no HTTP server. Every request uses the shared Bearer token.

| Method | Route | Purpose / format |
|---|---|---|
| GET | `/api/clock` | Clock calibration: returns `receive_ns`, `send_ns`, `server_id`, `clock_unstable`. |
| GET | `/api/camera/control` | Poll desired capture state: `experiment_id`, `server_id`; idle responses also include `preview_enabled`. |
| POST | `/api/camera/preview` | Upload temporary JPEG with `X-Frame-Metadata`. No persistent recording. |
| POST | `/api/experiments/{id}/frames` | Upload recorded JPEG with the same metadata header. Returns `{"duplicate":false}` or `true` for an identical retry. |
| POST | `/api/experiments/{id}/cameraStopped` | Acknowledge stopped capture and drained queue using final counters. |

When `experiment_id` is non-null, recording takes priority over preview. When it is null, `preview_enabled=true` permits temporary capture. The browser renews the lease; the camera only supplies frames.

## Frame wire format

```http
POST /api/experiments/{id}/frames
Authorization: Bearer <token>
Content-Type: image/jpeg
X-Frame-Metadata: <compact JSON object below>

<binary JPEG bytes>
```

Illustrative metadata; replace the hash and clock identity with real values:

```json
{
  "frame_id": 42,
  "timestamp_ns": 1792960000000000000,
  "monotonic_ns": 123456789000,
  "source_id": "0",
  "timestamp_source": "opencv-read-completion",
  "capture_uncertainty_ns": null,
  "width": 640,
  "height": 480,
  "encoding": "JPEG",
  "sha256": "<SHA-256 of exact JPEG bytes>",
  "clock": {
    "server_id": "<backend-process-uuid>",
    "offset_ns": 1000000,
    "uncertainty_ns": 2000000,
    "measured_camera_ns": 1792959999000000000,
    "measured_pc_ns": 1792959999001000000,
    "measured_monotonic_ns": 122456789000,
    "rtt_ns": 4000000,
    "source": "four-timestamp-http"
  }
}
```

`frame_id` is a non-negative sequence number; `sha256` hashes encoded bytes, not decoded pixels. `clock.offset_ns` maps camera Unix time to backend Unix time. Clock measurements identify a particular backend process and must be refreshed after its restart. Preview metadata may have `clock=null`; persistent frame uploads require a valid measurement.

Limits: JPEG ≤8 MiB, dimensions ≤4096 × 2160; recorded-frame metadata header ≤12,000 characters. Identical recorded-frame retries are idempotent; conflicting IDs are rejected. Late new frames after session closure are rejected.

Stop acknowledgement:

```json
{"captured":120,"queue_dropped":2,"rejected":1,"network_retries":3,"capture_errors":0}
```

## Optional standalone live mode

`--mode live --server http://COMPUTER-IP:8766` continuously sends the same JPEG/header contract to `POST /frame`, with no recording lifecycle. The separate `csi-backend server` detection receiver responds `{"ok":true}`. It is distinct from the dashboard API and does not provide its recording/export features.

See [system overview](overview.md) and [backend API](CSI-LAB-CV-ML.md).

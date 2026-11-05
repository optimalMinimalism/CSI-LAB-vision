# CSI-LAB-vision

CSI-LAB-vision supplies timestamped camera frames to CSI-LAB through an HTTP API. It captures and stores original JPEGs, runs visual inference, and provides preview and MP4 inspection tools.

Radio acquisition, radio data storage, signal processing and association with camera frames belong to CSI-LAB. This application processes camera data and optional Benewake distance telemetry.

## Components

```text
CSI-LAB-camera → JPEG + acquisition metadata → CSI-LAB-CV-ML → frame API → CSI-LAB
                                                       ├─ presence / posture / skeleton
                                                       └─ browser preview / MP4 / frame ZIP
```

- **CSI-LAB-camera** runs on the camera host. It captures images using OpenCV or Picamera2, timestamps them and uploads JPEGs with bounded buffering and retries.
- **CSI-LAB-CV-ML** runs on the computer. It stores camera recordings, exposes frames and metadata, and runs YOLO pose inference.
- **CSI-LAB** retrieves frames through the API and associates them with its own experiment data. Frame delivery uses HTTP retrieval; there is no automatic outbound push.

## Backend setup

Python 3.11–3.12, a local YOLO pose checkpoint and `ffmpeg` are required for visual inference and MP4 export.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[inference]'
cp .env.example .env
# Set a private token and the local model path in .env.
csi-video --env .env server
```

`CSI_VIDEO_MODEL` defaults to `./models/yolo11n-pose.pt`. The checkpoint must already exist; runtime inference does not download it. `.env` is loaded only through `--env`. Existing environment variables take precedence.

The backend listens on `127.0.0.1:8765` by default. Set `CSI_VIDEO_HOST=0.0.0.0` for access from another machine. All API calls require `Authorization: Bearer <CSI_VIDEO_TOKEN>` with a token of at least 16 characters. Use a trusted network or HTTPS.

## Camera setup

Install [CSI-LAB-camera](CSI-LAB-camera/README.md) on the camera host, then run:

```bash
export CSI_VIDEO_TOKEN='<same-token-as-backend>'
csi-framegrabber --mode recording --source 0 --server http://COMPUTER-IP:8765
```

Use `--backend picamera2` for a Raspberry Pi ribbon camera. Capture defaults to 640 × 480 at 10 FPS; `--width`, `--height` and `--fps` override these settings.

## Frame delivery

1. Create a camera session with `POST /api/recordings`, supplying `{"title":"Session A"}`. The response contains its UUID.
2. Read metadata using `GET /api/recordings/{id}/frames?offset=0&limit=100`.
3. Retrieve JPEG bytes using `GET /api/recordings/{id}/frames/{frame_id}`.
4. Stop capture with `POST /api/recordings/{id}/stop` and `{}`.

Frames are available during capture and after stop. The backend stores original JPEG bytes, SHA-256 hashes and acquisition metadata in `CSI_VIDEO_DATA/experiments.sqlite`. Frame IDs preserve capture gaps; identical upload retries are idempotent.

`timestamp_ns` is camera Unix time at read completion. `aligned_timestamp_ns` maps it to the backend computer's Unix clock using HTTP clock measurements. Neither is a measured sensor-exposure timestamp. Unknown capture uncertainty remains unknown. Browser clients should request `X-Timestamp-Format: string` and use `BigInt` for nanoseconds.

## Visual analysis and inspection

The separate **Distanze [experimental]** screen at `/distance` runs an on-demand pose/object pipeline for a fixed camera and a flat floor. Mark a measured floor rectangle before enabling metric estimates between visible ankle positions and object bases. Its separate low-priority process pauses during main preview, recording or analysis/export; it never replaces main live analysis. It displays TF02-Pro beam telemetry separately. [Distance-screen setup, calibration, API and limitations](docs/distance.md).

One YOLO pose extraction feeds presence detection, geometric posture estimates and 17-joint COCO skeletons. Live inference keeps only the latest pending frame and latest result. It does not provide a persistent annotation for every captured frame or track identities across frames.

Open `http://COMPUTER-IP:8765` to inspect the camera and start or stop recordings. Overlay toggles can be combined; **All** selects presence, posture and skeleton together. Preview alone creates no saved recording.

CSI-LAB can consume the annotated live video through `GET /api/camera/stream?mode=all` with the Bearer token. The yellow **Rilevamento cadute (Experimental)** button controls the shared live fall detector. Its API is `GET/POST /api/camera/fall-detection`; results appear in `GET /api/camera/analysis` under `result.fall_detection`. Fall detection adds a live status banner and a red warning for suspected falls; detections are not stored as dataset labels or included in MP4 exports. See [video integration and curl examples](api.md#consume-live-video-and-experimental-fall-detection).

MP4 export reanalyzes every saved frame and burns the selected overlays and capture timestamp into the video. It requires `ffmpeg` in the backend PATH, can take longer than playback, and can temporarily slow live inference. The download shows elapsed time and prevents duplicate requests. `mode=none` includes only timestamps; `mode=raw` omits annotations.

Offline `/inference`, `/skeletons`, `/predictions`, `/review` and `/export` routes provide saved visual-analysis artifacts, review history and a camera dataset ZIP. `/timing` reports frame gaps and camera clock information. `/api/experiments` and `/api/recordings` address the same camera sessions.

## Documentation and checks

- [System overview](docs/wiki/overview.md)
- [Camera and wire format](docs/wiki/CSI-LAB-camera.md)
- [Backend API and analysis format](docs/wiki/CSI-LAB-CV-ML.md)
- [API reference](api.md)

```bash
python -m unittest discover -s tests -p 'test*.py' -v
CSI_VIDEO_TEST_PYTHON=./.venv/bin/python node tests/testBrowser.mjs
```

The browser check requires Node 22+ and Chromium. `CSI_VIDEO_TEST_BROWSER` selects the Chromium executable. Optional model and native-worker checks use `CSI_VIDEO_TEST_MODEL` and `CSI_VIDEO_TEST_NATIVE`.

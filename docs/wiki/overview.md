# CSI-LAB-vision: camera frames for CSI-LAB

CSI-LAB-vision supplies **timestamped camera frames to CSI-LAB** as a visual reference for its experiments. It captures images, makes the original frames available through an HTTP API, and can attach visual information such as presence, posture and skeletons.

**CSI-LAB owns all radio CSI work:** acquisition, storage, processing and association with camera data. CSI-LAB-vision's responsibility is camera/video data only. It must not receive, generate, store, process or control radio CSI.

## Data flow

```text
Camera / Raspberry Pi             Computer                         Consumer
CSI-LAB-camera ── JPEG + time ──► CSI-LAB-CV-ML ── frames + time ──► CSI-LAB
                                      │
                                      ├─ visual analysis: presence, posture, skeleton
                                      └─ supporting tools: preview and MP4 download
```

| Component | Responsibility |
|---|---|
| [CSI-LAB-camera](CSI-LAB-camera.md) | Capture images, timestamp them, encode JPEGs and upload them with retries and bounded buffering. |
| [CSI-LAB-CV-ML](CSI-LAB-CV-ML.md) | Receive and retain camera frames, expose them to CSI-LAB through the API, and run optional visual analysis. |
| CSI-LAB, external to this application | Consume frames and their timestamps, manage the experiment and perform any association with its own radio measurements. |

## How CSI-LAB gets frames

1. The camera client sends original JPEG bytes and acquisition metadata to CSI-LAB-CV-ML.
2. The backend groups recorded frames under a session UUID and preserves each frame's ID, timestamp and SHA-256 hash.
3. CSI-LAB reads the session's frame list and downloads the JPEGs it needs through the authenticated API.
4. CSI-LAB uses the supplied timestamps to associate visual references with its own experiment data. Radio measurements stay in CSI-LAB.

The implemented delivery mechanism is **HTTP retrieval by CSI-LAB**. A live annotated MJPEG stream is also available for display (see below). There is currently no automatic outbound frame push or CSI-LAB callback.

| Operation | API |
|---|---|
| Start a camera recording | `POST /api/recordings` with `{"title":"Session A"}`; response contains `id`. |
| List recorded frame metadata | `GET /api/recordings/{id}/frames?offset=0&limit=100` |
| Retrieve an original frame | `GET /api/recordings/{id}/frames/{frame_id}` → JPEG |
| Stop the camera recording | `POST /api/recordings/{id}/stop` with `{}` |
| Watch annotated live video | `GET /api/camera/stream?mode=all` → MJPEG stream; Bearer token required. |
| Enable experimental fall detection | `POST /api/camera/fall-detection` with `{"enabled":true}` (`false` disables it). |
| Read the latest visual analysis | `GET /api/camera/analysis` → JSON |

CSI-LAB can list and retrieve frames while a recording is active or after it stops. Frame lists are paginated; frame IDs may contain gaps if capture or upload drops frames. Live analysis is latest-only, so it is not a complete per-frame annotation history.

## Frame and timestamp contract

Each frame carries its original JPEG bytes, `frame_id`, `timestamp_ns`, `monotonic_ns`, dimensions, source information and `sha256`. Stored metadata also provides `aligned_timestamp_ns`, mapping camera Unix time to the backend computer's Unix clock, and synchronization uncertainty when known.

- Timestamps use integer nanoseconds. Camera timestamping measures read completion, not sensor exposure.
- Monotonic timestamps belong to one host and cannot be compared across machines.
- CSI-LAB must account for clock domains and uncertainty when matching its measurements to images. An unknown uncertainty is not a calibrated timing guarantee.
- Original JPEGs and metadata are the reference data. An MP4 is a derived viewing artifact.

See the [camera page](CSI-LAB-camera.md#frame-wire-format) for the upload format and the [backend page](CSI-LAB-CV-ML.md) for API responses and visual-analysis formats.

## Supporting tools

The browser preview helps check camera position, connectivity and visual inference. Preview alone does not create a saved recording. Presence, posture and skeleton overlays can be combined or selected together with **All**.

### Fall detection (Experimental)

Enable **Rilevamento cadute (Experimental)** beside the preview overlay buttons to monitor suspected falls. The switch is off by default, applies to all viewers of this backend, and resets on server restart. Turning it off clears the temporal history and previous live result.

This single-person heuristic uses the existing pose model. It looks for an upright-to-lying transition within 1.5 seconds, a downward movement of at least 20% of the upright bounding-box height, and lying observations spanning at least 0.3 seconds. Missing people, multiple people, uncertain posture, a changed camera source, non-increasing capture times or gaps over one second reset the history. Someone already lying down is not classified as a fall. These thresholds are uncalibrated; camera movement, occlusion and person replacement can cause missed or false detections. This is an experimental visual annotation, not a validated safety alarm.

- `GET /api/camera/fall-detection` returns `enabled` and `experimental`.
- `POST /api/camera/fall-detection` with `{"enabled":true}` or `{"enabled":false}` changes the setting; live inference must be running.
- `GET /api/camera/analysis` includes `result.fall_detection` with `enabled`, `experimental` and `state`: `DISABLED`, `UNKNOWN`, `MONITORING` or `SUSPECTED_FALL`. The preview metadata contains the same field. A suspected fall is a transient result, not a saved event or latched alarm.

When enabled, fall detection adds a yellow status banner to live previews (including `mode=none`); a suspected fall adds a red banner and person box.

Fall detection runs only on live frames and is not included in MP4 exports or saved annotation history.

For setup instructions, request examples and response states, see [Fall detection (Experimental)](CSI-LAB-CV-ML.md#fall-detection-experimental).

### Live annotated stream for CSI-LAB

Connect to `GET /api/camera/stream?mode=all` using `Authorization: Bearer <CSI_VIDEO_TOKEN>`. The response is MJPEG (`multipart/x-mixed-replace; boundary=csi-frame`). Modes match the snapshot preview API: `presence`, `posture`, `skeleton`, comma-separated combinations, `all`, or `none`. Fall overlays follow the backend's shared fall detection switch.

Each part contains an `image/jpeg`, `Content-Length`, `X-Frame-Id`, and `X-Timestamp-Ns` (original camera acquisition time). Live inference must be enabled. Connecting keeps preview capture active without a separate preview lease request. The stream sends only fresh analyzed frames, up to 10 per second, skips intermediate frames for slow readers, and closes after 15 seconds without a fresh analyzed frame. Consumers should reconnect after disconnection. At most four stream clients are allowed; further connections receive HTTP 503.

See [Live video integration with CSI-LAB](CSI-LAB-CV-ML.md#live-video-integration-with-csi-lab) for connection examples and MP4 download instructions.

The stream and dashboard share inference and cached JPEG rendering for matching overlay modes. CSI-LAB can proxy this authenticated stream to its UI without repeating pose inference or overlay rendering. Each viewer still adds network traffic and JPEG decoding work; MJPEG does not provide inter-frame compression. The stream is a latest-only viewing artifact, not a complete dataset or an annotation history. Use the recorded-frame API and metadata for dataset synchronization. A plain browser `<img>` cannot supply the Bearer header; use an authenticated backend proxy or a client that supports request headers, rather than placing the token in the URL.

MP4 download helps inspect or share a completed camera recording. It can burn selected overlays and acquisition timestamps into the video. Export reanalyzes saved frames, requires `ffmpeg`, and may temporarily slow live inference. These tools support the frame-delivery workflow.

## Start and connect

On the backend computer, use Python 3.11–3.12, install `python -m pip install -e '.[inference]'`, configure `.env` from `.env.example`, provide a local pose checkpoint, and run:

```bash
csi-video --env .env server
```

On the camera host, after installing CSI-LAB-camera:

```bash
export CSI_VIDEO_TOKEN='<same-token-as-backend>'
csi-framegrabber --mode recording --source 0 --server http://COMPUTER-IP:8765
```

Set `CSI_VIDEO_HOST=0.0.0.0` for access from another host. Every API request requires `Authorization: Bearer <CSI_VIDEO_TOKEN>`; the token must contain at least 16 characters. JSON requests use `Content-Type: application/json`. Browser clients should request `X-Timestamp-Format: string` to preserve nanosecond precision.

The backend stores camera sessions, frame metadata and JPEG blobs in `CSI_VIDEO_DATA/experiments.sqlite`. The dashboard is available at `http://COMPUTER-IP:8765`. Use a trusted network or HTTPS.


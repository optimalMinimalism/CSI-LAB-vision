# CSI-LAB-camera

Camera capture and JPEG transport for CSI-LAB-vision. This package runs independently on the camera host and sends frames to CSI-LAB-CV-ML. Visual inference runs on the backend computer.

## Raspberry Pi installation

Python 3.11–3.12 and OpenCV are required. On Raspberry Pi OS:

```bash
sudo apt install python3-venv python3-numpy python3-opencv
cd CSI-LAB-camera
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --no-deps .
```

For a Raspberry Pi ribbon camera, install `python3-picamera2` and use `--backend picamera2`. USB cameras use the default OpenCV backend.

On a desktop, use a normal virtual environment and `python -m pip install .` to install NumPy and OpenCV with the package.

## Dashboard recording and frame delivery

```bash
export CSI_VIDEO_TOKEN='<same-token-as-backend>'
csi-framegrabber --mode recording --source 0 --server http://COMPUTER-IP:8765
```

The client follows the backend's camera controls. Preview keeps the newest pending frame; recordings use a bounded upload queue and retry identical JPEG bytes and metadata. The backend exposes recorded frames to CSI-LAB through its frame API.

Capture defaults to 640 × 480 at 10 FPS. Configure `--width`, `--height`, `--fps`, `--source` and `--backend` as needed. CLI options override `CSI_VIDEO_WIDTH`, `CSI_VIDEO_HEIGHT`, `CSI_VIDEO_FPS`, `CSI_VIDEO_DEVICE` and `CSI_VIDEO_CAPTURE_BACKEND`. `CSI_VIDEO_SERVER` supplies the default backend address. Stop with Ctrl-C.

The token must contain at least 16 characters and match the backend. Use a trusted network or HTTPS. Timestamps measure camera read completion; sensor exposure timing requires hardware measurements.

## Standalone detection stream

```bash
csi-framegrabber --mode live --source 0 --server http://COMPUTER-IP:8766
```

Live mode is the CLI default. It sends frames continuously to the standalone detection receiver's `POST /frame` endpoint. Dashboard recording requires `--mode recording` and the port-8765 backend.

## Deployment

`camera.env.example` contains environment defaults. `csi-framegrabber.service` is a systemd template for `/opt/CSI-LAB-camera`, a `csi-video` user in the `video` group, and `/etc/csi-framegrabber.env`.

The optional C++ capture worker lives in `native/` and requires CMake, a C++ compiler, OpenCV development libraries and OpenSSL development libraries. The Python client can run without this worker.

## Optional Benewake distance telemetry

The independent C++ UART reader supports the TF02-Pro's standard 9-byte, centimeter output format, with optional TFmini-S and TF-Luna profiles. It is disabled unless you explicitly configure a device. Camera operation continues if the device or worker is missing, disconnected, or produces invalid measurements.

Build just the distance reader on Linux (including Raspberry Pi OS), without OpenCV development libraries or PySerial:

```bash
sudo apt install build-essential cmake
cmake -S native -B native/build -DCSI_BUILD_CAMERA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target benewakeCapture -j2
```

From this directory, enable it for dashboard recording:

```bash
csi-framegrabber --mode recording --server http://COMPUTER-IP:8765 \
  --benewake-device /dev/serial0 --benewake-model tf02-pro \
  --benewake-worker "$PWD/native/build/benewakeCapture"
```

The same options work in `--mode live`. The existing `CSI_VIDEO_TOKEN` requirement still applies. The default `tf02-pro` profile accepts 0.10–40 m, rejects signal strength ≤60 or 65535, and rejects error-distance values such as 4500 cm. Select `tfmini-s` (0.10–12 m) or `tf-luna` (0.20–8 m) explicitly for those models. These are acceptance limits, not guaranteed usable range for every surface or lighting condition. The UART packet cannot identify the model or distinguish centimeter output from the sensor's optional millimeter mode: the sensor must be configured for centimeters.

`CSI_BENEWAKE_DEVICE`, `CSI_BENEWAKE_MODEL`, `CSI_BENEWAKE_BAUD`, and `CSI_BENEWAKE_WORKER` provide environment defaults. These also work with the existing `python -m backend.cli --env ... camera` deployment. For systemd, place the executable somewhere readable outside your home directory, such as `/usr/local/bin/benewakeCapture`, and use an absolute worker path. The service templates include the `dialout` group; verify the group that owns your serial device on your OS.

Frame metadata, recorded-frame API responses, and live inference results expose a `rangefinder` object. Its `distance_m` is populated only in the `available` state. Other states include `connected` (awaiting a reading), `disconnected`, `invalid`, `stale`, and `unmatched`. When disabled, the field is omitted entirely. The native reader can also run directly as `native/build/benewakeCapture /dev/serial0 115200 tf02-pro`; stdout contains JSONL telemetry.

The reader validates checksums, rejects weak/saturated returns, handles fragmented packets, and reconnects once per second. Sensor history is limited to 64 samples. `--benewake-tolerance-ms` defaults to 100 ms and `--benewake-stale-ms` to 250 ms. They gate association by **host UART reception time**, which does not establish simultaneous sensor exposures. Unknown measurement timing uncertainty is explicitly `null`.

This change reports single-beam sensor distance alongside images. It does not assign that distance to a detected person, change the model, or estimate a 3D position. [Protocol, timing, hardware setup, and validation details](../docs/benewake.md).

[Wire format and camera API](../docs/wiki/CSI-LAB-camera.md) · [Backend API](../api.md)

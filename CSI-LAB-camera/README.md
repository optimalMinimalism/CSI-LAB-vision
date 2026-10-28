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

[Wire format and camera API](../docs/wiki/CSI-LAB-camera.md) · [Backend API](../api.md)

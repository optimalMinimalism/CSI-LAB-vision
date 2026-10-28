# Verification

Run camera, frame-storage, HTTP, visual-inference and export checks:

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```

The tests use synthetic camera frames and deterministic pose outputs. Optional real-model checks use `CSI_VIDEO_TEST_MODEL` and `CSI_VIDEO_TEST_IMAGE`; optional compiled-camera checks use `CSI_VIDEO_TEST_NATIVE`. MP4 checks require `ffmpeg` in PATH.

Browser validation requires Node 22+ and Chromium:

```bash
CSI_VIDEO_TEST_PYTHON=./.venv/bin/python node tests/testBrowser.mjs
```

`CSI_VIDEO_TEST_BROWSER` selects the Chromium executable. The browser check covers authentication, preview, combined overlays, camera-only recording, MP4 download, mobile layout and logout.

Physical-camera timing, sensor exposure uncertainty and posture accuracy require measurements on the deployment hardware.

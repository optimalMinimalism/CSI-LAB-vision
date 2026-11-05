# Benewake integration progress — 2026-10-30

## Completed and current state

- Optional standalone C++17 Linux UART acquisition with fixed-size decoding, checksum resynchronization, signal/range validation, reconnect, bounded output waits, and signal-driven cleanup.
- Python process supervision, bounded history, explicit freshness/tolerance policy, and per-frame snapshots in live and recording paths.
- Environment and CLI configuration; compatible with both existing camera deployment entry points. Serial access remains disabled unless a device is configured.
- Worker source packaging, systemd serial group membership, installation and protocol/timing documentation.
- Hardware-independent tests covering native pseudo-terminal acquisition and application integration.
- TF02-Pro support verified against the vendor's REV 01/04/2024 manual and made the default profile after the user identified their hardware. Accepts 0.10–40 m, conservatively rejects strength ≤60, rejects saturated returns and error-distance sentinels. Explicit TFmini-S/TF-Luna profiles remain available.
- Equal UART reception timestamps now select the newest buffered packet, addressing packets batched in one native read.

## Decisions

- No source was imported from dog-fight; only its inspected UART conventions informed discovery. The reader implements the documented Benewake standard 9-byte/cm protocol.
- User identified the physical sensor as TF02-Pro. Model profile must match hardware; the packet stream cannot identify the model or distinguish cm/mm mode. Only standard centimeter UART output is supported.
- Host UART reception timestamps are distinct from sensor measurement times. Measurement timing uncertainty remains unknown; no simultaneous-exposure or 3D-position claim is made.
- Distance is independent telemetry. This change does not implement object association or model fusion.

## Tests executed and known failures

- CMake Release build of `benewakeCapture`, with `CSI_BUILD_CAMERA=OFF`: successful.
- Latest dedicated suite after TF02-Pro changes: 17 tests passed; native tests compile the worker with `-Wall -Wextra -Wpedantic -Werror` and require no hardware. Native Release rebuild also succeeded.
- Initial integration full Python suite, before the TF02-Pro profile update: 71 tests run, 62 passed, 6 skipped, 3 failed. Localhost tests required execution outside the network sandbox. This full suite was not rerun for the model-profile update; the dedicated acquisition/integration suite above was rerun.
- All three failures were reproduced in an untouched HEAD snapshot under `/tmp/csi-baseline-gp_p00cu`:
  - `testCameraScope.CameraScopeTests.test_radio_and_external_model_routes_are_not_available`: existing health response includes `adapter` despite the test expecting its absence.
  - `testExtensions.SkeletonTests.test_export_survives_encoder_timeout`: unavailable `ffmpeg` produces `absent` before the mocked encoder timeout.
  - `testSystem.SystemTests.test_export_structure_integrity_and_video`: unavailable `ffmpeg` prevents video generation.
- Skips concern optional native OpenCV camera-worker tests, real model weights, and `ffmpeg`-dependent tests. They do not establish hardware validation.
- No baseline tests were disabled or changed to obtain a passing result.

## Outstanding and next action

Confirm the TF02-Pro is configured for standard UART centimeter output, rebuild the worker on the Pi, grant serial-device access, and launch with explicit `--benewake-device`, `--benewake-model tf02-pro`, and worker path. Validate known stationary distances, disconnect/reconnect while recording, and long-running resource behavior on the intended Pi. Actual hardware, ARM performance, timing uncertainty, and range accuracy remain unvalidated. Investigate baseline health/export failures separately. Any future model-distance work needs its own calibration and validation design.

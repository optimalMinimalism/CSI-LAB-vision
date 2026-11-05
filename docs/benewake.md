# Optional Benewake telemetry in CSI-LAB

## Scope and provenance

This integration adds independent distance acquisition and frame metadata. It does not modify perception algorithms or associate a beam with an object. No sensor commands, calibration changes, or firmware writes are sent.

The previously inspected dog-fight reader established relevant deployment conventions: `/dev/serial0`, 115200 baud, standard 9-byte UART output, bounded reads, checksum resynchronization, and distance conversion to meters. It did not establish the exact physical sensor model. The implementation here is independently written in C++ against the published Benewake protocol; no code or perception/fusion algorithms were imported.

Protocol references:

- [Benewake TF02-Pro manual](https://ai.benewake.com/en/resources/technical-documents/tf02-pro-user-manual/) (REV: 01/04/2024)
- [Benewake TFmini-S manual](https://ai.benewake.com/en/resources/technical-documents/tfmini-s-user-manual/)
- [Benewake TF-Luna manual](https://ai.benewake.com/en/resources/technical-documents/tf-luna-user-manual/)

## Architecture and decisions

```text
UART → benewakeCapture (C++17, termios/poll, fixed parser buffer)
     → bounded JSONL records over a process pipe
     → Benewake (Python supervisor, 64-sample history)
     → rangefinder frame metadata → existing upload/store/inference result
```

The UART reader is separate from camera acquisition so disconnection does not halt frames. Native UART handling uses Linux APIs and needs neither OpenCV nor PySerial. Process isolation permits terminating a stalled worker. The pipe writer has a two-second timeout; the supervisor restarts an exited worker after one second. Missing executables are reported as unavailable and retried. Serial reads use a 256-byte buffer; decoding retains at most nine bytes. Telemetry lines are bounded to 4096 bytes in the supervisor.

Latest matching telemetry is a snapshot captured once with a frame. Recording retries retain that exact snapshot with the original JPEG. It is never refreshed during upload, preserving existing idempotence semantics.

## Supported configuration

| Setting | Default | Meaning |
|---|---|---|
| `CSI_BENEWAKE_DEVICE` / `--benewake-device` | unset | Explicit opt-in UART path; no port scanning |
| `CSI_BENEWAKE_MODEL` / `--benewake-model` | `tf02-pro` | Validation profile, not automatic identification |
| `CSI_BENEWAKE_BAUD` / `--benewake-baud` | `115200` | Must match sensor configuration |
| `CSI_BENEWAKE_WORKER` / `--benewake-worker` | `benewakeCapture` | Executable path or name on PATH |
| `--benewake-tolerance-ms` | `100` | Maximum absolute frame/reception timestamp difference |
| `--benewake-stale-ms` | `250` | Maximum reception age at snapshot time |

Baud rates: 9600, 19200, 38400, 57600, 115200, 230400; 8 data bits, no parity, one stop bit, no flow control. The sensor must already output the standard nine-byte/cm protocol. ASCII, alternate units, timestamp-extended packets, I²C, SPI, Ethernet, and other Benewake models are not implemented.

Packet layout: two `0x59` header bytes, little-endian distance in cm, strength, encoded chip temperature, and the low byte of the sum of the preceding eight bytes. Temperature converts to °C as `raw/8 - 256`. Strength is a return-quality indicator, not calibrated confidence or distance variance.

| Profile | Accepted distance | Rejected strength |
|---|---|---|
| `tf02-pro` (default) | 0.10–40 m | ≤60 or 65535 |
| `tfmini-s` | 0.10–12 m | <100 or 65535 |
| `tf-luna` | 0.20–8 m | <100 or 65535 |

TF02-Pro error distances 4500, 65534, and 65535 cm are outside the accepted range and are rejected even with otherwise valid strength/checksum. The threshold of 60 is conservatively rejected at the boundary. Neither valid checksum nor strong return establishes range accuracy. The factory frame rate is 100 Hz; this integration leaves sensor configuration untouched. Real usable range depends on surface reflectivity and lighting. A TF02-Pro configured for alternate millimeter output uses the same packet layout and cannot be distinguished by this reader: configure centimeter output before enabling it.

## Hardware and deployment

Confirm model, power requirements, pinout, UART voltage, and configured output mode using the manual for your actual unit. Raspberry Pi GPIO UART is 3.3 V logic; do not apply a 5 V UART signal to it. Share ground and connect sensor TX to Pi RX. This read-only integration does not need a sensor RX connection. Enable the Pi UART and disable the serial console on that same UART. For USB serial adapters, prefer a stable `/dev/serial/by-id/...` path. No physical device was probed by the implementation tests.

The TF02-Pro manual specifies a 5–12 V supply and 3.3 V LVTTL UART. Power it from a suitable supply, not a GPIO signal pin. This profile covers the UART TF02-Pro in standard/cm mode; it does not implement TF02-Pro-W, RS-485 variants, or I²C acquisition.

Use the build and launch commands in [the camera README](../CSI-LAB-camera/README.md). Both existing camera service templates include serial-device group membership; group names and permissions must match the deployment OS. The root package and standalone camera package include native source files; the worker must still be compiled for the target ARM system. No prebuilt desktop binary should be copied to a Pi.

## Metadata and timing

Enabled frames include `rangefinder` with model, source path, state, distance, and error counters. Available readings also include sequence, worker generation, strength, chip temperature, host reception wall time, and `CLOCK_MONOTONIC` reception time in nanoseconds. Counters and sequences belong to a native worker lifetime; worker generation distinguishes supervisor restarts.

The supervisor chooses the closest buffered reception time to the camera frame's monotonic timestamp, subject to tolerance and age limits. Equal differences prefer the most recently received packet, including packets batched into one native UART read. This is a bounded, non-waiting association over already received samples; it does not wait for future measurements. Missing, invalid, disconnected, stale, or unmatched data yields `distance_m: null`. Invalid measurements and reconnections clear the usable sample history. Disabling the feature omits the entire object and starts no worker.

`frame_delta_ns` is LiDAR reception time minus frame time; `age_ns` is time since UART reception at snapshot creation. All packets delivered in the same native read receive the same reception timestamp. These timestamps do not reveal physical measurement time or sensor/USB buffering delay. Both `sensor_timestamp_ns` and `measurement_uncertainty_ns` are `null`; synchronization tolerance is an acceptance policy, not a measured synchronization-accuracy guarantee. Values remain historical snapshots in stored frames.

Example subset:

```json
{"rangefinder":{"enabled":true,"model":"tf02-pro","state":"available","distance_m":1.23,"timestamp_source":"host-uart-reception","measurement_uncertainty_ns":null}}
```

Existing frame metadata transport, recording storage, live JSONL results, and dashboard analysis retain this object. The separate [floor-distance screen](distance.md) also displays the beam reading alongside its geometric floor estimates; it does not use LiDAR telemetry to attribute ranges to detected objects.

## Validation and remaining work

Run from the repository root:

```bash
.venv312/bin/python -m unittest discover -s tests -p testBenewake.py -v
```

Tests compile the C++ worker with warnings treated as errors and use Linux pseudo-terminals to exercise fragmented/noisy input, checksum recovery, invalid returns, profile limits, delayed device appearance, physical-link simulation via PTY hangup, reconnection with partial-packet reset, process restart, and shutdown. Other tests cover stale/unmatched readings, bounded history, absent workers, continuing camera frames without LiDAR, transport/result metadata, configuration, recording persistence, and exact retries. Native cases require Linux and `g++`; they report skips if unavailable.

TF02-Pro-specific coverage checks accepted 0.10 m and 40 m boundaries, readings beyond the previous 12 m limit, signal strengths 60/61/99/65535, rejected range/error values, supervisor restart with a 23.45 m reading, and preserved legacy profiles. These are simulated-protocol tests, not physical range validation.

Not hardware-validated: UART wiring, actual sensor model/firmware, range accuracy, ARM runtime behavior, camera exposure-to-range timing, resource use on Raspberry Pi 3 B+, thermal behavior, or long unattended operation. No performance or synchronization-accuracy benchmark is claimed. Physical validation should measure known stationary distances and check disconnect/reconnect behavior while camera recording continues.

The separate floor-distance estimator uses measured planar calibration for a fixed camera. General 3D distance inference or LiDAR/vision fusion would require additional calibration, an explicit beam association model, compatible depth information with uncertainty, and independent validation. A single beam measurement is not a distance estimate for every detected object.

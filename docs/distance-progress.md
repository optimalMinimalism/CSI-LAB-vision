# [experimental] Floor-distance screen progress — 2026-10-30

## Scope and completed work

The user confirmed a fixed camera and floor distances between a person and objects. Implemented a separate `/distance` screen, on-demand pose/object model wrapper, measured four-corner floor calibration, geometry estimates, skeleton/base overlays, same-frame JPEG/result metadata, calibration save/load/remove, authenticated APIs, bounded latest-frame processing, fresh-result gating and five-second session leases.

Primary preview, frame submission and live analysis retain their original implementations. Experimental inference runs in a separate spawned process at lower priority and at most one job per second; it is paused/preempted during main preview/stream, recording or analysis/export. Camera/LiDAR acquisition code was not changed for this screen. TF02-Pro telemetry is displayed separately and is not fused or assigned to objects.

## Decisions and limitations

- Estimates are ankle-midpoint to object-base **anchor separations**, not nearest physical clearance or general 3D positions.
- No metres without a measured floor calibration. No geometry outside its quadrilateral or for a mismatched source ID/resolution.
- Only selected classes actually resting on the same floor may be measured. Bounding-box bases and ankles are approximate contacts.
- Local pretrained pose and detection checkpoints run on CPU at the backend, without automatic downloads. No new network was trained. Models load only after enabled input inside an isolated process; stopping, expiry or primary-work preemption terminates that process and releases its models. Main process priority and Torch settings are untouched.
- Camera movement, focus/crop/zoom changes, lens distortion and floor deviations require physical validation/recalibration. A source ID and resolution cannot detect all mounting changes.
- Foot/object indices are frame local. No tracking identities, safety alerts, collision decisions or distance recording/export integration was added.
- Screen, navigation, JPEG overlay and API status explicitly identify the feature as `[experimental]`; API handlers live in a separate module under `/api/experimental/distance`.
- Exact camera/LiDAR extrinsics are unknown; the informal vertical offset does not establish beam association. LiDAR-based scale fusion remains unimplemented.

## Validation performed

- Thirteen dedicated Python geometry, lifecycle and authenticated API tests passed, including a known synthetic 2 m separation, occlusion/clipping/outside-region rejection, calibration persistence/removal, lease expiry, independent leases, untouched main preview/frame routing, child niceness, worker-crash isolation and killed in-flight work when primary functions start.
- Chromium browser test passed: authentication, opt-in start, no uncalibrated metres, four-click calibration, synthetic distance, mobile layout, calibration removal, stop and logout, plus preserved main annotated preview and continued recording during experimental pauses. Latest screenshot retained in `/tmp/csi-video-browser-wMHKKh/distance.png`.
- Both actual local YOLO checkpoints loaded and executed on a blank synthetic image in the spawned worker at niceness 10. Parent process niceness remained 0 and Torch's thread count remained 4. This verifies isolation and software execution only, not person/object detection quality, hardware accuracy or performance.
- Latest full Python suite: 86 tests run; 77 passed, 6 skipped, and the same three previously baseline-confirmed health/export failures remain. See [Benewake progress](benewake-progress.md) for their exact names and baseline evidence.
- The initial preemption test hang came from calling `multiprocessing.Event.set()` after killing a child blocked on that event. The test fixture was corrected to simulate blocked work with sleep, then all isolation tests passed. Production inference does not share such an event with its child.
- Structural comparison against HEAD confirmed unchanged implementations of main `latest_frame`, `preview_packet`, `set_preview`, `enable_live`, `start`, `stop`, `infer`, and `camera_stopped`. Experimental API logic is in its own module; minimal server/service hooks provide authenticated dispatch, an independent capture lease, and read-only JPEG snapshots.
- `git diff --check` and Python syntax compilation passed.

## Outstanding and next recommended action

Restart the backend, open `/distance`, mark a measured rectangle and verify known separations with a tape measure at several interior positions. Test representative actual people/objects and mounting geometry. Hardware accuracy, latency under representative load, lens distortion, Pi/backend resource behaviour and long unattended operation remain unvalidated. See [distance.md](distance.md) for operating instructions and API details.

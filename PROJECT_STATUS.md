# OpenGaze project status

Last updated: 2026-10-25.

Rule for this file: **"verified" means someone ran it and saw it work, on the stated setup.**
Emulator and synthetic runs are labelled as such. Nothing here claims real-hardware
success until it has happened.

## Confirmed hardware (from the hardware-team handoff)

Two Inland ESP32-CAM (OV2640) boards stream the left and right eye over 2.4 GHz Wi-Fi. An
MPU-6050 on the **right** board is read over I²C (GPIO13/14, unverified pins). The MacBook Air
M2 15" webcam gives head pose. USB-C carries 5 V power only. Contract:
[`docs/hardware-interface.md`](docs/hardware-interface.md).

## Priorities

| # | priority | state |
|---|---|---|
| 1 | Simultaneous wireless video from both ESP32-CAMs | **Mac side done, emulator-verified.** `esp32` adapter (`opengaze/cameras/esp32.py`). Real boards: not tested |
| 2 | MPU-6050 telemetry from the RIGHT ESP32 | **Mac side done, emulator-verified.** JSON-over-UDP with subscribe handshake (`opengaze/imu/sources.py`). Real sensor: not tested |
| 3 | Live hardware diagnostics dashboard | **Done, emulator-verified** (`opengaze diagnostics`, `opengaze headless`, `--metrics`) |
| 4 | Pupil detection on both eye streams | Existing `ellipse_fit` detector runs on ESP32 frames (emulator: both eyes detected, median confidence 0.96 on *synthetic* eyes). Real OV2640 eye images: **not tested**; likely needs tuning (visible light, no IR) |
| 5 | Webcam head tracking (MediaPipe) | Existed before this work (Mode A). Works with the ESP32 profile as the `head` camera |
| 6 | Sensor fusion + gaze calibration | Existing Mode C pipeline: per-eye features + IMU rotation (yaw anchored to webcam) + webcam position → poly-ridge mapper; 9-point and adaptive calibration; leave-one-out validation. **5-point quick calibration: not implemented** |
| 7 | macOS cursor + clicking | Quartz cursor, One Euro smoothing, confidence freeze, pause / e-stop exist. **Click = double blink (left click).** By decision there is no dwell click and no long-blink click |
| 8 | AI accessibility / sponsor APIs | Cartesia voice exists (opt-in). Claude message composition: not started |

## Verified (and how)

| what | how | date |
|---|---|---|
| Full test suite: 474 tests pass (9 new in `tests/test_consumer.py`) | `pytest` in a Python 3.13 venv on the dev Mac | 2026-10-25 |
| EyeMouse (`opengaze app`) screens render to the consumer design: setup pages, calibration, practice, keyboard, settings | offscreen Qt screenshots; calibration against a synthetic camera | 2026-10-25 |
| EyeMouse native windows: setup window, non-activating keyboard panel, floating mic + pill, menu bar item | short Cocoa run on the dev Mac (no camera, no gaze) | 2026-10-25 |
| Full test suite: 465 tests pass (417 pre-existing + 48 new) | `pytest` in a Python 3.13 venv on the dev Mac | 2026-10-24 |
| Two emulated ESP32 eye streams received concurrently at ~20 fps with device timestamps and sequence numbers | `opengaze headless --profile configs/hardware/esp32-emulator.yaml` against `opengaze fake-esp32` | 2026-10-24 |
| Automatic reconnect after stream drops; drop counting from sequence gaps (5% injected → ~4–5% counted) | same, with `--drop-rate 0.05 --jitter-ms 30 --disconnect-every 4` | 2026-10-24 |
| IMU telemetry via subscribe handshake at 100 Hz; drops, invalid readings, reordering, bad packets counted | `tests/test_imu_telemetry.py`; headless run | 2026-10-24 |
| Diagnostics window renders both eyes side by side, IMU values + plots, input badges, EMULATED labels | offscreen Qt screenshot against the emulator | 2026-10-24 |
| ESP32 frames go through the Mode C gaze tracker (pupil features + IMU head rotation) | scripted `GazeTracker` run against the emulator | 2026-10-24 |

## Not verified: EyeMouse

- A full first-run setup with a real webcam, real permission prompts and a real calibration.
- Practice steps driven by real eye gestures (they are tested with synthetic mouse events).
- Voice typing end to end (needs a Cartesia key), and the mic's caret placement across apps
  (Notes, Messages, Safari, Chrome, Electron apps report caret bounds differently).
- The login item actually starting EyeMouse at the next login.

## Not verified (needs hardware)

- [ ] Firmware compiles (arduino-esp32 2.0.x / 3.x) and flashes on both boards
- [ ] Real MJPEG stream from each board at 320×240, ≥15 fps, on the demo network
- [ ] Both boards streaming at once without Wi-Fi contention dropping below 15 fps
- [ ] MPU-6050 found at 0x68 on GPIO13/14 with the camera running (I²C port conflict risk)
- [ ] IMU at 100 Hz alongside camera streaming; reduce to 50 Hz if not
- [ ] IMU units confirmed (diagnostics units check passes at rest)
- [ ] IMU `axes` mapping on the mounted board (nod/shake test in diagnostics)
- [ ] mDNS names resolve on the demo network (else use `--set …host=<ip>`)
- [ ] macOS Local Network permission granted to the launching app
- [ ] Pupil detection on real OV2640 close-up eye images; `detector_2d.options` tuning; `crop`
- [ ] Eye image orientation per side (`rotation_degrees`, `mirror_horizontal`)
- [ ] End-to-end calibration accuracy in Mode C with real glasses
- [ ] Power: both boards on one USB-C supply without brown-outs (hardware team)
- [ ] RF-exposure review for head-worn use (hardware team); until then bench-only at 20 cm

## Open tasks (software)

1. Flash and bring up real boards; record results above (hardware access needed).
2. Tune `ellipse_fit` for visible-light OV2640 images; add an eye-ROI crop editor if needed.
3. 5-point quick calibration option (handoff §7).
4. Decide whether IMU fusion beats webcam-only head compensation with real data (handoff §6:
   keep it only if experiments show an improvement). Compare with and without `imu:`.
5. Calibration UI pre-flight: camera connection, eye detection and head tracking checks
   before the 9 dots (handoff §7 items 1–3). Today these are in diagnostics, not in the
   calibration flow.

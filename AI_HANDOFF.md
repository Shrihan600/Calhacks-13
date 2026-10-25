# AI / developer handoff

Read this, then [`PROJECT_STATUS.md`](PROJECT_STATUS.md), then
[`docs/hardware-interface.md`](docs/hardware-interface.md) and
[`docs/architecture.md`](docs/architecture.md).

## Ground rules for this repo

- **Preserve working code.** The camera/IMU contracts (`CameraSource`, `CameraFrame`,
  `ImuSource`, `ImuSample`) are what everything downstream depends on. Add adapters; don't
  fork the pipeline.
- **Never claim hardware success without hardware.** Emulator (`opengaze fake-esp32`) and
  synthetic runs prove the Mac side follows the contract, nothing more. Label them.
- Keep `PROJECT_STATUS.md` (verified / not verified / open) and this file current.
- Keep the emulator in step with `docs/hardware-interface.md` and the firmware.
- No raw eye or face video leaves the Mac without explicit consent.
- **Clicking is a double blink (left click).** No dwell clicking and no long-blink click,
  even though the hardware handoff lists dwell. This was decided on 2026-10-24.

## Wired eye cameras (2026-10-25)

The hardware team moved the eye cameras from Wi-Fi to USB serial: one Seeed XIAO RA4M1
bridge per ESP32-CAM (OV3660), 921600 baud, CRC-framed JPEGs, MPU-6050 now on the **left**
board. They ship the receiver (`opengaze-hw/receiver`: `serial_client.SerialCamera`,
`imu_client.TelemetryReceiver`). We don't reimplement the protocol:

- `cameras/wired.py`: `SerialEyeSource` (`esp32_serial`) loads the receiver from
  `options.receiver_dir`, `$OPENGAZE_HW_RECEIVER` or `opengaze-hw/receiver`.
  One `WiredLink` per port is shared by the camera and the IMU (one program per port), is
  reference counted, and lingers 3 s so a reconnect doesn't reopen the port. Never call
  `TelemetryReceiver.start()` (that is the UDP listener).
- `imu/sources.py` `WiredImuSource` (`wired`) polls `last_imu` (g, deg/s); its port defaults
  to the left camera's.
- Clock: the camera and the IMU on a board share one `DeviceClockSync`. IMU packets set the
  offset, and frames are only mapped with it, so frame and IMU times stay aligned.
- CLI: `--left-serial` / `--right-serial` expand to `--set cameras.*.options.port=…`.
- **Port `auto`** (the profile's default): `find_board_port` runs a `GlassesScan` (every
  `/dev/cu.usbmodem*` / `usbserial*` opened as a lingering `WiredLink`) until a link's
  `TelemetryReceiver.dev[1|2]` has fresh packets (device id 1 = left, 2 = right; STATUS
  comes once a second). One search at a time (`_DETECT_LOCK`); the lingering links mean the
  camera then reuses the scan's port instead of reopening it. The auto IMU raises when its
  port vanishes, so `ImuRuntime` reopens and searches again.
- **EyeMouse always runs on the glasses** (`cli.GLASSES_PROFILE`), by product decision: no
  webcam-only choice for users. `--profile configs/hardware/webcam-only.yaml` is the dev
  fallback (no eye cameras → setup skips the glasses page). Setup's `GlassesPage` polls a
  `GlassesScan`, shows both eyes and unlocks Continue when both boards are found; the menu
  shows the current hardware problem.
- **Problems are visible** (user requirement): `consumer/problems.py` `find_problem` maps
  camera + IMU health to one `Problem` (error / warning) with user wording, keyed off the
  manager's `last_error` text (`isn't connected` = the port search found nothing, `receiver`
  = receiver missing). `ProblemMonitor` debounces 2 s. `EyeMouseApp.check_hardware` (1 s
  timer) drives `WarningBanner` (errors), `MenuBar.notify` + the ⚠ menu line, and the log.
  `cmd_app` shows a dialog for an invalid profile. Fullscreen calibration hides other
  windows, so `EyeMouseCalibration` paints the problem itself (`paint_problem`) and blocks
  Start on an error; the floating banner is hidden meanwhile. A failed calibration save is
  shown on its card. `needs_setup` reruns setup when `AppSettings.setup_hardware` isn't the
  current profile's file name. Keep those error strings in step with
  `cameras/wired.py`; `tests/test_wired_detect.py` runs the real manager to check them.
- The receiver lives in `opengaze-hw/` (the unzipped handoff). The adapter reads
  `Frame.img / dev_ts_us / host_ts / frame_id` and the `TelemetryReceiver.imu` deque
  (`ImuSample.seq / dev_us / host_ts / ax… / valid`), starting from the newest sample on open.
- **Receiver bug, patched locally:** `SerialCamera` and `TelemetryReceiver` subclass
  `threading.Thread` and had a `_handle` method; since Python 3.13, `Thread.__init__` sets
  `self._handle`, so every packet raised `'_ThreadHandle' object is not callable`. Renamed to
  `_handle_packet` in `opengaze-hw/receiver/`. `load_receiver` refuses an unpatched copy with
  that explanation. Tell the hardware team.
- **macOS + simulator:** pyserial sets 921600 with an ioctl that pseudo-terminals reject
  (`Errno 25`). Against the simulator use `--set cameras.{left,right}_eye.options.baud=230400`
  (the simulator paces itself at 921600 regardless). Real USB CDC ports are unaffected.

## EyeMouse, the consumer app (2026-10-25)

`opengaze app` (`opengaze/consumer/`) is the end-user app built from the "EyeMouse — consumer
app" Claude Design canvas. It is UI only, over the existing pipeline:

- `engine.py`: one `GazeEngine` owns the camera runtime and runs `desktop_mode`'s
  `GazeCursorOverlay` (restyled subclass). Calibration stops the cursor (both would call
  `tracker.step`). The consumer binds triple blink to `app: dictate`.
- `calibration.py`: `CalibrationWindow` subclass; only painting and the end of the flow change.
- `setup_window.py`, `practice.py`, `art.py`: the setup pages. Practice uses real mouse events,
  so it works with gaze gestures and with a helper's mouse alike.
- `menubar.py`, `voice_typing.py`, `keyboard.py`, `settings_window.py`, `app.py` (controller).
- `opengaze/control/ax.py`: Accessibility API through ctypes (PyObjC's ApplicationServices
  isn't a dependency): the permission prompt and the focused text caret's bounds.
- Fonts (Fredoka, Figtree; OFL) ship in `opengaze/consumer/fonts/`.
- `float_everywhere` now no-ops off the Cocoa platform, so floating windows can be tested
  offscreen.

## How the handoff's suggested module layout maps onto this codebase

The hardware team suggested `opengaze/hardware/{left_camera,right_camera,imu_receiver,
device_manager}.py`. The codebase already had a hardware-agnostic layer with those roles,
so the ESP32 support plugs into it instead of adding a parallel package:

| suggested | actual |
|---|---|
| `hardware/left_camera.py`, `right_camera.py` | one adapter, `cameras/esp32.py` (`source_type: esp32`), instantiated per camera from `configs/hardware/esp32-glasses.yaml` |
| `hardware/imu_receiver.py` | `imu/sources.py` `UdpImuSource` (JSON telemetry, subscribe handshake) + `imu/runtime.py` `ImuRuntime` |
| `hardware/device_manager.py` | `cameras/manager.py` `CameraManager` (reconnect, latest-frame slots, health) + `ImuRuntime` |
| `vision/pupil_detection.py` | `gaze/detector2d.py` (`ellipse_fit`), `gaze/eye_model3d.py` |
| `vision/head_tracking.py` | `vision/mediapipe_detector.py`, `vision/head_pose.py` |
| `vision/feature_fusion.py` | `gaze/tracker.py`, `vision/observations.py`, `imu/head.py` |
| `calibration/*` | `calibration/` (session, mappers, validation, adaptive) + `desktop/calibration_ui.py` |
| `control/cursor.py`, `dwell.py`, `smoothing.py` | `control/cursor.py`, `dwell.py`, `filter.py`, `safety.py` |
| `ui/diagnostics.py` | `desktop/diagnostics.py` |
| `configs/devices.yaml` | `configs/hardware/esp32-glasses.yaml` (+ `--set` / `--overlay`) |

## What this session added (2026-10-24)

- `opengaze/cameras/mjpeg.py`: stdlib MJPEG-over-HTTP reader (chunked encoding, parts with
  or without Content-Length, `X-Sequence` / `X-Timestamp-Us` / `X-Timestamp` headers).
- `opengaze/cameras/esp32.py`: `ESP32StreamSource`, which reads `/status`, can apply
  `/control` settings, reads the resolution back, maps device timestamps, and skips and
  counts corrupt JPEGs.
- `opengaze/cameras/timing.py`: `DeviceClockSync` (sliding-window min-offset clock mapping
  with reboot detection and 32-bit unwrap).
- `opengaze/imu/types.py` / `sources.py` / `runtime.py`: JSON telemetry, the subscribe
  handshake, sequence and drop counters, `valid: false`, device-clock mapping (batch-aware),
  a units sanity check, and device-clock rate measurement.
- `opengaze/debug/fake_esp32.py` + `opengaze fake-esp32`: emulated glasses with fault
  injection.
- `opengaze/debug/metrics.py`: input status badges and a JSONL metrics recorder.
- Diagnostics: an IMU tile (values, plots, counters, units warning), input badges, the left
  and right eyes side by side, EMULATED labels, and `--metrics`. `headless` shows the IMU too.
- CLI: `--set dotted.key=value` overrides on every profile command.
- `configs/hardware/esp32-glasses.yaml`, `esp32-emulator.yaml`.
- `firmware/opengaze_esp32cam/`: reference sketch, **never compiled**.
- Tests: `tests/test_esp32.py`, `tests/test_imu_telemetry.py`.
- Camera permission is now requested only for AVFoundation sources
  (`CameraSource.needs_camera_permission`), not for network streams.

## Gotchas found while building this

- macOS 15+ blocks LAN connections from apps without **Local Network** permission, and the
  only symptom is `EHOSTUNREACH`. The error message now says so.
- A batched UDP datagram shares one receive time across its readings. Map them only after
  updating the clock with the whole batch (fixed; covered by a test).
- Measuring IMU rate from host receive times overestimates it whenever the reader thread is
  starved. It is measured from device timestamps now.
- The stock Espressif `CameraWebServer` works with the adapter but sends no `X-Sequence`, so
  drops can't be counted.
- The dev Python env is a scratch venv (no repo `.venv`); see the README Setup section.

## Continuation prompt (paste into the next session)

> You are continuing work on OpenGaze (CalHacks 13), a macOS eye-gaze cursor for people with
> limited hand mobility. Repo: this directory, branch `Hardware-specifics`. Hardware: two
> Wi-Fi ESP32-CAMs (left/right eye, MJPEG on :81/stream) and an MPU-6050 on the RIGHT board
> (JSON over UDP, subscribe handshake on 5006 → Mac 5005), plus the MacBook webcam for head
> pose. USB-C is power only. Read `AI_HANDOFF.md`, `PROJECT_STATUS.md` and
> `docs/hardware-interface.md` first. The Mac-side receive pipeline, diagnostics and the
> emulator (`opengaze fake-esp32`) are done and tested against the emulator only. Nothing
> has run on real boards. Next tasks, in order: (1) help bring up the real boards with
> `opengaze headless --profile configs/hardware/esp32-glasses.yaml --metrics` and record the
> results in PROJECT_STATUS.md; (2) tune pupil detection on real OV2640 eye images; (3) add a
> 5-point quick calibration and a pre-flight check to the calibration UI. Left click is a
> double blink; do not add dwell or long-blink clicking. Preserve existing working code, never claim hardware results you
> didn't observe, and keep PROJECT_STATUS.md / AI_HANDOFF.md current. Run tests with
> `pytest`.

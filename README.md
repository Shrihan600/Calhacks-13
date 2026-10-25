# OpenGaze

Eye-gaze control for an ordinary Mac, for people with limited hand or arm mobility. It's a CalHacks
13.0 prototype and **not a medically validated assistive device**.

The camera hardware (modules, lenses, glasses frame, mounting, USB vs. Raspberry Pi vs. ESP32) isn't
decided yet, so the software adapts to whatever hardware is described in a YAML **hardware
profile**. Nothing above the camera layer knows which hardware is in use. See
[docs/architecture.md](docs/architecture.md).

## Status

| phase | state |
|---|---|
| 1. Foundation: camera contract, adapters, profiles, capture manager, diagnostics UI, tests | **done** |
| 2. Head pose + eye features (MediaPipe, pupil detector) | next |
| 3. Calibration wizard + gaze model | planned |
| 4. Quartz cursor, smoothing, dwell click, safety stop | planned |
| 5. External eye cameras, Modes B/C: swappable 2D pupil / 3D eye-model pipelines, head-compensated screen mapping, eye-gesture keybinds | **done** (Pi / ESP32 stream adapters still planned) |
| 6. Communication board, Claude, Cartesia, Convex | planned |
| 7. Demo polish | planned |

Camera sources available now: `mac_webcam`, `usb`, `recorded`, `synthetic`. The `raspberry_pi` and
`esp32` sources are reserved names; the validator rejects them until Phase 5.

## Setup (macOS, Apple Silicon)

Use Python 3.11. MediaPipe wheels may not support newer versions yet.

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[all]"
```

No cameras or API keys are needed to run the tests or the simulated rigs.

## Run it

Live previews and health for a fully simulated two-eye rig (no hardware):

```bash
.venv/bin/opengaze diagnostics --profile configs/hardware/synthetic-dual-eye.yaml
```

The same rig with injected faults (dropped frames, latency, corrupt frames, disconnects, failed
connects):

```bash
.venv/bin/opengaze diagnostics --profile configs/hardware/synthetic-faults.yaml
```

The built-in webcam (Mode A). The first run asks for camera permission for the app you launched
from (Terminal, iTerm, VS Code…):

```bash
.venv/bin/opengaze diagnostics
```

Replaying a recorded eye video takes two steps. First, render the synthetic clip:

```bash
.venv/bin/python scripts/make_synthetic_recording.py
```

Then play it back:

```bash
.venv/bin/opengaze diagnostics --profile configs/hardware/replay.yaml
```

Other commands:

```bash
.venv/bin/opengaze validate configs/hardware/generic-usb.yaml
```

```bash
.venv/bin/opengaze probe-cameras
```

```bash
.venv/bin/opengaze headless --profile configs/hardware/synthetic-faults.yaml --seconds 10
```

- `validate` checks a profile, prints its tracking mode and geometry fingerprint, and exits 1 if it
  has errors.
- `probe-cameras` lists the OpenCV device indices that deliver frames. Use it to fill in `device_id`.
- `headless` runs capture with no UI and prints health tables (for SSH and CI).

In the diagnostics window, **Esc** or **Q** quits. *Normalized view* toggles between the raw sensor
image and the image after the profile's rotation, mirror and crop. Synthetic eyes show a green
**GT** marker at the true pupil position, so you can check by eye that the orientation settings are
right.

### Adaptive calibration

`opengaze calibrate --adaptive` (or **A** in `opengaze run`) replaces the fixed 9-dot grid with
a stream of dots. The centre and four corners train a first model. Each later dot is a test: if
the prediction lands within tolerance, the dot passes and the model stays as it is. If not, the dot
becomes training data and the model is refitted. New dots go to the least-covered parts of the
screen, with a pull toward regions where tests failed. The run stops after 4 passes in a row or
30 dots.

Every run writes a log to `~/.config/opengaze/calibration_logs/`, even if you discard the
calibration. The log holds each target, its raw samples, the prediction, the error and the
decision. To print the latest log:

```bash
.venv/bin/opengaze calibration-log
```

It also replays the log: it fits on the first k dots and tests on the rest, so you can see how
many calibration points your setup needs.

## Debugging the path to the cursor

Gaze reaches the cursor in stages: cameras → eyes (pupil detection, 3D eye model) → head pose
→ features → mapping (the calibrated model) → smoothed gaze → cursor. When the cursor doesn't
move, or moves wrong, these tools find the first stage that breaks. Fix that one first:
every later stage inherits the damage.

**Before the headset exists.** `configs/hardware/synthetic-headset.yaml` simulates it. Its eyes
are puppets: they look at each calibration dot, then wander, and a pink square in the run window
shows where they really look. Run the whole app on it, calibration included:

```bash
.venv/bin/opengaze run --profile configs/hardware/synthetic-headset.yaml
```

Or simulate a user headlessly. It calibrates, follows 16 fixations and a smooth pursuit, and
reports every stage against ground truth: accuracy, jitter, lag. Add fault injection to the
profile's camera `options` (`noise_std`, `drop_probability`, `latency_ms`, …) to check that a
fault shows up where you'd expect:

```bash
.venv/bin/opengaze simulate                       # synthetic headset, 9-point grid
.venv/bin/opengaze simulate --adaptive --profile my-rig.yaml
```

The IMU isn't simulated: its own real-time thread can't follow the simulator's virtual clock.

**With the headset.**

1. `opengaze cursor-test` checks the output stage alone: Accessibility permission, the
   displays, and that posted moves land. If this fails, nothing upstream matters.
   `opengaze run` also shows a red **Cursor control unavailable** banner, with the fix,
   whenever it can't drive the cursor. **P** opens the Accessibility settings, and the
   banner clears by itself once permission is granted.
2. In `opengaze run`, press **D** for the debug overlay. It shows every stage's health over
   the last 3 s, and why the cursor isn't moving when it isn't (cursor off, no driver, eye
   closed, low confidence, …).
3. `opengaze run --trace` (or **T**) records every tick to
   `~/.config/opengaze/traces/`: camera frames and latency, pupil detections, head pose,
   features, model output and why a tick was dropped, smoothed gaze, the cursor position sent
   and the position read back from macOS. `opengaze trace-report` turns the latest trace into
   the stage report:

   ```
   [OK  ] cameras   left_eye=fps:30,age95:0,lat95:283 ...
            - left_eye: frames reach the tracker 283 ms after capture (p95) — USB bandwidth, ...
   [WARN] features  skew95_ms=283.3
   ...
   First problem: cameras — left_eye: frames reach the tracker 283 ms after capture ...
   ```

4. `opengaze run --record-video` also saves what each camera gave the tracker, along with the
   calibration and the trace, in `~/.config/opengaze/sessions/`. `opengaze replay <session>`
   re-runs it, tick for tick, through the current code. Record a failing session once, then
   iterate on a fix offline. This saves eye video, so it needs `privacy.allow_recording: true`
   and shows **● REC VIDEO** while it runs. `opengaze simulate --record DIR` makes a
   synthetic session to try replay on.

With the 3D eye pipeline, the calibration intro screen waits until the eye model has fitted.
It fits from varied eye movement, so look around the screen first. Starting early would have
you staring at one dot that the unfitted model can never use.

## Eye cameras (Modes B/C)

Each eye camera runs through one of two pipelines, picked by `gaze.pipeline` in the hardware
profile. Each stage is a swappable component chosen by `type`:

| pipeline | stages | feature per eye |
|---|---|---|
| `pupil_2d` | 2D pupil detector | pupil centre in the eye image |
| `eye_3d` | 2D pupil detector → 3D eye model | unit gaze vector in eye-camera coordinates |

Built-in components are `ellipse_fit`, an OpenCV dark-pupil detector, and `sphere_fit`, a
weak-perspective eyeball fit in the style of pye3d. Pupil Labs' `pupil-detectors` and `pye3d`
have no Apple Silicon wheels, so they aren't wired in. Adding an adapter means implementing
`PupilDetector2D` / `EyeModel3D` (`opengaze/gaze/`) and registering it in `DETECTORS_2D` /
`EYE_MODELS_3D`.

Head pose and calibration turn the features into a screen position.
`calibration.geometry_estimation` picks how:

- `data_driven` works with any pipeline. It fits a polynomial of the eye features plus linear
  head rotation and position terms. Sub-models per eye subset and head part keep it working
  when one eye blinks or head pose drops out. It only compensates head movement that was seen
  during calibration.
- `geometric` needs gaze directions (`eye_3d`, or Mode A). It rotates each gaze ray by the
  fitted eye rotation and the measured head rotation, starts it at the measured head position,
  and intersects it with the screen plane. It compensates head movement that calibration
  never saw. It assumes the head camera sits top-centre on the screen and its image is
  mirrored to match the screen.

## Head pose

All head poses use one world frame: x right along the screen, y down, z from the user into
the screen, with the head camera at the origin (`opengaze/vision/head_pose.py`).

- **Mode A (webcam):** MediaPipe face landmarks give head rotation and head position in cm.
  Distance comes from the ~9 cm eye-corner span, so expect ±10% from person to person.
  The eye feature is the iris position *within the eye opening*, so moving the head doesn't
  move it. The webcam profile uses `geometric` mapping.
- **Modes B/C (IMU):** a gyro + accelerometer on the glasses frame gives head rotation.
  Pitch and roll are absolute (gravity). Yaw is relative: it is anchored when calibration
  starts and on **R** (or a gesture bound to `{app: recenter}`). If a head camera also sees
  the face, it pulls IMU yaw back toward the webcam's (cancelling drift) and supplies head
  position. With an IMU only, head position is unknown and is assumed fixed. Without an IMU,
  the webcam gives both, as in Mode A.

`opengaze diagnostics` shows live yaw/pitch/roll and x/y/z. Turning right, looking up,
tilting right and moving right should all read positive.

### Connecting an IMU

Add an `imu:` section to the hardware profile (see `configs/hardware/generic-usb.yaml`):
`source_type: serial` (USB, needs `pip install -e '.[imu]'`) or `udp` (ESP32 / Pi over
Wi-Fi), the device's units, and `axes`: which IMU axis points to the head's right, the chin,
and forward. The device sends one reading per line, gyro then accelerometer, optionally led
by a timestamp in µs:

```
gx,gy,gz,ax,ay,az
t_us,gx,gy,gz,ax,ay,az
```

Gyro bias is learned automatically whenever the head is still.

The 3D eye model learns during calibration, is frozen afterwards, and its state is saved with
the calibration. If the glasses move on the face, recalibrate. Try it without hardware:

```bash
.venv/bin/opengaze run --profile configs/hardware/synthetic-dual-eye-3d.yaml
```

## Eye gestures and keybinds

The same gestures work in every mode. Winks need both eyes observed.

| family | gestures |
|---|---|
| blinks | `long_blink`, `double_blink`, `triple_blink` (single natural blinks are ignored) |
| winks | `wink_left`, `wink_right` (quick repeats count up: 2 = double), `long_wink_left`, `long_wink_right`, `hold_wink_left`, `hold_wink_right` (start after 0.3 s, end on reopening) |
| glances | `glance_left/right/up/down`: look past a screen edge and back. `hold_left/right/up/down`: stay there; repeats while held |

The default mouse bindings work like a two-button mouse:

| gesture | does |
|---|---|
| double blink | left click |
| left wink, held | hold the left button: look to drag, reopen to release |
| right wink | right click (two quick right winks = double right-click) |
| long right wink | pause / resume |

While a left wink holds the button, the cursor follows the open eye. The other eye's natural
blinks don't end the hold or count as a double blink. Pausing, turning cursor control off,
calibrating or quitting always releases the button. `{mouse: hold}` only works on the
`hold_wink_*` gestures, because only they report when the hold ends.

Bind them under `gestures.bindings` in `configs/default.yaml`. Each binding is a system
keystroke (`{key: "cmd+shift+tab"}`), a mouse action at the gaze point (`{mouse: right_click}`,
`scroll_down`, …) or an OpenGaze action (`{app: pause}`, `toggle_cursor`, `recalibrate`, …).
Key and mouse bindings only fire while cursor control is on (Tab, or a gesture bound to
`toggle_cursor`). App bindings always fire, so the user can get back control hands-free.
The test environment shows each gesture as it fires.

## Adding or changing hardware

1. Copy a profile from `configs/hardware/` (`generic-usb.yaml` is a template).
2. Set each camera's `source_type`, `device_id`, `resolution` and `fps`. Then set `rotation_degrees`
   and mirroring so the eye appears upright in the normalized view, and `crop` if needed.
3. Leave physical measurements (`geometry`, `optics`) as `null` until they're known. Calibration
   is data-driven and doesn't need them.
4. Run `opengaze validate <profile>`, then `opengaze diagnostics --profile <profile>`.
5. Bump `version` or change `frame_design` whenever you rebuild the frame. Any setting that could
   move the eye in the image changes the profile's geometry fingerprint, and saved calibrations
   are rejected with a list of what changed (see `opengaze/config/fingerprint.py`).

Per-user tweaks (such as an eye crop) go in an overlay file that is merged over the profile:
`--overlay my-user.yaml`.

To add a new kind of camera, subclass `CameraSource` (or `BlockingCaptureSource` for blocking
driver APIs), give it a `source_type` and an `Options` model, and register it in
`opengaze/cameras/registry.py`. Nothing else changes.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite runs without any camera. It uses synthetic and recorded sources, fault injection, a fake
OpenCV driver for settings negotiation, and an offscreen Qt window. The one real-webcam test is
opt-in:

```bash
.venv/bin/python -m pytest -m hardware
```

Lint:

```bash
.venv/bin/ruff check opengaze tests
```

## Layout

```
configs/                 default.yaml + hardware profiles
opengaze/cameras/        CameraFrame contract, CameraSource ABC, adapters, registry,
                         manager (reconnect, latest-frame slots), runtime thread, permissions
opengaze/config/         pydantic schema, loader/validator, geometry fingerprint
opengaze/vision/         preprocessing, MediaPipe face/iris, feature schemas
opengaze/gaze/           eye pipelines (2D pupil detector, 3D eye model), GazeTracker, GazeLoop
opengaze/calibration/    session, LOO validation, storage, screen mappers (ridge/poly/geometric)
opengaze/gestures/       blink/wink/glance detectors, recognizer, keybind dispatch
opengaze/control/        Quartz cursor + keyboard events, smoothing, dwell, safety
opengaze/desktop/        PySide6 windows: diagnostics, calibration, run, activities
opengaze/debug/          pipeline traces, stage reports, simulated user, record/replay,
                         cursor self-test
opengaze/cli.py          `opengaze` command
scripts/                 make_synthetic_recording.py
tests/                   pytest suite (hardware-free)
docs/                    architecture.md, privacy.md
```

## Privacy

Everything runs locally. Nothing is recorded or uploaded, and synthetic or recorded data is always
labeled. See [docs/privacy.md](docs/privacy.md).

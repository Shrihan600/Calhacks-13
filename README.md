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

Head pose (MediaPipe on the head camera) and calibration turn the features into a screen
position. `calibration.geometry_estimation` picks how:

- `data_driven` works with either pipeline. It fits a polynomial of the eye features plus
  linear head-pose terms. Sub-models per eye subset keep it working when one eye blinks.
- `geometric` needs `eye_3d`. It rotates each gaze ray by the fitted eye-camera rotation and
  the measured head pose, then intersects it with the screen plane. It compensates head
  movement that calibration never saw. It assumes the head camera sits top-centre on the
  screen and its image is mirrored to match the screen.

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
| winks | `wink_left`, `wink_right`, `long_wink_left`, `long_wink_right` |
| glances | `glance_left/right/up/down`: look past a screen edge and back. `hold_left/right/up/down`: stay there; repeats while held |

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
opengaze/desktop/        PySide6 diagnostics window
opengaze/cli.py          `opengaze` command
scripts/                 make_synthetic_recording.py
tests/                   pytest suite (hardware-free)
docs/                    architecture.md, privacy.md
```

## Privacy

Everything runs locally. Nothing is recorded or uploaded, and synthetic or recorded data is always
labeled. See [docs/privacy.md](docs/privacy.md).

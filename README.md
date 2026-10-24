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
| 5. External eye cameras (USB / Pi / ESP32 streams), Modes B/C | planned |
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
opengaze/vision/         preprocessing (frame normalization)
opengaze/desktop/        PySide6 diagnostics window
opengaze/cli.py          `opengaze` command
scripts/                 make_synthetic_recording.py
tests/                   pytest suite (hardware-free)
docs/                    architecture.md, privacy.md
```

## Privacy

Everything runs locally. Nothing is recorded or uploaded, and synthetic or recorded data is always
labeled. See [docs/privacy.md](docs/privacy.md).

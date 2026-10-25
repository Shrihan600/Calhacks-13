# OpenGaze

Eye-gaze cursor control and communication tool for macOS, built for people with limited hand or arm mobility. CalHacks 13.0 prototype — **not a medically validated assistive device**.

You look at the screen. The software tracks where your eyes are pointing and moves the mouse cursor there. You hold a long blink (400 ms) to click. No hands required.

---

## What it does right now

| Capability | Status |
|---|---|
| Webcam-only gaze tracking (Mode A) | **Working** |
| 9-point gaze calibration wizard | **Working** |
| Fullscreen gaze test environment | **Working** |
| Long-blink click (hold blink 400 ms) | **Working** |
| Real mouse cursor control | **Working** |
| On-screen gaze keyboard | **Working** |
| Voice dictation — Cartesia Ink-Whisper STT | **Working** (opt-in, needs API key) |
| Speak typed text — Cartesia Sonic TTS (macOS `say` offline) | **Working** |
| System-wide voice bar — dictate into any app, read any selection aloud | **Working** |
| Target practice game | **Working** |
| Eye-camera glasses support (Mode C) | Architecture ready, hardware pending |
| AAC communication board + Claude AI | Planned |

---

## Phase completion

| Phase | What it covers | State |
|---|---|---|
| 1 | Camera infrastructure: frame contract, adapters, manager, reconnect, diagnostics UI | **done** |
| 2 | Vision: MediaPipe face landmarker, iris detection, head pose, feature extraction | **done** |
| 3 | Calibration: 9-point wizard, ridge regression model, validation, on-disk storage | **done** |
| 4 | Control: One Euro filter, blink click, safety gates, Quartz cursor driver, gaze test UI | **done** |
| 5 | Eye-camera glasses (USB/Pi/ESP32), Mode B/C pupil detector | next |
| 6 | Cartesia STT + TTS (**done**); AAC board, Claude AI text, Convex backend | in progress |

---

## How it works

### Tracking pipeline (Mode A — built-in webcam)

```
Webcam frame
  → FrameNormalizer  (rotation, mirror, crop from hardware profile)
  → MediaPipe FaceLandmarker  (68 landmarks + iris + blendshapes)
  → Feature extraction  (12 floats: left/right iris x/y, head yaw/pitch/roll, face center, scale)
  → Ridge regression model  (fitted during calibration, stored in ~/.config/opengaze/)
  → Post-filter remap  (stretch calibration range to full screen)
  → One Euro Filter  (adaptive smoothing: slow gaze = heavy smooth, fast = light smooth)
  → Gaze position (x, y) in normalised screen coordinates [0, 1]
  → BlinkClickDetector  (eyes closed ≥ 400 ms → click at frozen pre-blink position)
  → QuartzCursorDriver  (CGEvent mouse move + click, macOS only)
```

### Calibration

Before the gaze model can work, it must be calibrated to your face, eyes, and lighting. The calibration wizard:

1. Shows 9 dots one at a time at the corners, edges, and center of the screen (at 5% margins so the model learns edge data)
2. You look at each dot and hold still — a yellow ring fills over ~1.5 s to collect samples
3. After all 9 dots: ridge regression is fitted (z-score normalized, α=1.0)
4. Leave-one-out cross-validation scores each dot (green = accurate, red = less accurate)
5. The fitted model is saved to `~/.config/opengaze/calibrations/` keyed by hardware fingerprint
6. If you change hardware settings (resolution, mirror, rotation), the old calibration is automatically rejected and you must recalibrate

**Calibration quality notes for Mode A (webcam only):**
- Expected accuracy is 100–300 px depending on lighting, distance, and head stability
- Good lighting on your face and minimal head movement during collection improve accuracy
- The model handles head yaw/pitch/roll as features, but large head tilts hurt accuracy
- Mode C (eye-camera glasses) will be much more accurate

### Blink click

- **Natural blinks** (< 180 ms) are completely ignored
- **Long blink** (both eyes closed ≥ 400 ms) fires a click
- When eyes start closing, the gaze position **freezes** — the click fires at the last stable pre-blink position, not a noisy mid-blink estimate
- After a click there is an 800 ms cooldown to prevent double-fires
- A yellow arc fills around the gaze dot to show blink progress

### Safety systems

- **ConfidenceGate**: drops frames where fewer than 25% of features are valid (face not visible)
- **EmergencyStop**: internal stop state the UI can trigger
- **OneEuroFilter**: smooths jitter without adding perceptible lag

### Hardware profiles

Every camera setting that affects gaze accuracy is captured in a YAML **hardware profile** (`configs/hardware/`). The profile's SHA-256 **geometry fingerprint** is stored alongside each calibration. If you change resolution, mirroring, rotation, or crop, the fingerprint changes and the old calibration is rejected — the system tells you exactly which settings changed.

---

## Setup

Python 3.11 required. MediaPipe wheels do not yet support 3.12+.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[all]"
```

Download the MediaPipe face model (~5 MB, done once):

```bash
.venv/bin/opengaze download-models
```

---

## Running OpenGaze

### Main command — gaze test environment

```bash
.venv/bin/opengaze run
```

If no calibration exists for your hardware, the calibration wizard runs automatically first. After calibration, a fullscreen dark window opens showing:

- **Cyan crosshair** — where the software thinks you're looking
- **Camera preview** (bottom-right) — your face with iris circles drawn on it
- **L / R blink indicators** — yellow when that eye is closed
- **Yellow arc** around the crosshair — fills as you hold a blink; full = click fires
- **Top bar** — tracking mode, confidence score, click count, cursor state

**Keyboard shortcuts in the gaze test window:**

| Key | Action |
|---|---|
| `C` | Re-run calibration (camera stays on, no pause) |
| `K` | Open gaze keyboard |
| `G` | Open target practice game |
| `Tab` | Toggle real cursor control on/off (OFF by default — verify tracking first) |
| `Space` | Pause / resume tracking |
| `Esc` | Quit |

---

### Gaze keyboard

Press **K** in the gaze test window. A fullscreen on-screen keyboard appears.

```
TYPED TEXT area (top 30%)
─────────────────────────
Q  W  E  R  T  Y  U  I  O  P
 A  S  D  F  G  H  J  K  L
   Z  X  C  V  B  N  M
  MIC   SPACE   ⌫   CLEAR   SPEAK
```

**How to type:**
- Look at a key — it highlights immediately
- **Hold your gaze** for 0.8 s — the yellow dwell ring fills → key is typed
- OR **blink** (hold 400 ms) while looking at a key to type it instantly

**Special keys:**
- `SPACE` — inserts a space
- `⌫` — deletes the last character
- `CLEAR` — clears all typed text
- `MIC` — dictate instead of typing (Cartesia Ink-Whisper). The key turns red while listening and stops on its own after a 1.5 s pause, so you don't have to aim at it again. Look at it again or blink to stop early. The transcript is appended to the typed text so you can fix it with the keyboard before speaking it.
- `SPEAK` — reads the typed text aloud with Cartesia Sonic, or macOS `say` when offline. Activate it again while speaking to stop.

`MIC` and `SPEAK` fire once per look: move your gaze off the key (or blink) to trigger it again, so staring at the key never re-fires it.

The keyboard uses **nearest-key snapping**: if your gaze is slightly between two keys, the closer key center wins. This compensates for webcam accuracy limitations.

Press **Esc** to return to the gaze test window.

---

### Voice (Cartesia)

Typing by gaze takes about 2–5 seconds per letter. Many people who need eye-gaze input can still speak, at least partly or when they're not tired, and many can't speak at all. OpenGaze supports both: speak to write, then gaze-edit; or type with your eyes and have a natural voice say it.

**Setup:**

1. Get a key at [play.cartesia.ai/keys](https://play.cartesia.ai/keys)
2. `cp .env.example .env` and set `CARTESIA_API_KEY=sk_car_...`. Optionally set `CARTESIA_VOICE_ID` (any voice from the playground) and `CARTESIA_LANGUAGE` (ISO-639-1, default `en`).
3. Check it works without a camera or calibration:
   ```bash
   .venv/bin/opengaze voice-test --cloud
   ```
   This speaks a sentence with Sonic, then records you until you pause, transcribes with Ink-Whisper, and speaks back what it heard. Add `--tts-only` to skip the mic.
4. `.venv/bin/opengaze run --cloud` → press **K**. The status line under the typed text shows `Voice: Cartesia (Ink-Whisper + Sonic)`.

`--cloud` opts in for that session only. To opt in permanently, set `privacy.cloud_features_enabled: true` in `configs/default.yaml`.

Without both a key and the opt-in, nothing is sent anywhere: `SPEAK` uses macOS `say`, and `MIC` shows a message explaining how to enable it.

### System-wide voice bar

```bash
.venv/bin/opengaze dictate --cloud
```

A small bar floats at the top of the screen, over every app, Space and full-screen window:

| Action | Hotkey | Or click |
|---|---|---|
| Dictate into the focused app (Ink-Whisper). Recording stops when you pause | `⌃⌥D` (Control-Option-D) | 🎤 |
| Read the selected text aloud (Sonic). Press again to stop | `⌃⌥S` | 🔊 |

The transcript is typed into whatever app had focus: Notes, Messages, Slack, a browser text box, anything. Clicking the bar doesn't take focus away from that app; the hint line shows where text will go. Drag the bar to move it, right-click to quit, or press Ctrl+C in the terminal.

Typing and hotkeys need **Accessibility** permission for the app you launch from (Terminal, iTerm or VS Code): System Settings › Privacy & Security › Accessibility. Without it, transcripts are copied to the clipboard for ⌘V instead.

**Cartesia features used:**

| Feature | Endpoint | Used for |
|---|---|---|
| Ink-Whisper speech-to-text (batch) | `POST /stt` | `MIC` dictation on the gaze keyboard; system-wide dictation |
| Sonic-3.6 text-to-speech (raw PCM, 24 kHz) | `POST /tts/bytes` | `SPEAK` on the gaze keyboard; reading selected text aloud |

The code is in `opengaze/voice/`: `cartesia.py` (stdlib-only REST client), `audio.py` (mic capture with end-of-speech detection), `service.py` (non-blocking state machine the UI polls each frame).

**Known limitations:**

- Dictation is batch: words appear after you stop talking (usually under a second or two), not while you speak
- The system-wide bar's hotkeys are fixed (⌃⌥D / ⌃⌥S) and assume key positions on a US layout
- The gaze test window is fullscreen, so gaze-clicking the bar while using other apps needs a system-wide gaze cursor mode (not built yet). The hotkeys and a normal mouse work today
- Cartesia's batch `/stt` occasionally stalls; requests time out after 10 s and retry once
- End-of-speech detection uses a fixed loudness threshold, so a noisy room may keep it listening until the 20 s cap
- No offline speech-to-text fallback; if Cartesia is unreachable, an error shows and you type instead
- English is the default; other languages need `CARTESIA_LANGUAGE`

### Target practice game

Press **G** in the gaze test window. 5 coloured circles appear on a dark background.

- Look at a circle and **hold a blink** (400 ms) to hit it
- A new circle spawns immediately after each hit
- Circles pulse to make them easier to find
- The game runs for **60 seconds**
- Final screen shows: hits, misses, accuracy %
- **Space** starts or replays, **Esc** returns to the test window

Use this mode to:
1. Verify your calibration is working correctly
2. Practice the blink-click gesture
3. Build muscle memory before enabling real cursor control

---

### Calibration only

Run just the calibration wizard without opening the test environment:

```bash
.venv/bin/opengaze calibrate
```

### Check calibration status

```bash
.venv/bin/opengaze calibration-info
```

Prints whether a calibration exists for your current hardware profile and whether it is compatible.

### Diagnostics

Live camera previews and health monitoring (useful to verify cameras are working before calibrating):

```bash
.venv/bin/opengaze diagnostics
```

### Other commands

```bash
# List OpenCV camera indices that deliver frames
.venv/bin/opengaze probe-cameras

# Validate a hardware profile YAML
.venv/bin/opengaze validate configs/hardware/webcam-only.yaml

# Run capture headlessly and print health tables (SSH/CI)
.venv/bin/opengaze headless --seconds 10

# Check Cartesia voice (TTS + dictation) without a camera
.venv/bin/opengaze voice-test --cloud

# Re-download the MediaPipe model
.venv/bin/opengaze download-models --force
```

---

## Hardware modes

| Mode | Cameras | Accuracy | Status |
|---|---|---|---|
| **A — webcam only** | Built-in MacBook webcam | ~100–300 px | Working |
| **B — single eye camera** | One close-up eye camera + webcam | ~50–100 px | Architecture ready |
| **C — dual eye cameras** | Two close-up eye cameras + webcam | ~20–50 px | Architecture ready |

Mode A uses the whole face (iris position + head pose from MediaPipe). Modes B and C use dedicated cameras pointed at each eye mounted on glasses, giving much finer iris position data. The hardware profile YAML declares which cameras are present and their roles (`HEAD`, `LEFT_EYE`, `RIGHT_EYE`); the software automatically selects the correct tracking mode.

The team is building a glasses frame with two eye cameras and an accelerometer. The accelerometer will replace MediaPipe's head pose estimate (features 6–11) with IMU data, removing the head-tilt accuracy loss that affects Mode A.

---

## Enabling real cursor control

Cursor control is **off by default** so you can verify tracking quality before using it. To enable:

1. Run `opengaze run` and calibrate if prompted
2. Watch the cyan crosshair — confirm it tracks where you're actually looking
3. Press **G** (practice game) to verify blink clicks fire where you intend
4. Press **Tab** to enable cursor control — the top-right indicator turns green
5. Grant Accessibility permission if macOS prompts (required for cursor movement)

To disable at any time: press **Tab** again, or press **Esc** to quit entirely.

---

## Project layout

```
configs/
  default.yaml                    app config (log level, default profile path)
  hardware/
    webcam-only.yaml              Mode A — built-in MacBook webcam
    generic-usb.yaml              template for external USB cameras
    synthetic-dual-eye.yaml       simulated two-eye rig (no hardware needed)
    synthetic-faults.yaml         simulated rig with injected faults
    replay.yaml                   replay a recorded eye video

opengaze/
  cameras/
    types.py                      CameraFrame, CameraRole, Orientation, PixelFormat
    manager.py                    CameraManager: reconnect loop, latest-frame slots
    runtime.py                    CaptureRuntime: background thread
    opencv_source.py              mac_webcam and usb adapters
    simulated.py                  synthetic + recorded adapters
    permissions.py                macOS camera permission request
    registry.py                   source_type → class lookup
  config/
    schema.py                     Pydantic models: AppConfig, HardwareProfile, CameraConfig
    loader.py                     load_app_config, check_hardware_profile
    fingerprint.py                geometry snapshot → SHA-256 fingerprint + diff
  vision/
    preprocessing.py              FrameNormalizer (rotation, mirror, crop, debayer)
    mediapipe_detector.py         FaceLandmarkDetector: iris + head pose via MediaPipe
    observations.py               HeadObservation, EyeObservation, FeatureVector types
    feature_extractor.py          build_mode_a(): 12-feature vector from observations
  calibration/
    model.py                      GazeModel: ridge regression, z-score norm, predict
    session.py                    CalibrationSession: 9-point collection + fit
    validation.py                 leave-one-out cross-validation, ValidationReport
    storage.py                    CalibrationStore: save/load .npz by fingerprint
  control/
    filter.py                     OneEuroFilter: adaptive smoothing
    blink.py                      BlinkClickDetector: long blink → click
    dwell.py                      DwellDetector: hold gaze → click (unused in main UI)
    safety.py                     ConfidenceGate, JumpDetector, EmergencyStop
    cursor.py                     QuartzCursorDriver: macOS CGEvent move + click
    driver.py                     GazeDriver: full pipeline orchestrator
  desktop/
    calibration_ui.py             PySide6 calibration wizard (9 dots, validation heatmap)
    run_ui.py                     Fullscreen gaze test environment
    keyboard_ui.py                On-screen gaze keyboard with dictation + speech
    practice_ui.py                Target practice game
    diagnostics.py                Live camera health + preview window
  voice/
    cartesia.py                   Cartesia REST client: Ink-Whisper STT, Sonic TTS
    audio.py                      MicRecorder: capture + automatic end-of-speech
    service.py                    VoiceService: non-blocking dictation/speech, offline fallback
  modes.py                        TrackingMode enum, infer_mode()
  imaging.py                      Low-level image transforms
  cli.py                          opengaze CLI entry point

tests/
  test_cameras.py                 camera contract, manager, reconnect, fault injection
  test_config.py                  profile loading, fingerprint, overlay merging
  test_vision.py                  feature extractor, normalizer
  test_calibration.py             GazeModel, CalibrationSession, ValidationReport, store
  test_control.py                 OneEuroFilter, DwellDetector, safety, GazeDriver
  test_voice.py                   Cartesia client (mocked HTTP), end-of-speech, VoiceService

scripts/
  make_synthetic_recording.py     render a synthetic eye video for replay testing
docs/
  architecture.md
  privacy.md
```

---

## Tests

```bash
.venv/bin/python -m pytest
```

259 tests, all hardware-free and network-free. Uses synthetic cameras, fault injection, and a mocked cursor. Real-webcam tests are opt-in:

```bash
.venv/bin/python -m pytest -m hardware
```

---

## Privacy

Gaze tracking runs entirely on your Mac. No video and no gaze data ever leave the device. The only network traffic is optional Cartesia voice: when you opt in, audio you record with `MIC` and text you send with `SPEAK` go to Cartesia. Nothing is sent until you activate one of those keys. The calibration model is stored in `~/.config/opengaze/calibrations/` and is never uploaded anywhere. See [docs/privacy.md](docs/privacy.md).

---

## Known limitations (Mode A, webcam only)

- **Accuracy is ~100–300 px** — enough to click large targets and use the keyboard, not enough for pixel-precise work
- **Head tilt degrades accuracy** — the iris position in the camera frame shifts with head roll even when gaze direction doesn't change; keep your head upright during use
- **Lighting matters** — even overhead lighting on your face gives the best results; avoid strong backlight
- **Keyboard accuracy** — keys are ~200 px wide on a typical Mac display, which is within webcam accuracy bounds; the nearest-key snapping helps with off-by-one errors
- **Mode C glasses** will fix all of the above: dedicated eye cameras give 10× better iris precision and the accelerometer removes head-pose error entirely

# OpenGaze

Eye-gaze cursor control and communication tool for macOS, built for people with limited hand or arm mobility. CalHacks 13.0 prototype — **not a medically validated assistive device**.

You look at the screen. The software tracks where your eyes are pointing and moves the mouse cursor there. You blink twice quickly to left-click. No hands required.

---

## What it does right now

| Capability | Status |
|---|---|
| Webcam-only gaze tracking (Mode A) | **Working** |
| 9-point gaze calibration wizard | **Working** |
| Fullscreen gaze test environment | **Working** |
| Double-blink left click | **Working** |
| Real mouse cursor control | **Working** |
| On-screen gaze keyboard | **Working** |
| Voice dictation — Cartesia Ink-Whisper STT | **Working** (opt-in, needs API key) |
| Speak typed text — Cartesia Sonic TTS (macOS `say` offline) | **Working** |
| System-wide voice bar — dictate into any app, read any selection aloud | **Working** |
| System-wide gaze cursor — look + blink-click in every app (`opengaze desktop`) | **Working** |
| Target practice game | **Working** |
| Adaptive calibration (dots until accurate, with logs) | **Working** |
| Eye-gesture mouse: double-blink click, held-wink drag, right-wink right click | **Working** |
| Debug tooling: live overlay, traces, simulated headset, record/replay | **Working** |
| Eye-camera glasses support (Mode C) + head-mounted IMU | Architecture ready, hardware pending |
| **EyeMouse consumer app** — guided setup, menu bar, caret-following voice typing, gaze keyboard for any app (`opengaze app`) | **Working** (offscreen + native window checks; not yet tried by an end user) |
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

- **Double blink** (two short blinks, 40–300 ms each, within 450 ms of each other) fires a
  left click. It fires once the 450 ms gap after the second blink expires, so a triple blink
  (toggle cursor control) is never mistaken for it.
- A **single blink** does nothing. There is no long-blink click and no dwell click.
- While the eyes are closed the gaze position **freezes**, so the click lands at the last
  stable pre-blink position, not a noisy mid-blink estimate
- After a click there is an 800 ms cooldown to prevent double-fires
- Winks act as the other mouse buttons: see
  [Eye gestures and keybinds](#eye-gestures-and-keybinds)

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

### EyeMouse — the consumer app

```bash
.venv/bin/opengaze app          # or: .venv/bin/eyemouse
.venv/bin/opengaze app --setup  # run setup again
.venv/bin/opengaze app --fresh  # as if for the first time: setup, no tracking until you calibrate
.venv/bin/opengaze app --no-tracking  # gaze cursor off all session (combine with --fresh/--setup)
```

The everyday app, built from the consumer design. The developer commands below stay as they are.

**First run (done by a helper):** Welcome → **connect your glasses** (plug both cables into
any USB ports; each eye's picture appears when its board is found, and Continue unlocks when
both are) → allow the camera → turn on Accessibility (the page
moves on by itself once it's on) → allow the microphone or skip voice → 9-dot calibration
(fullscreen; a poor fit offers *Try again* / *Use it anyway*) → **Try it out**, where the gaze
cursor is live: double blink a circle, right wink a card, hold a left wink to drag a block.
If tracking is too poor to use, *Stop tracking* (or ⌃⌥T) turns the cursor off. **Done** moves EyeMouse to the menu bar and, if ticked, adds a login item
(`~/Library/LaunchAgents/com.opengaze.eyemouse.plist`).

**Every day:** the gaze cursor drives the pointer in every app. The menu bar item has the
gaze switch and its status (*Tracking well*, *Looking for your face*, *Mouse in control*,
*Paused*, *Tracking off*), Pause (⌃⌥G), Stop tracking (⌃⌥T), Voice typing, Gaze keyboard,
Recalibrate and Settings. Pause still listens for the long right wink that resumes; when
tracking is too bad to trust, ⌃⌥T stops the gaze loop outright (no cursor, clicks or
gestures) until you press it again. The voice typing mic keeps working meanwhile.

- **Voice typing:** when a text field has focus, a mic floats beside the caret (found through
  the Accessibility API). Click it or triple blink, then speak; the words are typed
  at the caret. Needs the microphone, `CARTESIA_API_KEY`, and the "Allow microphone" consent
  in setup (which turns on Cartesia for EyeMouse only).
- **Gaze keyboard:** a dark panel along the bottom of the screen. It never takes focus, so it
  types into whatever app you were in. Look at a key and double blink; word suggestions
  complete the current word; Send presses Return.

**Always the glasses:** EyeMouse loads `configs/hardware/esp32-wired.yaml` (the glasses plus
this Mac's camera for head position). There is no webcam-only option for users. The cameras keep looking for the
glasses if they're unplugged.

**Problems are shown, never only logged.** Once a problem lasts 2 s, EyeMouse shows it (a
camera still connecting gets 10 s after start; one whose port search has given up counts at
once, so unplugged glasses show within about 6 s):
- **Stops gaze control** (glasses or one eye camera not connected, no video, the receiver
  missing, this Mac's camera not working or not allowed): a red banner at the top of the
  screen until it's fixed (drawn inside calibration, whose Start waits for the fix), a
  notification, and a ⚠ line in the menu.
- **Gaze control still works** (the head motion sensor isn't sending): a notification and the
  ⚠ menu line.
- **Can't start** (invalid hardware settings): a dialog, since a login item has no terminal.
To try the 3D eye model instead of 2D pupil centres:
`eyemouse --overlay configs/overlays/eye-3d.yaml` (2D and 3D calibrate separately; the login
item keeps whatever options the app was started with).
Development fallback without glasses: `opengaze app --profile configs/hardware/webcam-only.yaml`
(setup then skips the glasses page). Setup remembers which profile it was done with
(`setup_hardware` in `eyemouse.json`) and runs again for glasses it wasn't done with, so a
webcam-only setup from before is followed by the glasses page. The webcam fallback never
re-runs setup.

EyeMouse's choices live in `~/.config/opengaze/eyemouse.json`; calibrations are shared with
the other commands. Run from a terminal, macOS grants camera and Accessibility to *that
terminal app*, and the setup pages name it.

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
| `A` | Adaptive calibration (dots until accurate) |
| `R` | Recenter head heading (IMU) |
| `K` | Open gaze keyboard |
| `G` | Open target practice game |
| `D` | Debug overlay: every pipeline stage's health, and why the cursor isn't moving |
| `T` | Start / stop a trace (`opengaze trace-report`) |
| `P` | Open Accessibility settings (when the cursor-permission error shows) |
| `Space` | Pause / resume cursor control (also `Tab` or a long right wink). Starts paused — verify tracking first. Tracking keeps running while paused |
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

### System-wide gaze control (every app)

```bash
.venv/bin/opengaze desktop --cloud
```

No fullscreen window: the real macOS cursor follows your gaze in every app, and your eye gestures fire their `gestures:` bindings there. These are the same ones as in `opengaze run`: by default a **double blink left-clicks** where you were looking, and a held wink drags. A yellow ring fills around the cursor as you hold a wink, so you can see the drag coming. The voice bar below floats on top, so you can blink-click 🎤 to dictate into the app you're using.

| Control | How |
|---|---|
| Pause / resume gaze cursor | `⌃⌥G`, or click 👁 on the voice bar |
| Take over with the mouse | Just move it; gaze waits 2 s after you stop |
| Quit | Right-click the voice bar → Quit, or Ctrl+C in the terminal |

Uses your saved calibration (runs the wizard first if there isn't one). Needs Accessibility permission for the terminal app. If tracking feels off, recalibrate with `opengaze calibrate`, and try the practice game (`opengaze run` → G) to get used to blink-clicking.

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
- Pausing gaze from the bar's 👁 button with your eyes means resuming needs `⌃⌥G` or the mouse, since the gaze cursor is off
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

Live camera previews, IMU readings and health monitoring (useful to verify hardware is working before calibrating):

```bash
.venv/bin/opengaze diagnostics
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-glasses.yaml --metrics
```

A badge per input (each camera, the IMU) shows OK / WARN / DOWN. Each camera tile shows fps,
jitter, frame age, delay, skipped (sequence gaps), partial and reconnect counts. The IMU tile
shows live accelerometer (m/s²) and gyro (rad/s) values with rolling plots, the rate, drops
and a units check. `--metrics` logs all of it (numbers only) to `logs/metrics/*.jsonl`.

### Other commands

```bash
# List OpenCV camera indices that deliver frames
.venv/bin/opengaze probe-cameras

# Validate a hardware profile YAML
.venv/bin/opengaze validate configs/hardware/webcam-only.yaml

# Run capture headlessly and print camera + IMU health tables (SSH/CI)
.venv/bin/opengaze headless --seconds 10

# Emulate the ESP32-CAM glasses on this Mac (two MJPEG eyes + UDP IMU, faults optional)
.venv/bin/opengaze fake-esp32 --drop-rate 0.05 --jitter-ms 30

# Override any profile setting for one run (IPs, ports, enable/disable a camera)
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-glasses.yaml \
  --set cameras.left_eye.options.host=192.168.1.41

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

The team is building a glasses frame with two eye cameras and an IMU. IMU head rotation is supported (see [Head pose](#head-pose)); with eye cameras, the head-tilt accuracy loss that affects Mode A goes away.

---

## Enabling real cursor control

Cursor control is **off by default** so you can verify tracking quality before using it. To enable:

1. Run `opengaze run` and calibrate if prompted
2. Watch the cyan crosshair — confirm it tracks where you're actually looking
3. Press **G** (practice game) to verify blink clicks fire where you intend
4. Press **Space** to resume cursor control — the top-right indicator turns green
5. Grant Accessibility permission (required for cursor movement). If it's missing, the window
   shows a red **Cursor control unavailable** banner with the fix; `opengaze cursor-test`
   checks it on its own

To pause at any time: press **Space** again, hold a right wink, or press **Esc** to quit entirely.

## Adaptive calibration

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

## ESP32-CAM glasses (Wi-Fi)

The confirmed hardware: two Inland ESP32-CAM boards (left and right eye) stream MJPEG over
2.4 GHz Wi-Fi. An MPU-6050 on the **right** board sends JSON telemetry over UDP. The MacBook
webcam provides head position and orientation, and USB-C carries power only. The full
contract (endpoints, headers, units, clock sync, failure handling) is in
[`docs/hardware-interface.md`](docs/hardware-interface.md). Reference firmware, **not yet
compiled or flashed**, is in [`firmware/`](firmware/).

```bash
# No hardware: run the emulator and point the ESP32 profile at it
.venv/bin/opengaze fake-esp32                                             # terminal 1
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-emulator.yaml   # terminal 2

# Real boards (mDNS names from the firmware; use --set …host=<ip> if mDNS is blocked)
.venv/bin/opengaze headless --profile configs/hardware/esp32-glasses.yaml --seconds 30 --metrics
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-glasses.yaml
```

The ESP32 profile is Mode C. Once both streams and the IMU are up, `opengaze calibrate` and
`opengaze run` use them like any other eye cameras. Bench-test live Wi-Fi at the boards'
documented 20 cm antenna separation, not on someone's head, until the RF-exposure review is
done.

## ESP32-CAM glasses (wired, USB serial)

The hardware team replaced Wi-Fi with USB serial (2026-10-25). Two ESP32-CAMs (OV3660,
color JPEG, 320×240) each reach the Mac through their own Seeed XIAO RA4M1 bridge, so the
Mac sees two `/dev/cu.usbmodem*` ports. The MPU-6050 is now on the **left** board, on the
same port as the left camera. The protocol is parsed by the hardware team's receiver
(`opengaze-hw.zip` → `receiver/`), which `opengaze/cameras/wired.py` wraps behind the usual
camera and IMU contracts. The Wi-Fi profile above still works as a fallback.

```bash
pip install pyserial
unzip opengaze-hw.zip                      # → opengaze-hw/receiver; or set OPENGAZE_HW_RECEIVER
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-wired.yaml
```

Ports are found automatically (`port: auto`): each camera, when it connects, opens every
`/dev/cu.usbmodem*` and `/dev/cu.usbserial*` port and waits for its board to name itself (the
device id in the telemetry each board sends once a second; the left one's IMU packets
too). Any USB socket works, and a reconnect looks again, so moving a cable doesn't need a
restart. To pin a port instead: `--left-serial /dev/cu.usbmodem1101 --right-serial …`. Port
names don't affect a calibration. Another serial device plugged in is opened briefly
during the search and then let go. Only one program can hold a port: quit the receiver's own
`main.py` dashboard first. Flip or mirror the images with `rotation_degrees` /
`mirror_horizontal` / `mirror_vertical` in the profile.

No hardware? The receiver's simulator serves pseudo-terminals (frames are labelled EMULATED).
macOS pseudo-terminals reject 921600 baud, so open them at 230400; the simulator still paces
itself at 921600:

```bash
.venv/bin/python opengaze-hw/receiver/tools/sim_serial_device.py --role left    # prints /dev/ttysN
.venv/bin/python opengaze-hw/receiver/tools/sim_serial_device.py --role right   # prints /dev/ttysM
.venv/bin/opengaze diagnostics --profile configs/hardware/esp32-wired.yaml \
  --left-serial /dev/ttysN --right-serial /dev/ttysM --set cameras.head.enabled=false \
  --set cameras.left_eye.options.baud=230400 --set cameras.right_eye.options.baud=230400
```

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
and forward. Over UDP the device can send JSON telemetry with sequence numbers and device
timestamps (the ESP32 glasses do; see [`docs/hardware-interface.md`](docs/hardware-interface.md)),
or the simple text protocol below. The text protocol sends one reading per line, gyro then
accelerometer, optionally led by a timestamp in µs:

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
blinks don't end the hold or count as a double blink. Pausing, calibrating or quitting always releases the button. `{mouse: hold}` only works on the
`hold_wink_*` gestures, because only they report when the hold ends.

Bind them under `gestures.bindings` in `configs/default.yaml`. Each binding is a system
keystroke (`{key: "cmd+shift+tab"}`), a mouse action at the gaze point (`{mouse: right_click}`,
`scroll_down`, …) or an OpenGaze action (`{app: pause}`, `recalibrate`, …; `toggle_cursor` is the same as `pause`).
Key and mouse bindings only fire while not paused. App bindings always fire, so the user can get back control hands-free.
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

---

## Project layout

```
configs/
  default.yaml                    app config (log level, default profile path)
  hardware/
    webcam-only.yaml              Mode A — built-in MacBook webcam
    esp32-glasses.yaml            Mode C — two Wi-Fi ESP32-CAM eyes + MPU-6050 (right) + webcam
    esp32-emulator.yaml           the same, pointed at `opengaze fake-esp32` on localhost
    generic-usb.yaml              template for external USB cameras
    synthetic-dual-eye.yaml       simulated two-eye rig (no hardware needed)
    synthetic-faults.yaml         simulated rig with injected faults
    synthetic-headset.yaml        simulated headset whose eyes follow calibration dots
    replay.yaml                   replay a recorded eye video

opengaze/
  cameras/
    types.py                      CameraFrame, CameraRole, Orientation, PixelFormat
    manager.py                    CameraManager: reconnect loop, latest-frame slots
    runtime.py                    CaptureRuntime: background thread
    opencv_source.py              mac_webcam and usb adapters
    esp32.py                      esp32 adapter: ESP32-CAM MJPEG over Wi-Fi
    mjpeg.py                      MJPEG-over-HTTP stream reader (stdlib, no FFmpeg)
    timing.py                     RateMeter, DeviceClockSync (device → host clock)
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
    mappers.py                    poly-ridge and geometric screen mappers (Modes B/C)
    adaptive.py                   adaptive calibration session, logs, points-needed replay
  imu/                            IMU sources (serial/UDP text or JSON/synthetic), fusion,
                                  head pose
  gaze/                           eye pipelines (2D pupil / 3D eye model), GazeTracker, GazeLoop
  gestures/                       blink/wink/glance detectors, recognizer, keybind dispatch
  debug/                          traces, stage reports, simulated user, record/replay,
                                  cursor self-test, fake_esp32 emulator, metrics recorder
  control/
    filter.py                     OneEuroFilter: adaptive smoothing
    dwell.py                      DwellDetector: hold gaze → click (not used; clicks are blinks)
    safety.py                     ConfidenceGate, JumpDetector, EmergencyStop, UserOverride
    cursor.py                     QuartzCursorDriver: macOS CGEvent move + click
    keystrokes.py                 Type Unicode into the focused app, copy selection
    driver.py                     GazeDriver: full pipeline orchestrator
  desktop/
    calibration_ui.py             PySide6 calibration wizard (9 dots, validation heatmap)
    adaptive_calibration_ui.py    adaptive calibration wizard (dots until accurate)
    run_ui.py                     Fullscreen gaze test environment
    keyboard_ui.py                On-screen gaze keyboard with dictation + speech
    practice_ui.py                Target practice game
    desktop_mode.py               System-wide gaze cursor: click-through overlay + voice bar
    dictation_ui.py               Floating voice bar: dictate into any app, read selection aloud
    macwin.py                     Floating, non-activating, all-Spaces window helpers
    diagnostics.py                Live camera + IMU health, previews, input badges
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
  test_esp32.py                   clock sync, MJPEG, esp32 adapter vs emulator, reconnects
  test_imu_telemetry.py           MPU-6050 JSON over UDP, handshake, units, metrics

scripts/
  make_synthetic_recording.py     render a synthetic eye video for replay testing
firmware/
  opengaze_esp32cam/              reference ESP32-CAM sketch (left/right; untested)
docs/
  architecture.md
  hardware-interface.md           ESP32 glasses ↔ Mac contract (endpoints, units, sync)
  privacy.md
PROJECT_STATUS.md                 what is verified vs. not, open tasks
AI_HANDOFF.md                     continuation notes for the next developer / AI session
```

---

## Tests

```bash
.venv/bin/python -m pytest
```

417 tests, all hardware-free and network-free. Uses synthetic cameras, fault injection, and a mocked cursor. Real-webcam tests are opt-in:

```bash
.venv/bin/python -m pytest -m hardware
```

Lint:

```bash
.venv/bin/ruff check opengaze tests
```

---

## Privacy

Gaze tracking runs entirely on your Mac. No video and no gaze data ever leave the device. The calibration model is stored in `~/.config/opengaze/calibrations/` and is never uploaded anywhere. Eye video is only saved when you ask for it (`run --record-video`, which also needs `privacy.allow_recording: true`). The only network traffic is optional Cartesia voice: when you opt in, audio you record with `MIC` (or ⌃⌥D) and text you send with `SPEAK` (or ⌃⌥S) go to Cartesia. Nothing is sent until you trigger one of those. See [docs/privacy.md](docs/privacy.md).

---

## Known limitations (Mode A, webcam only)

- **Accuracy is ~100–300 px** — enough to click large targets and use the keyboard, not enough for pixel-precise work
- **Head tilt degrades accuracy** — the iris position in the camera frame shifts with head roll even when gaze direction doesn't change; keep your head upright during use
- **Lighting matters** — even overhead lighting on your face gives the best results; avoid strong backlight
- **Keyboard accuracy** — keys are ~200 px wide on a typical Mac display, which is within webcam accuracy bounds; the nearest-key snapping helps with off-by-one errors
- **Mode C glasses** will fix all of the above: dedicated eye cameras give 10× better iris precision and the accelerometer removes head-pose error entirely

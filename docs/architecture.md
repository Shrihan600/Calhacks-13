# OpenGaze architecture

OpenGaze lets someone with limited hand or arm mobility control an ordinary Mac with their eyes,
minimal head movement, and optional AI-assisted communication. It is a hackathon prototype, **not a
medically validated assistive device**.

The governing rule: **the software adapts to the hardware.** Camera modules, lenses, glasses-frame
geometry, mounting angles, the processor (USB / Raspberry Pi / ESP32) and the link (wired / Wi-Fi) are
all unfinalized. None of them may leak past the camera layer and the hardware profile.

---

## 1. System architecture

```
 ┌──────────────────────── hardware (interchangeable) ─────────────────────────┐
 │  MacBook webcam   USB/UVC eye cams   Raspberry Pi stream   ESP32 stream      │
 │  recorded video   synthetic generator                                         │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        │  CameraSource adapters  (opengaze/cameras/*)  ← selected by hardware profile YAML
        ▼
 ┌─────────────────────────── capture thread (asyncio) ─────────────────────────┐
 │ CameraManager: one task per camera, reconnect with backoff, latest-frame      │
 │ slot per camera (old frames are overwritten, never queued), health stats      │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        │  CameraFrame (standard contract, §3)
        ▼
 ┌──────────────────────────── vision pipeline ─────────────────────────────────┐
 │ normalize → eye ROI → pupil/iris → confidence → head pose → temporal align    │
 │ → feature fusion (validity masks) → GazeEstimator → smoothing → screen map    │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        │  GazeSample (normalized screen [0,1]² + confidence)
        ▼
 ┌───────────────────────── main thread (PySide6) ──────────────────────────────┐
 │ diagnostics · calibration wizard · gaze overlay · communication board         │
 │ control/: Quartz cursor, dwell click state machine, safety (pause / e-stop)   │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        │  state snapshots (no images by default)
        ▼
 ┌──── optional, Phase 6+ ────┐        ┌──── optional cloud (opt-in only) ─────┐
 │ FastAPI + WebSocket (local)│ ─────▶ │ Claude (text) · Cartesia (speech)     │
 │ React dashboard            │        │ Convex (approved messages / status)   │
 └────────────────────────────┘        └───────────────────────────────────────┘
```

Why this shape:

- **One process, three kinds of thread.** Capture runs on its own asyncio loop in a background thread
  (`CaptureRuntime`). Each blocking camera driver gets its own daemon worker thread, so a hung USB
  read can't freeze other cameras, the UI, or process exit. The UI never waits on a camera. OpenCV and
  MediaPipe release the GIL in native code. If profiling on the M2 shows the UI stuttering, the vision
  pipeline moves to a subprocess. Its input and output are already plain data contracts, so nothing
  else has to change.
- **Latest-frame, not queues.** Each camera has a single slot that new frames overwrite. A slow
  consumer gets fresher frames, never a growing backlog. Overwritten frames are counted in diagnostics.
- **Native control stays native.** Cursor movement, clicks and the safety stop are Quartz calls from
  Python, never routed through the browser. React is only for the richer dashboard.
- **Local-first.** The cursor system works with no network. Cloud modules are separate and opt-in.
  They never receive images.

## 2. Camera abstraction and hardware independence

`opengaze/cameras/base.py` defines `CameraSource`:

| method | contract |
|---|---|
| `async connect()` | open the device/stream and apply requested settings. Raises `CameraConnectionError` (retryable) or `CameraConfigError` (not retryable). |
| `async read_frame()` | wait for the **next** frame and return a `CameraFrame`. Raises `CameraDisconnectedError` on loss or read timeout, `EndOfStream` for a finished recording. |
| `async disconnect()` | release resources. Idempotent. |
| `get_capabilities()` | `CameraCapabilities`: what the hardware reported, plus a per-setting result (supported / ignored / unsupported / unknown). |
| `get_status()` | `CameraStatus`: state, frames read, last error. |
| `frames()` | async iterator over `read_frame()`. |

Adapters register a `source_type` string. The hardware profile names the `source_type`, and the
registry builds the adapter. Everything downstream sees only `CameraFrame`, so swapping an ESP32 for a
Pi, or a Pi for USB, only edits YAML.

| `source_type` | class | status |
|---|---|---|
| `mac_webcam` | `MacWebcamSource` | Phase 1. OpenCV with the AVFoundation backend. |
| `usb` | `USBCameraSource` | Phase 1. Generic UVC through OpenCV; requested settings are read back and reported. |
| `recorded` | `RecordedVideoSource` | Phase 1. Video files, paced in real time or as fast as possible, optional loop. |
| `synthetic` | `SyntheticCameraSource` | Phase 1. Rendered eyes with ground truth, plus fault injection (drops, latency, jitter, corrupt frames, disconnects, failed connects). |
| `raspberry_pi` | `RaspberryPiStreamSource` | Phase 5. MJPEG over HTTP first, RTSP via FFmpeg as an option. |
| `esp32` | `ESP32StreamSource` | Phase 5. MJPEG over HTTP (the transport ESP32 camera web-server examples commonly use). |

The validator rejects a planned source type and names the phase. It never pretends to work.

Adapter-specific settings (a video path, synthetic parameters, later a stream URL) go in the camera's
`options:` block. Each adapter validates them with its own pydantic model, so a typo fails at load time.

## 3. Data schema

### CameraFrame (Phase 1, implemented)

| field | type | notes |
|---|---|---|
| `image` | `np.ndarray` | raw image as delivered, before orientation/crop |
| `camera_id`, `role` | `str`, `CameraRole` | role is one of `left_eye` / `right_eye` / `head` / `other` |
| `sequence` | `int` | per source; a gap means the source skipped frames |
| `host_receive_ns` | `int` | `time.monotonic_ns()` when the decoded frame reached the host |
| `capture_ns` | `int \| None` | exposure time on the host clock, **`None` when unknown** |
| `timestamp_source` | enum | `device`, `host_receive`, `media_time`, `synthetic` |
| `timestamp_uncertainty_ns` | `int \| None` | `None` = unknown. Network sources will report clock-offset uncertainty here. |
| `pixel_format` | enum | `bgr8`, `gray8`, `rgb8` |
| `source_kind` | enum | `live`, `recorded`, `synthetic`. Every UI labels non-live data. |
| `orientation` | `Orientation` | correction still to apply: mirror first, then clockwise right-angle rotation |
| `crop` | `CropRegion \| None` | fractional ROI of the oriented image |
| `fps`, `fps_measured` | `float \| None`, `bool` | measured from frame timestamps when possible |
| `intrinsics`, `distortion` | optional | only from the profile; never invented |
| `status`, `warnings` | enum, tuple | `ok` / `partial` (decoded but suspect) |
| `ground_truth` | mapping or `None` | synthetic sources only, so it can't be mistaken for a measurement |

`vision/preprocessing.normalize_frame()` applies undistortion (only if intrinsics and distortion are
both known), then orientation, then crop. It adjusts intrinsics to match and returns a frame with
identity orientation.

### Observations and fused features (Phase 2, planned)

```
EyeObservation  { camera_id, eye: left|right, t_ns, t_uncertainty_ns,
                  pupil_xy_norm (in normalized eye-ROI coords), pupil_radius_norm,
                  ellipse (optional), blink: bool, confidence: 0..1, detector: str }
HeadObservation { t_ns, yaw_deg, pitch_deg, roll_deg (relative to calibration baseline),
                  face_center_norm, relative_scale, confidence, transform_4x4 (optional) }
FeatureVector   { t_ns,
                  values:   float32[N]   # fixed order from a FeatureSchema
                  valid:    bool[N]      # explicit validity mask
                  sample_ts: {source: t_ns}, max_skew_ns,
                  schema_id: "A-webcam" | "B-single-eye" | "C-dual-eye" }
GazeSample      { t_ns, screen_xy_norm, confidence, mode, latency_ns }
```

**Phase 5 schemas.** Each mode × eye pipeline has its own `schema_id`: `A-webcam`, `B-pupil2d`,
`C-pupil2d`, `B-eye3d`, `C-eye3d`. `FEATURE_LAYOUTS` in `vision/observations.py` records which
columns hold each eye's values, its confidence, and the head pose. The `*-pupil2d` layout matches
Mode A's; the `*-eye3d` layout holds a unit gaze vector plus confidence per eye. Calibration
files are keyed by schema id, store the mapper kind, and store a *tracking snapshot* (eye
pipeline config + `geometry_estimation`). A changed pipeline or mapping is reported as
incompatible, the same way a geometry change is.

### Gaze pipeline (Phase 5)

```
eye camera ─▶ normalize ─▶ PupilDetector2D ─────────────────────────▶ pupil centre ─┐  pupil_2d
                                 └──▶ EyeModel3D (learns, then frozen) ─▶ gaze vector ─┤  eye_3d
head camera ─▶ normalize ─▶ MediaPipe ─▶ HeadObservation ─────────────────────────────┤
                                                                                      ▼
                     FeatureVector(schema) ─▶ ScreenMapper ─▶ OneEuro ─▶ cursor / gestures
                                              ├ GazeModel             (Mode A, linear ridge)
                                              ├ PolynomialRidgeMapper (B/C, data_driven)
                                              └ GeometricRayMapper    (B/C eye_3d, geometric)
```

`GazeTracker` (`gaze/tracker.py`) runs the vision front end for any mode. `GazeLoop`
(`gaze/loop.py`) adds the mapper, smoothing and gesture recognition, and every desktop window
uses it. Gestures see the unclipped, unsmoothed gaze, so glances past the screen edge are
detectable. The cursor freezes while an eye is closed, so blink and wink actions land at the
pre-blink position.

The feature order is the one in the project brief: left pupil x/y/conf, right pupil x/y/conf, head
yaw/pitch/roll, head center x/y, relative scale. A calibration model is fitted for one `schema_id` and
refuses vectors from another, so each mode gets its own model profile. Accuracy is never assumed to
carry over between modes.

### Coordinate spaces (kept as distinct types from Phase 3 on)

1. camera image pixels (raw)
2. normalized camera image (after orientation/crop)
3. normalized eye-ROI coordinates (pupil features)
4. normalized gaze/screen `[0,1]²`
5. macOS logical desktop points (what `CGEvent` mouse positions use)
6. physical display pixels (2880 × 1864 on the 15" Air, used only for metrics)

Logical size and backing scale are read at runtime from Quartz / Qt, never hardcoded.

## 4. Dependencies (Apple Silicon)

Checked on 2026-10-24: every package below installs from wheels and imports on an M2 MacBook
under Python 3.11.3 (python.org build). The versions are what pip resolved that day; `pyproject.toml`
keeps lower bounds only.

| extra | package | resolved |
|---|---|---|
| core | numpy | 2.4.6 |
| core | opencv-contrib-python | 5.0.0.93 |
| core | pydantic | 2.14.1 |
| core | PyYAML | 6.0.3 |
| core (macOS) | pyobjc-framework-AVFoundation | 12.2.2 |
| `desktop` | PySide6 | 6.12.0 |
| `desktop` (macOS) | pyobjc-framework-Quartz | 12.2.2 |
| `tracking` | mediapipe | 1.1.1 |
| `tracking` | scipy / scikit-learn | 1.17.1 / 1.9.1 |
| `api` | fastapi / uvicorn | 0.162.0 / 0.54.0 |
| `dev` | pytest / pytest-asyncio / ruff | 9.1.1 / 1.4.0 / 0.17.2 |

Notes:

- **OpenCV comes from `opencv-contrib-python` only.** MediaPipe depends on the contrib build.
  Installing `opencv-python` as well puts two packages into the same `cv2` directory, where they
  overwrite each other.
- **Use Python 3.11.** This Mac's default `python3` is 3.14, which wasn't tried. 3.12 is likely fine
  but untested.
- MediaPipe resolved to 1.x. Phase 2 must target its Tasks API (`FaceLandmarker`) and check that
  the API still matches the 0.10-era docs.

## 5. Repository structure

```
configs/            app defaults + hardware profiles (YAML)
opengaze/
  cameras/          types, base ABC, registry, adapters, manager, runtime thread
  config/           pydantic schema, loader + semantic validation, geometry fingerprint
  vision/           preprocessing (Phase 1); eye/pupil/head/fusion (Phase 2)
  calibration/      models, session, validation, storage (Phase 3)
  control/          cursor, smoothing, dwell, safety (Phase 4)
  desktop/          PySide6 apps: diagnostics (Phase 1), calibration UI, overlay
  services/         anthropic, cartesia, convex (Phase 6, opt-in)
  api/              FastAPI + WebSocket (Phase 6)
dashboard/          React + TS + Vite + Tailwind (Phase 6)
tests/              pytest; runs with no cameras attached
scripts/            dev utilities (e.g. render a synthetic recording)
docs/               architecture, hardware guide, calibration, privacy, sponsors
demo/               demo assets (generated recordings are git-ignored)
```

## 6. Roadmap (demo-critical path first)

1. **Foundation** (done in this phase): camera contract, adapters (webcam / USB / recorded /
   synthetic), profile loader and validator, geometry fingerprint, capture manager, diagnostics UI,
   tests.
2. **Webcam-only end to end (Mode A).** MediaPipe head pose and iris features, then 9-point
   calibration, Quartz cursor and dwell. This is the guaranteed fallback demo, so it is built first.
3. **Calibration quality**: validation targets, error report in points/cm, per-point retry, storage
   bound to the profile fingerprint.
4. **Safety and control polish**: One Euro filter, confidence freeze, jump detection, pause/e-stop,
   Accessibility permission wizard.
5. **External eye cameras (Modes B/C)**: classical pupil detector, 3D eye model, single/dual
   fusion, head-compensated screen mapping, eye-gesture keybinds (done); ROI editor, Pi and
   ESP32 stream adapters (planned). Hardware arrival gates this phase; the synthetic source keeps it
   testable before then.
6. **Communication + sponsors**: AAC board, Claude text help, Cartesia speech, optional Convex.
7. **Demo polish**: latency/accuracy measurement, scripted 2-minute judge flow, docs.

## 7. Highest-risk technical issues

1. **Near-eye imaging.** Many small webcams can't focus at eye-camera distances of a few centimetres.
   Visible-light pupil contrast is poor for dark irises. The usual fix is IR illumination with an
   IR-pass filter and an IR-sensitive sensor, but that raises **eye-safety** questions. Don't add IR
   emitters without checking exposure limits. Budget a fixed-focus versus adjustable-focus experiment
   early.
2. **Head vs. eye geometry.** Glasses-mounted cameras measure eye-in-head. Screen gaze also needs head
   pose, and a monocular webcam estimates it with noise and no reliable depth. A data-driven model
   extrapolates poorly outside the head poses seen during calibration. Mitigations: tell the user to
   keep their head roughly still, recalibrate quickly, and use drift correction.
3. **Frame slip.** Glasses moving on the nose silently shifts the eye-to-camera mapping. Detecting it
   (pupil baseline shift at a known target) and allowing a fast one-point correction matter more than
   model sophistication.
4. **Webcam-only accuracy is coarse.** The Mode A UI must use large targets and snapping. Don't promise
   pointer-level precision.
5. **macOS camera control.** OpenCV's AVFoundation backend honours few `CAP_PROP_*` settings, and
   exposure or focus are often ignored. Device indices can reorder when cameras are re-plugged or when
   an iPhone Continuity camera appears. We read settings back and report them instead of trusting
   them. True UVC control may need a dedicated library later.
6. **Two USB cameras on a MacBook Air.** The Air has only two ports, so a hub is likely. USB bandwidth
   and power with two UVC streams on one hub are unverified; use MJPEG and low resolutions (eye crops
   need little).
7. **Wireless streams.** Wi-Fi JPEG streams (ESP32 especially) have low and jittery frame rates and
   unknown capture timestamps. A crowded hackathon network makes this worse. Prefer wired for the demo.
8. **Permissions.** Camera and Accessibility permissions attach to the launching app (Terminal, IDE).
   The demo must launch from one already-authorized host. Confirmed while building Phase 1: OpenCV
   can't show the camera prompt from a worker thread. So `opengaze` requests camera access on the
   main thread before capture starts (`cameras/permissions.py`), and OpenCV's own request is
   disabled.
9. **Latency stack-up.** Exposure, transfer, processing, smoothing and dwell all add delay. The
   ~100 ms local target is a design goal and must be measured, not assumed.
10. **8 GB RAM demo machine.** MediaPipe, Qt and a browser dashboard all compete for memory. Keep the
    dashboard optional and avoid image copies.

## 8. How geometry changes trigger recalibration

`opengaze/config/fingerprint.py` builds a **geometry snapshot** of each enabled camera from every
setting that could change how eye features map to the screen: role, source type, resolution, rotation,
mirroring, crop, mounting position, geometry, optics, and adapter options. The profile's
`frame_design` and `version` are included too. Only an explicit allowlist of settings known to be
harmless is excluded: `device_id` (indices change on re-plug), `fps`, exposure/focus `controls`, and
`read_timeout_s`. The snapshot is hashed into a fingerprint.

A saved calibration (Phase 3) stores the full snapshot, not just the hash, plus the actual negotiated
resolution observed during calibration. On load, `check_calibration_compatibility()` diffs the stored
snapshot against the current profile and returns every change, e.g.
`cameras.left_eye.rotation_degrees: 0 → 90`. Any change means the calibration is reported as
**incompatible**. The UI must show the reasons and ask the user to recalibrate. It never reuses a
calibration silently. Bumping `version` or renaming `frame_design` in the profile is the manual way to
say "I rebuilt the frame".

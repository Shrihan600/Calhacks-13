# OpenGaze hardware interface: ESP32-CAM glasses ↔ MacBook

This is the contract between the glasses firmware and the Mac software. Change it here first,
then in both [`firmware/opengaze_esp32cam/`](../firmware/opengaze_esp32cam/) and the Mac
adapters, and keep [`opengaze/debug/fake_esp32.py`](../opengaze/debug/fake_esp32.py) (the
emulator) in step.

**Verification status (2026-10-24):** the Mac side is tested against the emulator only. No
real ESP32-CAM board, MPU-6050 or firmware build has been tested. See
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md).

## 1. Topology

```
 RIGHT ESP32-CAM ── MPU-6050 (I²C, GPIO13 SDA / GPIO14 SCL, 3.3 V — pins unverified)
   │  Wi-Fi 2.4 GHz: MJPEG :81/stream · /status · /control · UDP telemetry → Mac:5005
 LEFT ESP32-CAM
   │  Wi-Fi 2.4 GHz: MJPEG :81/stream · /status · /control
 USB-C → 5 V power only (no data)
                          ▼
 MacBook Air M2: ESP32StreamSource ×2 (cameras/esp32.py) · UdpImuSource (imu/sources.py)
                 · MacWebcamSource (head pose) → CameraManager / ImuRuntime → gaze pipeline
```

Each device is configured in `configs/hardware/esp32-glasses.yaml`. Addresses can be changed
without editing files (`--set cameras.left_eye.options.host=192.168.1.41`, or an `--overlay`
file). Changing an address, port or timeout doesn't invalidate a calibration.
Changing resolution, rotation, mirroring or crop does.

## 2. Video: MJPEG over HTTP (both boards)

| item | value |
|---|---|
| URL | `http://<host>:81/stream` (`options.host` / `port` / `path`, or `options.url`) |
| host | IP, or mDNS `opengaze-left.local` / `opengaze-right.local` |
| response | `200`, `Content-Type: multipart/x-mixed-replace;boundary=<b>`; chunked transfer encoding is fine |
| part headers | `Content-Type: image/jpeg`, `Content-Length` (optional but recommended), `X-Sequence: <uint>`, `X-Timestamp-Us: <int64>`, `X-Timestamp: <sec>.<usec>` |
| image | baseline JPEG from the OV2640, 320×240 (QVGA) to start, 640×480 at most |
| rate | target 15–20 fps, set by the firmware |
| clients | one at a time (a second client halves each one's frame rate and shows gaps in `X-Sequence`) |

- `X-Sequence` counts frames **captured** by the camera. A gap means a frame never reached the
  Mac. Diagnostics shows it as `skipped`.
- `X-Timestamp-Us` is the capture time on the ESP32's own clock, in microseconds. `X-Timestamp`
  is the same in Espressif's `seconds.microseconds` format. The Mac takes `X-Timestamp-Us`
  when present. Espressif's stock `CameraWebServer` sends only `X-Timestamp` and no sequence;
  the adapter works with it too (no skip counting).
- Undecodable JPEGs are skipped and counted. Ten in a row count as a broken stream and force a
  reconnect. A JPEG without an end-of-image marker is decoded but flagged `partial`.

### `/status` (port 80, JSON)

Fetched once per connection; shown in diagnostics. Fields are informational and may grow:

```json
{"device_id": "right_esp32", "role": "right_eye", "firmware": "opengaze-esp32cam/0.1.0",
 "camera_ok": true, "framesize": 5, "quality": 12, "fps": 19.8, "frames_sent": 1234,
 "clients": 1, "rssi": -58, "uptime_ms": 123456, "free_heap": 123456, "psram": true,
 "imu": {"sensor": "mpu6050", "present": true, "ok": true, "who_am_i": 104, "rate_hz": 100,
         "sequence": 5000, "sent": 4990, "read_errors": 0, "subscriber": "192.168.1.20:5005"}}
```

`imu` is `null` on the left board. `status_port: null` in the profile skips the request.

### `/control?var=<name>&val=<int>` (port 80)

Uses the `CameraWebServer` API. `framesize` takes an esp32-camera `framesize_t` (5 = QVGA,
8 = VGA in esp32-camera ≥ 2.0) and `quality` takes 4–63. The Mac only calls it with
`options.apply_settings: true`. Either way, the delivered resolution is read back from the first
frame and a mismatch is reported.

## 3. IMU: MPU-6050 JSON over UDP (right board)

### Subscribe handshake (so the Mac's IP is never compiled in)

1. The Mac binds UDP `0.0.0.0:5005` (`imu.options.port`).
2. Every second, it sends `{"type": "subscribe", "port": 5005}` from that socket to
   `<right-host>:5006` (`imu.options.device_host` / `device_port`).
3. The firmware streams telemetry to the sender address and port of the latest subscribe. It
   stops 5 s after the last one, so a crashed Mac app doesn't leave it transmitting forever.

The firmware can instead send to a fixed `OPENGAZE_IMU_FIXED_TARGET_IP`. The Mac then just
listens; leave `device_host` unset.

### Telemetry message

One JSON object per datagram, or a JSON array of objects (batching), or one object per line:

```json
{"device_id": "right_esp32", "sensor": "mpu6050", "sequence": 1024,
 "timestamp_us": 123456789,
 "accel_x": 0.02, "accel_y": 0.01, "accel_z": 9.79,
 "gyro_x": 0.12, "gyro_y": -0.08, "gyro_z": 0.03,
 "temp_c": 31.2, "valid": true}
```

| field | type | unit / meaning |
|---|---|---|
| `device_id` | string | `right_esp32`; others are dropped when `expected_device_id` is set |
| `sequence` | uint | +1 per **sample** (sent or not). Gaps = lost packets. A large drop = device restarted |
| `timestamp_us` | int64 | sample time on the ESP32 monotonic clock (`esp_timer_get_time()`). 32-bit wrap is tolerated |
| `accel_*` | float | **m/s²**, sensor axes. Reads ≈ +9.81 on the axis pointing *up* at rest |
| `gyro_*` | float | **rad/s**, sensor axes, raw (the Mac learns the bias) |
| `temp_c` | float | optional, °C |
| `valid` | bool | `false` = the sensor read failed; accel/gyro may be absent. The sequence still counts |

The default units are the ones in the table; the profile can override them
(`gyro_units: deg_s`, `accel_units: g`). **Validation:** diagnostics warns when the readings
don't fit the configured units. A median accelerometer magnitude near 1.0 means g was sent as
m/s². Near 96 means the reverse. Gyro peaks above the sensor's range mean °/s were sent as
rad/s.

The older text protocol (`t_us,gx,gy,gz,ax,ay,az` lines, deg/s and g) is still accepted
(`format: auto` / `text`).

### Rate

100 Hz to start (`OPENGAZE_IMU_RATE_HZ`). Drop to 50 Hz if camera streaming suffers. The Mac
measures the rate from device timestamps, so bursty Wi-Fi delivery doesn't distort it.

## 4. Time synchronization

The three clocks (left ESP32, right ESP32, Mac) are **never assumed to agree**.

- Every frame and IMU sample carries a **device timestamp** and a **sequence number**. The
  Mac stamps a **host receive time** (`time.monotonic_ns()`) the moment it reads each one.
- Per device, `DeviceClockSync` (`opengaze/cameras/timing.py`) keeps
  `offset = min(host_receive − device_time)` over a sliding 10 s window. The message that
  waited least in Wi-Fi retries and buffers gives the tightest bound. The device time mapped
  onto the Mac clock is `device_time + offset`. That lands *late* by the minimum one-way
  delay (unknown, a few ms on a quiet LAN), never early. The window absorbs crystal drift
  (tens of ppm).
- An offset jump > 1 s (reboot, clock change) resets the estimate. A batched datagram updates
  the estimate with all its readings before mapping any of them.
- `CameraFrame.capture_ns` holds the mapped time, with `timestamp_source = device` and
  `timestamp_uncertainty_ns` set to the median recent excess delay. IMU samples carry
  `capture_ns` the same way, and `ImuRuntime.pose_at(t)` looks up the head pose at an eye
  frame's capture time.
- Left and right eye frames are not captured in sync. The gaze tracker pairs the newest
  frame of each eye and records the skew between them. Precise stereo synchronization is out
  of scope.
- `delay` in diagnostics is each message's receive delay *above the best seen*. A climbing
  value means frames are queueing somewhere (weak signal, busy channel).

## 5. Failure behaviour (Mac side)

| situation | what happens |
|---|---|
| board off / wrong IP | connect fails with a specific message; retried with backoff (0.5 → 5 s) |
| Wi-Fi drop mid-stream | no frame within `read_timeout_s` (3 s) → reconnect; tile shows STALLED after 1 s |
| macOS blocks LAN access | "no route to host … Local Network" (System Settings → Privacy & Security → Local Network) |
| wrong endpoint | "expected a multipart MJPEG stream" |
| corrupt / truncated JPEG | skipped and counted / decoded and marked partial |
| IMU packets lost | `dropped` counter from sequence gaps; head rotation falls back to the webcam after 0.5 s of silence |
| IMU sensor fault | `valid: false` readings counted as `invalid`; firmware retries the sensor every second |
| no telemetry at all | IMU badge DOWN; with subscribes sent but nothing back, the hint says to check IP, port or the macOS firewall |

Webcam-only mode (`configs/hardware/webcam-only.yaml`) needs none of this hardware.

## 6. Testing without hardware

```bash
opengaze fake-esp32 --drop-rate 0.05 --jitter-ms 30 --disconnect-every 20   # terminal 1
opengaze diagnostics --profile configs/hardware/esp32-emulator.yaml           # terminal 2
opengaze headless --profile configs/hardware/esp32-emulator.yaml \
  --set cameras.head.enabled=false --seconds 30 --metrics                     # or headless
```

The emulator serves the same protocol (chunked MJPEG with sequence/timestamp headers,
`/status`, `/control`, subscribe + JSON telemetry) from synthetic eye images and a
synthetic head trajectory. Its device clocks start at an arbitrary offset with 20 ppm drift.
Everything it serves is marked emulated, and diagnostics labels it **EMULATED**. Passing
against the emulator shows the Mac side follows this document. It doesn't show the
firmware does.

`--metrics` writes a JSONL row per input per second (state, fps, age, drops, reconnects, IMU
counters; numbers only, no images) to `logs/metrics/` and prints a summary on exit.

## 7. Safety and privacy

- The Inland MD0456 documentation specifies **20 cm antenna separation**. Do live Wi-Fi tests
  on a bench at that distance, not on someone's head, until the hardware team has completed
  an RF-exposure review for head-worn use. The software never requires the glasses to be worn:
  diagnostics, the emulator and bench-mounted cameras all work.
- Eye and face video stays on the Mac. Diagnostics records metrics (numbers) only. Video
  recording (`opengaze run --record-video`) needs `privacy.allow_recording: true`. No module
  uploads images.

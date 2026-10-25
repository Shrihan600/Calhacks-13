# PROJECT_STATUS (updated 2026-10-24)

## Summary
Design, firmware source, Python receiver and docs are written. **No hardware test has been performed.** Firmware has **never been compiled**.

## Done
| Item | State | Evidence |
|---|---|---|
| Board research (MD0456 = Keyestudio, AI-Thinker pinout expected) | Done, partly unverified | docs/wiring.md §1 |
| USB-C sink + power design, budget, BOM, protection | Written | docs/wiring.md, docs/power-budget.md |
| Firmware: camera, MJPEG server, control/status API, MPU-6050 UDP, Wi-Fi watchdog, NVS config | Written, **uncompiled** | firmware/ |
| Python receiver (dual MJPEG, UDP, sync helpers, dashboard, CSV, headless) | Written, **tested vs simulator** | below |
| Simulator | Written, used | receiver/tools/sim_device.py |
| Docs set | Written | docs/ |

## Receiver test performed (simulator only, Linux sandbox, OpenCV 4.13 headless)
- Two simulated streams: both measured 25.0 fps; decode ≈0.3–0.4 ms/frame; relative delay <3 ms on loopback.
- Right server killed and restarted mid-run: client went `reconnecting`, then recovered automatically.
- IMU UDP with injected 5 % loss: receiver reported ≈5–6 % (a stray second sim was briefly also sending in one run, inflating a later reading; the first run is the clean one).
- Dashboard render checked visually (PNG). Interactive `cv2.imshow` window **not** exercised (no display).
- **Not representative of Wi-Fi:** loopback has no jitter, loss or buffering beyond what the sim injects.

## Added later (Oct 25)
- Wi-Fi firmware flashed and streaming on the first camera (user-reported: PSRAM present, sensor reports OV3660, ~8.5 fps at QVGA over a phone hotspot).
- Wired mode (`*_serial` envs, `receiver/serial_client.py`, `docs/wired-mode.md`): receiver tested against a simulated serial camera; **firmware not compiled or run**.
- Wi-Fi diagnostics (scan + failure reasons) and frame timing in `/status` added; the timing build has not been flashed.

## NOT done / unknown
- Firmware compile, flash, boot, camera init, PSRAM presence, GPIO13/14 I²C on the real board.
- Any current/voltage measurement; advertised current of the Mac port; cable drop.
- Real fps, latency, packet loss, sync skew, temperature.
- RF exposure assessment for head-worn use (open safety item).
- MD0456 FCC grant text and schematic were not read (only listing/record pages).
- Integration with the existing OpenGaze pipeline (Test H).

## Known risks / assumptions in code
- Targets Arduino-ESP32 core 3.x (pioarduino). `pin_sccb_*` fields, `Wire.begin(sda,scl,freq)`, `s6_addr` in the stream peer lookup are the likeliest compile surprises.
- Camera driver's `fb->timestamp` assumed to be in the `esp_timer` domain (code falls back to "now" if it looks wrong). Check that `X-Timestamp-Us` advances ~1e6 per second and matches IMU timestamps' clock.
- Only one `/stream` client per device at a time.
- IMU period uses a 1 ms FreeRTOS tick: rates not dividing 1000 are rounded.

## Next steps (in order)
1. Test A (power) with a USB-C meter and dummy load.
2. `pio run -e i2c_scan` → fix compile errors if any → Test C.
3. `pio run -e left` → Tests B, D. Then `right` → E, F, G.
4. Measure sync skew (testing.md), then Test H.
5. Resolve RF compliance before any head-worn demo.

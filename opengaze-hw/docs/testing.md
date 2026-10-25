# Staged bench validation

Rule: do not start stage N+1 until stage N passes. Record results in the tables; **empty = not run.** Nothing below has been executed on hardware yet.

Mac setup (once):
```bash
cd opengaze-hw/receiver
python3 -m venv ~/.venvs/opengaze-rx && source ~/.venvs/opengaze-rx/bin/activate
pip install -r requirements.txt
python tools/sim_device.py --role left  --http-port 8081 &     # optional: no-hardware self-test
python tools/sim_device.py --role right --http-port 8082 &
python main.py --left 127.0.0.1:8081 --right 127.0.0.1:8082
```

## Test A — Power (no ESP32, no MPU, nothing on glasses)
1. Unplugged: CC1–GND and CC2–GND = 5.1 kΩ each; VBUS–GND open circuit (no short).
2. Plug cable (via USB-C power meter if available) into the Mac, switch OFF. Measure CC1/CC2→GND voltage → read the advertised current from the table in wiring.md §6. **Record it.**
3. Switch ON, no load: VBUS 4.9–5.25 V. No port drop-out (watch the meter).
4. Dummy load: connect one 10 Ω/5 W across the rail (0.5 A) for 60 s; then two in parallel (≈1 A) for 60 s *only if the port advertised ≥1.5 A*. Pass: rail ≥ 4.75 V at the load end, no port shutdown, resistors/breakout/PTC not hot to the touch (> ~50 °C = stop).
5. Hot-plug/switch cycling ×10: no trips.
6. PTC check: a 3 Ω load must trip it within seconds and the port must survive (optional, skip if unsure).

| Item | Result |
|---|---|
| CC voltage / advertised current | |
| Rail @ 0.5 A / @ 1 A | |
| Max temp of breakout/PTC | |

If the port advertises only default current or the rail sags below 4.75 V at 1 A → use the wall-adapter bench alternative (power-budget.md §4) and continue.

## Test B — Single camera (LEFT, MPU not connected yet)
Flash `left`. Power from programmer or the rail (one source only). Open `http://IP:81/stream` in a browser or run the receiver with `--right` pointing at a dummy.
Pass: boot log shows `sensor PID=0x26`, stable image for 10 min, `/status` shows `fps`, `capture_fail` = 0, reset reason power-on.
Record: PSRAM yes/no ___, fps at qvga ___, vga ___, `fps=0` ceiling ___ (curl `…/control?var=fps&val=0`), RSSI ___.

## Test C — MPU-6050
Wire per wiring.md. Flash `i2c_scan`. Pass: device at 0x68 (or 0x69), WHO_AM_I printed, at rest |a| ≈ 1.00 ± 0.05 g, gyro within ±3 °/s, tilt changes ax/ay/az sensibly, temperature plausible (25–40 °C). Fail → wiring.md §2 verification list.

## Test D — MPU + camera simultaneously
Flash `left`. Stream open **and** receiver listening on UDP 5005.
Pass: `/status` `imu.hz` ≈ configured (100), `imu.errors` = 0, stream fps unchanged vs Test B (±10 %), no resets over 10 min. Receiver shows IMU rate ≈ 100 Hz, loss < 1 % (Wi-Fi UDP loss is not zero in general; record what you get).

## Test E — Second camera
Flash `right`. Both on the same network. Pass: both `/status` pages respond, both streams run (one client each), no cross-interference (compare each fps against its solo number).

## Test F — Combined Mac receiver
`python main.py --left <ip or name> --right <ip or name> --log-csv f1.csv`
Pass: two labeled feeds, IMU panel live, per-device link OK, reconnect works (unplug RIGHT's power for 5 s; the tile shows `reconnecting` then recovers), no growing lag (wave a hand: displayed motion stays near-real-time and `rel.delay` stays small).

## Test G — Reliability / soak (≥ 30 min, 2 h preferred)
Run headless with logging:
`python main.py --headless --duration 1800 --left … --right … --log-csv soak.csv`
Record from the final summary and CSV: achieved fps (each), stalls, reconnects/failures, IMU packet loss %, max `rel.delay`, RSSI range, free-heap trend (flat = no leak), reset reasons (any `brownout`?), enclosure/board temperature (IR thermometer at 0/15/30 min), rail voltage at the board under load, subjective responsiveness.
Suggested pass thresholds (adjust to your needs, they are not guarantees): 0 resets, 0 brownouts, both streams ≥ 15 fps sustained, IMU loss < 2 %, heap not trending down, surface temp ≤ 40 °C.

## Synchronization check (measures what the docs only estimate)
1. LED/phone-flash test: flash a bright light visible to **both** cameras; log the frame index where each first shows it, using `X-Timestamp-Us` → mapped Mac times (`ClockTracker.to_host`). The difference is your **actual cross-camera skew**; repeat ×20 and quote mean ± max.
2. IMU↔left-frame: tap the glasses on the table; find the gyro/accel spike time and the frame where the image jolts. Same clock, so the skew is purely measurement/pipeline.
3. Glass-to-glass latency: film a running stopwatch next to the live window with a phone at 60 fps+; difference in the stopwatch readings = end-to-end latency. (`rel.delay` in the dashboard is only the delay above the best recent frame, not absolute.)

## Test H — OpenGaze integration (only after A–G pass)
The receiver is built to be imported:
```python
from camera_client import MJPEGClient      # .get_latest() -> Frame(img BGR ndarray, dev_ts_us, host_ts, ...)
from imu_client import TelemetryReceiver   # .last_imu, .imu deque (left device), .dev[1|2].status
from synchronization import ImuAligner     # near_left_frame(ts_us), near_right_frame(ts_us)
```
Adapter work: wrap two `MJPEGClient`s behind the pipeline's existing "frame source" interface (frames are already latest-only, so the pipeline never sees a queue), feed `TelemetryReceiver.last_imu` as the head-motion side channel, keep the MacBook webcam + MediaPipe for head tracking. **Do not add eye-tracking models until Tests A–G are green.**

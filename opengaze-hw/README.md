# OpenGaze hardware bring-up: dual ESP32-CAM + MPU-6050 → MacBook

Two Inland MD0456 ESP32-CAM boards stream left/right eye video over Wi-Fi; the LEFT board also streams MPU-6050 telemetry (UDP). One USB-C cable powers the glasses (power only). The Mac does all processing.

```
opengaze-hw/
  firmware/            PlatformIO project (one codebase, envs: left, right, i2c_scan)
    include/           camera_config.h, network_config.h, telemetry_protocol.h, secrets.h.example, ...
    src/               main.cpp, camera_ctl.cpp, imu.cpp, stream_server.cpp, i2c_scan.cpp
    tools/             make_arduino_sketches.sh (optional Arduino-IDE layout)
  receiver/            Python app: main.py, camera_client.py, imu_client.py, synchronization.py, diagnostics.py
    tools/sim_device.py   hardware-free simulator that speaks the firmware's wire format
  docs/                wiring.md, power-budget.md, firmware-setup.md, testing.md
  PROJECT_STATUS.md  AI_HANDOFF.md
```

## Run the receiver (no hardware needed to try it)
```bash
cd receiver && python3 -m venv ~/.venvs/opengaze-rx && source ~/.venvs/opengaze-rx/bin/activate
pip install -r requirements.txt
python tools/sim_device.py --role left --http-port 8081 &  python tools/sim_device.py --role right --http-port 8082 &
python main.py --left 127.0.0.1:8081 --right 127.0.0.1:8082          # q quits, s saves snapshots
# real hardware:
python main.py --left opengaze-left.local --right opengaze-right.local --log-csv run.csv
```

## Build the firmware
See `docs/firmware-setup.md`. Short version: `pip install platformio`, copy `include/secrets.h.example` → `secrets.h`, then `pio run -e i2c_scan -t upload`, then `-e left`, then `-e right`.

## No Wi-Fi? Wired mode
See `docs/wired-mode.md`: `pio run -e left_serial` / `right_serial`, then `python main.py --left-serial PORT --right-serial PORT`. Uses an Arduino Uno/Nano (RESET to GND) per camera as the USB-serial bridge.

## Honest status (details in PROJECT_STATUS.md)
- **Tested:** Python receiver against the simulator (parsing, 25 fps ×2, reconnect, UDP loss accounting, dashboard drawing).
- **Not compiled, not flashed, not run on hardware:** all firmware. No power, I²C, camera, throughput, latency or sync numbers exist yet.
- **Open safety item:** head-worn Wi-Fi transmitters need an RF exposure assessment (docs/wiring.md §9). Bench and ≥20 cm setups only until resolved.

# Firmware setup, flashing, network

> I could not compile the firmware in my environment (the PlatformIO/Espressif toolchain hosts are not reachable from it), so **the first `pio run` is the first compile.** Expect to fix a typo or two; send me the error text. The receiver *was* run against simulated devices (see PROJECT_STATUS.md).

## 1. Toolchain choice
**PlatformIO CLI + pioarduino** (Arduino-ESP32 core 3.x). One source tree builds three targets: `left`, `right`, `i2c_scan`. Reproducible, no board-menu clicking, works from the terminal on Apple Silicon.
Arduino IDE alternative: `firmware/tools/make_arduino_sketches.sh` generates `arduino/left_camera/` and `arduino/right_camera/` sketch folders (untested path). In Arduino IDE 2.x install **"esp32 by Espressif Systems" ≥ 3.0**, Board = **AI Thinker ESP32-CAM**, PSRAM = Enabled, Partition = Huge APP.
Core 2.x is *not* supported: the code uses the 3.x field names `pin_sccb_sda/scl`.

```bash
# one-time (macOS, Apple Silicon)
xcode-select --install                     # if git/python3 are missing
python3 -m venv ~/.venvs/opengaze && source ~/.venvs/opengaze/bin/activate
pip install platformio
cd opengaze-hw/firmware
cp include/secrets.h.example include/secrets.h   # edit SSID / password (2.4 GHz WPA2-PSK)
```
The first build downloads the toolchain and framework (~1 GB).

## 2. Programmer
The ESP32-CAM has **no USB**. You need one of:
1. **ESP32-CAM-MB base (CH340, micro-USB)** — plug the module in; auto-reset/auto-boot-mode, no GPIO0 jumper. Easiest. (Remove the board from the base afterwards; do not power it from both base and glasses rail.)
2. **CP2102 USB-UART module that exposes both 3V3 and 5V pins** (logic is always 3.3 V). Manual wiring below.
Avoid FTDI boards whose single VCC pin follows a 3.3/5 V jumper unless the jumper is on **3.3 V** and 5 V power comes elsewhere; 5 V logic must never reach the ESP32 UART pins.
If no `/dev/cu.*` port appears after plugging in, install the vendor driver (WCH CH34x or Silicon Labs CP210x) and use a **data** cable.

### Manual wiring (CP2102)
| Programmer | ESP32-CAM |
|---|---|
| 5V | 5V (needs ≥ ~500 mA; if the board browns out, use a separate 5 V supply for the board and **do not connect the programmer's 5V**) |
| GND | GND (any) |
| TXD | U0R |
| RXD | U0T |
| — | **IO0 → GND (jumper wire) before reset = download mode** |

Procedure: jumper IO0–GND → press the board's RESET button (or replug power) → run upload → when "Hard resetting" appears **remove the IO0 jumper** → press RESET → open the monitor.
Power rule: **exactly one 5 V source on the board at any time** (programmer *or* glasses rail).

## 3. Bring-up order (do not skip)
```bash
source ~/.venvs/opengaze/bin/activate && cd opengaze-hw/firmware
ls /dev/cu.*                                      # find the port, e.g. /dev/cu.usbserial-0001

# (a) I2C/MPU check — on the LEFT board with the MPU wired
pio run -e i2c_scan -t upload --upload-port /dev/cu.usbserial-0001
pio device monitor -p /dev/cu.usbserial-0001 -b 115200
#   expect: "device at 0x68 <- MPU-6050 candidate", WHO_AM_I=0x68, |a| ≈ 1 g at rest

# (b) LEFT firmware
pio run -e left  -t upload --upload-port /dev/cu.usbserial-0001
# (c) swap boards, then RIGHT firmware
pio run -e right -t upload --upload-port /dev/cu.usbserial-0001
```
Serial monitor: **115200 baud, 8N1**. Boot log to expect (LEFT):
```
reset reason=1                   (1=power-on; 9=BROWNOUT → fix power)
PSRAM: yes (… bytes free)        (NO => see troubleshooting)
[cam] sensor PID=0x26            (OV2640)
[imu] device at 0x68 WHO_AM_I=0x68
[wifi] connected: ip=… rssi=…
[ready] stream: http://IP:81/stream   status: http://IP/status
```

## 4. Endpoints (actual, from the code)
| URL | Meaning |
|---|---|
| `http://IP:81/stream` | MJPEG, one client at a time. Per-frame headers `X-Timestamp-Us`, `X-Frame-Id` |
| `http://IP/status` | health JSON (fps, heap/PSRAM, RSSI, reconnects, reset reason, IMU state, config) |
| `http://IP/capture` | single JPEG |
| `http://IP/control?var=VAR&val=VAL[&save=1]` | live config; `save=1` persists to flash |
| `http://IP/reboot` | restart |

`VAR`: `framesize` (qqvga, qvga, cif, hvga, vga, svga), `quality` (4–63, lower=better), `fps` (0 = unlimited), `imu_hz` (10–500), `brightness/contrast/saturation` (−2..2), `hmirror`, `vflip`, `telem_host` (`auto` or IPv4).
Examples: `curl "http://IP/control?var=framesize&val=vga&save=1"`, `curl "http://IP/control?var=fps&val=30"`.
Defaults: QVGA 320×240, quality 12, 25 fps cap, IMU 100 Hz. **No FPS figure is promised; measure it** (Test B/G). `fps=0` shows the real ceiling.

UDP telemetry goes to port **5005** on the host that most recently opened `/stream` (`auto`) or to a fixed IP set with `telem_host`. mDNS names `opengaze-left.local` / `opengaze-right.local` are advertised; hotspots that block multicast need raw IPs (printed on serial, shown in your router's client list).

## 5. Network recommendation
- Use a **dedicated 2.4 GHz network you control**: a cheap travel router (offline LAN, WPA2) is best; a phone hotspot with "Maximize Compatibility" (2.4 GHz) works for bench tests. Pin the channel to 1, 6 or 11, away from congestion.
- **Do not use hackathon/venue/campus Wi-Fi**: client isolation, captive portals, WPA-Enterprise, and congestion will break discovery and multi-stream throughput. ESP32 cannot join 802.1X networks in this firmware.
- Mac and both ESP32s must be on the same subnet with **client/AP isolation off**.
- Security: streams are plain HTTP/UDP with no authentication. Anyone on the SSID can watch the eye cameras. Keep the network private, WPA2 with a strong passphrase, no internet uplink needed.
- Airtime: two ~320×240 JPEG streams at ~25 fps are expected to be a few Mb/s total, but both share one channel and the Mac; measure (Test F/G).

## 6. Troubleshooting
| Symptom | Likely cause / action |
|---|---|
| `Failed to connect to ESP32: Wrong boot mode` / timeout | IO0 not grounded at reset (manual wiring), TX/RX swapped, bad ground, or weak 5 V. Retry with IO0–GND, press RESET, upload. |
| Upload starts then fails mid-way | Supply droop: use a better 5 V source; lower `upload_speed` to 115200 in `platformio.ini`. |
| Boot loop, `Brownout detector was triggered` / reset reason 9 | Supply too weak/long thin cable: check ≥4.75 V at the board pin under load, add the 330 µF local cap, shorter cable, different port. Do **not** disable the detector. |
| `E (…) camera: Camera probe failed with error 0x105/0x106` | Ribbon cable not fully seated/locked (latch), camera not OV2640, or pin map differs from AI-Thinker. Reseat first. |
| `PSRAM: NO` / `cam_hal: FB-SIZE` / `Failed to get the frame on time` | PSRAM absent or not enabled (`-DBOARD_HAS_PSRAM`), board is a non-PSRAM variant (firmware falls back to QVGA single buffer), or marginal power. |
| `cam_dma_config` / `no mem for frame buffer` | PSRAM missing at larger sizes: stay at QVGA. |
| Stripes/noise in the image | Power ripple: add the cap, shorten wires, keep XCLK (IO0) path away from I²C leads. |
| I²C scan finds nothing | microSD slot not empty, 3V3/GND missing, SDA/SCL swapped, breakout lacks pull-ups, wires too long. Try the scanner at 100 kHz (it already does). |
| MPU found but WHO_AM_I ≠ 0x68 | Clone/MPU-6500-family part; likely works, verify values. |
| Stream stutters / big jumps in `rel.delay` | RSSI worse than ≈ −75 dBm, crowded channel, 640×480 at low quality value. Lower size or raise `quality` number. Only one `/stream` client at a time. |
| Mac cannot resolve `.local` | Use IPs. |

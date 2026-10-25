# OpenGaze ESP32-CAM firmware

> **Status: reference implementation, never compiled or flashed.** It was written against
> [`docs/hardware-interface.md`](../docs/hardware-interface.md) without access to the boards
> or an Arduino toolchain. Expect build fixes on first use. Until someone flashes it and
> records the result in `PROJECT_STATUS.md`, treat every claim here as unverified.

One sketch, [`opengaze_esp32cam/`](opengaze_esp32cam/), builds for both boards:

| board | `OPENGAZE_ROLE_RIGHT` | serves |
|---|---|---|
| LEFT ESP32-CAM | `0` | left-eye MJPEG on `:81/stream`, `/status`, `/control` |
| RIGHT ESP32-CAM | `1` | right-eye MJPEG, `/status`, `/control`, **and** MPU-6050 JSON over UDP |

## Build and flash

1. `cp opengaze_esp32cam/secrets.example.h opengaze_esp32cam/secrets.h` and enter the
   2.4 GHz Wi-Fi network (`secrets.h` is git-ignored).
2. Set `OPENGAZE_ROLE_RIGHT` in `config.h` (or pass `-DOPENGAZE_ROLE_RIGHT=1`).
3. Board: **AI Thinker ESP32-CAM**, partition scheme **Huge APP (3MB No OTA)**, PSRAM enabled.
   arduino-esp32 core 2.0.x or 3.x.
4. ESP32-CAM boards have no USB-serial chip: use a 3.3 V USB-serial adapter (or the
   ESP32-CAM-MB shield), hold **GPIO0 to GND** while resetting to enter the bootloader, flash,
   then remove the jumper and reset.

With arduino-cli (untested):

```bash
arduino-cli core install esp32:esp32
arduino-cli compile --fqbn esp32:esp32:esp32cam:PartitionScheme=huge_app \
  --build-property "build.extra_flags=-DOPENGAZE_ROLE_RIGHT=1" opengaze_esp32cam
arduino-cli upload -p /dev/cu.usbserial-XXXX --fqbn esp32:esp32:esp32cam opengaze_esp32cam
```

5. Open the serial monitor at 115200 baud. The board prints its IP, its mDNS name
   (`opengaze-left.local` / `opengaze-right.local`) and, on the right board, whether the
   MPU-6050 answered.

## Check it from the Mac

```bash
curl http://opengaze-right.local/status          # JSON: camera_ok, fps, rssi, imu.ok …
open http://opengaze-left.local:81/stream        # the browser shows the MJPEG stream
opengaze headless --profile configs/hardware/esp32-glasses.yaml --seconds 30 --metrics
opengaze diagnostics --profile configs/hardware/esp32-glasses.yaml
```

If mDNS doesn't resolve on the network, pass the IPs from the serial log:
`--set cameras.left_eye.options.host=… --set cameras.right_eye.options.host=… --set imu.options.device_host=…`

## Things the hardware team must verify

- **I²C pins.** GPIO13 (SDA) / GPIO14 (SCL) are free only while the microSD slot is unused.
  Confirm against the actual board revision. Keep I²C at 3.3 V logic. Most MPU-6050
  breakouts have 3.3 V pull-ups, but check.
- **I²C port.** The camera's SCCB bus normally uses I2C port 1, leaving `Wire` (port 0) for
  the MPU-6050. If the camera fails to initialise once `Wire.begin` runs, this is the first
  suspect.
- **MPU-6050 WHO_AM_I.** Genuine parts return `0x68`; clones often return `0x70`, `0x72` or
  `0x98`. The firmware reports the value in `/status` and does not refuse clones.
- **Power.** Both boards draw current spikes (~300 mA+) when Wi-Fi transmits. Brown-outs show
  as random resets in the serial log.
- **RF exposure.** The board documentation specifies 20 cm antenna separation. Bench-test
  first, at that distance. `OPENGAZE_WIFI_TX_POWER` lowers output power, but it is not a
  compliance argument.
- **Units.** The firmware sends m/s² and rad/s. `opengaze diagnostics` warns if the readings
  don't look like those units (for example, 1.0 at rest means g was sent).

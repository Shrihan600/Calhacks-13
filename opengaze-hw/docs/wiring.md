# Wiring, pinout, USB-C sink, BOM

> Status: **design only. Nothing here has been built or measured.** Items marked ⚠ VERIFY must be checked on your physical boards before power is applied.

## 1. What was verified about the Inland MD0456 (and what was not)

| Claim | Evidence | Confidence |
|---|---|---|
| MD0456 = Micro Center SKU 221739, "ESP32-CAM, OV2640" | Micro Center listing | Verified (listing) |
| Made by Shenzhen KEYES DIY Robot (Keyestudio), FCC ID 2ADMF-MD0456 | fccid.io record | Verified (record title only; I could not open the grant PDFs) |
| Uses the AI-Thinker ESP32-CAM pinout | Keyestudio's own ESP32-CAM docs list the AI-Thinker SD mapping (GPIO14 CLK, GPIO15 CMD, GPIO2 D0, GPIO4 D1/LED, GPIO12 D2, GPIO13 D3; GPIO1/3 UART; GPIO0 download mode). Their board is the same form factor. | **Probable, not proven for MD0456.** The firmware prints the OV2640 PID at boot; a successful camera init is the practical proof that the camera pin map is right. |
| PSRAM present | Listing says "SDRAM 520KB" — that is the ESP32's *internal* SRAM, so it does **not** confirm PSRAM | **Unverified.** Serial log prints `PSRAM: yes/NO`. Without PSRAM the firmware falls back to QVGA single-buffer. |
| MPU-6050 / BME280 / mic "reserved" | Boilerplate in the listing text | Ignore; the board has none of these |
| 20 cm RF separation | You reported it from the FCC grant. I could not read the grant text | Taken as given; see §9 |

Keyestudio's own page says a plain USB port cannot power the board and recommends ≥5 V / 500 mA external supply — consistent with our need for a proper 5 V rail.

## 2. ESP32-CAM header functions (AI-Thinker layout; read your silkscreen)

Two 1×8 headers. Do **not** rely on position from this document — match silkscreen labels:
`5V, GND, IO12, IO13, IO15, IO14, IO2, IO4` on one header; `3V3, IO16, IO0, GND, VCC, U0R, U0T, GND` on the other (three GND pins total on the module).

| Pin | Use in OpenGaze |
|---|---|
| 5V | Power input (from the 5 V rail). **Use this, not "VCC".** |
| "VCC" | Jumper-selectable 3.3 V/5 V on some boards. **Do not power the board from it; do not connect anything to it.** |
| 3V3 | Regulated output → MPU-6050 VCC (≈4 mA load) |
| GND | Common ground |
| IO13 | **I²C SDA** (microSD DAT3 line; not a strapping pin) |
| IO14 | **I²C SCL** (microSD CLK line; not a strapping pin) |
| IO0 | Camera XCLK **and** boot-mode select. Only touched during flashing (to GND). |
| IO12 | **Do not use** (flash-voltage strapping: high at boot can brick boot) |
| IO2, IO4, IO15 | Leave unconnected (IO4 = flash LED, IO2/IO15 strapping/SD) |
| IO16 | PSRAM chip-select on most boards. **Never use.** |
| U0T / U0R | UART0 for flashing and serial monitor only |

GPIO13/14 are free **only because the microSD card is not used. Keep the slot empty.**
Camera uses GPIO 0,5,18,19,21,22,23,25,26,27,32,34,35,36,39 (I²C camera bus on 26/27, hardware port separate from our `Wire` on 13/14).

### ⚠ VERIFY before wiring (5 minutes, multimeter)
1. Silkscreen shows `IO13` and `IO14` on the headers.
2. Continuity: IO13/IO14 header pins go to the microSD slot pads (DAT3 / CLK), not to anything else.
3. Power off, resistance IO13→3V3 and IO14→3V3: if you read ≈10 kΩ or less there is a board-level pull-up. Fine for I²C (parallel with the MPU's), but tell me the number.
4. With the board powered (bench, alone): 3V3 pin = 3.2–3.4 V; IO13/IO14 idle at ≤3.3 V (never 5 V).
5. Flash `i2c_scan` (see firmware-setup.md): it reports `device at 0x68`, WHO_AM_I and live data. **Only then is Wire.begin(13, 14) confirmed for your board.** Firmware uses exactly `Wire.begin(13, 14, 400000)`.

## 3. MPU-6050 breakout check
Typical GY-521: onboard 3.3 V LDO, 4.7 kΩ I²C pull-ups to its *regulated* rail, no level shifter, AD0 pulled low (address 0x68).
⚠ VERIFY on your module: (a) VCC accepts 3.3 V (LDO dropout means output ≈3.1–3.3 V — fine); (b) pull-ups go to the regulated 3.3 V, not to raw VCC; (c) no 5 V on SDA/SCL.
Power it **from the ESP32-CAM's 3V3 pin**, never from 5 V. A bare MPU-6050 chip breakout without pull-ups needs 4.7 kΩ from SDA and SCL to 3V3; a GY-521 needs none (adding more lowers the pull-up resistance).
AD0→GND = 0x68, AD0→3V3 = 0x69. Firmware tries both. INT, XDA, XCL: unconnected.

## 4. Schematic

```
 MacBook USB-C port (source, 5 V)
        |  USB-C cable (use a short, thick, 3 A-rated cable; data lines unused)
        v
 +--------------------------------------------+
 | USB-C sink breakout (Adafruit 4090 or equiv)|
 |   CC1 --[5.1k]-- GND   (Rd)                 |   <- both CC pins, one 5.1k each. ⚠ VERIFY
 |   CC2 --[5.1k]-- GND   (Rd)                 |
 |   D+  D-  SBU  : NOT CONNECTED (insulate)   |
 |   VBUS ----------------+      GND ------+   |
 +------------------------|---------------|---+
                          |               |
                     [SW1 slide switch, >=2 A, optional]
                          |               |
                     [F1 PTC 1.5 A hold]  |
                          |               |
   +5V_RAIL --+--------+--+------+         GND_RAIL (star point here)
              |        |         |            |
            C1 100uF  D1 TVS    C2 10uF       |
            (low ESR) SMAJ5.0A  ceramic       |
              |        |         |            |
   ===========+========+=========+============+==========  (wires to boards)

   +5V_RAIL --[C3 330uF]+[C4 10uF]+[C5 100nF]-- ESP32-CAM LEFT  5V   (caps on the board's 5V/GND pins, <3 cm)
   GND_RAIL ---------------------------------- ESP32-CAM LEFT  GND
   +5V_RAIL --[C6 330uF]+[C7 10uF]+[C8 100nF]-- ESP32-CAM RIGHT 5V
   GND_RAIL ---------------------------------- ESP32-CAM RIGHT GND

   LEFT 3V3 ---- MPU-6050 VCC
   LEFT GND ---- MPU-6050 GND
   LEFT IO13 --- MPU-6050 SDA
   LEFT IO14 --- MPU-6050 SCL
   MPU AD0 -> GND (0x68)

   NO connection from the rail to U0T/U0R/IO0/IO12. NO Arduino Nano. NO second 5 V source.
```
Only **one** source ever touches +5V_RAIL. Programming rule (§6) keeps it that way.

## 5. Pin-to-pin wiring table

| # | From | To | Wire | Notes |
|---|---|---|---|---|
| 1 | USB-C VBUS | SW1 in | 22 AWG | |
| 2 | SW1 out | F1 in | 22 AWG | |
| 3 | F1 out | +5V_RAIL node (C1, D1, C2) | 22 AWG | |
| 4 | USB-C GND | GND_RAIL node | 22 AWG | Same node as C1/D1/C2 negative |
| 5 | +5V_RAIL | LEFT 5V | 24 AWG | ≤ 20 cm |
| 6 | GND_RAIL | LEFT GND | 24 AWG | Route next to #5 (twist) |
| 7 | +5V_RAIL | RIGHT 5V | 24 AWG | |
| 8 | GND_RAIL | RIGHT GND | 24 AWG | |
| 9 | LEFT 3V3 | MPU VCC | 28–30 AWG silicone | |
| 10 | LEFT GND | MPU GND | 28–30 AWG | Twist with #11 |
| 11 | LEFT IO13 | MPU SDA | 28–30 AWG | ≤ 15 cm |
| 12 | LEFT IO14 | MPU SCL | 28–30 AWG | Twist with a ground wire; ≤ 15 cm |

## 6. USB-C sink configuration
- A sink presents **5.1 kΩ ±1 % from each CC pin to GND (Rd)**. Both pins need one, because the plug can be inserted either way. Without Rd the source (the Mac) keeps VBUS off.
- Adafruit 4090 is described as a downstream/sink breakout with CC resistors; **⚠ VERIFY unplugged: CC1–GND and CC2–GND ≈ 5.1 kΩ each.** If not, solder 5.1 k resistors.
- Rd-only means **no Power Delivery negotiation**: you get 5 V and whatever current the source advertises (below). Never connect D+/D−/SBU.
- Never connect the programmer's 5 V and the rail at the same time. Flash on the bench with the board unplugged from the glasses harness (use a 2-pin JST/Dupont connector for 5V/GND so each board unplugs).

### How to find out how much current the Mac port offers (do this first)
The source advertises current by the **voltage on the CC pin** through its pull-up (Rp) against our 5.1 k:

| CC voltage (meter, CC→GND, cable plugged into the Mac, nothing else connected) | Source advertises |
|---|---|
| ≈ 0.25 – 0.66 V | USB default (500 mA USB 2 / 900 mA USB 3) — **not enough for worst case** |
| ≈ 0.70 – 1.23 V | 1.5 A |
| ≈ 1.23 – 2.04 V | 3 A |

(Voltage bands per the USB Type-C sink-detection ranges; use them as bands, not exact values.) Then confirm with the dummy-load test in `testing.md` (Test A). Advertised ≠ delivered through your cable, so measure.

## 7. Wire gauge and strain relief
- Rail trunk and cable-side power: 22–24 AWG stranded silicone. Per-board 5 V/GND legs: 24–26 AWG. I²C leads: 28–30 AWG silicone, twisted with ground. Solid-core hookup wire fatigues and breaks near a head-mounted frame — avoid.
- Voltage drop at 1 A: 26 AWG ≈ 0.13 Ω/m ⇒ 0.2 m out+back ≈ 50 mV. Fine. The USB cable itself is usually the biggest loss (see power-budget.md).
- Strain relief: tie the USB cable to the frame with a printed clamp or a zip-tie + hot-glue anchor 5–8 cm *before* the breakout, leaving a slack loop so a tug never reaches solder joints. Cover all joints with heat-shrink; no exposed 5 V pad on the skin side.

## 8. Bill of materials (missing parts; prices are rough guesses, not quotes)

| Qty | Part | Why |
|---|---|---|
| 1 | USB-C sink breakout (Adafruit 4090, or any board with two 5.1 k Rd) | Power input |
| 1 | USB-to-UART **CP2102 module with separate 3V3 and 5V pins**, or ESP32-CAM-MB (CH340) programming base ×1–2 | Flashing (§ firmware-setup) |
| 1 | PTC resettable fuse 1.5 A hold (e.g. 1812/leaded) | Overcurrent |
| 1 | TVS diode SMAJ5.0A (or SMAJ6.0A) | Hot-plug/ESD spikes |
| 1 | Slide switch rated ≥ 2 A (optional) | Power control |
| 1 | 100 µF low-ESR electrolytic/polymer, 10 V+ | Rail bulk |
| 2 | 330 µF low-ESR, 10 V+ | One at each ESP32-CAM 5V pin |
| 3 | 10 µF ceramic + 3 × 100 nF ceramic | Local decoupling |
| 1 | USB-C power meter (voltage/current display) | Verify current |
| 2 | 10 Ω 5 W resistors | Dummy load (0.5 A each at 5 V) |
| 1 | Short 3 A-rated USB-C cable (≤1 m, thick) | Low drop |
| – | 22–30 AWG silicone wire, heat-shrink, 2-pin JST or Dupont for power legs | |
| opt | Load-switch breakout with soft-start (TPS22918-class) | Only if Mac port trips on switch-on inrush |

## 9. Wearable safety and RF compliance
Three situations must be kept separate:

| Situation | Status |
|---|---|
| **Powered bench testing** (boards on the table, away from anyone) | Fine. |
| **Wireless streaming with the radios ≥ 20 cm from any person** (boards on a stand/desk, or a frame held away from the body) | Consistent with the 20 cm separation condition you reported for FCC ID 2ADMF-MD0456. I could not read the grant myself, so confirm the exact condition in the grant's RF-exposure exhibit. |
| **Head-worn demo** (antennas a few cm from the face/temple) | **Not covered by that authorization as far as we know.** A certified module used outside its stated separation distance is the integrator's responsibility: it needs an RF exposure assessment (SAR or an exemption analysis) or a different radio arrangement (e.g., remote antenna ≥20 cm away from the head, a pre-certified wearable-class radio, or wired video). **Do not mount active transmitters beside a judge's or anyone's head until this is resolved.** |

Practical consequences for the project:
- Build and test everything on a stand. For a judging demo use a **table-mounted fixture or a mannequin/head model with the transmitters ≥ 20 cm from the nearest person** until compliance is settled; or demo the wired-power-only frame with the radios on a boom ≥ 20 cm away.
- Lowering `WIFI_TX_POWER` in `network_config.h` reduces output but is not a substitute for an assessment, and may hurt streaming reliability.
- Electrical: insulate every exposed 5 V node (heat-shrink/Kapton); no bare pads facing skin; trim solder-joint spikes and sand/cap sharp wire ends; PTC fuse on the rail; keep the glasses frame ≤ 40 °C; use strain relief on the single cable; keep wire-to-skin contact away from the 5 V rail.
- **No homemade near-eye IR illumination.** IR near the eye has exposure limits that cannot be verified with hobby tools. Stock OV2640 modules also normally carry an IR-cut filter, so pupil contrast under room light may be poor — evaluate that with the camera on a stand first, and use only commercially certified, eye-safe IR illuminators if you add any later.
- Streams are unencrypted and unauthenticated HTTP/UDP of images of a person's eyes. Run them only on a private network you control (see firmware-setup.md).

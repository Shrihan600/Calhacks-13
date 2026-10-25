# Power budget (planning calculation — **not measured**)

## Inputs and their honesty level
| Item | Planning value | Source / caveat |
|---|---|---|
| ESP32-CAM, Wi-Fi up + OV2640 streaming QVGA–VGA, flash LED off | 200–260 mA avg @5 V | Widely reported community range; **you must measure** |
| Same, short Wi-Fi TX bursts | up to ~350 mA | Reported; not measured |
| Boot/camera init/Wi-Fi association | up to ~500 mA for tens–hundreds of ms | Reported; brownouts here are the classic failure |
| MPU-6050 + breakout | ~4–6 mA | Datasheet ~3.9 mA typ. + LED |
| Flash LED (GPIO4) | not used | Firmware never drives it |

Conservative design numbers: **300 mA sustained per camera, 500 mA transient per camera.**

| Scenario | Calculation | Total @5 V |
|---|---|---|
| Typical streaming | 2 × 230 + 6 | ≈ 0.47 A (2.3 W) |
| Conservative sustained | 2 × 300 + 6 | ≈ 0.61 A (3.1 W) |
| Worst transient (both boot at once) | 2 × 500 + 6 | ≈ 1.0 A (5 W) |

## Compare with what the Mac port may give
- USB default power: 500 mA (USB 2 level) or 900 mA (USB 3 level) → **cannot be assumed to cover the 1.0 A transient or even 0.61 A with margin** (0.9 A is marginal; 0.5 A fails).
- 1.5 A advertised → OK (≈ 33 % margin over the worst transient).
- 3 A advertised → comfortable.
- A MacBook USB-C port is a dual-role port. With Rd-only on our side it acts as a **source**, so there is no backfeed path as long as the rail has no second source. Do not add a battery or second USB input without an ideal-diode/load-switch design.
- Ports can also cut VBUS when the Mac sleeps/lid closes, and may current-limit with a hard trip rather than sag.

**Verdict:** a single-cable design is *plausible* but **unproven until Test A shows (a) CC advertises ≥1.5 A and (b) the rail holds ≥4.75 V at 1 A with your cable.** If not, the fallback in §4 applies.

## Voltage-drop budget
The ESP32-CAM's onboard 3.3 V LDO needs headroom (typ. AMS1117 dropout ~1 V at 0.5 A) → keep the **board 5 V pin ≥ 4.75 V** under load, absolute floor ≈ 4.5 V.

V_board = 5.0 V − I × (R_cable + R_connectors + R_fuse + R_wiring)
- Cheap 28 AWG 1 m USB cable: ~0.4–0.5 Ω round trip → at 1 A: ~0.45 V drop ⇒ ~4.5 V. **Marginal.**
- 24 AWG 0.5 m cable: ~0.1 Ω → ~0.1 V drop.
- PTC 1.5 A hold: ~0.05–0.1 Ω cold → ≤ 0.1 V at 1 A.
- Wiring 0.2 m of 24 AWG: ~0.03 Ω.
Choose a short, thick cable; measure at the board pins during Test A/G.

## Heat
LDO loss per board ≈ (5 − 3.3) V × ~0.25 A ≈ 0.4 W, plus the ESP32 itself (~0.5 W under Wi-Fi). Two boards ≈ 2 W in a small plastic frame near the face. Target **exterior surfaces ≤ 40 °C** in a 25 °C room after 30 minutes; measure with an IR thermometer/thermocouple. Standoffs, vented frame, no covering of the module shields.

## Capacitors, protection (what each is for)
| Part | Where | Purpose |
|---|---|---|
| 100 µF low ESR + 10 µF ceramic | At the breakout output | Absorb cable inductance transients |
| 330 µF + 10 µF + 100 nF | At *each* ESP32-CAM 5V/GND, <3 cm | Supply the Wi-Fi TX/boot spikes locally (primary anti-brownout measure) |
| SMAJ5.0A TVS | Across rail | Clamp hot-plug/ESD spikes |
| PTC 1.5 A hold | Series after switch | Limits a short; hold current falls as it warms, so check for nuisance trips in a warm frame |
| Switch | Between breakout and fuse | Connect the cable first, then switch on |

**Inrush:** total rail capacitance ≈ 100 + 330 + 330 + small ≈ 0.8 mF. USB specifies ~10 µF for hot-attach; connecting the cable with the switch OFF then closing the switch is the usual practice, but a source may still trip. If the Mac drops VBUS on switch-on, reduce bulk caps to 100–150 µF each or add a soft-start load switch. Observe with the meter before concluding anything.

Do **not** disable the ESP32 brownout detector to hide resets; the firmware reports the reset reason in `/status` and the UDP status packet instead.

## 4. If direct Mac power is insufficient (safe bench alternative)
1. Power the glasses harness from a **5 V ≥ 2 A wall adapter or powered USB hub** through the same USB-C sink breakout (still Rd-only; no data).
2. The Mac then needs **no cable to the glasses at all** (Wi-Fi carries video + IMU). That removes any shared-ground/backfeed question.
3. Never have the Mac cable and the adapter both plugged into the rail.
4. Keep this as the demo-day contingency even if single-cable works on the bench.

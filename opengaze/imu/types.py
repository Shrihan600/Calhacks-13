"""IMU data contract and the wire formats IMU devices send (text lines and JSON)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

GRAVITY = 9.80665  # m/s²


@dataclass(frozen=True, slots=True, eq=False)
class ImuSample:
    """One reading in the IMU's own axes, canonical units."""

    host_receive_ns: int  # time.monotonic_ns() when the reading reached the host
    device_time_s: float | None  # device clock, if the device sends one (for integration dt)
    gyro: np.ndarray  # rad/s, float64[3]
    accel: np.ndarray  # m/s², float64[3]; at rest it reads +g pointing *up*
    source_kind: Literal["live", "synthetic"]
    # Synthetic sources only: true head angles (yaw, pitch, roll) in degrees.
    ground_truth: tuple[float, float, float] | None = None
    # Optional telemetry fields (JSON protocol). ``capture_ns`` is the device timestamp
    # mapped onto the host clock; None when the device sends no timestamp.
    sequence: int | None = None
    device_id: str | None = None
    temperature_c: float | None = None
    capture_ns: int | None = None

    @property
    def timestamp_ns(self) -> int:
        """Best available time: mapped capture time if known, else host receive time."""
        return self.capture_ns if self.capture_ns is not None else self.host_receive_ns


GyroUnits = Literal["deg_s", "rad_s"]
AccelUnits = Literal["g", "m_s2"]


def parse_line(
    text: str,
    host_receive_ns: int,
    *,
    gyro_units: GyroUnits = "deg_s",
    accel_units: AccelUnits = "g",
) -> ImuSample | None:
    """Parse one protocol line; None for blank lines and ``#`` comments.

    Protocol (one reading per line, comma- or whitespace-separated numbers)::

        gx,gy,gz,ax,ay,az            # 6 fields
        t_us,gx,gy,gz,ax,ay,az       # 7 fields: device timestamp in microseconds first

    This is what a few lines of Arduino/ESP32 code printing an MPU-6050, ICM-20948, BNO08x
    etc. produce. Units are set per device (``gyro_units``, ``accel_units``).
    Raises ValueError for anything else, so a misconfigured device is reported, not ignored.
    """
    text = text.strip()
    if not text or text.startswith("#"):
        return None
    fields = text.replace(",", " ").split()
    if len(fields) not in (6, 7):
        raise ValueError(f"expected 6 or 7 numbers, got {len(fields)}: {text[:60]!r}")
    values = [float(f) for f in fields]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite value in {text[:60]!r}")
    device_time_s = None
    if len(values) == 7:
        device_time_s = values.pop(0) * 1e-6
    gyro = np.array(values[:3])
    accel = np.array(values[3:])
    if gyro_units == "deg_s":
        gyro = np.radians(gyro)
    if accel_units == "g":
        accel = accel * GRAVITY
    return ImuSample(host_receive_ns, device_time_s, gyro, accel, "live")


# ---------------------------------------------------------------------------
# JSON telemetry (OpenGaze ESP32 firmware)
# ---------------------------------------------------------------------------

_JSON_FIELDS = ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z")


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """One decoded JSON telemetry message, before the host clock mapping."""

    sample: ImuSample | None  # None when the device flagged the reading ``valid: false``
    sequence: int | None
    device_time_us: int | None
    device_id: str | None
    emulated: bool


def parse_json_record(
    obj: Mapping[str, Any],
    host_receive_ns: int,
    *,
    gyro_units: GyroUnits = "rad_s",
    accel_units: AccelUnits = "m_s2",
) -> TelemetryRecord:
    """Decode one telemetry object (the format in ``docs/hardware-interface.md``)::

        {"device_id": "right_esp32", "sensor": "mpu6050", "sequence": 1024,
         "timestamp_us": 123456789, "accel_x": 0.02, "accel_y": 0.01, "accel_z": 9.79,
         "gyro_x": 0.12, "gyro_y": -0.08, "gyro_z": 0.03, "temp_c": 31.2, "valid": true}

    Default units are the protocol's: m/s² and rad/s. Raises ValueError on a malformed
    message, so a firmware mismatch is reported rather than read as zeros.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    sequence = _opt_int(obj, "sequence")
    device_time_us = _opt_int(obj, "timestamp_us")
    device_id = obj.get("device_id")
    device_id = str(device_id) if device_id is not None else None
    emulated = bool(obj.get("emulated", False))
    if obj.get("valid", True) is False:
        return TelemetryRecord(None, sequence, device_time_us, device_id, emulated)
    missing = [k for k in _JSON_FIELDS if k not in obj]
    if missing:
        raise ValueError(f"telemetry missing {', '.join(missing)}")
    try:
        values = [float(obj[k]) for k in _JSON_FIELDS]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric telemetry value: {exc}") from None
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite telemetry value")
    accel, gyro = np.array(values[:3]), np.array(values[3:])
    if gyro_units == "deg_s":
        gyro = np.radians(gyro)
    if accel_units == "g":
        accel = accel * GRAVITY
    temp = obj.get("temp_c", obj.get("temperature_c"))
    sample = ImuSample(
        host_receive_ns=host_receive_ns,
        device_time_s=device_time_us * 1e-6 if device_time_us is not None else None,
        gyro=gyro,
        accel=accel,
        source_kind="live",
        sequence=sequence,
        device_id=device_id,
        temperature_c=float(temp) if isinstance(temp, int | float) else None,
    )
    return TelemetryRecord(sample, sequence, device_time_us, device_id, emulated)


def iter_json_objects(text: str) -> Iterable[Any]:
    """A datagram holds one JSON object, a JSON array of them, or one object per line."""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def _opt_int(obj: Mapping[str, Any], key: str) -> int | None:
    value = obj.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return int(value)


def check_units(samples: Iterable[ImuSample]) -> str | None:
    """A warning if the readings don't look like the configured units, else None.

    The accelerometer reads 1 g at rest, so its median magnitude shows whether g and m/s²
    were mixed up. A gyro reading above the MPU-6050's ±2000 °/s range (35 rad/s) means
    degrees were sent where radians were configured.
    """
    readings = list(samples)
    if len(readings) < 20:
        return None
    mags = sorted(float(np.linalg.norm(s.accel)) for s in readings)
    median = mags[len(mags) // 2] / GRAVITY  # in g, after configured unit conversion
    if 0.07 < median < 0.14:
        return (f"accelerometer reads {median:.2f} g at rest: the device sends g but m_s2 "
                "is configured (set accel_units: g)")
    if 7.0 < median < 13.0:
        return (f"accelerometer reads {median:.1f} g at rest: the device sends m/s² but g "
                "is configured (set accel_units: m_s2)")
    if not 0.6 < median < 1.6:
        return f"accelerometer magnitude {median:.2f} g is far from 1 g: check units/wiring"
    peak = max(float(np.abs(s.gyro).max()) for s in readings)
    if peak > 35.0:
        return (f"gyro peak {peak:.0f} rad/s exceeds the sensor range: the device probably "
                "sends °/s (set gyro_units: deg_s)")
    return None

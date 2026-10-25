"""IMU data contract and the line protocol IMU devices send."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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

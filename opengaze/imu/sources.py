"""IMU sources: where readings come from. Selected by ``imu.source_type`` in the profile.

Sources are blocking (``read`` waits up to a timeout) because they run on ``ImuRuntime``'s
own thread. Each validates its ``options`` with its own pydantic model, like camera adapters.

| source_type | transport |
|---|---|
| ``synthetic`` | simulated head motion with ground truth, gyro noise and bias (no hardware) |
| ``serial``    | USB serial (Arduino / ESP32 / Pi Pico over USB), needs ``pyserial`` |
| ``udp``       | UDP datagrams (ESP32 / Pi over Wi-Fi), stdlib only |
"""

from __future__ import annotations

import math
import socket
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from opengaze.imu.types import GRAVITY, AccelUnits, GyroUnits, ImuSample, parse_line
from opengaze.vision.head_pose import rotation_from_angles

if TYPE_CHECKING:
    from opengaze.config.schema import ImuConfig


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImuSource(ABC):
    source_type: ClassVar[str]
    source_kind: ClassVar[Literal["live", "synthetic"]] = "live"
    Options: ClassVar[type[BaseModel]] = _Options

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = self.Options.model_validate(options or {})

    @abstractmethod
    def open(self) -> None:
        """Open the device / socket. Raises OSError (or a subclass) on failure."""

    @abstractmethod
    def read(self, timeout_s: float) -> ImuSample | None:
        """Next reading, or None if none arrived within ``timeout_s``."""

    @abstractmethod
    def close(self) -> None:
        """Release the device. Idempotent."""


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


class SyntheticImuOptions(_Options):
    rate_hz: float = Field(default=200.0, gt=0.0, le=2000.0)
    # sway: smooth, never-stopping head motion; sway_pause: sway with still pauses (closer
    # to real use, lets the gyro bias be learned); static: head still.
    motion: Literal["sway", "sway_pause", "static"] = "sway"
    amplitude_deg: tuple[float, float, float] = (15.0, 8.0, 5.0)  # yaw, pitch, roll
    period_s: float = Field(default=8.0, gt=0.0)
    gyro_noise_dps: float = Field(default=0.3, ge=0.0)
    gyro_bias_dps: tuple[float, float, float] = (0.6, -0.4, 0.3)
    accel_noise_g: float = Field(default=0.01, ge=0.0)
    seed: int = 0
    # false = generate as fast as possible on a virtual clock (tests).
    realtime: bool = True


class SyntheticImuSource(ImuSource):
    """Readings from a known head trajectory, in head axes (``axes: [+x, +y, +z]``)."""

    source_type: ClassVar[str] = "synthetic"
    source_kind: ClassVar[Literal["live", "synthetic"]] = "synthetic"
    Options: ClassVar[type[BaseModel]] = SyntheticImuOptions

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self._rng = np.random.default_rng(self.options.seed)
        self._index = 0
        self._start_ns = 0

    def angles(self, t: float) -> tuple[float, float, float]:
        o: SyntheticImuOptions = self.options
        if o.motion == "static":
            return 0.0, 0.0, 0.0
        if o.motion == "sway_pause":
            # Move for one period, then hold still for half a period, smoothly.
            cycle = 1.5 * o.period_s
            phase = t % cycle
            t = (t // cycle) * o.period_s + min(phase, o.period_s)
        w = 2 * math.pi / o.period_s
        a = o.amplitude_deg
        return (
            a[0] * math.sin(w * t),
            a[1] * math.sin(1.3 * w * t + 1.0),
            a[2] * math.sin(0.7 * w * t + 2.0),
        )

    def open(self) -> None:
        self._index = 0
        self._start_ns = time.monotonic_ns()

    def read(self, timeout_s: float) -> ImuSample | None:
        o: SyntheticImuOptions = self.options
        t = self._index / o.rate_hz
        self._index += 1
        if o.realtime:
            due_ns = self._start_ns + int(t * 1e9)
            wait = (due_ns - time.monotonic_ns()) / 1e9
            if wait > timeout_s:
                time.sleep(timeout_s)
                self._index -= 1
                return None
            if wait > 0:
                time.sleep(wait)
            host_ns = time.monotonic_ns()
        else:
            host_ns = self._start_ns + int(t * 1e9)

        r = rotation_from_angles(*self.angles(t))
        dt = 1e-4
        r_next = rotation_from_angles(*self.angles(t + dt))
        # Body angular velocity: R^T dR/dt is skew-symmetric with ω in it.
        skew = r.T @ (r_next - r) / dt
        omega = np.array([skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0],
                          skew[1, 0] - skew[0, 1]]) / 2.0
        gyro = (
            omega
            + np.radians(o.gyro_bias_dps)
            + self._rng.normal(0.0, math.radians(o.gyro_noise_dps), 3)
        )
        up_world = np.array([0.0, -1.0, 0.0])  # world y points down
        accel = (r.T @ up_world + self._rng.normal(0.0, o.accel_noise_g, 3)) * GRAVITY
        return ImuSample(host_ns, t, gyro, accel, "synthetic", self.angles(t))

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Serial (USB)
# ---------------------------------------------------------------------------


class _LineOptions(_Options):
    gyro_units: GyroUnits = "deg_s"
    accel_units: AccelUnits = "g"


class SerialImuOptions(_LineOptions):
    port: str  # e.g. /dev/cu.usbmodem1101 (macOS) — `ls /dev/cu.*` lists them
    baud: int = Field(default=115200, gt=0)


class SerialImuSource(ImuSource):
    source_type: ClassVar[str] = "serial"
    Options: ClassVar[type[BaseModel]] = SerialImuOptions

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self._serial = None

    def open(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise OSError("pyserial is required for serial IMUs: pip install -e '.[imu]'") from exc
        o: SerialImuOptions = self.options
        self._serial = serial.Serial(o.port, o.baud, timeout=0.1)

    def read(self, timeout_s: float) -> ImuSample | None:
        assert self._serial is not None
        o: SerialImuOptions = self.options
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            sample = parse_line(
                raw.decode("ascii", errors="replace"),
                time.monotonic_ns(),
                gyro_units=o.gyro_units,
                accel_units=o.accel_units,
            )
            if sample is not None:
                return sample
        return None

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


# ---------------------------------------------------------------------------
# UDP (Wi-Fi)
# ---------------------------------------------------------------------------


class UdpImuOptions(_LineOptions):
    bind_host: str = "0.0.0.0"
    port: int = Field(default=5005, ge=0, lt=65536)  # 0 = any free port


class UdpImuSource(ImuSource):
    """Each datagram holds one or more protocol lines."""

    source_type: ClassVar[str] = "udp"
    Options: ClassVar[type[BaseModel]] = UdpImuOptions

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self._sock: socket.socket | None = None
        self._pending: list[ImuSample] = []

    @property
    def bound_port(self) -> int | None:
        return self._sock.getsockname()[1] if self._sock is not None else None

    def open(self) -> None:
        o: UdpImuOptions = self.options
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((o.bind_host, o.port))
        self._sock = sock

    def read(self, timeout_s: float) -> ImuSample | None:
        assert self._sock is not None
        if self._pending:
            return self._pending.pop(0)
        o: UdpImuOptions = self.options
        self._sock.settimeout(timeout_s)
        try:
            data, _ = self._sock.recvfrom(4096)
        except TimeoutError:
            return None
        now = time.monotonic_ns()
        for line in data.decode("ascii", errors="replace").splitlines():
            sample = parse_line(line, now, gyro_units=o.gyro_units, accel_units=o.accel_units)
            if sample is not None:
                self._pending.append(sample)
        return self._pending.pop(0) if self._pending else None

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


IMU_SOURCES: dict[str, type[ImuSource]] = {
    cls.source_type: cls for cls in (SyntheticImuSource, SerialImuSource, UdpImuSource)
}


def make_imu_source(config: ImuConfig) -> ImuSource:
    try:
        cls = IMU_SOURCES[config.source_type]
    except KeyError:
        raise ValueError(
            f"unknown IMU source {config.source_type!r}; available: {sorted(IMU_SOURCES)}"
        ) from None
    return cls(config.options)

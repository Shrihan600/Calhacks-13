"""IMU sources: where readings come from. Selected by ``imu.source_type`` in the profile.

Sources are blocking (``read`` waits up to a timeout) because they run on ``ImuRuntime``'s
own thread. Each validates its ``options`` with its own pydantic model, like camera adapters.

| source_type | transport |
|---|---|
| ``synthetic`` | simulated head motion with ground truth, gyro noise and bias (no hardware) |
| ``serial``    | USB serial (Arduino / ESP32 / Pi Pico over USB), needs ``pyserial`` |
| ``udp``       | UDP datagrams (ESP32 / Pi over Wi-Fi): text lines or JSON telemetry, stdlib only |
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import socket
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from opengaze.cameras.timing import DeviceClockSync
from opengaze.imu.types import (
    GRAVITY,
    AccelUnits,
    GyroUnits,
    ImuSample,
    iter_json_objects,
    parse_json_record,
    parse_line,
)
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

    def stats(self) -> dict[str, Any]:
        """Transport counters for diagnostics (drops, bad packets, …). Empty if none."""
        return {}


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


class UdpImuOptions(_Options):
    bind_host: str = "0.0.0.0"
    port: int = Field(default=5005, ge=0, lt=65536)  # 0 = any free port
    # text: protocol lines (``parse_line``); json: telemetry objects (``parse_json_record``);
    # auto: decide per datagram (JSON starts with ``{`` or ``[``).
    format: Literal["auto", "text", "json"] = "auto"
    # null = the format's own default: text deg_s / g, json rad_s / m_s2.
    gyro_units: GyroUnits | None = None
    accel_units: AccelUnits | None = None
    # Accept only telemetry with this device_id (others are counted and dropped).
    expected_device_id: str | None = None
    # Subscribe handshake: with device_host set, a small JSON datagram is sent to
    # device_host:device_port every subscribe_interval_s, and the device streams to whoever
    # subscribed. So the Mac's IP never has to be compiled into the firmware.
    device_host: str | None = None
    device_port: int = Field(default=5006, gt=0, lt=65536)
    subscribe_interval_s: float = Field(default=1.0, gt=0.0)
    # Map device timestamps to the host clock (min-offset sync). false = host receive time.
    device_clock: bool = True


class UdpImuSource(ImuSource):
    """Each datagram holds one or more readings (text lines, or JSON objects)."""

    source_type: ClassVar[str] = "udp"
    Options: ClassVar[type[BaseModel]] = UdpImuOptions

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self._sock: socket.socket | None = None
        self._pending: collections.deque[ImuSample] = collections.deque()
        self._clock = DeviceClockSync()
        self._last_sequence: int | None = None
        self._next_subscribe = 0.0
        self._counters = collections.Counter[str]()
        self.last_error: str | None = None
        self.last_sender: tuple[str, int] | None = None
        self.emulated = False

    @property
    def bound_port(self) -> int | None:
        return self._sock.getsockname()[1] if self._sock is not None else None

    def open(self) -> None:
        o: UdpImuOptions = self.options
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((o.bind_host, o.port))
        self._sock = sock
        self._last_sequence = None
        self._next_subscribe = 0.0

    def _units(self, is_json: bool) -> tuple[GyroUnits, AccelUnits]:
        o: UdpImuOptions = self.options
        gyro = o.gyro_units or ("rad_s" if is_json else "deg_s")
        accel = o.accel_units or ("m_s2" if is_json else "g")
        return gyro, accel

    def _subscribe(self) -> None:
        o: UdpImuOptions = self.options
        if o.device_host is None or self._sock is None:
            return
        now = time.monotonic()
        if now < self._next_subscribe:
            return
        self._next_subscribe = now + o.subscribe_interval_s
        message = json.dumps({"type": "subscribe", "port": self.bound_port}).encode()
        try:
            self._sock.sendto(message, (o.device_host, o.device_port))
            self._counters["subscribes_sent"] += 1
        except OSError as exc:  # unresolvable name, no route: keep listening anyway
            from opengaze.cameras.mjpeg import describe_network_error

            self._counters["subscribe_errors"] += 1
            reason = describe_network_error(exc)
            self.last_error = f"subscribe to {o.device_host}:{o.device_port} failed: {reason}"

    def read(self, timeout_s: float) -> ImuSample | None:
        assert self._sock is not None
        deadline = time.monotonic() + timeout_s
        while not self._pending:
            self._subscribe()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            o: UdpImuOptions = self.options
            wait = remaining if o.device_host is None else min(remaining, o.subscribe_interval_s)
            self._sock.settimeout(max(wait, 1e-3))
            try:
                data, sender = self._sock.recvfrom(65535)
            except TimeoutError:
                continue
            self._handle_datagram(data, sender, time.monotonic_ns())
        return self._pending.popleft()

    def _handle_datagram(self, data: bytes, sender: tuple[str, int], now: int) -> None:
        o: UdpImuOptions = self.options
        self._counters["packets"] += 1
        self.last_sender = sender
        text = data.decode("utf-8", errors="replace")
        stripped = text.lstrip()
        is_json = o.format == "json" or (o.format == "auto" and stripped[:1] in ("{", "["))
        gyro_units, accel_units = self._units(is_json)
        try:
            if is_json:
                self._handle_json(text, now, gyro_units, accel_units)
            else:
                for line in text.splitlines():
                    sample = parse_line(
                        line, now, gyro_units=gyro_units, accel_units=accel_units
                    )
                    if sample is not None:
                        self._pending.append(sample)
                        self._counters["samples"] += 1
        except (ValueError, TypeError) as exc:
            # One bad packet must not end the stream: count it and keep listening.
            self._counters["parse_errors"] += 1
            self.last_error = f"bad packet from {sender[0]}: {exc}"

    def _handle_json(
        self, text: str, now: int, gyro_units: GyroUnits, accel_units: AccelUnits
    ) -> None:
        o: UdpImuOptions = self.options
        batch: list[tuple[ImuSample, int | None]] = []
        for obj in iter_json_objects(text):
            if isinstance(obj, dict) and obj.get("type") not in (None, "imu"):
                self._counters["other_messages"] += 1  # e.g. a status message
                continue
            record = parse_json_record(
                obj, now, gyro_units=gyro_units, accel_units=accel_units
            )
            if o.expected_device_id is not None and record.device_id != o.expected_device_id:
                self._counters["wrong_device"] += 1
                continue
            self.emulated = self.emulated or record.emulated
            if record.sequence is not None:
                last = self._last_sequence
                if last is not None:
                    if record.sequence <= last:
                        if last - record.sequence > 1000:  # the device restarted
                            self._counters["device_restarts"] += 1
                        else:
                            self._counters["out_of_order"] += 1
                            continue
                    elif record.sequence > last + 1:
                        self._counters["dropped"] += record.sequence - last - 1
                self._last_sequence = record.sequence
            if record.sample is None:
                self._counters["invalid"] += 1
                continue
            if record.device_time_us is not None and o.device_clock:
                self._clock.update(record.device_time_us, now)
                batch.append((record.sample, record.device_time_us))
            else:
                batch.append((record.sample, None))
        # Map only after the whole datagram updated the clock: its readings share one
        # receive time, and the newest of them bounds the offset best.
        for sample, device_us in batch:
            capture_ns = self._clock.map(device_us) if device_us is not None else None
            if capture_ns is not None:
                sample = _with_capture(sample, capture_ns)
            self._pending.append(sample)
            self._counters["samples"] += 1

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._counters)
        if self._clock.excess_delay_ns is not None:
            out["clock_excess_delay_ms"] = self._clock.excess_delay_ns / 1e6
        if self._clock.resets:
            out["clock_resets"] = self._clock.resets
        if self.last_sender is not None:
            out["sender"] = f"{self.last_sender[0]}:{self.last_sender[1]}"
        if self.emulated:
            out["emulated"] = True
        return out

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


def _with_capture(sample: ImuSample, capture_ns: int) -> ImuSample:
    return dataclasses.replace(sample, capture_ns=capture_ns)


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

"""ImuRuntime: reads an IMU on its own thread and keeps a short history of orientations.

The gaze loop runs at ~30 fps while IMUs report at 100–1000 Hz, so the runtime integrates
every reading itself and the loop just asks for the orientation at a frame's timestamp.
A failed or lost device is reopened with backoff; it never blocks the UI.
"""

from __future__ import annotations

import bisect
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from opengaze.imu.fusion import ImuOrientation
from opengaze.imu.sources import ImuSource, make_imu_source
from opengaze.imu.types import ImuSample, check_units

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImuHealth:
    state: str  # "connecting" | "streaming" | "stalled" | "stopped"
    samples: int
    rate_hz: float | None
    last_error: str | None
    last_sample_age_ms: float | None = None
    # Transport counters from the source (packets, dropped, invalid, parse_errors, …).
    transport: dict[str, Any] = field(default_factory=dict)
    units_warning: str | None = None


class ImuRuntime:
    def __init__(
        self,
        source: ImuSource,
        axes: tuple[str, str, str] = ("+x", "+y", "+z"),
        *,
        kp: float = 1.0,
        ki: float = 0.02,
        stale_timeout_s: float = 0.5,
        history: int = 1024,
    ) -> None:
        self.source = source
        self.orientation = ImuOrientation(axes, kp, ki)
        self.stale_timeout_ns = int(stale_timeout_s * 1e9)
        self._times: collections.deque[int] = collections.deque(maxlen=history)
        self._poses: collections.deque[np.ndarray] = collections.deque(maxlen=history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._state = "stopped"
        self._last_error: str | None = None
        self.last_sample: ImuSample | None = None
        self._recent: collections.deque[ImuSample] = collections.deque(maxlen=512)

    @classmethod
    def from_config(cls, config) -> ImuRuntime:  # config: ImuConfig
        return cls(
            make_imu_source(config),
            tuple(config.axes),
            kp=config.filter_kp,
            ki=config.filter_ki,
            stale_timeout_s=config.stale_timeout_s,
        )

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="imu", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._state = "stopped"

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            self._state = "connecting"
            try:
                self.source.open()
            except Exception as exc:
                self._last_error = f"open failed: {exc}"
                log.warning("IMU %s", self._last_error)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            self._state = "streaming"
            try:
                while not self._stop.is_set():
                    sample = self.source.read(0.25)
                    if sample is None:
                        continue
                    self.push(sample)
            except Exception as exc:
                self._last_error = f"read failed: {exc}"
                log.warning("IMU %s", self._last_error)
            finally:
                self.source.close()

    def push(self, sample: ImuSample) -> None:
        """Integrate one reading (called by the thread; tests call it directly)."""
        rot = self.orientation.update(sample)
        self.last_sample = sample
        self._samples += 1
        with self._lock:
            self._recent.append(sample)
            if rot is None:
                return
            # Device times mapped to the host clock can step back slightly when the clock
            # sync finds a smaller offset; keep the history sorted for bisect.
            t = sample.timestamp_ns
            if self._times and t < self._times[-1]:
                t = self._times[-1]
            self._times.append(t)
            self._poses.append(rot)

    def recent(self, n: int = 512) -> list[ImuSample]:
        """The newest ``n`` raw readings, oldest first (for plots and unit checks)."""
        with self._lock:
            items = list(self._recent)
        return items[-n:]

    # ------------------------------------------------------------------

    def pose_at(self, t_ns: int | None = None) -> np.ndarray | None:
        """Orientation (head → earth) nearest to ``t_ns`` (default: newest), or None if stale."""
        with self._lock:
            if not self._times:
                return None
            if t_ns is None:
                i = len(self._times) - 1
            else:
                times = list(self._times)
                i = bisect.bisect_left(times, t_ns)
                if i == len(times) or (i > 0 and t_ns - times[i - 1] < times[i] - t_ns):
                    i -= 1
            newest = self._times[-1]
            pose = self._poses[i]
        now = time.monotonic_ns() if t_ns is None else max(t_ns, newest)
        if now - newest > self.stale_timeout_ns:
            return None
        return pose

    def health(self) -> ImuHealth:
        with self._lock:
            recent = list(self._recent)[-200:]
        # Rate from the raw device clock when the device sends one: host receive times bunch
        # up whenever this thread is briefly starved and reads several queued datagrams at
        # once, and mapped times step while the clock sync settles.
        # Age and staleness use host receive times: they measure the link.
        times = [s.host_receive_ns for s in recent]
        device = [s.device_time_s for s in recent]
        rate = None
        if len(recent) > 1:
            if all(t is not None for t in device) and device[-1] > device[0]:  # type: ignore[operator]
                rate = (len(device) - 1) / (device[-1] - device[0])  # type: ignore[operator]
            elif times[-1] > times[0]:
                rate = (len(times) - 1) / ((times[-1] - times[0]) / 1e9)
        now = time.monotonic_ns()
        age_ms = (now - times[-1]) / 1e6 if times else None
        state = self._state
        if state == "streaming" and (not times or now - times[-1] > self.stale_timeout_ns):
            state = "stalled"
        transport = self.source.stats()
        error = self._last_error or getattr(self.source, "last_error", None)
        return ImuHealth(
            state,
            self._samples,
            rate,
            error,
            last_sample_age_ms=age_ms,
            transport=transport,
            units_warning=check_units(recent) if recent and recent[-1].source_kind == "live"
            else None,
        )


def start_profile_imu(profile) -> ImuRuntime | None:  # profile: HardwareProfile
    """Start the profile's IMU if it has an enabled one (diagnostics show it in any mode)."""
    imu_cfg = profile.imu
    if imu_cfg is None or not imu_cfg.enabled:
        return None
    runtime = ImuRuntime.from_config(imu_cfg)
    runtime.start()
    return runtime

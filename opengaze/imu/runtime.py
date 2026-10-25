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
from dataclasses import dataclass

import numpy as np

from opengaze.imu.fusion import ImuOrientation
from opengaze.imu.sources import ImuSource, make_imu_source

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImuHealth:
    state: str  # "connecting" | "streaming" | "stalled" | "stopped"
    samples: int
    rate_hz: float | None
    last_error: str | None


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
        self.last_sample = None

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

    def push(self, sample) -> None:
        """Integrate one reading (called by the thread; tests call it directly)."""
        rot = self.orientation.update(sample)
        self.last_sample = sample
        self._samples += 1
        if rot is None:
            return
        with self._lock:
            self._times.append(sample.host_receive_ns)
            self._poses.append(rot)

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
            times = list(self._times)[-200:]
        rate = None
        if len(times) > 1 and times[-1] > times[0]:
            rate = (len(times) - 1) / ((times[-1] - times[0]) / 1e9)
        state = self._state
        if state == "streaming" and (
            not times or time.monotonic_ns() - times[-1] > self.stale_timeout_ns
        ):
            state = "stalled"
        return ImuHealth(state, self._samples, rate, self._last_error)

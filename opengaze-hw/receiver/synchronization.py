"""Approximate time alignment between independent device clocks and the Mac.

What is and is NOT achievable (read before trusting any number):

* Every ESP32 stamps its data with esp_timer_get_time() (microseconds since ITS boot).
  The LEFT camera frames (X-Timestamp-Us) and LEFT IMU packets use the SAME clock, so
  IMU<->left-frame alignment needs no cross-device sync. Residual error = (a) camera
  frame timestamp is "frame done" time at the driver, not mid-exposure (offset of the order of
  one frame period or less, not measured), (b) IMU timestamp precedes the I2C read by <1 ms
  and the MPU low-pass filter adds group delay (~5 ms at the 44 Hz setting).
* LEFT vs RIGHT clocks are unrelated. We estimate each device's offset to the Mac clock with a
  sliding-window minimum filter on (host_arrival - device_timestamp), using UDP packets (low,
  symmetric-ish delay). This assumes the minimum network delay is constant; real error is
  roughly the jitter floor of the Wi-Fi link: expect a few ms best case, tens of ms when the
  channel is busy. It has NOT been measured on this hardware. Plan on +/- one frame period
  (~33-40 ms) until you measure it with the LED-flash test in docs/testing.md.
* Crystal drift (tens of ppm) is negligible over the 10 s window (<1 ms).
* "Relative delay" reported for video = how much later than the best recent frame a frame
  arrived. It reveals buffering/jitter but is NOT absolute glass-to-glass latency.
"""
from __future__ import annotations
import bisect
import time
from collections import deque
from typing import Deque, Optional, Tuple


class ClockTracker:
    """Maps a device microsecond clock onto the Mac's time.monotonic() seconds."""

    def __init__(self, window_s: float = 10.0):
        self.window_s = window_s
        self._obs: Deque[Tuple[float, float]] = deque()  # (host_s, offset_s)
        self._min = None  # type: Optional[float]

    def update(self, dev_us: int, host_s: float) -> None:
        off = host_s - dev_us * 1e-6
        self._obs.append((host_s, off))
        cutoff = host_s - self.window_s
        while self._obs and self._obs[0][0] < cutoff:
            self._obs.popleft()
        self._min = min(o for _, o in self._obs)

    @property
    def ready(self) -> bool:
        return self._min is not None and len(self._obs) >= 5

    def to_host(self, dev_us: int) -> Optional[float]:
        """Estimated host time (monotonic seconds) at which the device event happened."""
        return None if self._min is None else dev_us * 1e-6 + self._min

    def rel_delay_ms(self, dev_us: int, host_s: float) -> Optional[float]:
        """Arrival delay above the recent minimum (ms). >=0 by construction (approximately)."""
        if self._min is None:
            return None
        return (host_s - dev_us * 1e-6 - self._min) * 1e3


class ImuAligner:
    """Finds IMU samples near a camera-frame timestamp."""

    def __init__(self, imu_samples, left_clock: ClockTracker, right_clock: ClockTracker):
        self._imu = imu_samples          # deque of ImuSample from imu_client (left device)
        self._lc = left_clock
        self._rc = right_clock

    def _snapshot(self):
        return list(self._imu)

    def near_left_frame(self, frame_dev_ts_us: int, max_gap_ms: float = 30.0):
        """Same-clock lookup. Returns (sample, dt_ms) or (None, None). dt = sample - frame."""
        return self._nearest(self._snapshot(), frame_dev_ts_us, max_gap_ms)

    def near_right_frame(self, frame_dev_ts_us: int, max_gap_ms: float = 40.0):
        """Cross-clock lookup through the Mac time base. Less accurate (see module docstring)."""
        if not (self._lc.ready and self._rc.ready):
            return None, None
        host_s = self._rc.to_host(frame_dev_ts_us)
        left_us = int((host_s - self._lc._min) * 1e6)  # back to the LEFT device clock
        return self._nearest(self._snapshot(), left_us, max_gap_ms)

    @staticmethod
    def _nearest(samples, target_us: int, max_gap_ms: float):
        if not samples:
            return None, None
        ts = [s.dev_us for s in samples]
        i = bisect.bisect_left(ts, target_us)
        cands = [j for j in (i - 1, i) if 0 <= j < len(samples)]
        j = min(cands, key=lambda k: abs(ts[k] - target_us))
        dt_ms = (ts[j] - target_us) / 1e3
        if abs(dt_ms) > max_gap_ms:
            return None, None
        return samples[j], dt_ms

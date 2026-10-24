"""One Euro Filter for real-time gaze smoothing.

Adapts its cutoff frequency to movement speed:
  - slow / still   → heavy smoothing (removes jitter)
  - fast movement  → light smoothing (keeps up with motion)

Parameters
----------
min_cutoff : float
    Minimum cutoff frequency in Hz.  Lower = more smoothing when still.
    Default 0.5 works well for gaze at ~30 fps.
beta : float
    Speed coefficient.  Higher = less lag when moving quickly.
    Default 0.05 for gaze; raise to 0.3+ for fast pointer movements.
d_cutoff : float
    Cutoff for the derivative low-pass filter.  1.0 Hz is usually fine.
"""

from __future__ import annotations

import math
import time
from collections import deque


class _LowPassFilter:
    __slots__ = ("_y", "_cutoff")

    def __init__(self, cutoff: float, initial: float = 0.0) -> None:
        self._y = initial
        self._cutoff = cutoff

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, freq: float) -> float:
        a = self._alpha(self._cutoff, freq)
        self._y = a * x + (1.0 - a) * self._y
        return self._y

    @property
    def value(self) -> float:
        return self._y


class OneEuroFilter:
    """Single-axis One Euro Filter.  Create one instance per axis (x, y)."""

    __slots__ = ("_min_cutoff", "_beta", "_d_cutoff", "_xf", "_dxf", "_last_t", "_last_x")

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._xf = _LowPassFilter(min_cutoff)
        self._dxf = _LowPassFilter(d_cutoff)
        self._last_t: float | None = None
        self._last_x: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.monotonic()

        if self._last_t is None:
            self._last_t = t
            self._last_x = x
            self._xf._y = x
            self._dxf._y = 0.0
            return x

        dt = t - self._last_t
        if dt <= 1e-9:
            return self._xf.value

        freq = 1.0 / dt

        dx = (x - (self._last_x or x)) * freq
        dx_hat = self._dxf(dx, freq)
        self._dxf._cutoff = self._d_cutoff

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        self._xf._cutoff = cutoff
        x_hat = self._xf(x, freq)

        self._last_t = t
        self._last_x = x
        return x_hat

    def reset(self, value: float | None = None) -> None:
        self._last_t = None
        self._last_x = None
        if value is not None:
            self._xf._y = value


class TemporalMedianFilter:
    """Sliding-window median to suppress single-frame MediaPipe outliers.

    A window of 5 frames at 30 fps removes spikes with ~67 ms of lag.
    """

    __slots__ = ("_buf", "_window")

    def __init__(self, window: int = 5) -> None:
        self._buf: deque[float] = deque(maxlen=window)
        self._window = window

    def __call__(self, x: float) -> float:
        self._buf.append(x)
        s = sorted(self._buf)
        return s[len(s) // 2]

    def reset(self, value: float | None = None) -> None:
        self._buf.clear()
        if value is not None:
            self._buf.append(value)

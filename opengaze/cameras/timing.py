"""Frame-rate and timing measurement helpers."""

from __future__ import annotations


class RateMeter:
    """Exponential moving average of frame interval, giving fps and interval jitter.

    Fed with timestamps in nanoseconds. Call ``reset()`` after a reconnect so the outage gap
    doesn't pollute the estimate.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self.reset()

    def reset(self) -> None:
        self._last_ns: int | None = None
        self._mean_interval_ns: float | None = None
        self._jitter_ns: float = 0.0

    def update(self, timestamp_ns: int) -> None:
        if self._last_ns is not None:
            dt = timestamp_ns - self._last_ns
            if dt > 0:
                if self._mean_interval_ns is None:
                    self._mean_interval_ns = float(dt)
                else:
                    self._mean_interval_ns += self._alpha * (dt - self._mean_interval_ns)
                    deviation = abs(dt - self._mean_interval_ns)
                    self._jitter_ns += self._alpha * (deviation - self._jitter_ns)
        self._last_ns = timestamp_ns

    @property
    def fps(self) -> float | None:
        if not self._mean_interval_ns:
            return None
        return 1e9 / self._mean_interval_ns

    @property
    def jitter_ms(self) -> float | None:
        if self._mean_interval_ns is None:
            return None
        return self._jitter_ns / 1e6

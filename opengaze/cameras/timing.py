"""Frame-rate and timing measurement helpers."""

from __future__ import annotations

import collections


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


class DeviceClockSync:
    """Maps a device's monotonic clock (an ESP32's microsecond timer) onto ``monotonic_ns``.

    The clocks are never synchronized. Every message carries a device timestamp and gets a
    host receive time, so ``host - device`` is the clock offset plus that message's
    transport delay. The smallest value seen recently is the best offset estimate: it is
    the message that waited least in Wi-Fi retries and buffers. Mapped times are therefore
    late by the minimum transport delay (unknown, a few ms on a quiet LAN) and never early.

    The minimum is taken over a sliding window so crystal drift (tens of ppm) can't
    accumulate. A device reboot or clock jump resets the estimate. 32-bit microsecond
    counters (Arduino ``micros()``) are unwrapped.
    """

    def __init__(self, window_s: float = 10.0, jump_tolerance_s: float = 1.0) -> None:
        self.window_ns = int(window_s * 1e9)
        self.jump_tolerance_ns = int(jump_tolerance_s * 1e9)
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self._samples: collections.deque[tuple[int, int]] = collections.deque()  # (host, offset)
        self._last_device_us: int | None = None
        self._wrap_us = 0
        self.offset_ns: int | None = None
        self.excess_delay_ns: int | None = None

    def _unwrap(self, device_us: int) -> int:
        last = self._last_device_us
        if last is not None and device_us + self._wrap_us < last - 2**31:
            self._wrap_us += 2**32  # a 32-bit counter rolled over (a reboot also lands here;
            # its offset jump then resets everything in ``update``)
        unwrapped = device_us + self._wrap_us
        self._last_device_us = unwrapped
        return unwrapped

    def update(self, device_us: int, host_ns: int) -> int:
        """Record one message; returns its device time mapped to the host clock."""
        device_ns = self._unwrap(device_us) * 1000
        offset = host_ns - device_ns
        if self.offset_ns is not None and abs(offset - self.offset_ns) > self.jump_tolerance_ns:
            # The device rebooted or its clock jumped: old offsets no longer apply.
            self.resets += 1
            self.reset()
            device_ns = self._unwrap(device_us) * 1000
            offset = host_ns - device_ns
        self._samples.append((host_ns, offset))
        cutoff = host_ns - self.window_ns
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            self._samples.popleft()
        self.offset_ns = min(o for _, o in self._samples)
        self.excess_delay_ns = offset - self.offset_ns
        return device_ns + self.offset_ns

    def map(self, device_us: int) -> int | None:
        """Map a device time with the current estimate, without recording a message.

        Use after ``update``-ing every message of a batch (one datagram holding several
        readings): they share one receive time, so only the whole batch gives the offset.
        """
        if self.offset_ns is None or self._last_device_us is None:
            return None
        unwrapped = device_us + self._wrap_us
        if unwrapped > self._last_device_us + 2**31:  # from before the latest roll-over
            unwrapped -= 2**32
        return unwrapped * 1000 + self.offset_ns

    @property
    def uncertainty_ns(self) -> int | None:
        """Typical spread of transport delay over the window: how far a mapped time may be
        from the true capture time beyond the (unknown) minimum delay."""
        if self.offset_ns is None or len(self._samples) < 2:
            return None
        excess = sorted(o - self.offset_ns for _, o in self._samples)
        return int(excess[len(excess) // 2])

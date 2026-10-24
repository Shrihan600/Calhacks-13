"""Dwell click detector.

The user looks at a point for `dwell_ms` milliseconds without moving more
than `radius_px` pixels — that fires a left click.  A cooldown period
prevents immediate re-click after a click fires.

State machine
-------------
IDLE      → cursor enters radius around current anchor → DWELLING
DWELLING  → cursor leaves radius → IDLE (reset anchor)
DWELLING  → dwell_ms elapsed → fire click → COOLDOWN
COOLDOWN  → cooldown_ms elapsed → IDLE
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass


class DwellState(enum.Enum):
    IDLE = "idle"
    DWELLING = "dwelling"
    COOLDOWN = "cooldown"


@dataclass
class DwellResult:
    clicked: bool
    state: DwellState
    progress: float  # 0.0 → 1.0 fill fraction (useful for UI ring)
    anchor_x: int | None
    anchor_y: int | None


class DwellDetector:
    """Detects dwell clicks from a stream of (x, y) cursor positions.

    Coordinates are in screen pixels.  Call ``update()`` on every frame;
    it returns a ``DwellResult`` with ``clicked=True`` on the frame that
    triggers the click.
    """

    def __init__(
        self,
        dwell_ms: float = 600.0,
        radius_px: float = 40.0,
        cooldown_ms: float = 800.0,
        enabled: bool = True,
    ) -> None:
        self.dwell_ms = dwell_ms
        self.radius_px = radius_px
        self.cooldown_ms = cooldown_ms
        self.enabled = enabled

        self._state = DwellState.IDLE
        self._anchor_x: int | None = None
        self._anchor_y: int | None = None
        self._state_entered: float = time.monotonic()
        self._last_t: float = time.monotonic()

    def update(self, x: int, y: int, t: float | None = None) -> DwellResult:
        if t is None:
            t = time.monotonic()
        self._last_t = t

        if not self.enabled:
            return DwellResult(False, DwellState.IDLE, 0.0, None, None)

        elapsed = t - self._state_entered

        if self._state == DwellState.COOLDOWN:
            if elapsed >= self.cooldown_ms / 1000.0:
                self._enter(DwellState.IDLE, t)
                self._anchor_x = x
                self._anchor_y = y
            return DwellResult(False, self._state, 0.0, self._anchor_x, self._anchor_y)

        if self._state == DwellState.IDLE:
            self._anchor_x = x
            self._anchor_y = y
            self._enter(DwellState.DWELLING, t)
            return DwellResult(False, self._state, 0.0, self._anchor_x, self._anchor_y)

        # DWELLING
        assert self._anchor_x is not None and self._anchor_y is not None
        dist = _dist(x, y, self._anchor_x, self._anchor_y)
        if dist > self.radius_px:
            # Moved — reset anchor to new position
            self._anchor_x = x
            self._anchor_y = y
            self._enter(DwellState.DWELLING, t)
            return DwellResult(False, self._state, 0.0, self._anchor_x, self._anchor_y)

        progress = min(elapsed / (self.dwell_ms / 1000.0), 1.0)
        if elapsed >= self.dwell_ms / 1000.0:
            self._enter(DwellState.COOLDOWN, t)
            return DwellResult(True, DwellState.COOLDOWN, 1.0, self._anchor_x, self._anchor_y)

        return DwellResult(False, self._state, progress, self._anchor_x, self._anchor_y)

    def _enter(self, state: DwellState, t: float) -> None:
        self._state = state
        self._state_entered = t

    def reset(self) -> None:
        self._state = DwellState.IDLE
        self._anchor_x = None
        self._anchor_y = None
        now = time.monotonic()
        self._state_entered = now
        self._last_t = now

    @property
    def state(self) -> DwellState:
        return self._state

    @property
    def progress(self) -> float:
        if self._state != DwellState.DWELLING:
            return 0.0
        elapsed = self._last_t - self._state_entered
        return min(elapsed / (self.dwell_ms / 1000.0), 1.0)


def _dist(x1: int, y1: int, x2: int, y2: int) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

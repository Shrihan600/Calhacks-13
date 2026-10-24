"""Long-blink click detector.

Natural blinks are ~100–150 ms — ignored.
An intentional long blink (both eyes closed >= long_blink_ms) fires a click.
A cooldown prevents re-firing immediately after.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BlinkResult:
    clicked: bool
    any_blink: bool    # either eye is currently closed (pause cursor)
    long_blink: bool   # currently in a long blink (for visual feedback)
    blink_progress: float  # 0→1 how far into long-blink threshold we are


class BlinkClickDetector:
    """Fires a click when both eyes stay closed >= long_blink_ms.

    Natural blinks are short (<= short_blink_ms) and are ignored.
    During any blink the cursor movement is optionally paused by the caller.
    """

    def __init__(
        self,
        long_blink_ms: float = 400.0,
        short_blink_ms: float = 180.0,
        cooldown_ms: float = 800.0,
    ) -> None:
        self.long_blink_ms = long_blink_ms
        self.short_blink_ms = short_blink_ms
        self.cooldown_ms = cooldown_ms

        self._both_closed_since: float | None = None
        self._cooldown_until: float = 0.0
        self._fired_this_blink = False

    def update(
        self,
        left_blink: bool,
        right_blink: bool,
        t: float | None = None,
    ) -> BlinkResult:
        if t is None:
            t = time.monotonic()

        any_blink = left_blink or right_blink
        both_closed = left_blink and right_blink

        in_cooldown = t < self._cooldown_until

        if both_closed:
            if self._both_closed_since is None:
                self._both_closed_since = t
                self._fired_this_blink = False
            elapsed_ms = (t - self._both_closed_since) * 1000.0
            progress = min(elapsed_ms / self.long_blink_ms, 1.0)
            long_blink = elapsed_ms >= self.long_blink_ms

            clicked = False
            if long_blink and not self._fired_this_blink and not in_cooldown:
                clicked = True
                self._fired_this_blink = True
                self._cooldown_until = t + self.cooldown_ms / 1000.0

            return BlinkResult(clicked=clicked, any_blink=True, long_blink=long_blink, blink_progress=progress)
        else:
            self._both_closed_since = None
            self._fired_this_blink = False
            return BlinkResult(clicked=False, any_blink=any_blink, long_blink=False, blink_progress=0.0)

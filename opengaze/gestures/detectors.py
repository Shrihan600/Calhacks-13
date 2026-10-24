"""Gesture detectors. Each consumes ``EyeSignal`` frames and emits ``GestureEvent``s.

Detectors are independent and stateful; ``GestureRecognizer`` runs them side by side. They
never look at the clock themselves, so tests drive them with synthetic timestamps.

Timing rules that keep natural behaviour from triggering anything:

* A natural blink (both eyes, ~100–150 ms) is not a gesture on its own. Only a *long* blink,
  or two/three short blinks in quick succession, is.
* A wink requires the other eye to stay open for the whole closure; if both close it's a blink.
* A glance requires looking *past* the screen edge (configurable margin), not just near it.
"""

from __future__ import annotations

from opengaze.gestures.config import BlinkSettings, GlanceSettings, WinkSettings
from opengaze.gestures.types import EyeSignal, Gesture, GestureEvent


class BlinkPatternDetector:
    """Long blink, double blink, triple blink.

    A long blink fires while the eyes are still closed (as soon as the threshold is reached).
    Double/triple blinks fire once the gap after the last short blink has expired, so a
    triple blink is never also reported as a double.
    """

    def __init__(self, settings: BlinkSettings | None = None) -> None:
        self.s = settings or BlinkSettings()
        self._closed_since: float | None = None
        self._long_fired = False
        self._short_count = 0
        self._last_short_end: float | None = None
        self._cooldown_until = 0.0

    @property
    def closed_since(self) -> float | None:
        return self._closed_since

    def progress(self, t: float) -> float:
        """0 → 1 toward a long blink (for a fill ring in the UI)."""
        if self._closed_since is None or self._long_fired:
            return 0.0
        return min(1.0, (t - self._closed_since) * 1000.0 / self.s.long_blink_ms)

    def update(self, sig: EyeSignal) -> list[GestureEvent]:
        s, t = self.s, sig.t
        events: list[GestureEvent] = []
        both_closed = bool(sig.left_closed) and bool(sig.right_closed)

        if both_closed:
            if self._closed_since is None:
                self._closed_since = t
                self._long_fired = False
            held_ms = (t - self._closed_since) * 1000.0
            if held_ms >= s.long_blink_ms and not self._long_fired:
                self._long_fired = True
                self._short_count = 0  # a long blink ends any multi-blink sequence
                if t >= self._cooldown_until:
                    events.append(GestureEvent(Gesture.LONG_BLINK, t))
                    self._cooldown_until = t + s.cooldown_ms / 1000.0
            return events

        if self._closed_since is not None:
            held_ms = (t - self._closed_since) * 1000.0
            if not self._long_fired and s.short_blink_min_ms <= held_ms <= s.short_blink_max_ms:
                self._short_count += 1
                self._last_short_end = t
            elif not self._long_fired:
                self._short_count = 0  # too short (noise) or between short and long
            self._closed_since = None
            self._long_fired = False

        if self._short_count and self._last_short_end is not None:
            gap_ms = (t - self._last_short_end) * 1000.0
            if gap_ms > s.multi_blink_gap_ms or self._short_count >= 3:
                n, self._short_count = self._short_count, 0
                gesture = {2: Gesture.DOUBLE_BLINK, 3: Gesture.TRIPLE_BLINK}.get(n)
                if gesture is not None and t >= self._cooldown_until:
                    events.append(GestureEvent(gesture, t))
                    self._cooldown_until = t + s.cooldown_ms / 1000.0
        return events

    def reset(self) -> None:
        self._closed_since = None
        self._long_fired = False
        self._short_count = 0
        self._last_short_end = None


class WinkDetector:
    """Left/right wink (fires on reopening) and long wink (fires while held)."""

    def __init__(self, settings: WinkSettings | None = None) -> None:
        self.s = settings or WinkSettings()
        self._eye: str | None = None  # eye currently winking
        self._since = 0.0
        self._long_fired = False
        self._spoiled = False  # the other eye closed during this wink
        self._cooldown_until = 0.0

    def progress(self, t: float) -> float:
        if self._eye is None or self._spoiled or self._long_fired:
            return 0.0
        return min(1.0, (t - self._since) * 1000.0 / self.s.long_wink_ms)

    def update(self, sig: EyeSignal) -> list[GestureEvent]:
        s, t = self.s, sig.t
        # A wink needs both eyes observed; an unobserved eye is not "open".
        if sig.left_closed is None or sig.right_closed is None:
            self.reset()
            return []
        left, right = sig.left_closed, sig.right_closed
        events: list[GestureEvent] = []

        if self._eye is None:
            if left != right:
                self._eye = "left" if left else "right"
                self._since = t
                self._long_fired = False
                self._spoiled = False
            return events

        winking_closed = left if self._eye == "left" else right
        other_closed = right if self._eye == "left" else left
        if other_closed:
            self._spoiled = True  # turned into a blink; never report this as a wink
        if winking_closed:
            held_ms = (t - self._since) * 1000.0
            if not self._spoiled and not self._long_fired and held_ms >= s.long_wink_ms:
                self._long_fired = True
                if t >= self._cooldown_until:
                    g = Gesture.LONG_WINK_LEFT if self._eye == "left" else Gesture.LONG_WINK_RIGHT
                    events.append(GestureEvent(g, t))
                    self._cooldown_until = t + s.cooldown_ms / 1000.0
            return events

        # Winking eye reopened.
        held_ms = (t - self._since) * 1000.0
        if (
            not self._spoiled
            and not self._long_fired
            and s.min_wink_ms <= held_ms < s.long_wink_ms
            and t >= self._cooldown_until
        ):
            g = Gesture.WINK_LEFT if self._eye == "left" else Gesture.WINK_RIGHT
            events.append(GestureEvent(g, t))
            self._cooldown_until = t + s.cooldown_ms / 1000.0
        self._eye = None
        # The other eye may have started its own wink on this very frame.
        if left != right:
            return events + self.update(sig)
        return events

    def reset(self) -> None:
        self._eye = None
        self._long_fired = False
        self._spoiled = False


_GLANCE = {
    "left": Gesture.GLANCE_LEFT,
    "right": Gesture.GLANCE_RIGHT,
    "up": Gesture.GLANCE_UP,
    "down": Gesture.GLANCE_DOWN,
}
_HOLD = {
    "left": Gesture.HOLD_LEFT,
    "right": Gesture.HOLD_RIGHT,
    "up": Gesture.HOLD_UP,
    "down": Gesture.HOLD_DOWN,
}


class GlanceDetector:
    """Glance (out past an edge and back quickly) and hold (out past an edge and stay).

    The zone is entered when gaze passes ``-edge_margin`` / ``1 + edge_margin`` and left only
    once gaze is back inside by ``hysteresis``, so jitter at the boundary doesn't toggle it.
    Frames with no gaze (blinks, no model) neither enter nor leave a zone.
    """

    def __init__(self, settings: GlanceSettings | None = None) -> None:
        self.s = settings or GlanceSettings()
        self._zone: str | None = None
        self._since = 0.0
        self._holds_fired = 0
        self._next_repeat = 0.0
        self._cooldown_until = 0.0

    @property
    def zone(self) -> str | None:
        return self._zone

    def progress(self, t: float) -> float:
        if self._zone is None or self._holds_fired:
            return 0.0
        return min(1.0, (t - self._since) * 1000.0 / self.s.hold_ms)

    def _zone_of(self, x: float, y: float, inner: float) -> str | None:
        m = self.s.edge_margin + inner
        # Horizontal wins on a corner: most bindings are left/right.
        if x < -m:
            return "left"
        if x > 1.0 + m:
            return "right"
        if y < -m:
            return "up"
        if y > 1.0 + m:
            return "down"
        return None

    def update(self, sig: EyeSignal) -> list[GestureEvent]:
        s, t = self.s, sig.t
        if sig.gaze_x is None or sig.gaze_y is None:
            return []
        events: list[GestureEvent] = []

        if self._zone is None:
            zone = self._zone_of(sig.gaze_x, sig.gaze_y, 0.0)
            if zone is not None:
                self._zone, self._since, self._holds_fired = zone, t, 0
            return events

        # Still in the zone? (leave only once back inside by the hysteresis band)
        still = self._zone_of(sig.gaze_x, sig.gaze_y, -s.hysteresis) == self._zone
        held_ms = (t - self._since) * 1000.0
        if still:
            if self._holds_fired == 0 and held_ms >= s.hold_ms:
                events.append(GestureEvent(_HOLD[self._zone], t))
                self._holds_fired = 1
                if s.hold_repeat_ms is not None:
                    self._next_repeat = t + s.hold_repeat_ms / 1000.0
            elif (
                self._holds_fired
                and s.hold_repeat_ms is not None
                and t >= self._next_repeat
            ):
                events.append(GestureEvent(_HOLD[self._zone], t, repeat=self._holds_fired))
                self._holds_fired += 1
                self._next_repeat = t + s.hold_repeat_ms / 1000.0
            return events

        if (
            self._holds_fired == 0
            and s.min_glance_ms <= held_ms <= s.max_glance_ms
            and t >= self._cooldown_until
        ):
            events.append(GestureEvent(_GLANCE[self._zone], t))
            self._cooldown_until = t + s.cooldown_ms / 1000.0
        self._zone = None
        return events

    def reset(self) -> None:
        self._zone = None
        self._holds_fired = 0

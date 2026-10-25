"""Runs every gesture detector on each frame and reports events plus UI feedback."""

from __future__ import annotations

from dataclasses import dataclass

from opengaze.gestures.config import GestureSettings
from opengaze.gestures.detectors import BlinkPatternDetector, GlanceDetector, WinkDetector
from opengaze.gestures.types import EyeSignal, Gesture, GestureEvent


@dataclass(frozen=True, slots=True)
class GestureFrame:
    events: tuple[GestureEvent, ...]
    any_closed: bool  # freeze the cursor: gaze is unreliable while an eye is closing
    blink_progress: float  # 0 → 1 toward a long blink
    wink_progress: float  # 0 → 1 toward a long wink
    hold_progress: float  # 0 → 1 toward an edge hold
    glance_zone: str | None  # edge currently looked past, if any


class GestureRecognizer:
    """Combines the blink, wink and glance detectors.

    With a single eye camera (Mode B) the unobserved eye is reported as None; for blink
    purposes it is assumed to mirror the observed eye, and winks are unavailable.

    While a wink is held (``hold_wink_*``, e.g. a drag), blinks are not detected: the other
    eye's natural blinks close both eyes for a moment and would otherwise count toward a
    double blink.
    """

    def __init__(self, settings: GestureSettings | None = None) -> None:
        self.settings = settings or GestureSettings()
        self.blink = BlinkPatternDetector(self.settings.blink)
        self.wink = WinkDetector(self.settings.wink)
        self.glance = GlanceDetector(self.settings.glance)

    def update(self, sig: EyeSignal) -> GestureFrame:
        left, right = sig.left_closed, sig.right_closed
        any_closed = bool(left) or bool(right)
        if not self.settings.enabled:
            return GestureFrame((), any_closed, 0.0, 0.0, 0.0, None)

        blink_sig = sig
        if (left is None) != (right is None):
            observed = left if left is not None else right
            blink_sig = EyeSignal(sig.t, observed, observed, sig.gaze_x, sig.gaze_y)

        wink_events = self.wink.update(sig)
        if self.wink.holding or any(e.phase == "end" for e in wink_events):
            self.blink.reset()
            blink_events = []
        else:
            blink_events = self.blink.update(blink_sig)
        events = [*blink_events, *wink_events, *self.glance.update(sig)]
        return GestureFrame(
            events=tuple(events),
            any_closed=any_closed,
            # The fill ring only promises something when a long blink does something.
            blink_progress=(
                self.blink.progress(sig.t)
                if Gesture.LONG_BLINK in self.settings.bindings
                else 0.0
            ),
            wink_progress=self.wink.progress(sig.t),
            hold_progress=self.glance.progress(sig.t),
            glance_zone=self.glance.zone,
        )

    def reset(self) -> None:
        self.blink.reset()
        self.wink.reset()
        self.glance.reset()

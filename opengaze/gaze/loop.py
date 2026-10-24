"""GazeLoop: tracker → screen mapper → smoothing → gestures, once per UI tick.

Every desktop window (test environment, keyboard, practice) runs the same loop, so it lives
here instead of being copied into each window.

* Gaze freezes while either eye is closed: pupil position is unreliable as the lid moves, and
  a blink/wink gesture should act at the pre-blink position.
* Gestures see the *unclipped, unsmoothed* gaze, so glances past the screen edge are
  detectable and not delayed by the filter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from opengaze.calibration.model import ScreenMapper
from opengaze.calibration.session import GRID_HI, GRID_LO
from opengaze.control.filter import OneEuroFilter
from opengaze.gaze.tracker import GazeTracker, TrackerFrame
from opengaze.gestures.config import GestureSettings
from opengaze.gestures.recognizer import GestureFrame, GestureRecognizer
from opengaze.gestures.types import EyeSignal


def expand(v: float) -> float:
    """Remap the Mode A ridge output range [GRID_LO, GRID_HI] → [0, 1] (unclipped).

    Ridge shrinkage compresses Mode A predictions toward the centre; Modes B/C mappers are
    fitted with much less regularisation and are used as is.
    """
    return (v - GRID_LO) / (GRID_HI - GRID_LO)


@dataclass
class GazeState:
    frame: TrackerFrame
    gaze_x: float  # smoothed, [0, 1]
    gaze_y: float
    confidence: float
    gesture: GestureFrame
    updated: bool  # gaze moved on this tick (False while frozen / low confidence / no model)


class GazeLoop:
    def __init__(
        self,
        tracker: GazeTracker,
        model: ScreenMapper | None,
        gestures: GestureSettings | None = None,
        *,
        min_confidence: float = 0.25,
        filter_min_cutoff: float = 0.8,
        filter_beta: float = 0.07,
    ) -> None:
        self.tracker = tracker
        self.model = model
        self.recognizer = GestureRecognizer(gestures)
        self.min_confidence = min_confidence
        self._fx = OneEuroFilter(filter_min_cutoff, filter_beta)
        self._fy = OneEuroFilter(filter_min_cutoff, filter_beta)
        self._expand = tracker.is_mode_a
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.confidence = 0.0

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        self.recognizer.reset()

    def tick(self, manager: object, t: float | None = None) -> GazeState:
        t = time.monotonic() if t is None else t
        frame = self.tracker.step(manager)
        any_closed = bool(frame.left_closed) or bool(frame.right_closed)

        raw: tuple[float, float] | None = None
        updated = False
        fv = frame.features
        if self.model is not None and fv is not None and not any_closed:
            try:
                sample = self.model.predict(fv)
            except Exception:
                sample = None
            if sample is not None and sample.confidence >= self.min_confidence:
                rx = sample.raw_x_norm if sample.raw_x_norm is not None else sample.screen_x_norm
                ry = sample.raw_y_norm if sample.raw_y_norm is not None else sample.screen_y_norm
                if self._expand:
                    rx, ry = expand(rx), expand(ry)
                raw = (rx, ry)
                self.confidence = sample.confidence
                self.gaze_x = min(1.0, max(0.0, self._fx(min(1.0, max(0.0, rx)), t)))
                self.gaze_y = min(1.0, max(0.0, self._fy(min(1.0, max(0.0, ry)), t)))
                updated = True

        signal = EyeSignal(
            t,
            frame.left_closed,
            frame.right_closed,
            raw[0] if raw else None,
            raw[1] if raw else None,
        )
        gesture = self.recognizer.update(signal)
        return GazeState(frame, self.gaze_x, self.gaze_y, self.confidence, gesture, updated)


def clicked(state: GazeState, settings: GestureSettings) -> bool:
    """True if a gesture bound to ``mouse: click`` fired this tick (activities use it to
    select the key/target under the gaze, whatever gesture the user chose for clicking)."""
    for event in state.gesture.events:
        binding = settings.bindings.get(event.gesture)
        if binding is not None and binding.mouse == "click":
            return True
    return False

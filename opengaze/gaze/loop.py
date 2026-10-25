"""GazeLoop: tracker → screen mapper → smoothing → gestures, once per UI tick.

Every desktop window (test environment, keyboard, practice) runs the same loop, so it lives
here instead of being copied into each window.

* Gaze freezes while either eye is closed: pupil position is unreliable as the lid moves, and
  a blink/wink gesture should act at the pre-blink position. The exception is
  ``track_one_eye`` (set while a held wink holds the mouse button down): then gaze follows the
  open eye, so the held button drags. The mapper falls back to that eye's sub-model, since
  a closed eye's features are marked invalid.
* Gestures see the *unclipped, unsmoothed* gaze, so glances past the screen edge are
  detectable and not delayed by the filter.
* Every tick that doesn't move the gaze says why (``GazeState.drop_reason``), so a cursor
  that won't move can be traced to the stage that stopped it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

from opengaze.calibration.model import ScreenMapper
from opengaze.control.filter import OneEuroFilter, TemporalMedianFilter
from opengaze.gaze.tracker import GazeTracker, TrackerFrame
from opengaze.gestures.config import GestureSettings
from opengaze.gestures.recognizer import GestureFrame, GestureRecognizer
from opengaze.gestures.types import EyeSignal

log = logging.getLogger(__name__)

DropReason = Literal[
    "no_model", "no_frames", "eye_closed", "predict_error", "no_prediction", "low_confidence"
]


@dataclass
class GazeState:
    frame: TrackerFrame
    gaze_x: float  # smoothed, [0, 1]
    gaze_y: float
    confidence: float
    gesture: GestureFrame
    updated: bool  # gaze moved on this tick (False while frozen / low confidence / no model)
    raw: tuple[float, float] | None = None  # unclipped, unsmoothed model output
    drop_reason: DropReason | None = None   # why the gaze didn't move (None when updated)


class GazeLoop:
    def __init__(
        self,
        tracker: GazeTracker,
        model: ScreenMapper | None,
        gestures: GestureSettings | None = None,
        *,
        min_confidence: float = 0.25,
        filter_min_cutoff: float = 0.3,
        filter_beta: float = 0.1,
        median_window: int = 5,
    ) -> None:
        self.tracker = tracker
        self.model = model
        self.recognizer = GestureRecognizer(gestures)
        self.min_confidence = min_confidence
        # Median pre-filter drops single-frame outliers (e.g. a spike when the face is
        # partly visible); the low One Euro cutoff steadies fixations, beta keeps saccades fast.
        self._mx = TemporalMedianFilter(median_window)
        self._my = TemporalMedianFilter(median_window)
        self._fx = OneEuroFilter(filter_min_cutoff, filter_beta)
        self._fy = OneEuroFilter(filter_min_cutoff, filter_beta)
        self.gaze_x = 0.5
        self.gaze_y = 0.5
        self.confidence = 0.0
        self.track_one_eye = False  # keep tracking with one eye closed (drag in progress)

    def reset(self) -> None:
        self._mx.reset()
        self._my.reset()
        self._fx.reset()
        self._fy.reset()
        self.recognizer.reset()

    def tick(self, manager: object, t: float | None = None) -> GazeState:
        t = time.monotonic() if t is None else t
        frame = self.tracker.step(manager)
        both_closed = bool(frame.left_closed) and bool(frame.right_closed)
        any_closed = bool(frame.left_closed) or bool(frame.right_closed)
        frozen = both_closed or (any_closed and not self.track_one_eye)

        raw: tuple[float, float] | None = None
        updated = False
        reason: DropReason | None = None
        fv = frame.features
        if self.model is None:
            reason = "no_model"
        elif fv is None:
            reason = "no_frames"
        elif frozen:
            reason = "eye_closed"
        else:
            try:
                sample = self.model.predict(fv)
            except Exception as exc:
                log.debug("predict failed: %s", exc)
                sample, reason = None, "predict_error"
            if sample is not None:
                if sample.confidence <= 0.0 and sample.raw_x_norm is None:
                    reason = "no_prediction"  # the valid features couldn't support one
                elif sample.confidence < self.min_confidence:
                    reason = "low_confidence"
                    self.confidence = sample.confidence
                else:
                    rx, ry = sample.raw_x_norm, sample.raw_y_norm
                    if rx is None or ry is None:
                        rx, ry = sample.screen_x_norm, sample.screen_y_norm
                    raw = (rx, ry)
                    self.confidence = sample.confidence
                    mx = self._mx(min(1.0, max(0.0, rx)))
                    my = self._my(min(1.0, max(0.0, ry)))
                    self.gaze_x = min(1.0, max(0.0, self._fx(mx, t)))
                    self.gaze_y = min(1.0, max(0.0, self._fy(my, t)))
                    updated = True

        signal = EyeSignal(
            t,
            frame.left_closed,
            frame.right_closed,
            raw[0] if raw else None,
            raw[1] if raw else None,
        )
        gesture = self.recognizer.update(signal)
        return GazeState(
            frame, self.gaze_x, self.gaze_y, self.confidence, gesture, updated, raw, reason
        )


def clicked(state: GazeState, settings: GestureSettings) -> bool:
    """True if a gesture bound to ``mouse: click`` fired this tick (activities use it to
    select the key/target under the gaze, whatever gesture the user chose for clicking)."""
    for event in state.gesture.events:
        binding = settings.bindings.get(event.gesture)
        if binding is not None and binding.mouse == "click":
            return True
    return False

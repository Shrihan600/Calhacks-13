"""Gesture names, the per-frame eye signal gestures are detected from, and gesture events."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal


class Gesture(enum.StrEnum):
    # Blink family (both eyes).
    LONG_BLINK = "long_blink"
    DOUBLE_BLINK = "double_blink"
    TRIPLE_BLINK = "triple_blink"
    # Winks (one eye closed, the other open).
    WINK_LEFT = "wink_left"
    WINK_RIGHT = "wink_right"
    LONG_WINK_LEFT = "long_wink_left"
    LONG_WINK_RIGHT = "long_wink_right"
    # Held winks: a "start" event once the wink has lasted hold_press_ms, an "end" event when
    # the eye reopens. Bound to ``mouse: hold``, the button is down in between (drag).
    HOLD_WINK_LEFT = "hold_wink_left"
    HOLD_WINK_RIGHT = "hold_wink_right"
    # Glances: look past a screen edge and come straight back.
    GLANCE_LEFT = "glance_left"
    GLANCE_RIGHT = "glance_right"
    GLANCE_UP = "glance_up"
    GLANCE_DOWN = "glance_down"
    # Holds: look past a screen edge and stay there (repeats while held, if configured).
    HOLD_LEFT = "hold_left"
    HOLD_RIGHT = "hold_right"
    HOLD_UP = "hold_up"
    HOLD_DOWN = "hold_down"


@dataclass(frozen=True, slots=True)
class EyeSignal:
    """What the gesture detectors see on one frame.

    ``*_closed`` is None when that eye isn't observed at all (no camera, face lost), which is
    different from "open". Gaze is in normalised screen coordinates and is **not** clipped, so
    values outside [0, 1] mean the user is looking past the screen edge. It is None while no
    gaze model is available or the eyes are closed.
    """

    t: float  # seconds, monotonic
    left_closed: bool | None
    right_closed: bool | None
    gaze_x: float | None = None
    gaze_y: float | None = None


@dataclass(frozen=True, slots=True)
class GestureEvent:
    gesture: Gesture
    t: float
    repeat: int = 0  # 0 for the first firing of a hold, 1, 2, … for repeats
    # Winks: position in a run of quick winks of the same eye (2 = second wink of a double).
    count: int = 1
    # Held winks: "start" when the hold begins, "end" when it ends; None for other gestures.
    phase: Literal["start", "end"] | None = None

"""Gesture → action dispatch.

``ActionDispatcher`` looks up each ``GestureEvent`` in the configured bindings and either
posts a system keystroke / mouse event through an ``InputBackend`` (Quartz on macOS) or calls
an OpenGaze app handler registered by the UI (pause, recalibrate, …).

System input is gated: while ``system_input_enabled`` is False (cursor control off, or
paused), key and mouse bindings are reported as suppressed instead of being posted. App
actions always run, so the user can always un-pause with a gesture.

Held gestures (``hold_wink_*``) come as a "start" and an "end" event. Bound to ``mouse:
hold`` they press the left button at the start and release it at the end; the release is
posted even if system input was switched off meanwhile, so the button can never stay
stuck down. Other bindings act on the start only.

Repeated winks (``GestureEvent.count`` > 1) bound to a click become multi-clicks: the click
is posted with that click count at the position of the run's first click, so gaze drift
between winks doesn't break a double click. Gaze that has moved further than
``MULTI_CLICK_RADIUS_PX`` starts a new single click instead.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from opengaze.gestures.config import Binding
from opengaze.gestures.keys import KeyCombo, parse_key_combo
from opengaze.gestures.types import Gesture, GestureEvent

log = logging.getLogger(__name__)

MULTI_CLICK_RADIUS_PX = 60.0
_COUNTED = frozenset({"click", "right_click"})  # mouse actions that take a click count


class InputBackend(Protocol):
    def key_combo(self, combo: KeyCombo) -> None: ...

    def mouse(self, action: str, x: int, y: int, count: int = 1) -> None: ...


@dataclass(frozen=True, slots=True)
class DispatchResult:
    event: GestureEvent
    binding: Binding | None
    # "done" | "unbound" | "suppressed" | "no_handler" | "no_backend" | "error" | "ignored"
    outcome: str


class ActionDispatcher:
    def __init__(
        self,
        bindings: Mapping[Gesture, Binding],
        backend: InputBackend | None = None,
        app_handlers: Mapping[str, Callable[[], None]] | None = None,
    ) -> None:
        self.bindings = dict(bindings)
        self.backend = backend
        self.app_handlers = dict(app_handlers or {})
        self.system_input_enabled = False
        self._combos = {
            g: parse_key_combo(b.key) for g, b in self.bindings.items() if b.key is not None
        }
        self._last_click: tuple[str, tuple[int, int]] | None = None  # action, position

    def dispatch(self, event: GestureEvent, gaze_px: tuple[int, int] | None) -> DispatchResult:
        binding = self.bindings.get(event.gesture)
        if binding is None:
            return DispatchResult(event, None, "unbound")
        try:
            outcome = self._run(event, binding, gaze_px)
        except Exception as exc:  # a failed keybind must never take the gaze loop down
            log.warning("binding %s → %s failed: %s", event.gesture, binding.describe(), exc)
            outcome = "error"
        if outcome == "done":
            log.info("gesture %s → %s", event.gesture.value, binding.describe())
        return DispatchResult(event, binding, outcome)

    def _run(self, event: GestureEvent, b: Binding, gaze_px: tuple[int, int] | None) -> str:
        gesture = event.gesture
        if event.phase == "end":
            if b.mouse != "hold":
                return "ignored"  # other bindings act when a hold starts
            if self.backend is None:
                return "no_backend"
            self.backend.mouse("release", *(gaze_px or (-1, -1)))  # never gated: see above
            return "done"
        if b.app is not None:
            handler = self.app_handlers.get(b.app)
            if handler is None:
                return "no_handler"
            handler()
            return "done"
        if not self.system_input_enabled:
            return "suppressed"
        if self.backend is None:
            return "no_backend"
        if b.key is not None:
            self.backend.key_combo(self._combos[gesture])
            return "done"
        if gaze_px is None:
            return "suppressed"
        assert b.mouse is not None
        if b.mouse == "hold":
            self.backend.mouse("press", *gaze_px)
            return "done"
        if b.mouse not in _COUNTED:
            self.backend.mouse(b.mouse, *gaze_px)
            return "done"
        pos, count = gaze_px, 1
        last = self._last_click
        if (
            event.count > 1
            and last is not None
            and last[0] == b.mouse
            and math.dist(last[1], gaze_px) <= MULTI_CLICK_RADIUS_PX
        ):
            pos, count = last[1], event.count
        self._last_click = (b.mouse, pos)
        self.backend.mouse(b.mouse, *pos, count=count)
        return "done"

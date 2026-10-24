"""Gesture → action dispatch.

``ActionDispatcher`` looks up each ``GestureEvent`` in the configured bindings and either
posts a system keystroke / mouse event through an ``InputBackend`` (Quartz on macOS) or calls
an OpenGaze app handler registered by the UI (pause, recalibrate, …).

System input is gated: while ``system_input_enabled`` is False (cursor control off, or
paused), key and mouse bindings are reported as suppressed instead of being posted. App
actions always run, so the user can always un-pause with a gesture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from opengaze.gestures.config import Binding
from opengaze.gestures.keys import KeyCombo, parse_key_combo
from opengaze.gestures.types import Gesture, GestureEvent

log = logging.getLogger(__name__)


class InputBackend(Protocol):
    def key_combo(self, combo: KeyCombo) -> None: ...

    def mouse(self, action: str, x: int, y: int) -> None: ...


@dataclass(frozen=True, slots=True)
class DispatchResult:
    event: GestureEvent
    binding: Binding | None
    outcome: str  # "done" | "unbound" | "suppressed" | "no_handler" | "no_backend" | "error"


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

    def dispatch(self, event: GestureEvent, gaze_px: tuple[int, int] | None) -> DispatchResult:
        binding = self.bindings.get(event.gesture)
        if binding is None:
            return DispatchResult(event, None, "unbound")
        try:
            outcome = self._run(event.gesture, binding, gaze_px)
        except Exception as exc:  # a failed keybind must never take the gaze loop down
            log.warning("binding %s → %s failed: %s", event.gesture, binding.describe(), exc)
            outcome = "error"
        if outcome == "done":
            log.info("gesture %s → %s", event.gesture.value, binding.describe())
        return DispatchResult(event, binding, outcome)

    def _run(self, gesture: Gesture, b: Binding, gaze_px: tuple[int, int] | None) -> str:
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
        self.backend.mouse(b.mouse, *gaze_px)
        return "done"

"""Eye gestures: blink family, winks, glances, recognizer, bindings and key combos."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from opengaze.config.loader import load_app_config
from opengaze.gestures import (
    ActionDispatcher,
    Binding,
    EyeSignal,
    Gesture,
    GestureEvent,
    GestureRecognizer,
    GestureSettings,
)
from opengaze.gestures.keys import parse_key_combo
from tests.helpers import REPO_ROOT

DT = 1 / 30


def run(rec: GestureRecognizer, frames: list[tuple]) -> list[Gesture]:
    """frames: (duration_s, left_closed, right_closed, gaze_x, gaze_y)."""
    t = 0.0
    out: list[Gesture] = []
    for duration, left, right, gx, gy in frames:
        end = t + duration
        while t < end - 1e-9:
            out += [e.gesture for e in rec.update(EyeSignal(t, left, right, gx, gy)).events]
            t += DT
    return out


OPEN = (False, False, 0.5, 0.5)
CLOSED = (True, True, None, None)


def eyes(duration, state):
    return (duration, *state)


class TestBlinks:
    def test_natural_blink_is_ignored(self):
        frames = [eyes(0.5, OPEN), eyes(0.12, CLOSED), eyes(1.0, OPEN)]
        assert run(GestureRecognizer(), frames) == []

    def test_long_blink_fires_once_while_closed(self):
        got = run(GestureRecognizer(), [eyes(0.3, OPEN), eyes(1.2, CLOSED), eyes(0.5, OPEN)])
        assert got == [Gesture.LONG_BLINK]

    def test_double_and_triple_blink(self):
        blink = [eyes(0.12, CLOSED), eyes(0.15, OPEN)]
        assert run(GestureRecognizer(), [eyes(0.3, OPEN), *blink * 2, eyes(1.0, OPEN)]) == [
            Gesture.DOUBLE_BLINK
        ]
        assert run(GestureRecognizer(), [eyes(0.3, OPEN), *blink * 3, eyes(1.0, OPEN)]) == [
            Gesture.TRIPLE_BLINK
        ]

    def test_blinks_far_apart_are_not_a_double(self):
        frames = [eyes(0.12, CLOSED), eyes(1.0, OPEN), eyes(0.12, CLOSED), eyes(1.0, OPEN)]
        assert run(GestureRecognizer(), frames) == []

    def test_single_eye_camera_still_detects_blinks(self):
        rec = GestureRecognizer()
        got = run(rec, [(0.3, False, None, 0.5, 0.5), (1.0, True, None, None, None)])
        assert got == [Gesture.LONG_BLINK]


class TestWinks:
    def test_left_and_right_wink(self):
        left = (0.35, True, False, 0.5, 0.5)
        right = (0.35, False, True, 0.5, 0.5)
        assert run(GestureRecognizer(), [eyes(0.3, OPEN), left, eyes(0.5, OPEN)]) == [
            Gesture.WINK_LEFT
        ]
        assert run(GestureRecognizer(), [eyes(0.3, OPEN), right, eyes(0.5, OPEN)]) == [
            Gesture.WINK_RIGHT
        ]

    def test_long_wink(self):
        frames = [eyes(0.3, OPEN), (1.2, True, False, 0.5, 0.5), eyes(0.5, OPEN)]
        assert run(GestureRecognizer(), frames) == [Gesture.LONG_WINK_LEFT]

    def test_wink_that_becomes_a_blink_is_not_a_wink(self):
        frames = [
            eyes(0.3, OPEN),
            (0.1, True, False, 0.5, 0.5),
            eyes(0.25, CLOSED),
            (0.1, False, True, 0.5, 0.5),
            eyes(0.6, OPEN),
        ]
        got = run(GestureRecognizer(), frames)
        assert not {Gesture.WINK_LEFT, Gesture.WINK_RIGHT} & set(got)

    def test_no_winks_with_one_eye_unobserved(self):
        frames = [(0.3, False, None, 0.5, 0.5), (0.35, True, None, None, None)]
        assert Gesture.WINK_LEFT not in run(GestureRecognizer(), frames)


class TestGlances:
    def test_glance_left_and_back(self):
        frames = [eyes(0.3, OPEN), (0.3, False, False, -0.1, 0.5), eyes(0.5, OPEN)]
        assert run(GestureRecognizer(), frames) == [Gesture.GLANCE_LEFT]

    def test_looking_near_edge_is_not_a_glance(self):
        frames = [eyes(0.3, OPEN), (0.3, False, False, 0.995, 0.5), eyes(0.5, OPEN)]
        assert run(GestureRecognizer(), frames) == []

    def test_hold_repeats_and_suppresses_glance(self):
        frames = [eyes(0.3, OPEN), (2.0, False, False, 0.5, 1.2), eyes(0.5, OPEN)]
        got = run(GestureRecognizer(), frames)
        assert got[0] == Gesture.HOLD_DOWN
        assert len(got) >= 3 and set(got) == {Gesture.HOLD_DOWN}

    def test_hold_without_repeat(self):
        settings = GestureSettings.model_validate({"glance": {"hold_repeat_ms": None}})
        frames = [eyes(0.3, OPEN), (2.0, False, False, 0.5, -0.3), eyes(0.5, OPEN)]
        assert run(GestureRecognizer(settings), frames) == [Gesture.HOLD_UP]

    def test_closed_frames_do_not_end_a_glance_zone(self):
        rec = GestureRecognizer()
        run(rec, [(0.2, False, False, 1.2, 0.5)])
        assert rec.glance.zone == "right"


class TestDispatcher:
    class Backend:
        def __init__(self):
            self.calls = []

        def key_combo(self, combo):
            self.calls.append(("key", combo.modifiers, combo.key))

        def mouse(self, action, x, y):
            self.calls.append(("mouse", action, x, y))

    def make(self):
        backend = self.Backend()
        called = []
        d = ActionDispatcher(
            {
                Gesture.LONG_BLINK: Binding(mouse="click"),
                Gesture.WINK_LEFT: Binding(key="cmd+shift+tab"),
                Gesture.TRIPLE_BLINK: Binding(app="pause"),
            },
            backend,
            {"pause": lambda: called.append("pause")},
        )
        return d, backend, called

    def test_system_input_gated(self):
        d, backend, _ = self.make()
        r = d.dispatch(GestureEvent(Gesture.LONG_BLINK, 0.0), (10, 20))
        assert r.outcome == "suppressed" and backend.calls == []
        d.system_input_enabled = True
        d.dispatch(GestureEvent(Gesture.LONG_BLINK, 0.0), (10, 20))
        d.dispatch(GestureEvent(Gesture.WINK_LEFT, 0.0), None)
        assert backend.calls == [("mouse", "click", 10, 20), ("key", ("cmd", "shift"), "tab")]

    def test_app_actions_always_run(self):
        d, _, called = self.make()
        assert d.dispatch(GestureEvent(Gesture.TRIPLE_BLINK, 0.0), None).outcome == "done"
        assert called == ["pause"]

    def test_unbound(self):
        d, _, _ = self.make()
        assert d.dispatch(GestureEvent(Gesture.GLANCE_UP, 0.0), None).outcome == "unbound"

    def test_failing_backend_does_not_raise(self):
        d, backend, _ = self.make()
        d.system_input_enabled = True
        backend.mouse = lambda *a: 1 / 0
        assert d.dispatch(GestureEvent(Gesture.LONG_BLINK, 0.0), (1, 1)).outcome == "error"


class TestConfig:
    def test_key_combo_parsing(self):
        c = parse_key_combo("Command+Option+Left")
        assert c.modifiers == ("cmd", "alt") and c.key == "left"
        for bad in ("cmd+", "hyper+a", "cmd+nope"):
            with pytest.raises(ValueError):
                parse_key_combo(bad)

    def test_binding_needs_exactly_one_action(self):
        with pytest.raises(ValidationError):
            Binding()
        with pytest.raises(ValidationError):
            Binding(key="tab", app="pause")
        with pytest.raises(ValidationError):
            Binding(key="cmd+bogus")

    def test_unknown_gesture_name_rejected(self):
        with pytest.raises(ValidationError):
            GestureSettings.model_validate({"bindings": {"sneeze": {"app": "pause"}}})

    def test_timing_order_validated(self):
        with pytest.raises(ValidationError):
            GestureSettings.model_validate({"blink": {"long_blink_ms": 100}})

    def test_default_config_bindings_load(self):
        config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
        assert config.gestures.bindings[Gesture.LONG_BLINK] == Binding(mouse="click")
        assert len(config.gestures.bindings) > 5

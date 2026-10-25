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


_HOLDS = {Gesture.HOLD_WINK_LEFT, Gesture.HOLD_WINK_RIGHT}


def no_holds(gestures: list[Gesture]) -> list[Gesture]:
    return [g for g in gestures if g not in _HOLDS]


class TestWinks:
    def test_left_and_right_wink(self):
        left = (0.35, True, False, 0.5, 0.5)
        right = (0.35, False, True, 0.5, 0.5)
        assert no_holds(run(GestureRecognizer(), [eyes(0.3, OPEN), left, eyes(0.5, OPEN)])) == [
            Gesture.WINK_LEFT
        ]
        assert no_holds(run(GestureRecognizer(), [eyes(0.3, OPEN), right, eyes(0.5, OPEN)])) == [
            Gesture.WINK_RIGHT
        ]

    def test_long_wink(self):
        frames = [eyes(0.3, OPEN), (1.2, True, False, 0.5, 0.5), eyes(0.5, OPEN)]
        assert no_holds(run(GestureRecognizer(), frames)) == [Gesture.LONG_WINK_LEFT]

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

    def test_pause_progress_only_for_the_wink_bound_to_pause(self):
        rec = GestureRecognizer(GestureSettings(
            bindings={Gesture.LONG_WINK_RIGHT: Binding(app="pause")}
        ))
        rec.update(EyeSignal(0.0, False, False, 0.5, 0.5))
        rec.update(EyeSignal(0.1, False, True, None, None))  # right wink starts
        assert rec.update(EyeSignal(0.5, False, True, None, None)).pause_progress == (
            pytest.approx(0.5)
        )
        rec.update(EyeSignal(0.6, False, False, 0.5, 0.5))
        rec.update(EyeSignal(1.5, True, False, None, None))  # left wink: unbound
        frame = rec.update(EyeSignal(1.9, True, False, None, None))
        assert frame.pause_progress == 0.0 and frame.wink_progress > 0


def events(rec: GestureRecognizer, frames: list[tuple]) -> list[GestureEvent]:
    t = 0.0
    out: list[GestureEvent] = []
    for duration, left, right, gx, gy in frames:
        end = t + duration
        while t < end - 1e-9:
            out += rec.update(EyeSignal(t, left, right, gx, gy)).events
            t += DT
    return out


LEFT_SHUT = (True, False, 0.5, 0.5)
BOTH_SHUT = (True, True, None, None)


class TestLongWinkCooldown:
    def test_second_long_wink_waits_for_cooldown(self):
        wink = eyes(0.9, LEFT_SHUT)
        frames = [eyes(0.3, OPEN), wink, eyes(0.3, OPEN), wink, eyes(0.3, OPEN)]
        got = [e.gesture for e in events(GestureRecognizer(), frames)]
        assert got.count(Gesture.LONG_WINK_LEFT) == 1

    def test_long_wink_after_cooldown(self):
        wink = eyes(0.9, LEFT_SHUT)
        frames = [eyes(0.3, OPEN), wink, eyes(1.6, OPEN), wink, eyes(0.3, OPEN)]
        got = [e.gesture for e in events(GestureRecognizer(), frames)]
        assert got.count(Gesture.LONG_WINK_LEFT) == 2

    def test_wink_held_through_a_reset_does_not_fire_again(self):
        # Pausing resets the recognizer while the eye is still shut from the pausing wink.
        rec = GestureRecognizer()
        t, fired = 0.0, []
        while t < 6.0:
            closed = 0.3 <= t < 5.0
            frame = rec.update(EyeSignal(t, closed, False, 0.5, 0.5))
            for e in frame.events:
                if e.gesture is Gesture.LONG_WINK_LEFT:
                    fired.append(e)
                    rec.reset()
            t += DT
        assert len(fired) == 1


class TestWinkRunsAndHolds:
    def test_quick_winks_count_up_without_waiting(self):
        wink = [eyes(0.25, LEFT_SHUT), eyes(0.2, OPEN)]
        got = [e for e in events(GestureRecognizer(), [eyes(0.3, OPEN), *wink * 3])
               if e.gesture is Gesture.WINK_LEFT]
        assert [e.count for e in got] == [1, 2, 3]  # the cooldown doesn't block a run

    def test_slow_winks_start_new_runs(self):
        wink = [eyes(0.25, LEFT_SHUT), eyes(1.0, OPEN)]
        got = [e for e in events(GestureRecognizer(), [eyes(0.3, OPEN), *wink * 2])
               if e.gesture is Gesture.WINK_LEFT]
        assert [e.count for e in got] == [1, 1]

    def test_other_eye_breaks_a_run(self):
        frames = [eyes(0.3, OPEN), eyes(0.25, LEFT_SHUT), eyes(0.2, OPEN),
                  eyes(0.25, (False, True, 0.5, 0.5)), eyes(0.2, OPEN),
                  eyes(0.25, LEFT_SHUT), eyes(1.0, OPEN)]
        got = [(e.gesture, e.count) for e in events(GestureRecognizer(), frames)
               if e.gesture in (Gesture.WINK_LEFT, Gesture.WINK_RIGHT)]
        assert got == [(Gesture.WINK_LEFT, 1), (Gesture.WINK_LEFT, 1)]  # right was in cooldown

    def test_hold_starts_after_press_time_and_ends_on_reopen(self):
        got = events(GestureRecognizer(), [eyes(0.3, OPEN), eyes(0.6, LEFT_SHUT),
                                           eyes(0.5, OPEN)])
        holds = [(e.phase, round(e.t, 2)) for e in got if e.gesture is Gesture.HOLD_WINK_LEFT]
        assert [p for p, _ in holds] == ["start", "end"]
        assert holds[0][1] == pytest.approx(0.6, abs=0.04)  # 0.3 s into the wink
        assert holds[1][1] == pytest.approx(0.9, abs=0.04)

    def test_twitch_never_presses(self):
        got = events(GestureRecognizer(), [eyes(0.3, OPEN), eyes(0.15, LEFT_SHUT),
                                           eyes(0.5, OPEN)])
        assert not any(e.gesture is Gesture.HOLD_WINK_LEFT for e in got)

    def test_natural_blink_during_a_hold_keeps_it_and_isnt_a_double_blink(self):
        blink = [eyes(0.12, BOTH_SHUT), eyes(0.15, LEFT_SHUT)]
        frames = [eyes(0.3, OPEN), eyes(0.5, LEFT_SHUT), *blink * 2, eyes(0.4, LEFT_SHUT),
                  eyes(0.5, OPEN)]
        got = events(GestureRecognizer(), frames)
        holds = [e.phase for e in got if e.gesture is Gesture.HOLD_WINK_LEFT]
        assert holds == ["start", "end"]
        assert Gesture.DOUBLE_BLINK not in {e.gesture for e in got}

    def test_closing_both_eyes_for_long_ends_the_hold(self):
        frames = [eyes(0.3, OPEN), eyes(0.5, LEFT_SHUT), eyes(0.6, BOTH_SHUT)]
        holds = [e.phase for e in events(GestureRecognizer(), frames)
                 if e.gesture is Gesture.HOLD_WINK_LEFT]
        assert holds == ["start", "end"]

    def test_losing_an_eye_ends_the_hold(self):
        frames = [eyes(0.3, OPEN), eyes(0.5, LEFT_SHUT), (0.2, True, None, None, None)]
        holds = [e.phase for e in events(GestureRecognizer(), frames)
                 if e.gesture is Gesture.HOLD_WINK_LEFT]
        assert holds == ["start", "end"]


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

        def mouse(self, action, x, y, count=1):
            self.calls.append(("mouse", action, x, y) if count == 1 else
                              ("mouse", action, x, y, count))

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

    def test_hold_presses_then_releases_even_when_gated(self):
        backend = self.Backend()
        d = ActionDispatcher({Gesture.HOLD_WINK_LEFT: Binding(mouse="hold")}, backend)
        d.system_input_enabled = True
        d.dispatch(GestureEvent(Gesture.HOLD_WINK_LEFT, 0.0, phase="start"), (5, 6))
        d.system_input_enabled = False  # e.g. cursor control switched off mid-drag
        r = d.dispatch(GestureEvent(Gesture.HOLD_WINK_LEFT, 1.0, phase="end"), (50, 60))
        assert r.outcome == "done"
        assert backend.calls == [("mouse", "press", 5, 6), ("mouse", "release", 50, 60)]

    def test_hold_end_is_ignored_by_other_bindings(self):
        backend = self.Backend()
        d = ActionDispatcher({Gesture.HOLD_WINK_RIGHT: Binding(key="cmd+tab")}, backend)
        d.system_input_enabled = True
        d.dispatch(GestureEvent(Gesture.HOLD_WINK_RIGHT, 0.0, phase="start"), (1, 1))
        r = d.dispatch(GestureEvent(Gesture.HOLD_WINK_RIGHT, 1.0, phase="end"), (1, 1))
        assert r.outcome == "ignored" and len(backend.calls) == 1

    def test_repeated_clicks_count_and_snap_to_the_first(self):
        backend = self.Backend()
        d = ActionDispatcher({Gesture.WINK_RIGHT: Binding(mouse="right_click")}, backend)
        d.system_input_enabled = True
        d.dispatch(GestureEvent(Gesture.WINK_RIGHT, 0.0, count=1), (100, 100))
        d.dispatch(GestureEvent(Gesture.WINK_RIGHT, 0.4, count=2), (120, 110))  # drifted
        d.dispatch(GestureEvent(Gesture.WINK_RIGHT, 0.8, count=3), (400, 400))  # moved away
        assert backend.calls == [
            ("mouse", "right_click", 100, 100),
            ("mouse", "right_click", 100, 100, 2),
            ("mouse", "right_click", 400, 400),
        ]

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

    def test_hold_only_on_held_gestures(self):
        with pytest.raises(ValidationError):
            GestureSettings.model_validate({"bindings": {"long_blink": {"mouse": "hold"}}})
        GestureSettings.model_validate({"bindings": {"hold_wink_left": {"mouse": "hold"}}})

    def test_unknown_gesture_name_rejected(self):
        with pytest.raises(ValidationError):
            GestureSettings.model_validate({"bindings": {"sneeze": {"app": "pause"}}})

    def test_timing_order_validated(self):
        with pytest.raises(ValidationError):
            GestureSettings.model_validate({"blink": {"long_blink_ms": 100}})

    def test_default_config_bindings_load(self):
        config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
        assert config.gestures.bindings[Gesture.DOUBLE_BLINK] == Binding(mouse="click")
        assert Gesture.LONG_BLINK not in config.gestures.bindings  # no long-blink click
        assert len(config.gestures.bindings) > 5

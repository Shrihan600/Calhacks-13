"""Tests for filter, dwell, safety, and GazeDriver.

No camera, no display, no Quartz required — cursor is mocked.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from opengaze.control.dwell import DwellDetector, DwellState
from opengaze.control.filter import OneEuroFilter
from opengaze.control.safety import ConfidenceGate, EmergencyStop, JumpDetector
from opengaze.control.driver import GazeDriver
from opengaze.calibration.model import GazeModel
from opengaze.vision.observations import FeatureVector
from opengaze.vision.observations import FEATURE_LAYOUTS

_W = FEATURE_LAYOUTS["A-webcam"].width  # Mode A feature width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockCursor:
    def __init__(self, width=1920, height=1080):
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[tuple[int, int]] = []
        self._w = width
        self._h = height

    def move(self, x: int, y: int) -> None:
        self.moves.append((x, y))

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))

    def norm_to_px(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self._w), int(y * self._h)


def _fitted_model() -> GazeModel:
    from tests.test_calibration import GRID_9, _accepted_point
    pts = [_accepted_point(sx, sy) for sx, sy in GRID_9]
    X = np.stack([p.representative() for p in pts]).astype(np.float32)
    y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)
    model = GazeModel(schema_id="A-webcam")
    model.fit(X, y)
    return model


def _fv(x: float = 0.5, y: float = 0.5) -> FeatureVector:
    values = np.zeros(_W, dtype=np.float32)
    values[0] = x; values[1] = y
    values[3] = x; values[4] = y
    valid = np.ones(_W, dtype=bool)
    return FeatureVector(
        t_ns=time.time_ns(), values=values, valid=valid,
        schema_id="A-webcam", max_skew_ns=0,
        head_t_ns=None, left_eye_t_ns=None, right_eye_t_ns=None,
    )


# ---------------------------------------------------------------------------
# OneEuroFilter
# ---------------------------------------------------------------------------

class TestOneEuroFilter:
    def test_static_signal_converges(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        t = 0.0
        for _ in range(60):
            out = f(1.0, t)
            t += 1.0 / 30
        assert abs(out - 1.0) < 0.01

    def test_step_response_reaches_target(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
        t = 0.0
        for _ in range(30):
            f(0.0, t)
            t += 1.0 / 30
        for _ in range(60):
            out = f(1.0, t)
            t += 1.0 / 30
        assert out > 0.9

    def test_output_first_call_equals_input(self):
        f = OneEuroFilter()
        assert f(0.42, 0.0) == pytest.approx(0.42)

    def test_smoothed_less_than_raw_noise(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
        t = 0.0
        raw, smoothed = [], []
        for _ in range(100):
            x = 0.5 + rng.normal(0, 0.05)
            out = f(x, t)
            raw.append(x)
            smoothed.append(out)
            t += 1.0 / 30
        assert np.std(smoothed) < np.std(raw)

    def test_reset_forgets_state(self):
        f = OneEuroFilter()
        t = 0.0
        for _ in range(30):
            f(0.0, t); t += 1/30
        f.reset(value=1.0)
        out = f(1.0, t)
        assert abs(out - 1.0) < 0.01

    def test_duplicate_timestamps_dont_crash(self):
        f = OneEuroFilter()
        f(0.5, 1.0)
        out = f(0.5, 1.0)   # same timestamp
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# DwellDetector
# ---------------------------------------------------------------------------

class TestDwellDetector:
    def test_fires_after_dwell_time(self):
        d = DwellDetector(dwell_ms=100, radius_px=50, cooldown_ms=200)
        t = 0.0
        fired = False
        for _ in range(20):
            r = d.update(100, 100, t)
            if r.clicked:
                fired = True
            t += 0.01
        assert fired

    def test_resets_on_movement(self):
        d = DwellDetector(dwell_ms=500, radius_px=30, cooldown_ms=200)
        t = 0.0
        for _ in range(5):
            d.update(100, 100, t); t += 0.05
        # big jump — should reset
        d.update(500, 500, t); t += 0.05
        # progress should be near zero
        assert d.progress < 0.1

    def test_no_double_click_during_cooldown(self):
        d = DwellDetector(dwell_ms=50, radius_px=50, cooldown_ms=500)
        t = 0.0
        clicks = 0
        for _ in range(40):
            r = d.update(100, 100, t)
            if r.clicked:
                clicks += 1
            t += 0.01
        assert clicks == 1

    def test_disabled_never_clicks(self):
        d = DwellDetector(dwell_ms=10, radius_px=50, cooldown_ms=10, enabled=False)
        t = 0.0
        for _ in range(20):
            r = d.update(100, 100, t)
            assert not r.clicked
            t += 0.01

    def test_progress_zero_to_one(self):
        d = DwellDetector(dwell_ms=200, radius_px=50)
        t = 0.0
        progresses = []
        for _ in range(10):
            r = d.update(100, 100, t)
            progresses.append(r.progress)
            t += 0.03
        # Progress should reach 1.0 at some point (dwell fires at 200ms = ~7 steps)
        assert max(progresses) == pytest.approx(1.0)

    def test_state_transitions(self):
        d = DwellDetector(dwell_ms=100, radius_px=50, cooldown_ms=100)
        assert d.state == DwellState.IDLE
        d.update(100, 100, 0.0)
        assert d.state == DwellState.DWELLING
        t = 0.0
        while d.state == DwellState.DWELLING:
            d.update(100, 100, t); t += 0.02
        assert d.state == DwellState.COOLDOWN


# ---------------------------------------------------------------------------
# ConfidenceGate
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    def test_allows_above_threshold(self):
        g = ConfidenceGate(min_confidence=0.3)
        assert g.allow(0.31)
        assert g.allow(1.0)

    def test_blocks_below_threshold(self):
        g = ConfidenceGate(min_confidence=0.3)
        assert not g.allow(0.29)
        assert not g.allow(0.0)

    def test_exactly_threshold_passes(self):
        g = ConfidenceGate(min_confidence=0.5)
        assert g.allow(0.5)


# ---------------------------------------------------------------------------
# JumpDetector
# ---------------------------------------------------------------------------

class TestJumpDetector:
    def test_first_sample_always_passes(self):
        j = JumpDetector(max_jump_norm=0.1)
        assert j.allow(0.5, 0.5)

    def test_small_move_passes(self):
        j = JumpDetector(max_jump_norm=0.1)
        j.allow(0.5, 0.5)
        assert j.allow(0.55, 0.55)

    def test_large_jump_blocked(self):
        j = JumpDetector(max_jump_norm=0.1)
        j.allow(0.5, 0.5)
        assert not j.allow(0.9, 0.9)

    def test_jumps_counted(self):
        j = JumpDetector(max_jump_norm=0.05)
        j.allow(0.0, 0.0)          # first — always passes, _last = (0,0)
        j.allow(1.0, 1.0)          # jump — rejected, _last stays (0,0)
        j.allow(0.5, 0.5)          # jump from (0,0) — rejected, _last stays (0,0)
        assert j.jumps_detected == 2

    def test_reset_clears_history(self):
        j = JumpDetector(max_jump_norm=0.05)
        j.allow(0.0, 0.0)
        j.reset()
        assert j.allow(1.0, 1.0)  # first sample after reset always passes


# ---------------------------------------------------------------------------
# EmergencyStop
# ---------------------------------------------------------------------------

class TestEmergencyStop:
    def test_starts_inactive(self):
        e = EmergencyStop()
        assert not e.active

    def test_trigger_activates(self):
        e = EmergencyStop()
        e.trigger()
        assert e.active

    def test_reset_deactivates(self):
        e = EmergencyStop()
        e.trigger()
        e.reset()
        assert not e.active

    def test_toggle(self):
        e = EmergencyStop()
        assert e.toggle() is True
        assert e.active
        assert e.toggle() is False
        assert not e.active


# ---------------------------------------------------------------------------
# GazeDriver (full pipeline with mock cursor)
# ---------------------------------------------------------------------------

class TestGazeDriver:
    def _driver(self, **kw) -> tuple[GazeDriver, _MockCursor]:
        cursor = _MockCursor()
        model = _fitted_model()
        defaults = dict(dwell_ms=50, dwell_radius_px=200, dwell_cooldown_ms=100)
        defaults.update(kw)
        driver = GazeDriver(model, cursor, **defaults)
        return driver, cursor

    def test_push_moves_cursor(self):
        driver, cursor = self._driver()
        driver.push(_fv(0.5, 0.5))
        assert len(cursor.moves) == 1

    def test_low_confidence_drops_frame(self):
        driver, cursor = self._driver(min_confidence=0.9)
        fv = _fv(0.5, 0.5)
        # confidence = fraction of valid features; _fv has every feature valid → 1.0
        # So let's make most invalid
        values = np.zeros(_W, dtype=np.float32)
        valid = np.zeros(_W, dtype=bool)
        fv2 = FeatureVector(
            t_ns=0, values=values, valid=valid,
            schema_id="A-webcam", max_skew_ns=0,
            head_t_ns=None, left_eye_t_ns=None, right_eye_t_ns=None,
        )
        result = driver.push(fv2)
        assert result.drop_reason == "confidence"
        assert len(cursor.moves) == 0

    def test_paused_drops_all_frames(self):
        driver, cursor = self._driver()
        driver.pause()
        for _ in range(5):
            r = driver.push(_fv(0.5, 0.5))
            assert r.drop_reason == "paused"
        assert len(cursor.moves) == 0

    def test_toggle_pause_resumes(self):
        driver, cursor = self._driver()
        driver.pause()
        driver.toggle_pause()  # resume
        driver.push(_fv(0.5, 0.5))
        assert len(cursor.moves) == 1

    def test_dwell_click_fires(self):
        driver, cursor = self._driver(dwell_ms=50, dwell_radius_px=500)
        t_end = time.monotonic() + 0.3
        while time.monotonic() < t_end:
            driver.push(_fv(0.5, 0.5))
            time.sleep(0.01)
        assert len(cursor.clicks) >= 1

    def test_no_dwell_when_disabled(self):
        driver, cursor = self._driver(dwell_enabled=False, dwell_ms=10)
        for _ in range(20):
            driver.push(_fv(0.5, 0.5))
            time.sleep(0.005)
        assert len(cursor.clicks) == 0

    def test_frames_processed_counted(self):
        driver, cursor = self._driver()
        for _ in range(10):
            driver.push(_fv(0.5, 0.5))
        assert driver.frames_processed == 10


# ---------------------------------------------------------------------------
# QuartzCursorDriver event logic (Quartz faked: no display or permission needed)
# ---------------------------------------------------------------------------


class _FakeQuartz:
    kCGEventMouseMoved, kCGEventLeftMouseDragged = "moved", "dragged"
    kCGEventLeftMouseDown, kCGEventLeftMouseUp = "left_down", "left_up"
    kCGEventRightMouseDown, kCGEventRightMouseUp = "right_down", "right_up"
    kCGEventOtherMouseDown, kCGEventOtherMouseUp = "other_down", "other_up"
    kCGMouseButtonLeft, kCGMouseButtonRight, kCGMouseButtonCenter = 0, 1, 2
    kCGMouseEventClickState = "click_state"
    kCGHIDEventTap = "tap"

    def __init__(self):
        self.posted = []

    def CGEventCreateMouseEvent(self, _src, kind, pos, _button):  # noqa: N802
        return {"kind": kind, "pos": tuple(pos)}

    def CGEventSetIntegerValueField(self, event, _field, value):  # noqa: N802
        event["clicks"] = value

    def CGEventPost(self, _tap, event):  # noqa: N802
        self.posted.append(event)

    def CGEventCreate(self, _src):  # noqa: N802
        return None

    def CGEventGetLocation(self, _event):  # noqa: N802
        class _P:
            x, y = 77.0, 88.0
        return _P()


def _driver():
    from opengaze.control.cursor import QuartzCursorDriver, ScreenSize

    d = object.__new__(QuartzCursorDriver)
    d._Q = _FakeQuartz()
    d._screen = ScreenSize(1000, 800)
    d._origin = (0, 0)
    d._pressed = False
    return d


class TestQuartzDriverEvents:
    def test_hold_drags_then_releases(self):
        d = _driver()
        d.mouse("press", 10, 20)
        d.move(30, 40)
        d.mouse("release", 50, 60)
        d.move(70, 80)
        assert [(e["kind"], e["pos"]) for e in d._Q.posted] == [
            ("left_down", (10, 20)),
            ("dragged", (30, 40)),   # moves while held are drags
            ("left_up", (50, 60)),
            ("moved", (70, 80)),
        ]
        assert not d.pressed

    def test_release_is_safe_and_uses_cursor_position(self):
        d = _driver()
        d.release()  # nothing held: no event
        assert d._Q.posted == []
        d.press(1, 2)
        d.release()
        assert d._Q.posted[-1] == {"kind": "left_up", "pos": (77, 88)}

    def test_click_count_marks_double_clicks(self):
        d = _driver()
        d.mouse("click", 5, 5)
        d.mouse("click", 5, 5, count=2)
        d.mouse("right_click", 9, 9, count=2)
        assert [(e["kind"], e["clicks"]) for e in d._Q.posted] == [
            ("left_down", 1), ("left_up", 1),
            ("left_down", 2), ("left_up", 2),
            ("right_down", 2), ("right_up", 2),
        ]


# ---------------------------------------------------------------------------
# UserOverride (system-wide gaze yields to a physical mouse)
# ---------------------------------------------------------------------------

def test_override_allows_gaze_when_cursor_where_we_left_it():
    from opengaze.control.safety import UserOverride

    o = UserOverride(tolerance_px=20, hold_s=2.0)
    o.moved_to(100, 100)
    assert o.gaze_allowed((105, 98), now=10.0)


def test_override_holds_after_physical_mouse_move():
    from opengaze.control.safety import UserOverride

    o = UserOverride(tolerance_px=20, hold_s=2.0)
    o.moved_to(100, 100)
    assert not o.gaze_allowed((400, 300), now=10.0)
    assert not o.gaze_allowed((410, 300), now=11.5)
    assert o.gaze_allowed((410, 300), now=12.1)


def test_override_before_any_gaze_move_allows():
    from opengaze.control.safety import UserOverride

    assert UserOverride().gaze_allowed((0, 0), now=0.0)

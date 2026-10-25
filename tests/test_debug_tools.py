"""Debug tooling: drop reasons, traces, stage reports, simulated users, record/replay.

No camera, display or MediaPipe model needed (the run-window test uses Qt offscreen).
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np
import pytest

from opengaze.config.loader import check_hardware_profile, load_app_config
from opengaze.debug import puppet
from opengaze.debug.analyze import analyze
from opengaze.debug.harness import SimulatedUser, replay, simulate
from opengaze.debug.trace import TraceWriter, load_trace, parse_trace
from tests.helpers import HARDWARE_DIR, REPO_ROOT, camera_config


@pytest.fixture
def config():
    return load_app_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture
def headset():
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-headset.yaml")
    assert report.ok, report.errors
    return report.profile


# ---------------------------------------------------------------------------
# Hand-built ticks for the analyzer
# ---------------------------------------------------------------------------


def _tick(t, *, detected=True, closed=False, drop=None, raw=(0.5, 0.5), truth=None,
          cursor=None, fresh=True):
    rec = {
        "kind": "tick", "t": t,
        "cams": {"left_eye": {"seq": int(t * 30), "fresh": fresh, "age_ms": 5.0,
                              "status": "ok", "kind": "synthetic"}},
        "eyes": {"left": {"detected": detected, "closed": closed, "conf": 0.9}},
        "head": None,
        "feat": {"schema": "B-pupil2d", "values": [0.5] * 6, "valid": [True] * 6,
                 "skew_ms": 0.0},
        "raw": list(raw) if drop is None else None,
        "conf": 0.9, "drop": drop, "gaze": list(raw), "updated": drop is None,
        "gestures": [],
    }
    if truth is not None:
        rec["truth"] = list(truth)
    if cursor is not None:
        rec["cursor"] = cursor
    return rec


def _ticks(n=90, **kw):
    return [_tick(i / 30, **kw) for i in range(n)]


class TestAnalyzer:
    def test_healthy_trace_has_no_problems(self):
        report = analyze(_ticks(truth=(0.5, 0.5)), {"screen_px": [1440, 900]})
        assert report.first_problem is None, report.format()

    def test_missing_model_is_named_with_a_fix(self):
        report = analyze(_ticks(drop="no_model"))
        first = report.first_problem
        assert first is not None and first.name == "mapping"
        assert "calibrate" in first.findings[0]

    def test_pupil_detection_failure_comes_first(self):
        ticks = _ticks(detected=False, drop="no_prediction")
        first = analyze(ticks).first_problem
        assert first is not None and first.name == "eyes" and first.status == "fail"

    def test_accuracy_against_ground_truth(self):
        report = analyze(_ticks(raw=(0.3, 0.3), truth=(0.5, 0.5)), {"screen_px": [1440, 900]})
        gaze = report.stage("gaze")
        assert gaze.status == "fail"
        assert gaze.metrics["err_median"] == pytest.approx(0.283, abs=0.01)

    def test_cursor_without_backend(self):
        cur = {"enabled": True, "backend": False, "cmd": None, "actual": None,
               "blocked": "no_backend"}
        st = analyze(_ticks(cursor=cur)).stage("cursor")
        assert st.status == "fail" and "Accessibility" in st.findings[0]

    def test_cursor_that_doesnt_land(self):
        ticks = _ticks(cursor={"enabled": True, "backend": True, "cmd": [100, 100],
                               "actual": [700, 400], "blocked": None})
        st = analyze(ticks).stage("cursor")
        assert st.status == "fail" and st.metrics["follow_px"] > 600

    def test_cursor_left_off(self):
        cur = {"enabled": False, "backend": True, "cmd": None, "actual": None,
               "blocked": "cursor_off"}
        st = analyze(_ticks(cursor=cur)).stage("cursor")
        assert st.status == "warn" and "Tab" in st.findings[0]

    def test_camera_that_never_delivered(self):
        st = analyze(_ticks(), {"cameras": ["left_eye", "right_eye"]}).stage("cameras")
        assert st.status == "fail" and "right_eye" in st.findings[0]

    def test_jitter_ignores_saccades(self):
        rng = np.random.default_rng(0)
        ticks = []
        for i in range(300):
            target = (0.2, 0.2) if (i // 30) % 2 else (0.8, 0.7)  # a saccade every second
            raw = np.array(target) + rng.normal(0, 0.003, 2)
            ticks.append(_tick(i / 30, raw=tuple(raw)))
        jit = analyze(ticks, {"screen_px": [1000, 1000]}).stage("gaze").metrics["jitter_px"]
        assert 2.0 < jit < 8.0  # ~4.2 px RMS spread, not the 700 px saccades

    def test_format_mentions_every_stage(self):
        text = analyze(_ticks()).format()
        for name in ("cameras", "eyes", "head", "features", "mapping", "gaze", "cursor"):
            assert name in text


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


def test_trace_round_trip_survives_a_truncated_line(tmp_path):
    w = TraceWriter(tmp_path / "x.trace.jsonl", {"profile_id": "p"})
    w.tick(_tick(0.0))
    w.event(0.01, "cursor on")
    w.tick(_tick(0.033))
    w.close()
    with (tmp_path / "x.trace.jsonl").open("a") as f:
        f.write('{"kind": "tick", "t": 0.0')  # crash mid-write
    trace = load_trace(tmp_path / "x.trace.jsonl")
    assert trace.header["profile_id"] == "p"
    assert len(trace.ticks) == 2 and trace.events[0]["msg"] == "cursor on"
    assert parse_trace([json.dumps({"kind": "tick", "t": 1})]).ticks == [{"kind": "tick", "t": 1}]


# ---------------------------------------------------------------------------
# Gaze loop drop reasons
# ---------------------------------------------------------------------------


class _Tracker:
    def __init__(self, fv, closed=False):
        from opengaze.gaze.tracker import TrackerFrame

        self.frame = TrackerFrame(fv, None, None, None, closed, closed, None)

    def step(self, manager):
        return self.frame


class _Model:
    def __init__(self, confidence):
        self.c = confidence

    def predict(self, fv):
        from opengaze.calibration.model import GazeSample

        return GazeSample(0, 0.4, 0.6, self.c, "A-webcam", 0.4, 0.6)


@pytest.mark.parametrize(
    ("model", "fv", "closed", "reason"),
    [
        (None, object(), False, "no_model"),
        (_Model(1.0), None, False, "no_frames"),
        (_Model(1.0), object(), True, "eye_closed"),
        (_Model(0.1), object(), False, "low_confidence"),
        (_Model(1.0), object(), False, None),
    ],
)
def test_gaze_loop_says_why_gaze_did_not_move(model, fv, closed, reason):
    from opengaze.gaze.loop import GazeLoop

    loop = GazeLoop(_Tracker(fv, closed), model)
    state = loop.tick(None, 0.0)
    assert state.drop_reason == reason
    assert state.updated == (reason is None)
    if reason is None:
        assert state.raw == (0.4, 0.6)


def test_gaze_follows_the_open_eye_only_while_tracking_one_eye():
    from opengaze.gaze.loop import GazeLoop
    from opengaze.gaze.tracker import TrackerFrame

    class OneEyeShut(_Tracker):
        def __init__(self):
            self.frame = TrackerFrame(object(), None, None, None, True, False, None)

    loop = GazeLoop(OneEyeShut(), _Model(1.0))
    assert loop.tick(None, 0.0).drop_reason == "eye_closed"  # normally: frozen
    loop.track_one_eye = True  # a held wink is holding the button down
    state = loop.tick(None, 0.1)
    assert state.updated and state.raw == (0.4, 0.6)


# ---------------------------------------------------------------------------
# Puppet eyes
# ---------------------------------------------------------------------------


def test_puppet_eyes_look_where_told():
    from opengaze.cameras.simulated import SyntheticCameraSource

    src = SyntheticCameraSource(camera_config(
        "left_eye", options={"motion": "puppet", "realtime": False, "noise_std": 0.0}
    ))
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(src.connect())
        puppet.look_at(0.2, 0.7)
        truth = loop.run_until_complete(src.read_frame()).ground_truth
        assert (truth["gaze_x"], truth["gaze_y"]) == (0.2, 0.7)
        puppet.release()  # falls back to saccades
        truth = loop.run_until_complete(src.read_frame()).ground_truth
        assert (truth["gaze_x"], truth["gaze_y"]) != (0.2, 0.7)
    finally:
        puppet.release()
        loop.close()


def test_simulated_user_reaction_delay():
    user = SimulatedUser(reaction_s=0.2, jitter_norm=0.0)
    user.ask((0.9, 0.1), 1.0)
    assert user.eye(1.1) == (0.5, 0.5)
    assert user.eye(1.25) == (0.9, 0.1)


# ---------------------------------------------------------------------------
# Headless simulation and replay
# ---------------------------------------------------------------------------


def test_simulated_headset_drives_the_cursor_accurately(headset, config, tmp_path):
    res = simulate(headset, config, fixations=4, pursuit_s=3.0, trace_dir=tmp_path)
    assert res.calibration_points == 9
    assert not [s for s in res.report.stages if s.status == "fail"], res.report.format()
    gaze = res.report.stage("gaze")
    assert gaze.metrics["err_median"] < 0.03
    assert res.report.stage("cursor").metrics["moved"] > 0.9
    assert puppet.target() is None


def test_simulation_finds_an_injected_camera_delay(headset, config, tmp_path):
    left = headset.cameras["left_eye"]
    slow = left.model_copy(update={"options": {**left.options, "latency_ms": 250}})
    profile = headset.model_copy(update={"cameras": {**headset.cameras, "left_eye": slow}})
    res = simulate(profile, config, fixations=3, pursuit_s=0.0, trace_dir=tmp_path)
    first = res.report.first_problem
    assert first is not None and first.name == "cameras"
    assert "left_eye" in first.findings[0]


def test_record_and_replay_reproduce_the_session(headset, config, tmp_path):
    session = tmp_path / "session"
    live = simulate(headset, config, fixations=3, pursuit_s=0.0, record=session)
    assert (session / "profile.yaml").exists() and (session / "calibration.npz").exists()
    assert sorted(p.name for p in session.glob("*.avi")) == [
        "head.avi", "left_eye.avi", "right_eye.avi"
    ]
    result = replay(session, config)
    assert result.original is not None
    n_live = live.report.n_ticks
    assert result.report.n_ticks == n_live
    live_err = live.report.stage("gaze").metrics["err_median"]
    replay_err = result.report.stage("gaze").metrics["err_median"]
    assert replay_err == pytest.approx(live_err, abs=0.01)


# ---------------------------------------------------------------------------
# Run window: debug overlay + trace, offscreen
# ---------------------------------------------------------------------------


def test_run_window_traces_and_draws_debug_overlay(headset, config, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from opengaze.cameras.manager import CameraManager
    from opengaze.cameras.runtime import CaptureRuntime
    from opengaze.debug import trace as trace_mod
    from opengaze.desktop import run_ui

    monkeypatch.setattr(trace_mod, "TRACE_DIR", tmp_path)
    problem = "Accessibility permission not granted (test)"
    monkeypatch.setattr(run_ui, "_try_make_cursor", lambda: (None, problem))  # as in CI
    app = QApplication.instance() or QApplication([])
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-headset.yaml")
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    try:
        win = run_ui.GazeTestWindow(runtime, report, config, "C-eye3d", trace=True)
        win.show()
        app.processEvents()  # the trace starts once the window is up
        win._toggle_debug()
        assert win._cursor_problem == problem
        win._toggle_cursor()  # refused: no driver, so it stays off and flashes the error
        assert not win._cursor_enabled
        assert "error" in win._status
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        win.repaint()
        assert win._trace is not None and win._trace.n_ticks > 10
        path = win._trace.path
        win.close()
    finally:
        runtime.stop()
    trace = load_trace(path)
    assert trace.header["profile_id"] == "synthetic-headset"
    blocked = {t["cursor"]["blocked"] for t in trace.ticks}
    assert "no_backend" in blocked
    assert any(t.get("truth") for t in trace.ticks)  # puppet eyes report where they look


def test_cursor_problem_explains_the_fix(monkeypatch):
    pytest.importorskip("PySide6")
    from opengaze.control import cursor
    from opengaze.desktop import run_ui

    monkeypatch.setattr(cursor, "is_accessibility_granted", lambda: False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    driver, problem = run_ui._try_make_cursor()
    assert driver is None
    assert "Accessibility" in problem and "Visual Studio Code" in problem

"""Tests for adaptive (test-or-train) calibration and its saved logs. No camera or display."""

from __future__ import annotations

import numpy as np
import pytest

from opengaze.calibration.adaptive import (
    SEED_TARGETS,
    AdaptiveCalibrationSession,
    AdaptiveLog,
    list_logs,
    points_needed_curve,
)
from opengaze.calibration.model import GazeModel, ScreenMapper
from opengaze.calibration.session import CalibrationPoint
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureVector

_WIDTH = FEATURE_LAYOUTS["A-webcam"].width


def _fv(x: float, y: float, t_ns: int = 0) -> FeatureVector:
    values = np.zeros(_WIDTH, dtype=np.float32)
    values[[0, 3]] = x
    values[[1, 4]] = y
    return FeatureVector(
        t_ns=t_ns, values=values, valid=np.ones(_WIDTH, dtype=bool), schema_id="A-webcam",
        max_skew_ns=0, head_t_ns=None, left_eye_t_ns=None, right_eye_t_ns=None,
    )


def _collect(sx: float, sy: float, warp=lambda x, y: (x, y), n: int = 15) -> CalibrationPoint:
    rng = np.random.default_rng(int(sx * 1000 + sy * 10))
    pt = CalibrationPoint(sx, sy)
    fx, fy = warp(sx, sy)
    pt.samples = [_fv(fx + e[0], fy + e[1], i) for i, e in enumerate(rng.normal(0, 0.002, (n, 2)))]
    return pt


def _run(session: AdaptiveCalibrationSession, warp=lambda x, y: (x, y)) -> None:
    for _ in range(200):
        if session.done:
            return
        session.submit(_collect(*session.next_target(), warp=warp))
    raise AssertionError("session never finished")


def _ridge() -> ScreenMapper:
    return GazeModel(schema_id="A-webcam", alpha=0.01)


class _Broken(GazeModel):
    """Always predicts the top-left corner."""

    def predict_xy(self, x, valid=None):
        return 0.0, 0.0


class TestAdaptiveSession:
    def test_seed_targets_come_first(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge)
        for target in SEED_TARGETS:
            assert s.next_target() == target
            rec = s.submit(_collect(*target))
            assert rec.phase == "seed" and rec.action == "train"
        assert s.model is not None

    def test_linear_features_converge_after_seed(self):
        s = AdaptiveCalibrationSession(
            "A-webcam", mapper_factory=_ridge, required_passes=4, screen_px=(1000, 800)
        )
        _run(s)
        assert s.converged
        assert len(s.training) == len(SEED_TARGETS)
        tests = [r for r in s.log.records if r.phase == "test"]
        assert len(tests) == 4 and all(r.action == "pass" for r in tests)
        result = s.result()
        assert result.converged
        assert result.final_error_norm is not None and result.final_error_norm < 0.02
        assert result.final_error_px is not None

    def test_failed_tests_become_training(self):
        s = AdaptiveCalibrationSession(
            "A-webcam", mapper_factory=lambda: _Broken("A-webcam"), max_points=12
        )
        _run(s)
        assert not s.converged
        assert s.n_accepted == 12
        assert len(s.training) == 12
        assert all(r.action == "train" for r in s.log.records)

    def test_pass_and_train_follow_tolerance(self):
        """A non-linear feature warp: the linear model fails somewhere and must add points."""
        warp = lambda x, y: (x, y + 2 * (x - 0.5) ** 2)  # noqa: E731
        s = AdaptiveCalibrationSession(
            "A-webcam", mapper_factory=_ridge, tolerance_norm=0.03, max_points=25
        )
        _run(s, warp)
        tests = [r for r in s.log.records if r.phase == "test"]
        assert any(r.action == "train" for r in tests)
        for r in tests:
            assert r.error_norm is not None
            assert (r.error_norm <= 0.03) == (r.action == "pass")

    def test_too_few_samples_rejected_and_retried(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge, min_samples_per_point=10)
        target = s.next_target()
        rec = s.submit(_collect(*target, n=3))
        assert rec.action == "reject"
        assert s.next_target() == target
        assert s.n_accepted == 0
        assert s.log.reps[0] is None

    def test_adaptive_targets_never_repeat(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=lambda: _Broken("A-webcam"))
        _run(s)
        targets = [(r.target_x, r.target_y) for r in s.log.records]
        assert len(set(targets)) == len(targets)

    def test_first_adaptive_target_fills_a_gap(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge)
        for t in SEED_TARGETS:
            s.submit(_collect(*t))
        x, y = s.next_target()
        # Farthest from centre and corners: an edge midpoint.
        assert (round(x, 3), round(y, 3)) in {(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)}

    def test_refeature_updates_log(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge)
        for t in SEED_TARGETS:
            s.submit(_collect(*t))
        s.refeature(lambda p: [_fv(p.screen_x_norm, p.screen_y_norm, 99)] * 5)
        assert all(len(group) == 5 for group in s.log.samples)
        assert s.log.reps[0] is not None
        assert float(s.log.reps[0][0]) == pytest.approx(0.5)

    def test_result_without_model_raises(self):
        s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge)
        with pytest.raises(ValueError):
            s.result()


class TestAdaptiveLog:
    def _finished(self) -> AdaptiveCalibrationSession:
        s = AdaptiveCalibrationSession(
            "A-webcam", mapper_factory=_ridge, screen_px=(1440, 900), max_points=10
        )
        s.submit(_collect(*s.next_target(), n=2))  # one rejected attempt
        _run(s)
        return s

    def test_round_trip(self, tmp_path):
        s = self._finished()
        s.log.profile_fingerprint = "abc123"
        path = s.log.save(tmp_path)
        assert list_logs(tmp_path) == [path]
        loaded = AdaptiveLog.load(path)
        assert loaded.records == s.log.records
        assert loaded.screen_px == (1440, 900)
        assert loaded.settings == s.log.settings
        assert loaded.profile_fingerprint == "abc123"
        assert loaded.reps[0] is None
        for a, b in zip(loaded.reps[1:], s.log.reps[1:], strict=True):
            np.testing.assert_allclose(a, b)
        assert [len(g) for g in loaded.samples] == [len(g) for g in s.log.samples]
        assert "rejected" in loaded.summary()

    def test_points_needed_curve(self, tmp_path):
        s = self._finished()
        curve = points_needed_curve(AdaptiveLog.load(s.log.save(tmp_path)), _ridge)
        assert curve
        assert [c.n_training for c in curve] == sorted(c.n_training for c in curve)
        assert all(c.n_eval > 0 for c in curve)
        assert curve[-1].mean_error_norm < 0.05
        assert curve[-1].mean_error_px is not None

    def test_curve_default_mapper(self):
        curve = points_needed_curve(self._finished().log)
        assert curve


def test_window_runs_to_result_and_saves(tmp_path, monkeypatch):
    """Drive the adaptive wizard offscreen; the simulated headset's eyes follow the dots."""
    import os
    import time

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from opengaze.calibration.storage import CalibrationStore
    from opengaze.cameras.manager import CameraManager
    from opengaze.cameras.runtime import CaptureRuntime
    from opengaze.config.loader import check_hardware_profile, load_app_config
    from opengaze.desktop import adaptive_calibration_ui as ui
    from tests.helpers import HARDWARE_DIR, REPO_ROOT

    monkeypatch.setattr(ui, "_SETTLE_S", 0.0)
    monkeypatch.setattr(ui, "_COLLECT_S", 0.3)
    monkeypatch.setattr(ui, "_FLASH_S", 0.01)
    app = QApplication.instance() or QApplication([])
    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-headset.yaml")
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    store = CalibrationStore(tmp_path / "calib")
    try:
        win = ui.AdaptiveCalibrationWindow(
            runtime, report, config, store, owns_runtime=False, log_dir=tmp_path / "logs"
        )
        win.show()
        # The intro screen holds until the 3D eye model has fitted from the wandering eyes.
        deadline = time.monotonic() + 20.0
        while win._asession is None and time.monotonic() < deadline:
            app.processEvents()
            win._tick()
            win._on_advance()
            time.sleep(0.01)
        assert win._asession is not None, "3D eye model never became ready"
        win._asession.max_points = 12
        win._asession.min_samples_per_point = 3
        deadline = time.monotonic() + 20.0
        while win._astate != ui._AState.RESULT and time.monotonic() < deadline:
            app.processEvents()
            win._tick()
            time.sleep(0.01)
        assert win._astate == ui._AState.RESULT, win._error
        win.repaint()
        assert win._log_path is not None and win._log_path.exists()
        log = AdaptiveLog.load(win._log_path)
        assert log.n_training >= len(SEED_TARGETS)
        assert win._aresult is not None and win._aresult.converged, log.summary()
        win._on_advance()  # save
        assert win._saved
        assert store.list_all()
        win.close()
    finally:
        runtime.stop()


def test_curve_empty_when_every_point_rejected():
    s = AdaptiveCalibrationSession("A-webcam", mapper_factory=_ridge)
    s.submit(_collect(*s.next_target(), n=2))
    assert points_needed_curve(s.log, _ridge) == []

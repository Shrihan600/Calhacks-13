"""Tests for the calibration model, session, validation, and storage.

All tests run without a camera, display, or MediaPipe model.
FeatureVectors are built by hand from synthetic data.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from opengaze.calibration.model import GazeModel
from opengaze.calibration.session import CalibrationPoint, CalibrationSession, GRID_9
from opengaze.calibration.storage import CalibrationStore
from opengaze.calibration.validation import validate_calibration
from opengaze.vision.observations import FeatureVector
from opengaze.vision.observations import FEATURE_LAYOUTS

_W = FEATURE_LAYOUTS["A-webcam"].width  # Mode A feature width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fv(x: float, y: float, t_ns: int = 0, schema: str = "A-webcam") -> FeatureVector:
    """Minimal FeatureVector: left pupil (x, y), rest zeroed but valid."""
    values = np.zeros(_W, dtype=np.float32)
    values[0] = x   # left_pupil_x
    values[1] = y   # left_pupil_y
    values[3] = x   # right_pupil_x (same as left for simplicity)
    values[4] = y
    valid = np.ones(_W, dtype=bool)
    return FeatureVector(
        t_ns=t_ns,
        values=values,
        valid=valid,
        schema_id=schema,  # type: ignore[arg-type]
        max_skew_ns=0,
        head_t_ns=None,
        left_eye_t_ns=None,
        right_eye_t_ns=None,
    )


def _make_samples(sx: float, sy: float, n: int = 20) -> list[FeatureVector]:
    """n samples with slight noise around (sx, sy)."""
    rng = np.random.default_rng(42)
    out = []
    for i in range(n):
        noise = rng.normal(0, 0.01, 2)
        out.append(_fv(sx + noise[0], sy + noise[1], t_ns=i))
    return out


def _accepted_point(sx: float, sy: float, n: int = 20) -> CalibrationPoint:
    pt = CalibrationPoint(sx, sy)
    pt.samples = _make_samples(sx, sy, n)
    pt.accepted = True
    return pt


# ---------------------------------------------------------------------------
# GazeModel
# ---------------------------------------------------------------------------

class TestGazeModel:
    def test_fit_and_predict_identity(self):
        """With perfectly separable data the model should recover near-perfect mapping."""
        targets = GRID_9
        pts = [_accepted_point(sx, sy) for sx, sy in targets]
        X = np.stack([p.representative() for p in pts]).astype(np.float32)
        y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)

        model = GazeModel(schema_id="A-webcam", alpha=0.01)
        model.fit(X, y)

        for p in pts:
            rep = p.representative()
            assert rep is not None
            px, py = model.predict_raw(rep)
            assert abs(px - p.screen_x_norm) < 0.05
            assert abs(py - p.screen_y_norm) < 0.05

    def test_schema_id_mismatch_raises(self):
        pts = [_accepted_point(sx, sy) for sx, sy in GRID_9]
        X = np.stack([p.representative() for p in pts]).astype(np.float32)
        y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)
        model = GazeModel(schema_id="A-webcam")
        model.fit(X, y)

        wrong_fv = _fv(0.5, 0.5, schema="B-single-eye")
        with pytest.raises(ValueError, match="A-webcam"):
            model.predict(wrong_fv)

    def test_predict_unfitted_raises(self):
        model = GazeModel(schema_id="A-webcam")
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_raw(np.zeros(_W, dtype=np.float32))

    def test_output_clamped_to_unit_square(self):
        pts = [_accepted_point(sx, sy) for sx, sy in GRID_9]
        X = np.stack([p.representative() for p in pts]).astype(np.float32)
        y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)
        model = GazeModel(schema_id="A-webcam")
        model.fit(X, y)

        extreme = np.full(_W, 99.0, dtype=np.float32)
        px, py = model.predict_raw(extreme)
        assert 0.0 <= px <= 1.0
        assert 0.0 <= py <= 1.0

    def test_roundtrip_serialisation(self):
        pts = [_accepted_point(sx, sy) for sx, sy in GRID_9]
        X = np.stack([p.representative() for p in pts]).astype(np.float32)
        y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)
        model = GazeModel(schema_id="A-webcam", alpha=2.0)
        model.fit(X, y)

        arrays = model.to_arrays()
        model2 = GazeModel.from_arrays(arrays)

        assert model2.schema_id == model.schema_id
        for p in pts[:3]:
            rep = p.representative()
            assert rep is not None
            px1, py1 = model.predict_raw(rep)
            px2, py2 = model2.predict_raw(rep)
            assert abs(px1 - px2) < 1e-6
            assert abs(py1 - py2) < 1e-6

    def test_need_at_least_2_samples(self):
        X = np.zeros((1, 12), dtype=np.float32)
        y = np.zeros((1, 2), dtype=np.float32)
        model = GazeModel(schema_id="A-webcam")
        with pytest.raises(ValueError, match="2 samples"):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------------------

class TestCalibrationSession:
    def test_schema_id_stored(self):
        session = CalibrationSession(schema_id="A-webcam")
        assert session.schema_id == "A-webcam"

    def test_n_targets_matches_grid(self):
        session = CalibrationSession(schema_id="A-webcam")
        assert session.n_targets == 9

    def test_wrong_schema_samples_skipped(self):
        session = CalibrationSession(schema_id="A-webcam", collect_duration_s=100.0)
        session.start_point(0)
        session.add_sample(_fv(0.5, 0.5, schema="B-single-eye"))
        assert session.points[0].n_samples == 0

    def test_fit_recovers_targets(self):
        session = CalibrationSession(schema_id="A-webcam", min_samples_per_point=5)
        for i, (sx, sy) in enumerate(GRID_9):
            pt = session.points[i]
            pt.samples = _make_samples(sx, sy, 20)
            pt.accepted = True

        result = session.fit(screen_width_px=1920, screen_height_px=1080)
        assert result.model is not None
        assert result.schema_id == "A-webcam"
        assert result.mean_train_error_norm < 0.1
        assert result.mean_train_error_px is not None

    def test_fit_fails_with_too_few_accepted(self):
        session = CalibrationSession(schema_id="A-webcam")
        # Accept only 3 points (need 4)
        for i in range(3):
            session.points[i].samples = _make_samples(GRID_9[i][0], GRID_9[i][1])
            session.points[i].accepted = True
        with pytest.raises(ValueError, match="4"):
            session.fit()

    def test_accepted_count(self):
        session = CalibrationSession(schema_id="A-webcam")
        session.points[0].accepted = True
        session.points[2].accepted = True
        assert session.n_accepted == 2


# ---------------------------------------------------------------------------
# CalibrationPoint
# ---------------------------------------------------------------------------

class TestCalibrationPoint:
    def test_representative_none_when_too_few_samples(self):
        pt = CalibrationPoint(0.5, 0.5)
        pt.samples = _make_samples(0.5, 0.5, n=2)
        assert pt.representative() is None

    def test_representative_median(self):
        pt = _accepted_point(0.3, 0.7, n=30)
        rep = pt.representative()
        assert rep is not None
        assert rep.dtype == np.float32
        assert abs(float(rep[0]) - 0.3) < 0.05  # left_pupil_x ≈ 0.3

    def test_mean_confidence_full(self):
        pt = _accepted_point(0.5, 0.5)
        assert pt.mean_confidence() == pytest.approx(1.0)

    def test_mean_confidence_empty(self):
        pt = CalibrationPoint(0.5, 0.5)
        assert pt.mean_confidence() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _nine_points(self) -> list[CalibrationPoint]:
        return [_accepted_point(sx, sy) for sx, sy in GRID_9]

    def test_passes_on_clean_data(self):
        points = self._nine_points()
        report = validate_calibration(points, schema_id="A-webcam", pass_threshold_norm=0.15)
        assert report.passed
        assert report.mean_error_norm < 0.15

    def test_fails_with_high_threshold(self):
        points = self._nine_points()
        report = validate_calibration(points, schema_id="A-webcam", pass_threshold_norm=0.0001)
        assert not report.passed

    def test_needs_five_points(self):
        points = [_accepted_point(sx, sy) for sx, sy in GRID_9[:4]]
        with pytest.raises(ValueError, match="5"):
            validate_calibration(points, schema_id="A-webcam")

    def test_pixel_errors_present_when_size_given(self):
        points = self._nine_points()
        report = validate_calibration(
            points, schema_id="A-webcam",
            screen_width_px=1920, screen_height_px=1080
        )
        assert report.mean_error_px is not None
        assert report.max_error_px is not None
        assert report.mean_error_px >= 0

    def test_n_point_errors_equals_n_accepted(self):
        points = self._nine_points()
        report = validate_calibration(points, schema_id="A-webcam")
        assert len(report.point_errors) == len(points)

    def test_summary_contains_pass_or_fail(self):
        points = self._nine_points()
        report = validate_calibration(points, schema_id="A-webcam")
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TestCalibrationStorage:
    def _fitted_model(self) -> GazeModel:
        pts = [_accepted_point(sx, sy) for sx, sy in GRID_9]
        X = np.stack([p.representative() for p in pts]).astype(np.float32)
        y = np.array([[p.screen_x_norm, p.screen_y_norm] for p in pts], dtype=np.float32)
        model = GazeModel(schema_id="A-webcam")
        model.fit(X, y)
        return model

    def _profile(self):
        from tests.helpers import camera_config
        from opengaze.config.schema import HardwareProfile
        cam = camera_config(source_type="synthetic", role="head", camera_id="head")
        return HardwareProfile(id="test", cameras={"head": cam})

    def test_save_and_load(self, tmp_path: Path):
        store = CalibrationStore(store_dir=tmp_path)
        model = self._fitted_model()
        profile = self._profile()

        path = store.save(model, profile)
        assert path.exists()

        loaded = store.load("A-webcam", profile)
        assert loaded.schema_id == "A-webcam"
        assert loaded._fitted

    def test_exists_true_after_save(self, tmp_path: Path):
        store = CalibrationStore(store_dir=tmp_path)
        model = self._fitted_model()
        profile = self._profile()
        assert not store.exists("A-webcam", profile)
        store.save(model, profile)
        assert store.exists("A-webcam", profile)

    def test_load_missing_raises(self, tmp_path: Path):
        store = CalibrationStore(store_dir=tmp_path)
        profile = self._profile()
        with pytest.raises(FileNotFoundError):
            store.load("A-webcam", profile)

    def test_delete(self, tmp_path: Path):
        store = CalibrationStore(store_dir=tmp_path)
        model = self._fitted_model()
        profile = self._profile()
        store.save(model, profile)
        assert store.delete("A-webcam", profile)
        assert not store.exists("A-webcam", profile)

    def test_predictions_survive_roundtrip(self, tmp_path: Path):
        store = CalibrationStore(store_dir=tmp_path)
        model = self._fitted_model()
        profile = self._profile()
        store.save(model, profile)
        loaded = store.load("A-webcam", profile)

        fv = _fv(0.5, 0.5)
        px1, py1 = model.predict_raw(fv.values)
        px2, py2 = loaded.predict_raw(fv.values)
        assert abs(px1 - px2) < 1e-5
        assert abs(py1 - py2) < 1e-5

    def test_incompatible_profile_raises(self, tmp_path: Path):
        from opengaze.config.schema import HardwareProfile
        from tests.helpers import camera_config

        store = CalibrationStore(store_dir=tmp_path)
        model = self._fitted_model()
        profile = self._profile()
        store.save(model, profile)

        # Different resolution → different geometry fingerprint
        cam2 = camera_config(source_type="synthetic", role="head", camera_id="head",
                              resolution=(640, 480))
        profile2 = HardwareProfile(id="test", cameras={"head": cam2})
        with pytest.raises((ValueError, FileNotFoundError)):
            store.load("A-webcam", profile2)

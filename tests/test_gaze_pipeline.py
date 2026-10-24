"""Phase 5 gaze estimation: 2D detector, 3D eye model, pipelines, mappers, tracker, storage."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from opengaze.calibration.mappers import (
    GeometricRayMapper,
    PolynomialRidgeMapper,
    head_rotation,
    kabsch,
    make_mapper,
    mapper_from_arrays,
)
from opengaze.calibration.model import GazeModel
from opengaze.calibration.session import GRID_9, CalibrationSession
from opengaze.calibration.storage import CalibrationStore
from opengaze.calibration.validation import validate_calibration
from opengaze.cameras.simulated import render_eye, skin_background, spherical_gaze_vector
from opengaze.cameras.types import CameraFrame, CameraRole, PixelFormat, SourceKind, TimestampSource
from opengaze.config.loader import check_profile_data
from opengaze.config.schema import AppConfig
from opengaze.gaze.detector2d import EllipseFitPupilDetector, make_pupil_detector
from opengaze.gaze.eye_model3d import SphereFitEyeModel, make_eye_model
from opengaze.gaze.tracker import GazeTracker
from opengaze.modes import TrackingMode, schema_for
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureVector

BG = skin_background(320, 240, 0.25)


def noisy_eye(gaze, closure=0.0, spherical=False, rng=None):
    img, truth = render_eye(BG, gaze, closure, 0.08, spherical=spherical)
    if rng is not None:
        img = np.clip(img + rng.standard_normal(img.shape) * 4, 0, 255).astype(np.uint8)
    return img, truth


# ---------------------------------------------------------------------------
# 2D detector
# ---------------------------------------------------------------------------


class TestEllipseFitDetector:
    def test_finds_pupil_centre(self):
        rng = np.random.default_rng(0)
        det = EllipseFitPupilDetector()
        for gaze in [(0.5, 0.5), (0.05, 0.1), (0.95, 0.9), (0.2, 0.8)]:
            img, truth = noisy_eye(gaze, rng=rng)
            d = det.detect(img)
            assert d is not None and d.confidence > 0.6
            assert math.hypot(d.center_x - truth["pupil_x"], d.center_y - truth["pupil_y"]) < 1.5

    def test_measures_foreshortening(self):
        img, truth = noisy_eye((0.95, 0.5), spherical=True)
        d = EllipseFitPupilDetector().detect(img)
        assert d is not None
        assert d.axis_ratio == pytest.approx(abs(truth["gaze_vec_z"]), abs=0.08)
        # minor axis is radial: horizontal for a sideways look
        assert min(d.minor_axis_angle_deg, 180 - d.minor_axis_angle_deg) < 10

    def test_closed_eye_has_no_pupil(self):
        img, _ = noisy_eye((0.5, 0.5), closure=1.0)
        assert EllipseFitPupilDetector().detect(img) is None

    def test_accepts_bgr(self):
        import cv2

        img, _ = noisy_eye((0.4, 0.6))
        assert EllipseFitPupilDetector().detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) is not None

    def test_registry(self):
        assert isinstance(make_pupil_detector("ellipse_fit"), EllipseFitPupilDetector)
        with pytest.raises(ValueError, match="unknown 2D pupil detector"):
            make_pupil_detector("pupil_labs_2d")


# ---------------------------------------------------------------------------
# 3D eye model
# ---------------------------------------------------------------------------


def _warm_model(n=200, seed=0) -> SphereFitEyeModel:
    rng = np.random.default_rng(seed)
    det, model = EllipseFitPupilDetector(), SphereFitEyeModel()
    for k in range(n):
        img, _ = noisy_eye(tuple(rng.uniform(0.02, 0.98, 2)), spherical=True, rng=rng)
        d = det.detect(img)
        if d is not None:
            model.update(d, k)
    return model


class TestSphereFitModel:
    def test_unfitted_until_enough_observations(self):
        model = SphereFitEyeModel()
        img, _ = noisy_eye((0.9, 0.5), spherical=True)
        d = EllipseFitPupilDetector().detect(img)
        assert model.update(d, 0) is None
        assert not model.fitted

    def test_recovers_eyeball_and_gaze(self):
        model = _warm_model()
        assert model.fitted
        center = model.state_arrays()["center"]
        assert np.allclose(center, [160, 120], atol=3)
        rng = np.random.default_rng(5)
        det = EllipseFitPupilDetector()
        errors = []
        for _ in range(40):
            gaze = tuple(rng.uniform(0.05, 0.95, 2))
            img, _ = noisy_eye(gaze, spherical=True, rng=rng)
            g = model.project(det.detect(img))
            assert g is not None
            cos = float(g.direction @ spherical_gaze_vector(gaze))
            errors.append(math.degrees(math.acos(min(1.0, cos))))
        assert np.median(errors) < 2.0
        assert max(errors) < 5.0

    def test_frozen_model_ignores_new_data_and_round_trips(self):
        model = _warm_model()
        model.freeze()
        before = model.state_arrays()
        img, _ = noisy_eye((0.5, 0.5), spherical=True)
        for k in range(50):
            model.update(EllipseFitPupilDetector().detect(img), k)
        after = model.state_arrays()
        assert np.array_equal(before["center"], after["center"])

        restored = make_eye_model("sphere_fit")
        restored.load_state_arrays(after)
        assert restored.fitted

    def test_registry(self):
        with pytest.raises(ValueError, match="unknown 3D eye model"):
            make_eye_model("pye3d")


# ---------------------------------------------------------------------------
# Tracker + calibration end to end on rendered eye images
# ---------------------------------------------------------------------------


class FakeManager:
    """Serves rendered eye images looking at whatever ``target`` is set to."""

    def __init__(self, eye_ids, spherical: bool, seed: int = 0) -> None:
        self.eye_ids = eye_ids
        self.spherical = spherical
        self.target = (0.5, 0.5)
        self.closure = 0.0
        self.rng = np.random.default_rng(seed)
        self.seq = 0

    def latest(self, cid):
        if cid not in self.eye_ids:
            return None
        self.seq += 1
        img, truth = noisy_eye(self.target, self.closure, self.spherical, self.rng)
        return CameraFrame(
            image=img,
            camera_id=cid,
            role=CameraRole(cid),
            sequence=self.seq,
            host_receive_ns=time.monotonic_ns(),
            capture_ns=None,
            timestamp_source=TimestampSource.SYNTHETIC,
            timestamp_uncertainty_ns=0,
            pixel_format=PixelFormat.GRAY8,
            source_kind=SourceKind.SYNTHETIC,
            ground_truth=truth,
        )


def eye_profile(pipeline: str, estimation: str = "data_driven", eyes=("left_eye", "right_eye")):
    cams = {
        cid: {"source_type": "synthetic", "options": {"eye_model": "spherical"}} for cid in eyes
    }
    data = {
        "hardware_profile": {
            "id": "test-eyes",
            "cameras": cams,
            "gaze": {"pipeline": pipeline},
            "calibration": {"geometry_estimation": estimation},
        }
    }
    report = check_profile_data(data)
    assert report.ok, report.errors
    return report.profile


def app_config() -> AppConfig:
    return AppConfig.model_validate({"app": {"hardware_profile": "x.yaml"}})


def calibrate(tracker: GazeTracker, manager: FakeManager, profile, frames_per_point=12):
    rng = np.random.default_rng(1)
    # Look around first so a 3D eye model has something to fit (the wizard's intro screen).
    for _ in range(120):
        manager.target = tuple(rng.uniform(0.0, 1.0, 2))
        tracker.step(manager)
    session = CalibrationSession(
        tracker.schema_id,
        targets=GRID_9,
        min_samples_per_point=5,
        collect_duration_s=60,
        mapper_factory=lambda: make_mapper(
            tracker.schema_id, profile.calibration.geometry_estimation
        ),
    )
    records = {}
    for i, target in enumerate(session.targets):
        manager.target = target
        session.start_point(i)
        records[i] = []
        for _ in range(frames_per_point):
            frame = tracker.step(manager)
            if session.add_sample(frame.features):
                records[i].append(frame)
        session.finish_point(i)
    tracker.freeze()
    for i, pt in enumerate(session.points):
        pt.samples = [tracker.refeature(r) for r in records[i]]
    return session, session.fit()


@pytest.mark.parametrize(
    ("pipeline", "estimation", "eyes", "schema"),
    [
        ("pupil_2d", "data_driven", ("left_eye", "right_eye"), "C-pupil2d"),
        ("pupil_2d", "data_driven", ("left_eye",), "B-pupil2d"),
        ("eye_3d", "data_driven", ("left_eye", "right_eye"), "C-eye3d"),
        ("eye_3d", "geometric", ("left_eye", "right_eye"), "C-eye3d"),
        ("eye_3d", "geometric", ("right_eye",), "B-eye3d"),
    ],
)
def test_end_to_end_calibration(pipeline, estimation, eyes, schema):
    profile = eye_profile(pipeline, estimation, eyes)
    tracker = GazeTracker(profile, app_config())
    assert tracker.schema_id == schema
    manager = FakeManager(eyes, spherical=True)
    session, result = calibrate(tracker, manager, profile)

    # Predict at fresh positions inside the calibrated area.
    rng = np.random.default_rng(9)
    errors = []
    for _ in range(25):
        manager.target = tuple(rng.uniform(0.1, 0.9, 2))
        sample = result.model.predict(tracker.step(manager).features)
        errors.append(math.dist((sample.screen_x_norm, sample.screen_y_norm), manager.target))
    assert np.median(errors) < 0.03, errors
    assert max(errors) < 0.08

    report = validate_calibration(
        session.points,
        schema,
        mapper_factory=lambda: make_mapper(schema, estimation),
    )
    assert report.mean_error_norm < 0.08


def test_closed_eyes_are_reported_and_invalidate_features():
    profile = eye_profile("pupil_2d")
    tracker = GazeTracker(profile, app_config())
    manager = FakeManager(("left_eye", "right_eye"), spherical=False)
    frame = tracker.step(manager)
    assert frame.left_closed is False and frame.right_closed is False
    manager.closure = 1.0
    frame = tracker.step(manager)
    assert frame.left_closed and frame.right_closed
    layout = FEATURE_LAYOUTS["C-pupil2d"]
    assert not frame.features.valid[list(layout.eye_values["left"])].any()


def test_single_eye_profile_leaves_other_eye_unobserved():
    profile = eye_profile("pupil_2d", eyes=("left_eye",))
    frame = GazeTracker(profile, app_config()).step(FakeManager(("left_eye",), False))
    assert frame.left_closed is False
    assert frame.right_closed is None


def test_schema_for():
    assert schema_for(TrackingMode.WEBCAM_ONLY, "eye_3d") == "A-webcam"
    assert schema_for(TrackingMode.SINGLE_EYE, "pupil_2d") == "B-pupil2d"
    assert schema_for(TrackingMode.DUAL_EYE, "eye_3d") == "C-eye3d"


# ---------------------------------------------------------------------------
# Mappers on synthetic features (head-pose compensation)
# ---------------------------------------------------------------------------

SCREEN_W_CM, SCREEN_H_CM = 34.0, 22.0


def _true_r_ce(seed: int) -> np.ndarray:
    # Eye camera below the eye, tilted up at it, plus a little arbitrary roll.
    return head_rotation(5 * seed, -25.0, 10.0 - 5 * seed) @ np.diag([1.0, -1.0, -1.0])


def synthetic_3d_rows(targets, heads, mapper: GeometricRayMapper):
    """Feature rows generated with the geometric model's own conventions."""
    layout = FEATURE_LAYOUTS["C-eye3d"]
    head0 = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.15])
    rows = []
    for (x, y), h in zip(targets, heads, strict=True):
        h = np.asarray(h, dtype=np.float64)
        d = 60.0 * head0[5] / h[5]
        origin = np.array(
            [mapper.kx * d * (h[3] - head0[3]), mapper.ky * d * (h[4] - head0[4]), -d]
        )
        target = np.array([(x - 0.5) * SCREEN_W_CM, y * SCREEN_H_CM + 1.0, 0.0])
        u = target - origin
        u /= np.linalg.norm(u)
        row = np.zeros(layout.width, dtype=np.float32)
        for i, side in enumerate(("left", "right")):
            v = (head_rotation(*h[:3]) @ _true_r_ce(i)).T @ u
            row[list(layout.eye_values[side])] = v
            row[layout.eye_conf[side]] = 1.0
        row[list(layout.head)] = h
        rows.append(row)
    return np.stack(rows)


def test_geometric_mapper_compensates_unseen_head_poses():
    rng = np.random.default_rng(3)
    mapper = GeometricRayMapper("C-eye3d")
    targets = np.array(GRID_9, dtype=np.float32)
    # Calibration: head nearly still.
    cal_heads = [
        (*rng.normal(0, 1.0, 3), 0.5 + rng.normal(0, 0.005), 0.5 + rng.normal(0, 0.005), 0.15)
        for _ in targets
    ]
    X = synthetic_3d_rows(targets, cal_heads, mapper)
    mapper.fit(X, targets)

    data_driven = PolynomialRidgeMapper("C-eye3d")
    data_driven.fit(X, targets)

    # Test: head turned/tilted well beyond anything seen during calibration.
    test_targets = rng.uniform(0.1, 0.9, (30, 2))
    test_heads = [
        (*rng.uniform(-12, 12, 3), 0.5 + rng.uniform(-0.05, 0.05), 0.5, 0.15) for _ in test_targets
    ]
    Xt = synthetic_3d_rows(test_targets, test_heads, mapper)
    geo_err = [math.dist(mapper.predict_xy(x), t) for x, t in zip(Xt, test_targets, strict=True)]
    dd_err = [
        math.dist(data_driven.predict_xy(x), t) for x, t in zip(Xt, test_targets, strict=True)
    ]
    assert np.median(geo_err) < 0.02
    assert np.median(geo_err) < np.median(dd_err) / 3


def test_geometric_mapper_uses_remaining_eye():
    mapper = GeometricRayMapper("C-eye3d")
    targets = np.array(GRID_9, dtype=np.float32)
    heads = [(0, 0, 0, 0.5, 0.5, 0.15)] * len(targets)
    X = synthetic_3d_rows(targets, heads, mapper)
    mapper.fit(X, targets)
    layout = FEATURE_LAYOUTS["C-eye3d"]
    valid = np.ones(layout.width, dtype=bool)
    valid[list(layout.eye_values["left"])] = False
    x, y = mapper.predict_xy(X[4], valid)
    assert math.dist((x, y), targets[4]) < 0.01


def test_poly_mapper_falls_back_to_one_eye():
    layout = FEATURE_LAYOUTS["C-pupil2d"]
    rng = np.random.default_rng(0)
    targets = rng.uniform(0, 1, (15, 2))
    X = np.zeros((15, layout.width), dtype=np.float32)
    X[:, 0:2] = 0.3 + 0.4 * targets  # left pupil
    X[:, 3:5] = 0.35 + 0.4 * targets  # right pupil
    X[:, [2, 5]] = 1.0
    mapper = PolynomialRidgeMapper("C-pupil2d", alpha=1e-4)
    valid = np.ones_like(X, dtype=bool)
    valid[:, 6:] = False  # no head camera
    mapper.fit(X, targets, valid)
    one_eye = valid[0].copy()
    one_eye[0:3] = False
    for x, t in zip(X, targets, strict=True):
        assert math.dist(mapper.predict_xy(x, one_eye), t) < 0.01
    none = np.zeros(layout.width, dtype=bool)
    assert mapper.predict_xy(X[0], none) is None
    fv = FeatureVector(0, X[0], none, "C-pupil2d", 0, None, None, None)
    assert mapper.predict(fv).confidence == 0.0


@pytest.mark.parametrize("cls", [PolynomialRidgeMapper, GeometricRayMapper])
def test_mapper_arrays_round_trip(cls):
    mapper = cls("C-eye3d")
    targets = np.array(GRID_9, dtype=np.float32)
    X = synthetic_3d_rows(targets, [(0, 0, 0, 0.5, 0.5, 0.15)] * 9, GeometricRayMapper("C-eye3d"))
    mapper.fit(X, targets)
    arrays = {k: (np.bytes_(v) if isinstance(v, str) else v) for k, v in mapper.to_arrays().items()}
    loaded = mapper_from_arrays(arrays)
    assert type(loaded) is cls
    for x in X:
        assert np.allclose(mapper.predict_xy(x), loaded.predict_xy(x))


def test_geometric_needs_direction_schema():
    with pytest.raises(ValueError, match="3D gaze vectors"):
        GeometricRayMapper("C-pupil2d")


def test_make_mapper_kinds():
    assert isinstance(make_mapper("A-webcam", "data_driven"), GazeModel)
    assert isinstance(make_mapper("B-pupil2d"), PolynomialRidgeMapper)
    assert isinstance(make_mapper("C-eye3d", "geometric"), GeometricRayMapper)


def test_kabsch_recovers_rotation():
    r = head_rotation(20, -10, 35)
    a = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(kabsch(a, a @ r.T), r, atol=1e-9)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_store_round_trips_mapper_and_eye_state(tmp_path):
    profile = eye_profile("eye_3d", "geometric")
    tracker = GazeTracker(profile, app_config())
    manager = FakeManager(("left_eye", "right_eye"), spherical=True)
    _, result = calibrate(tracker, manager, profile)
    store = CalibrationStore(tmp_path)
    store.save(result.model, profile, eye_states=tracker.eye_states())

    bundle = store.load_bundle("C-eye3d", profile)
    assert isinstance(bundle.model, GeometricRayMapper)
    assert set(bundle.eye_states) == {"left", "right"}

    fresh = GazeTracker(profile, app_config())
    fresh.load_eye_states(bundle.eye_states)
    manager.target = (0.3, 0.7)
    sample = bundle.model.predict(fresh.step(manager).features)
    assert math.dist((sample.screen_x_norm, sample.screen_y_norm), (0.3, 0.7)) < 0.05


def test_store_rejects_changed_tracking_settings(tmp_path):
    profile = eye_profile("eye_3d", "geometric")
    targets = np.array(GRID_9, dtype=np.float32)
    mapper = GeometricRayMapper("C-eye3d")
    mapper.fit(synthetic_3d_rows(targets, [(0, 0, 0, 0.5, 0.5, 0.15)] * 9, mapper), targets)
    store = CalibrationStore(tmp_path)
    store.save(mapper, profile)

    changed = eye_profile("eye_3d", "data_driven")
    compat = store.compatibility("C-eye3d", changed)
    assert compat is not None and not compat.compatible
    assert any(c.path == "tracking.geometry_estimation" for c in compat.changes)
    with pytest.raises(ValueError, match="recalibrate"):
        store.load("C-eye3d", changed)

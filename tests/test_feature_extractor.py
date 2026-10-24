"""Tests for Mode A feature vector assembly.  No MediaPipe model required."""

from __future__ import annotations

import pytest

from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.observations import FEATURE_NAMES_MODE_A, EyeObservation, HeadObservation

_T = 1_000_000_000  # 1 s in ns


def _eye(
    side: str,
    *,
    x: float = 0.5,
    y: float = 0.4,
    conf: float = 0.9,
    blink: bool = False,
    t_ns: int = _T,
) -> EyeObservation:
    return EyeObservation(
        camera_id="head",
        eye=side,
        t_ns=t_ns,
        t_uncertainty_ns=None,
        pupil_x_norm=x,
        pupil_y_norm=y,
        pupil_radius_norm=0.02,
        blink=blink,
        confidence=conf,
        detector="test",
    )


def _head(
    yaw: float = 0.0,
    pitch: float = 5.0,
    roll: float = -2.0,
    t_ns: int = _T,
) -> HeadObservation:
    return HeadObservation(
        t_ns=t_ns,
        t_uncertainty_ns=None,
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        face_center_x_norm=0.5,
        face_center_y_norm=0.45,
        relative_scale=0.3,
        confidence=1.0,
        transform_4x4=None,
    )


def test_schema_id() -> None:
    fv = build_mode_a(None, None, None)
    assert fv.schema_id == "A-webcam"


def test_feature_count_matches_names() -> None:
    fv = build_mode_a(_head(), _eye("left"), _eye("right"))
    assert len(fv.values) == len(FEATURE_NAMES_MODE_A)
    assert len(fv.valid) == len(FEATURE_NAMES_MODE_A)


def test_all_none_produces_all_invalid() -> None:
    fv = build_mode_a(None, None, None)
    assert not fv.valid.any()


def test_full_sample_all_valid() -> None:
    fv = build_mode_a(_head(), _eye("left"), _eye("right"))
    assert fv.valid.all()


def test_iris_values_correct() -> None:
    fv = build_mode_a(_head(), _eye("left", x=0.3, y=0.4, conf=0.8), _eye("right", x=0.7, y=0.4))
    assert fv.values[0] == pytest.approx(0.3)   # left_pupil_x
    assert fv.values[1] == pytest.approx(0.4)   # left_pupil_y
    assert fv.values[2] == pytest.approx(0.8)   # left_pupil_conf
    assert fv.values[3] == pytest.approx(0.7)   # right_pupil_x


def test_head_values_correct() -> None:
    fv = build_mode_a(_head(yaw=10.0, pitch=-5.0, roll=2.0), None, None)
    assert fv.values[6] == pytest.approx(10.0)
    assert fv.values[7] == pytest.approx(-5.0)
    assert fv.values[8] == pytest.approx(2.0)


def test_blink_marks_eye_invalid() -> None:
    fv = build_mode_a(_head(), _eye("left", blink=True), _eye("right"))
    assert not fv.valid[0]  # left_pupil_x invalid
    assert not fv.valid[1]
    assert not fv.valid[2]
    assert fv.valid[3]      # right_pupil_x still valid
    assert fv.valid[6]      # head features valid


def test_missing_head_marks_pose_invalid() -> None:
    fv = build_mode_a(None, _eye("left"), _eye("right"))
    assert fv.valid[0]      # left eye valid
    assert not fv.valid[6]  # head_yaw invalid


def test_max_skew_zero_when_same_timestamp() -> None:
    fv = build_mode_a(_head(t_ns=_T), _eye("left", t_ns=_T), _eye("right", t_ns=_T))
    assert fv.max_skew_ns == 0


def test_max_skew_reflects_gap() -> None:
    late = _T + 10_000_000  # 10 ms later
    fv = build_mode_a(_head(t_ns=late), _eye("left", t_ns=_T), _eye("right", t_ns=_T))
    assert fv.max_skew_ns == 10_000_000


def test_timestamp_fields() -> None:
    fv = build_mode_a(_head(t_ns=_T + 1), _eye("left", t_ns=_T), _eye("right", t_ns=_T + 2))
    assert fv.head_t_ns == _T + 1
    assert fv.left_eye_t_ns == _T
    assert fv.right_eye_t_ns == _T + 2


def test_missing_sources_have_none_timestamps() -> None:
    fv = build_mode_a(_head(), None, None)
    assert fv.left_eye_t_ns is None
    assert fv.right_eye_t_ns is None
    assert fv.head_t_ns == _T

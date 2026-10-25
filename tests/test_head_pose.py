"""Webcam head pose from landmarks: conventions checked against a synthetic posed face."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opengaze.vision.head_pose import (
    EYE_RADIUS_PER_WIDTH,
    MP_CHIN,
    MP_FOREHEAD,
    MP_LEFT_EYE_INNER,
    MP_LEFT_EYE_OUTER,
    MP_LEFT_IRIS,
    MP_RIGHT_EYE_INNER,
    MP_RIGHT_EYE_OUTER,
    MP_RIGHT_IRIS,
    angles_from_rotation,
    estimate_from_landmarks,
    focal_px,
    rotation_from_angles,
)

W, H, HFOV = 1280, 720, 60.0
F = focal_px(W, HFOV)


def face_model(eye_rot=(0.0, 0.0)) -> dict[str, np.ndarray]:
    """Points in the head frame (cm): x = user's right, y = chin, z = out of the face."""
    pts = {
        "r_outer": (4.5, 0.0, 0.0),
        "r_inner": (1.5, 0.0, 0.0),
        "l_inner": (-1.5, 0.0, 0.0),
        "l_outer": (-4.5, 0.0, 0.0),
        "forehead": (0.0, -7.0, 0.8),
        "chin": (0.0, 12.0, 0.0),
    }
    width = 3.0
    r = EYE_RADIUS_PER_WIDTH * width
    sx, sy = math.sin(eye_rot[0]), math.sin(eye_rot[1])
    pts["r_iris"] = (3.0 + r * sx, r * sy, 0.5)
    pts["l_iris"] = (-3.0 + r * sx, r * sy, 0.5)
    return {k: np.array(v) for k, v in pts.items()}


def landmarks(yaw, pitch, roll, position, *, mirrored, eye_rot=(0.0, 0.0)) -> np.ndarray:
    rot = rotation_from_angles(yaw, pitch, roll)
    world = {k: rot @ p + np.asarray(position) for k, p in face_model(eye_rot).items()}
    ref_depth = -np.mean([world[k][2] for k in ("r_outer", "l_outer")])
    lm = np.zeros((478, 3))

    def put(index, p):
        # Real camera (OpenCV): X = −world x, Y = world y, Z = −world z.
        x_cam, y_cam, z_cam = -p[0], p[1], -p[2]
        u = F * x_cam / z_cam + W / 2
        v = F * y_cam / z_cam + H / 2
        if mirrored:
            u = W - u
        lm[index] = (u / W, v / H, (z_cam - ref_depth) * F / ref_depth / W)

    # MediaPipe names eyes after the face it sees: mirrored, its "right" is the user's left.
    mp_right, mp_left = ("l", "r") if mirrored else ("r", "l")
    put(MP_RIGHT_EYE_OUTER, world[f"{mp_right}_outer"])
    put(MP_RIGHT_EYE_INNER, world[f"{mp_right}_inner"])
    put(MP_RIGHT_IRIS, world[f"{mp_right}_iris"])
    put(MP_LEFT_EYE_OUTER, world[f"{mp_left}_outer"])
    put(MP_LEFT_EYE_INNER, world[f"{mp_left}_inner"])
    put(MP_LEFT_IRIS, world[f"{mp_left}_iris"])
    put(MP_FOREHEAD, world["forehead"])
    put(MP_CHIN, world["chin"])
    return lm


@pytest.mark.parametrize("mirrored", [True, False])
@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"), [(0, 0, 0), (15, 0, 0), (0, 12, 0), (0, 0, -10), (-12, 8, 6)]
)
def test_recovers_rotation_and_position(mirrored, yaw, pitch, roll):
    position = (6.0, 9.0, -55.0)
    lm = landmarks(yaw, pitch, roll, position, mirrored=mirrored)
    pose = estimate_from_landmarks(lm, W, H, mirrored=mirrored, focal_length_px=F)
    assert pose is not None
    # Weak perspective: a few degrees / cm of error is expected for an off-centre face.
    assert pose.yaw_deg == pytest.approx(yaw, abs=3.0)
    assert pose.pitch_deg == pytest.approx(pitch, abs=3.0)
    assert pose.roll_deg == pytest.approx(roll, abs=3.0)
    assert np.allclose(pose.position_cm, position, atol=3.0)


def test_turning_right_is_positive_yaw_in_both_image_conventions():
    for mirrored in (True, False):
        lm = landmarks(20, 0, 0, (0, 0, -60), mirrored=mirrored)
        pose = estimate_from_landmarks(lm, W, H, mirrored=mirrored, focal_length_px=F)
        assert pose.yaw_deg > 15


def test_moving_right_is_positive_x():
    for mirrored in (True, False):
        lm = landmarks(0, 0, 0, (10, 0, -60), mirrored=mirrored)
        pose = estimate_from_landmarks(lm, W, H, mirrored=mirrored, focal_length_px=F)
        assert pose.position_cm[0] == pytest.approx(10, abs=1.0)


def test_eye_in_head_direction_ignores_head_motion():
    eye = (math.radians(15), math.radians(-10))
    dirs = []
    for yaw, pos in [(0, (0, 0, -60)), (14, (8, 4, -50)), (-10, (-6, 3, -70))]:
        lm = landmarks(yaw, 0, 0, pos, mirrored=True, eye_rot=eye)
        pose = estimate_from_landmarks(lm, W, H, mirrored=True, focal_length_px=F)
        dirs.append(pose.eye_dirs["left"])
    expected = np.array([math.sin(eye[0]), math.sin(eye[1])])
    for d in dirs:
        assert np.allclose(d[:2], expected, atol=0.04)
        assert d[2] > 0


def test_angles_round_trip():
    for angles in [(10, -20, 5), (-35, 15, -25), (0, 0, 0)]:
        assert np.allclose(angles_from_rotation(rotation_from_angles(*angles)), angles)


def test_too_few_landmarks():
    lm = np.zeros((10, 3))
    assert estimate_from_landmarks(lm, W, H, mirrored=True, focal_length_px=F) is None

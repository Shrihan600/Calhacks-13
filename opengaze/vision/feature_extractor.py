"""Assembles a Mode A FeatureVector from raw vision observations.

Each tracking mode has its own builder (Modes B/C: ``gaze/features.py``). A calibration
model is fitted for one ``schema_id`` and refuses vectors from another, so modes are never
mixed.
"""

from __future__ import annotations

import time

import numpy as np

from opengaze.vision.observations import (
    FEATURE_LAYOUTS,
    EyeObservation,
    FeatureVector,
    HeadObservation,
    head_values,
)

_LAYOUT = FEATURE_LAYOUTS["A-webcam"]


def write_head(v: np.ndarray, ok: np.ndarray, layout, head: HeadObservation | None) -> None:
    """Fill a layout's head slots; rotation and position are valid independently."""
    if head is None:
        return
    rot, rot_ok, pos, pos_ok = head_values(head)
    v[list(layout.head_rot)] = rot
    ok[list(layout.head_rot)] = rot_ok
    v[list(layout.head_pos)] = pos
    ok[list(layout.head_pos)] = pos_ok


def build_mode_a(
    head: HeadObservation | None,
    left_eye: EyeObservation | None,
    right_eye: EyeObservation | None,
    *,
    t_ns: int | None = None,
) -> FeatureVector:
    """Build a Mode A (webcam-only) FeatureVector.

    Feature order (14): per eye (left, right) the eye-in-head gaze direction x/y/z and its
    confidence, then head yaw/pitch/roll (deg) and head x/y/z (cm, world frame). Because the
    eye features are measured relative to the eye corners, head translation doesn't move
    them; head rotation and position are compensated by the screen mapper.
    """
    if t_ns is None:
        t_ns = _newest_timestamp(head, left_eye, right_eye)

    v = np.zeros(_LAYOUT.width, dtype=np.float32)
    ok = np.zeros(_LAYOUT.width, dtype=bool)

    for side, eye in (("left", left_eye), ("right", right_eye)):
        if eye is None or eye.blink or eye.gaze_in_head is None:
            continue
        cols = list(_LAYOUT.eye_values[side])
        v[cols] = eye.gaze_in_head
        v[_LAYOUT.eye_conf[side]] = eye.confidence
        ok[cols] = True
        ok[_LAYOUT.eye_conf[side]] = True

    write_head(v, ok, _LAYOUT, head)

    timestamps = [obs.t_ns for obs in (head, left_eye, right_eye) if obs is not None]
    max_skew = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

    return FeatureVector(
        t_ns=t_ns,
        values=v,
        valid=ok,
        schema_id="A-webcam",
        max_skew_ns=max_skew,
        head_t_ns=head.t_ns if head is not None else None,
        left_eye_t_ns=left_eye.t_ns if left_eye is not None else None,
        right_eye_t_ns=right_eye.t_ns if right_eye is not None else None,
    )


def _newest_timestamp(
    head: HeadObservation | None,
    left: EyeObservation | None,
    right: EyeObservation | None,
) -> int:
    stamps = [obs.t_ns for obs in (head, left, right) if obs is not None]
    return max(stamps) if stamps else time.monotonic_ns()

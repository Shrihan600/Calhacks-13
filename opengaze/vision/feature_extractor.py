"""Assembles a FeatureVector from raw vision observations.

Each tracking mode has its own builder function. A calibration model is fitted for
one ``schema_id`` and refuses vectors from another, so modes are never mixed.
"""

from __future__ import annotations

import time

import numpy as np

from opengaze.vision.observations import (
    FEATURE_NAMES_MODE_A,
    EyeObservation,
    FeatureVector,
    HeadObservation,
)

_N_MODE_A = len(FEATURE_NAMES_MODE_A)


def build_mode_a(
    head: HeadObservation | None,
    left_eye: EyeObservation | None,
    right_eye: EyeObservation | None,
    *,
    t_ns: int | None = None,
) -> FeatureVector:
    """Build a Mode A (webcam-only) FeatureVector.

    Feature order (12 total):
        left_pupil_x/y/conf, right_pupil_x/y/conf,
        head_yaw/pitch/roll, head_center_x/y, head_scale
    """
    if t_ns is None:
        t_ns = _newest_timestamp(head, left_eye, right_eye)

    v = np.zeros(_N_MODE_A, dtype=np.float32)
    ok = np.zeros(_N_MODE_A, dtype=bool)

    if left_eye is not None and not left_eye.blink:
        v[0], v[1], v[2] = left_eye.pupil_x_norm, left_eye.pupil_y_norm, left_eye.confidence
        ok[0:3] = True

    if right_eye is not None and not right_eye.blink:
        v[3], v[4], v[5] = right_eye.pupil_x_norm, right_eye.pupil_y_norm, right_eye.confidence
        ok[3:6] = True

    if head is not None:
        v[6], v[7], v[8] = head.yaw_deg, head.pitch_deg, head.roll_deg
        v[9], v[10], v[11] = head.face_center_x_norm, head.face_center_y_norm, head.relative_scale
        ok[6:12] = True

    timestamps = [
        obs.t_ns
        for obs in (head, left_eye, right_eye)
        if obs is not None
    ]
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

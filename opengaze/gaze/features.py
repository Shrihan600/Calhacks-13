"""Feature vectors for Modes B/C: per-eye pipeline results + head pose → ``FeatureVector``."""

from __future__ import annotations

import time

import numpy as np

from opengaze.gaze.pipelines import EyeResult
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureVector, HeadObservation


def build_eye_features(
    schema_id: str,
    head: HeadObservation | None,
    left: EyeResult | None,
    right: EyeResult | None,
    *,
    t_ns: int | None = None,
) -> FeatureVector:
    """Pack eye results into the schema's layout (see ``FEATURE_LAYOUTS``).

    An eye's slots are valid only when it is open and its feature exists (for ``eye_3d``, once
    the eye model is fitted). Head slots are valid when a head observation is present.
    """
    layout = FEATURE_LAYOUTS[schema_id]
    v = np.zeros(layout.width, dtype=np.float32)
    ok = np.zeros(layout.width, dtype=bool)

    for side, result in (("left", left), ("right", right)):
        if result is None or result.closed:
            continue
        values = _eye_values(layout.eye_kind, result)
        if values is None:
            continue
        cols = list(layout.eye_values[side])
        v[cols] = values
        v[layout.eye_conf[side]] = result.confidence
        ok[cols] = True
        ok[layout.eye_conf[side]] = True

    if head is not None:
        cols = list(layout.head)
        v[cols] = (
            head.yaw_deg,
            head.pitch_deg,
            head.roll_deg,
            head.face_center_x_norm,
            head.face_center_y_norm,
            head.relative_scale,
        )
        ok[cols] = True

    stamps = [o.t_ns for o in (head, left, right) if o is not None]
    if t_ns is None:
        t_ns = max(stamps) if stamps else time.monotonic_ns()
    return FeatureVector(
        t_ns=t_ns,
        values=v,
        valid=ok,
        schema_id=schema_id,  # type: ignore[arg-type]
        max_skew_ns=max(stamps) - min(stamps) if len(stamps) > 1 else 0,
        head_t_ns=head.t_ns if head is not None else None,
        left_eye_t_ns=left.t_ns if left is not None else None,
        right_eye_t_ns=right.t_ns if right is not None else None,
    )


def _eye_values(kind: str, result: EyeResult) -> tuple[float, ...] | None:
    if kind == "position":
        if result.detection is None:
            return None
        return result.detection.center_norm
    if result.gaze is None:
        return None
    return tuple(float(c) for c in result.gaze.direction)

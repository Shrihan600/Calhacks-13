"""Tracking modes, determined by which camera roles are present."""

from __future__ import annotations

import enum
from collections.abc import Iterable

from opengaze.cameras.types import CameraRole


class TrackingMode(enum.StrEnum):
    WEBCAM_ONLY = "A"  # MacBook webcam only: MediaPipe head pose + approximate iris
    SINGLE_EYE = "B"  # one eye camera (+ webcam for head pose)
    DUAL_EYE = "C"  # two eye cameras (+ webcam for head pose)

    @property
    def label(self) -> str:
        return {
            TrackingMode.WEBCAM_ONLY: "Mode A · webcam only",
            TrackingMode.SINGLE_EYE: "Mode B · single eye camera",
            TrackingMode.DUAL_EYE: "Mode C · dual eye cameras",
        }[self]


def infer_mode(roles: Iterable[CameraRole]) -> TrackingMode | None:
    """The richest mode the given roles support, or None if there's nothing to track with.

    Calibration models are fitted per mode; never assume one mode's accuracy for another.
    """
    present = set(roles)
    eyes = len(present & {CameraRole.LEFT_EYE, CameraRole.RIGHT_EYE})
    if eyes == 2:
        return TrackingMode.DUAL_EYE
    if eyes == 1:
        return TrackingMode.SINGLE_EYE
    if CameraRole.HEAD in present:
        return TrackingMode.WEBCAM_ONLY
    return None

"""Tracking modes, determined by which camera roles are present."""

from __future__ import annotations

import enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from opengaze.cameras.types import CameraRole

if TYPE_CHECKING:
    from opengaze.config.schema import HardwareProfile


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


EyePipelineKind = Literal["pupil_2d", "eye_3d"]


def schema_for(mode: TrackingMode | None, pipeline: EyePipelineKind = "pupil_2d") -> str:
    """Feature schema id for a mode and eye pipeline. Mode A has no eye cameras, so no pipeline."""
    if mode is None or mode is TrackingMode.WEBCAM_ONLY:
        return "A-webcam"
    prefix = "B" if mode is TrackingMode.SINGLE_EYE else "C"
    return f"{prefix}-{'pupil2d' if pipeline == 'pupil_2d' else 'eye3d'}"


def profile_schema(profile: HardwareProfile) -> str:
    """Schema id the given hardware profile calibrates and tracks with."""
    mode = infer_mode(cam.role for cam in profile.enabled_cameras.values())
    return schema_for(mode, profile.gaze.pipeline)

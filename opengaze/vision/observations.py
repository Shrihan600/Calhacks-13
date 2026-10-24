"""Observation data types produced by vision detectors.

All x/y coordinates are normalised to [0, 1] within the frame image unless noted.
Nothing is invented: a field is ``None`` when genuinely unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class EyeObservation:
    """Pupil / iris features for one eye extracted from one frame."""

    camera_id: str
    eye: Literal["left", "right"]
    t_ns: int
    t_uncertainty_ns: int | None
    # Normalised to [0, 1] within the normalised camera image.
    pupil_x_norm: float
    pupil_y_norm: float
    pupil_radius_norm: float  # fraction of the image's shorter side
    blink: bool
    confidence: float  # 0 … 1
    detector: str  # e.g. "mediapipe_face_landmarker"


@dataclass(frozen=True, slots=True, eq=False)
class HeadObservation:
    """Head pose and face-centre position in one frame.

    ``eq=False``: the optional ``transform_4x4`` numpy array cannot be compared
    element-wise with ``==`` without raising an ambiguity error.
    """

    t_ns: int
    t_uncertainty_ns: int | None
    yaw_deg: float    # positive = turning right
    pitch_deg: float  # positive = looking up
    roll_deg: float   # positive = tilting right
    face_center_x_norm: float  # midpoint between irises, normalised [0, 1]
    face_center_y_norm: float
    relative_scale: float  # inter-ocular distance / reference; 1.0 at calibration distance
    confidence: float  # 0 … 1
    transform_4x4: np.ndarray | None  # 4×4 head-to-camera transform from MediaPipe, if available


# Canonical order for Mode A. Calibration models are trained to this fixed layout.
FEATURE_NAMES_MODE_A: tuple[str, ...] = (
    "left_pupil_x",
    "left_pupil_y",
    "left_pupil_conf",
    "right_pupil_x",
    "right_pupil_y",
    "right_pupil_conf",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "head_center_x",
    "head_center_y",
    "head_scale",
)


@dataclass(frozen=True, slots=True, eq=False)
class FeatureVector:
    """Fixed-length feature sample ready for a calibration or gaze model.

    ``valid[i]`` is ``False`` when that feature was unavailable (blink, no face
    detected, missing camera). Models must consult the validity mask before using
    a value; invalid slots hold 0.0.

    ``eq=False``: contains numpy arrays.
    """

    t_ns: int
    values: np.ndarray  # float32[N]
    valid: np.ndarray   # bool[N]
    schema_id: Literal["A-webcam", "B-single-eye", "C-dual-eye"]
    max_skew_ns: int    # time gap between the newest and oldest contributing observation
    head_t_ns: int | None
    left_eye_t_ns: int | None
    right_eye_t_ns: int | None

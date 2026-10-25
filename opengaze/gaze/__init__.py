"""Modes B/C gaze estimation from eye cameras, and the per-frame tracker shared by all UIs.

Swappable components (each a Protocol with a registry of implementations):

    detector2d.PupilDetector2D   eye image → pupil ellipse        (built-in: ellipse_fit)
    eye_model3d.EyeModel3D       pupil ellipses → 3D gaze vector  (built-in: sphere_fit)
    pipelines.EyePipeline        pupil_2d | eye_3d, built from the profile's ``gaze:`` section

Screen mapping (with head-pose compensation) lives in ``opengaze/calibration/mappers.py``.
"""

from opengaze.gaze.detector2d import (
    DETECTORS_2D,
    EllipseFitPupilDetector,
    PupilDetection2D,
    PupilDetector2D,
    make_pupil_detector,
)
from opengaze.gaze.eye_model3d import (
    EYE_MODELS_3D,
    EyeModel3D,
    GazeVector3D,
    SphereFitEyeModel,
    make_eye_model,
)
from opengaze.gaze.features import build_eye_features
from opengaze.gaze.pipelines import (
    Eye3DPipeline,
    EyePipeline,
    EyeResult,
    Pupil2DPipeline,
    make_eye_pipeline,
)

__all__ = [
    "DETECTORS_2D",
    "EYE_MODELS_3D",
    "EllipseFitPupilDetector",
    "Eye3DPipeline",
    "EyeModel3D",
    "EyePipeline",
    "EyeResult",
    "GazeVector3D",
    "PupilDetection2D",
    "PupilDetector2D",
    "Pupil2DPipeline",
    "SphereFitEyeModel",
    "build_eye_features",
    "make_eye_model",
    "make_eye_pipeline",
    "make_pupil_detector",
]

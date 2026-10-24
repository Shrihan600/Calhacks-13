"""Calibration: collect, fit, validate, and persist gaze models."""

from opengaze.calibration.mappers import (
    GeometricRayMapper,
    PolynomialRidgeMapper,
    make_mapper,
    mapper_from_arrays,
)
from opengaze.calibration.model import GazeModel, GazeSample, ScreenMapper
from opengaze.calibration.session import CalibrationPoint, CalibrationSession
from opengaze.calibration.storage import CalibrationBundle, CalibrationStore
from opengaze.calibration.validation import ValidationReport, validate_calibration

__all__ = [
    "CalibrationBundle",
    "CalibrationPoint",
    "CalibrationSession",
    "CalibrationStore",
    "GazeModel",
    "GazeSample",
    "GeometricRayMapper",
    "PolynomialRidgeMapper",
    "ScreenMapper",
    "ValidationReport",
    "make_mapper",
    "mapper_from_arrays",
    "validate_calibration",
]

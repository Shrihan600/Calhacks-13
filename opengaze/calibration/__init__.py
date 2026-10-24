"""Calibration: collect, fit, validate, and persist gaze models."""

from opengaze.calibration.model import GazeModel, GazeSample
from opengaze.calibration.session import CalibrationPoint, CalibrationSession
from opengaze.calibration.storage import CalibrationStore
from opengaze.calibration.validation import ValidationReport, validate_calibration

__all__ = [
    "CalibrationPoint",
    "CalibrationSession",
    "CalibrationStore",
    "GazeModel",
    "GazeSample",
    "ValidationReport",
    "validate_calibration",
]

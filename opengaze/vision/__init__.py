"""Vision pipeline stages. Each stage consumes and produces plain data contracts."""

from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.observations import (
    FEATURE_NAMES_MODE_A,
    EyeObservation,
    FeatureVector,
    HeadObservation,
)
from opengaze.vision.preprocessing import FrameNormalizer

__all__ = [
    "build_mode_a",
    "EyeObservation",
    "FeatureVector",
    "FEATURE_NAMES_MODE_A",
    "FrameNormalizer",
    "HeadObservation",
]

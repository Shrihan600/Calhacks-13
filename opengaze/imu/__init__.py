"""Head-mounted IMU (gyroscope + accelerometer) for head rotation in Modes B/C."""

from opengaze.imu.fusion import ImuOrientation, MahonyFilter, axis_matrix
from opengaze.imu.head import HeadPoseFusion
from opengaze.imu.runtime import ImuRuntime
from opengaze.imu.sources import IMU_SOURCES, ImuSource, make_imu_source
from opengaze.imu.types import ImuSample, parse_line

__all__ = [
    "IMU_SOURCES",
    "HeadPoseFusion",
    "ImuOrientation",
    "ImuRuntime",
    "ImuSample",
    "ImuSource",
    "MahonyFilter",
    "axis_matrix",
    "make_imu_source",
    "parse_line",
]

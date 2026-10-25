"""Head pose for Modes B/C: IMU rotation, anchored to the screen.

* Rotation comes from the IMU: fast, smooth and in the same rigid frame as the eye cameras.
* The IMU's yaw is relative, so it is anchored: on the first reading and on ``recenter()``
  the current heading is taken as "facing the screen" (or as the webcam's yaw, when a head
  camera sees the face). With a webcam, yaw is also pulled slowly toward the webcam's
  estimate every frame, which cancels gyro drift; pitch and roll are already absolute.
* Position needs the webcam (an IMU can't measure it). IMU only, position is unknown and
  the screen mapper treats the head as staying where it was during calibration.
"""

from __future__ import annotations

import math

import numpy as np

from opengaze.vision.head_pose import angles_from_rotation, rotation_from_angles
from opengaze.vision.observations import HeadObservation


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class HeadPoseFusion:
    def __init__(self, webcam_yaw_rate: float = 0.5) -> None:
        """``webcam_yaw_rate``: fraction per second of the IMU-vs-webcam yaw difference that
        is corrected (0 disables; 0.5 removes drift within a few seconds without passing
        webcam jitter through)."""
        self.webcam_yaw_rate = webcam_yaw_rate
        self._yaw_offset: float | None = None  # degrees added to the IMU's yaw
        self._last_t_ns: int | None = None
        self._last_imu: np.ndarray | None = None
        self._last_webcam: HeadObservation | None = None

    def recenter(self) -> None:
        """Take the current heading as facing the screen (or the webcam's yaw, if seen)."""
        if self._last_imu is not None:
            self._anchor(self._last_imu, self._last_webcam)

    def _anchor(self, imu_rot: np.ndarray, webcam: HeadObservation | None) -> None:
        imu_yaw = angles_from_rotation(imu_rot)[0]
        target = webcam.yaw_deg if webcam is not None else 0.0
        self._yaw_offset = _wrap_deg(target - imu_yaw)

    def update(
        self, imu_rot: np.ndarray | None, webcam: HeadObservation | None, t_ns: int
    ) -> HeadObservation | None:
        if imu_rot is None:
            return webcam  # IMU missing or stale: fall back to the webcam alone
        self._last_imu, self._last_webcam = imu_rot, webcam
        if self._yaw_offset is None:
            self._anchor(imu_rot, webcam)
        assert self._yaw_offset is not None

        rot = rotation_from_angles(self._yaw_offset, 0.0, 0.0) @ imu_rot
        yaw, pitch, roll = angles_from_rotation(rot)
        dt = 0.0 if self._last_t_ns is None else max(0.0, (t_ns - self._last_t_ns) / 1e9)
        self._last_t_ns = t_ns
        if webcam is not None and self.webcam_yaw_rate > 0:
            step = _wrap_deg(webcam.yaw_deg - yaw) * min(1.0, self.webcam_yaw_rate * dt)
            self._yaw_offset = _wrap_deg(self._yaw_offset + step)

        return HeadObservation(
            t_ns=t_ns,
            t_uncertainty_ns=None,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            face_center_x_norm=webcam.face_center_x_norm if webcam else math.nan,
            face_center_y_norm=webcam.face_center_y_norm if webcam else math.nan,
            relative_scale=webcam.relative_scale if webcam else math.nan,
            confidence=1.0,
            transform_4x4=None,
            position_cm=webcam.position_cm if webcam else None,
            rotation=rot,
            source="imu+webcam" if webcam else "imu",
        )

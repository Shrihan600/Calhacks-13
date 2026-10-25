"""IMU orientation: axis mapping, gyro-bias estimation and a Mahony complementary filter.

Output is the head → earth rotation in the world-frame convention of ``vision/head_pose.py``
(x right, y down, z forward) with y aligned to gravity. Pitch and roll are absolute (the
accelerometer sees gravity); yaw is only relative: it starts at 0 and drifts slowly with
residual gyro bias. ``imu/head.py`` anchors yaw to the screen.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from opengaze.imu.types import GRAVITY, ImuSample
from opengaze.vision.head_pose import angles_from_rotation, rotation_from_angles

_AXES = {"x": 0, "y": 1, "z": 2}


def axis_matrix(axes: tuple[str, str, str]) -> np.ndarray:
    """M with ``head_vector = M @ imu_vector``.

    ``axes`` names the IMU axis pointing toward the head's right, chin and forward, e.g.
    ``("+x", "-z", "+y")``. Raises ValueError unless it is a proper rotation: IMUs are
    right-handed, so a mirrored mapping means the axes were written down wrong.
    """
    m = np.zeros((3, 3))
    for row, spec in enumerate(axes):
        spec = spec.strip().lower()
        if len(spec) != 2 or spec[0] not in "+-" or spec[1] not in _AXES:
            raise ValueError(f"bad IMU axis {spec!r}: use +x, -x, +y, -y, +z or -z")
        m[row, _AXES[spec[1]]] = 1.0 if spec[0] == "+" else -1.0
    if sorted(abs(m).argmax(axis=1)) != [0, 1, 2]:
        raise ValueError(f"IMU axes {list(axes)} must use x, y and z once each")
    if np.linalg.det(m) < 0:
        raise ValueError(
            f"IMU axes {list(axes)} describe a mirrored frame; flip the sign of one axis"
        )
    return m


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class MahonyFilter:
    """Gyro integration corrected toward the accelerometer's gravity direction."""

    def __init__(self, kp: float = 1.0, ki: float = 0.02) -> None:
        self.kp, self.ki = kp, ki
        self.q: np.ndarray | None = None  # head → earth
        self._integral = np.zeros(3)

    def reset(self) -> None:
        self.q = None
        self._integral = np.zeros(3)

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        up = accel / np.linalg.norm(accel) if np.linalg.norm(accel) > 1e-6 else None
        if self.q is None:
            if up is None:
                return
            # Level from gravity, heading 0.
            r0 = _align(up, np.array([0.0, -1.0, 0.0]))
            yaw, _, _ = angles_from_rotation(r0)
            self.q = _matrix_to_quat(rotation_from_angles(-yaw, 0.0, 0.0) @ r0)
            return
        omega = gyro.copy()
        norm = np.linalg.norm(accel)
        # Trust the accelerometer only near 1 g (not while the head is accelerating).
        if up is not None and abs(norm - GRAVITY) < 0.25 * GRAVITY:
            r = _quat_to_matrix(self.q)
            expected_up = r.T @ np.array([0.0, -1.0, 0.0])
            err = np.cross(up, expected_up)
            self._integral += self.ki * err * dt
            omega = omega + self.kp * err + self._integral
        dq = 0.5 * _quat_mul(self.q, np.array([0.0, *omega]))
        q = self.q + dq * dt
        self.q = q / np.linalg.norm(q)

    def rotation(self) -> np.ndarray | None:
        return None if self.q is None else _quat_to_matrix(self.q)


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


class GyroBiasEstimator:
    """Learns the gyro's zero-rate offset whenever the head is still.

    Pitch and roll bias is also corrected by the accelerometer, but yaw bias isn't, so this
    is what keeps yaw from drifting by degrees per minute.

    "Still" means the readings are *constant* over a window (low spread), not merely small:
    a slowly turning head also passes through near-zero rates, and learning those as bias
    would cause the very drift this is meant to remove.
    """

    def __init__(
        self,
        window_s: float = 0.5,
        still_gyro_std_dps: float = 0.6,
        still_accel_std_g: float = 0.02,
        max_bias_dps: float = 10.0,
    ) -> None:
        self.bias = np.zeros(3)
        self.window_s = window_s
        self.still_gyro_std = math.radians(still_gyro_std_dps)
        self.still_accel_std = still_accel_std_g * GRAVITY
        self.max_bias = math.radians(max_bias_dps)
        self._buf: collections.deque[tuple[float, np.ndarray, np.ndarray]] = collections.deque()
        self._t = 0.0
        self._n = 0

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        """Feed a raw reading; returns the bias-corrected gyro."""
        self._t += dt
        self._buf.append((self._t, gyro, accel))
        while self._buf and self._t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        span = self._buf[-1][0] - self._buf[0][0]
        if span >= 0.8 * self.window_s and len(self._buf) >= 10:
            g = np.stack([b[1] for b in self._buf])
            a = np.stack([b[2] for b in self._buf])
            mean = g.mean(axis=0)
            still = (
                float(g.std(axis=0).max()) < self.still_gyro_std
                and float(a.std(axis=0).max()) < self.still_accel_std
                and abs(np.linalg.norm(a.mean(axis=0)) - GRAVITY) < 0.1 * GRAVITY
                and float(np.linalg.norm(mean)) < self.max_bias
            )
            if still:
                self._n += 1
                alpha = max(0.005, 1.0 / self._n)  # running mean first, then slow tracking
                self.bias += alpha * (mean - self.bias)
        return gyro - self.bias


class ImuOrientation:
    """Axis mapping → bias removal → Mahony filter, one reading at a time."""

    def __init__(self, axes: tuple[str, str, str], kp: float = 1.0, ki: float = 0.02) -> None:
        self.m = axis_matrix(axes)
        self.bias = GyroBiasEstimator()
        self.filter = MahonyFilter(kp, ki)
        self._last_time_s: float | None = None

    def update(self, sample: ImuSample) -> np.ndarray | None:
        t = sample.device_time_s
        if t is None:
            t = sample.host_receive_ns / 1e9
        dt = 0.0 if self._last_time_s is None else t - self._last_time_s
        self._last_time_s = t
        dt = min(max(dt, 0.0), 0.1)  # tolerate gaps / clock jumps
        gyro = self.m @ sample.gyro
        accel = self.m @ sample.accel
        gyro = self.bias.update(gyro, accel, dt)
        self.filter.update(gyro, accel, dt)
        return self.filter.rotation()

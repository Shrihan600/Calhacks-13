"""Head pose (rotation + position) and eye-in-head gaze from face landmarks.

Every head pose in OpenGaze, whether from the webcam or an IMU, is expressed in one
**world frame**: x to the right along the screen (as the user sees it), y down, z from the
user into the screen. The head camera sits at the origin on the screen plane, so a user in
front of it is at z < 0.

The **head frame** is x toward the user's right ear, y toward the chin, z out of the face.
``R`` (head → world) is the identity when the user faces the screen upright.

Angles follow ``HeadObservation``: ``R = Ry(yaw) · Rx(pitch) · Rz(roll)``, with positive yaw
turning right, positive pitch looking up and positive roll tilting the head to the right.

Landmark input is MediaPipe-style: x, y normalised to the image width/height, z a relative
depth on the same scale as x, smaller = closer to the camera. ``mirrored`` says whether the
image was flipped horizontally (so that image-right is screen-right). That matters because
MediaPipe names landmarks by the face it *sees*: in a mirrored image its "right eye" is the
user's left eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MediaPipe FaceMesh indices, named from the perspective of the face MediaPipe sees.
MP_RIGHT_EYE_OUTER = 33
MP_RIGHT_EYE_INNER = 133
MP_LEFT_EYE_INNER = 362
MP_LEFT_EYE_OUTER = 263
MP_RIGHT_IRIS = 468
MP_LEFT_IRIS = 473
MP_FOREHEAD = 10
MP_CHIN = 152

# Adult outer-eye-corner separation (exocanthion to exocanthion) ≈ 9 cm. Sets the distance
# scale; a ±10% person-to-person difference scales the estimated distance by ±10%.
DEFAULT_EYE_CORNER_SPAN_CM = 9.0
# Eyeball radius / eye width (corner to corner): ≈ 12 mm / 30 mm. Converts iris offset
# within the eye opening into an eye rotation; calibration fits a gain on top.
EYE_RADIUS_PER_WIDTH = 0.4


def rotation_from_angles(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Head → world rotation ``Ry(yaw) · Rx(pitch) · Rz(roll)`` (conventions above)."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def angles_from_rotation(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of ``rotation_from_angles``: (yaw, pitch, roll) in degrees."""
    f = r[:, 2]  # forward axis in world
    yaw = math.atan2(f[0], f[2])
    pitch = math.asin(max(-1.0, min(1.0, -f[1])))
    m = rotation_from_angles(math.degrees(yaw), math.degrees(pitch), 0.0).T @ r
    roll = math.atan2(m[1, 0], m[0, 0])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


@dataclass(frozen=True, slots=True, eq=False)
class WebcamHeadPose:
    rotation: np.ndarray  # head → world
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    position_cm: tuple[float, float, float]  # eye midpoint, world frame
    # Per eye (user's "left"/"right"): unit gaze direction in the head frame, from the iris
    # position within the eye opening. z > 0 = looking forward.
    eye_dirs: dict[str, np.ndarray]


def estimate_from_landmarks(
    landmarks: np.ndarray,
    width: int,
    height: int,
    *,
    mirrored: bool,
    focal_length_px: float,
    principal_point: tuple[float, float] | None = None,
    eye_corner_span_cm: float = DEFAULT_EYE_CORNER_SPAN_CM,
) -> WebcamHeadPose | None:
    """Head rotation, head position and eye-in-head gaze from normalised landmarks (N×3)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] <= max(MP_LEFT_IRIS, MP_RIGHT_IRIS):
        return None
    s = 1.0 if mirrored else -1.0  # world x = s · image x

    # MediaPipe's "right" is the user's right only in an unmirrored image.
    if mirrored:
        user = {"right": (MP_LEFT_EYE_OUTER, MP_LEFT_EYE_INNER, MP_LEFT_IRIS),
                "left": (MP_RIGHT_EYE_OUTER, MP_RIGHT_EYE_INNER, MP_RIGHT_IRIS)}
    else:
        user = {"right": (MP_RIGHT_EYE_OUTER, MP_RIGHT_EYE_INNER, MP_RIGHT_IRIS),
                "left": (MP_LEFT_EYE_OUTER, MP_LEFT_EYE_INNER, MP_LEFT_IRIS)}
    r_outer, l_outer = user["right"][0], user["left"][0]

    cx, cy = principal_point if principal_point is not None else (width / 2.0, height / 2.0)
    u, v = lm[:, 0] * width, lm[:, 1] * height
    rel_depth_px = lm[:, 2] * width  # same scale as x, in pixels at the face's depth

    def back_project(distance: float) -> np.ndarray:
        """Landmarks in world coordinates (cm), full perspective."""
        z = distance * (1.0 + rel_depth_px / focal_length_px)
        return np.stack(
            [s * (u - cx) / focal_length_px * z, (v - cy) / focal_length_px * z, -z], axis=1
        )

    # Distance from the known eye-corner span: a weak-perspective guess, then one refinement
    # on the back-projected 3D span (which no longer shrinks when the head turns).
    span_img = math.hypot(u[r_outer] - u[l_outer], v[r_outer] - v[l_outer])
    span_img = math.hypot(span_img, rel_depth_px[r_outer] - rel_depth_px[l_outer])
    if span_img < 1e-6:
        return None
    distance = focal_length_px * eye_corner_span_cm / span_img
    pts = back_project(distance)
    span_cm = float(np.linalg.norm(pts[r_outer] - pts[l_outer]))
    if span_cm < 1e-9:
        return None
    distance *= eye_corner_span_cm / span_cm
    pts = back_project(distance)

    right_axis = pts[r_outer] - pts[l_outer]
    down_axis = pts[MP_CHIN] - pts[MP_FOREHEAD]
    if np.linalg.norm(down_axis) < 1e-9:
        return None
    x_axis = right_axis / np.linalg.norm(right_axis)
    y_axis = down_axis - (down_axis @ x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    yaw, pitch, roll = angles_from_rotation(rot)
    mid = (pts[r_outer] + pts[l_outer]) / 2.0
    position = (float(mid[0]), float(mid[1]), float(mid[2]))

    eye_dirs: dict[str, np.ndarray] = {}
    for side, (outer, inner, iris) in user.items():
        center = (pts[outer] + pts[inner]) / 2.0
        eye_width = float(np.linalg.norm(pts[outer] - pts[inner]))
        if eye_width < 1e-6:
            continue
        offset = pts[iris] - center
        sx = (offset @ x_axis) / eye_width / EYE_RADIUS_PER_WIDTH
        sy = (offset @ y_axis) / eye_width / EYE_RADIUS_PER_WIDTH
        n = math.hypot(sx, sy)
        if n > 0.95:  # clamp implausible rotations instead of producing NaNs
            sx, sy = sx * 0.95 / n, sy * 0.95 / n
        eye_dirs[side] = np.array([sx, sy, math.sqrt(1.0 - sx * sx - sy * sy)])

    return WebcamHeadPose(rot, yaw, pitch, roll, position, eye_dirs)

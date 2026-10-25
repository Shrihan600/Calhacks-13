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

# Landmarks that move with the skull, not with expression: forehead, nose bridge and tip,
# eye corners, cheekbones and temples. No mouth, jaw (it moves when talking), brows or lids.
RIGID_LANDMARKS = (
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1,  # midline: forehead → nose tip
    67, 297, 109, 338, 54, 284, 103, 332,  # forehead
    21, 251, 127, 356, 234, 454,  # temples, face sides
    33, 133, 362, 263,  # eye corners
    116, 345, 117, 346, 123, 352, 50, 280,  # cheekbones
    98, 327,  # nostrils
)


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


def _similarity_fit(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(R, t, scale, rms) minimising Σ‖scale·R·aᵢ + t − bᵢ‖² (Umeyama; rows are points)."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T)) or 1.0])
    rot = vt.T @ d @ u.T
    scale = float(np.trace(np.diag(sv) @ d) / np.sum(a0 * a0))
    t = cb - scale * rot @ ca
    rms = float(np.sqrt(np.mean(np.sum((scale * a @ rot.T + t - b) ** 2, axis=1))))
    return rot, t, scale, rms


class RigidFaceFit:
    """Per-user face template for a head pose fitted to all ``RIGID_LANDMARKS``.

    The 4-point pose (eye corners, forehead, chin) carries the full noise of each of those
    landmarks, and the chin moves whenever the user talks. This learns the user's face shape
    in the head frame by averaging 4-point poses over the first ``frames`` near-frontal
    frames; from then on each frame's pose is a similarity fit of that template to ~40
    landmarks, which averages their noise out. Distance comes from the fitted scale, and the
    eye centres from the template, instead of from two jittery corners each.

    The template is rebuilt once it stops fitting for a while (someone else sat down).
    """

    def __init__(self, frames: int = 30, *, max_rms: float = 0.08, max_bad: int = 15) -> None:
        self.frames = frames
        self.max_rms = max_rms  # bad fit: rms residual above this × eye-corner span
        self.max_bad = max_bad  # consecutive bad fits before the template is rebuilt
        self.reset()

    def reset(self) -> None:
        self._samples: list[np.ndarray] = []
        self.template: np.ndarray | None = None  # RIGID_LANDMARKS × 3, head frame, cm
        self._bad = 0

    @property
    def ready(self) -> bool:
        return self.template is not None

    def add_sample(self, head_pts: np.ndarray) -> None:
        self._samples.append(head_pts)
        if len(self._samples) >= self.frames:
            self.template = np.mean(self._samples, axis=0)
            self._samples = []

    def fit(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(R, origin, scale) for world points of ``RIGID_LANDMARKS``; None on a bad fit."""
        assert self.template is not None
        rot, t, scale, rms = _similarity_fit(self.template, pts)
        span = float(np.linalg.norm(self.template[_TPL_R_OUTER] - self.template[_TPL_L_OUTER]))
        if not scale > 0 or rms > self.max_rms * span * scale:
            self._bad += 1
            if self._bad >= self.max_bad:
                self.reset()
            return None
        self._bad = 0
        return rot, t, scale

    def eye(self, outer: int, inner: int) -> tuple[np.ndarray, float]:
        """(centre, width) of an eye in the template, by its corner landmark indices."""
        assert self.template is not None
        o, i = self.template[_RIGID_INDEX[outer]], self.template[_RIGID_INDEX[inner]]
        return (o + i) / 2.0, float(np.linalg.norm(o - i))


_RIGID_INDEX = {lm: i for i, lm in enumerate(RIGID_LANDMARKS)}
# Either face's outer corners; the span between them doesn't depend on which is which.
_TPL_R_OUTER, _TPL_L_OUTER = _RIGID_INDEX[MP_RIGHT_EYE_OUTER], _RIGID_INDEX[MP_LEFT_EYE_OUTER]
# Template samples only from near-frontal poses, where the 4-point pose is most accurate.
_TEMPLATE_MAX_ANGLE_DEG = 20.0


def estimate_from_landmarks(
    landmarks: np.ndarray,
    width: int,
    height: int,
    *,
    mirrored: bool,
    focal_length_px: float,
    principal_point: tuple[float, float] | None = None,
    eye_corner_span_cm: float = DEFAULT_EYE_CORNER_SPAN_CM,
    face_fit: RigidFaceFit | None = None,
) -> WebcamHeadPose | None:
    """Head rotation, head position and eye-in-head gaze from normalised landmarks (N×3).

    With ``face_fit`` (and a full face mesh) the pose is a rigid fit of the user's face
    template once it has been learnt; until then, and without one, it comes from 4 points.
    """
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
    mid = (pts[r_outer] + pts[l_outer]) / 2.0

    # Eye centre and width in the head frame, per eye (4-point: from this frame's corners).
    eyes = {
        side: (rot.T @ ((pts[o] + pts[i]) / 2.0 - mid), float(np.linalg.norm(pts[o] - pts[i])))
        for side, (o, i, _) in user.items()
    }
    rigid = list(RIGID_LANDMARKS)
    if face_fit is not None and lm.shape[0] > max(RIGID_LANDMARKS):
        if not face_fit.ready:
            yaw, pitch, _ = angles_from_rotation(rot)
            if max(abs(yaw), abs(pitch)) < _TEMPLATE_MAX_ANGLE_DEG:
                face_fit.add_sample((pts[rigid] - mid) @ rot)
        elif (fitted := face_fit.fit(pts[rigid])) is not None:
            # Points scale with the assumed distance: correct it by the fitted scale, refit.
            distance /= fitted[2]
            pts = back_project(distance)
            if (fitted := face_fit.fit(pts[rigid])) is not None:
                rot, mid = fitted[0], fitted[1]
                eyes = {side: face_fit.eye(o, i) for side, (o, i, _) in user.items()}

    yaw, pitch, roll = angles_from_rotation(rot)
    position = (float(mid[0]), float(mid[1]), float(mid[2]))

    eye_dirs: dict[str, np.ndarray] = {}
    for side, (_, _, iris) in user.items():
        center, eye_width = eyes[side]
        if eye_width < 1e-6:
            continue
        offset = rot.T @ (pts[iris] - mid) - center
        sx = offset[0] / eye_width / EYE_RADIUS_PER_WIDTH
        sy = offset[1] / eye_width / EYE_RADIUS_PER_WIDTH
        n = math.hypot(sx, sy)
        if n > 0.95:  # clamp implausible rotations instead of producing NaNs
            sx, sy = sx * 0.95 / n, sy * 0.95 / n
        eye_dirs[side] = np.array([sx, sy, math.sqrt(1.0 - sx * sx - sy * sy)])

    return WebcamHeadPose(rot, yaw, pitch, roll, position, eye_dirs)

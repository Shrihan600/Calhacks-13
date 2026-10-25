"""Screen mappers for every mode, plus the factory that picks one for a profile.

Head pose arrives as two independently valid parts: rotation (webcam or IMU) and position
(webcam only). Both mappers use whatever the frame has; see ``vision/head_pose.py`` for the
world and head frames.

``PolynomialRidgeMapper`` (``geometry_estimation: data_driven``, any schema)
    Second-order polynomial of the eye features plus linear head rotation / position terms,
    by ridge regression. One sub-model is fitted per usable combination of eyes (both / left /
    right) and head parts (rotation, position), and prediction uses the richest one the
    current frame supports, so a blink in one eye or a lost head pose doesn't wreck the
    estimate. Head compensation is learned, so it only covers head movement seen during
    calibration.

``GeometricRayMapper`` (``geometry_estimation: geometric``, gaze-direction schemas)
    Rotates each eye's gaze direction into the world with a fitted eye-to-head rotation and
    the measured head rotation, starts the ray at the measured head position, and intersects
    it with the screen plane z = 0. Head movement is compensated by geometry, so it
    extrapolates to head poses calibration never saw.

    With a measured head position, the head camera's position on the screen
    (``camera_screen_pos``, default top centre as on a MacBook) anchors the plane-to-screen
    offset and only the screen scale is fitted. Fitting the offset too would leave a
    still-head calibration unable to tell a screen offset from a slightly wrong eye rotation,
    and the wrong choice only shows up once the head moves. Without head position (IMU only,
    or no head sensor) the eyes are assumed to stay put and the offset is fitted.

    Mode A's eye directions come from the iris position within the eye opening through an
    assumed eyeball size, so a horizontal and vertical gain is fitted on top.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import ClassVar

import numpy as np

from opengaze.calibration.model import GazeModel, ScreenMapper, decode_str
from opengaze.vision.head_pose import rotation_from_angles
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureLayout, SchemaId

_SIDES = ("left", "right")
# Smallest standard deviation used when z-scoring, per kind of feature, so a feature that
# barely moved during calibration (typically head pose) can't be amplified into noise.
_STD_FLOOR_EYE = {"position": 0.01, "direction": 0.01}
_STD_FLOOR_ROT = np.array([2.0, 2.0, 2.0])  # degrees
_STD_FLOOR_POS = np.array([1.0, 1.0, 2.0])  # cm

# Shared with the rest of the code base; kept under this name for the mapper's callers.
head_rotation = rotation_from_angles


def _eye_inputs(layout: FeatureLayout, x: np.ndarray, side: str) -> np.ndarray:
    """2 numbers per eye: pupil position, or gaze direction as (horizontal, vertical) angles."""
    vals = x[list(layout.eye_values[side])].astype(np.float64)
    if layout.eye_kind == "position":
        return vals
    gx, gy, gz = vals
    # Eye cameras see the gaze coming back toward them (z < 0); head-frame gaze has z > 0.
    return np.array([math.atan2(gx, abs(gz)), math.atan2(gy, abs(gz))])


def _valid_mask(x: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    return np.ones(x.shape[-1], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)


def _eye_valid(layout: FeatureLayout, valid: np.ndarray, side: str) -> bool:
    return bool(valid[list(layout.eye_values[side])].all())


def _rot_valid(layout: FeatureLayout, valid: np.ndarray) -> bool:
    return bool(valid[list(layout.head_rot)].all())


def _pos_valid(layout: FeatureLayout, valid: np.ndarray) -> bool:
    return bool(valid[list(layout.head_pos)].all())


def _poly2(z: np.ndarray) -> np.ndarray:
    """Linear and quadratic terms (no constant) of each row of z."""
    cols = [z]
    for i, j in combinations(range(z.shape[1]), 2):
        cols.append((z[:, i] * z[:, j])[:, None])
    cols.append(z**2)
    return np.hstack(cols)


def _key(eyes: tuple[str, ...], rot: bool, pos: bool) -> str:
    return "+".join(eyes) + ("|rot" if rot else "") + ("|pos" if pos else "")


class PolynomialRidgeMapper(ScreenMapper):
    kind: ClassVar[str] = "poly_ridge"

    def __init__(self, schema_id: SchemaId, alpha: float = 0.05, min_points: int = 5) -> None:
        self.schema_id = schema_id
        self.alpha = alpha
        self.min_points = min_points
        self.layout = FEATURE_LAYOUTS[schema_id]
        self._subs: dict[str, dict[str, np.ndarray]] = {}  # _key(...) → arrays

    @property
    def fitted(self) -> bool:
        return bool(self._subs)

    # ------------------------------------------------------------------

    def _design(self, rows: np.ndarray, eyes: tuple[str, ...], rot: bool, pos: bool) -> np.ndarray:
        eye_cols = [np.stack([_eye_inputs(self.layout, r, s) for r in rows]) for s in eyes]
        parts = [np.hstack(eye_cols)]
        if rot:
            parts.append(rows[:, list(self.layout.head_rot)].astype(np.float64))
        if pos:
            parts.append(rows[:, list(self.layout.head_pos)].astype(np.float64))
        return np.hstack(parts)

    def _floors(self, eyes: tuple[str, ...], rot: bool, pos: bool) -> np.ndarray:
        f = [np.full(2 * len(eyes), _STD_FLOOR_EYE[self.layout.eye_kind])]
        if rot:
            f.append(_STD_FLOOR_ROT)
        if pos:
            f.append(_STD_FLOOR_POS)
        return np.concatenate(f)

    def _expand(self, z: np.ndarray, n_eye: int) -> np.ndarray:
        """Polynomial of the eye inputs; head inputs stay linear (first-order compensation)."""
        return np.hstack([_poly2(z[:, :n_eye]), z[:, n_eye:]])

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")
        valid = np.ones_like(X, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        rot_rows = np.array([_rot_valid(self.layout, v) for v in valid])
        pos_rows = np.array([_pos_valid(self.layout, v) for v in valid])
        self._subs = {}
        for n in (2, 1):
            for eyes in combinations(_SIDES, n):
                eye_rows = np.array(
                    [all(_eye_valid(self.layout, v, s) for s in eyes) for v in valid]
                )
                for rot in (True, False):
                    for pos in (True, False):
                        rows = eye_rows & (rot_rows if rot else True) & (pos_rows if pos else True)
                        if rows.sum() >= self.min_points:
                            self._fit_sub(X[rows], y[rows], eyes, rot, pos)
        if not self._subs:
            raise ValueError(
                f"no eye had valid features at {self.min_points}+ calibration points"
            )

    def _fit_sub(
        self, X: np.ndarray, y: np.ndarray, eyes: tuple[str, ...], rot: bool, pos: bool
    ) -> None:
        raw = self._design(X, eyes, rot, pos)
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), self._floors(eyes, rot, pos))
        phi = self._expand((raw - mean) / std, 2 * len(eyes))
        phi_b = np.hstack([phi, np.ones((phi.shape[0], 1))])
        a = phi_b.T @ phi_b
        a[:-1, :-1] += self.alpha * np.eye(phi.shape[1])
        w = np.linalg.solve(a, phi_b.T @ y.astype(np.float64))
        self._subs[_key(eyes, rot, pos)] = {"mean": mean, "std": std, "w": w}

    def _choose(self, valid: np.ndarray) -> tuple[tuple[str, ...], bool, bool] | None:
        eyes_ok = tuple(s for s in _SIDES if _eye_valid(self.layout, valid, s))
        rot_opts = (True, False) if _rot_valid(self.layout, valid) else (False,)
        pos_opts = (True, False) if _pos_valid(self.layout, valid) else (False,)
        for n in (2, 1):
            for eyes in combinations(eyes_ok, n):
                for rot in rot_opts:
                    for pos in pos_opts:
                        if _key(eyes, rot, pos) in self._subs:
                            return eyes, rot, pos
        return None

    def predict_xy(
        self, x: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        if not self._subs:
            raise RuntimeError("model not fitted")
        choice = self._choose(_valid_mask(x, valid))
        if choice is None:
            return None
        eyes, rot, pos = choice
        sub = self._subs[_key(eyes, rot, pos)]
        raw = self._design(x[None, :], eyes, rot, pos)
        phi = self._expand((raw - sub["mean"]) / sub["std"], 2 * len(eyes))
        pred = np.hstack([phi, np.ones((1, 1))]) @ sub["w"]
        return float(pred[0, 0]), float(pred[0, 1])

    def confidence(self, fv: object) -> float:
        """Mean confidence of the eyes used, scaled down when a fitted head part is missing."""
        values, valid = fv.values, fv.valid  # type: ignore[attr-defined]
        choice = self._choose(valid)
        if choice is None:
            return 0.0
        eyes, rot, pos = choice
        conf = float(np.mean([values[self.layout.eye_conf[s]] for s in eyes]))
        if not rot and any("|rot" in k for k in self._subs):
            conf *= 0.8
        if not pos and any("|pos" in k for k in self._subs):
            conf *= 0.9
        return conf

    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray | str | float]:
        if not self._subs:
            raise RuntimeError("model not fitted")
        out: dict[str, np.ndarray | str | float] = {
            "mapper_kind": self.kind,
            "schema_id": self.schema_id,
            "alpha": np.float64(self.alpha),
            "sub_keys": json.dumps(sorted(self._subs)),
        }
        for i, key in enumerate(sorted(self._subs)):
            for name, arr in self._subs[key].items():
                out[f"sub{i}_{name}"] = arr
        return out

    @classmethod
    def from_arrays(cls, arrays: dict) -> PolynomialRidgeMapper:
        m = cls(decode_str(arrays["schema_id"]), alpha=float(arrays["alpha"]))  # type: ignore[arg-type]
        for i, key in enumerate(json.loads(decode_str(arrays["sub_keys"]))):
            m._subs[key] = {n: np.asarray(arrays[f"sub{i}_{n}"]) for n in ("mean", "std", "w")}
        return m


# ---------------------------------------------------------------------------
# Geometric
# ---------------------------------------------------------------------------


def kabsch(
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray | None = None,
    *,
    allow_reflection: bool = False,
) -> np.ndarray:
    """Rotation R minimising Σ w‖R·aᵢ − bᵢ‖² (rows of a and b are vectors).

    With ``allow_reflection`` the best orthogonal matrix is returned, which may be a
    reflection: needed when an eye image was mirrored by the profile (``mirror_horizontal``),
    which flips the handedness of eye-camera coordinates.
    """
    w = np.ones(len(a)) if w is None else w
    h = (a * w[:, None]).T @ b
    u, _, vt = np.linalg.svd(h)
    if allow_reflection:
        return vt.T @ u.T
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _rodrigues(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def _levenberg_marquardt(fun, p0: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Minimise ‖fun(p)‖² (small problems: numerical Jacobian)."""
    p = p0.astype(np.float64).copy()
    r = fun(p)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iterations):
        jac = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros_like(p)
            dp[j] = 1e-6 * max(1.0, abs(p[j]))
            jac[:, j] = (fun(p + dp) - fun(p - dp)) / (2 * dp[j])
        jtj, jtr = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(10):
            step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), -jtr)
            r_new = fun(p + step)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                p, r, cost = p + step, r_new, cost_new
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 4
        if not improved or float(np.linalg.norm(step)) < 1e-10:
            break
    return p


class GeometricRayMapper(ScreenMapper):
    kind: ClassVar[str] = "geometric"

    def __init__(
        self,
        schema_id: SchemaId,
        *,
        distance_cm: float = 60.0,
        camera_screen_pos: tuple[float, float] = (0.5, 0.0),
        iterations: int = 15,
    ) -> None:
        layout = FEATURE_LAYOUTS[schema_id]
        if layout.eye_kind != "direction":
            raise ValueError(f"geometric mapping needs gaze directions; {schema_id} has none")
        self.schema_id = schema_id
        self.layout = layout
        self.distance_cm = distance_cm  # assumed eye distance when head position is unknown
        self.iterations = iterations
        self.fit_gain = layout.eye_frame == "head"
        self._r_ce: dict[str, np.ndarray] = {}  # eye frame → head, per eye (may be a reflection)
        # sx, ox, sy, oy:  screen = s·P + o per axis.
        self._screen = np.array([1.0, camera_screen_pos[0], 1.0, camera_screen_pos[1]])
        self._gain = np.ones(2)
        self._rot0 = np.zeros(3)  # baseline head angles (calibration mean)
        self._pos0 = np.array([0.0, 0.0, -distance_cm])  # baseline head position
        self._has_rot = False
        self._has_pos = False

    @property
    def fitted(self) -> bool:
        return bool(self._r_ce)

    # ------------------------------------------------------------------

    def _head(self, x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(rotation relative to the calibration baseline, ray origin) for one row."""
        r_rel = np.eye(3)
        if self._has_rot and _rot_valid(self.layout, valid):
            angles = x[list(self.layout.head_rot)].astype(np.float64)
            r_rel = rotation_from_angles(*angles) @ rotation_from_angles(*self._rot0).T
        origin = self._pos0
        if self._has_pos and _pos_valid(self.layout, valid):
            origin = x[list(self.layout.head_pos)].astype(np.float64)
        return r_rel, origin

    def _eye_dir(self, x: np.ndarray, side: str, gain: np.ndarray | None = None) -> np.ndarray:
        v = x[list(self.layout.eye_values[side])].astype(np.float64)
        g = self._gain if gain is None else gain
        if self.fit_gain and abs(v[2]) > 1e-6:
            t = v[:2] / v[2]
            v = np.array([g[0] * t[0], g[1] * t[1], 1.0]) * math.copysign(1.0, v[2])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    @staticmethod
    def _hit(origin: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Where the ray origin + t·w meets z = 0 (rays pointing away are pushed far out)."""
        t = -origin[2] / max(w[2], 1e-3)
        return (origin + t * w)[:2]

    def _plane_point(self, x: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
        r_rel, origin = self._head(x, valid)
        hits = [
            self._hit(origin, r_rel @ self._r_ce[s] @ self._eye_dir(x, s))
            for s in _SIDES
            if s in self._r_ce and _eye_valid(self.layout, valid, s)
        ]
        return np.mean(hits, axis=0) if hits else None

    def _to_screen(self, p: np.ndarray) -> np.ndarray:
        sx, ox, sy, oy = self._screen
        return np.array([sx * p[0] + ox, sy * p[1] + oy])

    def _from_screen(self, target: np.ndarray) -> np.ndarray:
        sx, ox, sy, oy = self._screen
        return np.array([(target[0] - ox) / sx, (target[1] - oy) / sy])

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        valid = np.ones_like(X, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        n = X.shape[0]
        if n < 4:
            raise ValueError("geometric mapping needs at least 4 calibration points")
        y = y.astype(np.float64)
        rot_rows = np.array([_rot_valid(self.layout, v) for v in valid])
        pos_rows = np.array([_pos_valid(self.layout, v) for v in valid])
        self._has_rot, self._has_pos = bool(rot_rows.any()), bool(pos_rows.any())
        if self._has_rot:
            self._rot0 = X[rot_rows][:, list(self.layout.head_rot)].astype(np.float64).mean(0)
        if self._has_pos:
            self._pos0 = X[pos_rows][:, list(self.layout.head_pos)].astype(np.float64).mean(0)
        self._gain = np.ones(2)

        per_eye = {s: np.array([_eye_valid(self.layout, v, s) for v in valid]) for s in _SIDES}
        per_eye = {s: np.flatnonzero(rows) for s, rows in per_eye.items() if rows.sum() >= 4}
        if not per_eye:
            raise ValueError("no eye had a valid gaze direction at 4+ calibration points")

        # 1) Initial guess, refined by alternation: the mean gaze direction looks straight at
        #    the screen (+z); then fit the screen scale (offset too without head position),
        #    then the rotation each eye needed to hit every target, and repeat.
        self._r_ce = {}
        for s, idx in per_eye.items():
            mean = np.stack([self._eye_dir(X[i], s) for i in idx]).mean(axis=0)
            self._r_ce[s] = _align(mean / np.linalg.norm(mean), np.array([0.0, 0.0, 1.0]))
        for _ in range(self.iterations):
            self._fit_screen(X, y, valid)
            for s, idx in per_eye.items():
                a_vecs, b_vecs = [], []
                for i in idx:
                    r_rel, origin = self._head(X[i], valid[i])
                    u = np.array([*self._from_screen(y[i]), 0.0]) - origin
                    a_vecs.append(self._eye_dir(X[i], s))
                    b_vecs.append(r_rel.T @ (u / np.linalg.norm(u)))
                self._r_ce[s] = kabsch(np.stack(a_vecs), np.stack(b_vecs), allow_reflection=True)

        # 2) Joint least-squares refinement of everything, with per-eye residuals.
        sides = list(per_eye)
        base = {s: self._r_ce[s].copy() for s in sides}
        heads = [self._head(X[i], valid[i]) for i in range(n)]
        free_offset = not self._has_pos
        j_screen = 3 * len(sides)
        j_offset = j_screen + 2
        j_gain = j_offset + (2 if free_offset else 0)

        def unpack(p: np.ndarray) -> None:
            for k, s in enumerate(sides):
                self._r_ce[s] = base[s] @ _rodrigues(p[3 * k : 3 * k + 3])
            self._screen[0], self._screen[2] = p[j_screen], p[j_screen + 1]
            if free_offset:
                self._screen[1], self._screen[3] = p[j_offset], p[j_offset + 1]
            if self.fit_gain:
                self._gain = p[j_gain : j_gain + 2]

        def residuals(p: np.ndarray) -> np.ndarray:
            unpack(p)
            res = []
            for s in sides:
                for i in per_eye[s]:
                    r_rel, origin = heads[i]
                    hit = self._hit(origin, r_rel @ self._r_ce[s] @ self._eye_dir(X[i], s))
                    res.append(self._to_screen(hit) - y[i])
            return np.concatenate(res)

        p0 = [np.zeros(3 * len(sides)), [self._screen[0], self._screen[2]]]
        if free_offset:
            p0.append([self._screen[1], self._screen[3]])
        if self.fit_gain:
            p0.append(self._gain)
        unpack(_levenberg_marquardt(residuals, np.concatenate(p0)))

    def _fit_screen(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray) -> None:
        pts = [self._plane_point(X[i], valid[i]) for i in range(X.shape[0])]
        used = [i for i, p in enumerate(pts) if p is not None]
        p_arr = np.stack([pts[i] for i in used])
        y_arr = y[used]
        for axis in (0, 1):
            px = p_arr[:, axis]
            if self._has_pos:  # offset anchored at the camera's screen position
                o_ = self._screen[2 * axis + 1]
                denom = float(px @ px)
                s_ = float(px @ (y_arr[:, axis] - o_)) / denom if denom > 0 else 0.0
            else:  # eye position unknown: fit the offset as well
                a = np.vstack([px, np.ones(len(used))]).T
                (s_, o_), *_ = np.linalg.lstsq(a, y_arr[:, axis], rcond=None)
                self._screen[2 * axis + 1] = o_
            if abs(s_) < 1e-9:
                raise ValueError("calibration targets don't constrain the screen scale")
            self._screen[2 * axis] = s_

    def predict_xy(
        self, x: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        if not self._r_ce:
            raise RuntimeError("model not fitted")
        p = self._plane_point(x, _valid_mask(x, valid))
        if p is None:
            return None
        sx, sy = self._to_screen(p)
        return float(sx), float(sy)

    def confidence(self, fv: object) -> float:
        values, valid = fv.values, fv.valid  # type: ignore[attr-defined]
        confs = [
            float(values[self.layout.eye_conf[s]])
            for s in self._r_ce
            if _eye_valid(self.layout, valid, s)
        ]
        if not confs:
            return 0.0
        conf = float(np.mean(confs))
        if self._has_rot and not _rot_valid(self.layout, valid):
            conf *= 0.8
        if self._has_pos and not _pos_valid(self.layout, valid):
            conf *= 0.9
        return conf

    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray | str | float]:
        if not self._r_ce:
            raise RuntimeError("model not fitted")
        out: dict[str, np.ndarray | str | float] = {
            "mapper_kind": self.kind,
            "schema_id": self.schema_id,
            "distance_cm": np.float64(self.distance_cm),
            "screen": self._screen,
            "gain": self._gain,
            "rot0": self._rot0,
            "pos0": self._pos0,
            "has_rot": np.bool_(self._has_rot),
            "has_pos": np.bool_(self._has_pos),
        }
        for s, r in self._r_ce.items():
            out[f"r_ce_{s}"] = r
        return out

    @classmethod
    def from_arrays(cls, arrays: dict) -> GeometricRayMapper:
        m = cls(decode_str(arrays["schema_id"]), distance_cm=float(arrays["distance_cm"]))  # type: ignore[arg-type]
        m._screen = np.asarray(arrays["screen"], dtype=np.float64)
        m._gain = np.asarray(arrays["gain"], dtype=np.float64)
        m._rot0 = np.asarray(arrays["rot0"], dtype=np.float64)
        m._pos0 = np.asarray(arrays["pos0"], dtype=np.float64)
        m._has_rot = bool(arrays["has_rot"])
        m._has_pos = bool(arrays["has_pos"])
        m._r_ce = {s: np.asarray(arrays[f"r_ce_{s}"]) for s in _SIDES if f"r_ce_{s}" in arrays}
        return m


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MAPPERS: dict[str, type[ScreenMapper]] = {
    GazeModel.kind: GazeModel,
    PolynomialRidgeMapper.kind: PolynomialRidgeMapper,
    GeometricRayMapper.kind: GeometricRayMapper,
}


def mapper_kind_for(schema_id: str, geometry_estimation: str = "data_driven") -> str:
    if geometry_estimation == "geometric":
        return GeometricRayMapper.kind
    return PolynomialRidgeMapper.kind


def make_mapper(schema_id: str, geometry_estimation: str = "data_driven") -> ScreenMapper:
    """A new, unfitted mapper for this schema and calibration policy."""
    cls = MAPPERS[mapper_kind_for(schema_id, geometry_estimation)]
    return cls(schema_id)  # type: ignore[call-arg]


def mapper_from_arrays(arrays: dict) -> ScreenMapper:
    """Load any mapper; files written before mappers existed have no kind and are ridge."""
    kind = decode_str(arrays["mapper_kind"]) if "mapper_kind" in arrays else GazeModel.kind
    try:
        cls = MAPPERS[kind]
    except KeyError:
        raise ValueError(f"calibration uses unknown mapper {kind!r}") from None
    return cls.from_arrays(arrays)

"""Screen mappers for Modes B/C, plus the factory that picks one for a profile.

``PolynomialRidgeMapper`` (``geometry_estimation: data_driven``, either eye pipeline)
    Second-order polynomial of the eye features plus linear head-pose terms, by ridge
    regression. One sub-model is fitted per usable eye subset (both / left only / right only,
    each with and without head pose), and prediction uses the richest one the current frame's
    validity mask supports, so a blink in one eye doesn't wreck the estimate.

``GeometricRayMapper`` (``geometry_estimation: geometric``, ``eye_3d`` pipeline only)
    Rotates each eye's 3D gaze vector from eye-camera into world coordinates using a fitted
    camera-to-head rotation and the head pose, then intersects the ray with the screen plane.
    Head movement is compensated by geometry rather than learned, so it extrapolates to head
    poses calibration never saw — as long as the head pose signs follow the
    ``HeadObservation`` contract.

World frame (geometric mapper): x right and y down along the screen, z from the user into
the screen. The screen is the plane z = 0 with the head camera at the origin; the user's
eyes are at z = −D.

Where the eyes are matters, not just where they point. With a head camera, the face's
position in its (normalised, mirrored-to-match-the-screen) image gives the eye position
directly, and the camera's position on the screen (``camera_screen_pos``, default top centre
as on a MacBook) anchors the plane-to-screen offset, so only the screen scale and the
eye-camera rotations are fitted. Fitting the offset too would leave a still-head calibration
unable to tell a screen offset from a slightly wrong eye-camera rotation, and the wrong choice
only shows up once the head turns. Without a head camera the head can't be compensated
anyway, so the eye position is fitted as a free offset.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import ClassVar

import numpy as np

from opengaze.calibration.model import GazeModel, ScreenMapper, decode_str
from opengaze.vision.observations import FEATURE_LAYOUTS, FeatureLayout, SchemaId

_SIDES = ("left", "right")
# Smallest standard deviation used when z-scoring, per kind of feature, so a feature that
# barely moved during calibration (typically head pose) can't be amplified into noise.
_STD_FLOOR_EYE = {"position": 0.01, "direction": 0.01}
_STD_FLOOR_HEAD = np.array([2.0, 2.0, 2.0, 0.02, 0.02, 0.01])


def _eye_inputs(layout: FeatureLayout, x: np.ndarray, side: str) -> np.ndarray:
    """2 numbers per eye: pupil position, or gaze direction as (horizontal, vertical) angles."""
    vals = x[list(layout.eye_values[side])].astype(np.float64)
    if layout.eye_kind == "position":
        return vals
    gx, gy, gz = vals
    return np.array([math.atan2(gx, -gz), math.atan2(gy, -gz)])


def _valid_mask(x: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    return np.ones(x.shape[-1], dtype=bool) if valid is None else np.asarray(valid, dtype=bool)


def _eye_valid(layout: FeatureLayout, valid: np.ndarray, side: str) -> bool:
    return bool(valid[list(layout.eye_values[side])].all())


def _head_valid(layout: FeatureLayout, valid: np.ndarray) -> bool:
    return bool(valid[list(layout.head)].all())


def _poly2(z: np.ndarray) -> np.ndarray:
    """Linear and quadratic terms (no constant) of each row of z."""
    cols = [z]
    for i, j in combinations(range(z.shape[1]), 2):
        cols.append((z[:, i] * z[:, j])[:, None])
    cols.append(z**2)
    return np.hstack(cols)


class PolynomialRidgeMapper(ScreenMapper):
    kind: ClassVar[str] = "poly_ridge"

    def __init__(self, schema_id: SchemaId, alpha: float = 0.05, min_points: int = 5) -> None:
        self.schema_id = schema_id
        self.alpha = alpha
        self.min_points = min_points
        self.layout = FEATURE_LAYOUTS[schema_id]
        # key "left+right|head" → arrays
        self._subs: dict[str, dict[str, np.ndarray]] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._subs)

    # ------------------------------------------------------------------

    def _design(self, rows: np.ndarray, eyes: tuple[str, ...], head: bool) -> np.ndarray:
        eye_cols = [np.stack([_eye_inputs(self.layout, r, s) for r in rows]) for s in eyes]
        parts = [np.hstack(eye_cols)]
        if head:
            parts.append(rows[:, list(self.layout.head)].astype(np.float64))
        return np.hstack(parts)

    def _floors(self, eyes: tuple[str, ...], head: bool) -> np.ndarray:
        f = [np.full(2 * len(eyes), _STD_FLOOR_EYE[self.layout.eye_kind])]
        if head:
            f.append(_STD_FLOOR_HEAD)
        return np.concatenate(f)

    def _expand(self, z: np.ndarray, n_eye: int) -> np.ndarray:
        """Polynomial of the eye inputs; head inputs stay linear (first-order compensation)."""
        return np.hstack([_poly2(z[:, :n_eye]), z[:, n_eye:]])

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit")
        valid = np.ones_like(X, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self._subs = {}
        for n in (2, 1):
            for eyes in combinations(_SIDES, n):
                eye_rows = np.array(
                    [all(_eye_valid(self.layout, v, s) for s in eyes) for v in valid]
                )
                for head in (True, False):
                    rows = eye_rows & (
                        np.array([_head_valid(self.layout, v) for v in valid]) if head else True
                    )
                    if rows.sum() < self.min_points:
                        continue
                    self._fit_sub(X[rows], y[rows], eyes, head)
        if not self._subs:
            raise ValueError(
                f"no eye had valid features at {self.min_points}+ calibration points"
            )

    def _fit_sub(self, X: np.ndarray, y: np.ndarray, eyes: tuple[str, ...], head: bool) -> None:
        raw = self._design(X, eyes, head)
        mean = raw.mean(axis=0)
        std = np.maximum(raw.std(axis=0), self._floors(eyes, head))
        phi = self._expand((raw - mean) / std, 2 * len(eyes))
        phi_b = np.hstack([phi, np.ones((phi.shape[0], 1))])
        a = phi_b.T @ phi_b
        a[:-1, :-1] += self.alpha * np.eye(phi.shape[1])
        w = np.linalg.solve(a, phi_b.T @ y.astype(np.float64))
        self._subs[_key(eyes, head)] = {"mean": mean, "std": std, "w": w}

    def _choose(self, valid: np.ndarray) -> tuple[tuple[str, ...], bool] | None:
        eyes_ok = tuple(s for s in _SIDES if _eye_valid(self.layout, valid, s))
        head_ok = _head_valid(self.layout, valid)
        for n in (2, 1):
            for eyes in combinations(eyes_ok, n):
                for head in ((True, False) if head_ok else (False,)):
                    if _key(eyes, head) in self._subs:
                        return eyes, head
        return None

    def predict_xy(
        self, x: np.ndarray, valid: np.ndarray | None = None
    ) -> tuple[float, float] | None:
        if not self._subs:
            raise RuntimeError("model not fitted")
        choice = self._choose(_valid_mask(x, valid))
        if choice is None:
            return None
        eyes, head = choice
        sub = self._subs[_key(eyes, head)]
        raw = self._design(x[None, :], eyes, head)
        phi = self._expand((raw - sub["mean"]) / sub["std"], 2 * len(eyes))
        pred = np.hstack([phi, np.ones((1, 1))]) @ sub["w"]
        return float(pred[0, 0]), float(pred[0, 1])

    def confidence(self, fv: object) -> float:
        """Mean confidence of the eyes used, scaled down when head pose is missing."""
        valid = fv.valid  # type: ignore[attr-defined]
        choice = self._choose(valid)
        if choice is None:
            return 0.0
        eyes, head = choice
        conf = float(np.mean([fv.values[self.layout.eye_conf[s]] for s in eyes]))  # type: ignore[attr-defined]
        has_head_model = any(k.endswith("|head") for k in self._subs)
        return conf * (0.8 if has_head_model and not head else 1.0)

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


def _key(eyes: tuple[str, ...], head: bool) -> str:
    return "+".join(eyes) + ("|head" if head else "|nohead")


# ---------------------------------------------------------------------------
# Geometric
# ---------------------------------------------------------------------------


def head_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Head-to-world rotation from ``HeadObservation`` angles, in the world frame above.

    Yaw right turns the forward axis (+z) toward +x, pitch up turns it toward −y (up), and
    roll right turns the head's up axis (−y) toward +x.
    """
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


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
        head_camera_hfov_deg: float = 60.0,
        head_camera_aspect: float = 9.0 / 16.0,
        camera_screen_pos: tuple[float, float] = (0.5, 0.0),
        iterations: int = 15,
    ) -> None:
        layout = FEATURE_LAYOUTS[schema_id]
        if layout.eye_kind != "direction":
            raise ValueError(f"geometric mapping needs 3D gaze vectors; {schema_id} has none")
        self.schema_id = schema_id
        self.layout = layout
        self.distance_cm = distance_cm
        self.kx = 2.0 * math.tan(math.radians(head_camera_hfov_deg) / 2.0)
        self.ky = self.kx * head_camera_aspect
        self.iterations = iterations
        self._r_ce: dict[str, np.ndarray] = {}  # eye camera → head, per eye (may be a reflection)
        # sx, ox, sy, oy:  screen = s·P + o per axis.
        self._screen = np.array([1.0, camera_screen_pos[0], 1.0, camera_screen_pos[1]])
        self._head0 = np.zeros(6)  # baseline yaw, pitch, roll, centre x, centre y, scale
        self._has_head = False
        self._eye_offset = np.zeros(2)  # fitted eye position (cm) when there is no head pose

    @property
    def fitted(self) -> bool:
        return bool(self._r_ce)

    # ------------------------------------------------------------------

    def _head(self, x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(R_rel, ray origin) for one row; baseline pose when head pose is missing."""
        if not self._has_head:
            return np.eye(3), np.array([*self._eye_offset, -self.distance_cm])
        h = (
            x[list(self.layout.head)].astype(np.float64)
            if _head_valid(self.layout, valid)
            else self._head0
        )
        r_rel = head_rotation(*h[:3]) @ head_rotation(*self._head0[:3]).T
        scale = h[5] if h[5] > 0 else self._head0[5]
        d = self.distance_cm * (self._head0[5] / scale if scale > 0 else 1.0)
        # Face position relative to the head camera's optical axis.
        origin = np.array([self.kx * d * (h[3] - 0.5), self.ky * d * (h[4] - 0.5), -d])
        return r_rel, origin

    def _eye_dir(self, x: np.ndarray, side: str) -> np.ndarray:
        v = x[list(self.layout.eye_values[side])].astype(np.float64)
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

    def fit(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray | None = None) -> None:
        valid = np.ones_like(X, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        n = X.shape[0]
        if n < 4:
            raise ValueError("geometric mapping needs at least 4 calibration points")
        y = y.astype(np.float64)
        head_rows = np.array([_head_valid(self.layout, v) for v in valid])
        self._has_head = bool(head_rows.any())
        if self._has_head:
            self._head0 = X[head_rows][:, list(self.layout.head)].astype(np.float64).mean(axis=0)
        self._eye_offset = np.zeros(2)

        per_eye = {s: np.array([_eye_valid(self.layout, v, s) for v in valid]) for s in _SIDES}
        per_eye = {s: np.flatnonzero(rows) for s, rows in per_eye.items() if rows.sum() >= 4}
        if not per_eye:
            raise ValueError("no eye had a valid gaze vector at 4+ calibration points")
        dirs = {s: np.stack([self._eye_dir(X[i], s) for i in idx]) for s, idx in per_eye.items()}

        # 1) Initial guess, refined by alternation: the mean gaze direction looks straight at
        #    the screen (+z); then fit the screen scale (offset too without head pose), then
        #    the rotation each eye needed to hit every target, and repeat.
        self._r_ce = {}
        for s in per_eye:
            mean = dirs[s].mean(axis=0)
            self._r_ce[s] = _align(mean / np.linalg.norm(mean), np.array([0.0, 0.0, 1.0]))
        for _ in range(self.iterations):
            self._fit_screen(X, y, valid)
            for s, idx in per_eye.items():
                b_vecs = []
                for i in idx:
                    r_rel, origin = self._head(X[i], valid[i])
                    u = np.array([*self._from_screen(y[i]), 0.0]) - origin
                    b_vecs.append(r_rel.T @ (u / np.linalg.norm(u)))
                self._r_ce[s] = kabsch(dirs[s], np.stack(b_vecs), allow_reflection=True)

        # 2) Joint least-squares refinement of everything, per-eye residuals.
        sides = list(per_eye)
        base = {s: self._r_ce[s].copy() for s in sides}
        heads = [self._head(X[i], valid[i]) for i in range(n)]
        free_offset = not self._has_head

        def unpack(p: np.ndarray) -> None:
            for k, s in enumerate(sides):
                self._r_ce[s] = base[s] @ _rodrigues(p[3 * k : 3 * k + 3])
            j = 3 * len(sides)
            self._screen[0], self._screen[2] = p[j], p[j + 1]
            if free_offset:
                self._eye_offset = p[j + 2 : j + 4]

        def residuals(p: np.ndarray) -> np.ndarray:
            unpack(p)
            res = []
            for s in sides:
                for row, i in enumerate(per_eye[s]):
                    r_rel, origin = (
                        heads[i] if not free_offset else self._head(X[i], valid[i])
                    )
                    hit = self._hit(origin, r_rel @ self._r_ce[s] @ dirs[s][row])
                    res.append(self._to_screen(hit) - y[i])
            return np.concatenate(res)

        p0 = np.concatenate(
            [np.zeros(3 * len(sides)), [self._screen[0], self._screen[2]]]
            + ([self._eye_offset] if free_offset else [])
        )
        unpack(_levenberg_marquardt(residuals, p0))

    def _fit_screen(self, X: np.ndarray, y: np.ndarray, valid: np.ndarray) -> None:
        pts = [self._plane_point(X[i], valid[i]) for i in range(X.shape[0])]
        used = [i for i, p in enumerate(pts) if p is not None]
        p_arr = np.stack([pts[i] for i in used])
        y_arr = y[used]
        for axis in (0, 1):
            px = p_arr[:, axis]
            if self._has_head:  # offset anchored at the camera's screen position
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

    def _from_screen(self, target: np.ndarray) -> np.ndarray:
        sx, ox, sy, oy = self._screen
        return np.array([(target[0] - ox) / sx, (target[1] - oy) / sy])

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
        valid = fv.valid  # type: ignore[attr-defined]
        confs = [
            float(fv.values[self.layout.eye_conf[s]])  # type: ignore[attr-defined]
            for s in self._r_ce
            if _eye_valid(self.layout, valid, s)
        ]
        if not confs:
            return 0.0
        head_ok = not self._has_head or _head_valid(self.layout, valid)
        return float(np.mean(confs)) * (1.0 if head_ok else 0.8)

    # ------------------------------------------------------------------

    def to_arrays(self) -> dict[str, np.ndarray | str | float]:
        if not self._r_ce:
            raise RuntimeError("model not fitted")
        out: dict[str, np.ndarray | str | float] = {
            "mapper_kind": self.kind,
            "schema_id": self.schema_id,
            "distance_cm": np.float64(self.distance_cm),
            "kx": np.float64(self.kx),
            "ky": np.float64(self.ky),
            "screen": self._screen,
            "head0": self._head0,
            "has_head": np.bool_(self._has_head),
            "eye_offset": self._eye_offset,
        }
        for s, r in self._r_ce.items():
            out[f"r_ce_{s}"] = r
        return out

    @classmethod
    def from_arrays(cls, arrays: dict) -> GeometricRayMapper:
        m = cls(decode_str(arrays["schema_id"]), distance_cm=float(arrays["distance_cm"]))  # type: ignore[arg-type]
        m.kx, m.ky = float(arrays["kx"]), float(arrays["ky"])
        m._screen = np.asarray(arrays["screen"], dtype=np.float64)
        m._head0 = np.asarray(arrays["head0"], dtype=np.float64)
        m._has_head = bool(arrays["has_head"])
        m._eye_offset = np.asarray(arrays["eye_offset"], dtype=np.float64)
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
    if schema_id == "A-webcam":
        return GazeModel.kind
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

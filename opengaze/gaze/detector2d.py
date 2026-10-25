"""2D pupil detectors: one grayscale eye image in, one pupil ellipse out.

Detectors are swappable behind ``PupilDetector2D``. The built-in ``ellipse_fit`` detector is
pure OpenCV so it runs anywhere (Pupil Labs' ``pupil-detectors`` has no Apple Silicon wheels).
An adapter for ``pupil-detectors``' ``Detector2D`` only has to convert its result dict
(``ellipse.center/axes/angle``, ``confidence``) into a ``PupilDetection2D`` and register here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class PupilDetection2D:
    """A pupil ellipse in eye-image pixels.

    ``minor_axis_angle_deg`` is the direction of the minor axis (image coordinates, x right,
    y down). For a pupil on a sphere seen from a camera, the minor axis points toward the
    projected eyeball centre, which is what the 3D eye model relies on.
    """

    center_x: float
    center_y: float
    major_px: float  # full axis length (diameter)
    minor_px: float
    minor_axis_angle_deg: float
    confidence: float  # 0 … 1
    image_width: int
    image_height: int
    detector: str

    @property
    def axis_ratio(self) -> float:
        return self.minor_px / self.major_px if self.major_px > 0 else 1.0

    @property
    def center_norm(self) -> tuple[float, float]:
        return self.center_x / self.image_width, self.center_y / self.image_height

    @property
    def diameter_norm(self) -> float:
        """Major axis as a fraction of the image's shorter side."""
        return self.major_px / min(self.image_width, self.image_height)

    @property
    def minor_axis_dir(self) -> tuple[float, float]:
        a = math.radians(self.minor_axis_angle_deg)
        return math.cos(a), math.sin(a)


class PupilDetector2D(Protocol):
    name: ClassVar[str]

    def detect(self, gray: np.ndarray) -> PupilDetection2D | None:
        """Detect the pupil in a uint8 grayscale image; None when nothing plausible is found."""
        ...


class EllipseFitOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Plausible pupil diameter, as a fraction of the image's shorter side.
    min_diameter: float = Field(default=0.04, gt=0.0, lt=1.0)
    max_diameter: float = Field(default=0.45, gt=0.0, le=1.0)
    # Images are downscaled so the shorter side is at most this many pixels (speed).
    max_side_px: int = Field(default=240, ge=48)
    # Pixels darker than (darkest + threshold_offset) count as pupil.
    threshold_offset: float = Field(default=18.0, gt=0.0)
    # Pupil must be at least this much darker than its surround, in grey levels.
    min_contrast: float = Field(default=25.0, ge=0.0)


class EllipseFitPupilDetector:
    """Dark-blob pupil detector: centre-surround search, threshold, contour, ellipse fit.

    Confidence combines how well the contour fills its ellipse (an eyelid cutting the pupil
    lowers it), how elliptical the contour is, and pupil/iris contrast.
    """

    name: ClassVar[str] = "ellipse_fit"
    Options: ClassVar[type[BaseModel]] = EllipseFitOptions

    def __init__(self, options: EllipseFitOptions | None = None) -> None:
        self.options = options or EllipseFitOptions()

    def detect(self, gray: np.ndarray) -> PupilDetection2D | None:
        opts = self.options
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape
        scale = min(1.0, opts.max_side_px / min(h0, w0))
        img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if (
            scale < 1.0
        ) else gray
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        short = min(h, w)
        r_min = max(2, int(opts.min_diameter * short / 2))
        r_max = max(r_min + 1, int(opts.max_diameter * short / 2))

        coarse = _coarse_center(img, r_min, r_max)
        if coarse is None:
            return None
        cx0, cy0, r0 = coarse

        # Local ROI around the coarse estimate.
        pad = int(2.5 * r0) + 2
        x0, y0 = max(0, cx0 - pad), max(0, cy0 - pad)
        x1, y1 = min(w, cx0 + pad), min(h, cy0 + pad)
        roi = img[y0:y1, x0:x1]
        darkest = float(np.percentile(roi, 1))
        surround = float(np.median(roi))
        contrast = surround - darkest
        if contrast < opts.min_contrast:
            return None

        thr = darkest + max(opts.threshold_offset, 0.25 * contrast)
        mask = (roi <= thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = _pick_contour(contours, (cx0 - x0, cy0 - y0), r_min, r_max)
        if contour is None:
            return None

        hull = cv2.convexHull(contour)
        if len(hull) < 5:
            return None
        (ex, ey), (d1, d2), angle = cv2.fitEllipse(hull)
        major, minor = max(d1, d2), min(d1, d2)
        if not (2 * r_min <= major <= 2.4 * r_max) or minor <= 0:
            return None

        ellipse_area = math.pi * major * minor / 4.0
        fill = min(1.0, cv2.contourArea(contour) / ellipse_area) if ellipse_area > 0 else 0.0
        residual = _ellipse_residual(hull[:, 0, :].astype(np.float64), ex, ey, d1, d2, angle)
        shape = max(0.0, 1.0 - residual / 0.15)
        contrast_term = min(1.0, contrast / (2.0 * max(opts.min_contrast, 1.0)))
        confidence = float(np.clip(fill * shape * contrast_term, 0.0, 1.0))

        # cv2 reports the rotated-rect angle of axis d1; the minor axis is whichever is shorter.
        minor_angle = angle if d1 <= d2 else angle + 90.0
        inv = 1.0 / scale
        return PupilDetection2D(
            center_x=(ex + x0) * inv,
            center_y=(ey + y0) * inv,
            major_px=major * inv,
            minor_px=minor * inv,
            minor_axis_angle_deg=float(minor_angle % 180.0),
            confidence=confidence,
            image_width=w0,
            image_height=h0,
            detector=self.name,
        )


def _coarse_center(img: np.ndarray, r_min: int, r_max: int) -> tuple[int, int, int] | None:
    """Darkest centre-surround response over a few radii, using an integral image."""
    h, w = img.shape
    integral = cv2.integral(img, sdepth=cv2.CV_64F)
    best: tuple[float, int, int, int] | None = None
    for r in np.unique(np.linspace(r_min, r_max, 5).astype(int)):
        r = int(r)
        outer = 2 * r
        if 2 * outer + 1 > min(h, w):
            continue
        inner_mean = _box_means(integral, r)
        outer_mean = _box_means(integral, outer)
        # Align both maps on centres valid for the outer box.
        d = outer - r
        inner = inner_mean[d : inner_mean.shape[0] - d, d : inner_mean.shape[1] - d]
        n_in = (2 * r + 1) ** 2
        n_out = (2 * outer + 1) ** 2
        ring = (outer_mean * n_out - inner * n_in) / (n_out - n_in)
        response = ring - inner
        iy, ix = np.unravel_index(int(np.argmax(response)), response.shape)
        score = float(response[iy, ix])
        if best is None or score > best[0]:
            best = (score, int(ix) + outer, int(iy) + outer, r)
    if best is None or best[0] <= 0:
        return None
    return best[1], best[2], best[3]


def _box_means(integral: np.ndarray, r: int) -> np.ndarray:
    """Mean over every (2r+1)² box whose centre is at least r pixels from the border."""
    k = 2 * r + 1
    s = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return s / (k * k)


def _pick_contour(
    contours: Any, center: tuple[int, int], r_min: int, r_max: int
) -> np.ndarray | None:
    best, best_d = None, math.inf
    a_min = math.pi * (0.5 * r_min) ** 2
    a_max = math.pi * (1.5 * r_max) ** 2
    for c in contours:
        if len(c) < 5:
            continue
        area = cv2.contourArea(c)
        if not a_min <= area <= a_max:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        d = math.hypot(m["m10"] / m["m00"] - center[0], m["m01"] / m["m00"] - center[1])
        if d < best_d:
            best, best_d = c, d
    return best


def _ellipse_residual(
    pts: np.ndarray, cx: float, cy: float, d1: float, d2: float, angle_deg: float
) -> float:
    """Mean |r - 1| of contour points in the ellipse's unit-circle frame."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    u = (c * dx + s * dy) / (d1 / 2)
    v = (-s * dx + c * dy) / (d2 / 2)
    return float(np.mean(np.abs(np.hypot(u, v) - 1.0)))


DETECTORS_2D: dict[str, type] = {EllipseFitPupilDetector.name: EllipseFitPupilDetector}


def make_pupil_detector(type_name: str, options: dict[str, Any] | None = None) -> PupilDetector2D:
    try:
        cls = DETECTORS_2D[type_name]
    except KeyError:
        raise ValueError(
            f"unknown 2D pupil detector {type_name!r}; available: {sorted(DETECTORS_2D)}"
        ) from None
    return cls(cls.Options.model_validate(options or {}))

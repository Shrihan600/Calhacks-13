"""Pipeline stage 1: frame normalization.

Turns a raw ``CameraFrame`` into the canonical one the rest of the pipeline expects:
undistorted (only when both intrinsics and distortion are known), upright and unmirrored,
and cropped to the configured region of interest. Intrinsics are updated to match the new
image; nothing is invented when they are unknown.

Synthetic ground truth is left as rendered: it refers to the oriented image *before* cropping.
"""

from __future__ import annotations

import cv2
import numpy as np

from opengaze.cameras.types import IDENTITY_ORIENTATION, CameraFrame, CameraIntrinsics, Orientation
from opengaze.imaging import apply_orientation


def orient_intrinsics(intrinsics: CameraIntrinsics, orientation: Orientation) -> CameraIntrinsics:
    """Intrinsics of the image after ``apply_orientation``."""
    w, h = intrinsics.width, intrinsics.height
    cx, cy = orientation.transform_point(intrinsics.cx, intrinsics.cy, w, h)
    new_w, new_h = orientation.output_size(w, h)
    fx, fy = intrinsics.fx, intrinsics.fy
    if orientation.rotation_degrees in (90, 270):
        fx, fy = fy, fx
    return CameraIntrinsics(fx, fy, cx, cy, new_w, new_h)


class FrameNormalizer:
    """Stateful only to cache undistortion maps, which are expensive to build per frame."""

    def __init__(self) -> None:
        self._maps: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, frame: CameraFrame) -> CameraFrame:
        if frame.is_normalized:
            return frame
        image = frame.image
        warnings = list(frame.warnings)

        intrinsics = None
        if frame.intrinsics is not None:
            intrinsics = frame.intrinsics.scaled_to(frame.width, frame.height)
            if intrinsics is None:
                warnings.append("intrinsics ignored: frame aspect ratio differs from calibration")

        if frame.distortion is not None:
            if intrinsics is None:
                warnings.append("distortion ignored: no usable intrinsics")
            else:
                image = self._undistort(frame.camera_id, image, intrinsics, frame.distortion)

        image = apply_orientation(image, frame.orientation)
        if intrinsics is not None:
            intrinsics = orient_intrinsics(intrinsics, frame.orientation)

        if frame.crop is not None:
            x0, y0, w, h = frame.crop.to_pixels(image.shape[1], image.shape[0])
            image = image[y0 : y0 + h, x0 : x0 + w]
            if intrinsics is not None:
                intrinsics = CameraIntrinsics(
                    intrinsics.fx, intrinsics.fy, intrinsics.cx - x0, intrinsics.cy - y0, w, h
                )

        return frame.replace(
            image=image,
            orientation=IDENTITY_ORIENTATION,
            crop=None,
            intrinsics=intrinsics,
            distortion=None,
            warnings=tuple(warnings),
        )

    def _undistort(
        self,
        camera_id: str,
        image: np.ndarray,
        intrinsics: CameraIntrinsics,
        distortion: tuple[float, ...],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        key = (camera_id, w, h, intrinsics, distortion)
        maps = self._maps.get(key)
        if maps is None:
            k = intrinsics.as_matrix()
            d = np.asarray(distortion, dtype=np.float64)
            maps = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), cv2.CV_16SC2)
            self._maps[key] = maps
        return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)

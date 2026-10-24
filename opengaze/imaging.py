"""Low-level image operations shared by camera adapters and vision stages."""

from __future__ import annotations

import cv2
import numpy as np

from opengaze.cameras.types import Orientation

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_orientation(image: np.ndarray, orientation: Orientation) -> np.ndarray:
    """Mirror, then rotate clockwise, exactly as ``Orientation`` documents.

    Returns the input array unchanged (no copy) for the identity orientation.
    """
    if orientation.mirror_horizontal and orientation.mirror_vertical:
        image = cv2.flip(image, -1)
    elif orientation.mirror_horizontal:
        image = cv2.flip(image, 1)
    elif orientation.mirror_vertical:
        image = cv2.flip(image, 0)
    code = _ROTATE_CODES.get(orientation.rotation_degrees)
    if code is not None:
        image = cv2.rotate(image, code)
    return image

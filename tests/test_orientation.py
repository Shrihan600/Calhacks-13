"""Orientation algebra must match the actual image operations exactly."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from opengaze.cameras.types import CameraIntrinsics, CropRegion, Orientation
from opengaze.imaging import apply_orientation
from opengaze.vision.preprocessing import orient_intrinsics

ALL_ORIENTATIONS = [
    Orientation(r, mh, mv)
    for r, mh, mv in itertools.product((0, 90, 180, 270), (False, True), (False, True))
]


@pytest.fixture
def image() -> np.ndarray:
    # Non-square with unique pixel values, so any wrong permutation is detected.
    return np.arange(5 * 7, dtype=np.int32).reshape(5, 7)


@pytest.mark.parametrize("o", ALL_ORIENTATIONS, ids=str)
def test_inverse_restores_image(image: np.ndarray, o: Orientation) -> None:
    restored = apply_orientation(apply_orientation(image, o), o.inverse())
    np.testing.assert_array_equal(restored, image)


@pytest.mark.parametrize("o", ALL_ORIENTATIONS, ids=str)
def test_canonical_form_is_equivalent(image: np.ndarray, o: Orientation) -> None:
    np.testing.assert_array_equal(
        apply_orientation(image, o), apply_orientation(image, o.canonical())
    )


@pytest.mark.parametrize("o", ALL_ORIENTATIONS, ids=str)
def test_transform_point_matches_image(image: np.ndarray, o: Orientation) -> None:
    h, w = image.shape
    out = apply_orientation(image, o)
    assert (out.shape[1], out.shape[0]) == o.output_size(w, h)
    for y, x in itertools.product(range(h), range(w)):
        nx, ny = o.transform_point(x, y, w, h)
        assert out[int(ny), int(nx)] == image[y, x]


def test_identity_detection() -> None:
    assert Orientation().is_identity
    assert Orientation(180, True, True).is_identity  # both flips == rotate 180, then rotate 180
    assert not Orientation(90).is_identity
    assert not Orientation(0, mirror_horizontal=True).is_identity


def test_rejects_non_right_angles() -> None:
    with pytest.raises(ValueError):
        Orientation(45)


@pytest.mark.parametrize("o", ALL_ORIENTATIONS, ids=str)
def test_intrinsics_follow_principal_point(o: Orientation) -> None:
    k = CameraIntrinsics(fx=500.0, fy=400.0, cx=100.0, cy=60.0, width=320, height=240)
    k2 = orient_intrinsics(k, o)
    assert (k2.width, k2.height) == o.output_size(320, 240)
    assert (k2.cx, k2.cy) == o.transform_point(100.0, 60.0, 320, 240)
    if o.rotation_degrees in (90, 270):
        assert (k2.fx, k2.fy) == (400.0, 500.0)
    else:
        assert (k2.fx, k2.fy) == (500.0, 400.0)


def test_intrinsics_scaling() -> None:
    k = CameraIntrinsics(600.0, 600.0, 320.0, 240.0, 640, 480)
    half = k.scaled_to(320, 240)
    assert half is not None and (half.fx, half.cx, half.cy) == (300.0, 160.0, 120.0)
    assert k.scaled_to(640, 360) is None  # aspect ratio changed: can't just rescale


def test_crop_region_bounds() -> None:
    assert CropRegion(0.25, 0.5, 0.5, 0.5).to_pixels(200, 100) == (50, 50, 100, 50)
    with pytest.raises(ValueError):
        CropRegion(0.6, 0.0, 0.5, 0.5)

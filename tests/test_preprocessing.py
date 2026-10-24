from __future__ import annotations

import numpy as np

from opengaze.config.schema import CropConfig, IntrinsicsConfig, OpticsConfig
from opengaze.vision.preprocessing import FrameNormalizer


async def test_crop_after_rotation_and_intrinsics_follow(synthetic) -> None:
    source = synthetic(
        resolution=(320, 240),
        rotation_degrees=90,
        crop=CropConfig(x=0.25, y=0.0, width=0.5, height=1.0),
        optics=OpticsConfig(
            intrinsics=IntrinsicsConfig(fx=300, fy=310, cx=160, cy=120, width=320, height=240)
        ),
    )
    async with source:
        raw = await source.read_frame()
    assert not raw.is_normalized
    out = FrameNormalizer()(raw)
    assert out.is_normalized
    assert (out.width, out.height) == (120, 320)  # rotated to 240x320, then half the width
    k = out.intrinsics
    assert (k.width, k.height) == (120, 320)
    assert (k.fx, k.fy) == (310, 300)  # axes swap under a 90° rotation
    assert k.cx == (240 - 1 - 120) - 60 and k.cy == 160


async def test_identity_frames_pass_through(synthetic) -> None:
    source = synthetic()
    async with source:
        raw = await source.read_frame()
    assert FrameNormalizer()(raw) is raw


async def test_zero_distortion_undistort_is_near_identity(synthetic) -> None:
    source = synthetic(
        optics=OpticsConfig(
            intrinsics=IntrinsicsConfig(fx=300, fy=300, cx=160, cy=120, width=320, height=240),
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
    )
    async with source:
        raw = await source.read_frame()
    out = FrameNormalizer()(raw)
    assert out.distortion is None
    diff = np.abs(out.image.astype(int) - raw.image.astype(int))
    assert diff.mean() < 1.0


async def test_mismatched_intrinsics_are_dropped_with_warning(synthetic) -> None:
    source = synthetic(
        resolution=(320, 180),
        optics=OpticsConfig(
            intrinsics=IntrinsicsConfig(fx=300, fy=300, cx=160, cy=120, width=320, height=240),
            distortion=(0.1, 0.0, 0.0, 0.0),
        ),
    )
    async with source:
        raw = await source.read_frame()
    out = FrameNormalizer()(raw)
    assert out.intrinsics is None
    assert any("intrinsics ignored" in w for w in out.warnings)

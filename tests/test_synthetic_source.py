from __future__ import annotations

import time

import numpy as np
import pytest

from opengaze.cameras.simulated import GazeTrajectory, eyelid_closure
from opengaze.cameras.types import (
    CameraConnectionError,
    CameraDisconnectedError,
    CameraState,
    ControlSupport,
    FrameStatus,
    PixelFormat,
    SourceKind,
    TimestampSource,
)
from opengaze.vision.preprocessing import FrameNormalizer


async def test_frames_are_labeled_and_carry_ground_truth(synthetic) -> None:
    source = synthetic(resolution=(320, 240), fps=60)
    async with source:
        frames = [await source.read_frame() for _ in range(5)]
    assert [f.sequence for f in frames] == [0, 1, 2, 3, 4]
    for f in frames:
        assert f.source_kind is SourceKind.SYNTHETIC
        assert f.timestamp_source is TimestampSource.SYNTHETIC
        assert f.timestamp_uncertainty_ns == 0
        assert f.pixel_format is PixelFormat.BGR8
        assert (f.width, f.height) == (320, 240)
        assert f.ground_truth is not None
        assert 0 <= f.ground_truth["pupil_x"] < 320 and 0 <= f.ground_truth["pupil_y"] < 240
    # Virtual-clock timestamps advance exactly one frame period.
    periods = np.diff([f.capture_ns for f in frames])
    assert np.allclose(periods, 1e9 / 60, atol=1)
    assert frames[-1].fps == pytest.approx(60, rel=1e-3)
    assert frames[-1].fps_measured


async def test_pupil_is_darkest_region_at_ground_truth(synthetic) -> None:
    source = synthetic(options={"motion": "static", "blink_interval_s": None})
    async with source:
        frame = await source.read_frame()
    gray = frame.image[..., 0]
    x, y = round(frame.ground_truth["pupil_x"]), round(frame.ground_truth["pupil_y"])
    assert gray[y, x] < 40  # pupil
    assert gray[5, 5] > 100  # skin


async def test_same_seed_is_deterministic(synthetic) -> None:
    images = []
    for _ in range(2):
        source = synthetic(
            options={"motion": "saccades", "motion_seed": 3, "noise_std": 5.0, "noise_seed": 9}
        )
        async with source:
            images.append([(await source.read_frame()).image for _ in range(3)])
    for a, b in zip(*images, strict=True):
        np.testing.assert_array_equal(a, b)


async def test_eyes_share_trajectory_but_not_noise(synthetic) -> None:
    left = synthetic("left_eye", options={"motion_seed": 1, "noise_seed": 1, "noise_std": 6.0})
    right = synthetic("right_eye", options={"motion_seed": 1, "noise_seed": 2, "noise_std": 6.0})
    async with left, right:
        fl, fr = await left.read_frame(), await right.read_frame()
    assert fl.ground_truth["gaze_x"] == fr.ground_truth["gaze_x"]
    assert not np.array_equal(fl.image, fr.image)


@pytest.mark.parametrize("resolution", [(160, 120), (320, 240), (640, 480), (400, 400)])
async def test_any_resolution(synthetic, resolution) -> None:
    source = synthetic(resolution=resolution)
    async with source:
        frame = await source.read_frame()
    assert (frame.width, frame.height) == resolution


@pytest.mark.parametrize(
    ("mount", "mirror", "correction", "correction_mirror"),
    [
        (90, False, 270, False),
        (180, False, 180, False),
        (270, False, 90, False),
        (0, True, 0, True),
        (90, True, 90, True),
    ],
)
async def test_profile_orientation_undoes_mount(
    synthetic, mount, mirror, correction, correction_mirror
) -> None:
    """A rotated/mirrored mount plus the matching profile correction gives the upright eye."""
    upright = synthetic(options={"motion_seed": 4})
    mounted = synthetic(
        rotation_degrees=correction,
        mirror_horizontal=correction_mirror,
        options={
            "motion_seed": 4,
            "mount_rotation_degrees": mount,
            "mount_mirror_horizontal": mirror,
        },
    )
    normalize = FrameNormalizer()
    async with upright, mounted:
        expected = await upright.read_frame()
        raw = await mounted.read_frame()
    if mount in (90, 270):
        assert raw.image.shape[:2] == expected.image.shape[1::-1]
    np.testing.assert_array_equal(normalize(raw).image, expected.image)
    assert raw.ground_truth == expected.ground_truth


async def test_drops_create_sequence_gaps(synthetic) -> None:
    source = synthetic(options={"drop_probability": 0.3, "noise_seed": 5})
    async with source:
        seqs = [(await source.read_frame()).sequence for _ in range(50)]
    assert seqs == sorted(seqs)
    assert seqs[-1] > 49  # some sequence numbers never arrived


async def test_latency_offsets_receive_time(synthetic) -> None:
    source = synthetic(options={"latency_ms": 40.0})
    async with source:
        frame = await source.read_frame()
    assert frame.host_receive_ns - frame.capture_ns == 40_000_000


async def test_injected_disconnect_then_reconnect(synthetic) -> None:
    source = synthetic(options={"disconnect_after_frames": 3})
    await source.connect()
    for _ in range(3):
        await source.read_frame()
    with pytest.raises(CameraDisconnectedError):
        await source.read_frame()
    assert source.get_status().state is CameraState.IDLE
    await source.connect()
    frame = await source.read_frame()
    assert frame.sequence > 3  # the timeline continues across reconnects
    await source.disconnect()


async def test_injected_connect_failures(synthetic) -> None:
    source = synthetic(options={"fail_connect_attempts": 2})
    for _ in range(2):
        with pytest.raises(CameraConnectionError):
            await source.connect()
    await source.connect()
    assert source.get_status().state is CameraState.STREAMING
    await source.disconnect()


async def test_corrupt_frames_are_flagged(synthetic) -> None:
    source = synthetic(options={"corrupt_probability": 1.0})
    async with source:
        frame = await source.read_frame()
    assert frame.status is FrameStatus.PARTIAL
    assert frame.warnings


async def test_read_before_connect_fails(synthetic) -> None:
    with pytest.raises(CameraDisconnectedError):
        await synthetic().read_frame()


async def test_realtime_pacing(synthetic) -> None:
    source = synthetic(fps=50, options={"realtime": True})
    async with source:
        start = time.monotonic()
        for _ in range(10):
            await source.read_frame()
        elapsed = time.monotonic() - start
    assert 0.12 < elapsed < 0.5  # ~0.18 s for 9 periods at 50 fps, with scheduler slack


def test_capabilities_report_unsupported_controls(synthetic) -> None:
    from opengaze.config.schema import ControlsConfig

    caps = synthetic(controls=ControlsConfig(exposure=100.0)).get_capabilities()
    assert caps.source_kind is SourceKind.SYNTHETIC
    assert caps.controls["exposure"] is ControlSupport.UNSUPPORTED


def test_trajectories_stay_on_screen() -> None:
    for motion in ("lissajous", "saccades", "static"):
        trajectory = GazeTrajectory(motion, seed=11, saccade_interval_s=0.5)
        for t in np.linspace(0, 30, 600):
            x, y = trajectory(float(t))
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_blink_profile() -> None:
    assert eyelid_closure(0.0, 4.0, 0.2) == 0.0
    assert eyelid_closure(2.1, 4.0, 0.2) == pytest.approx(1.0)  # mid-blink
    assert eyelid_closure(3.0, 4.0, 0.2) == 0.0
    assert eyelid_closure(2.1, None, 0.2) == 0.0

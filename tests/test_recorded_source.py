from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from opengaze.cameras.recorded import RecordedVideoSource
from opengaze.cameras.types import CameraConfigError, EndOfStream, SourceKind, TimestampSource
from tests.helpers import camera_config


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
    assert writer.isOpened(), "OpenCV build can't write MJPG/AVI"
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    writer.release()
    return path


def recorded(path: Path, **options) -> RecordedVideoSource:
    opts = {"path": path.name, "realtime": False, **options}
    return RecordedVideoSource(
        camera_config("left_eye", source_type="recorded", base_dir=path.parent, options=opts)
    )


async def test_plays_file_with_media_timestamps(video: Path) -> None:
    source = recorded(video)
    async with source:
        frames = [await source.read_frame() for _ in range(6)]
        with pytest.raises(EndOfStream):
            await source.read_frame()
    assert all(f.source_kind is SourceKind.RECORDED for f in frames)
    assert all(f.timestamp_source is TimestampSource.MEDIA_TIME for f in frames)
    assert (frames[0].width, frames[0].height) == (64, 48)
    gaps = np.diff([f.capture_ns for f in frames])
    assert np.allclose(gaps, 1e9 / 20, rtol=1e-6)
    # Brightness ramps like the written frames did (JPEG is lossy, so compare loosely).
    means = [f.image.mean() for f in frames]
    assert means == sorted(means)


async def test_loops(video: Path) -> None:
    source = recorded(video, loop=True)
    async with source:
        frames = [await source.read_frame() for _ in range(14)]
    assert [f.sequence for f in frames] == list(range(14))
    assert np.all(np.diff([f.capture_ns for f in frames]) > 0)  # time keeps moving forward


async def test_playback_rate_scales_time(video: Path) -> None:
    source = recorded(video, playback_rate=2.0)
    async with source:
        a, b = await source.read_frame(), await source.read_frame()
    assert b.capture_ns - a.capture_ns == pytest.approx(1e9 / 40, rel=1e-6)


async def test_missing_file_is_config_error(tmp_path: Path) -> None:
    source = recorded(tmp_path / "nope.avi")
    with pytest.raises(CameraConfigError):
        await source.connect()
    issues = RecordedVideoSource.check_config(source.config)
    assert any(i.severity == "error" and "not found" in i.message for i in issues)

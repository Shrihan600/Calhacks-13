"""Recorded video files, for repeatable development and regression tests.

Frames are labeled ``SourceKind.RECORDED``. Timestamps come from the frame index and the file's
frame rate, mapped onto the host clock at playback (``TimestampSource.MEDIA_TIME``);
variable-frame-rate files are therefore approximated as constant-rate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, ConfigDict, Field

from opengaze.cameras.base import BlockingCaptureSource, ConfigIssue, RawCapture
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConfigError,
    CameraConnectionError,
    CameraFrame,
    ControlSupport,
    EndOfStream,
    PixelFormat,
    SettingResult,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CameraConfig


class RecordedOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    loop: bool = False
    # true = play at the recorded frame rate (× playback_rate); false = as fast as decoded.
    realtime: bool = True
    playback_rate: float = Field(default=1.0, gt=0.0)
    fps_override: float | None = Field(
        default=None, gt=0.0, description="use when the file has no usable frame rate"
    )


@dataclass
class _Playback:
    cap: Any
    fps: float
    frame_index: int = 0  # frames delivered since playback started, across loops


def resolve_video_path(config: CameraConfig, options: RecordedOptions) -> Path:
    path = Path(options.path).expanduser()
    if not path.is_absolute() and config.base_dir is not None:
        path = config.base_dir / path
    return path.resolve()


class RecordedVideoSource(BlockingCaptureSource):
    source_type = "recorded"
    source_kind = SourceKind.RECORDED
    Options = RecordedOptions
    non_geometric_options = frozenset({"loop", "realtime", "playback_rate"})

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self.path = resolve_video_path(config, self.options)
        self._play_start_ns = 0

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        if issues:
            return issues
        path = resolve_video_path(config, cls.parse_options(config))
        if not path.is_file():
            issues.append(ConfigIssue("error", f"{config.camera_id}: video file not found: {path}"))
        issues.append(
            ConfigIssue("info", f"{config.camera_id}: RECORDED source — replay, not live")
        )
        return issues

    def _open_blocking(self) -> _Playback:
        if not self.path.is_file():
            raise CameraConfigError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(f"OpenCV could not open video {self.path}")
        file_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = self.options.fps_override or (file_fps if file_fps > 0 else None)
        if fps is None:
            cap.release()
            raise CameraConfigError(f"{self.path.name} reports no frame rate; set fps_override")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._capabilities = CameraCapabilities(
            source_type=self.source_type,
            source_kind=self.source_kind,
            backend=cap.getBackendName(),
            device_name=self.path.name,
            resolution=(width, height) if width and height else None,
            reported_fps=fps,
            pixel_formats=(PixelFormat.BGR8,),
            controls={"resolution": ControlSupport.UNSUPPORTED, "fps": ControlSupport.UNSUPPORTED},
            setting_results=tuple(
                SettingResult(name, value, None, ControlSupport.UNSUPPORTED, "fixed by the file")
                for name, value in (
                    ("resolution", self.config.resolution),
                    ("fps", self.config.fps),
                )
                if value is not None
            ),
            supports_reconnect=True,
            notes=(
                f"RECORDED: {self.path.name}"
                + (f", {frame_count} frames" if frame_count > 0 else ""),
                "timestamps are media time mapped onto the host clock",
            ),
        )
        return _Playback(cap=cap, fps=fps)

    def _read_blocking(self, handle: _Playback) -> RawCapture:
        ok, image = handle.cap.read()
        if not ok and self.options.loop and handle.frame_index > 0:
            handle.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = handle.cap.read()
        if not ok or image is None:
            raise EndOfStream(f"end of {self.path.name}")
        handle.frame_index += 1
        return RawCapture(image=image, host_receive_ns=time.monotonic_ns())

    def _close_blocking(self, handle: _Playback) -> None:
        handle.cap.release()

    async def connect(self) -> None:
        await super().connect()
        self._play_start_ns = time.monotonic_ns()

    async def read_frame(self) -> CameraFrame:
        raw = await self._read_raw()
        playback: _Playback = self._handle
        media_ns = int(
            (playback.frame_index - 1) * 1e9 / (playback.fps * self.options.playback_rate)
        )
        capture_ns = self._play_start_ns + media_ns
        received_ns = raw.host_receive_ns
        if self.options.realtime:
            wait_s = (capture_ns - time.monotonic_ns()) / 1e9
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            received_ns = time.monotonic_ns()
        return self._frame_from_raw(
            raw,
            host_receive_ns=received_ns,
            capture_ns=capture_ns,
            timestamp_source=TimestampSource.MEDIA_TIME,
        )

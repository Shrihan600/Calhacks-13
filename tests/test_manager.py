"""CameraManager behaviour under normal operation and hardware failures."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest

from opengaze.cameras.base import BlockingCaptureSource, CameraSource, RawCapture
from opengaze.cameras.manager import CameraManager
from opengaze.cameras.recorded import RecordedVideoSource
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraFrame,
    CameraState,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CaptureSettings
from tests.helpers import camera_config

FAST = CaptureSettings(
    stale_frame_timeout_s=0.15, reconnect_initial_delay_s=0.02, reconnect_max_delay_s=0.1
)


def live(synthetic, camera_id="left_eye", fps=100, **options):
    return synthetic(camera_id, fps=fps, options={"realtime": True, **options})


@asynccontextmanager
async def running(*sources: CameraSource, settings: CaptureSettings = FAST):
    manager = CameraManager(sources, settings)
    task = asyncio.create_task(manager.run())
    try:
        yield manager
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 3.0)


async def wait_for(predicate, timeout=2.0, interval=0.01) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(interval)


async def test_independent_rates_and_latest_frame(synthetic) -> None:
    fast, slow = live(synthetic, "left_eye", fps=100), live(synthetic, "right_eye", fps=25)
    async with running(fast, slow) as manager:
        await asyncio.sleep(0.5)
        health = manager.health()
        assert manager.latest("left_eye") is not None and manager.latest("right_eye") is not None
    assert health["left_eye"].state is CameraState.STREAMING
    assert health["left_eye"].measured_fps == pytest.approx(100, rel=0.25)
    assert health["right_eye"].measured_fps == pytest.approx(25, rel=0.25)
    assert manager.health()["left_eye"].state is CameraState.STOPPED


async def test_slow_consumer_gets_fresh_frames_not_a_backlog(synthetic) -> None:
    async with running(live(synthetic, fps=100)) as manager:
        await wait_for(lambda: manager.latest("left_eye", consume=False) is not None)
        sequences = []
        for _ in range(4):
            await asyncio.sleep(0.1)  # a consumer 10x slower than the camera
            frame = manager.latest("left_eye")
            sequences.append(frame.sequence)
            age_ms = (time.monotonic_ns() - frame.host_receive_ns) / 1e6
            assert age_ms < 50  # always the newest frame, never a queued old one
        health = manager.health()["left_eye"]
    assert all(b - a >= 5 for a, b in zip(sequences, sequences[1:], strict=False))
    assert health.frames_overwritten > 0


async def test_reconnects_after_disconnect(synthetic) -> None:
    source = live(synthetic, disconnect_after_frames=5)
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["left_eye"].reconnects >= 2)
        received = manager.health()["left_eye"].frames_received
        await wait_for(lambda: manager.health()["left_eye"].frames_received > received)
    assert "injected disconnect" in manager.health()["left_eye"].last_error


async def test_connect_retries_then_streams(synthetic) -> None:
    source = live(synthetic, fail_connect_attempts=3)
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["left_eye"].state is CameraState.STREAMING)


async def test_missing_camera_does_not_block_others(synthetic) -> None:
    broken = live(synthetic, "right_eye", fail_connect_attempts=10_000)
    async with running(live(synthetic, "left_eye"), broken) as manager:
        await wait_for(lambda: manager.health()["left_eye"].frames_received > 10)
        health = manager.health()
    assert health["right_eye"].state in (CameraState.CONNECTING, CameraState.RECONNECTING)
    assert health["right_eye"].frames_received == 0
    assert "injected connect failure" in health["right_eye"].last_error


async def test_config_error_fails_without_retry(tmp_path) -> None:
    source = RecordedVideoSource(
        camera_config(
            "left_eye", source_type="recorded", options={"path": str(tmp_path / "missing.avi")}
        )
    )
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["left_eye"].state is CameraState.FAILED)
        await asyncio.sleep(0.1)
        assert manager.health()["left_eye"].state is CameraState.FAILED
    assert manager.health()["left_eye"].state is CameraState.FAILED


async def test_transport_delay_and_drops_are_measured(synthetic) -> None:
    source = live(synthetic, latency_ms=30.0, drop_probability=0.2, noise_seed=3)
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["left_eye"].frames_received > 30)
        health = manager.health()["left_eye"]
    assert health.transport_delay_ms == pytest.approx(30.0, abs=8.0)
    assert health.frames_skipped_by_source > 0


class HangingSource(CameraSource):
    """Delivers one frame, then never another (a frozen stream that doesn't error)."""

    source_type = "test_hanging"
    source_kind = SourceKind.SYNTHETIC

    def __init__(self, config) -> None:
        super().__init__(config)
        self.sent = False

    async def connect(self) -> None:
        self._set_state(CameraState.STREAMING)

    async def disconnect(self) -> None:
        self._set_state(CameraState.STOPPED)

    async def read_frame(self) -> CameraFrame:
        if self.sent:
            await asyncio.Event().wait()
        self.sent = True
        now = time.monotonic_ns()
        return self._make_frame(
            np.zeros((4, 4, 3), np.uint8),
            host_receive_ns=now,
            capture_ns=None,
            timestamp_source=TimestampSource.HOST_RECEIVE,
            timestamp_uncertainty_ns=None,
        )

    def get_capabilities(self) -> CameraCapabilities:
        return CameraCapabilities(self.source_type, self.source_kind)


async def test_frozen_stream_is_reported_stalled() -> None:
    source = HangingSource(camera_config("head", source_type="test_hanging"))
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["head"].frames_received == 1)
        assert manager.health()["head"].state is CameraState.STREAMING
        await wait_for(lambda: manager.health()["head"].state is CameraState.STALLED)


class StuckDriver(BlockingCaptureSource):
    """A blocking driver whose read never returns in time — like an unplugged USB camera."""

    source_type = "test_stuck"
    source_kind = SourceKind.LIVE

    def _open_blocking(self):
        return object()

    def _read_blocking(self, handle) -> RawCapture:
        time.sleep(1.0)
        return RawCapture(np.zeros((4, 4, 3), np.uint8), time.monotonic_ns())

    def _close_blocking(self, handle) -> None:
        pass


async def test_blocking_read_timeout_triggers_reconnect() -> None:
    source = StuckDriver(camera_config("left_eye", source_type="test_stuck", read_timeout_s=0.1))
    started = time.monotonic()
    async with running(source) as manager:
        await wait_for(lambda: manager.health()["left_eye"].reconnects >= 1)
        assert "no frame within" in manager.health()["left_eye"].last_error
    assert time.monotonic() - started < 3.0  # a stuck driver thread must not block shutdown


def test_runtime_thread_serves_frames_to_another_thread(synthetic) -> None:
    manager = CameraManager([live(synthetic, fps=60)], FAST)
    with CaptureRuntime(manager) as runtime:
        deadline = time.monotonic() + 2.0
        while manager.latest("left_eye") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager.latest("left_eye") is not None
        assert runtime.running
    assert not runtime.running


def test_stop_before_run_returns_immediately(synthetic) -> None:
    manager = CameraManager([live(synthetic)], FAST)
    manager.request_stop()
    asyncio.run(asyncio.wait_for(manager.run(), 1.0))

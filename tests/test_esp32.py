"""ESP32-CAM glasses: clock sync, MJPEG transport, the esp32 adapter, and the reconnect path.

Everything here runs against ``opengaze.debug.fake_esp32`` on localhost. That checks the Mac
side of the interface in docs/hardware-interface.md; it says nothing about real boards.
"""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager

import pytest

from opengaze.cameras.esp32 import ESP32StreamSource
from opengaze.cameras.manager import CameraManager, is_emulated
from opengaze.cameras.mjpeg import MjpegPart, boundary_from_content_type, jpeg_is_complete
from opengaze.cameras.registry import create_source
from opengaze.cameras.timing import DeviceClockSync
from opengaze.cameras.types import (
    CameraConnectionError,
    CameraDisconnectedError,
    CameraState,
    ControlSupport,
    FrameStatus,
    TimestampSource,
)
from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import check_hardware_profile, check_profile_data
from opengaze.config.schema import CaptureSettings
from opengaze.debug.fake_esp32 import EyeSettings, FakeEyeCamera, Faults
from tests.helpers import HARDWARE_DIR, camera_config

FAST = CaptureSettings(
    stale_frame_timeout_s=0.5, reconnect_initial_delay_s=0.05, reconnect_max_delay_s=0.2
)


def esp32_config(port: int, camera_id: str = "left_eye", **options):
    return camera_config(
        camera_id,
        source_type="esp32",
        resolution=(320, 240),
        connect_timeout_s=2.0,
        read_timeout_s=1.0,
        options={"host": "127.0.0.1", "port": port, "status_port": port, **options},
    )


@pytest.fixture
def fake_eye():
    cameras: list[FakeEyeCamera] = []

    def make(**settings) -> FakeEyeCamera:
        faults = settings.pop("faults", Faults())
        cam = FakeEyeCamera(EyeSettings("left", fps=40.0, faults=faults, **settings)).start()
        cameras.append(cam)
        return cam

    yield make
    for cam in cameras:
        cam.stop()


@asynccontextmanager
async def running(*sources, settings: CaptureSettings = FAST):
    manager = CameraManager(sources, settings)
    task = asyncio.create_task(manager.run())
    try:
        yield manager
    finally:
        await manager.stop()
        await asyncio.wait_for(task, 5.0)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Clock sync
# ---------------------------------------------------------------------------


class TestDeviceClockSync:
    def test_maps_to_host_clock_within_min_delay(self):
        sync = DeviceClockSync()
        offset_ns = 5_000_000_000_000  # host clock far ahead of the device's
        delays_ms = [12, 3, 25, 1, 8, 40, 2]  # transport delay per message
        for i, d in enumerate(delays_ms):
            device_us = 1_000_000 + i * 50_000
            host_ns = device_us * 1000 + offset_ns + d * 1_000_000
            mapped = sync.update(device_us, host_ns)
        true_capture = device_us * 1000 + offset_ns
        # Late by exactly the smallest delay seen (1 ms), never early.
        assert mapped - true_capture == 1_000_000
        assert sync.excess_delay_ns == 1_000_000  # last message: 2 ms vs best 1 ms
        assert sync.uncertainty_ns is not None and sync.uncertainty_ns > 0

    def test_device_reboot_resets_estimate(self):
        sync = DeviceClockSync()
        sync.update(900_000_000, 1_000_000_000_000)
        mapped = sync.update(5_000, 1_000_010_000_000)  # the device restarted from ~0
        assert sync.resets == 1
        assert mapped == 1_000_010_000_000  # new offset learned from this message alone

    def test_map_after_batch_update(self):
        sync = DeviceClockSync()
        assert sync.map(5) is None
        host = 10**15
        for t_us in (1_000, 2_000, 3_000):  # three readings in one datagram
            sync.update(t_us, host)
        assert [sync.map(t) for t in (1_000, 2_000, 3_000)] == [
            host - 2_000_000, host - 1_000_000, host
        ]

    def test_unwraps_32_bit_micros(self):
        sync = DeviceClockSync()
        host = 10**15
        before = 2**32 - 10_000
        sync.update(before, host)
        mapped = sync.update(10_000, host + 20_000_000)  # rolled over 20 ms later
        assert sync.resets == 0
        assert mapped == pytest.approx(host + 20_000_000, abs=1)


# ---------------------------------------------------------------------------
# MJPEG helpers
# ---------------------------------------------------------------------------


def test_boundary_parsing():
    assert boundary_from_content_type("multipart/x-mixed-replace;boundary=1234") == "1234"
    assert boundary_from_content_type('multipart/x-mixed-replace; boundary="--abc"') == "abc"
    assert boundary_from_content_type("image/jpeg") is None


def test_part_timestamps_and_sequence():
    stock = MjpegPart(b"", {"x-timestamp": "12.000345"}, 0)  # Espressif CameraWebServer
    assert stock.device_time_us == 12_000_345 and stock.sequence is None
    ours = MjpegPart(b"", {"x-timestamp-us": "987654321", "x-sequence": "17"}, 0)
    assert ours.device_time_us == 987_654_321 and ours.sequence == 17
    assert MjpegPart(b"", {"x-timestamp": "junk"}, 0).device_time_us is None


def test_jpeg_completeness():
    assert jpeg_is_complete(b"\xff\xd8abc\xff\xd9\r\n")
    assert not jpeg_is_complete(b"\xff\xd8abc")


# ---------------------------------------------------------------------------
# Adapter against the emulator
# ---------------------------------------------------------------------------


async def test_reads_frames_with_device_timestamps(fake_eye):
    cam = fake_eye()
    src = ESP32StreamSource(esp32_config(cam.port))
    await src.connect()
    try:
        frames = [await src.read_frame() for _ in range(5)]
    finally:
        await src.disconnect()
    f = frames[-1]
    assert (f.width, f.height) == (320, 240)
    assert f.timestamp_source is TimestampSource.DEVICE
    assert f.capture_ns is not None and f.capture_ns <= f.host_receive_ns
    seqs = [fr.sequence for fr in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5  # the device's own counter
    assert "emulated device" in f.warnings
    caps = src.get_capabilities()
    assert caps.backend == "mjpeg-http" and caps.resolution == (320, 240)
    assert caps.device_name == "left_esp32" and caps.provides_device_timestamps
    assert caps.notes[0].startswith("EMULATED")


async def test_parts_without_content_length(fake_eye):
    cam = fake_eye(content_length=False)
    src = ESP32StreamSource(esp32_config(cam.port))
    await src.connect()
    try:
        frames = [await src.read_frame() for _ in range(4)]
    finally:
        await src.disconnect()
    assert all(fr.status is FrameStatus.OK and fr.width == 320 for fr in frames)


async def test_resolution_mismatch_is_reported_not_hidden(fake_eye):
    cam = fake_eye(resolution=(640, 480))
    src = ESP32StreamSource(esp32_config(cam.port))
    await src.connect()
    await src.disconnect()
    result = src.get_capabilities().setting_results[0]
    assert result.name == "resolution" and result.support is ControlSupport.IGNORED
    assert result.actual == (640, 480)


async def test_apply_settings_uses_control_endpoint(fake_eye):
    cam = fake_eye(resolution=(640, 480))
    src = ESP32StreamSource(esp32_config(cam.port, apply_settings=True, jpeg_quality=12))
    await src.connect()
    frame = await src.read_frame()
    await src.disconnect()
    assert (frame.width, frame.height) == (320, 240)
    caps = src.get_capabilities()
    assert caps.setting_results[0].support is ControlSupport.SUPPORTED
    assert {r.name for r in caps.setting_results} >= {"framesize", "jpeg_quality"}


async def test_corrupt_frames_are_skipped(fake_eye):
    cam = fake_eye()
    real_render = cam.render
    calls = {"n": 0}

    def flaky():
        jpeg, seq, ts = real_render()
        calls["n"] += 1
        return (b"not a jpeg" if calls["n"] % 2 else jpeg), seq, ts

    cam.render = flaky  # type: ignore[method-assign]
    src = ESP32StreamSource(esp32_config(cam.port))
    await src.connect()
    try:
        for _ in range(3):
            assert (await src.read_frame()).width == 320
    finally:
        await src.disconnect()
    assert src.corrupt_frames >= 3


async def test_connection_refused_is_retryable_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once closed
    src = ESP32StreamSource(esp32_config(port, status_port=None))
    with pytest.raises(CameraConnectionError, match="refused"):
        await src.connect()


async def test_wrong_endpoint_explains_itself(fake_eye):
    cam = fake_eye()
    src = ESP32StreamSource(esp32_config(cam.port, path="/status"))
    with pytest.raises(CameraConnectionError, match="multipart"):
        await src.connect()


async def test_stalled_stream_raises_disconnected(fake_eye):
    cam = fake_eye(faults=Faults(stall_every_s=0.3, stall_s=5.0))
    src = ESP32StreamSource(esp32_config(cam.port))
    await src.connect()
    with pytest.raises(CameraDisconnectedError):
        for _ in range(100):
            await src.read_frame()
    await src.disconnect()


# ---------------------------------------------------------------------------
# Through the CameraManager: two cameras, drops, reconnects
# ---------------------------------------------------------------------------


async def test_two_cameras_concurrently_with_drops_and_reconnects(fake_eye):
    faults = Faults(drop_rate=0.2, disconnect_every_s=0.6)
    left, right = fake_eye(faults=faults), fake_eye(faults=faults, seed=9)
    sources = [
        ESP32StreamSource(esp32_config(left.port, "left_eye")),
        ESP32StreamSource(esp32_config(right.port, "right_eye")),
    ]
    async with running(*sources) as manager:
        await wait_for(
            lambda: all(h.reconnects >= 1 and h.frames_received > 30
                        for h in manager.health().values()),
            timeout=8.0,
        )
        health = manager.health()
    for h in health.values():
        assert h.frames_skipped_by_source > 0  # sequence gaps from the injected drops
        assert h.measured_fps is not None and h.measured_fps > 10
        assert is_emulated(h)


async def test_stall_shows_as_stalled_then_recovers(fake_eye):
    cam = fake_eye(faults=Faults(stall_every_s=0.5, stall_s=0.8))
    src = ESP32StreamSource(esp32_config(cam.port))
    async with running(src, settings=CaptureSettings(
        stale_frame_timeout_s=0.3, reconnect_initial_delay_s=0.05, reconnect_max_delay_s=0.2
    )) as manager:
        await wait_for(lambda: manager.health()["left_eye"].state is CameraState.STALLED)
        await wait_for(lambda: manager.health()["left_eye"].state is CameraState.STREAMING)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_registry_and_missing_address():
    assert isinstance(create_source(esp32_config(81)), ESP32StreamSource)
    report = check_profile_data(
        {"hardware_profile": {"id": "t", "cameras": {"left_eye": {"source_type": "esp32"}}}}
    )
    assert not report.ok and any("set host" in str(e) for e in report.errors)


def test_changing_address_keeps_calibration_fingerprint():
    path = HARDWARE_DIR / "esp32-glasses.yaml"
    base = check_hardware_profile(path)
    moved = check_hardware_profile(
        path, overrides=["cameras.left_eye.options.host=192.168.4.20", "imu.options.port=6000"]
    )
    rotated = check_hardware_profile(path, overrides=["cameras.left_eye.rotation_degrees=90"])
    assert base.ok and moved.ok and rotated.ok
    assert moved.profile.cameras["left_eye"].options["host"] == "192.168.4.20"
    assert profile_fingerprint(base.profile) == profile_fingerprint(moved.profile)
    assert profile_fingerprint(base.profile) != profile_fingerprint(rotated.profile)


def test_same_stream_twice_is_rejected():
    report = check_hardware_profile(
        HARDWARE_DIR / "esp32-glasses.yaml",
        overrides=[
            "cameras.left_eye.options.host=10.0.0.5",
            "cameras.right_eye.options.host=10.0.0.5",
        ],
    )
    assert any("same device" in str(e) for e in report.errors)


def test_bad_override_is_reported():
    report = check_hardware_profile(HARDWARE_DIR / "esp32-glasses.yaml", overrides=["nonsense"])
    assert not report.ok and "dotted.key=value" in str(report.errors[0])


@pytest.mark.parametrize("name", ["esp32-glasses.yaml", "esp32-emulator.yaml"])
def test_shipped_esp32_profiles_validate(name):
    report = check_hardware_profile(HARDWARE_DIR / name)
    assert report.ok, report.errors
    assert report.mode is not None and report.mode.value == "C"
    assert report.profile.imu is not None and report.profile.imu.source_type == "udp"

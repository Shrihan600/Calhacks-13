"""Wired mode: the esp32_serial adapter and the wired IMU.

Most tests use a small fake with the receiver's API (``SerialCamera``, ``TelemetryReceiver``)
so they don't depend on the hardware team's code. The last one runs the real receiver
(opengaze-hw/receiver) against its own serial simulator, when both are present. None of this
says anything about real boards.
"""

from __future__ import annotations

import asyncio
import textwrap
import time

import pytest

from opengaze.cameras.registry import create_source
from opengaze.cameras.types import CameraConnectionError, CameraState, TimestampSource
from opengaze.cameras.wired import _LINKS, close_all_links
from opengaze.config.fingerprint import profile_fingerprint
from opengaze.config.loader import check_hardware_profile
from opengaze.imu.sources import WiredImuSource
from tests.helpers import HARDWARE_DIR, camera_config
from tests.test_esp32 import running, wait_for

FAKE_IMU_CLIENT = '''
from collections import deque

class TelemetryReceiver:
    def __init__(self):
        self.imu = deque(maxlen=4000)
        self.last_imu = None
    def start(self):
        raise AssertionError("no UDP in wired mode")
'''

FAKE_SERIAL_CLIENT = '''
import threading, time
from dataclasses import dataclass
import numpy as np

OPENED = []          # ports opened, in order (the tests inspect it)
CLOCK_OFFSET_US = 7_000_000

@dataclass
class Frame:
    img: np.ndarray
    dev_ts_us: int
    host_ts: float
    frame_id: int

@dataclass
class Imu:
    dev_us: int; host_ts: float; seq: int
    ax: float; ay: float; az: float; gx: float; gy: float; gz: float
    temp_c: float; valid: bool

class SerialCamera:
    def __init__(self, name, port, baud, tel):
        self.name, self.port, self.baud, self.tel = name, port, baud, tel
        self.state = "connecting"
        self._latest = None
        self._stop = threading.Event()
        self.paused = port.endswith("silent")

    @property
    def frame_age_s(self):
        return 0.0

    def get_latest(self):
        return self._latest

    def start(self):
        OPENED.append(self.port)
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        seq = imu_seq = 0
        next_frame = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            now_us = int(now * 1e6) - CLOCK_OFFSET_US
            if self.name == "left":
                s = Imu(now_us, now, imu_seq, 0.0, 0.0, 1.0, 90.0, 0.0, 0.0, 30.0,
                        imu_seq % 10 != 9)  # every tenth reading flagged invalid
                self.tel.imu.append(s)
                self.tel.last_imu = s
                imu_seq += 1
            if not self.paused and now >= next_frame:
                img = np.full((240, 320, 3), seq % 256, np.uint8)
                # A JPEG takes ~30 ms over the link: the frame is stamped at capture.
                self._latest = Frame(img, now_us - 30_000, now, seq)
                self.state = "streaming"
                seq += 1
                next_frame += 1 / 30
            time.sleep(0.01)
'''


@pytest.fixture(scope="session")
def receiver_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("receiver")
    (d / "imu_client.py").write_text(textwrap.dedent(FAKE_IMU_CLIENT))
    (d / "serial_client.py").write_text(textwrap.dedent(FAKE_SERIAL_CLIENT))
    return d


@pytest.fixture(autouse=True)
def _links_closed():
    yield
    close_all_links()


def wired_config(receiver_dir, camera_id="left_eye", port="/dev/cu.usbmodemA", **options):
    return camera_config(
        camera_id,
        source_type="esp32_serial",
        resolution=(320, 240),
        connect_timeout_s=2.0,
        read_timeout_s=0.5,
        options={"port": port, "receiver_dir": str(receiver_dir), **options},
    )


async def test_reads_frames_with_device_timestamps(receiver_dir):
    src = create_source(wired_config(receiver_dir))
    await src.connect()
    try:
        frames = [await src.read_frame() for _ in range(4)]
    finally:
        await src.disconnect()
    seqs = [f.sequence for f in frames]
    assert len(set(seqs)) == 4 and seqs == sorted(seqs)  # newest only, never repeated
    f = frames[-1]
    assert (f.width, f.height) == (320, 240)
    assert f.timestamp_source is TimestampSource.DEVICE
    assert f.capture_ns is not None and f.capture_ns <= f.host_receive_ns
    caps = src.get_capabilities()
    assert caps.backend == "serial-receiver" and caps.resolution == (320, 240)
    assert not caps.notes[0].startswith("EMULATED")


async def test_pty_port_is_labelled_emulated(receiver_dir):
    src = create_source(wired_config(receiver_dir, port="/dev/ttys004"))
    await src.connect()
    await src.disconnect()
    assert src.get_capabilities().notes[0].startswith("EMULATED")


async def test_missing_receiver_explains_setup(tmp_path):
    src = create_source(wired_config(tmp_path))
    with pytest.raises(CameraConnectionError, match="opengaze-hw.zip"):
        await src.connect()


async def test_no_frames_fails_connect_and_frees_port(receiver_dir):
    src = create_source(wired_config(receiver_dir, port="/dev/cu.silent"))
    with pytest.raises(CameraConnectionError, match="no new frame"):
        await src.connect()
    await asyncio.sleep(0)
    assert "/dev/cu.silent" in _LINKS  # lingers briefly, so a retry doesn't reopen it


async def test_camera_and_imu_share_one_port(receiver_dir):
    import serial_client  # the fake, loaded by the adapter

    serial_client.OPENED.clear()
    cam = create_source(wired_config(receiver_dir))
    imu = WiredImuSource({"board": "left", "receiver_dir": str(receiver_dir)})
    await cam.connect()
    imu.open()
    try:
        await asyncio.sleep(0.3)
        samples = []
        end = time.monotonic() + 0.1  # the fake never stops, so drain for a fixed time
        while time.monotonic() < end and (sample := imu.read(0.05)) is not None:
            samples.append(sample)
        await cam.read_frame()  # buffered at connect, before the IMU set the clock
        frame = await cam.read_frame()
    finally:
        imu.close()
        await cam.disconnect()
    assert serial_client.OPENED == ["/dev/cu.usbmodemA"]  # opened once, by whoever came first
    # Every reading since open is delivered once, in order; invalid ones counted, not passed.
    seqs = [s.sequence for s in samples]
    assert 10 < len(seqs) < 200 and seqs == sorted(set(seqs))
    assert all(q % 10 != 9 for q in seqs) and imu.stats()["invalid"] > 0
    sample = samples[-1]
    assert sample.device_id == "left_esp32"
    assert sample.accel[2] == pytest.approx(9.80665)  # g → m/s²
    assert sample.gyro[0] == pytest.approx(1.5707963, rel=1e-6)  # deg/s → rad/s
    # One clock, set by the IMU: the frame's ~30 ms on the wire shows up as transport delay
    # instead of being absorbed into the clock offset.
    assert frame.capture_ns is not None
    delay_ms = (frame.host_receive_ns - frame.capture_ns) / 1e6
    assert 25 <= delay_ms <= 60


def test_imu_without_a_port_says_how_to_set_one(receiver_dir):
    imu = WiredImuSource({"board": "nowhere", "receiver_dir": str(receiver_dir)})
    with pytest.raises(OSError, match="--nowhere-serial"):
        imu.open()


async def test_manager_streams_both_eyes(receiver_dir):
    left = create_source(wired_config(receiver_dir))
    right = create_source(wired_config(receiver_dir, "right_eye", port="/dev/cu.usbmodemB"))
    async with running(left, right) as manager:
        await wait_for(
            lambda: all(h.state is CameraState.STREAMING for h in manager.health().values())
            and all(
                manager.latest(cid, consume=False) is not None for cid in ("left_eye", "right_eye")
            )
        )


def test_profile_ports_from_cli_and_fingerprint_stable():
    path = HARDWARE_DIR / "esp32-wired.yaml"
    auto = check_hardware_profile(path, [], [])
    assert auto.ok  # ports are found when the cameras connect
    placeholder = check_hardware_profile(path, [], ["cameras.right_eye.options.port=PORT_B"])
    assert not placeholder.ok and any("--right-serial" in str(i) for i in placeholder.issues)
    a = check_hardware_profile(path, [], ["cameras.left_eye.options.port=/dev/cu.a",
                                          "cameras.right_eye.options.port=/dev/cu.b"])
    b = check_hardware_profile(path, [], ["cameras.left_eye.options.port=/dev/cu.c",
                                          "cameras.right_eye.options.port=/dev/cu.d"])
    assert a.ok and b.ok
    # Moving a board to another USB socket must not invalidate a calibration.
    assert profile_fingerprint(a.profile) == profile_fingerprint(b.profile)
    assert profile_fingerprint(auto.profile) == profile_fingerprint(a.profile)


def test_cli_serial_flags_expand_to_sets():
    from opengaze.cli import _sets, build_parser

    args = build_parser().parse_args(
        ["validate", "x.yaml", "--left-serial", "/dev/cu.a", "--set", "k=v"]
    )
    assert _sets(args) == ["k=v", "cameras.left_eye.options.port=/dev/cu.a"]


def test_python_313_thread_handle_clash_is_explained(tmp_path):
    import sys

    from opengaze.cameras.wired import load_receiver

    if sys.version_info < (3, 13):
        pytest.skip("threading.Thread._handle exists from Python 3.13")
    (tmp_path / "imu_client.py").write_text("class TelemetryReceiver: pass\n")
    (tmp_path / "serial_client.py").write_text(
        "import threading\n"
        "class SerialCamera(threading.Thread):\n"
        "    def _handle(self, *a): pass\n"
    )
    try:
        with pytest.raises(CameraConnectionError, match="_handle_packet"):
            load_receiver(str(tmp_path))
    finally:
        sys.path.remove(str(tmp_path.resolve()))


REAL_RECEIVER = HARDWARE_DIR.parents[1] / "opengaze-hw" / "receiver"


@pytest.mark.skipif(
    not (REAL_RECEIVER / "tools" / "sim_serial_device.py").is_file(),
    reason="the hardware team's receiver (opengaze-hw/) isn't in this checkout",
)
async def test_real_receiver_against_its_serial_simulator():
    """The hardware team's receiver and simulator, through our adapter. Still emulated."""
    import subprocess
    import sys

    sim = subprocess.Popen(
        [sys.executable, str(REAL_RECEIVER / "tools" / "sim_serial_device.py"), "--role", "left"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        port = sim.stdout.readline().strip()
        # macOS pseudo-terminals reject 921600; the simulator paces itself at 921600 anyway.
        config = camera_config(
            "left_eye", source_type="esp32_serial", resolution=(320, 240),
            connect_timeout_s=8.0, read_timeout_s=3.0,
            options={"port": port, "baud": 230400, "receiver_dir": str(REAL_RECEIVER)},
        )
        cam = create_source(config)
        imu = WiredImuSource({"board": "left", "receiver_dir": str(REAL_RECEIVER)})
        await cam.connect()
        imu.open()
        try:
            frames = [await cam.read_frame() for _ in range(5)]
            sample = imu.read(2.0)
        finally:
            imu.close()
            await cam.disconnect()
    finally:
        sim.kill()
        sim.wait()
    assert all((f.width, f.height) == (320, 240) for f in frames)
    assert frames[-1].timestamp_source is TimestampSource.DEVICE
    assert cam.get_capabilities().notes[0].startswith("EMULATED")
    assert sample is not None and sample.accel[2] == pytest.approx(9.80665, rel=0.01)

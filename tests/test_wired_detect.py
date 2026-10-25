"""Wired mode: finding the glasses on whichever USB ports they're plugged into.

A fake receiver stands in for the hardware team's: the port's name decides what it pretends
to be (``…L`` the left board, ``…R`` the right, anything else a device that streams frames but
sends no telemetry). Ports are real files in a temporary folder, so unplugging is deleting
one. None of this says anything about real boards.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
import time

import pytest

from opengaze.cameras import wired
from opengaze.cameras.registry import create_source
from opengaze.cameras.types import CameraConnectionError
from opengaze.cameras.wired import GlassesScan, close_all_links, find_board_port
from opengaze.imu.sources import WiredImuSource
from tests.helpers import camera_config

FAKE_IMU_CLIENT = '''
from collections import deque
from types import SimpleNamespace

class TelemetryReceiver:
    def __init__(self):
        self.imu = deque(maxlen=4000)
        self.last_imu = None
        self.dev = {1: SimpleNamespace(last_host_ts=0.0, status_host_ts=0.0),
                    2: SimpleNamespace(last_host_ts=0.0, status_host_ts=0.0)}
'''

FAKE_SERIAL_CLIENT = '''
import os, threading, time
from dataclasses import dataclass
import numpy as np

OPENED = []

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
        self.port, self.tel = port, tel
        self.device = {"L": 1, "R": 2}.get(port[-1])
        self.state = "connecting"
        self._latest = None
        self._stop = threading.Event()

    def get_latest(self):
        return self._latest

    def start(self):
        OPENED.append(self.port)
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        seq = 0
        while not self._stop.is_set():
            now = time.monotonic()
            if not os.path.exists(self.port):  # unplugged: nothing arrives
                self.state = "reconnecting"
                time.sleep(0.02)
                continue
            if self.device is not None:
                self.tel.dev[self.device].status_host_ts = now
            if self.device == 1:
                s = Imu(int(now * 1e6), now, seq, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0, True)
                self.tel.imu.append(s)
                self.tel.dev[1].last_host_ts = now
            img = np.full((240, 320, 3), 40 * (self.device or 0), np.uint8)
            self._latest = Frame(img, int(now * 1e6), now, seq)
            self.state = "streaming"
            seq += 1
            time.sleep(0.02)
'''


@pytest.fixture(scope="module")
def receiver_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("detect_receiver")
    (d / "imu_client.py").write_text(textwrap.dedent(FAKE_IMU_CLIENT))
    (d / "serial_client.py").write_text(textwrap.dedent(FAKE_SERIAL_CLIENT))
    return d


@pytest.fixture
def ports(tmp_path, monkeypatch):
    """Plug in fake ports: ``ports("modemL", "modemR")`` → their paths."""
    monkeypatch.setattr(wired, "PORT_PATTERNS", (str(tmp_path / "cu.usb*"),))

    def plug(*names: str) -> list[str]:
        paths = [str(tmp_path / f"cu.usb{name}") for name in names]
        for path in paths:
            open(path, "w").close()
        return paths

    yield plug
    close_all_links()


def opened():
    import serial_client  # the fake, loaded by the scan

    return serial_client.OPENED


def poll_until(scan: GlassesScan, boards: int, timeout_s: float = 2.0) -> dict[str, str]:
    deadline = time.monotonic() + timeout_s
    while True:
        found = scan.poll()
        if len(found) >= boards or time.monotonic() > deadline:
            return found
        time.sleep(0.02)


def test_scan_tells_left_from_right_and_ignores_other_devices(receiver_dir, ports):
    other, right, left = ports("serialX", "modemR", "modemL")
    scan = GlassesScan(receiver=str(receiver_dir))
    try:
        found = poll_until(scan, 2)
        assert found == {"left": left, "right": right}
        time.sleep(0.1)  # the other device's first frame
        image = scan.image("right", found)
        assert image is not None and image.shape == (240, 320, 3) and image[0, 0, 0] == 80
        assert f"{other}: sends frames but no board id" in scan.describe(found)
    finally:
        scan.close()


def test_scan_lets_go_of_unplugged_ports(receiver_dir, ports):
    (left,) = ports("modemL")
    scan = GlassesScan(receiver=str(receiver_dir))
    try:
        assert poll_until(scan, 1) == {"left": left}
        os.remove(left)
        assert scan.poll() == {} and left not in scan.links
    finally:
        scan.close()


def test_missing_board_is_explained(receiver_dir, ports):
    with pytest.raises(CameraConnectionError, match="no USB serial devices"):
        find_board_port("left", None, str(receiver_dir), timeout_s=0.2)
    ports("modemR")
    with pytest.raises(CameraConnectionError, match=r"left camera isn't connected.*right camera"):
        find_board_port("left", None, str(receiver_dir), timeout_s=0.5)


async def test_auto_port_cameras_find_their_boards_and_open_each_port_once(receiver_dir, ports):
    left, right = ports("modemL", "modemR")
    opened().clear()

    def config(camera_id):
        return camera_config(camera_id, source_type="esp32_serial", resolution=(320, 240),
                             connect_timeout_s=4.0, read_timeout_s=0.5,
                             options={"port": "auto", "receiver_dir": str(receiver_dir)})

    cams = [create_source(config("left_eye")), create_source(config("right_eye"))]
    await asyncio.gather(*(c.connect() for c in cams))  # connect together, as the manager does
    try:
        frames = [await c.read_frame() for c in cams]
    finally:
        for c in cams:
            await c.disconnect()
    assert [int(f.image[0, 0, 0]) for f in frames] == [40, 80]  # left board, right board
    assert "found automatically" in cams[0].get_capabilities().notes[0]
    assert sorted(opened()) == sorted([left, right])  # the scan's links were reused


def test_auto_profile_gives_each_eye_its_own_device_key():
    from opengaze.cameras.wired import SerialEyeSource

    keys = {SerialEyeSource.device_key(camera_config(cid, source_type="esp32_serial",
                                                     options={"port": "auto"}))
            for cid in ("left_eye", "right_eye")}
    assert len(keys) == 2


def test_imu_finds_the_left_board_and_reopens_when_it_moves(receiver_dir, ports):
    ports("modemR")
    (left,) = ports("modemL")
    imu = WiredImuSource({"board": "left", "port": "auto", "receiver_dir": str(receiver_dir)})
    imu.open()
    try:
        assert imu._link.port == left
        assert imu.read(1.0) is not None
        os.remove(left)  # unplugged; a reopen would look for it again
        with pytest.raises(OSError, match="is gone"):
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                imu.read(0.1)
    finally:
        imu.close()


# ---------------------------------------------------------------------------
# EyeMouse: the setup page, and the app running on the glasses
# ---------------------------------------------------------------------------


class FakeScan:
    def __init__(self) -> None:
        self.found: dict[str, str] = {}
        self.closed = False

    def poll(self) -> dict[str, str]:
        return dict(self.found)

    def image(self, board, found):
        return None

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def setup_window(scan):
    from opengaze.consumer.setup_window import SetupHooks, SetupWindow

    hooks = SetupHooks(
        calibrate=lambda: None, finish=lambda login: None, quit=lambda: None,
        voice_chosen=lambda on: None, has_voice_key=False, permission_app="Terminal",
        scan_glasses=(lambda: scan) if scan is not None else None,
    )
    return SetupWindow(hooks, open_at_login=False)


def test_glasses_page_waits_for_both_cameras(qapp):
    from opengaze.consumer.setup_window import GlassesPage, SetupWindow

    scan = FakeScan()
    window = setup_window(scan)
    window.go(SetupWindow.GLASSES)
    page = window.pages[SetupWindow.GLASSES]
    assert isinstance(page, GlassesPage)
    assert not page.primary.isEnabled()
    scan.found = {"left": "/dev/a"}
    page._check()
    assert not page.primary.isEnabled() and page.previews["left"].connected
    scan.found = {"left": "/dev/a", "right": "/dev/b"}
    page._check()
    assert page.primary.isEnabled()
    page.primary.click()
    assert scan.closed
    assert window.stack.currentIndex() == SetupWindow.CAMERA


def test_glasses_page_skipped_for_a_webcam_dev_profile(qapp):
    from opengaze.consumer.setup_window import SetupWindow, WelcomePage

    window = setup_window(None)
    welcome = window.pages[SetupWindow.WELCOME]
    assert isinstance(welcome, WelcomePage)
    welcome.findChildren(type(window.pages[SetupWindow.CAMERA].primary))[0].click()
    assert window.stack.currentIndex() == SetupWindow.CAMERA


def test_app_runs_on_the_glasses_unless_a_developer_says_otherwise():
    from opengaze.cli import GLASSES_PROFILE, build_parser

    args = build_parser().parse_args(["app"])
    assert args.profile is None  # cmd_app then loads GLASSES_PROFILE
    assert GLASSES_PROFILE.name == "esp32-wired.yaml" and GLASSES_PROFILE.is_file()


# ---------------------------------------------------------------------------
# Problems the person is shown (never only logged)
# ---------------------------------------------------------------------------


def _manager_problem(receiver_dir, seconds: float):
    """Run the glasses profile's eye cameras for ``seconds``; the problem they add up to."""
    from opengaze.cameras.manager import CameraManager
    from opengaze.config.loader import check_hardware_profile
    from opengaze.config.schema import CaptureSettings
    from opengaze.consumer.problems import STARTUP_GRACE_S, find_problem
    from tests.helpers import HARDWARE_DIR

    sets = ["cameras.head.enabled=false"]
    for side in ("left", "right"):
        sets += [f"cameras.{side}_eye.options.receiver_dir={receiver_dir}",
                 f"cameras.{side}_eye.connect_timeout_s=1.0"]
    report = check_hardware_profile(HARDWARE_DIR / "esp32-wired.yaml", [], sets)
    assert report.ok, report.issues
    manager = CameraManager.from_profile(
        report.profile, CaptureSettings(reconnect_max_delay_s=0.5)
    )

    async def run():
        task = asyncio.create_task(manager.run())
        await asyncio.sleep(seconds)
        health = manager.health()  # while running: stopping marks every camera stopped
        manager.request_stop()
        await task
        return health

    return find_problem(asyncio.run(run()), None, STARTUP_GRACE_S)


def test_unplugged_glasses_become_a_visible_problem(receiver_dir, ports):
    problem = _manager_problem(receiver_dir, 1.5)
    assert problem is not None and problem.severity == "error"
    assert problem.title == "Glasses not connected"
    assert "Plug both glasses cables" in problem.detail
    assert "no USB serial devices" in problem.technical  # the real reason, for the log


def test_one_missing_eye_camera_is_named(receiver_dir, ports):
    ports("modemL")
    problem = _manager_problem(receiver_dir, 2.0)
    assert problem is not None and problem.title == "Right eye camera not connected"


def _health(camera_id, state, error=None):
    from opengaze.cameras.manager import CameraHealth
    from opengaze.cameras.types import CameraRole, SourceKind

    return CameraHealth(camera_id, CameraRole(camera_id), "esp32_serial", SourceKind.LIVE,
                        state, 0, 0, 0, 0, None, None, None, None, 0, error, None)


def test_find_problem_cases():
    from opengaze.cameras.types import CameraState as S
    from opengaze.consumer.problems import find_problem
    from opengaze.imu.runtime import ImuHealth

    ok = {cid: _health(cid, S.STREAMING) for cid in ("left_eye", "right_eye", "head")}
    assert find_problem(ok, ImuHealth("streaming", 10, 100.0, None), 60) is None

    down = {**ok, "left_eye": _health("left_eye", S.RECONNECTING, "the glasses' left camera "
                                      "isn't connected: no USB serial devices are plugged in")}
    # The search has given up: shown at once, even while starting up.
    assert find_problem(down, None, 3).title == "Left eye camera not connected"
    connecting = {**ok, "left_eye": _health("left_eye", S.CONNECTING)}
    assert find_problem(connecting, None, 3) is None  # no answer yet
    assert find_problem(connecting, None, 60).title == "The left eye camera isn’t sending video"

    stalled = {**ok, "right_eye": _health("right_eye", S.STALLED)}
    assert find_problem(stalled, None, 60).title == "The right eye camera isn’t sending video"

    failed = {**ok, "left_eye": _health("left_eye", S.FAILED, "bad option")}
    problem = find_problem(failed, None, 0)  # shown at once: it won't fix itself
    assert problem.title == "The glasses can’t start" and "bad option" in problem.detail

    missing = {**ok, "left_eye": _health("left_eye", S.RECONNECTING,
                                         "wired-mode receiver not found at /x")}
    assert find_problem(missing, None, 60).title == "The glasses’ software is missing"

    head = {**ok, "head": _health("head", S.RECONNECTING, "camera busy")}
    assert find_problem(head, None, 60).title == "This Mac’s camera isn’t working"
    assert find_problem(head, None, 60, camera_denied=True).title == "Camera access is off"

    imu = find_problem(ok, ImuHealth("connecting", 0, None, "no MPU-6050"), 60)
    assert imu.severity == "warning" and imu.technical == "no MPU-6050"

    assert find_problem(ok, None, 60, capture_running=False).severity == "error"


def test_problem_monitor_waits_then_clears_at_once():
    from opengaze.consumer.problems import Problem, ProblemMonitor

    monitor = ProblemMonitor(show_after_s=2.0)
    p = Problem("Glasses not connected", "Plug them in.", "error")
    assert monitor.update(p, 0.0) is None
    assert monitor.update(p, 1.0) is None  # a reconnect would be over by now
    assert monitor.update(p, 2.0) == p
    assert monitor.update(None, 2.5) is None and monitor.shown is None
    assert monitor.update(p, 3.0) is None  # back: waits again


def test_banner_and_menu_show_the_problem(qapp):
    from opengaze.consumer import theme
    from opengaze.consumer.menubar import MenuBar, MenuHooks
    from opengaze.consumer.problems import Problem
    from opengaze.consumer.warning_banner import WarningBanner

    theme.load_fonts()
    problem = Problem("Glasses not connected",
                      "Plug both glasses cables into this Mac. Gaze control is off until "
                      "they’re back.", "error")
    banner = WarningBanner()
    banner.show_problem(problem)
    assert banner.isVisible() and banner.height() > 60
    banner.grab()  # paints without errors
    banner.clear()
    assert not banner.isVisible()

    shown: list[Problem | None] = [problem]
    status = ["tracking"]
    hooks = MenuHooks(gaze_on=lambda: True, status=lambda: status[0], set_gaze=lambda on: None,
                      toggle_pause=lambda: None, toggle_tracking=lambda: None,
                      voice_on=lambda: False,
                      set_voice=lambda on: None, keyboard=lambda: None,
                      recalibrate=lambda: None, settings=lambda: None, quit=lambda: None,
                      problem=lambda: shown[0])
    menu = MenuBar(hooks)
    menu.refresh()
    assert menu.note.isVisible() and menu.note.text() == "⚠ Glasses not connected"
    assert "Glasses not connected" in menu.tray.toolTip()
    shown[0] = None
    menu.refresh()
    assert not menu.note.isVisible()

    assert menu.tracking.text() == "Stop tracking" and menu.pause.isEnabled()
    status[0] = "off"  # ⌃⌥T: the gaze loop is stopped, so pausing means nothing
    menu.refresh()
    assert menu.tracking.text() == "Start tracking" and not menu.pause.isEnabled()
    assert menu.status.text() == "Tracking off"


def test_setup_runs_again_on_other_hardware(tmp_path):
    from opengaze.config.loader import check_hardware_profile
    from opengaze.consumer.app import needs_setup
    from opengaze.consumer.settings import AppSettings
    from tests.helpers import HARDWARE_DIR

    glasses = check_hardware_profile(HARDWARE_DIR / "esp32-wired.yaml")
    webcam = check_hardware_profile(HARDWARE_DIR / "webcam-only.yaml")
    old = AppSettings.load(tmp_path / "eyemouse.json")
    old.setup_done = True  # a settings file from before setup_hardware: webcam-only setup
    assert needs_setup(old, glasses)
    assert not needs_setup(old, webcam)  # the dev fallback goes straight to the menu bar
    old.setup_hardware = "esp32-wired.yaml"
    assert not needs_setup(old, glasses) and needs_setup(old, glasses, force=True)
    assert not needs_setup(old, webcam)


def test_calibration_shows_the_problem_and_waits(qapp, monkeypatch):
    from opengaze.cameras.manager import CameraManager
    from opengaze.cameras.runtime import CaptureRuntime
    from opengaze.config.loader import check_hardware_profile, load_app_config
    from opengaze.consumer import calibration as cal
    from opengaze.consumer import theme
    from opengaze.consumer.problems import Problem
    from opengaze.desktop.calibration_ui import _State
    from tests.helpers import HARDWARE_DIR, REPO_ROOT

    theme.load_fonts()
    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-dual-eye.yaml")
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    problem: list[Problem | None] = [Problem("Glasses not connected", "Plug them in.", "error")]
    outcomes: list[str] = []
    win = cal.EyeMouseCalibration(runtime, report, config, outcomes.append, lambda: problem[0])
    try:
        win._eye_ready = True
        win._on_advance()  # Start, while the glasses are unplugged
        assert win._state == _State.IDLE
        win.grab()  # paints the banner over the intro card
        problem[0] = Problem("Head motion sensor not working", "Still works.", "warning")
        win._on_advance()  # a warning doesn't block
        assert win._state == _State.COLLECTING
    finally:
        win.close()
        runtime.stop()
    assert outcomes == ["cancelled"]

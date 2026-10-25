"""ESP32-CAM eye cameras on USB serial (wired mode), through the hardware team's receiver.

Each ESP32-CAM is bridged to the Mac by its own Seeed XIAO RA4M1, so the Mac sees one serial
port per camera (``/dev/cu.usbmodem…``). The firmware sends CRC-checked, framed JPEGs at
921600 baud; the left board also sends MPU-6050 packets on the same port. The protocol is
parsed by the hardware team's receiver (``opengaze-hw/receiver``: ``serial_client.py``,
``imu_client.py``, …), which this module loads from ``receiver_dir`` and wraps behind the
``CameraSource`` and ``ImuSource`` contracts. Nothing downstream changes.

Only one program can hold a port, and the left camera's port carries the IMU too, so the
camera adapter and ``WiredImuSource`` share one ``WiredLink`` per port. The receiver
reconnects by itself when a cable is unplugged, so a link outlives short reconnects of its
users instead of reopening the port each time.

Port ``auto`` (the default) finds the board when the camera connects: every port a bridge
could be on is opened and watched until the board names itself (each board sends a STATUS
packet with its device id once a second). Reconnects look again, so moving a cable to another
USB socket doesn't need a new port name.

Timing: frames and IMU samples from one board share its clock (``dev_ts_us``), so they share
one ``DeviceClockSync``. IMU packets are small and give the tighter offset bound; once they
arrive, frames are mapped with that offset instead of updating it, which keeps frame and IMU
times aligned with each other. A camera with no IMU maps its own frames, so its times land
late by the shortest time a JPEG takes over the link. Left/right alignment is approximate
(no better than about one frame period until it is measured).
"""

from __future__ import annotations

import glob
import importlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from opengaze.cameras.base import BlockingCaptureSource, ConfigIssue, RawCapture
from opengaze.cameras.opencv_source import resolution_result
from opengaze.cameras.timing import DeviceClockSync
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConnectionError,
    CameraDisconnectedError,
    FrameStatus,
    PixelFormat,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CameraConfig

DEFAULT_BAUD = 921600
# Where the receiver is looked for when neither ``receiver_dir`` nor the environment says.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIVER_DIR = REPO_ROOT / "opengaze-hw" / "receiver"
RECEIVER_ENV = "OPENGAZE_HW_RECEIVER"
_RECEIVER_MODULES = ("serial_client", "imu_client", "camera_client", "synchronization")
# How often the receiver's latest-frame slot is polled (bounds the added latency).
_FRAME_POLL_S = 0.002
# A link with no users is closed after this long, so a reconnect doesn't reopen the port.
_LINGER_S = 3.0

AUTO_PORT = "auto"
# Where USB-serial bridges appear on macOS: the XIAO (native USB) as usbmodem, CH340 and FTDI
# bridges as usbserial.
PORT_PATTERNS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*")
# The firmware's telemetry device ids.
DEVICE_BOARDS = {1: "left", 2: "right"}
# A board counts as present while its telemetry is this recent (STATUS comes once a second).
_FRESH_S = 2.5
# How long a search waits for the boards to name themselves.
DETECT_TIMEOUT_S = 4.0


# ---------------------------------------------------------------------------
# Loading the receiver
# ---------------------------------------------------------------------------


def receiver_dir(configured: str | None = None) -> Path:
    return Path(configured or os.environ.get(RECEIVER_ENV) or DEFAULT_RECEIVER_DIR).expanduser()


def load_receiver(configured: str | None = None) -> tuple[type, type]:
    """``(SerialCamera, TelemetryReceiver)`` from the hardware team's receiver package.

    Its modules import each other by top-level name, so their directory goes first on
    ``sys.path`` while they are imported, then last: it also has a ``main.py`` and a
    ``diagnostics.py``, which must not shadow anything else afterwards.
    Raises ``CameraConnectionError`` with setup instructions if it is missing.
    """
    directory = receiver_dir(configured)
    if not (directory / "serial_client.py").is_file():
        raise CameraConnectionError(
            f"wired-mode receiver not found at {directory}: unzip opengaze-hw.zip into this "
            f"repo or set options.receiver_dir (or ${RECEIVER_ENV}) to its receiver/ folder, "
            "then `pip install pyserial`"
        )
    path = str(directory.resolve())
    loaded = sys.modules.get("serial_client")
    if loaded is not None and Path(loaded.__file__ or "").resolve().parent != Path(path):
        for name in _RECEIVER_MODULES:  # imported from another receiver folder
            sys.modules.pop(name, None)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    try:
        serial_client = importlib.import_module("serial_client")
        imu_client = importlib.import_module("imu_client")
    except ImportError as exc:
        raise CameraConnectionError(
            f"could not import the wired-mode receiver from {directory}: {exc} "
            "(its requirements.txt needs pyserial)"
        ) from exc
    finally:
        sys.path.remove(path)
        sys.path.append(path)
    classes = (serial_client.SerialCamera, imu_client.TelemetryReceiver)
    for cls in classes:
        if sys.version_info >= (3, 13) and callable(vars(cls).get("_handle")):
            # threading.Thread sets self._handle in __init__ since 3.13, hiding a method of
            # that name: every packet would raise "'_ThreadHandle' object is not callable".
            raise CameraConnectionError(
                f"the receiver in {directory} defines {cls.__name__}._handle, which Python "
                "3.13+ hides behind threading.Thread._handle; rename it (e.g. _handle_packet)"
            )
    return classes


# ---------------------------------------------------------------------------
# One serial port, shared by a camera and (left board) the IMU
# ---------------------------------------------------------------------------


class WiredLink:
    """One ``SerialCamera`` + its ``TelemetryReceiver`` on one port, reference counted."""

    def __init__(self, board: str, port: str, baud: int, receiver: str | None) -> None:
        self.board = board
        self.port = port
        self.baud = baud
        self.receiver = receiver
        self.clock = DeviceClockSync()
        self.clock_lock = threading.Lock()
        self.imu_seen = False  # IMU packets arrived: they, not frames, set the clock offset
        self.camera: Any = None
        self.telemetry: Any = None
        self._users = 0
        self._close_timer: threading.Timer | None = None

    def _start(self) -> None:
        serial_camera, telemetry_receiver = load_receiver(self.receiver)
        # No telemetry.start(): it is the Wi-Fi UDP listener. In wired mode the serial
        # client feeds IMU packets into it.
        self.telemetry = telemetry_receiver()
        self.camera = serial_camera(self.board, self.port, self.baud, self.telemetry)
        self.camera.start()

    def _stop(self) -> None:
        camera, self.camera, self.telemetry = self.camera, None, None
        stop = getattr(camera, "stop", None) or getattr(camera, "close", None)
        if callable(stop):
            stop()

    @property
    def state(self) -> str:
        return str(getattr(self.camera, "state", "stopped"))

    def identity(self) -> str | None:
        """The board on this port (``left`` / ``right``), from the device id of its recent
        telemetry; None until it has sent some."""
        devices = getattr(self.telemetry, "dev", None) or {}
        now = time.monotonic()
        for device_id, board in DEVICE_BOARDS.items():
            d = devices.get(device_id)
            if d is None:
                continue
            last = max(getattr(d, "last_host_ts", 0.0), getattr(d, "status_host_ts", 0.0))
            if last and now - last < _FRESH_S:
                return board
        return None

    def latest_image(self) -> np.ndarray | None:
        frame = self.camera.get_latest() if self.camera is not None else None
        image = _get(frame, "img", "image") if frame is not None else None
        return image if isinstance(image, np.ndarray) and image.size else None


_LINKS: dict[str, WiredLink] = {}
_LINKS_LOCK = threading.Lock()
# The (port, baud) each board's camera was configured with, so the IMU can find its board.
_BOARD_PORTS: dict[str, tuple[str, int]] = {}


def acquire_link(board: str, port: str, baud: int | None, receiver: str | None) -> WiredLink:
    """The link on ``port``, opened if needed. ``baud=None`` accepts the open link's."""
    with _LINKS_LOCK:
        link = _LINKS.get(port)
        if link is None:
            link = WiredLink(board, port, baud or DEFAULT_BAUD, receiver)
            link._start()
            _LINKS[port] = link
        elif baud is not None and link.baud != baud:
            raise CameraConnectionError(
                f"{port} is already open at {link.baud} baud; both users must agree"
            )
        if link._close_timer is not None:
            link._close_timer.cancel()
            link._close_timer = None
        link._users += 1
        return link


def release_link(link: WiredLink, linger_s: float = _LINGER_S) -> None:
    with _LINKS_LOCK:
        link._users -= 1
        if link._users > 0:
            return
        if linger_s <= 0:
            _close_unused(link)
            return
        timer = threading.Timer(linger_s, _close_if_unused, (link,))
        timer.daemon = True
        link._close_timer = timer
        timer.start()


def _close_if_unused(link: WiredLink) -> None:
    with _LINKS_LOCK:
        if link._users <= 0:
            _close_unused(link)


def _close_unused(link: WiredLink) -> None:
    if _LINKS.get(link.port) is link:
        del _LINKS[link.port]
    link._close_timer = None
    link._stop()


def close_all_links() -> None:
    """Stop every link now (tests, and shutdown paths that must free the ports)."""
    with _LINKS_LOCK:
        for link in list(_LINKS.values()):
            if link._close_timer is not None:
                link._close_timer.cancel()
            link._users = 0
            _close_unused(link)


def board_port(board: str) -> tuple[str, int] | None:
    return _BOARD_PORTS.get(board)


# ---------------------------------------------------------------------------
# Finding the boards
# ---------------------------------------------------------------------------


def candidate_ports() -> list[str]:
    """Serial ports a bridge could be on, right now."""
    return sorted({port for pattern in PORT_PATTERNS for port in glob.glob(pattern)})


class GlassesScan:
    """Opens every candidate port and watches which board answers on each.

    Each port is held as a shared link until ``close()``, which lets it linger: a camera
    opened right after a scan reuses the link instead of reopening the port. Other devices on
    those ports (another Arduino, say) are opened too, and let go of when the scan closes.
    """

    def __init__(self, baud: int | None = None, receiver: str | None = None) -> None:
        self.baud, self.receiver = baud, receiver
        self.links: dict[str, WiredLink] = {}
        self.skipped: dict[str, str] = {}  # port → why it couldn't be opened

    def poll(self) -> dict[str, str]:
        """``{board: port}`` for the boards heard from recently. Opens ports that appeared
        since the last poll and lets go of ones that disappeared."""
        load_receiver(self.receiver)  # missing receiver: one clear error, not one per port
        present = set(candidate_ports())
        for port in [p for p in self.links if p not in present]:
            release_link(self.links.pop(port))
        for port in sorted(present - self.links.keys() - self.skipped.keys()):
            try:
                self.links[port] = acquire_link("glasses", port, self.baud, self.receiver)
            except CameraConnectionError as exc:
                self.skipped[port] = str(exc)
        found: dict[str, str] = {}
        for port, link in self.links.items():
            board = link.identity()
            if board is not None:
                found.setdefault(board, port)
        return found

    def image(self, board: str, found: dict[str, str]) -> np.ndarray | None:
        port = found.get(board)
        link = self.links.get(port) if port is not None else None
        return link.latest_image() if link is not None else None

    def describe(self, found: dict[str, str]) -> str:
        """What each port is doing, for a "not found" message."""
        if not self.links and not self.skipped:
            return "no USB serial devices are plugged in (looked for " + ", ".join(
                PORT_PATTERNS) + ")"
        parts = []
        for port, link in sorted(self.links.items()):
            board = next((b for b, p in found.items() if p == port), None)
            if board is not None:
                parts.append(f"{port}: {board} camera")
            elif link.latest_image() is not None:
                parts.append(f"{port}: sends frames but no board id")
            else:
                parts.append(f"{port}: nothing from it (state: {link.state})")
        parts += [f"{port}: {why}" for port, why in sorted(self.skipped.items())]
        return "; ".join(parts)

    def close(self) -> None:
        links, self.links = self.links, {}
        for link in links.values():
            release_link(link)


# One search at a time: the left and right cameras connect together and both want every port.
_DETECT_LOCK = threading.Lock()


def find_board_port(
    board: str, baud: int | None, receiver: str | None, timeout_s: float = DETECT_TIMEOUT_S
) -> str:
    """The port the ``board`` camera is plugged into. Raises ``CameraConnectionError``."""
    deadline = time.monotonic() + timeout_s
    if not _DETECT_LOCK.acquire(timeout=timeout_s):
        # The other camera's search ran the whole time, so it found nothing either.
        raise CameraConnectionError(
            f"the glasses' {board} camera isn't connected (the other camera's search found "
            "nothing in time)"
        )
    scan = GlassesScan(baud, receiver)
    try:
        while True:
            found = scan.poll()
            if board in found:
                return found[board]
            if time.monotonic() >= deadline:
                raise CameraConnectionError(
                    f"the glasses' {board} camera isn't connected: {scan.describe(found)}"
                )
            time.sleep(0.05)
    finally:
        scan.close()
        _DETECT_LOCK.release()


def _get(obj: Any, *names: str) -> Any:
    """First present attribute (or mapping key) of ``names``; the receiver's objects may be
    dataclasses, namedtuples or dicts."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


def _inside(timeout_s: float) -> float:
    """Give up just before the base class's timeout does: it would abandon the worker
    without releasing the link, and its message wouldn't say why."""
    return max(0.05, timeout_s - max(0.1, 0.1 * timeout_s))


_PTY_BAUD_HINT = (
    ". macOS pseudo-terminals reject non-standard baud rates such as 921600; for the "
    "simulator use --set cameras.left_eye.options.baud=230400 (and right_eye); the simulator "
    "still paces itself at 921600"
)


def _is_pty(port: str) -> bool:
    # tools/sim_serial_device.py serves pseudo-terminals; real XIAOs are /dev/cu.usbmodem*.
    return port.startswith(("/dev/pts/", "/dev/ttys"))


# ---------------------------------------------------------------------------
# Camera adapter
# ---------------------------------------------------------------------------


class SerialEyeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # /dev/cu.usbmodem… on macOS, or ``auto`` to find the board on whichever port it's on
    # (it changes when a board moves to another USB socket). --left-serial / --right-serial
    # set it on the command line.
    port: str = AUTO_PORT
    baud: int = Field(default=DEFAULT_BAUD, gt=0)
    # Which board this is (the receiver's name for it); the left one carries the IMU.
    board: str | None = None  # default: from the camera role (left_eye → left)
    # The hardware team's receiver/ folder (default: $OPENGAZE_HW_RECEIVER, then
    # opengaze-hw/receiver in this repo).
    receiver_dir: str | None = None
    # Map dev_ts_us onto the host clock. false = use host receive times only.
    device_clock: bool = True


class SerialEyeSource(BlockingCaptureSource):
    source_type = "esp32_serial"
    source_kind = SourceKind.LIVE
    Options: ClassVar[type[BaseModel]] = SerialEyeOptions
    # Port names move between USB sockets; none of these move the image.
    non_geometric_options = frozenset({"port", "baud", "board", "receiver_dir", "device_clock"})

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        o: SerialEyeOptions = self.options
        self.board = o.board or config.role.value.removesuffix("_eye")
        _BOARD_PORTS[self.board] = (o.port, o.baud)
        self._link: WiredLink | None = None
        self._last_key: Any = None
        self._pending: RawCapture | None = None

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        issues = super().check_config(config)
        if issues:
            return issues
        o: SerialEyeOptions = cls.parse_options(config)
        if not (receiver_dir(o.receiver_dir) / "serial_client.py").is_file():
            issues.append(
                ConfigIssue(
                    "warning",
                    f"{config.camera_id}: wired-mode receiver not found at "
                    f"{receiver_dir(o.receiver_dir)} (set options.receiver_dir or "
                    f"${RECEIVER_ENV})",
                )
            )
        if o.port.startswith("PORT_") or not o.port:
            board = o.board or config.role.value.removesuffix("_eye")
            issues.append(
                ConfigIssue(
                    "error",
                    f"{config.camera_id}: set the serial port to auto, or e.g. "
                    f"--{board}-serial /dev/cu.usbmodem1101 (`ls /dev/cu.usbmodem*` lists them)",
                )
            )
        return issues

    @classmethod
    def device_key(cls, config: CameraConfig) -> tuple[Any, ...] | None:
        try:
            options: SerialEyeOptions = cls.parse_options(config)
        except Exception:
            return None
        if options.port == AUTO_PORT:  # one auto port per board
            return ("serial", AUTO_PORT, options.board or config.role.value.removesuffix("_eye"))
        return ("serial", options.port)

    # -- blocking driver ---------------------------------------------------------------

    def _open_blocking(self) -> Any:
        o: SerialEyeOptions = self.options
        budget = _inside(self.config.connect_timeout_s)
        deadline = time.monotonic() + budget
        port = o.port
        if port == AUTO_PORT:
            port = find_board_port(self.board, o.baud, o.receiver_dir,
                                   min(DETECT_TIMEOUT_S, 0.75 * budget))
            _BOARD_PORTS[self.board] = (port, o.baud)
        link = acquire_link(self.board, port, o.baud, o.receiver_dir)
        try:
            first = self._next_capture(link, max(0.5, deadline - time.monotonic()))
        except CameraDisconnectedError as exc:
            release_link(link)
            raise CameraConnectionError(f"{port}: {exc}") from None
        self._link = link
        self._pending = first
        actual = (int(first.image.shape[1]), int(first.image.shape[0]))
        notes = [
            f"serial {port} @ {o.baud} baud ({self.board} board"
            + (", found automatically)" if o.port == AUTO_PORT else ")"),
            "wired mode: hardware team's receiver (CRC-checked JPEG frames)",
        ]
        if _is_pty(port):
            notes.insert(
                0, "EMULATED (pseudo-terminal: tools/sim_serial_device.py), not a USB device"
            )
        has_device_ts = first.timestamp_source is TimestampSource.DEVICE
        notes.append(
            "capture times: ESP32 clock mapped to host (min-offset sync, shared with the IMU)"
            if has_device_ts
            else "capture times: host receive"
        )
        self._capabilities = CameraCapabilities(
            source_type=self.source_type,
            source_kind=self.source_kind,
            backend="serial-receiver",
            device_name=f"{self.board}_esp32",
            resolution=actual,
            pixel_formats=(PixelFormat.BGR8,),
            provides_device_timestamps=has_device_ts,
            intrinsics=self._intrinsics,
            distortion=self._distortion,
            setting_results=(resolution_result(self.config.resolution, actual),),
            notes=tuple(notes),
        )
        return link

    def _read_blocking(self, handle: Any) -> RawCapture:
        if self._pending is not None:
            raw, self._pending = self._pending, None
            return raw
        return self._next_capture(handle, _inside(self.config.read_timeout_s))

    def _close_blocking(self, handle: Any) -> None:
        self._pending = None
        self._link = None
        release_link(handle)

    # -- helpers -----------------------------------------------------------------------

    def _next_capture(self, link: WiredLink, timeout_s: float) -> RawCapture:
        """Wait for a frame newer than the last one returned."""
        deadline = time.monotonic() + timeout_s
        while True:
            frame = link.camera.get_latest() if link.camera is not None else None
            if frame is not None:
                key = _get(frame, "frame_id", "dev_ts_us")
                key = id(frame) if key is None else key
                if key != self._last_key:
                    self._last_key = key
                    # The receiver stamps arrival with time.monotonic(): same clock.
                    host_ts = _get(frame, "host_ts")
                    host_ns = int(host_ts * 1e9) if host_ts else time.monotonic_ns()
                    raw = self._to_raw(link, frame, host_ns)
                    if raw is not None:
                        return raw
            if time.monotonic() >= deadline:
                error = getattr(link.camera, "last_error", "")
                raise CameraDisconnectedError(
                    f"no new frame within {timeout_s:.1f}s (receiver state: {link.state}"
                    + (f": {error}" if error else "")
                    + ")"
                    + (_PTY_BAUD_HINT if "Errno 25" in error and _is_pty(link.port) else "")
                )
            time.sleep(_FRAME_POLL_S)

    def _to_raw(self, link: WiredLink, frame: Any, host_ns: int) -> RawCapture | None:
        o: SerialEyeOptions = self.options
        image = _get(frame, "img", "image")
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None
        capture_ns = None
        source = TimestampSource.HOST_RECEIVE
        uncertainty = None
        device_us = _get(frame, "dev_ts_us")
        if device_us is not None and o.device_clock:
            with link.clock_lock:
                mapped = link.clock.map(int(device_us)) if link.imu_seen else None
                if mapped is None:
                    mapped = link.clock.update(int(device_us), host_ns)
                uncertainty = link.clock.uncertainty_ns
            capture_ns = min(mapped, host_ns)  # never after it was received
            source = TimestampSource.DEVICE
        sequence = _get(frame, "frame_id")
        if sequence is not None and int(sequence) < 0:  # the Wi-Fi client's "no header"
            sequence = None
        return RawCapture(
            image=image,
            host_receive_ns=host_ns,
            capture_ns=capture_ns,
            timestamp_source=source,
            timestamp_uncertainty_ns=uncertainty,
            status=FrameStatus.OK,
            warnings=("emulated device",) if _is_pty(link.port) else (),
            sequence=int(sequence) if sequence is not None else None,
        )

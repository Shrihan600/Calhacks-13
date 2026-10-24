"""Hardware-neutral camera data contract.

Every ``CameraSource`` produces ``CameraFrame`` objects, whatever the hardware. Anything a device
cannot report is ``None`` (unknown), never a guessed value.

Timestamps use the host's ``time.monotonic_ns()`` clock throughout.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


class CameraRole(enum.StrEnum):
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    HEAD = "head"
    OTHER = "other"


class SourceKind(enum.StrEnum):
    """Whether frames come from a live sensor. Every UI must label non-live data as such."""

    LIVE = "live"
    RECORDED = "recorded"
    SYNTHETIC = "synthetic"


class PixelFormat(enum.StrEnum):
    BGR8 = "bgr8"
    GRAY8 = "gray8"
    RGB8 = "rgb8"


class TimestampSource(enum.StrEnum):
    """Where ``CameraFrame.capture_ns`` came from, i.e. how far it can be trusted."""

    DEVICE = "device"  # exposure/presentation time from the device, mapped to the host clock
    HOST_RECEIVE = "host_receive"  # nothing better than host receive time is known
    MEDIA_TIME = "media_time"  # position in a recording, mapped onto the host clock at playback
    SYNTHETIC = "synthetic"  # generated; exact by construction


class FrameStatus(enum.StrEnum):
    OK = "ok"
    PARTIAL = "partial"  # decoded, but the data is known or suspected to be damaged


class CameraState(enum.StrEnum):
    IDLE = "idle"  # created, never connected
    CONNECTING = "connecting"
    STREAMING = "streaming"
    STALLED = "stalled"  # connected, but no new frame within the stale timeout
    RECONNECTING = "reconnecting"
    FAILED = "failed"  # unrecoverable (e.g. invalid configuration); no further retries
    STOPPED = "stopped"  # disconnected on purpose, or a recording ended


class ControlSupport(enum.StrEnum):
    SUPPORTED = "supported"  # applied and confirmed by reading it back
    IGNORED = "ignored"  # the driver accepted the request but reports a different value
    UNSUPPORTED = "unsupported"  # the driver rejected the request
    UNKNOWN = "unknown"  # not requested, or the driver cannot report it


@dataclass(frozen=True, slots=True)
class Orientation:
    """Correction that turns a raw image into the canonical upright, unmirrored image.

    Applied in this order: horizontal mirror, vertical mirror, then clockwise rotation.
    Only right angles are supported; arbitrary mounting angles belong in the profile's geometry
    block, where the calibration model accounts for them.
    """

    rotation_degrees: int = 0
    mirror_horizontal: bool = False
    mirror_vertical: bool = False

    def __post_init__(self) -> None:
        if self.rotation_degrees not in (0, 90, 180, 270):
            raise ValueError(
                f"rotation_degrees must be 0, 90, 180 or 270, got {self.rotation_degrees!r}"
            )

    def canonical(self) -> Orientation:
        """The same transform expressed with horizontal mirroring and rotation only."""
        # A vertical flip equals a horizontal flip followed by a 180° rotation.
        rotation = (self.rotation_degrees + (180 if self.mirror_vertical else 0)) % 360
        return Orientation(rotation, self.mirror_horizontal != self.mirror_vertical, False)

    @property
    def is_identity(self) -> bool:
        c = self.canonical()
        return c.rotation_degrees == 0 and not c.mirror_horizontal

    def inverse(self) -> Orientation:
        c = self.canonical()
        if c.mirror_horizontal:
            # A mirror followed by a rotation is a reflection, which is its own inverse.
            return c
        return Orientation((360 - c.rotation_degrees) % 360)

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        if self.rotation_degrees in (90, 270):
            return height, width
        return width, height

    def transform_point(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        """Map a pixel coordinate in the raw image to the corrected image.

        Uses pixel-index coordinates (0 … width-1), matching ``cv2.flip`` / ``cv2.rotate``.
        """
        if self.mirror_horizontal:
            x = (width - 1) - x
        if self.mirror_vertical:
            y = (height - 1) - y
        if self.rotation_degrees == 90:
            return (height - 1) - y, x
        if self.rotation_degrees == 180:
            return (width - 1) - x, (height - 1) - y
        if self.rotation_degrees == 270:
            return y, (width - 1) - x
        return x, y


IDENTITY_ORIENTATION = Orientation()


@dataclass(frozen=True, slots=True)
class CropRegion:
    """Region of interest as fractions of the (already oriented) image, so it survives
    resolution changes."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        eps = 1e-9
        if not (0.0 <= self.x < 1.0 and 0.0 <= self.y < 1.0):
            raise ValueError("crop x and y must be in [0, 1)")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("crop width and height must be positive")
        if self.x + self.width > 1.0 + eps or self.y + self.height > 1.0 + eps:
            raise ValueError("crop region must lie inside the image")

    def to_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """(x0, y0, width, height) in pixels, at least 1×1 and clipped to the image."""
        x0 = min(int(round(self.x * image_width)), image_width - 1)
        y0 = min(int(round(self.y * image_height)), image_height - 1)
        w = max(1, min(int(round(self.width * image_width)), image_width - x0))
        h = max(1, min(int(round(self.height * image_height)), image_height - y0))
        return x0, y0, w, h


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics, valid at the resolution they were measured at."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> CameraIntrinsics | None:
        """Intrinsics for a different capture resolution, or ``None`` if the aspect ratio
        changed (likely a sensor crop mode, which a simple rescale cannot describe)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) > 0.01 * max(sx, sy):
            return None
        return CameraIntrinsics(
            self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height
        )

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


@dataclass(frozen=True, slots=True, eq=False)
class CameraFrame:
    """One image from one camera, plus everything known (and unknown) about it."""

    image: np.ndarray
    camera_id: str
    role: CameraRole
    sequence: int
    host_receive_ns: int
    capture_ns: int | None
    timestamp_source: TimestampSource
    timestamp_uncertainty_ns: int | None
    pixel_format: PixelFormat
    source_kind: SourceKind
    orientation: Orientation = IDENTITY_ORIENTATION
    crop: CropRegion | None = None
    fps: float | None = None
    fps_measured: bool = False
    intrinsics: CameraIntrinsics | None = None
    distortion: tuple[float, ...] | None = None
    status: FrameStatus = FrameStatus.OK
    warnings: tuple[str, ...] = ()
    # Only synthetic sources set this, so generated truth can't be mistaken for a measurement.
    ground_truth: Mapping[str, float] | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def timestamp_ns(self) -> int:
        """Best available time for this observation: capture time if known, else receive time."""
        return self.capture_ns if self.capture_ns is not None else self.host_receive_ns

    @property
    def is_normalized(self) -> bool:
        return self.orientation.is_identity and self.crop is None and self.distortion is None

    def replace(self, **changes: object) -> CameraFrame:
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True, slots=True)
class SettingResult:
    """Outcome of applying one requested setting to a device."""

    name: str
    requested: object
    actual: object | None
    support: ControlSupport
    message: str = ""


@dataclass(frozen=True, slots=True)
class CameraCapabilities:
    """What a source reported about itself. ``None`` means the hardware didn't say."""

    source_type: str
    source_kind: SourceKind
    backend: str | None = None
    device_name: str | None = None
    resolution: tuple[int, int] | None = None
    supported_resolutions: tuple[tuple[int, int], ...] | None = None
    reported_fps: float | None = None
    pixel_formats: tuple[PixelFormat, ...] = ()
    controls: Mapping[str, ControlSupport] = field(default_factory=dict)
    provides_device_timestamps: bool = False
    supports_reconnect: bool = True
    intrinsics: CameraIntrinsics | None = None
    distortion: tuple[float, ...] | None = None
    setting_results: tuple[SettingResult, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CameraStatus:
    camera_id: str
    role: CameraRole
    source_type: str
    source_kind: SourceKind
    state: CameraState
    frames_read: int
    last_frame_host_ns: int | None
    last_error: str | None


class CameraError(Exception):
    """Base class for camera failures."""


class CameraConnectionError(CameraError):
    """The device or stream could not be opened. Retryable."""


class CameraDisconnectedError(CameraError):
    """An open device stopped delivering frames (unplugged, stream dropped, read timeout).
    Retryable."""


class CameraConfigError(CameraError):
    """The configuration can't work (bad options, missing file). Not retryable."""


class EndOfStream(CameraError):
    """A finite source (a recording without looping) has no more frames."""

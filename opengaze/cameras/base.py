"""The ``CameraSource`` interface every camera adapter implements.

Gaze estimation, calibration, the UI and cursor control depend only on this interface and on
``CameraFrame``. Adding hardware means adding an adapter, never touching those layers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from opengaze.cameras.timing import RateMeter
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConfigError,
    CameraConnectionError,
    CameraDisconnectedError,
    CameraError,
    CameraFrame,
    CameraRole,
    CameraState,
    CameraStatus,
    FrameStatus,
    PixelFormat,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CameraConfig

log = logging.getLogger(__name__)


class NoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """A problem an adapter found in its configuration, without touching the device."""

    severity: str  # "error" | "warning" | "info"
    message: str


class CameraSource(ABC):
    """One camera, whatever its hardware or transport.

    Lifecycle: ``connect()`` → ``read_frame()`` repeatedly → ``disconnect()``. After a
    ``CameraConnectionError`` or ``CameraDisconnectedError`` the caller may ``connect()`` again.
    """

    source_type: ClassVar[str]
    source_kind: ClassVar[SourceKind]
    Options: ClassVar[type[BaseModel]] = NoOptions
    # Option names that cannot change how eye features map to the screen (timing, fault
    # injection). Everything else in ``options`` is part of the calibration fingerprint.
    non_geometric_options: ClassVar[frozenset[str]] = frozenset()
    # True for adapters that open cameras through AVFoundation, which needs macOS camera
    # permission. Network streams don't.
    needs_camera_permission: ClassVar[bool] = False

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.options: Any = self.parse_options(config)
        self._state = CameraState.IDLE
        self._frames_read = 0
        self._next_sequence = 0
        self._last_frame_host_ns: int | None = None
        self._last_error: str | None = None
        self._rate = RateMeter()
        self._orientation = config.orientation()
        self._crop = config.crop.to_region() if config.crop else None
        optics = config.optics
        self._intrinsics = optics.intrinsics.to_intrinsics() if optics.intrinsics else None
        self._distortion = optics.distortion

    # -- configuration -------------------------------------------------------------------

    @classmethod
    def parse_options(cls, config: CameraConfig) -> Any:
        try:
            return cls.Options.model_validate(config.options)
        except ValidationError as exc:
            details = "; ".join(
                f"options.{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            raise CameraConfigError(
                f"{config.camera_id}: invalid options for source_type "
                f"'{cls.source_type}': {details}"
            ) from exc

    @classmethod
    def check_config(cls, config: CameraConfig) -> list[ConfigIssue]:
        """Static checks used by the profile validator. Must not open the device."""
        try:
            cls.parse_options(config)
        except CameraConfigError as exc:
            return [ConfigIssue("error", str(exc))]
        return []

    @classmethod
    def device_key(cls, config: CameraConfig) -> tuple[Any, ...] | None:
        """Identity of the physical device, so a profile can't assign one device twice."""
        return None

    # -- the interface -------------------------------------------------------------------

    @property
    def camera_id(self) -> str:
        return self.config.camera_id

    @property
    def role(self) -> CameraRole:
        return self.config.role

    @abstractmethod
    async def connect(self) -> None:
        """Open the device and apply requested settings."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Release the device. Safe to call repeatedly or after a failure."""

    @abstractmethod
    async def read_frame(self) -> CameraFrame:
        """Wait for the next frame.

        Raises ``CameraDisconnectedError`` when the stream is lost or stalls past the read
        timeout, and ``EndOfStream`` when a finite source is exhausted.
        """

    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        """What the source reported about itself, including per-setting results."""

    def get_status(self) -> CameraStatus:
        return CameraStatus(
            camera_id=self.camera_id,
            role=self.role,
            source_type=self.source_type,
            source_kind=self.source_kind,
            state=self._state,
            frames_read=self._frames_read,
            last_frame_host_ns=self._last_frame_host_ns,
            last_error=self._last_error,
        )

    async def frames(self) -> AsyncIterator[CameraFrame]:
        while True:
            yield await self.read_frame()

    async def __aenter__(self) -> CameraSource:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.disconnect()

    # -- helpers for adapters ------------------------------------------------------------

    def _set_state(self, state: CameraState, error: str | None = None) -> None:
        self._state = state
        if error is not None:
            self._last_error = error

    def _reset_timing(self) -> None:
        self._rate.reset()

    def _make_frame(
        self,
        image: np.ndarray,
        *,
        host_receive_ns: int,
        capture_ns: int | None,
        timestamp_source: TimestampSource,
        timestamp_uncertainty_ns: int | None,
        sequence: int | None = None,
        status: FrameStatus = FrameStatus.OK,
        warnings: tuple[str, ...] = (),
        ground_truth: Mapping[str, float] | None = None,
    ) -> CameraFrame:
        if sequence is None:
            sequence = self._next_sequence
        self._next_sequence = sequence + 1
        timestamp = capture_ns if capture_ns is not None else host_receive_ns
        self._rate.update(timestamp)
        self._frames_read += 1
        self._last_frame_host_ns = host_receive_ns
        return CameraFrame(
            image=image,
            camera_id=self.camera_id,
            role=self.role,
            sequence=sequence,
            host_receive_ns=host_receive_ns,
            capture_ns=capture_ns,
            timestamp_source=timestamp_source,
            timestamp_uncertainty_ns=timestamp_uncertainty_ns,
            pixel_format=pixel_format_of(image),
            source_kind=self.source_kind,
            orientation=self._orientation,
            crop=self._crop,
            fps=self._rate.fps,
            fps_measured=self._rate.fps is not None,
            intrinsics=self._intrinsics,
            distortion=self._distortion,
            status=status,
            warnings=warnings,
            ground_truth=ground_truth,
        )


def pixel_format_of(image: np.ndarray) -> PixelFormat:
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return PixelFormat.GRAY8
    if image.ndim == 3 and image.shape[2] == 3:
        return PixelFormat.BGR8
    raise CameraError(f"unsupported image shape {image.shape}")


# -- blocking drivers ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawCapture:
    """What a blocking driver read returns, before it becomes a CameraFrame."""

    image: np.ndarray
    host_receive_ns: int
    capture_ns: int | None = None
    timestamp_source: TimestampSource = TimestampSource.HOST_RECEIVE
    timestamp_uncertainty_ns: int | None = None
    status: FrameStatus = FrameStatus.OK
    warnings: tuple[str, ...] = ()
    sequence: int | None = None  # the device's own frame counter, when it sends one


class _BlockingWorker:
    """A single daemon thread that runs one camera's blocking driver calls in order.

    Daemon, so a driver call that never returns (an unplugged camera, a dead stream) can't
    block process exit. One per camera, so a stuck camera can't stall the others.
    """

    def __init__(self, name: str) -> None:
        self._jobs: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[..., Any], *args: Any) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._jobs.put((loop, future, fn, args))
        return future

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn`` eventually, ignoring its result (used for cleanup on a stuck worker)."""
        self._jobs.put((None, None, fn, args))

    def close(self) -> None:
        self._jobs.put(None)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            loop, future, fn, args = job
            try:
                result, error = fn(*args), None
            except Exception as exc:  # delivered to the awaiting coroutine
                result, error = None, exc
            if future is None:
                if error is not None:
                    log.debug("background camera call failed: %s", error)
                continue
            with contextlib.suppress(RuntimeError):  # loop already closed; nobody is waiting
                loop.call_soon_threadsafe(_resolve, future, result, error)


def _resolve(future: asyncio.Future[Any], result: Any, error: Exception | None) -> None:
    if future.done():  # cancelled by a timeout while the driver call was running
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(result)


class BlockingCaptureSource(CameraSource):
    """Base for adapters whose driver API blocks (OpenCV ``VideoCapture`` and friends).

    Subclasses implement ``_open_blocking``, ``_read_blocking`` and ``_close_blocking``; this
    class runs them on the camera's own thread and enforces connect/read timeouts.
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._worker: _BlockingWorker | None = None
        self._handle: Any = None
        self._capabilities = CameraCapabilities(
            source_type=self.source_type, source_kind=self.source_kind
        )

    @abstractmethod
    def _open_blocking(self) -> Any:
        """Open the device; return a handle. Raise CameraConnectionError/CameraConfigError."""

    @abstractmethod
    def _read_blocking(self, handle: Any) -> RawCapture:
        """Read one frame. Raise CameraDisconnectedError (or EndOfStream) on failure."""

    @abstractmethod
    def _close_blocking(self, handle: Any) -> None:
        """Release the handle."""

    def get_capabilities(self) -> CameraCapabilities:
        return self._capabilities

    async def _call(self, timeout: float, fn: Callable[..., Any], *args: Any) -> Any:
        if self._worker is None:
            self._worker = _BlockingWorker(f"camera-{self.camera_id}")
        return await asyncio.wait_for(self._worker.submit(fn, *args), timeout)

    def _abandon_worker(self) -> None:
        """Give up on a worker stuck in a driver call; release its handle if it ever returns."""
        if self._worker is None:
            return
        if self._handle is not None:
            self._worker.post(self._close_blocking, self._handle)
        self._worker.close()
        self._worker = None
        self._handle = None

    async def connect(self) -> None:
        if self._handle is not None:
            return
        self._set_state(CameraState.CONNECTING)
        timeout = self.config.connect_timeout_s
        try:
            self._handle = await self._call(timeout, self._open_blocking)
        except TimeoutError:
            self._abandon_worker()
            self._fail_connect(CameraConnectionError(f"timed out after {timeout:.1f}s opening"))
        except CameraError as exc:
            self._fail_connect(exc)
        except Exception as exc:
            self._fail_connect(CameraConnectionError(f"{type(exc).__name__}: {exc}"), exc)
        self._reset_timing()
        self._set_state(CameraState.STREAMING)

    def _fail_connect(self, error: CameraError, cause: Exception | None = None) -> None:
        state = CameraState.FAILED if isinstance(error, CameraConfigError) else CameraState.IDLE
        self._set_state(state, str(error))
        raise error from cause

    async def _read_raw(self) -> RawCapture:
        if self._handle is None:
            raise CameraDisconnectedError("not connected")
        timeout = self.config.read_timeout_s
        try:
            return await self._call(timeout, self._read_blocking, self._handle)
        except TimeoutError:
            self._abandon_worker()
            self._set_state(CameraState.IDLE, f"no frame within {timeout:.1f}s")
            raise CameraDisconnectedError(f"no frame within {timeout:.1f}s") from None
        except CameraError as exc:
            self._set_state(CameraState.IDLE, str(exc))
            raise
        except Exception as exc:
            self._set_state(CameraState.IDLE, str(exc))
            raise CameraDisconnectedError(f"{type(exc).__name__}: {exc}") from exc

    async def read_frame(self) -> CameraFrame:
        raw = await self._read_raw()
        return self._frame_from_raw(raw)

    def _frame_from_raw(self, raw: RawCapture, **overrides: Any) -> CameraFrame:
        fields: dict[str, Any] = {
            "host_receive_ns": raw.host_receive_ns,
            "capture_ns": raw.capture_ns,
            "timestamp_source": raw.timestamp_source,
            "timestamp_uncertainty_ns": raw.timestamp_uncertainty_ns,
            "status": raw.status,
            "warnings": raw.warnings,
            "sequence": raw.sequence,
        }
        fields.update(overrides)
        return self._make_frame(raw.image, **fields)

    async def disconnect(self) -> None:
        if self._worker is not None:
            if self._handle is not None:
                try:
                    await self._call(2.0, self._close_blocking, self._handle)
                except Exception as exc:  # a failed release must not prevent shutdown
                    log.warning("%s: error releasing camera: %s", self.camera_id, exc)
                    self._abandon_worker()
            if self._worker is not None:
                self._worker.close()
                self._worker = None
        self._handle = None
        if self._state is not CameraState.FAILED:
            self._set_state(CameraState.STOPPED)

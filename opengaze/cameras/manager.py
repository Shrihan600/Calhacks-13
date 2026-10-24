"""Runs every configured camera concurrently and keeps the newest frame from each.

Design rules:

- Cameras run independently at their own rates; nothing waits for a slow camera.
- Each camera has a single latest-frame slot. A new frame replaces the old one, so a slow
  consumer gets fresher frames, never a growing backlog. Overwritten frames are counted.
- Lost cameras reconnect with exponential backoff. Configuration errors don't retry.
- ``latest()`` and ``health()`` are thread-safe, so UI threads can poll them directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from opengaze.cameras.base import CameraSource
from opengaze.cameras.registry import create_source
from opengaze.cameras.timing import RateMeter
from opengaze.cameras.types import (
    CameraCapabilities,
    CameraConfigError,
    CameraError,
    CameraFrame,
    CameraRole,
    CameraState,
    EndOfStream,
    FrameStatus,
    SourceKind,
    TimestampSource,
)
from opengaze.config.schema import CaptureSettings, HardwareProfile

log = logging.getLogger(__name__)

_TRANSPORT_DELAY_SOURCES = (TimestampSource.DEVICE, TimestampSource.SYNTHETIC)


@dataclass(frozen=True, slots=True)
class CameraHealth:
    camera_id: str
    role: CameraRole
    source_type: str
    source_kind: SourceKind
    state: CameraState
    frames_received: int
    frames_skipped_by_source: int  # gaps in sequence numbers (lost before reaching the host)
    frames_overwritten: int  # replaced in the slot before any consumer took them
    partial_frames: int
    measured_fps: float | None
    interval_jitter_ms: float | None
    last_frame_age_ms: float | None
    transport_delay_ms: float | None  # capture → host receive, when capture time is known
    reconnects: int
    last_error: str | None
    capabilities: CameraCapabilities | None


class _Slot:
    def __init__(self, source: CameraSource) -> None:
        self.source = source
        self.state = CameraState.IDLE
        self.latest: CameraFrame | None = None
        self.consumed = True
        self.last_sequence: int | None = None
        self.received = 0
        self.skipped = 0
        self.overwritten = 0
        self.partial = 0
        self.reconnects = 0
        self.last_error: str | None = None
        self.capabilities: CameraCapabilities | None = None
        self.rate = RateMeter()
        self.transport_delay_ms: float | None = None
        self.streaming_since_ns = 0


class CameraManager:
    def __init__(
        self,
        sources: Iterable[CameraSource],
        settings: CaptureSettings | None = None,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.settings = settings or CaptureSettings()
        self._clock = clock
        self._slots: dict[str, _Slot] = {}
        for source in sources:
            if source.camera_id in self._slots:
                raise ValueError(f"duplicate camera id '{source.camera_id}'")
            self._slots[source.camera_id] = _Slot(source)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._stop_requested = False

    @classmethod
    def from_profile(
        cls, profile: HardwareProfile, settings: CaptureSettings | None = None
    ) -> CameraManager:
        return cls((create_source(cam) for cam in profile.enabled_cameras.values()), settings)

    @property
    def camera_ids(self) -> list[str]:
        return list(self._slots)

    def source(self, camera_id: str) -> CameraSource:
        return self._slots[camera_id].source

    # -- running ---------------------------------------------------------------------------

    async def run(self) -> None:
        """Capture until ``stop()`` / ``request_stop()``; then disconnect everything."""
        self._stop_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        if self._stop_requested:
            self._stop_event.set()
        tasks = [
            asyncio.create_task(self._camera_loop(slot), name=f"camera-{cid}")
            for cid, slot in self._slots.items()
        ]
        try:
            await self._stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(
                *(self._safe_disconnect(slot) for slot in self._slots.values()),
                return_exceptions=True,
            )
            for slot in self._slots.values():
                if slot.state is not CameraState.FAILED:
                    slot.state = CameraState.STOPPED

    async def stop(self) -> None:
        self.request_stop()

    def request_stop(self) -> None:
        """Thread-safe."""
        self._stop_requested = True
        loop, event = self._loop, self._stop_event
        if loop is not None and event is not None:
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(event.set)

    def _stopping(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Sleep, waking early on stop. Returns True if stopping."""
        assert self._stop_event is not None
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), seconds)
        return self._stopping()

    async def _camera_loop(self, slot: _Slot) -> None:
        cid = slot.source.camera_id
        delay = self.settings.reconnect_initial_delay_s
        connected_before = False
        while not self._stopping():
            slot.state = CameraState.RECONNECTING if connected_before else CameraState.CONNECTING
            try:
                await slot.source.connect()
            except CameraConfigError as exc:
                slot.state, slot.last_error = CameraState.FAILED, str(exc)
                log.error("%s: %s (not retrying)", cid, exc)
                return
            except CameraError as exc:
                slot.last_error = str(exc)
                log.warning("%s: connect failed: %s; retrying in %.1fs", cid, exc, delay)
                if await self._sleep_or_stop(delay):
                    return
                delay = min(delay * 2, self.settings.reconnect_max_delay_s)
                continue
            except Exception as exc:  # adapter bug: log it fully, keep retrying
                slot.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("%s: unexpected error while connecting", cid)
                if await self._sleep_or_stop(delay):
                    return
                delay = min(delay * 2, self.settings.reconnect_max_delay_s)
                continue

            with self._lock:
                if connected_before:
                    slot.reconnects += 1
                slot.capabilities = slot.source.get_capabilities()
                slot.rate.reset()
                slot.last_sequence = None
                slot.streaming_since_ns = self._clock()
                slot.state = CameraState.STREAMING
            connected_before = True
            delay = self.settings.reconnect_initial_delay_s
            log.info("%s: streaming (%s)", cid, slot.source.source_type)

            try:
                while not self._stopping():
                    self._on_frame(slot, await slot.source.read_frame())
            except EndOfStream as exc:
                log.info("%s: %s", cid, exc)
                await self._safe_disconnect(slot)
                slot.state, slot.last_error = CameraState.STOPPED, str(exc)
                return
            except CameraError as exc:
                slot.last_error = str(exc)
                log.warning("%s: lost stream: %s", cid, exc)
            except Exception as exc:
                slot.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("%s: unexpected error while reading", cid)
            await self._safe_disconnect(slot)
            slot.state = CameraState.RECONNECTING
            if await self._sleep_or_stop(delay):
                return

    async def _safe_disconnect(self, slot: _Slot) -> None:
        try:
            await asyncio.wait_for(slot.source.disconnect(), 5.0)
        except Exception as exc:
            log.warning("%s: disconnect failed: %s", slot.source.camera_id, exc)

    def _on_frame(self, slot: _Slot, frame: CameraFrame) -> None:
        with self._lock:
            if slot.latest is not None and not slot.consumed:
                slot.overwritten += 1
            if slot.last_sequence is not None and frame.sequence > slot.last_sequence + 1:
                slot.skipped += frame.sequence - slot.last_sequence - 1
            slot.last_sequence = frame.sequence
            slot.latest, slot.consumed = frame, False
            slot.received += 1
            if frame.status is FrameStatus.PARTIAL:
                slot.partial += 1
            slot.rate.update(frame.host_receive_ns)
            if frame.capture_ns is not None and frame.timestamp_source in _TRANSPORT_DELAY_SOURCES:
                delay_ms = (frame.host_receive_ns - frame.capture_ns) / 1e6
                previous = slot.transport_delay_ms
                slot.transport_delay_ms = (
                    delay_ms if previous is None else previous + 0.1 * (delay_ms - previous)
                )

    # -- consumers (thread-safe) ---------------------------------------------------------

    def latest(self, camera_id: str, *, consume: bool = True) -> CameraFrame | None:
        with self._lock:
            slot = self._slots[camera_id]
            if consume:
                slot.consumed = True
            return slot.latest

    def health(self) -> dict[str, CameraHealth]:
        now = self._clock()
        stale_ns = self.settings.stale_frame_timeout_s * 1e9
        out: dict[str, CameraHealth] = {}
        with self._lock:
            for cid, slot in self._slots.items():
                age_ns = None if slot.latest is None else now - slot.latest.host_receive_ns
                state = slot.state
                if state is CameraState.STREAMING:
                    # Stalled = no frame for the timeout, counted from the newer of the last
                    # frame and the moment this connection started streaming.
                    last_activity = slot.streaming_since_ns
                    if slot.latest is not None:
                        last_activity = max(last_activity, slot.latest.host_receive_ns)
                    if now - last_activity > stale_ns:
                        state = CameraState.STALLED
                source = slot.source
                out[cid] = CameraHealth(
                    camera_id=cid,
                    role=source.role,
                    source_type=source.source_type,
                    source_kind=source.source_kind,
                    state=state,
                    frames_received=slot.received,
                    frames_skipped_by_source=slot.skipped,
                    frames_overwritten=slot.overwritten,
                    partial_frames=slot.partial,
                    measured_fps=slot.rate.fps,
                    interval_jitter_ms=slot.rate.jitter_ms,
                    last_frame_age_ms=None if age_ns is None else age_ns / 1e6,
                    transport_delay_ms=slot.transport_delay_ms,
                    reconnects=slot.reconnects,
                    last_error=slot.last_error,
                    capabilities=slot.capabilities,
                )
        return out

"""Runs a CameraManager on a background asyncio loop, so UI threads never block on cameras."""

from __future__ import annotations

import asyncio
import logging
import threading

from opengaze.cameras.manager import CameraManager

log = logging.getLogger(__name__)


class CaptureRuntime:
    def __init__(self, manager: CameraManager) -> None:
        self.manager = manager
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("capture runtime already started")
        self._thread = threading.Thread(target=self._run, name="opengaze-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            asyncio.run(self.manager.run())
        except Exception:
            log.exception("capture loop crashed")

    def stop(self, timeout: float = 5.0) -> None:
        self.manager.request_stop()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("capture thread did not stop within %.1fs", timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> CaptureRuntime:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

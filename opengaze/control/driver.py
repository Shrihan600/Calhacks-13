"""GazeDriver: the main pipeline from FeatureVector → cursor move.

Pipeline per frame
------------------
FeatureVector
  → GazeModel.predict()    → GazeSample (screen_x/y_norm, confidence)
  → EmergencyStop          → drop if paused
  → ConfidenceGate         → drop if confidence too low (blink, face lost)
  → JumpDetector           → drop if position jumped impossibly far
  → OneEuroFilter (x, y)   → smoothed normalised position
  → norm_to_px()           → screen pixel coordinates
  → DwellDetector          → fire click if dwell complete
  → QuartzCursorDriver     → move (and optionally click) the real cursor

All state lives here.  The caller just feeds FeatureVectors at ~30 fps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from opengaze.calibration.model import GazeModel, GazeSample
from opengaze.control.dwell import DwellDetector, DwellResult
from opengaze.control.filter import OneEuroFilter
from opengaze.control.safety import ConfidenceGate, EmergencyStop, JumpDetector
from opengaze.vision.observations import FeatureVector

log = logging.getLogger(__name__)


@dataclass
class DriveResult:
    """What the driver did on this frame."""
    sample: GazeSample | None      # None if the frame was dropped
    screen_x_px: int | None
    screen_y_px: int | None
    dwell: DwellResult | None
    drop_reason: str | None        # "paused" | "confidence" | "jump" | None


class GazeDriver:
    """Ties gaze model, filters, safety, dwell, and cursor together.

    Usage::

        driver = GazeDriver(model, cursor)
        # call on every frame at ~30 fps:
        result = driver.push(feature_vector)
    """

    def __init__(
        self,
        model: GazeModel,
        cursor: object,   # QuartzCursorDriver or any duck-typed mock
        *,
        min_confidence: float = 0.3,
        max_jump_norm: float = 0.15,
        filter_min_cutoff: float = 0.5,
        filter_beta: float = 0.05,
        dwell_ms: float = 600.0,
        dwell_radius_px: float = 40.0,
        dwell_cooldown_ms: float = 800.0,
        dwell_enabled: bool = True,
    ) -> None:
        self.model = model
        self.cursor = cursor

        self._confidence_gate = ConfidenceGate(min_confidence)
        self._jump = JumpDetector(max_jump_norm)
        self._filter_x = OneEuroFilter(filter_min_cutoff, filter_beta)
        self._filter_y = OneEuroFilter(filter_min_cutoff, filter_beta)
        self._dwell = DwellDetector(dwell_ms, dwell_radius_px, dwell_cooldown_ms, dwell_enabled)
        self._estop = EmergencyStop()

        self.frames_processed: int = 0
        self.frames_dropped: int = 0
        self.clicks_fired: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def push(self, fv: FeatureVector) -> DriveResult:
        self.frames_processed += 1
        t = time.monotonic()

        if self._estop.active:
            self.frames_dropped += 1
            return DriveResult(None, None, None, None, "paused")

        try:
            sample = self.model.predict(fv)
        except Exception as exc:
            log.debug("model predict failed: %s", exc)
            self.frames_dropped += 1
            return DriveResult(None, None, None, None, "model_error")

        if not self._confidence_gate.allow(sample.confidence):
            self.frames_dropped += 1
            return DriveResult(sample, None, None, None, "confidence")

        if not self._jump.allow(sample.screen_x_norm, sample.screen_y_norm):
            self.frames_dropped += 1
            return DriveResult(sample, None, None, None, "jump")

        # Smooth
        sx = self._filter_x(sample.screen_x_norm, t)
        sy = self._filter_y(sample.screen_y_norm, t)

        # To screen pixels
        px, py = self.cursor.norm_to_px(sx, sy)

        # Dwell
        dwell_result = self._dwell.update(px, py, t)

        # Move cursor
        self.cursor.move(px, py)

        # Click
        if dwell_result.clicked:
            self.cursor.click(px, py)
            self.clicks_fired += 1
            log.info("dwell click at (%d, %d)", px, py)

        return DriveResult(sample, px, py, dwell_result, None)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._estop.trigger()
        self._filter_x.reset()
        self._filter_y.reset()
        self._dwell.reset()
        self._jump.reset()
        log.info("gaze driver paused")

    def resume(self) -> None:
        self._estop.reset()
        log.info("gaze driver resumed")

    def toggle_pause(self) -> bool:
        """Toggle pause. Returns True if now paused."""
        if self._estop.active:
            self.resume()
            return False
        else:
            self.pause()
            return True

    @property
    def paused(self) -> bool:
        return self._estop.active

    @property
    def dwell_progress(self) -> float:
        return self._dwell.progress

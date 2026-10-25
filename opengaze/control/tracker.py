"""Headless gaze pipeline: camera frame → filtered screen gaze + blink state.

Same stages as the gaze test window (detector → features → model → median →
One Euro → edge remap, blink freeze), packaged so modes without a fullscreen
window, like system-wide cursor control, can run it from any timer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from opengaze.calibration.session import GRID_HI, GRID_LO
from opengaze.control.blink import BlinkClickDetector, BlinkResult
from opengaze.control.filter import OneEuroFilter, TemporalMedianFilter
from opengaze.control.safety import ConfidenceGate
from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.preprocessing import FrameNormalizer


def expand(v: float) -> float:
    """Remap calibration output range [GRID_LO, GRID_HI] → [0, 1]."""
    return max(0.0, min(1.0, (v - GRID_LO) / (GRID_HI - GRID_LO)))


@dataclass
class GazeUpdate:
    x: float           # normalised screen position (frozen during blinks)
    y: float
    fresh: bool        # a new model sample landed this step
    face: bool         # a face was detected
    blink: BlinkResult


class GazeTracker:
    def __init__(self, manager, head_ids: set[str], detector, model) -> None:
        self._manager = manager
        self._head_ids = head_ids
        self._detector = detector
        self._model = model
        self._normalizer = FrameNormalizer()
        self._mx = TemporalMedianFilter(5)
        self._my = TemporalMedianFilter(5)
        self._fx = OneEuroFilter(min_cutoff=0.3, beta=0.1)
        self._fy = OneEuroFilter(min_cutoff=0.3, beta=0.1)
        self._conf = ConfidenceGate(min_confidence=0.25)
        self._blink = BlinkClickDetector(long_blink_ms=400, short_blink_ms=180)
        self.x = 0.5
        self.y = 0.5

    def reset(self) -> None:
        for f in (self._mx, self._my, self._fx, self._fy):
            f.reset()

    def step(self) -> GazeUpdate | None:
        """Process the newest head-camera frame. None if no frame or no detector."""
        if self._detector is None:
            return None
        for cid in self._head_ids:
            frame = self._manager.latest(cid)
            if frame is None:
                continue
            try:
                head, eyes = self._detector.detect(self._normalizer(frame))
            except Exception:
                continue
            left = next((e for e in eyes if e.eye == "left"), None)
            right = next((e for e in eyes if e.eye == "right"), None)
            blink = self._blink.update(bool(left and left.blink), bool(right and right.blink))
            fresh = False
            # Freeze gaze while eyes are closed so a blink click lands where the
            # user was looking, not at a noisy mid-blink estimate.
            if self._model is not None and not blink.any_blink:
                fv = build_mode_a(head, left, right)
                if self._conf.allow(fv.valid.mean()):
                    try:
                        s = self._model.predict(fv)
                        t = time.monotonic()
                        self.x = expand(self._fx(self._mx(s.screen_x_norm), t))
                        self.y = expand(self._fy(self._my(s.screen_y_norm), t))
                        fresh = True
                    except Exception:
                        pass
            return GazeUpdate(self.x, self.y, fresh, head is not None or bool(eyes), blink)
        return None


class UserOverride:
    """Yields the cursor to a physical mouse/trackpad.

    After each gaze move we remember where we put the cursor. If it's found
    somewhere else, a person moved it, so gaze stays hands-off for ``hold_s``.
    """

    def __init__(self, tolerance_px: float = 24.0, hold_s: float = 2.0) -> None:
        self.tolerance_px = tolerance_px
        self.hold_s = hold_s
        self._expected: tuple[float, float] | None = None
        self._hold_until = 0.0

    def moved_to(self, x: float, y: float) -> None:
        self._expected = (x, y)

    def gaze_allowed(self, actual: tuple[float, float], now: float) -> bool:
        if self._expected is not None and math.dist(actual, self._expected) > self.tolerance_px:
            self._hold_until = now + self.hold_s
            self._expected = None
        return now >= self._hold_until

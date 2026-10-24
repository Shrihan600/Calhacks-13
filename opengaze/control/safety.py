"""Safety gates between the gaze model and the cursor.

ConfidenceGate  — drops samples where the model isn't sure (blink, face lost).
JumpDetector    — drops samples that jump impossibly far in one frame (outlier).
EmergencyStop   — hard pause; call trigger() to freeze, reset() to resume.

All gates are stateless except JumpDetector (needs previous position) and
EmergencyStop (toggled by the user).
"""

from __future__ import annotations

import math


class ConfidenceGate:
    """Passes a GazeSample only if confidence >= threshold.

    Typical threshold: 0.3 — requires at least 30% of features to be valid.
    Raise it if the cursor is jumpy when partially occluded.
    """

    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence

    def allow(self, confidence: float) -> bool:
        return confidence >= self.min_confidence


class JumpDetector:
    """Drops frames where the predicted position moves too far in one step.

    Jumps happen when MediaPipe briefly loses and re-acquires the face, or
    when a stale frame produces an outlier prediction.  The threshold is in
    normalised [0, 1]² screen coordinates; 0.15 ≈ 150–290 px depending on
    screen size.
    """

    def __init__(self, max_jump_norm: float = 0.15) -> None:
        self.max_jump_norm = max_jump_norm
        self._last_x: float | None = None
        self._last_y: float | None = None
        self.jumps_detected: int = 0

    def allow(self, x_norm: float, y_norm: float) -> bool:
        if self._last_x is None:
            self._last_x = x_norm
            self._last_y = y_norm
            return True
        dist = math.sqrt((x_norm - self._last_x) ** 2 + (y_norm - (self._last_y or 0)) ** 2)
        if dist > self.max_jump_norm:
            self.jumps_detected += 1
            return False
        self._last_x = x_norm
        self._last_y = y_norm
        return True

    def reset(self) -> None:
        self._last_x = None
        self._last_y = None


class EmergencyStop:
    """Hard pause — freezes cursor movement without killing the process.

    Bind trigger() to a hotkey or foot-switch.  The gaze loop continues
    running; frames are processed but cursor moves are suppressed.
    """

    def __init__(self) -> None:
        self._active = False
        self.trigger_count: int = 0

    def trigger(self) -> None:
        self._active = True
        self.trigger_count += 1

    def reset(self) -> None:
        self._active = False

    def toggle(self) -> bool:
        self._active = not self._active
        if self._active:
            self.trigger_count += 1
        return self._active

    @property
    def active(self) -> bool:
        return self._active

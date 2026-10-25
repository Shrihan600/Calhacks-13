"""Where the simulated user is looking.

Synthetic eye cameras with ``motion: puppet`` render the eye looking at ``target()`` instead
of following their own trajectory. Calibration windows call ``look_at`` for every dot, so a
simulated rig can be calibrated through the real UI; the headless simulator scripts it.
With no target set, puppet eyes fall back to saccades.

The target is one process-wide tuple: writes are atomic, and the capture thread only reads it.
"""

from __future__ import annotations

_target: tuple[float, float] | None = None


def look_at(x: float, y: float) -> None:
    global _target
    _target = (float(x), float(y))


def release() -> None:
    global _target
    _target = None


def target() -> tuple[float, float] | None:
    return _target

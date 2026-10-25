"""Webcam eye closure from MediaPipe blink scores: winks survive the open eye squinting."""

from __future__ import annotations

import pytest

from opengaze.vision.mediapipe_detector import closed_from_scores


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0.05, 0.08, (False, False)),  # both open
        (0.92, 0.88, (True, True)),  # blink
        (0.90, 0.10, (True, False)),  # clean wink
        (0.92, 0.55, (True, False)),  # wink, open eye squinting past the threshold
        (0.50, 0.95, (False, True)),  # same, other eye
        (0.60, 0.45, (True, True)),  # half-closed both: still a blink, not a wink
    ],
)
def test_closed_from_scores(left, right, expected):
    assert closed_from_scores(left, right) == expected

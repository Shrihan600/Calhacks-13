"""Webcam eye closure from MediaPipe blink scores: winks survive the open eye squinting."""

from __future__ import annotations

import pytest

from opengaze.vision.mediapipe_detector import EyeClosure, closed_from_scores


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


def _closures(scores, closure=None):
    closure = closure or EyeClosure()
    return [closure.update(v, v)[0] for v in scores]


def test_wavering_score_is_one_blink_not_two():
    # The score dips just under the closing threshold mid-blink: still one closure.
    states = _closures([0.05] * 30 + [0.6, 0.45, 0.33, 0.5, 0.7, 0.2, 0.05, 0.05])
    closings = sum(1 for a, b in zip(states, states[1:]) if b and not a)
    assert closings == 1


def test_high_resting_score_is_not_a_closed_eye():
    # Someone whose open eyes score 0.45 (hooded eyes, looking down): open once adapted,
    # and a real blink still closes.
    closure = EyeClosure()
    states = _closures([0.45] * 600, closure)
    assert not any(states[-300:])
    assert closure.update(0.95, 0.95) == (True, True)


def test_wink_with_adapted_baseline():
    closure = EyeClosure()
    for _ in range(300):
        closure.update(0.3, 0.3)
    assert closure.update(0.92, 0.55) == (True, False)

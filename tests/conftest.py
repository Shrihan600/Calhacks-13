from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from opengaze.cameras.simulated import SyntheticCameraSource
from tests.helpers import camera_config


@pytest.fixture
def synthetic() -> Callable[..., SyntheticCameraSource]:
    """Build a synthetic source; by default unpaced (virtual clock) and noise-free."""

    def make(
        camera_id: str = "left_eye", *, options: dict[str, Any] | None = None, **fields: Any
    ) -> SyntheticCameraSource:
        opts = {"realtime": False, "noise_std": 0.0, **(options or {})}
        return SyntheticCameraSource(camera_config(camera_id, options=opts, **fields))

    return make

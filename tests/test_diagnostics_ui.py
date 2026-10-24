"""Diagnostics window renders synthetic cameras offscreen (no display or camera needed)."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from opengaze.cameras.manager import CameraManager  # noqa: E402
from opengaze.cameras.runtime import CaptureRuntime  # noqa: E402
from opengaze.config.loader import check_hardware_profile, load_app_config  # noqa: E402
from tests.helpers import HARDWARE_DIR, REPO_ROOT  # noqa: E402


def test_window_shows_previews_and_synthetic_labels() -> None:
    from opengaze.desktop.diagnostics import DiagnosticsWindow

    app = QApplication.instance() or QApplication([])
    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-dual-eye.yaml")
    runtime = CaptureRuntime(CameraManager.from_profile(report.profile, config.capture))
    runtime.start()
    try:
        window = DiagnosticsWindow(runtime, report, config)
        window.show()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            window.refresh()
            if all(
                tile.preview.pixmap() and not tile.preview.pixmap().isNull()
                for tile in window.tiles.values()
            ):
                break
            time.sleep(0.02)
        assert set(window.tiles) == {"left_eye", "right_eye", "head"}
        for tile in window.tiles.values():
            assert not tile.preview.pixmap().isNull()
            assert tile.kind.text() == "SYNTHETIC"
            assert tile.state.text() == "STREAMING"
        window.close()
    finally:
        runtime.stop()

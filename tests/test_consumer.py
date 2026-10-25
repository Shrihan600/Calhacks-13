"""EyeMouse consumer app: settings, login item, keyboard suggestions, setup flow, practice."""

from __future__ import annotations

import os
import plistlib

import pytest

from opengaze.config.loader import load_app_config
from opengaze.consumer.settings import AppSettings, LoginItem
from opengaze.gestures.types import Gesture
from tests.helpers import REPO_ROOT

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    from opengaze.consumer import theme

    theme.load_fonts()
    return app


def test_settings_round_trip_and_bad_file(tmp_path) -> None:
    path = tmp_path / "eyemouse.json"
    assert AppSettings.load(path) == AppSettings()
    s = AppSettings.load(path)
    s.setup_done, s.voice_typing, s.open_at_login = True, True, False
    s.save()
    again = AppSettings.load(path)
    assert (again.setup_done, again.voice_typing, again.open_at_login) == (True, True, False)

    path.write_text('{"setup_done": "yes", "voice_typing": true}')  # wrong type is ignored
    loaded = AppSettings.load(path)
    assert loaded.setup_done is False and loaded.voice_typing is True
    path.write_text("not json")
    assert AppSettings.load(path) == AppSettings()


def test_login_item_writes_and_removes_launch_agent(tmp_path) -> None:
    item = LoginItem(agents_dir=tmp_path, label="test.eyemouse")
    assert not item.enabled
    item.apply(True, ["/usr/bin/python3", "-m", "opengaze.cli", "app"], REPO_ROOT)
    plist = plistlib.loads(item.path.read_bytes())
    assert plist["Label"] == "test.eyemouse"
    assert plist["ProgramArguments"][-1] == "app"
    assert plist["RunAtLoad"] is True
    item.apply(False, [], REPO_ROOT)
    assert not item.enabled


def test_keyboard_suggestions() -> None:
    from opengaze.consumer.keyboard import suggest

    assert suggest("") == ["I", "Thanks", "Yes"]
    assert suggest("umb")[0] == "umbrella"
    assert suggest("Wat") == ["Water", "Watch"]
    assert "water" not in suggest("water")  # an already-complete word isn't offered
    assert len(suggest("t")) == 3


def test_triple_blink_starts_voice_typing() -> None:
    from opengaze.consumer.engine import consumer_bindings

    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    bindings = consumer_bindings(config)
    assert bindings[Gesture.TRIPLE_BLINK].app == "dictate"
    assert bindings[Gesture.DOUBLE_BLINK].mouse == "click"  # clicking stays a double blink



def test_fresh_start_ignores_the_saved_calibration(monkeypatch) -> None:
    from opengaze.config.loader import check_hardware_profile
    from opengaze.consumer.engine import GazeEngine
    from tests.helpers import HARDWARE_DIR

    config = load_app_config(REPO_ROOT / "configs" / "default.yaml")
    report = check_hardware_profile(HARDWARE_DIR / "synthetic-headset.yaml")
    engine = GazeEngine(report, config, {})
    monkeypatch.setattr(engine.store, "exists", lambda *a: True)
    monkeypatch.setattr(engine, "start_camera", lambda: None)
    assert engine.calibrated
    engine.ignore_saved_calibration = True  # `opengaze app --fresh`
    assert not engine.calibrated
    assert not engine.start_cursor() and not engine.running

def _hooks(calls: list):
    from opengaze.consumer.setup_window import SetupHooks

    return SetupHooks(
        calibrate=lambda: calls.append("calibrate"),
        finish=lambda login: calls.append(("finish", login)),
        quit=lambda: calls.append("quit"),
        voice_chosen=lambda on: calls.append(("voice", on)),
        has_voice_key=False,
        permission_app="Terminal",
    )


def test_setup_flow_skipping_voice(qapp, monkeypatch) -> None:
    from opengaze.cameras.permissions import CameraPermission
    from opengaze.consumer import setup_window as sw

    monkeypatch.setattr(sw, "request_camera_permission", lambda: CameraPermission.AUTHORIZED)
    monkeypatch.setattr(sw, "is_accessibility_granted", lambda: True)
    calls: list = []
    win = sw.SetupWindow(_hooks(calls), open_at_login=False)
    win.show()
    win.pages[win.WELCOME].findChildren(sw.QPushButton)[0].click()
    assert win.stack.currentIndex() == win.CAMERA
    win.pages[win.CAMERA].primary.click()
    assert win.stack.currentIndex() == win.ACCESS
    win.pages[win.ACCESS].primary.click()  # already granted: "Continue"
    assert win.stack.currentIndex() == win.MIC
    skip = next(b for b in win.pages[win.MIC].findChildren(sw.QPushButton)
                if b.text().startswith("Skip"))
    skip.click()
    assert calls == [("voice", False), "calibrate"]

    win.go(win.PRACTICE)
    done = next(b for b in win.practice.findChildren(sw.QPushButton) if b.text() == "Done")
    done.click()
    assert calls[-1] == ("finish", False)
    assert "quit" not in calls  # finishing closes setup without quitting



def test_try_it_out_can_stop_tracking(qapp) -> None:
    from opengaze.consumer.setup_window import SetupWindow

    calls: list = []
    hooks = _hooks(calls)
    hooks.toggle_tracking = lambda: calls.append("tracking")
    win = SetupWindow(hooks, open_at_login=True)
    win.go(SetupWindow.PRACTICE)
    page = win.practice
    assert page.tracking.text().startswith("Stop tracking")
    QTest.mouseClick(page.tracking, Qt.MouseButton.LeftButton)
    assert calls == ["tracking"]
    page.set_tracking(False)
    assert page.tracking.text().startswith("Start tracking")

def test_closing_setup_early_quits(qapp) -> None:
    from opengaze.consumer.setup_window import SetupWindow

    calls: list = []
    win = SetupWindow(_hooks(calls), open_at_login=True)
    win.show()
    win.close()
    assert calls == ["quit"]


def test_denied_camera_explains_and_opens_settings(qapp, monkeypatch) -> None:
    from opengaze.cameras.permissions import CameraPermission
    from opengaze.consumer import setup_window as sw

    opened: list[str] = []
    monkeypatch.setattr(sw, "request_camera_permission", lambda: CameraPermission.DENIED)
    monkeypatch.setattr(sw, "camera_permission_status", lambda: CameraPermission.DENIED)
    monkeypatch.setattr(sw, "open_privacy_pane", opened.append)
    win = sw.SetupWindow(_hooks([]), open_at_login=True)
    win.go(win.CAMERA)
    page = win.pages[win.CAMERA]
    page.primary.click()
    assert win.stack.currentIndex() == win.CAMERA
    assert "Terminal" in page.problem.text()
    assert page.primary.text() == "Open System Settings"
    page.primary.click()
    assert opened == ["Privacy_Camera"]


def _finish_step(qapp, stage) -> None:
    assert stage.done
    stage.done_at -= 2.0  # skip the "Nice!" pause
    stage._animate()
    qapp.processEvents()


def test_practice_needs_the_right_gesture_for_each_step(qapp) -> None:
    from opengaze.consumer.practice import PracticeStage

    stage = PracticeStage()
    stage.resize(400, 300)
    stage.show()
    centre = stage._circle().center().toPoint()
    QTest.mouseClick(stage, Qt.MouseButton.RightButton, pos=centre)
    assert not stage.done  # step 1 wants a left click (double blink)
    QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=centre)
    _finish_step(qapp, stage)

    card = stage._card().center().toPoint()
    QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=card)
    assert not stage.done  # step 2 wants a right click (right wink)
    QTest.mouseClick(stage, Qt.MouseButton.RightButton, pos=card)
    _finish_step(qapp, stage)

    block = stage._block().center().toPoint()
    slot = stage._slot().center().toPoint()
    QTest.mousePress(stage, Qt.MouseButton.LeftButton, pos=block)
    QTest.mouseMove(stage, QPoint(block.x() + 20, block.y()))
    QTest.mouseRelease(stage, Qt.MouseButton.LeftButton, pos=QPoint(block.x() + 20, block.y()))
    assert not stage.done and stage._block_offset == QPointF(0, 0)  # short drag snaps back

    QTest.mousePress(stage, Qt.MouseButton.LeftButton, pos=block)
    QTest.mouseMove(stage, slot)
    QTest.mouseRelease(stage, Qt.MouseButton.LeftButton, pos=slot)
    _finish_step(qapp, stage)
    assert stage.finished

    stage.repaint()
    QTest.mouseClick(stage, Qt.MouseButton.LeftButton, pos=stage._again.center())
    assert stage.step == 0


def test_floating_windows_render_offscreen(qapp) -> None:
    from opengaze.consumer.keyboard import GazeKeyboard
    from opengaze.consumer.voice_typing import ListeningPill, MicButton

    kb = GazeKeyboard()
    kb.open()
    kb._toggle_shift()
    assert kb._rows[0][0].text() == "Q"
    kb._toggle_symbols()
    assert kb._rows[0][0].text() == "1"
    assert not kb.grab().isNull()
    kb.close()
    MicButton(lambda: None).show()
    pill = ListeningPill()
    pill.show_text("Listening", 0.5)
    assert pill.width() > 60

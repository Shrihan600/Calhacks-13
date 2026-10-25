"""EyeMouse setup, done by a helper: Welcome → Glasses → Camera → Accessibility → Microphone
→ (calibration, fullscreen) → Try it out → Done.

The window only collects permissions and choices; ``EyeMouseApp`` runs calibration and
starts the gaze cursor when this window asks it to.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from opengaze.cameras.permissions import (
    CameraPermission,
    camera_permission_status,
    microphone_permission_status,
    request_camera_permission,
    request_microphone_permission,
)
from opengaze.cameras.types import CameraError
from opengaze.consumer import art, theme
from opengaze.consumer.practice import PracticeStage, paint_check, paint_eyes
from opengaze.control.ax import prompt_for_accessibility
from opengaze.control.cursor import is_accessibility_granted, open_accessibility_prefs

_W, _H = 880, 600
_ACCESS_POLL_MS = 1000
_GLASSES_POLL_MS = 200

# What each eye gesture does, as the practice page teaches it:
# (left eye open, right eye open, gesture, action).
GESTURE_GUIDE = [
    (False, False, "Double blink", "Click"),
    (False, True, "Hold left wink", "Drag"),
    (True, False, "Right wink", "Right-click"),
    (False, False, "Triple blink", "Voice typing"),
    (True, False, "Long right wink", "Pause"),
]
PAUSE_SHORTCUT = "⌃⌥G"
TRACKING_SHORTCUT = "⌃⌥T"


def open_privacy_pane(pane: str) -> None:
    """Open System Settings › Privacy & Security at ``pane`` (e.g. ``Privacy_Camera``)."""
    subprocess.run(
        ["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"], check=False
    )


class Scan(Protocol):
    """What the glasses page needs of ``opengaze.cameras.wired.GlassesScan``."""

    def poll(self) -> dict[str, str]: ...
    def image(self, board: str, found: dict[str, str]) -> np.ndarray | None: ...
    def close(self) -> None: ...


@dataclass
class SetupHooks:
    """What the setup window asks of the app."""

    calibrate: Callable[[], None]
    finish: Callable[[bool], None]   # open_at_login
    quit: Callable[[], None]
    voice_chosen: Callable[[bool], None]
    has_voice_key: bool
    permission_app: str              # whose permissions macOS checks (us, or the terminal)
    # Finds the glasses. None: a developer's webcam-only profile, so there's nothing to connect.
    scan_glasses: Callable[[], Scan] | None = None
    # Turns the gaze cursor off or back on while trying it out (the app's ⌃⌥T).
    toggle_tracking: Callable[[], None] = lambda: None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _label(text: str, role: str = "", wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    lbl.setWordWrap(wrap)
    lbl.setTextFormat(Qt.TextFormat.RichText)
    return lbl


def _button(text: str, kind: str, on_click: Callable[[], None]) -> QPushButton:
    btn = QPushButton(text)
    btn.setProperty("kind", kind)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    btn.clicked.connect(on_click)
    return btn


class CheckBadge(QWidget):
    """A lime tick in a circle, before a short note."""

    def __init__(self, text: str) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._tick = _Tick()
        self.label = _label(text, "note", wrap=False)
        row.addWidget(self._tick)
        row.addWidget(self.label)
        row.addStretch(1)


class _Tick(QWidget):
    def __init__(self, size: int = 18) -> None:
        super().__init__()
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.LIME))
        p.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        c = QPointF(self.width() / 2, self.height() / 2)
        paint_check(p, c, self.width() * 0.9, theme.PURPLE)
        p.end()


class StepHeader(QWidget):
    """"Allow access · Calibrate" progress, centred in the window's top strip."""

    STEPS = ("Allow access", "Calibrate")

    def __init__(self, current: int) -> None:
        super().__init__()
        self.current = current
        self.setFixedHeight(48)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = theme.text_font(13, 600)
        p.setFont(font)
        fm = p.fontMetrics()
        widths = [16 + fm.horizontalAdvance(s) for s in self.STEPS]
        x = (self.width() - (sum(widths) + 24 * (len(widths) - 1))) / 2
        cy = self.height() / 2
        for i, (label, w) in enumerate(zip(self.STEPS, widths, strict=True)):
            if i < self.current:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.qcolor(theme.LIME))
                p.drawEllipse(QPointF(x + 5, cy), 7, 7)
                paint_check(p, QPointF(x + 5, cy), 13, theme.PURPLE)
                color = theme.BODY
            elif i == self.current:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.qcolor(theme.PURPLE))
                p.drawEllipse(QPointF(x + 4, cy), 4, 4)
                color = theme.PURPLE
            else:
                p.setPen(QPen(theme.qcolor(theme.FAINT), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(x + 4, cy), 3, 3)
                color = theme.MUTED
            p.setPen(theme.qcolor(color))
            p.drawText(QRectF(x + 16, 0, w, self.height()), Qt.AlignmentFlag.AlignVCenter, label)
            x += w + 24
        p.end()


class Page(QWidget):
    """Header, a text column beside an illustration, and a footer of buttons."""

    def __init__(self, step: int | None, art_widget: QWidget | None = None) -> None:
        super().__init__()
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        if step is not None:
            outer.addWidget(StepHeader(step))
        body = QHBoxLayout()
        body.setContentsMargins(64, 24, 48, 0)
        body.setSpacing(48)
        self.text = QVBoxLayout()
        self.text.setSpacing(0)
        body.addLayout(self.text, 1)
        if art_widget is not None:
            body.addWidget(art_widget, 0, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(body, 1)
        self.footer = QHBoxLayout()
        self.footer.setContentsMargins(46, 24, 40, 32)
        self.footer.setSpacing(12)
        outer.addLayout(self.footer)

    def heading(self, title: str, body: str | None = None) -> QLabel | None:
        t = _label(title, "title")
        self.text.addWidget(t)
        self.text.addSpacing(14)
        if body is None:
            return None
        b = _label(body, "body")
        b.setMaximumWidth(400)
        self.text.addWidget(b)
        return b

    def on_shown(self) -> None:
        """Called each time the page comes to the front."""

    def on_hidden(self) -> None:
        """Called when another page replaces this one."""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

class WelcomePage(Page):
    def __init__(self, start: Callable[[], None]) -> None:
        super().__init__(None, art.SvgArt(art.welcome(), 340, 500))
        logo = art.SvgArt(theme.LOGO_SVG, 64, 64)
        self.text.addSpacing(20)
        self.text.addWidget(logo)
        self.text.addSpacing(28)
        title = _label(f"Let’s set up<br>{theme.APP_NAME}", "title")
        title.setStyleSheet("font-size: 44px;")
        self.text.addWidget(title)
        self.text.addSpacing(32)
        for n, (label, current) in enumerate((("Allow access", True),
                                              ("Calibrate and practice", False)), start=1):
            self.text.addWidget(_NumberedRow(n, label, current))
            self.text.addSpacing(14)
        self.text.addStretch(1)
        start_btn = _button("Start setup", "primary", start)
        row = QHBoxLayout()
        row.addWidget(start_btn)
        row.addStretch(1)
        self.text.addLayout(row)
        self.footer.setContentsMargins(0, 0, 0, 24)


class _NumberedRow(QWidget):
    def __init__(self, n: int, label: str, current: bool) -> None:
        super().__init__()
        self._n, self._current = n, current
        row = QHBoxLayout(self)
        row.setContentsMargins(40, 0, 0, 0)
        text = _label(label, wrap=False)
        text.setStyleSheet("font-size: 15px;")
        row.addWidget(text)
        self.setFixedHeight(26)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PURPLE if self._current else theme.INK))
        p.drawEllipse(QRectF(0, 0, 26, 26))
        p.setPen(theme.qcolor("#FFFFFF" if self._current else theme.CREAM))
        p.setFont(theme.display_font(13))
        p.drawText(QRectF(0, 0, 26, 26), Qt.AlignmentFlag.AlignCenter, str(self._n))
        p.end()


class EyePreview(QWidget):
    """One eye camera's live picture, or what it's waiting for."""

    def __init__(self, side: str) -> None:
        super().__init__()
        self.side = side
        self.image: QImage | None = None
        self.connected = False
        self.setFixedSize(170, 150)

    def show_frame(self, connected: bool, image: np.ndarray | None) -> None:
        from opengaze.desktop.diagnostics import to_qimage

        self.connected = connected
        self.image = to_qimage(image) if connected and image is not None else None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        box = QRectF(0, 0, self.width(), self.height() - 28).adjusted(1, 1, -1, -1)
        p.setPen(QPen(theme.qcolor(theme.LINE), 1))
        p.setBrush(theme.qcolor("#FFFFFF"))
        p.drawRoundedRect(box, 14, 14)
        if self.image is not None:
            scaled = self.image.scaled(box.size().toSize(),
                                       Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                       Qt.TransformationMode.SmoothTransformation)
            clip = QPainterPath()
            clip.addRoundedRect(box, 14, 14)
            p.save()
            p.setClipPath(clip)
            p.drawImage(QPointF(box.center().x() - scaled.width() / 2,
                                box.center().y() - scaled.height() / 2), scaled)
            p.restore()
        else:
            p.setPen(theme.qcolor(theme.MUTED))
            p.setFont(theme.text_font(13, 500))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter,
                       "Connected" if self.connected else "Waiting…")
        label = QRectF(0, self.height() - 22, self.width(), 22)
        if self.connected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(theme.LIME))
            p.drawEllipse(QPointF(9, label.center().y()), 7, 7)
            paint_check(p, QPointF(9, label.center().y()), 13, theme.PURPLE)
        else:
            p.setPen(QPen(theme.qcolor(theme.FAINT), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(9, label.center().y()), 5, 5)
        p.setPen(theme.qcolor(theme.INK))
        p.setFont(theme.text_font(14, 600))
        p.drawText(label.adjusted(24, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter,
                   f"{self.side.capitalize()} eye")
        p.end()


class GlassesPage(Page):
    """Finds the glasses on whichever USB ports they're in, and shows both eye cameras."""

    def __init__(self, hooks: SetupHooks, back: Callable[[], None],
                 forward: Callable[[], None]) -> None:
        super().__init__(0)
        self._hooks = hooks
        self._scan: Scan | None = None
        self.found: dict[str, str] = {}
        self.text.addStretch(1)
        self.heading("Connect your glasses",
                     "Plug both glasses cables into this Mac. Any USB ports work. "
                     "Each camera shows up here when it’s found.")
        self.text.addSpacing(24)
        row = QHBoxLayout()
        row.setSpacing(20)
        self.previews = {side: EyePreview(side) for side in ("left", "right")}
        for preview in self.previews.values():
            row.addWidget(preview)
        row.addStretch(1)
        self.text.addLayout(row)
        self.text.addSpacing(14)
        self.problem = _label("", "error")
        self.problem.setMaximumWidth(400)
        self.problem.hide()
        self.text.addWidget(self.problem)
        self.text.addStretch(1)
        self.footer.addWidget(_button("Back", "ghost", back))
        self.footer.addStretch(1)
        self.primary = _button("Continue", "primary", forward)
        self.primary.setEnabled(False)
        self.footer.addWidget(self.primary)
        self._poll = QTimer(self)
        self._poll.setInterval(_GLASSES_POLL_MS)
        self._poll.timeout.connect(self._check)

    def on_shown(self) -> None:
        assert self._hooks.scan_glasses is not None
        self._scan = self._hooks.scan_glasses()
        self._check()
        self._poll.start()

    def on_hidden(self) -> None:
        self._poll.stop()
        if self._scan is not None:
            self._scan.close()
            self._scan = None

    def _check(self) -> None:
        if self._scan is None:
            return
        try:
            self.found = self._scan.poll()
        except CameraError as exc:  # the receiver isn't installed
            self._poll.stop()
            self.problem.setText(str(exc))
            self.problem.show()
            return
        for side, preview in self.previews.items():
            preview.show_frame(side in self.found, self._scan.image(side, self.found))
        self.primary.setEnabled(len(self.found) == len(self.previews))


class CameraPage(Page):
    def __init__(self, hooks: SetupHooks, back: Callable[[], None],
                 forward: Callable[[], None]) -> None:
        super().__init__(0, art.SvgArt(art.camera_prompt(), 330, 330))
        self._hooks, self._forward = hooks, forward
        self.text.addStretch(1)
        self.body = self.heading("Allow the camera", "")
        self.text.addSpacing(18)
        self.problem = _label("", "error")
        self.problem.setMaximumWidth(400)
        self.problem.hide()
        self.text.addWidget(self.problem)
        self.text.addStretch(1)
        self.footer.addWidget(_button("Back", "ghost", back))
        self.footer.addStretch(1)
        self.primary = _button("Allow camera", "primary", self._allow)
        self.footer.addWidget(self.primary)
        self._denied = False

    def on_shown(self) -> None:
        follows = ("your head through this Mac’s camera, and your eyes through the glasses"
                   if self._hooks.scan_glasses is not None
                   else "your eyes through this Mac’s camera")
        assert self.body is not None
        self.body.setText(f"{theme.APP_NAME} follows {follows}. "
                          "Video is processed on this Mac and isn’t sent anywhere.")
        if camera_permission_status() is CameraPermission.AUTHORIZED:
            self.primary.setText("Continue")

    def _allow(self) -> None:
        if self._denied and camera_permission_status() is not CameraPermission.AUTHORIZED:
            open_privacy_pane("Privacy_Camera")
            return
        status = request_camera_permission()
        if status in (CameraPermission.AUTHORIZED, CameraPermission.UNAVAILABLE):
            self._forward()
            return
        self._denied = True
        app = self._hooks.permission_app
        self.problem.setText(
            f"The camera is off for {app}. Turn it on in System Settings › Privacy &amp; "
            f"Security › Camera, then quit and reopen {app}."
        )
        self.problem.show()
        self.primary.setText("Open System Settings")


class AccessibilityPage(Page):
    def __init__(self, hooks: SetupHooks, back: Callable[[], None],
                 forward: Callable[[], None]) -> None:
        app = hooks.permission_app
        super().__init__(0, art.SvgArt(art.accessibility_settings(app), 380, 320))
        self._forward = forward
        self.text.addStretch(1)
        self.heading("Turn on Accessibility",
                     f"This lets {theme.APP_NAME} move the pointer and click for you.")
        self.text.addSpacing(10)
        for n, line in enumerate((
            "Click <b>Open System Settings</b>",
            f"Switch on <b>{app}</b> in the list",
            "Come back here. This page moves on by itself.",
        ), start=1):
            row = QHBoxLayout()
            row.setSpacing(12)
            num = _label(f"<span style='color:{theme.PURPLE}'>{n}</span>", wrap=False)
            num.setStyleSheet(f"font-family: '{theme.DISPLAY}'; font-size: 15px; "
                              "font-weight: 600;")
            num.setFixedWidth(14)
            text = _label(line)
            text.setStyleSheet("font-size: 15px;")
            row.addWidget(num, 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(text, 1)
            self.text.addLayout(row)
            self.text.addSpacing(12)
        self.text.addSpacing(6)
        fine = _label(f"Not in the list? Click <b>+</b> under it and choose {app} "
                      "from Applications.", "fine")
        fine.setMaximumWidth(400)
        self.text.addWidget(fine)
        self.status = CheckBadge("Accessibility is on")
        self.status.hide()
        self.text.addSpacing(12)
        self.text.addWidget(self.status)
        self.text.addStretch(1)
        self.footer.addWidget(_button("Back", "ghost", back))
        self.footer.addStretch(1)
        self.primary = _button("Open System Settings", "primary", self._open)
        self.footer.addWidget(self.primary)
        self._asked = False
        self._poll = QTimer(self)
        self._poll.setInterval(_ACCESS_POLL_MS)
        self._poll.timeout.connect(self._check)

    def on_shown(self) -> None:
        granted = is_accessibility_granted()
        self.status.setVisible(granted)
        self.primary.setText("Continue" if granted else "Open System Settings")
        self._poll.start()

    def on_hidden(self) -> None:
        self._poll.stop()

    def _open(self) -> None:
        if is_accessibility_granted():
            self._forward()
            return
        self._asked = True
        prompt_for_accessibility()
        open_accessibility_prefs()

    def _check(self) -> None:
        if not is_accessibility_granted():
            return
        self.status.show()
        if self._asked:
            self._poll.stop()
            self._asked = False
            QTimer.singleShot(600, self._forward)
        else:
            self.primary.setText("Continue")


class MicrophonePage(Page):
    def __init__(self, hooks: SetupHooks, back: Callable[[], None],
                 forward: Callable[[], None]) -> None:
        super().__init__(0, art.SvgArt(art.voice_typing(), 330, 330))
        self._hooks, self._forward = hooks, forward
        self.text.addStretch(1)
        self.badge = CheckBadge("Camera and Accessibility are on")
        self.text.addWidget(self.badge)
        self.text.addSpacing(18)
        self.heading("Turn on Microphone",
                     f"Let {theme.APP_NAME} turn your words into text.")
        self.text.addSpacing(20)
        fine = "Speech is turned into text by Cartesia, online. Audio is sent only while the " \
               "mic is on."
        if not hooks.has_voice_key:
            fine += " Voice typing also needs a Cartesia API key (CARTESIA_API_KEY in .env)."
        note = _label(fine, "fine")
        note.setMaximumWidth(400)
        self.text.addWidget(note)
        self.text.addSpacing(12)
        self.problem = _label("", "error")
        self.problem.setMaximumWidth(400)
        self.problem.hide()
        self.text.addWidget(self.problem)
        self.text.addStretch(1)
        self.footer.addWidget(_button("Back", "ghost", back))
        self.footer.addStretch(1)
        self.footer.addWidget(_button("Skip, no voice", "ghost", self._skip))
        self.primary = _button("Allow microphone", "primary", self._allow)
        self.footer.addWidget(self.primary)
        self._denied = False

    def on_shown(self) -> None:
        camera = camera_permission_status() in (CameraPermission.AUTHORIZED,
                                                CameraPermission.UNAVAILABLE)
        access = is_accessibility_granted()
        text = {(True, True): "Camera and Accessibility are on",
                (True, False): "Camera is on", (False, True): "Accessibility is on"}
        key = (camera, access)
        self.badge.setVisible(key in text)
        self.badge.label.setText(text.get(key, ""))

    def _skip(self) -> None:
        self._hooks.voice_chosen(False)
        self._forward()

    def _allow(self) -> None:
        if self._denied and microphone_permission_status() is not CameraPermission.AUTHORIZED:
            open_privacy_pane("Privacy_Microphone")
            return
        status = request_microphone_permission()
        if status in (CameraPermission.AUTHORIZED, CameraPermission.UNAVAILABLE):
            self._hooks.voice_chosen(True)
            self._forward()
            return
        self._denied = True
        app = self._hooks.permission_app
        self.problem.setText(
            f"The microphone is off for {app}. Turn it on in System Settings › Privacy &amp; "
            f"Security › Microphone, or skip voice for now."
        )
        self.problem.show()
        self.primary.setText("Open System Settings")


class GestureList(QWidget):
    """The gesture → action table, with eye glyphs."""

    ROW_H = 41

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(self.ROW_H * len(GESTURE_GUIDE) + 1)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        p.setPen(QPen(theme.qcolor(theme.LINE), 1))
        p.drawLine(0, 0, w, 0)
        for i, (left_open, right_open, how, does) in enumerate(GESTURE_GUIDE):
            top = i * self.ROW_H
            paint_eyes(p, 22, top + self.ROW_H / 2, left_open, right_open)
            p.setPen(theme.qcolor(theme.INK))
            p.setFont(theme.text_font(15, 600))
            p.drawText(QRectF(58, top, w - 58, self.ROW_H), Qt.AlignmentFlag.AlignVCenter, how)
            p.setPen(theme.qcolor(theme.MUTED))
            p.setFont(theme.text_font(14, 400))
            p.drawText(QRectF(0, top, w, self.ROW_H),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, does)
            p.setPen(QPen(theme.qcolor(theme.LINE), 1))
            p.drawLine(0, top + self.ROW_H, w, top + self.ROW_H)
        p.end()


class PracticePage(QWidget):
    def __init__(self, redo: Callable[[], None], done: Callable[[bool], None],
                 open_at_login: bool, toggle_tracking: Callable[[], None]) -> None:
        super().__init__()
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(StepHeader(1))

        body = QHBoxLayout()
        body.setContentsMargins(64, 16, 48, 0)
        body.setSpacing(44)
        left = QVBoxLayout()
        left.setSpacing(0)
        cal = QHBoxLayout()
        cal.setSpacing(8)
        cal.addWidget(_Tick())
        cal.addWidget(_label("Calibrated", "note", wrap=False))
        cal.addWidget(_button("Redo", "link", redo))
        cal.addStretch(1)
        # A way out when tracking is too bad to use: no cursor, clicks or gestures.
        self.tracking = _button("", "link", toggle_tracking)
        cal.addWidget(self.tracking)
        self.set_tracking(True)
        left.addLayout(cal)
        left.addSpacing(14)
        title = _label("Try it out", "title")
        title.setStyleSheet("font-size: 38px;")
        left.addWidget(title)
        left.addSpacing(16)
        self.stage = PracticeStage()
        left.addWidget(self.stage, 1)
        body.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(0)
        right.setContentsMargins(0, 6, 0, 0)
        h = _label("Eye gestures", wrap=False)
        h.setStyleSheet(f"font-family: '{theme.DISPLAY}'; font-size: 18px; font-weight: 600;")
        right.addWidget(h)
        right.addSpacing(12)
        right.addWidget(GestureList())
        right.addSpacing(16)
        right.addWidget(self._after_setup(open_at_login))
        right.addStretch(1)
        side = QWidget()
        side.setFixedWidth(330)
        side.setLayout(right)
        body.addWidget(side)
        outer.addLayout(body, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(46, 20, 48, 32)
        footer.addWidget(_button("Back", "ghost", redo))
        footer.addStretch(1)
        footer.addWidget(_button("Done", "primary", lambda: done(self.login.isChecked())))
        outer.addLayout(footer)

    def set_tracking(self, on: bool) -> None:
        self.tracking.setText(f"{'Stop' if on else 'Start'} tracking  {TRACKING_SHORTCUT}")

    def _after_setup(self, open_at_login: bool) -> QWidget:
        box = QFrame()
        box.setObjectName("after")
        box.setStyleSheet(f"QFrame#after {{ background: #FFFFFF; border: 1px solid {theme.LINE};"
                          " border-radius: 16px; }")
        col = QVBoxLayout(box)
        col.setContentsMargins(16, 14, 16, 16)
        col.setSpacing(12)
        h = _label("After setup", wrap=False)
        h.setStyleSheet(f"font-family: '{theme.DISPLAY}'; font-size: 15px; font-weight: 600;")
        col.addWidget(h)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(art.SvgArt(theme.MENU_LOGO_SVG, 22, 22))
        lives = _label(f"{theme.APP_NAME} lives in the menu bar.")
        lives.setStyleSheet(f"font-size: 14px; color: {theme.BODY};")
        row.addWidget(lives, 1)
        col.addLayout(row)
        self.login = QCheckBox(f"Open {theme.APP_NAME} when this Mac starts")
        self.login.setChecked(open_at_login)
        col.addWidget(self.login)
        keys = QHBoxLayout()
        helper = _label("Helper: pause or resume", wrap=False)
        helper.setStyleSheet(f"font-size: 14px; color: {theme.BODY};")
        keys.addWidget(helper)
        keys.addStretch(1)
        kbd = _label(PAUSE_SHORTCUT, wrap=False)
        kbd.setStyleSheet("font-size: 13px; font-weight: 600; padding: 5px 8px; "
                          "border-radius: 6px; background: #EFE8DA; letter-spacing: 1px;")
        keys.addWidget(kbd)
        col.addLayout(keys)
        return box


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class SetupWindow(QWidget):
    WELCOME, GLASSES, CAMERA, ACCESS, MIC, PRACTICE = range(6)

    def __init__(self, hooks: SetupHooks, open_at_login: bool) -> None:
        super().__init__()
        self._hooks = hooks
        self._closing_for_good = False
        self.setWindowTitle(f"{theme.APP_NAME} Setup")
        self.setFixedSize(_W, _H)
        self.setStyleSheet(theme.stylesheet())
        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        go = self.go
        # Without a glasses scan the glasses page is skipped (but keeps its index).
        ask = hooks.scan_glasses is not None
        before_camera = self.GLASSES if ask else self.WELCOME
        self.pages: list[QWidget] = [
            WelcomePage(lambda: go(self.GLASSES if ask else self.CAMERA)),
            GlassesPage(hooks, lambda: go(self.WELCOME), lambda: go(self.CAMERA))
            if ask else QWidget(),
            CameraPage(hooks, lambda: go(before_camera), lambda: go(self.ACCESS)),
            AccessibilityPage(hooks, lambda: go(self.CAMERA), lambda: go(self.MIC)),
            MicrophonePage(hooks, lambda: go(self.ACCESS), hooks.calibrate),
            PracticePage(hooks.calibrate, self._done, open_at_login, hooks.toggle_tracking),
        ]
        for page in self.pages:
            self.stack.addWidget(page)
        self.go(self.WELCOME)

    @property
    def practice(self) -> PracticePage:
        return self.pages[self.PRACTICE]  # type: ignore[return-value]

    def go(self, index: int) -> None:
        old = self.stack.currentWidget()
        if isinstance(old, Page) and old is not self.pages[index]:
            old.on_hidden()
        self.stack.setCurrentIndex(index)
        page = self.pages[index]
        if isinstance(page, Page):
            page.on_shown()
        if index == self.PRACTICE:
            self.practice.stage.restart()

    def _done(self, open_at_login: bool) -> None:
        self._closing_for_good = True
        self.close()
        self._hooks.finish(open_at_login)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        super().closeEvent(event)
        if not self._closing_for_good:
            self._hooks.quit()  # closing setup before it's done quits the app

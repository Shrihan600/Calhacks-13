"""EyeMouse's calibration screen: the standard 9-dot wizard, restyled and simplified.

Collection, fitting and validation are ``CalibrationWindow``'s. What changes is what the
person sees: a "Look at the series of dots" card to start, the lime-and-purple target, and
no error numbers. A good fit saves and closes by itself; a poor one offers "Try again" or
"Use it anyway".

A hardware problem (e.g. the glasses unplugged) is drawn on this screen, since it covers the
whole display, and Start waits until it's fixed. A failure to save is shown too.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QApplication

from opengaze.calibration.storage import CalibrationStore
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.consumer import theme
from opengaze.consumer.problems import Problem
from opengaze.consumer.warning_banner import banner_rect, paint_problem
from opengaze.desktop.calibration_ui import _FLASH_MS, CalibrationWindow, _State

Outcome = Literal["saved", "cancelled", "retry"]

_CARD_W, _CARD_H = 440, 330
_BTN_H = 52


class EyeMouseCalibration(CalibrationWindow):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        on_done: Callable[[Outcome], None],
        problem: Callable[[], Problem | None] = lambda: None,
    ) -> None:
        super().__init__(runtime, report, config, CalibrationStore(), owns_runtime=False)
        self._problem = problem
        self.setWindowTitle(f"{theme.APP_NAME} — Calibrate")
        self.setStyleSheet(f"background: {theme.CREAM};")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self._on_done = on_done
        self._outcome: Outcome = "cancelled"
        self._failed = False  # the last attempt couldn't be fitted
        self._save_failed = False
        self._buttons: dict[str, QRect] = {}

    # -- flow --------------------------------------------------------------

    def _finish(self) -> None:
        super()._finish()
        if self._state == _State.IDLE:  # the fit failed; the intro card says so
            self._failed = True
        elif self._validation is None or self._validation.passed:
            self._accept()

    def _blocking(self) -> Problem | None:
        problem = self._problem()
        return problem if problem is not None and problem.severity == "error" else None

    def _accept(self) -> None:
        self._save_and_close()
        if not self._saved:  # logged by the base class; say so here, and let them retry
            self._save_failed = True
            self._state = _State.IDLE
            self.update()
            return
        self._outcome = "saved"
        self.close()

    def _retry(self) -> None:
        self._outcome = "retry"
        self.close()

    def _on_advance(self) -> None:
        if self._state == _State.RESULT:
            self._retry()
        elif self._state == _State.IDLE and self._blocking() is not None:
            return  # the banner on screen says why
        else:
            self._save_failed = False
            super()._on_advance()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Return and self._state == _State.RESULT:
            self._accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._state == _State.RESULT:
            pos = event.position().toPoint()
            if self._buttons.get("again", QRect()).contains(pos):
                self._retry()
            elif self._buttons.get("anyway", QRect()).contains(pos):
                self._accept()
            return
        super().mousePressEvent(event)

    def _tick(self) -> None:
        super()._tick()
        # The pointer is only shown where there's something to press.
        shown = self._state in (_State.IDLE, _State.RESULT)
        self.setCursor(Qt.CursorShape.ArrowCursor if shown else Qt.CursorShape.BlankCursor)

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
        on_done, self._on_done = self._on_done, None
        if on_done is not None:
            on_done(self._outcome)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor(theme.CREAM))
        if self._state == _State.IDLE:
            self._paint_dots(p, active=None)
            self._paint_intro(p)
        elif self._state in (_State.COLLECTING, _State.FLASH):
            self._paint_dots(p, active=self._current_point)
            self._paint_target(p)
        elif self._state == _State.RESULT:
            self._paint_dots(p, active=None)
            self._paint_result(p)
        problem = self._blocking()
        if problem is not None:
            paint_problem(p, QRectF(banner_rect(self.rect(), problem)), problem)
        p.end()

    def _point_px(self, i: int) -> QPointF:
        pt = self.session.points[i]
        return QPointF(pt.screen_x_norm * self.width(), pt.screen_y_norm * self.height())

    def _paint_dots(self, p: QPainter, active: int | None) -> None:
        for i, pt in enumerate(self.session.points):
            if i == active:
                continue
            c = self._point_px(i)
            if pt.accepted:
                p.setPen(QPen(theme.qcolor(theme.PURPLE), 2))
                p.setBrush(theme.qcolor(theme.LIME))
            else:
                p.setPen(QPen(theme.qcolor(theme.DOT_OFF), 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, 6, 6)

    def _paint_ring_target(self, p: QPainter, c: QPointF, scale: float, progress: float) -> None:
        r = 36 * scale
        track = QPen(theme.qcolor(theme.PURPLE, 0.14), 8 * scale)
        p.setPen(track)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        if progress > 0:
            arc = QPen(theme.qcolor(theme.PURPLE), 8 * scale)
            arc.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(arc)
            p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 90 * 16,
                      int(-min(progress, 1.0) * 360 * 16))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.LIME))
        p.drawEllipse(c, 17 * scale, 17 * scale)
        p.setBrush(theme.qcolor(theme.PURPLE))
        p.drawEllipse(c, 7 * scale, 7 * scale)

    def _paint_target(self, p: QPainter) -> None:
        c = self._point_px(self._current_point)
        if self._state == _State.FLASH and self._flash_start is not None:
            age = min(1.0, (time.monotonic() - self._flash_start) * 1000 / _FLASH_MS)
            if self._flash_ok:
                burst = theme.qcolor(theme.LIME, 0.7 * (1 - age))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(burst)
                p.drawEllipse(c, 48 + 30 * age, 48 + 30 * age)
            self._paint_ring_target(p, c, 1.0, 1.0 if self._flash_ok else 0.0)
            if not self._flash_ok:
                self._paint_hint(p, c, "Hold still and look at the dot")
        else:
            self._paint_ring_target(p, c, 1.0, self.session.collect_progress)
        p.setPen(theme.qcolor(theme.MUTED))
        p.setFont(theme.text_font(15, 600))
        n = self.session.n_targets
        p.drawText(QRect(0, self.height() - 120, self.width(), 24), Qt.AlignmentFlag.AlignCenter,
                   f"{min(self._current_point + 1, n)} of {n}")

    def _paint_hint(self, p: QPainter, c: QPointF, text: str) -> None:
        p.setFont(theme.text_font(16, 600))
        p.setPen(theme.qcolor(theme.BODY))
        y = int(c.y() + 64) if c.y() < self.height() * 0.75 else int(c.y() - 88)
        p.drawText(QRect(int(c.x()) - 200, y, 400, 24), Qt.AlignmentFlag.AlignCenter, text)

    def _card(self, p: QPainter, h: int) -> QRect:
        p.fillRect(self.rect(), theme.qcolor(theme.INK, 0.45))
        card = QRect(0, 0, _CARD_W, h)
        card.moveCenter(self.rect().center())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor("#FFFFFF"))
        p.drawRoundedRect(card, 24, 24)
        return card

    def _paint_copy(self, p: QPainter, card: QRect, top: int, title: str, body: str) -> int:
        p.setPen(theme.qcolor(theme.INK))
        p.setFont(theme.display_font(30))
        p.drawText(QRect(card.left() + 32, top, card.width() - 64, 40),
                   Qt.AlignmentFlag.AlignCenter, title)
        p.setPen(theme.qcolor(theme.BODY))
        p.setFont(theme.text_font(17, 400))
        body_rect = QRect(card.left() + 40, top + 50, card.width() - 80, 56)
        p.drawText(body_rect, Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap, body)
        return body_rect.bottom()

    def _button(self, p: QPainter, rect: QRect, label: str, primary: bool) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PURPLE if primary else theme.SAND))
        p.drawRoundedRect(rect, 14, 14)
        p.setPen(theme.qcolor("#FFFFFF" if primary else theme.BODY))
        p.setFont(theme.text_font(17, 600))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_intro(self, p: QPainter) -> None:
        card = self._card(p, _CARD_H)
        self._paint_ring_target(p, QPointF(card.center().x(), card.top() + 76), 0.75, 0.7)
        blocked = self._blocking() is not None
        if self._save_failed:
            title = "Couldn’t save that"
            body = "Please try again. If it keeps happening, ask for help."
        elif self._failed:
            title, body = "Let’s try that again", "Sit facing the screen with light on your face."
        else:
            title, body = "Look at the series of dots", "Keep your head still. Only move your eyes."
        if blocked:
            body = "Fix the problem at the top of the screen first."
        elif not self._eye_ready:
            body = "Look around the whole screen for a few seconds first."
        self._paint_copy(p, card, card.top() + 130, title, body)
        btn = QRect(0, 0, 140, _BTN_H)
        btn.moveCenter(card.center())
        btn.moveBottom(card.bottom() - 36)
        self._button(p, btn, "Start", primary=self._eye_ready and not blocked)

    def _paint_result(self, p: QPainter) -> None:
        card = self._card(p, 300)
        self._paint_copy(p, card, card.top() + 48, "Not quite accurate yet",
                         "Good light on your face and sitting an arm’s length away both help.")
        again = QRect(card.left() + 40, card.bottom() - 36 - _BTN_H, 170, _BTN_H)
        anyway = QRect(card.right() - 40 - 170, again.top(), 170, _BTN_H)
        self._button(p, again, "Try again", primary=True)
        self._button(p, anyway, "Use it anyway", primary=False)
        self._buttons = {"again": again, "anyway": anyway}


def run_calibration_once(
    runtime: CaptureRuntime,
    report: ProfileReport,
    config: AppConfig,
    on_done: Callable[[Outcome], None],
    problem: Callable[[], Problem | None] = lambda: None,
) -> EyeMouseCalibration:
    """Open the calibration screen; ``on_done`` gets the outcome when it closes. ``problem``:
    the current hardware problem, drawn on the screen."""
    window = EyeMouseCalibration(runtime, report, config, on_done, problem)
    window.show()
    window.raise_()
    window.activateWindow()
    QApplication.processEvents()
    return window

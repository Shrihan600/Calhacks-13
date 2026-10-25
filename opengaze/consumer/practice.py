"""The "Try it out" stage: one eye gesture at a time, on real mouse events.

By the time this shows, the gaze cursor is live, so a double blink arrives here as an
ordinary left click, a right wink as a right click and a held left wink as a press, drag
and release. A helper's mouse works the same way.

1. Double blink to click the circle.
2. Right wink to right-click the card.
3. Hold a left wink to drag the block into the slot.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from opengaze.consumer import theme

_STEPS = 3
_ADVANCE_S = 1.2   # "Nice!" shows this long before the next gesture
_BURST_S = 0.8

# (left eye open, right eye open, label) for each step's instruction.
_INSTRUCTIONS = [
    (False, False, "Double blink to click"),
    (True, False, "Right wink to right-click"),
    (False, True, "Hold left wink to drag"),
]


def paint_eyes(p: QPainter, cx: float, cy: float, left_open: bool, right_open: bool,
               scale: float = 1.0) -> None:
    """The design's eye glyph: a dot for an open eye, a dash for a shut one."""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.qcolor(theme.PURPLE))
    for dx, is_open in ((-8 * scale, left_open), (8 * scale, right_open)):
        x = cx + dx
        if is_open:
            p.drawEllipse(QPointF(x, cy), 5.5 * scale, 5.5 * scale)
        else:
            p.drawRoundedRect(QRectF(x - 6.5 * scale, cy - 2 * scale, 13 * scale, 4 * scale),
                              2 * scale, 2 * scale)


def paint_check(p: QPainter, c: QPointF, size: float, color: str, progress: float = 1.0) -> None:
    path = QPainterPath(QPointF(c.x() - size * 0.25, c.y() + size * 0.02))
    path.lineTo(c.x() - size * 0.08, c.y() + size * 0.19)
    path.lineTo(c.x() + size * 0.25, c.y() - size * 0.19)
    pen = QPen(theme.qcolor(color), max(2.0, size * 0.125))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    if progress < 1.0:
        path = _partial(path, progress)
    p.drawPath(path)


def _partial(path: QPainterPath, t: float) -> QPainterPath:
    out = QPainterPath(path.pointAtPercent(0))
    steps = 24
    for i in range(1, int(steps * max(0.0, t)) + 1):
        out.lineTo(path.pointAtPercent(i / steps))
    return out


class PracticeStage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(380, 280)
        self.step = 0
        self.done_at: float | None = None
        self._hover = False
        self._drag_from: QPointF | None = None
        self._block_offset = QPointF(0, 0)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate)
        self._again = QRect()

    @property
    def done(self) -> bool:
        return self.done_at is not None

    @property
    def finished(self) -> bool:
        return self.step >= _STEPS

    def restart(self) -> None:
        self.step, self.done_at = 0, None
        self._drag_from, self._block_offset = None, QPointF(0, 0)
        self._timer.stop()
        self.update()

    # -- geometry ----------------------------------------------------------

    def _toy_center(self) -> QPointF:
        return QPointF(self.width() / 2, 34 + (self.height() - 34 - 64) / 2)

    def _circle(self) -> QRectF:
        c = self._toy_center()
        return QRectF(c.x() - 60, c.y() - 60, 120, 120)

    def _card(self) -> QRectF:
        c = self._toy_center()
        return QRectF(c.x() - 70, c.y() - 62, 140, 124)

    def _slot(self) -> QRectF:
        c = self._toy_center()
        return QRectF(c.x() + 24 - 10, c.y() - 42, 84, 84)

    def _block_home(self) -> QRectF:
        c = self._toy_center()
        return QRectF(c.x() - 24 - 64 - 10, c.y() - 32, 64, 64)

    def _block(self) -> QRectF:
        if self.done:
            slot = self._slot()
            return QRectF(slot.center().x() - 32, slot.center().y() - 32, 64, 64)
        return self._block_home().translated(self._block_offset)

    def _hit(self, pos: QPointF) -> bool:
        if self.step == 0:
            c = self._circle().center()
            return (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2 <= 60 ** 2
        if self.step == 1:
            return self._card().contains(pos)
        if self.step == 2:
            return self._block().contains(pos)
        return False

    # -- input -------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if self.finished:
            if self._again.contains(pos.toPoint()):
                self.restart()
            return
        if self.done or not self._hit(pos):
            return
        button = event.button()
        wanted = Qt.MouseButton.RightButton if self.step == 1 else Qt.MouseButton.LeftButton
        if button != wanted:
            return
        if self.step < 2:
            self._complete()
        else:
            self._drag_from = pos - self._block_offset

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        event.accept()  # the right click is handled in mousePressEvent

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if self._drag_from is not None:
            self._block_offset = pos - self._drag_from
        hover = self._hit(pos) or (self.finished and self._again.contains(pos.toPoint()))
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.CursorShape.PointingHandCursor if hover
                           else Qt.CursorShape.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        self._drag_from = None
        block = self._block_home().translated(self._block_offset)
        if self._slot().contains(block.center()):
            self._complete()
        else:
            self._block_offset = QPointF(0, 0)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def _complete(self) -> None:
        self.done_at = time.monotonic()
        self._timer.start()
        self.update()

    def _animate(self) -> None:
        if self.done_at is not None and time.monotonic() - self.done_at >= _ADVANCE_S:
            self.step += 1
            self.done_at = None
            self._block_offset = QPointF(0, 0)
            self._timer.stop()
        self.update()

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.SAND))
        p.drawRoundedRect(QRectF(self.rect()), 22, 22)
        self._paint_progress(p)
        age = time.monotonic() - self.done_at if self.done_at is not None else None
        if self.step == 0:
            self._paint_circle(p, age)
        elif self.step == 1:
            self._paint_card(p, age)
        elif self.step == 2:
            self._paint_blocks(p, age)
        else:
            self._paint_ready(p)
        self._paint_caption(p, age)
        p.end()

    def _paint_progress(self, p: QPainter) -> None:
        x0 = self.width() / 2 - (3 * 28 + 2 * 8) / 2
        for i in range(_STEPS):
            lit = i < self.step or (i == self.step and self.done) or self.finished
            p.setBrush(theme.qcolor(theme.PURPLE if lit else theme.DOT_OFF))
            p.drawRoundedRect(QRectF(x0 + i * 36, 18, 28, 6), 3, 3)

    def _burst(self, p: QPainter, c: QPointF, age: float | None) -> None:
        if age is None or age > _BURST_S:
            return
        t = age / _BURST_S
        p.setPen(QPen(theme.qcolor(theme.PURPLE, 0.8 * (1 - t)), 4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        r = 45 + 97 * t
        p.drawEllipse(c, r, r)

    def _paint_circle(self, p: QPainter, age: float | None) -> None:
        r = self._circle()
        self._burst(p, r.center(), age)
        grow = 4 if (self._hover and not self.done) else 0
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PURPLE if self.done else theme.LIME))
        p.drawEllipse(r.adjusted(-grow, -grow, grow, grow))
        if age is not None:
            paint_check(p, r.center(), 48, theme.LIME, min(1.0, age / 0.4))

    def _paint_card(self, p: QPainter, age: float | None) -> None:
        r = self._card()
        self._burst(p, r.center(), age)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.INK, 0.10))
        p.drawRoundedRect(r.translated(0, 6), 16, 16)
        p.setBrush(theme.qcolor("#FFFFFF"))
        p.drawRoundedRect(r, 16, 16)
        if not self.done:
            p.setBrush(theme.qcolor(theme.PURPLE_TINT if self._hover else theme.LINE))
            for i, w in enumerate((1.0, 0.7, 0.85)):
                p.drawRoundedRect(QRectF(r.left() + 22, r.top() + 26 + i * 20,
                                         (r.width() - 44) * w, 9), 4.5, 4.5)
            return
        p.setFont(theme.text_font(15, 500))
        for i, label in enumerate(("Copy", "Paste", "Share")):
            row = QRectF(r.left() + 8, r.top() + 8 + i * 36, r.width() - 16, 34)
            if i == 0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.qcolor(theme.PURPLE_TINT))
                p.drawRoundedRect(row, 8, 8)
            p.setPen(theme.qcolor(theme.INK))
            p.drawText(row.adjusted(10, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, label)

    def _paint_blocks(self, p: QPainter, age: float | None) -> None:
        slot = self._slot()
        pen = QPen(theme.qcolor(theme.PURPLE if self.done else theme.FAINT), 2)
        if not self.done:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(theme.qcolor(theme.PURPLE_TINT) if self.done else Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(slot.adjusted(1, 1, -1, -1), 18, 18)
        block = self._block()
        self._burst(p, block.center(), age)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PURPLE))
        p.drawRoundedRect(block, 14, 14)
        if age is not None:
            paint_check(p, block.center(), 28, theme.LIME, min(1.0, age / 0.4))

    def _paint_ready(self, p: QPainter) -> None:
        c = self._toy_center()
        c.setY(c.y() - 14)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PURPLE))
        p.drawEllipse(c, 42, 42)
        paint_check(p, c, 40, theme.LIME)
        p.setPen(theme.qcolor(theme.INK))
        p.setFont(theme.display_font(22))
        p.drawText(QRectF(0, c.y() + 54, self.width(), 30), Qt.AlignmentFlag.AlignCenter,
                   "You’re ready")

    def _paint_caption(self, p: QPainter, age: float | None) -> None:
        band = QRectF(0, self.height() - 64, self.width(), 56)
        if self.finished:
            self._again = QRect(0, 0, 120, 44)
            self._again.moveCenter(band.center().toPoint())
            if self._hover:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(theme.qcolor(theme.SAND_HOVER))
                p.drawRoundedRect(QRectF(self._again), 12, 12)
            p.setPen(theme.qcolor(theme.BODY))
            p.setFont(theme.text_font(15, 600))
            p.drawText(self._again, Qt.AlignmentFlag.AlignCenter, "Try again")
            return
        if age is not None:
            p.setPen(theme.qcolor(theme.PURPLE))
            p.setFont(theme.display_font(22))
            rise = max(0.0, 6 * (1 - age / 0.35))
            p.drawText(band.translated(0, rise), Qt.AlignmentFlag.AlignCenter, "Nice!")
            return
        left_open, right_open, label = _INSTRUCTIONS[self.step]
        paint_eyes(p, band.center().x(), band.top() + 10, left_open, right_open, 0.85)
        p.setPen(theme.qcolor(theme.INK))
        p.setFont(theme.display_font(18))
        p.drawText(band.adjusted(0, 22, 0, 0), Qt.AlignmentFlag.AlignHCenter, label)

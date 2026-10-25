"""A hardware problem, shown at the top of the screen until it's fixed.

Floats over every app (and over calibration), never takes focus, and lets clicks through: the
person may be fixing the problem with the mouse.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

from opengaze.consumer import theme
from opengaze.consumer.problems import Problem
from opengaze.desktop.macwin import float_everywhere

BANNER_W = 560
_PAD = 20
_TOP = 44  # below the menu bar
_ACCENT = "#A3262A"  # the setup pages' error red


class WarningBanner(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.problem: Problem | None = None

    def show_problem(self, problem: Problem) -> None:
        self.problem = problem
        rect = banner_rect(QApplication.primaryScreen().geometry(), problem)
        self.setFixedSize(rect.size())
        self.move(rect.topLeft())
        self.update()
        self.show()

    def clear(self) -> None:
        self.problem = None
        self.hide()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        float_everywhere(self, click_through=True)

    def paintEvent(self, event) -> None:  # noqa: N802
        if self.problem is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_problem(p, QRectF(self.rect()), self.problem)
        p.end()


def _detail_rect(card: QRectF) -> QRectF:
    return QRectF(card.left() + _PAD + 40, card.top() + _PAD + 28,
                  card.width() - 2 * _PAD - 40, 200)


def problem_height(problem: Problem, width: int = BANNER_W) -> int:
    """The banner's height for ``problem`` (the detail wraps)."""
    fm = QFontMetrics(theme.text_font(14, 500))
    rect = _detail_rect(QRectF(0, 0, width, 0)).toRect()
    text = fm.boundingRect(rect, Qt.TextFlag.TextWordWrap, problem.detail)
    return _PAD + 28 + text.height() + _PAD


def banner_rect(screen: QRect, problem: Problem) -> QRect:
    """Where the banner goes on ``screen``: top centre, below the menu bar."""
    return QRect(screen.x() + (screen.width() - BANNER_W) // 2, screen.y() + _TOP,
                 BANNER_W, problem_height(problem))


def paint_problem(p: QPainter, rect: QRectF, problem: Problem) -> None:
    """The banner itself, into ``rect`` (also drawn inside the fullscreen calibration)."""
    card = rect.adjusted(1, 1, -1, -1)
    path = QPainterPath()
    path.addRoundedRect(card, 16, 16)
    p.setBrush(theme.qcolor("#FFFFFF"))
    p.setPen(QPen(theme.qcolor(_ACCENT), 2))
    p.drawPath(path)
    # Warning sign: a filled circle with "!".
    c = QPointF(card.left() + _PAD + 14, card.top() + _PAD + 12)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.qcolor(_ACCENT))
    p.drawEllipse(c, 13, 13)
    p.setPen(theme.qcolor("#FFFFFF"))
    p.setFont(theme.display_font(17, 700))
    p.drawText(QRectF(c.x() - 13, c.y() - 13, 26, 26), Qt.AlignmentFlag.AlignCenter, "!")
    p.setPen(theme.qcolor(theme.INK))
    p.setFont(theme.display_font(17, 600))
    p.drawText(QRectF(card.left() + _PAD + 40, card.top() + _PAD - 2,
                      card.width() - 2 * _PAD - 40, 26),
               Qt.AlignmentFlag.AlignVCenter, problem.title)
    p.setPen(theme.qcolor(theme.BODY))
    p.setFont(theme.text_font(14, 500))
    p.drawText(_detail_rect(card), Qt.TextFlag.TextWordWrap, problem.detail)

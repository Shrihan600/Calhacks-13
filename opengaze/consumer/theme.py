"""EyeMouse's look, from the consumer design: colours, type, the logo and shared styles.

Fredoka (display) and Figtree (text) ship in ``fonts/`` under the SIL Open Font License, so
the app looks the same on every Mac.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

log = logging.getLogger("opengaze.consumer.theme")

APP_NAME = "EyeMouse"

PURPLE = "#5512F0"
PURPLE_DARK = "#3D07B8"
PURPLE_TINT = "#E3DAFB"
LIME = "#CCF95E"
CREAM = "#FBF7EE"
INK = "#1C1433"
BODY = "#3E3654"
MUTED = "#625B76"
FAINT = "#B9B1C9"
DOT_OFF = "#CFC7DD"
SAND = "#F3EDE0"
SAND_HOVER = "#F0EADB"
LINE = "#E6DECD"
LAVENDER = "#EEE7F9"
GREEN = "#3FA34D"
AMBER = "#C77700"
NIGHT = "#1C1433"
NIGHT_KEY = "#2E2449"
NIGHT_MOD = "#241C3B"
NIGHT_HOVER = "#3E3260"

_FONT_DIR = Path(__file__).with_name("fonts")
_ASSETS = Path(__file__).with_name("assets")
DISPLAY = "Fredoka"
TEXT = "Figtree"
_FALLBACK = "Helvetica Neue"


def load_fonts() -> None:
    """Register the bundled fonts; missing ones fall back to Helvetica Neue."""
    global DISPLAY, TEXT
    families: set[str] = set()
    for path in _FONT_DIR.glob("*.ttf"):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            log.warning("could not load font %s", path.name)
            continue
        families.update(QFontDatabase.applicationFontFamilies(font_id))
    DISPLAY = "Fredoka" if "Fredoka" in families else _FALLBACK
    TEXT = "Figtree" if "Figtree" in families else _FALLBACK


def display_font(px: int, weight: int = 600) -> QFont:
    f = QFont(DISPLAY)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


def text_font(px: int, weight: int = 500) -> QFont:
    f = QFont(TEXT)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


# The mark: two eyes (one winking) and a smile, on a lime tile.
LOGO_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
<rect width="100" height="100" rx="26" fill="{LIME}"/>
<circle cx="30" cy="38" r="9.5" fill="{PURPLE}"/>
<path d="M48 41 L72 32" stroke="{PURPLE}" stroke-width="12" stroke-linecap="round"/>
<path d="M19 64 Q42 82 66 67" stroke="{PURPLE}" stroke-width="5.5" fill="none"
 stroke-linecap="round"/>
<circle cx="70" cy="62" r="7" fill="{PURPLE}"/>
</svg>"""

# Menu bar version: white mark on a purple tile.
MENU_LOGO_SVG = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 124 124">
<rect width="124" height="124" rx="28" fill="{PURPLE}"/>
<g transform="translate(12 12)">
<circle cx="30" cy="38" r="11" fill="#FFFFFF"/>
<path d="M48 41 L72 32" stroke="#FFFFFF" stroke-width="13" stroke-linecap="round"/>
<path d="M19 64 Q42 82 66 67" stroke="#FFFFFF" stroke-width="7" fill="none"
 stroke-linecap="round"/>
<circle cx="70" cy="62" r="8" fill="#FFFFFF"/>
</g>
</svg>"""


def svg_pixmap(svg: str, width: int, height: int, dpr: float = 2.0) -> QPixmap:
    pm = QPixmap(int(width * dpr), int(height * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(p, QRectF(0, 0, width, height))
    p.end()
    return pm


def app_icon() -> QIcon:
    return QIcon(svg_pixmap(LOGO_SVG, 256, 256))


def menu_bar_icon() -> QIcon:
    return QIcon(svg_pixmap(MENU_LOGO_SVG, 22, 22))


def qcolor(hex_: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_)
    c.setAlphaF(alpha)
    return c


def stylesheet() -> str:
    """Widget styles for the setup and settings windows."""
    return f"""
QWidget#page {{ background: {CREAM}; }}
QLabel {{ color: {INK}; font-family: "{TEXT}"; }}
QLabel[role="title"] {{ font-family: "{DISPLAY}"; font-size: 40px; font-weight: 600; }}
QLabel[role="body"] {{ color: {BODY}; font-size: 17px; }}
QLabel[role="fine"] {{ color: {MUTED}; font-size: 14px; }}
QLabel[role="note"] {{ color: {BODY}; font-size: 14px; font-weight: 600; }}
QLabel[role="error"] {{ color: #A3262A; font-size: 14px; font-weight: 600; }}
QPushButton {{ font-family: "{TEXT}"; }}
QPushButton[kind="primary"] {{
    background: {PURPLE}; color: #FFFFFF; border: 0; border-radius: 14px;
    min-height: 52px; padding: 0 28px; font-size: 17px; font-weight: 600;
}}
QPushButton[kind="primary"]:hover {{ background: {PURPLE_DARK}; }}
QPushButton[kind="primary"]:disabled {{ background: {DOT_OFF}; }}
QPushButton[kind="ghost"] {{
    background: transparent; color: {BODY}; border: 0; border-radius: 14px;
    min-height: 52px; padding: 0 18px; font-size: 16px; font-weight: 600;
}}
QPushButton[kind="ghost"]:hover {{ background: {SAND_HOVER}; color: {INK}; }}
QPushButton[kind="link"] {{
    background: transparent; color: {PURPLE}; border: 0; padding: 0;
    font-size: 14px; font-weight: 600; text-decoration: underline;
}}
QPushButton[kind="link"]:hover {{ color: {PURPLE_DARK}; }}
QPushButton:focus {{ outline: none; }}
QCheckBox {{ color: {INK}; font-family: "{TEXT}"; font-size: 14px; spacing: 10px; }}
QCheckBox::indicator {{
    width: 20px; height: 20px; border-radius: 5px; border: 2px solid {FAINT};
    background: #FFFFFF;
}}
QCheckBox::indicator:checked {{
    background: {PURPLE}; border-color: {PURPLE}; image: url({(_ASSETS / "check.svg").as_posix()});
}}
"""

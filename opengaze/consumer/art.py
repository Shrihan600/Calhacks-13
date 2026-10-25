"""Illustrations for the setup pages, as SVG (drawn with Qt's SVG renderer)."""

from __future__ import annotations

from html import escape

from PySide6.QtCore import QByteArray, QRectF
from PySide6.QtGui import QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from opengaze.consumer import theme
from opengaze.consumer.theme import BODY, CREAM, INK, LAVENDER, LIME, PURPLE

_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">{body}</svg>'
_CURSOR = ('<path d="M240 410 L240 446 L249 438 L256 453 L263 450 L256 435 L268 435 Z" '
           f'fill="{INK}" stroke="{CREAM}" stroke-width="2.5" stroke-linejoin="round"/>')
_MIC = ('<rect x="{x}" y="{y}" width="6" height="11" rx="3" fill="none" stroke="#FFFFFF" '
        'stroke-width="2"/><path d="M{a} {b} a7 7 0 0 0 14 0 M{c} {d} v3" fill="none" '
        'stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>')


def _text(x: float, y: float, s: str, size: int, weight: int = 500, fill: str = INK,
          anchor: str = "start", family: str | None = None) -> str:
    return (f'<text x="{x}" y="{y}" font-family="{family or theme.TEXT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def welcome() -> str:
    """The lime panel: the mark's face, a dotted gaze path and a clicking cursor."""
    body = f"""
<rect width="340" height="536" rx="22" fill="{LIME}"/>
<circle cx="96" cy="128" r="30" fill="{PURPLE}"/>
<path d="M156 138 L234 108" stroke="{PURPLE}" stroke-width="38" stroke-linecap="round"/>
<path d="M60 220 Q 140 280 222 232" stroke="{PURPLE}" stroke-width="16" fill="none"
 stroke-linecap="round"/>
<circle cx="236" cy="222" r="22" fill="{PURPLE}"/>
<path d="M96 170 C 110 300, 190 360, 236 410" stroke="{INK}" stroke-width="3"
 stroke-dasharray="2 10" stroke-linecap="round" fill="none" opacity="0.55"/>
<circle cx="244" cy="424" r="34" fill="none" stroke="{PURPLE}" stroke-width="6" opacity="0.18"/>
<path d="M244 424 m0 -34 a34 34 0 1 1 -29.4 51" fill="none" stroke="{PURPLE}" stroke-width="6"
 stroke-linecap="round"/>
{_CURSOR}"""
    return _SVG.format(w=340, h=536, body=body)


def camera_prompt() -> str:
    """The macOS camera prompt, with Allow picked out."""
    logo = theme.LOGO_SVG.split(">", 1)[1].rsplit("</svg>", 1)[0]
    body = f"""
<rect width="330" height="330" rx="22" fill="{LAVENDER}"/>
<rect x="33" y="72" width="264" height="190" rx="14" fill="#28145A" opacity="0.10"
 transform="translate(0 8)"/>
<rect x="33" y="72" width="264" height="190" rx="14" fill="#FFFFFF"/>
<g transform="translate(141 92) scale(0.48)">{logo}</g>
{_text(165, 166, "“EyeMouse” would like to", 13, 700, anchor="middle")}
{_text(165, 183, "access the camera.", 13, 700, anchor="middle")}
{_text(165, 203, "Used to follow your eyes and move the pointer.", 10, 400, "#5C5866", "middle")}
<rect x="53" y="220" width="108" height="26" rx="7" fill="#ECEAF0"/>
{_text(107, 237, "Don’t Allow", 12, 500, "#2A2732", "middle")}
<rect x="164" y="215" width="118" height="36" rx="11" fill="{PURPLE}"/>
<rect x="166" y="217" width="114" height="32" rx="9" fill="{LIME}"/>
<rect x="169" y="220" width="108" height="26" rx="7" fill="{PURPLE}"/>
{_text(223, 237, "Allow", 12, 600, "#FFFFFF", "middle")}"""
    return _SVG.format(w=330, h=330, body=body)


def accessibility_settings(app_name: str) -> str:
    """System Settings › Privacy & Security › Accessibility, with our row switched on."""
    logo = theme.LOGO_SVG.split(">", 1)[1].rsplit("</svg>", 1)[0]
    side = ["Appearance", "Displays", "Privacy & Security", "Desktop & Dock", "Keyboard"]
    items = []
    for i, label in enumerate(side):
        y = 40 + i * 26
        if i == 2:
            items.append(f'<rect x="8" y="{y - 14}" width="104" height="21" rx="6" '
                         f'fill="{PURPLE}"/>')
            items.append(_text(16, y, label, 11, 600, "#FFFFFF"))
        else:
            items.append(_text(16, y, label, 11, 400, "#4A4656"))
    rows = []
    for i, w in enumerate((70, 54)):
        y = 158 + i * 40
        rows.append(f'<rect x="146" y="{y}" width="22" height="22" rx="6" fill="#E2DFE7"/>'
                    f'<rect x="178" y="{y + 8}" width="{w}" height="7" rx="4" fill="#E2DFE7"/>'
                    f'<rect x="320" y="{y + 2}" width="30" height="18" rx="9" fill="#DEDBE3"/>'
                    f'<circle cx="329" cy="{y + 11}" r="7" fill="#FFFFFF"/>')
    body = f"""
<rect x="0" y="6" width="380" height="320" rx="12" fill="#28145A" opacity="0.08"/>
<rect width="380" height="320" rx="12" fill="#FFFFFF"/>
<path d="M12 0 H120 V320 H12 A12 12 0 0 1 0 308 V12 A12 12 0 0 1 12 0 Z" fill="#F4F2F6"/>
{''.join(items)}
{_text(136, 30, "‹ Accessibility", 13, 700, "#1F1C26")}
{_text(136, 50, "Allow the apps below to control your computer.", 10, 400, "#6A6575")}
<rect x="136" y="64" width="228" height="170" rx="8" fill="#F7F6F9"/>
<rect x="130" y="66" width="240" height="52" rx="12" fill="{LIME}"/>
<rect x="134" y="70" width="232" height="44" rx="9" fill="{PURPLE}"/>
<rect x="136" y="72" width="228" height="40" rx="8" fill="#FFFFFF"/>
<g transform="translate(146 81) scale(0.22)">{logo}</g>
{_text(178, 97, app_name[:24], 12, 600, "#1F1C26")}
<rect x="320" y="83" width="30" height="18" rx="9" fill="{PURPLE}"/>
<circle cx="341" cy="92" r="7" fill="#FFFFFF"/>
{''.join(rows)}
<rect x="136" y="246" width="22" height="20" rx="4" fill="#F1EFF4"/>
<rect x="159" y="246" width="22" height="20" rx="4" fill="#F1EFF4"/>
{_text(147, 261, "+", 13, 500, "#4A4656", "middle")}
{_text(170, 261, "−", 13, 500, "#4A4656", "middle")}"""
    return _SVG.format(w=380, h=320, body=body)


def voice_typing() -> str:
    """A chat reply being spoken: the text field, the live mic and the Listening pill."""
    bars = "".join(
        f'<rect x="{70 + i * 6}" y="{246 - h / 2}" width="3" height="{h}" rx="1.5" '
        f'fill="{PURPLE}"/>' for i, h in enumerate((8, 14, 10, 16, 6))
    )
    body = f"""
<rect width="330" height="330" rx="22" fill="{LIME}"/>
<rect x="30" y="60" width="270" height="144" rx="16" fill="{INK}" opacity="0.14"
 transform="translate(0 8)"/>
<rect x="30" y="60" width="270" height="144" rx="16" fill="#FFFFFF"/>
<circle cx="55" cy="85" r="11" fill="#E3DAFB"/>
{_text(74, 90, "Sam", 13, 600)}
<rect x="44" y="108" width="118" height="30" rx="14" fill="#EFEBF4"/>
{_text(56, 128, "Need anything?", 13, 400)}
<rect x="45" y="149" width="240" height="42" rx="21" fill="none" stroke="{PURPLE}"
 stroke-width="2"/>
{_text(62, 175, "Just the big umbrella", 14, 400)}
<rect x="204" y="162" width="2" height="16" fill="{PURPLE}"/>
<circle cx="262" cy="170" r="20" fill="{PURPLE}" opacity="0.18"/>
<circle cx="262" cy="170" r="16" fill="{PURPLE}"/>
{_MIC.format(x=259, y=162, a=255, b=168, c=262, d=175)}
<rect x="56" y="228" width="128" height="36" rx="12" fill="{CREAM}"/>
{bars}
{_text(108, 251, "Listening", 13, 500, BODY)}
{_text(165, 290, "Select any text box, then turn on", 13, 600, "#3B2A66", "middle")}
{_text(165, 308, "the mic to talk instead of type.", 13, 600, "#3B2A66", "middle")}"""
    return _SVG.format(w=330, h=330, body=body)


class SvgArt(QWidget):
    """A fixed-size illustration."""

    def __init__(self, svg: str, width: int, height: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._renderer = QSvgRenderer(QByteArray(svg.encode()), self)

    def set_svg(self, svg: str) -> None:
        self._renderer.load(QByteArray(svg.encode()))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._renderer.render(p, QRectF(0, 0, self.width(), self.height()))
        p.end()

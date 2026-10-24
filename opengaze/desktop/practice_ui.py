"""Gaze target practice game.

Five coloured circles appear on screen.  Look at one and blink to hit it;
a new target spawns immediately.  60-second round, then score screen.

Space = start / play again   Esc = back to test window.
"""

from __future__ import annotations

import logging
import math
import random
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from opengaze.cameras.types import CameraRole
from opengaze.cameras.runtime import CaptureRuntime
from opengaze.config.loader import ProfileReport
from opengaze.config.schema import AppConfig
from opengaze.control.blink import BlinkClickDetector
from opengaze.control.filter import OneEuroFilter
from opengaze.control.safety import ConfidenceGate
from opengaze.vision.feature_extractor import build_mode_a
from opengaze.vision.preprocessing import FrameNormalizer

log = logging.getLogger("opengaze.practice_ui")

_GAME_S      = 60.0
_N_TARGETS   = 5
_RADIUS_NORM = 0.055   # fraction of min(w, h)
_MIN_DIST    = 0.22    # normalised distance between target centres
_MARGIN      = 0.13    # keep away from screen edges

_COLORS = [
    QColor("#e84393"), QColor("#f0b429"), QColor("#2e9e5b"),
    QColor("#00d4ff"), QColor("#9b59b6"), QColor("#e74c3c"),
    QColor("#1abc9c"), QColor("#e67e22"),
]

_BG   = QColor("#0d0f12")
_TEXT = QColor("#c8c8c8")
_DIM  = QColor("#555")


class _Target:
    def __init__(self, x: float, y: float, color: QColor) -> None:
        self.x, self.y = x, y
        self.color = color
        self.born = time.monotonic()
        self.hit_at: float = 0.0

    def radius_px(self, w: int, h: int) -> int:
        return int(_RADIUS_NORM * min(w, h))

    @property
    def alive(self) -> bool:
        return self.hit_at == 0.0


class PracticeWindow(QWidget):
    def __init__(
        self,
        runtime: CaptureRuntime,
        report: ProfileReport,
        config: AppConfig,
        schema_id: str,
        model: object,
        detector: object | None = None,
    ) -> None:
        super().__init__()
        assert report.profile is not None
        self._runtime = runtime
        self._model = model
        self._head_ids = {
            cid for cid, cam in report.profile.enabled_cameras.items()
            if cam.role is CameraRole.HEAD
        }
        self._normalizer = FrameNormalizer()
        if detector is not None:
            self._detector = detector
            self._owns_detector = False
        else:
            self._detector = _make_detector(config)
            self._owns_detector = True

        self._fx = OneEuroFilter(min_cutoff=0.8, beta=0.07)
        self._fy = OneEuroFilter(min_cutoff=0.8, beta=0.07)
        self._conf = ConfidenceGate(min_confidence=0.25)
        self._blink = BlinkClickDetector(long_blink_ms=400)

        self._gaze_x: float = 0.5
        self._gaze_y: float = 0.5
        self._blink_prog: float = 0.0

        self._state = "idle"   # idle | playing | done
        self._targets: list[_Target] = []
        self._score = 0
        self._misses = 0
        self._start_time: float = 0.0
        self._miss_flash_until: float = 0.0

        self.setWindowTitle("OpenGaze — Practice")
        self.setStyleSheet("background: #0d0f12;")
        self.showFullScreen()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def _start(self) -> None:
        self._score = 0
        self._misses = 0
        self._targets = []
        self._start_time = time.monotonic()
        self._state = "playing"
        for _ in range(_N_TARGETS):
            self._spawn()

    def _spawn(self) -> None:
        for _ in range(120):
            x = random.uniform(_MARGIN, 1.0 - _MARGIN)
            y = random.uniform(_MARGIN + 0.08, 1.0 - _MARGIN)
            if all(math.hypot(x - t.x, y - t.y) > _MIN_DIST for t in self._targets if t.alive):
                self._targets.append(_Target(x, y, random.choice(_COLORS)))
                return

    def _on_click(self) -> None:
        w, h = self.width(), self.height()
        gx, gy = self._gaze_x * w, self._gaze_y * h
        for t in self._targets:
            if not t.alive:
                continue
            if math.hypot(gx - t.x * w, gy - t.y * h) <= t.radius_px(w, h):
                t.hit_at = time.monotonic()
                self._score += 1
                return
        self._misses += 1
        self._miss_flash_until = time.monotonic() + 0.22

    # ------------------------------------------------------------------
    # Gaze tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        for cid in self._head_ids:
            frame = self._runtime.manager.latest(cid)
            if frame is None:
                continue
            frame = self._normalizer(frame)
            if self._detector is None:
                continue
            try:
                head, eyes = self._detector.detect(frame)
            except Exception:
                continue

            left_eye  = next((e for e in eyes if e.eye == "left"),  None)
            right_eye = next((e for e in eyes if e.eye == "right"), None)
            br = self._blink.update(
                bool(left_eye  and left_eye.blink),
                bool(right_eye and right_eye.blink),
            )
            self._blink_prog = br.blink_progress

            if self._model is not None and not br.any_blink:
                fv = build_mode_a(head, left_eye, right_eye)
                if self._conf.allow(fv.valid.mean()):
                    try:
                        s = self._model.predict(fv)
                        t = time.monotonic()
                        self._gaze_x = self._fx(s.screen_x_norm, t)
                        self._gaze_y = self._fy(s.screen_y_norm, t)
                    except Exception:
                        pass

            if br.clicked and self._state == "playing":
                self._on_click()

        now = time.monotonic()
        if self._state == "playing":
            # Remove hit targets, spawn replacements
            expired = [t for t in self._targets if not t.alive and now - t.hit_at > 0.28]
            for t in expired:
                self._targets.remove(t)
                self._spawn()
            if now - self._start_time >= _GAME_S:
                self._state = "done"

        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), _BG)
        if self._state == "idle":
            self._draw_idle(p, w, h)
        elif self._state == "playing":
            self._draw_game(p, w, h)
        else:
            self._draw_done(p, w, h)
        p.end()

    def _draw_idle(self, p: QPainter, w: int, h: int) -> None:
        self._center(p, w, h,
            "Gaze Target Practice\n\n"
            "Coloured circles appear on screen.\n"
            "Look at one and hold a blink (400 ms) to hit it.\n\n"
            "Space = start  •  Esc = back", size=24)

    def _draw_game(self, p: QPainter, w: int, h: int) -> None:
        now = time.monotonic()

        for t in self._targets:
            cx, cy = int(t.x * w), int(t.y * h)
            r = t.radius_px(w, h)
            if not t.alive:
                age = now - t.hit_at
                alpha = max(0, int(255 * (1.0 - age / 0.28)))
                c = QColor(t.color); c.setAlpha(alpha)
                p.setBrush(c); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(cx - r * 2, cy - r * 2, r * 4, r * 4)
            else:
                pulse = 0.5 + 0.5 * math.sin((now - t.born) * 2.8)
                outer = r + int(7 * pulse)
                ring = QColor(t.color); ring.setAlpha(70)
                p.setBrush(ring); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(cx - outer, cy - outer, outer * 2, outer * 2)
                p.setBrush(t.color)
                p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        if now < self._miss_flash_until:
            p.fillRect(0, 0, w, h, QColor(214, 69, 65, 45))

        # Gaze dot + blink ring
        gx, gy = int(self._gaze_x * w), int(self._gaze_y * h)
        if self._blink_prog > 0:
            pen = QPen(QColor("#f0b429"), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawArc(gx - 22, gy - 22, 44, 44,
                      90 * 16, int(-self._blink_prog * 360 * 16))
        p.setBrush(QColor(255, 255, 255, 210))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(gx - 5, gy - 5, 10, 10)

        # HUD
        elapsed = now - self._start_time
        remaining = max(0.0, _GAME_S - elapsed)
        p.setFont(QFont("Helvetica Neue", 24, QFont.Weight.Bold))
        p.setPen(_TEXT)
        p.drawText(32, 46, f"Hits: {self._score}")
        time_color = QColor("#e84393") if remaining < 10 else _TEXT
        p.setPen(time_color)
        p.drawText(w // 2 - 40, 46, f"{remaining:.0f}s")
        p.setPen(_TEXT)
        p.drawText(w - 220, 46, f"Misses: {self._misses}")
        p.setFont(QFont("Menlo", 11))
        p.setPen(_DIM)
        p.drawText(32, h - 14, "Hold blink 400ms to hit  •  Esc = back")

    def _draw_done(self, p: QPainter, w: int, h: int) -> None:
        total = self._score + self._misses
        acc = f"{int(100 * self._score / total)}%" if total else "—"
        self._center(p, w, h,
            f"Time's up!\n\n"
            f"Hits: {self._score}    Misses: {self._misses}    Accuracy: {acc}\n\n"
            "Space = play again  •  Esc = back", size=26)

    def _center(self, p: QPainter, w: int, h: int, text: str, size: int = 22) -> None:
        p.setPen(_TEXT)
        p.setFont(QFont("Helvetica Neue", size))
        p.drawText(
            int(w * 0.1), int(h * 0.2), int(w * 0.8), int(h * 0.6),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter |
            Qt.TextFlag.TextWordWrap,
            text,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        k = event.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space and self._state in ("idle", "done"):
            self._start()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        if self._owns_detector and self._detector is not None:
            self._detector.close()
        super().closeEvent(event)


def _make_detector(config: AppConfig) -> object | None:
    try:
        from opengaze.vision.mediapipe_detector import FaceLandmarkDetector, default_model_path
        path = config.vision.face_landmarker_model_path or default_model_path()
        if path.exists():
            return FaceLandmarkDetector(path)
    except Exception:
        pass
    return None

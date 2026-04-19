"""Snake board drawn with QPainter; optional Q-value overlay."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from env.config import GRID_COLS, GRID_ROWS
from env.snake_core import Dir


CELL = 20
BG = QColor(20, 24, 32)
GRID = QColor(55, 62, 75)
SNAKE = QColor(80, 200, 120)
HEAD = QColor(120, 255, 160)
FOOD = QColor(220, 90, 90)
DEAD_HEAD = QColor(200, 80, 80)
BAR_BG = QColor(40, 44, 55)
BAR_FG = QColor(100, 180, 255)


class BoardWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snake: list[tuple[int, int]] = []
        self._food: tuple[int, int] = (0, 0)
        self._alive = True
        self._q: np.ndarray | None = None
        self._direction: Dir | None = None
        gw = GRID_COLS * CELL
        gh = GRID_ROWS * CELL
        self._overlay_h = 72
        self.setMinimumSize(gw, gh + self._overlay_h)
        self.setMaximumWidth(gw + 4)

    def set_state(
        self,
        snake: list[tuple[int, int]],
        food: tuple[int, int],
        alive: bool,
        direction: Dir | None = None,
    ) -> None:
        self._snake = list(snake)
        self._food = tuple(food)
        self._alive = alive
        self._direction = direction
        self.update()

    def set_q_values(self, q: np.ndarray | None) -> None:
        self._q = np.asarray(q, dtype=np.float64).copy() if q is not None else None
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(GRID_COLS * CELL, GRID_ROWS * CELL + self._overlay_h)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        gw = GRID_COLS * CELL
        gh = GRID_ROWS * CELL

        p.fillRect(0, 0, gw, gh, BG)
        p.setPen(QPen(GRID, 1))
        for x in range(0, gw + 1, CELL):
            p.drawLine(x, 0, x, gh)
        for y in range(0, gh + 1, CELL):
            p.drawLine(0, y, gw, y)

        for i, (cx, cy) in enumerate(self._snake):
            rect = QRectF(cx * CELL + 1, cy * CELL + 1, CELL - 2, CELL - 2)
            col = HEAD if i == 0 else SNAKE
            if i == 0 and not self._alive:
                col = DEAD_HEAD
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(rect, 4, 4)

        fx, fy = self._food
        p.setBrush(FOOD)
        p.drawRoundedRect(QRectF(fx * CELL + 1, fy * CELL + 1, CELL - 2, CELL - 2), 4, 4)

        # Q overlay
        oy = gh + 4
        p.fillRect(0, gh, gw, self._overlay_h, QColor(14, 18, 26))
        if self._q is not None and self._q.size == 3:
            labels = ("straight", "right", "left")
            qmin = float(np.min(self._q))
            qmax = float(np.max(self._q))
            span = max(1e-6, qmax - qmin)
            bw = (gw - 20) // 3
            for i in range(3):
                x0 = 10 + i * (bw + 6)
                norm = (float(self._q[i]) - qmin) / span
                h = max(4, int(norm * (self._overlay_h - 28)))
                p.setPen(QPen(GRID, 1))
                p.setBrush(BAR_BG)
                p.drawRect(x0, oy + 18, bw, self._overlay_h - 24)
                p.setBrush(BAR_FG)
                p.drawRect(x0, oy + 18 + (self._overlay_h - 24 - h), bw, h)
                p.setPen(QColor(220, 224, 230))
                p.setFont(QFont("Consolas", 9))
                p.drawText(QRectF(x0, oy, bw, 16), Qt.AlignmentFlag.AlignCenter, labels[i])
                p.drawText(
                    QRectF(x0, oy + self._overlay_h - 22, bw, 14),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{float(self._q[i]):.2f}",
                )
        else:
            p.setPen(QColor(140, 150, 170))
            p.setFont(QFont("Consolas", 10))
            p.drawText(QRectF(0, oy, gw, self._overlay_h), Qt.AlignmentFlag.AlignCenter, "Q overlay (play mode)")

"""Second window: live training curves (pyqtgraph)."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QWidget

pg.setConfigOptions(antialias=True)


class CurvesWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training curves")
        self.resize(720, 640)

        cw = pg.GraphicsLayoutWidget(show=False)
        self.setCentralWidget(cw)

        self.plot_return = cw.addPlot(row=0, col=0, title="Episode return")
        self.plot_return.showGrid(x=True, y=True, alpha=0.3)
        self.curve_return = self.plot_return.plot(pen=pg.mkPen("#6cf", width=2))

        cw.nextRow()
        self.plot_score = cw.addPlot(row=1, col=0, title="Score + mean(30)")
        self.plot_score.showGrid(x=True, y=True, alpha=0.3)
        self.curve_score = self.plot_score.plot(pen=pg.mkPen("#8e8", width=2), name="score")
        self.curve_mean30 = self.plot_score.plot(pen=pg.mkPen("#fa8", width=2, style=Qt.PenStyle.DashLine), name="mean30")

        cw.nextRow()
        self.plot_loss = cw.addPlot(row=2, col=0, title="Loss (train step) + EMA")
        self.plot_loss.showGrid(x=True, y=True, alpha=0.3)
        self.plot_loss.setLogMode(False, False)
        self.curve_loss = self.plot_loss.plot(pen=pg.mkPen("#aaf", width=1))
        self.curve_loss_ema = self.plot_loss.plot(pen=pg.mkPen("#f8f", width=2))

        cw.nextRow()
        self.plot_eps = cw.addPlot(row=3, col=0, title="Epsilon vs env frames")
        self.plot_eps.showGrid(x=True, y=True, alpha=0.3)
        self.curve_eps = self.plot_eps.plot(pen=pg.mkPen("#fc8", width=2))

        self._episodes: list[int] = []
        self._returns: list[float] = []
        self._scores: list[int] = []
        self._mean30: list[float] = []
        self._loss_x: list[int] = []
        self._loss_y: list[float] = []
        self._loss_ema: list[float] = []
        self._eps_x: list[int] = []
        self._eps_y: list[float] = []
        self._ema_loss = 0.0
        self._ema_beta = 0.02

    def clear(self) -> None:
        self._episodes.clear()
        self._returns.clear()
        self._scores.clear()
        self._mean30.clear()
        self._loss_x.clear()
        self._loss_y.clear()
        self._loss_ema.clear()
        self._eps_x.clear()
        self._eps_y.clear()
        self._ema_loss = 0.0
        self._redraw_all()

    def on_metrics(self, d: dict) -> None:
        ep = int(d["episode"])
        self._episodes.append(ep)
        self._returns.append(float(d["return"]))
        self._scores.append(int(d["score"]))
        m = d.get("mean_score_30")
        if m is not None:
            self._mean30.append(float(m))
        elif self._mean30:
            self._mean30.append(self._mean30[-1])
        else:
            self._mean30.append(float(d["score"]))

        cap_ep = 20_000
        if len(self._episodes) > cap_ep:
            self._episodes = self._episodes[-cap_ep:]
            self._returns = self._returns[-cap_ep:]
            self._scores = self._scores[-cap_ep:]
            self._mean30 = self._mean30[-cap_ep:]

        self.curve_return.setData(self._episodes, self._returns)
        self.curve_score.setData(self._episodes, self._scores)
        self.curve_mean30.setData(self._episodes, self._mean30)

    def on_loss(self, d: dict) -> None:
        ts = int(d["train_step"])
        lo = float(d["loss"])
        self._loss_x.append(ts)
        self._loss_y.append(lo)
        if self._ema_loss == 0.0:
            self._ema_loss = lo
        else:
            self._ema_loss = self._ema_beta * lo + (1 - self._ema_beta) * self._ema_loss
        self._loss_ema.append(self._ema_loss)
        cap = 50_000
        if len(self._loss_x) > cap:
            self._loss_x = self._loss_x[-cap:]
            self._loss_y = self._loss_y[-cap:]
            self._loss_ema = self._loss_ema[-cap:]
        self.curve_loss.setData(self._loss_x, self._loss_y)
        self.curve_loss_ema.setData(self._loss_x, self._loss_ema)

    def on_epsilon_frame(self, frames: int, eps: float) -> None:
        self._eps_x.append(frames)
        self._eps_y.append(eps)
        cap = 8000
        if len(self._eps_x) > cap:
            self._eps_x = self._eps_x[-cap:]
            self._eps_y = self._eps_y[-cap:]
        self.curve_eps.setData(self._eps_x, self._eps_y)

    def _redraw_all(self) -> None:
        self.curve_return.setData(self._episodes, self._returns)
        self.curve_score.setData(self._episodes, self._scores)
        self.curve_mean30.setData(self._episodes, self._mean30)
        self.curve_loss.setData(self._loss_x, self._loss_y)
        self.curve_loss_ema.setData(self._loss_x, self._loss_ema)
        self.curve_eps.setData(self._eps_x, self._eps_y)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.hide()
        event.ignore()
        par = self.parentWidget()
        if par is not None:
            act = getattr(par, "act_curves", None)
            if act is not None:
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)

"""Train tab: sliders, controls, board, summary."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from agent.dqn import DQNAgent, DQNConfig, default_config
from env.snake_env import RewardConfig
from gui.board_widget import BoardWidget
from gui.worker import TrainWorker


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _lr_from_slider(v: int) -> float:
    lo, hi = 1e-4, 1e-2
    t = v / 1000.0
    return float(10 ** (math.log10(lo) + t * (math.log10(hi) - math.log10(lo))))


def _lr_to_slider(lr: float) -> int:
    lo, hi = 1e-4, 1e-2
    t = (math.log10(max(lo, min(hi, lr))) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return int(round(max(0.0, min(1.0, t)) * 1000))


class SummaryDialog(QDialog):
    def __init__(self, summary: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training finished")

        lay = QVBoxLayout(self)
        text = []
        for k, v in summary.items():
            if k == "agent":
                continue
            text.append(f"{k}: {v}")
        lab = QLabel("\n".join(text))
        lab.setWordWrap(True)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(lab)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        lay.addWidget(bb)
        bb.accepted.connect(self.accept)


class TrainPanel(QWidget):
    """Emits metrics_out / loss_out / frame_out for CurvesWindow."""

    metrics_out = pyqtSignal(dict)
    loss_out = pyqtSignal(dict)
    frame_out = pyqtSignal(dict)
    training_started = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: TrainWorker | None = None
        self._running = False

        dc = default_config()

        left = QVBoxLayout()
        hyp = QGroupBox("Hyperparameters (lock while training)")
        fl = QFormLayout(hyp)

        self.sp_episodes = QSpinBox()
        self.sp_episodes.setRange(1, 500_000)
        self.sp_episodes.setValue(2000)
        fl.addRow("Episodes", self.sp_episodes)

        self.sp_render = QSpinBox()
        self.sp_render.setRange(1, 200)
        self.sp_render.setValue(4)
        fl.addRow("Board update every N env steps", self.sp_render)

        self.sl_lr = QSlider(Qt.Orientation.Horizontal)
        self.sl_lr.setRange(0, 1000)
        self.sl_lr.setValue(_lr_to_slider(dc.lr))
        self.lb_lr = QLabel(f"{_lr_from_slider(self.sl_lr.value()):.2e}")
        self.sl_lr.valueChanged.connect(lambda _: self.lb_lr.setText(f"{_lr_from_slider(self.sl_lr.value()):.2e}"))
        lr_row = QHBoxLayout()
        lr_row.addWidget(self.sl_lr)
        lr_row.addWidget(self.lb_lr)
        w_lr = QWidget()
        w_lr.setLayout(lr_row)
        fl.addRow("Learning rate (log)", w_lr)

        self.sl_gamma = QSlider(Qt.Orientation.Horizontal)
        self.sl_gamma.setRange(800, 999)
        self.sl_gamma.setValue(int(dc.gamma * 1000))
        self.lb_gamma = QLabel(f"{self.sl_gamma.value() / 1000:.3f}")
        self.sl_gamma.valueChanged.connect(lambda v: self.lb_gamma.setText(f"{v / 1000:.3f}"))
        g_row = QHBoxLayout()
        g_row.addWidget(self.sl_gamma)
        g_row.addWidget(self.lb_gamma)
        wg = QWidget()
        wg.setLayout(g_row)
        fl.addRow("Gamma", wg)

        self.cb_batch = QComboBox()
        for b in (16, 32, 64, 128, 256):
            self.cb_batch.addItem(str(b), b)
        self.cb_batch.setCurrentText(str(dc.batch))
        fl.addRow("Batch size", self.cb_batch)

        self.sp_buf = QSpinBox()
        self.sp_buf.setRange(2000, 500_000)
        self.sp_buf.setSingleStep(1000)
        self.sp_buf.setValue(dc.buffer_cap)
        fl.addRow("Replay buffer cap", self.sp_buf)

        self.sp_warm = QSpinBox()
        self.sp_warm.setRange(100, 50_000)
        self.sp_warm.setValue(dc.warmup)
        fl.addRow("Warmup (min buffer)", self.sp_warm)

        self.sp_target = QSpinBox()
        self.sp_target.setRange(50, 20_000)
        self.sp_target.setValue(dc.target_sync)
        fl.addRow("Target sync every N grads", self.sp_target)

        self.sp_eps_decay = QSpinBox()
        self.sp_eps_decay.setRange(5000, 2_000_000)
        self.sp_eps_decay.setSingleStep(5000)
        self.sp_eps_decay.setValue(dc.eps_decay_steps)
        fl.addRow("Epsilon decay steps", self.sp_eps_decay)

        self.d_eps_end = QDoubleSpinBox()
        self.d_eps_end.setRange(0.0, 0.5)
        self.d_eps_end.setDecimals(4)
        self.d_eps_end.setValue(dc.eps_end)
        fl.addRow("Epsilon end", self.d_eps_end)

        left.addWidget(hyp)

        rew = QGroupBox("Rewards (env)")
        rfl = QFormLayout(rew)
        self.d_r_eat = QDoubleSpinBox()
        self.d_r_eat.setRange(0.1, 100.0)
        self.d_r_eat.setDecimals(3)
        self.d_r_eat.setValue(10.0)
        self.d_r_eat.setSingleStep(0.5)
        rfl.addRow("Eat (food)", self.d_r_eat)

        self.d_r_death = QDoubleSpinBox()
        self.d_r_death.setRange(-100.0, -0.01)
        self.d_r_death.setDecimals(3)
        self.d_r_death.setValue(-10.0)
        self.d_r_death.setSingleStep(0.5)
        rfl.addRow("Death (wall/self)", self.d_r_death)

        self.d_r_step = QDoubleSpinBox()
        self.d_r_step.setRange(-2.0, 0.0)
        self.d_r_step.setDecimals(5)
        self.d_r_step.setValue(-0.005)
        self.d_r_step.setSingleStep(0.001)
        rfl.addRow("Each step (living cost)", self.d_r_step)

        left.addWidget(rew)

        self.chk_autosave = QCheckBox("Auto-save without asking (uses models/ timestamp name)")
        left.addWidget(self.chk_autosave)

        self.lbl_curves = QLabel(
            "Training curves: open automatically when you press Start, or use View → Training curves."
        )
        self.lbl_curves.setWordWrap(True)
        self.lbl_curves.setStyleSheet("color: #8ab;")
        left.addWidget(self.lbl_curves)

        row_btn = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_pause)
        row_btn.addWidget(self.btn_stop)
        left.addLayout(row_btn)

        self.lbl_hud = QLabel("Idle")
        self.lbl_hud.setWordWrap(True)
        self.lbl_hud.setMinimumWidth(200)
        left.addWidget(self.lbl_hud)
        left.addStretch()

        self.board = BoardWidget()
        self.board.set_q_values(None)

        root = QHBoxLayout(self)
        root.addLayout(left, stretch=0)
        root.addWidget(self.board, stretch=0)

        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self._stop)

    def _rewards_from_ui(self) -> RewardConfig:
        return RewardConfig(
            r_eat=float(self.d_r_eat.value()),
            r_death=float(self.d_r_death.value()),
            r_step=float(self.d_r_step.value()),
        )

    def _cfg_from_ui(self) -> DQNConfig:
        return DQNConfig(
            lr=_lr_from_slider(self.sl_lr.value()),
            gamma=self.sl_gamma.value() / 1000.0,
            batch=int(self.cb_batch.currentData()),
            buffer_cap=int(self.sp_buf.value()),
            warmup=int(self.sp_warm.value()),
            target_sync=int(self.sp_target.value()),
            eps_start=1.0,
            eps_end=float(self.d_eps_end.value()),
            eps_decay_steps=int(self.sp_eps_decay.value()),
        )

    def _set_controls_enabled(self, on: bool) -> None:
        self.sp_episodes.setEnabled(on)
        self.sp_render.setEnabled(on)
        self.sl_lr.setEnabled(on)
        self.sl_gamma.setEnabled(on)
        self.cb_batch.setEnabled(on)
        self.sp_buf.setEnabled(on)
        self.sp_warm.setEnabled(on)
        self.sp_target.setEnabled(on)
        self.sp_eps_decay.setEnabled(on)
        self.d_eps_end.setEnabled(on)
        self.d_r_eat.setEnabled(on)
        self.d_r_death.setEnabled(on)
        self.d_r_step.setEnabled(on)

    def _start(self) -> None:
        if self._running:
            return
        cfg = self._cfg_from_ui()
        rewards = self._rewards_from_ui()
        episodes = int(self.sp_episodes.value())
        render_every = int(self.sp_render.value())

        self.training_started.emit()
        self._set_controls_enabled(False)
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("Pause")

        self._thread = QThread()
        self._worker = TrainWorker(cfg, episodes, render_every, rewards=rewards)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run_training)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)

        self._worker.metrics.connect(self._on_metrics)
        self._worker.metrics.connect(self.metrics_out.emit)
        self._worker.loss_step.connect(self._on_loss)
        self._worker.loss_step.connect(self.loss_out.emit)
        self._worker.frame.connect(self._on_frame)
        self._worker.frame.connect(self.frame_out.emit)

        self._thread.start()

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _toggle_pause(self) -> None:
        if not self._worker:
            return
        if self.btn_pause.text() == "Pause":
            self._worker.set_pause(True)
            self.btn_pause.setText("Resume")
        else:
            self._worker.set_pause(False)
            self.btn_pause.setText("Pause")

    def _stop(self) -> None:
        if self._worker:
            self._worker.request_stop()

    def _on_metrics(self, d: dict) -> None:
        self.lbl_hud.setText(
            f"Ep {d['episode']}  score={d['score']}  ret={d['return']:.2f}  "
            f"eps={d['epsilon']:.4f}  buf={d['buffer']}"
        )

    def _on_loss(self, d: dict) -> None:
        pass

    def _on_frame(self, d: dict) -> None:
        self.board.set_state(d["snake"], d["food"], d["alive"], None)
        self.lbl_hud.setText(
            f"Ep {d['episode']} step {d['step_in_ep']}  score={d['score']}  "
            f"ret={d['ep_return']:.2f}  eps={d['epsilon']:.4f}  buf={d['buffer']}"
        )

    def _on_worker_finished(self, summary: dict[str, Any]) -> None:
        self._running = False
        self._set_controls_enabled(True)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)

        agent: DQNAgent | None = summary.get("agent")

        dlg = SummaryDialog(summary, self)
        dlg.exec()

        if agent is not None:
            if self.chk_autosave.isChecked():
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                path = MODELS_DIR / f"dqn_{datetime.now():%Y%m%d_%H%M%S}_s{summary.get('peak_score', 0)}.pt"
                agent.save(
                    path,
                    meta={"autosave": True, "summary": {k: v for k, v in summary.items() if k != "agent"}},
                )
            else:
                ans = QMessageBox.question(
                    self,
                    "Save model",
                    "Save trained policy weights to a .pt file?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    path, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save model",
                        str(MODELS_DIR / f"dqn_{datetime.now():%Y%m%d_%H%M%S}.pt"),
                        "PyTorch (*.pt)",
                    )
                    if path:
                        agent.save(path, meta={"from": "save_prompt", "summary": {k: v for k, v in summary.items() if k != "agent"}})

    def stop_for_close(self) -> None:
        if self._worker and self._running:
            self._worker.request_stop()
            if self._thread:
                self._thread.wait(3000)

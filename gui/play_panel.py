"""Play tab: load .pt model, AI steps with pause / step, Q overlay."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from agent.dqn import DQNAgent
from env.snake_env import SnakeEnv
from gui.board_widget import BoardWidget

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class PlayPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._agent: DQNAgent | None = None
        self._env = SnakeEnv()
        self._obs = self._env.reset()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        top = QHBoxLayout()
        self.cb_model = QComboBox()
        self.cb_model.setMinimumWidth(260)
        btn_ref = QPushButton("Refresh")
        btn_load = QPushButton("Load")
        top.addWidget(QLabel("Model:"))
        top.addWidget(self.cb_model, stretch=1)
        top.addWidget(btn_ref)
        top.addWidget(btn_load)

        row2 = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_pause = QPushButton("Pause")
        self.btn_step = QPushButton("Step")
        self.btn_reset = QPushButton("Reset env")
        row2.addWidget(self.btn_play)
        row2.addWidget(self.btn_pause)
        row2.addWidget(self.btn_step)
        row2.addWidget(self.btn_reset)
        row2.addWidget(QLabel("FPS:"))
        self.sl_fps = QSlider(Qt.Orientation.Horizontal)
        self.sl_fps.setRange(1, 60)
        self.sl_fps.setValue(12)
        self.lb_fps = QLabel("12")
        self.sl_fps.valueChanged.connect(lambda v: self.lb_fps.setText(str(v)))
        row2.addWidget(self.sl_fps)
        row2.addWidget(self.lb_fps)

        self.board = BoardWidget()
        self.board.set_q_values(None)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(row2)
        lay.addWidget(self.board)

        btn_ref.clicked.connect(self._refresh_models)
        btn_load.clicked.connect(self._load_selected)
        self.btn_play.clicked.connect(self._play)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_step.clicked.connect(self._step_once)
        self.btn_reset.clicked.connect(self._reset_env)
        self.sl_fps.valueChanged.connect(self._apply_fps)

        self._refresh_models()
        self._apply_fps()
        self._paint_board()

    def _model_paths(self) -> list[Path]:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    def _refresh_models(self) -> None:
        self.cb_model.clear()
        for p in self._model_paths():
            self.cb_model.addItem(p.name, str(p))

    def _load_selected(self) -> None:
        if self.cb_model.count() == 0:
            QMessageBox.warning(self, "Models", f"No .pt files in {MODELS_DIR}")
            return
        path = self.cb_model.currentData()
        try:
            self._agent = DQNAgent.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self._pause()
        self._reset_env()

    def _apply_fps(self) -> None:
        fps = max(1, int(self.sl_fps.value()))
        self._timer.setInterval(int(1000 / fps))

    def _reset_env(self) -> None:
        self._obs = self._env.reset()
        self._paint_board()

    def _paint_board(self) -> None:
        self.board.set_state(self._env.snake, self._env.food, self._env.alive, self._env.direction)
        if self._agent is not None:
            self.board.set_q_values(self._agent.q_values(self._obs))
        else:
            self.board.set_q_values(None)

    def _play(self) -> None:
        if self._agent is None:
            QMessageBox.information(self, "Play", "Load a model first.")
            return
        self._apply_fps()
        self._timer.start()

    def _pause(self) -> None:
        self._timer.stop()

    def _step_once(self) -> None:
        if self._agent is None:
            QMessageBox.information(self, "Step", "Load a model first.")
            return
        self._timer.stop()
        self._do_step()

    def _tick(self) -> None:
        self._do_step()

    def _do_step(self) -> None:
        if self._agent is None:
            return
        a = self._agent.act(self._obs, explore=False)
        self._obs, _, done, _ = self._env.step(a)
        if done:
            self._obs = self._env.reset()
        self.board.set_state(self._env.snake, self._env.food, self._env.alive, self._env.direction)
        self.board.set_q_values(self._agent.q_values(self._obs))

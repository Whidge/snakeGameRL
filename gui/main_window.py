"""Main app: Train / Play tabs + curves window."""

from __future__ import annotations

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QTabWidget, QWidget

from gui.curves_window import CurvesWindow
from gui.play_panel import PlayPanel
from gui.train_panel import TrainPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Snake DQN — Train / Play")
        self.resize(1100, 720)

        self.curves = CurvesWindow(self)
        self.curves.hide()

        self.train_panel = TrainPanel()
        self.play_panel = PlayPanel()
        tabs = QTabWidget()
        tabs.addTab(self.train_panel, "Train")
        tabs.addTab(self.play_panel, "Play")
        self.setCentralWidget(tabs)

        self.train_panel.training_started.connect(self._on_training_started)
        self.train_panel.metrics_out.connect(self.curves.on_metrics)
        self.train_panel.loss_out.connect(self.curves.on_loss)
        self.train_panel.frame_out.connect(self._on_train_frame)

        m_file = self.menuBar().addMenu("File")
        a_models = QAction("Open models folder…", self)
        a_models.triggered.connect(self._open_models_dir)
        m_file.addAction(a_models)

        m_view = self.menuBar().addMenu("View")
        self.act_curves = QAction("Training curves", self)
        self.act_curves.setCheckable(True)
        self.act_curves.toggled.connect(self._toggle_curves)
        m_view.addAction(self.act_curves)

    def _on_training_started(self) -> None:
        self.curves.clear()
        self.act_curves.blockSignals(True)
        self.act_curves.setChecked(True)
        self.act_curves.blockSignals(False)
        self.curves.show()
        self.curves.raise_()
        self.curves.activateWindow()

    def _on_train_frame(self, d: dict) -> None:
        # Update even when hidden so reopening shows full epsilon trace
        self.curves.on_epsilon_frame(int(d["frames"]), float(d["epsilon"]))

    def _toggle_curves(self, checked: bool) -> None:
        if checked:
            self.curves.show()
        else:
            self.curves.hide()

    def _open_models_dir(self) -> None:
        from gui.train_panel import MODELS_DIR

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        QFileDialog.getOpenFileName(self, "Models", str(MODELS_DIR), "*.pt")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.train_panel.stop_for_close()
        self.curves.hide()
        super().closeEvent(event)

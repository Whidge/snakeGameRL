"""Qt GUI entry: Train / Play / curves."""

from __future__ import annotations

import importlib.util
import os
import sys


def _pyqt6_root() -> str | None:
    spec = importlib.util.find_spec("PyQt6")
    if not spec:
        return None
    if spec.submodule_search_locations:
        return next(iter(spec.submodule_search_locations))
    if spec.origin:
        return os.path.dirname(spec.origin)
    return None


def _fix_qt_plugin_path(root: str) -> None:
    """macOS/venv: set paths *before* importing PyQt6.* (importing PyQt6 loads Qt and locks plugin search)."""
    plugins = os.path.join(root, "Qt6", "plugins")
    platforms = os.path.join(plugins, "platforms")
    if os.path.isdir(plugins):
        os.environ["QT_PLUGIN_PATH"] = plugins
    if os.path.isdir(platforms):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms


def _qt_runtime_ok(root: str) -> bool:
    """PyQt6-Qt6 must ship QtCore next to bindings; missing folder = broken / partial pip."""
    here = os.path.join(root, "Qt6", "lib")
    if not os.path.isdir(here):
        return False
    mac = os.path.join(here, "QtCore.framework", "Versions", "A", "QtCore")
    linux = os.path.join(here, "libQt6Core.so.6")
    win = os.path.join(root, "Qt6", "bin", "Qt6Core.dll")
    return os.path.isfile(mac) or os.path.isfile(linux) or os.path.isfile(win)


def _die_qt_broken() -> None:
    sys.stderr.write(
        "\nPyQt6 Qt runtime missing: PyQt6/Qt6/lib is not there or incomplete.\n"
        "Common after a failed/partial pip or version mismatch.\n\n"
        "Fix (same venv you use to run app.py):\n"
        "  pip install --force-reinstall --no-cache-dir PyQt6 PyQt6-Qt6\n\n"
        "If that still fails:\n"
        "  pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip -y\n"
        "  pip install PyQt6 PyQt6-Qt6\n\n"
    )
    raise SystemExit(1)


_root = _pyqt6_root()
if _root is None:
    sys.stderr.write("PyQt6 is not installed in this environment.\n")
    raise SystemExit(1)
if not _qt_runtime_ok(_root):
    _die_qt_broken()
_fix_qt_plugin_path(_root)

try:
    from PyQt6.QtWidgets import QApplication
except ImportError as e:
    if "Library not loaded" in str(e) or "QtWidgets" in str(e) or "QtCore" in str(e):
        _die_qt_broken()
    raise

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from pose_craft.diagnostics import configure_logging, install_excepthooks
from pose_craft.gui.main_window import MainWindow


def _asset_path(relative: str) -> Path:
    # sys._MEIPASS is the extraction root PyInstaller sets at runtime when frozen
    # (see PoseCraft.spec's `datas`, which bundles assets/ there); outside a frozen
    # build it falls back to the source tree so `python -m pose_craft.app` works too.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


ICON_PATH = _asset_path("assets/icon.png")


class _ErrorDialogRelay(QObject):
    """Shows uncaught-exception messages on the GUI thread, whichever thread
    raised (a queued signal hops threads safely; a QMessageBox can't)."""

    message = Signal(str)

    def __init__(self):
        super().__init__()
        self.message.connect(self._show, Qt.QueuedConnection)

    def _show(self, text: str) -> None:
        QMessageBox.critical(None, "PoseCraft error", text)


def main() -> int:
    from pose_craft.selftest import run_self_test, wants_self_test

    if wants_self_test(sys.argv[1:]):
        # Models, a real CPU/GPU solve against a golden output, exports and the
        # main window offscreen, then exit -- no visible window (pose_craft/selftest.py).
        return run_self_test()
    # The windowed build has no console: log to a file and turn any uncaught
    # exception into a dialog pointing at it (see pose_craft/diagnostics.py).
    configure_logging()
    app = QApplication(sys.argv)
    relay = _ErrorDialogRelay()
    install_excepthooks(relay.message.emit)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    # warm_up: load the remembered model on the GPU in the background, so the
    # first Solve doesn't pay the cold start (only when it's already cached).
    window = MainWindow(warm_up=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

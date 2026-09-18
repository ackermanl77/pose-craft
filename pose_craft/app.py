from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from pose_craft.gui.main_window import MainWindow


def _asset_path(relative: str) -> Path:
    # sys._MEIPASS is the extraction root PyInstaller sets at runtime when frozen
    # (see PoseCraft.spec's `datas`, which bundles assets/ there); outside a frozen
    # build it falls back to the source tree so `python -m pose_craft.app` works too.
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


ICON_PATH = _asset_path("assets/icon.png")


def main() -> int:
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

"""MainWindow.show_update: a found update is announced and opens the download page only on Yes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt install; CI's lightweight venv skips it

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from pose_craft.gui.main_window import MainWindow  # noqa: E402
from pose_craft.update_check import UpdateInfo  # noqa: E402


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("answer, opened", [(QMessageBox.Yes, True), (QMessageBox.No, False)])
def test_update_notice_opens_the_download_page_only_on_yes(qapp, answer, opened):
    window = MagicMock()
    info = UpdateInfo("9.0.0", "https://example.com/get", "Better hands")
    with patch("pose_craft.gui.main_window.QMessageBox.question", return_value=answer) as question, \
            patch("PySide6.QtGui.QDesktopServices.openUrl") as open_url:
        MainWindow.show_update(window, info)
    assert "9.0.0" in window.statusBar().showMessage.call_args.args[0]
    assert "Better hands" in question.call_args.args[2]
    assert open_url.called is opened

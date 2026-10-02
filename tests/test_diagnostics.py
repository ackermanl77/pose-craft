from __future__ import annotations

import logging
import sys
import threading

import pytest

from pose_craft import diagnostics


@pytest.fixture(autouse=True)
def _restore_hooks_and_handlers():
    saved = sys.excepthook, threading.excepthook, list(diagnostics.log.handlers), diagnostics.log.level
    yield
    sys.excepthook, threading.excepthook = saved[0], saved[1]
    for handler in list(diagnostics.log.handlers):
        if handler not in saved[2]:
            diagnostics.log.removeHandler(handler)
            handler.close()
    diagnostics.log.setLevel(saved[3])


def test_configure_logging_writes_pose_craft_logs_to_a_file(tmp_path):
    path = diagnostics.configure_logging(tmp_path / "logs")
    logging.getLogger("pose_craft.gui.main_window").warning("export failed for C:/x.json")
    for handler in diagnostics.log.handlers:
        handler.flush()

    assert path == tmp_path / "logs" / diagnostics.LOG_FILE_NAME
    assert "export failed for C:/x.json" in path.read_text(encoding="utf-8")
    assert str(path) in diagnostics.log_file_hint()


def test_configure_logging_never_raises_when_the_folder_cannot_be_created(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert diagnostics.configure_logging(blocker / "logs") is None


def test_default_log_dir_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert diagnostics.default_log_dir() == tmp_path / "PoseCraft" / "logs"


def test_uncaught_exceptions_are_logged_and_reported_from_any_thread(tmp_path):
    path = diagnostics.configure_logging(tmp_path)
    messages = []
    diagnostics.install_excepthooks(messages.append)

    try:
        raise ValueError("main thread boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    worker = threading.Thread(target=lambda: 1 / 0, name="worker-1")
    worker.start()
    worker.join()
    for handler in diagnostics.log.handlers:
        handler.flush()

    assert len(messages) == 2
    assert "ValueError: main thread boom" in messages[0] and str(path) in messages[0]
    assert "ZeroDivisionError" in messages[1]
    text = path.read_text(encoding="utf-8")
    assert "main thread boom" in text and "Traceback" in text and "worker-1" in text

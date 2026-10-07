"""Logging and last-resort error reporting for the windowed app.

The built .exe has no console (PoseCraft.spec: console=False), so anything
printed to stderr -- an uncaught exception in a Qt slot, a failed worker's
traceback -- used to vanish. `configure_logging` writes a rotating log file
under %LOCALAPPDATA%\\PoseCraft\\logs (~/.local/state/PoseCraft/logs
elsewhere), and `install_excepthooks` logs every uncaught exception, on the
main thread or any other, and hands a short message to a callback (the app
shows it in a dialog). Pure stdlib, so it's testable without Qt.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

LOG_FILE_NAME = "posecraft.log"
_MAX_LOG_BYTES = 1_000_000
_BACKUP_COUNT = 3

_log_file: Path | None = None
log = logging.getLogger("pose_craft")


def default_log_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PoseCraft" / "logs"
    return Path.home() / ".local" / "state" / "PoseCraft" / "logs"


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO) -> Path | None:
    """Sends the "pose_craft" loggers to a rotating file in `log_dir` (default:
    default_log_dir()). Returns the log file, or None if the directory can't be
    created (logging must never stop the app from starting)."""
    global _log_file
    directory = default_log_dir() if log_dir is None else Path(log_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILE_NAME
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=_MAX_LOG_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for existing in list(log.handlers):
        if isinstance(existing, logging.handlers.RotatingFileHandler):
            log.removeHandler(existing)
            existing.close()
    log.addHandler(handler)
    log.setLevel(level)
    _log_file = path
    return path


def log_file_hint() -> str:
    """Where the log is, for error dialogs."""
    return str(_log_file) if _log_file is not None else "(logging to a file is not enabled)"


def install_excepthooks(on_error: Callable[[str], None] | None = None) -> None:
    """Logs uncaught exceptions from the main thread (sys.excepthook) and from
    other threads (threading.excepthook), then calls on_error(message). A
    failing on_error is logged, never raised."""

    def _report(exc_type, exc_value, exc_traceback, where: str) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        log.critical("uncaught exception %s", where, exc_info=(exc_type, exc_value, exc_traceback))
        if on_error is not None:
            try:
                on_error(
                    f"PoseCraft hit an unexpected error: {exc_type.__name__}: {exc_value}\n\n"
                    f"Details were saved to: {log_file_hint()}"
                )
            except Exception:  # noqa: BLE001 -- the error reporter must never raise
                log.exception("error callback failed")

    sys.excepthook = lambda t, v, tb: _report(t, v, tb, "on the main thread")
    threading.excepthook = lambda args: _report(
        args.exc_type, args.exc_value, args.exc_traceback, f"in thread {getattr(args.thread, 'name', '?')}"
    )

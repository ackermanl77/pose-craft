"""Suite-wide isolation: the GUI remembers settings (QSettings, an INI file
under the user's config folder) -- point that at a temporary folder so tests
never read or write the real user's settings."""
from __future__ import annotations

import pytest


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "gpu: compares GPU and CPU results; needs a GPU provider and cached models")


@pytest.fixture(autouse=True, scope="session")
def _isolated_qt_settings(tmp_path_factory):
    try:
        from PySide6.QtCore import QSettings
    except ImportError:  # CI's lightweight venv has no Qt
        yield
        return
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path_factory.mktemp("qt-settings")))
    yield


@pytest.fixture(autouse=True)
def _isolated_pose_library(tmp_path, monkeypatch):
    """The pose library (F15) lives in a per-test folder, never the user's."""
    monkeypatch.setenv("POSECRAFT_LIBRARY_DIR", str(tmp_path / "pose_library"))

"""R4: Help > Diagnostics says which runtime/GPU is in use and why a GPU isn't."""
from __future__ import annotations

import pytest

from pose_craft import diagnostics


class _Estimator:
    variant = "heavy"
    device = "CPU"
    gpu_fallback_reason = "the GPU's results didn't match the CPU's"


def test_the_report_names_the_runtime_the_model_and_the_fallback_reason():
    rows = dict(diagnostics.diagnostics_report(_Estimator()))
    assert rows["Pose model"] == "heavy on CPU"
    assert rows["GPU not used because"] == "the GPU's results didn't match the CPU's"
    assert "CPUExecutionProvider" in rows["Providers in this build"]
    assert "DirectX 12 GPUs" in rows and "Log file" in rows


def test_the_report_survives_a_broken_gpu_probe(monkeypatch):
    from pose_craft.estimation import onnx_pose

    monkeypatch.setattr(onnx_pose, "directml_adapters", lambda: (_ for _ in ()).throw(OSError("dxgi gone")))
    rows = dict(diagnostics.diagnostics_report())
    assert rows["DirectX 12 GPUs"].startswith("could not be listed") and rows["Pose model"].startswith("not loaded")
    assert "PoseCraft" in diagnostics.format_report(list(rows.items()))


def test_the_help_menu_opens_the_dialog_with_the_report(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    from pose_craft.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(settings=QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat))
    window._estimator = _Estimator()
    dialog = window._on_show_diagnostics()
    text = dialog.findChild(QPlainTextEdit, "diagnosticsText").toPlainText()
    assert "heavy on CPU" in text and "didn't match the CPU's" in text
    assert window.diagnostics_action.text() == "&Diagnostics..."
    dialog.close()
    window._estimator = None
    window.close()
    app.processEvents()

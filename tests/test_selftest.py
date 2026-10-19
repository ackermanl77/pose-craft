"""--self-test (pose_craft/selftest.py): routing from main(), the bundled photo,
the report and the exit code, and a full run when the model is cached."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pose_craft import app as app_module
from pose_craft import selftest

ROOT = Path(__file__).resolve().parent.parent


def test_main_runs_the_self_test_without_the_window(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["PoseCraft.exe", "--self-test"])
    monkeypatch.setattr(selftest, "run_self_test", lambda: calls.append(1) or 0)
    monkeypatch.setattr(app_module, "MainWindow", lambda *a, **k: pytest.fail("the window was built"))
    assert app_module.main() == 0
    assert calls == [1]


def test_the_photo_and_its_golden_output_are_bundled():
    assert (selftest.photo_dir() / selftest.PHOTO).is_file()
    spec = (ROOT / "PoseCraft.spec").read_text(encoding="utf-8")
    for name in (selftest.PHOTO, Path(selftest.PHOTO).stem + ".golden.json"):
        assert re.search(rf"\('tests/data/golden/{re.escape(name)}', 'selftest'\)", spec)


def test_a_failing_check_fails_the_run_and_writes_the_report(monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(selftest, "_check_models", lambda log: True)
    monkeypatch.setattr(selftest, "_check_window", lambda workdir, log: True)

    def broken(workdir, log):
        raise RuntimeError("no onnx session")

    monkeypatch.setattr(selftest, "_check_solve", broken)
    lines = []
    assert selftest.run_self_test(lines.append) == 1
    report = (tmp_path / selftest.REPORT_NAME).read_text(encoding="utf-8")
    assert "FAIL  RuntimeError: no onnx session" in report
    assert "RESULT: FAIL" in lines


def test_full_self_test_passes_with_the_cached_model(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    from pose_craft.estimation.landmarker import is_model_available

    if not is_model_available(selftest.MODEL_VARIANT):
        pytest.skip("the 'full' pose model isn't cached")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    lines = []
    assert selftest.run_self_test(lines.append) == 0, "\n".join(lines)

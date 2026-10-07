"""B13: folder batch processing -- pose_craft.batch, the CLI and the GUI dialog."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from pose_craft import batch, cli
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_gui_flows import _person  # noqa: E402


class _Estimator:
    """Finds two people in a photo named "*two*", one elsewhere, nobody in "*empty*"."""

    device = "CPU"

    def __init__(self):
        self.seen, self.hands, self.group_photo, self.min_detection_confidence = [], False, False, 0.5

    def detect(self, image):
        self.seen.append(image.shape)
        brightness = int(image[0, 0, 0])
        return [] if brightness == 1 else [_person(), _person(x_offset=0.2)] if brightness == 2 else [_person()]

    def enable_hands(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass


def _folder(tmp_path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    for name, value in (("a_one.jpg", 0), ("b_two.png", 2), ("c_empty.png", 1)):
        cv2.imwrite(str(folder / name), np.full((60, 40, 3), value, np.uint8))
    (folder / "d_broken.jpg").write_bytes(b"not an image")
    (folder / "notes.txt").write_text("ignored")
    return folder


def test_every_photo_gets_one_file_per_person_named_like_export_all(tmp_path):
    result = batch.solve_folder(_folder(tmp_path), tmp_path / "out", _Estimator())
    names = sorted(path.name for path in result.written)
    assert names == ["a_one_person1.json", "b_two_person1.json", "b_two_person2.json"]
    errors = {outcome.photo.name: outcome.error for outcome in result.outcomes if outcome.error}
    assert errors == {"c_empty.png": "no person detected", "d_broken.jpg": "couldn't be read as an image"}
    document = json.loads((tmp_path / "out" / "a_one_person1.json").read_text(encoding="utf-8"))
    assert set(document["bones"]) >= set(sk.BONE_NAMES) and "batch: a_one.jpg" in document["source_note"]
    assert result.summary() == "3 pose(s) from 2 of 4 photo(s)"


def test_bvh_mirror_overwrite_and_cancel(tmp_path):
    folder, out = _folder(tmp_path), tmp_path / "out"
    first = batch.solve_folder(folder, out, _Estimator(), fmt="bvh", mirror=True)
    assert {path.suffix for path in first.written} == {".bvh"}
    kept = batch.solve_folder(folder, out, _Estimator(), fmt="bvh", overwrite=False)
    assert kept.written == []
    cancelled = batch.solve_folder(folder, out, _Estimator(), should_cancel=lambda: True)
    assert cancelled.cancelled and cancelled.outcomes == []
    with pytest.raises(ValueError):
        batch.solve_folder(folder, out, _Estimator(), fmt="fbx")


def test_the_cli_runs_the_batch_and_reports_per_photo(tmp_path, monkeypatch, capsys):
    estimator = _Estimator()
    created = []
    monkeypatch.setattr(cli.PoseEstimator, "for_variant", classmethod(lambda cls, variant, **kw: created.append((variant, kw)) or estimator))
    code = cli.main(["--input", str(_folder(tmp_path)), "--output", str(tmp_path / "out"), "--model", "lite", "--cpu",
                     "--detection-threshold", "0.3", "--group-photo"])
    out = capsys.readouterr().out
    assert code == 1  # two photos gave nothing
    assert created == [("lite", {"max_people": 4, "use_gpu": False})]
    assert estimator.min_detection_confidence == 0.3 and estimator.group_photo
    assert "c_empty.png: no person detected" in out and "b_two.png: b_two_person1.json, b_two_person2.json" in out
    assert cli.main(["--input", str(tmp_path / "missing"), "--output", str(tmp_path / "out")]) == 2


def test_the_gui_dialog_runs_a_batch_with_the_windows_settings(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from test_gui_flows import _wait_until

    from pose_craft.gui import batch_dialog
    from pose_craft.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    estimator = _Estimator()
    monkeypatch.setattr(batch_dialog.PoseEstimator, "for_variant", classmethod(lambda cls, variant, **kw: estimator))
    window = MainWindow(settings=QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat))
    try:
        window.detection_threshold_spin.setValue(0.35)
        dialog = window._on_batch_folder()
        dialog.input_edit.setText(str(_folder(tmp_path)))
        dialog.output_edit.setText(str(tmp_path / "out"))
        dialog.start()
        assert _wait_until(app, lambda: dialog.result is not None)
        assert len(dialog.result.written) == 3 and estimator.min_detection_confidence == pytest.approx(0.35)
        assert "3 pose(s) from 2 of 4 photo(s)" in dialog.log_view.toPlainText()
        dialog.close()
    finally:
        window._closing = True
        window.close()
        app.processEvents()

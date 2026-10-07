"""B10: per-landmark presence (out of frame vs. hidden) and adjustable thresholds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pose_craft.export import blender_pose
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _standing_t_pose  # noqa: E402


def _presence_with_feet_cut_off() -> np.ndarray:
    presence = np.ones(sk.NUM_LANDMARKS)
    for name in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"):
        presence[getattr(sk, name)] = 0.1
    return presence


def test_bones_reaching_a_landmark_outside_the_frame_are_flagged():
    flagged = sk.bones_out_of_frame(_presence_with_feet_cut_off())
    assert {"shin.L", "shin.R", "foot.L", "foot.R"} <= flagged
    assert "spine" not in flagged and "upper_arm.L" not in flagged
    assert sk.bones_out_of_frame(None) == set()
    assert sk.bones_out_of_frame(np.ones(sk.NUM_LANDMARKS)) == set()
    with pytest.raises(ValueError):
        sk.bones_out_of_frame(np.ones(5))


def test_the_export_marks_out_of_frame_bones_apart_from_low_confidence():
    rotations = sk.solve_bone_rotations(_standing_t_pose())
    document = blender_pose.build_pose_document(rotations, bone_confidence={"shin.L": 0.9}, out_of_frame={"shin.L"})
    assert document["bones"]["shin.L"]["out_of_frame"] is True
    assert document["bones"]["shin.L"]["low_confidence"] is False
    assert "out_of_frame" not in document["bones"]["spine"]


def test_the_low_confidence_threshold_is_adjustable_and_validated(tmp_path):
    rotations = sk.solve_bone_rotations(_standing_t_pose())
    try:
        blender_pose.set_low_confidence_threshold(0.95)
        path = tmp_path / "pose.json"
        blender_pose.save_pose_file(rotations, path, bone_confidence={"spine": 0.9})
        assert json.loads(path.read_text(encoding="utf-8"))["bones"]["spine"]["low_confidence"] is True
        with pytest.raises(ValueError):
            blender_pose.set_low_confidence_threshold(1.5)
    finally:
        blender_pose.set_low_confidence_threshold(blender_pose.LOW_CONFIDENCE_THRESHOLD)


def test_the_estimator_passes_its_thresholds_to_the_model():
    from pose_craft.estimation.landmarker import PoseEstimator

    calls = []

    class _Pose:
        def detect(self, image, max_people, min_detection_confidence, min_presence_confidence, tiled):
            calls.append((min_detection_confidence, min_presence_confidence))
            return []

    estimator = PoseEstimator.__new__(PoseEstimator)
    estimator._pose, estimator.max_people, estimator._hand_model = _Pose(), 1, None
    estimator.min_detection_confidence = 0.3
    assert estimator.detect(np.zeros((64, 64, 3), np.uint8)) == []
    assert calls == [(0.3, 0.5)]


def test_the_gui_flags_cut_off_bones_and_applies_the_advanced_settings(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from test_gui_flows import _FakeEstimator, _person, _use_estimator, _wait_until

    from pose_craft.gui import main_window as mw

    app = QApplication.instance() or QApplication([])
    person = _person()
    person.landmark_presence = _presence_with_feet_cut_off()
    estimator = _FakeEstimator(people=[person])
    _use_estimator(monkeypatch, estimator)
    settings = QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat)
    window = mw.MainWindow(settings=settings)
    try:
        window._image_bgr = np.zeros((600, 600, 3), dtype=np.uint8)
        window.solve_btn.setEnabled(True)
        window.detection_threshold_spin.setValue(0.3)
        window._on_solve_pose()
        assert _wait_until(app, lambda: bool(window._people))
        assert estimator.min_detection_confidence == pytest.approx(0.3)
        overlay = window.photo_view._overlay_people[0]
        assert {"shin.L", "foot.R"} <= overlay.out_of_frame
        assert "shin.L" in window._out_of_frame_for(0)

        window.low_confidence_spin.setValue(0.8)
        assert blender_pose.low_confidence_threshold() == pytest.approx(0.8)
        assert "80%" in window.legend.toolTip()
        window._save_settings()
        assert float(settings.value("detect/low_confidence_threshold")) == pytest.approx(0.8)
    finally:
        blender_pose.set_low_confidence_threshold(blender_pose.LOW_CONFIDENCE_THRESHOLD)
        window._closing = True
        window.close()
        app.processEvents()

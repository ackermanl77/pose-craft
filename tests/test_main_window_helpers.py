from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt install; CI's lightweight venv skips it

from pose_craft.estimation.landmarker import DetectedPerson  # noqa: E402
from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD  # noqa: E402
from pose_craft.gui.main_window import (  # noqa: E402
    MainWindow,
    _low_confidence_bone_count,
    _low_confidence_status_suffix,
    _nearest_person_index,
)
from pose_craft.retarget import skeleton as sk  # noqa: E402


class _StubPerson:
    def __init__(self, x: float, y: float):
        self._xy = (x, y)

    def centroid_xy(self) -> tuple[float, float]:
        return self._xy


def test_nearest_person_index_picks_closest_centroid():
    people = [_StubPerson(0.1, 0.5), _StubPerson(0.9, 0.5), _StubPerson(0.5, 0.1)]

    assert _nearest_person_index(people, 0.12, 0.48) == 0
    assert _nearest_person_index(people, 0.85, 0.5) == 1
    assert _nearest_person_index(people, 0.5, 0.05) == 2


def test_low_confidence_bone_count_counts_only_bones_below_threshold():
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    confidence["hand.L"] = LOW_CONFIDENCE_THRESHOLD - 0.01
    confidence["foot.R"] = LOW_CONFIDENCE_THRESHOLD - 0.01

    assert _low_confidence_bone_count(confidence) == 2


def test_low_confidence_bone_count_is_zero_when_every_bone_is_confident():
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    assert _low_confidence_bone_count(confidence) == 0


def test_low_confidence_bone_count_boundary_is_exclusive():
    """bone_confidence scores strictly below the threshold count as low -- a score
    exactly at the threshold does not (matches photo_view.py's `< LOW_CONFIDENCE_THRESHOLD`
    and blender_pose.py's own low_confidence flag, both strict-less-than).
    """
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    confidence["spine"] = LOW_CONFIDENCE_THRESHOLD
    assert _low_confidence_bone_count(confidence) == 0


def test_low_confidence_status_suffix_empty_when_nothing_is_low():
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    assert _low_confidence_status_suffix(confidence) == ""


def test_low_confidence_status_suffix_reports_count_and_total():
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    confidence["hand.L"] = 0.1
    confidence["foot.R"] = 0.1

    suffix = _low_confidence_status_suffix(confidence)

    assert suffix == f" -- 2 of {len(sk.BONE_NAMES)} bones low-confidence"


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_mirrored_confidence_matches_the_landmark_data_actually_used(qapp):
    """Regression test: _current_person_document_inputs() used to mirror the
    landmarks (for rotations) but not the visibility array (for confidence), so
    a mirrored thigh.L's reported confidence came from the ORIGINAL left leg's
    visibility instead of the real (swapped-in) right leg's -- inconsistent with
    what solve_bone_rotations actually used to compute that bone's rotation.
    """
    landmarks = np.zeros((sk.NUM_LANDMARKS, 3))
    landmarks[sk.LEFT_HIP], landmarks[sk.RIGHT_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
    landmarks[sk.LEFT_KNEE], landmarks[sk.RIGHT_KNEE] = [0.1, -0.5, 0.0], [-0.1, -0.5, 0.0]

    visibility = np.ones(sk.NUM_LANDMARKS)
    visibility[sk.RIGHT_HIP] = 0.05
    visibility[sk.RIGHT_KNEE] = 0.05  # the real right leg is barely visible

    person = DetectedPerson(landmarks=landmarks, image_landmarks_xy=np.zeros((sk.NUM_LANDMARKS, 2)), visibility=visibility)

    window = MainWindow()
    try:
        window._people = [person]
        window.person_combo.addItem("Person 1")
        window.mirror_checkbox.setChecked(True)

        _rotations, confidence = window._current_person_document_inputs()

        # Mirrored: thigh.L's position now comes from the original RIGHT hip/knee data,
        # so its confidence must reflect that side's (low) visibility, not the
        # original left leg's (high) visibility.
        assert confidence["thigh.L"] == pytest.approx(0.05)
        assert confidence["thigh.R"] == pytest.approx(1.0)
    finally:
        window.close()


def test_orientation_combo_defaults_to_no_change(qapp):
    """The "Default" entry must be identity so existing behavior (and every
    other test/export path) is unaffected by the new control out of the box.
    """
    window = MainWindow()
    try:
        assert window.orientation_combo.currentText() == "Default"
        assert np.linalg.norm(window._orientation_correction().as_rotvec()) < 1e-9
    finally:
        window.close()


def test_orientation_correction_is_applied_to_the_exported_rotations(qapp):
    """Selecting a body-orientation fix must reach the rotations dict used for
    JSON/BVH/Live Link export -- the same composed-onto-root-bone effect
    skeleton.apply_orientation_correction documents.
    """
    landmarks = np.zeros((sk.NUM_LANDMARKS, 3))
    landmarks[sk.LEFT_HIP], landmarks[sk.RIGHT_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
    landmarks[sk.LEFT_SHOULDER], landmarks[sk.RIGHT_SHOULDER] = [0.2, 0.5, 0.0], [-0.2, 0.5, 0.0]

    person = DetectedPerson(
        landmarks=landmarks,
        image_landmarks_xy=np.zeros((sk.NUM_LANDMARKS, 2)),
        visibility=np.ones(sk.NUM_LANDMARKS),
    )

    window = MainWindow()
    try:
        window._people = [person]
        window.person_combo.addItem("Person 1")

        baseline_rotations, _ = window._current_person_document_inputs()

        index = window.orientation_combo.findText("Flip Upside-Down")
        assert index >= 0
        window.orientation_combo.setCurrentIndex(index)

        corrected_rotations, _ = window._current_person_document_inputs()

        expected_spine = sk.ORIENTATION_CORRECTIONS["Flip Upside-Down"] * baseline_rotations["spine"]
        relative = corrected_rotations["spine"] * expected_spine.inv()
        assert np.linalg.norm(relative.as_rotvec()) < 1e-6

        # Every other bone's local rotation is unaffected, same as
        # apply_orientation_correction's own guarantee.
        for name in sk.BONE_NAMES:
            if name == "spine":
                continue
            assert np.allclose(corrected_rotations[name].as_quat(), baseline_rotations[name].as_quat())
    finally:
        window.close()


class _FakeEstimator:
    device = "CPU"
    max_people = 4

    def close(self) -> None:
        pass


def _standing_person() -> DetectedPerson:
    landmarks = np.zeros((sk.NUM_LANDMARKS, 3))
    landmarks[sk.LEFT_HIP], landmarks[sk.RIGHT_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
    landmarks[sk.LEFT_SHOULDER], landmarks[sk.RIGHT_SHOULDER] = [0.2, 0.5, 0.0], [-0.2, 0.5, 0.0]
    return DetectedPerson(
        landmarks=landmarks,
        image_landmarks_xy=np.full((sk.NUM_LANDMARKS, 2), 0.5),
        visibility=np.ones(sk.NUM_LANDMARKS),
    )


def test_resolve_that_finds_nobody_clears_the_previous_result(qapp, monkeypatch):
    """Regression test: a re-solve that detected nobody used to return early
    with the previous solve's person list, photo overlay and 3D skeleton still
    showing, and every export button still enabled for those stale people."""
    from pose_craft.gui import main_window as mw

    monkeypatch.setattr(mw.QMessageBox, "information", lambda *args, **kwargs: None)
    window = MainWindow()
    try:
        window._on_solve_finished([_standing_person()], _FakeEstimator())
        assert window.person_combo.count() == 1
        assert window.export_btn.isEnabled() and window.export_all_btn.isEnabled()
        assert window.viewport._bone_lines is not None
        assert window.photo_view._overlay_people

        window._on_solve_finished([], _FakeEstimator())

        assert window.person_combo.count() == 0
        assert window._people == []
        assert window.viewport._bone_lines is None and window.viewport._joints is None
        assert window.photo_view._overlay_people == []
        for button in (window.export_btn, window.export_all_btn, window.export_bvh_btn, window.send_to_blender_btn):
            assert not button.isEnabled()
    finally:
        window.close()

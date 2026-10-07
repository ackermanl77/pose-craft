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


# --- F2: the window fits a 1080p screen ----------------------------------------


def test_main_window_minimum_size_fits_a_1080p_screen(qapp):
    """Regression test: the stacked left column made the minimum height 1124 px,
    taller than a 1080p screen's usable ~1030 px (less at 125-150% scaling)."""
    window = MainWindow()
    try:
        hint = window.minimumSizeHint()
        assert hint.height() <= 900, hint
        assert hint.width() <= 1280, hint
    finally:
        window.close()


# --- F1: the 3D preview shows the exported rotations ------------------------------


def _yawed_person() -> DetectedPerson:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from scipy.spatial.transform import Rotation
    from test_retarget_frames import _articulated_pose, _rigidly_rotated

    landmarks = _rigidly_rotated(_articulated_pose(), Rotation.from_euler("y", 60, degrees=True))
    return DetectedPerson(
        landmarks=landmarks, image_landmarks_xy=np.full((sk.NUM_LANDMARKS, 2), 0.5), visibility=np.ones(sk.NUM_LANDMARKS)
    )


def test_preview_draws_forward_kinematics_of_the_exported_rotations(qapp, monkeypatch):
    """The preview must show what the export says, not the raw landmarks: with
    the solver stubbed to return an all-identity pose, the drawn skeleton must be
    a T-pose even though the landmarks are articulated and turned."""
    from scipy.spatial.transform import Rotation

    from pose_craft import posing

    # The export's solve (posing.document_inputs, shared with the batch) -- what the preview draws.
    monkeypatch.setattr(posing, "solve_bone_rotations", lambda *a, **k: {n: Rotation.identity() for n in sk.BONE_NAMES})
    window = MainWindow()
    drawn = {}
    monkeypatch.setattr(window.viewport, "show_skeleton", lambda positions, **kwargs: drawn.update(positions=positions, **kwargs))
    try:
        window._people = [_yawed_person()]
        window.person_combo.addItem("Person 1")
        window._update_preview_for_selected_person()

        points = drawn["positions"]
        forearm = points["forearm.L.tail"] - points["forearm.L.head"]
        assert np.dot(forearm / np.linalg.norm(forearm), sk.LEFT_X) > 0.999  # T-pose arm
        # The raw landmarks are still shown, as a ghost for comparison.
        assert drawn.get("ghost_points") is not None and len(drawn["ghost_points"]) > 0
    finally:
        window.close()


def test_preview_of_a_turned_body_lands_on_its_landmarks(qapp, monkeypatch):
    window = MainWindow()
    drawn = {}
    monkeypatch.setattr(window.viewport, "show_skeleton", lambda positions, **kwargs: drawn.update(positions=positions))
    try:
        person = _yawed_person()
        window._people = [person]
        window.person_combo.addItem("Person 1")
        window._update_preview_for_selected_person()

        # The preview is lifted onto the floor (B6), so compare relative to the hips.
        expected = sk.joint_positions(person.landmarks)
        drawn_hips, expected_hips = drawn["positions"]["spine.head"], expected["spine.head"]
        for name in ("upper_arm.L.head", "upper_arm.R.head", "thigh.L.head", "forearm.L.tail", "shin.R.tail"):
            np.testing.assert_allclose(
                drawn["positions"][name] - drawn_hips, expected[name] - expected_hips, atol=1e-6, err_msg=name
            )
    finally:
        window.close()


def test_preview_stands_the_skeleton_on_the_grid(qapp, monkeypatch):
    """B6/F9: the world landmarks are hip-centred, so the preview used to draw
    every skeleton with its hips on the grid and its legs through the floor."""
    window = MainWindow()
    drawn = {}
    monkeypatch.setattr(window.viewport, "show_skeleton", lambda positions, **kwargs: drawn.update(positions=positions, **kwargs))
    try:
        window._people = [_yawed_person()]
        window.person_combo.addItem("Person 1")
        window._update_preview_for_selected_person()

        lowest = min(point[1] for point in drawn["positions"].values())
        assert lowest == pytest.approx(0.0, abs=1e-9)
        assert drawn["positions"]["spine.head"][1] > 0.5  # the hips are well above the floor
        assert set(drawn["lateral_axes"]) == set(sk.BONE_NAMES)  # octahedral bones get each bone's roll
    finally:
        window.close()


# --- R1: export errors are reported -----------------------------------------------


@pytest.mark.parametrize("handler", ["_on_export_pose", "_on_export_bvh", "_on_export_all"])
def test_export_to_an_unwritable_path_shows_a_warning(qapp, monkeypatch, tmp_path, handler):
    from pose_craft.gui import main_window as mw

    blocker = tmp_path / "not_a_folder"
    blocker.write_text("a file where a folder should be", encoding="utf-8")
    bad_path = str(blocker / "pose.out")
    monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (bad_path, ""))
    monkeypatch.setattr(mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(blocker))
    warnings = []
    monkeypatch.setattr(mw.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args))

    window = MainWindow()
    try:
        window._people = [_standing_person()]
        window.person_combo.addItem("Person 1")
        getattr(window, handler)()  # must not raise out of the Qt slot
        assert warnings, "no warning dialog for a failed export"
        assert str(blocker) in warnings[0][2]
    finally:
        window.close()


# --- R2: closing never hangs on a stuck worker -------------------------------------


def test_close_event_does_not_hang_on_a_stuck_solve(qapp, monkeypatch):
    import threading
    import time

    from PySide6.QtCore import QThread
    from PySide6.QtGui import QCloseEvent

    release = threading.Event()

    class _StuckWorker(QThread):
        def cancel(self) -> None:
            pass  # e.g. blocked inside a network read that can't be interrupted

        def run(self) -> None:
            release.wait(30)

    window = MainWindow()
    worker = _StuckWorker()
    worker.start()
    window._worker = worker
    try:
        start = time.monotonic()
        event = QCloseEvent()
        window.closeEvent(event)
        assert time.monotonic() - start < 10
        assert not event.isAccepted()  # deferred until the worker finishes
        assert not window.isVisible()
    finally:
        release.set()
        worker.wait(5000)


# --- B15: the Live Link reply is shown in the app ---------------------------------


def test_live_link_reply_message_reports_what_blender_applied():
    from pose_craft.gui.main_window import _live_link_reply_message

    message, ok = _live_link_reply_message(
        {"ok": True, "applied": ["a"] * 13, "requested": 15, "unmatched": ["foot.L", "foot.R"], "armature": "Armature"}
    )
    assert ok
    assert "13/15" in message and "Armature" in message and "foot.L" in message

    message, ok = _live_link_reply_message({"ok": False, "error": "no armature is selected"})
    assert not ok and "no armature is selected" in message

    message, ok = _live_link_reply_message(None)  # an add-on older than the reply protocol
    assert ok and "update" in message.lower()

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt install; CI's lightweight venv skips it

from pose_craft.estimation.landmarker import DetectedPerson  # noqa: E402
from pose_craft.gui.main_window import MainWindow, _nearest_person_index  # noqa: E402
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

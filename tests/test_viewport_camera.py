from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt/OpenGL install; CI's lightweight venv skips it

from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD  # noqa: E402
from pose_craft.gui.viewport_3d import (  # noqa: E402
    BONE_COLOR,
    LOW_CONFIDENCE_BONE_COLOR,
    _bone_line_colors,
    _compute_camera_frame,
    _engine_to_display,
    _MIN_CAMERA_DISTANCE,
)
from pose_craft.retarget.skeleton import BONE_NAMES  # noqa: E402


def test_compute_camera_frame_centers_on_the_points():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float32)
    center, _distance = _compute_camera_frame(points)
    assert np.allclose(center, points.mean(axis=0), atol=1e-5)


def test_compute_camera_frame_distance_scales_with_skeleton_size():
    small = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    large = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    _, small_distance = _compute_camera_frame(small)
    _, large_distance = _compute_camera_frame(large)
    assert large_distance > small_distance


def test_compute_camera_frame_has_a_minimum_distance_for_degenerate_points():
    coincident = np.zeros((5, 3), dtype=np.float32)
    _, distance = _compute_camera_frame(coincident)
    assert distance == pytest.approx(_MIN_CAMERA_DISTANCE)


def test_bone_line_colors_without_confidence_matches_old_single_color_behavior():
    """confidence=None must preserve show_skeleton's original behavior exactly --
    one shared RGBA tuple for the whole GLLinePlotItem, not a per-vertex array.
    """
    assert _bone_line_colors(None) == BONE_COLOR


def test_bone_line_colors_with_confidence_flags_only_low_confidence_bones():
    # Every bone confident except "thigh.L", which is below LOW_CONFIDENCE_THRESHOLD.
    confidence = {name: 1.0 for name in BONE_NAMES}
    confidence["thigh.L"] = LOW_CONFIDENCE_THRESHOLD - 0.01

    colors = _bone_line_colors(confidence)

    assert colors.shape == (2 * len(BONE_NAMES), 4)
    low_index = BONE_NAMES.index("thigh.L")
    # Two rows (head, tail) per bone, in BONES' order.
    assert np.allclose(colors[2 * low_index], LOW_CONFIDENCE_BONE_COLOR)
    assert np.allclose(colors[2 * low_index + 1], LOW_CONFIDENCE_BONE_COLOR)
    other_index = BONE_NAMES.index("thigh.R")
    assert np.allclose(colors[2 * other_index], BONE_COLOR)
    assert np.allclose(colors[2 * other_index + 1], BONE_COLOR)


def test_bone_line_colors_differs_with_vs_without_confidence_data():
    """The whole point of the new parameter: passing confidence data must actually
    change what gets drawn compared to the default (None) path, when a bone is low.
    """
    confidence = {name: 1.0 for name in BONE_NAMES}
    confidence["spine"] = 0.0

    with_confidence = _bone_line_colors(confidence)
    without_confidence = _bone_line_colors(None)

    assert isinstance(without_confidence, tuple)
    assert isinstance(with_confidence, np.ndarray)
    spine_index = BONE_NAMES.index("spine")
    assert np.allclose(with_confidence[2 * spine_index], LOW_CONFIDENCE_BONE_COLOR)
    assert not np.allclose(LOW_CONFIDENCE_BONE_COLOR, BONE_COLOR)


def test_bone_line_colors_defaults_missing_bones_to_full_confidence():
    """confidence.get(bone.name, 1.0) in _bone_line_colors: a bone absent from the
    dict must not be misdrawn as low-confidence.
    """
    colors = _bone_line_colors({})
    assert np.allclose(colors, np.tile(BONE_COLOR, (2 * len(BONE_NAMES), 1)))


def test_engine_to_display_stands_the_skeleton_up_in_the_z_up_view():
    """pyqtgraph's view is Z-up; the engine is Y-up with the subject facing +Z.
    Engine up must become display up (+Z) and the subject's facing direction
    must become -Y (Blender's front), with their left staying on +X."""
    up, forward, left = _engine_to_display(np.eye(3)[[1, 2, 0]])
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(forward, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(left, [1.0, 0.0, 0.0])

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")  # this module needs a real Qt/OpenGL install; CI's lightweight venv skips it

from pose_craft.gui.viewport_3d import _MIN_CAMERA_DISTANCE, _compute_camera_frame  # noqa: E402


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

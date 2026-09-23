"""A minimal 3D skeleton viewer built on pyqtgraph's OpenGL view, so the app can
show a solved pose next to the source photo before the user exports it.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD
from pose_craft.retarget.skeleton import BONES

BONE_COLOR = (0.3, 0.8, 1.0, 1.0)
JOINT_COLOR = (1.0, 0.6, 0.1, 1.0)
# Matches photo_view._SELECTED_LOW_CONFIDENCE_COLOR (QColor(255, 100, 60)) so the 2D
# photo overlay and the 3D preview agree on what a low-confidence bone looks like.
LOW_CONFIDENCE_BONE_COLOR = (1.0, 100 / 255, 60 / 255, 1.0)

# How much farther back than the skeleton's own radius to place the camera, and
# the floor on that distance so a degenerate (all-coincident) skeleton doesn't
# put the camera implausibly close.
_FRAME_MARGIN = 2.2
_MIN_CAMERA_DISTANCE = 0.5


def _bone_line_colors(confidence: dict[str, float] | None) -> np.ndarray | tuple[float, float, float, float]:
    """Pure math behind show_skeleton's per-bone coloring, pulled out the same way
    _compute_camera_frame is: testable without a real GLViewWidget/OpenGL context.

    Returns either a single RGBA tuple (confidence is None -- every bone segment
    shares BONE_COLOR, matching the pre-confidence-parameter behavior exactly) or a
    (2 * len(BONES), 4) per-vertex array, two identical rows per bone (its head and
    tail endpoint), colored LOW_CONFIDENCE_BONE_COLOR when that bone's score is below
    LOW_CONFIDENCE_THRESHOLD and BONE_COLOR otherwise -- GLLinePlotItem has no
    per-segment coloring primitive, only a single color or one color per vertex.
    """
    if confidence is None:
        return BONE_COLOR
    colors = []
    for bone in BONES:
        bone_color = (
            LOW_CONFIDENCE_BONE_COLOR if confidence.get(bone.name, 1.0) < LOW_CONFIDENCE_THRESHOLD else BONE_COLOR
        )
        colors.append(bone_color)
        colors.append(bone_color)
    return np.array(colors, dtype=np.float32)


def _compute_camera_frame(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure math behind auto-framing: given every drawn point, returns (center,
    distance) so the whole skeleton fits in view regardless of a person's size,
    proportions, or how far from a canonical T-pose they are -- the fixed
    center/distance this replaced only happened to look right for poses near
    the origin at roughly T-pose scale.
    """
    center = points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1))) if len(points) else 0.0
    distance = max(radius * _FRAME_MARGIN, _MIN_CAMERA_DISTANCE)
    return center, distance


class SkeletonViewport(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=3.0, elevation=10, azimuth=45)
        self.opts["center"] = Vector(0, 0.8, 0)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.1, 0.1)
        self.addItem(grid)

        self._bone_lines: gl.GLLinePlotItem | None = None
        self._joints: gl.GLScatterPlotItem | None = None

    def clear_skeleton(self) -> None:
        if self._bone_lines is not None:
            self.removeItem(self._bone_lines)
            self._bone_lines = None
        if self._joints is not None:
            self.removeItem(self._joints)
            self._joints = None

    def show_skeleton(
        self, joint_positions: dict[str, np.ndarray], confidence: dict[str, float] | None = None
    ) -> None:
        """joint_positions: output of retarget.skeleton.joint_positions (engine Y-up frame).

        confidence: optional per-bone score from retarget.skeleton.bone_confidence(). When
        given, each bone segment below LOW_CONFIDENCE_THRESHOLD is drawn in
        LOW_CONFIDENCE_BONE_COLOR instead of BONE_COLOR -- the same threshold/color
        convention photo_view.py already uses for the 2D overlay, reused here so the two
        previews agree. Left as None (the default) for any other caller, which preserves the
        original single-color-for-everything behavior exactly.

        GLLinePlotItem's `color` accepts either one RGBA tuple for the whole item or a
        per-vertex (N, 4) array -- there's no per-segment coloring primitive, so a
        per-bone color is achieved by giving both endpoints of a segment the same color.
        """
        self.clear_skeleton()

        segments = []
        for bone in BONES:
            segments.append(joint_positions[f"{bone.name}.head"])
            segments.append(joint_positions[f"{bone.name}.tail"])

        points = np.array(segments, dtype=np.float32)
        line_color = _bone_line_colors(confidence)
        self._bone_lines = gl.GLLinePlotItem(pos=points, color=line_color, width=3, mode="lines")
        self.addItem(self._bone_lines)

        joints = np.array(list(joint_positions.values()), dtype=np.float32)
        self._joints = gl.GLScatterPlotItem(pos=joints, color=JOINT_COLOR, size=8)
        self.addItem(self._joints)

        # Reframe on every update rather than only once -- a wildly different pose
        # (crouching vs. arms flung wide) can change the skeleton's bounding box
        # enough that the old distance/center would clip it. Passing only `distance`
        # (not elevation/azimuth) leaves the user's own camera rotation alone.
        center, distance = _compute_camera_frame(joints)
        self.opts["center"] = Vector(*center)
        self.setCameraPosition(distance=distance)

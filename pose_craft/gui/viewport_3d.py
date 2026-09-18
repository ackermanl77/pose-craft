"""A minimal 3D skeleton viewer built on pyqtgraph's OpenGL view, so the app can
show a solved pose next to the source photo before the user exports it.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose_craft.retarget.skeleton import BONES

BONE_COLOR = (0.3, 0.8, 1.0, 1.0)
JOINT_COLOR = (1.0, 0.6, 0.1, 1.0)


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

    def show_skeleton(self, joint_positions: dict[str, np.ndarray]) -> None:
        """joint_positions: output of retarget.skeleton.joint_positions (engine Y-up frame)."""
        self.clear_skeleton()

        segments = []
        for bone in BONES:
            head = joint_positions[f"{bone.name}.head"]
            tail = joint_positions[f"{bone.name}.tail"]
            segments.append(head)
            segments.append(tail)

        points = np.array(segments, dtype=np.float32)
        self._bone_lines = gl.GLLinePlotItem(pos=points, color=BONE_COLOR, width=3, mode="lines")
        self.addItem(self._bone_lines)

        joints = np.array(list(joint_positions.values()), dtype=np.float32)
        self._joints = gl.GLScatterPlotItem(pos=joints, color=JOINT_COLOR, size=8)
        self.addItem(self._joints)

"""A minimal 3D skeleton viewer built on pyqtgraph's OpenGL view, so the app can
show a solved pose next to the source photo before the user exports it.

The skeleton is drawn standing on the grid (the caller passes floor-relative
points, see main_window._preview_positions), as tapered octahedral bones --
Blender's own bone shape, which shows each bone's roll -- with line segments
on top; a low-confidence bone's segment is dashed, so it reads without
relying on colour. A small XYZ gizmo, front/side/top camera presets and
optional bone-name labels help judge depth.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD
from pose_craft.retarget.skeleton import BONES

BONE_COLOR = (0.3, 0.8, 1.0, 1.0)
JOINT_COLOR = (1.0, 1.0, 1.0, 1.0)  # neutral: the joints say nothing about confidence
# Matches photo_view.SELECTED_LOW_CONFIDENCE_COLOR (QColor(255, 100, 60)) so the 2D
# photo overlay and the 3D preview agree on what a low-confidence bone looks like.
LOW_CONFIDENCE_BONE_COLOR = (1.0, 100 / 255, 60 / 255, 1.0)
# The raw landmark joints, drawn faintly behind the solved skeleton so any
# disagreement between the landmarks and the exported rotations is visible.
GHOST_COLOR = (0.8, 0.8, 0.8, 0.35)
# Octahedral bone bodies: translucent, so the lines and joints stay visible.
BONE_BODY_COLOR = (0.3, 0.8, 1.0, 0.35)
LOW_CONFIDENCE_BODY_COLOR = (1.0, 100 / 255, 60 / 255, 0.35)
LABEL_COLOR = (230, 230, 230, 255)

# A low-confidence bone's line is split into this many dashes (with equal gaps).
DASHES_PER_BONE = 5
# Octahedron shape: the widest point sits this far along the bone, with this
# half-width relative to the bone's length (Blender's look).
_OCTAHEDRON_WAIST = 0.1
_OCTAHEDRON_HALF_WIDTH = 0.1

# How much farther back than the skeleton's own radius to place the camera, and
# the floor on that distance so a degenerate (all-coincident) skeleton doesn't
# put the camera implausibly close.
_FRAME_MARGIN = 2.2
_MIN_CAMERA_DISTANCE = 0.5

# pyqtgraph's GL view is Z-up (its grid lies in the XY plane and the orbit
# camera's elevation is measured from it), while the engine frame is Y-up --
# drawn as-is, a standing skeleton lay flat on the grid. Engine -> display is
# the same axis change export/blender_pose.py applies (engine +X -> +X, +Y -> +Z,
# +Z -> -Y), so the preview matches how the pose lands in Blender: upright,
# facing -Y. The default azimuth puts the camera in front of the subject
# (-Y side), turned slightly toward their left (+X), so the preview roughly
# matches the photo's viewpoint with a little depth visible.
_ENGINE_TO_DISPLAY = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
_DEFAULT_AZIMUTH = -70
_DEFAULT_ELEVATION = 10

# Camera presets (elevation, azimuth in pyqtgraph's Z-up display frame): the
# subject faces -Y, so "front" looks from -Y (azimuth -90), "side" from the
# subject's left (+X, azimuth 0), "top" straight down.
CAMERA_PRESETS: dict[str, tuple[float, float]] = {
    "front": (0.0, -90.0),
    "side": (0.0, 0.0),
    "top": (90.0, -90.0),
    "default": (_DEFAULT_ELEVATION, _DEFAULT_AZIMUTH),
}


def _engine_to_display(points: np.ndarray) -> np.ndarray:
    """(N, 3) engine-frame (Y-up) points -> pyqtgraph's Z-up display frame."""
    return np.asarray(points, dtype=np.float64) @ _ENGINE_TO_DISPLAY.T


def _is_low_confidence(confidence: dict[str, float] | None, bone_name: str) -> bool:
    return confidence is not None and confidence.get(bone_name, 1.0) < LOW_CONFIDENCE_THRESHOLD


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
        bone_color = LOW_CONFIDENCE_BONE_COLOR if _is_low_confidence(confidence, bone.name) else BONE_COLOR
        colors.append(bone_color)
        colors.append(bone_color)
    return np.array(colors, dtype=np.float32)


def _bone_segments(
    joint_positions: dict[str, np.ndarray], confidence: dict[str, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, per-vertex colors) for one GLLinePlotItem in "lines" mode,
    engine frame: a solid head-tail segment per bone, except low-confidence
    bones, which become DASHES_PER_BONE dashes -- the same solid/dashed
    distinction the photo overlay draws, so confidence doesn't rely on hue."""
    vertices, colors = [], []
    for bone in BONES:
        head = np.asarray(joint_positions[f"{bone.name}.head"], dtype=float)
        tail = np.asarray(joint_positions[f"{bone.name}.tail"], dtype=float)
        if _is_low_confidence(confidence, bone.name):
            steps = np.linspace(0.0, 1.0, 2 * DASHES_PER_BONE)
            for start, end in zip(steps[0::2], steps[1::2]):
                vertices += [head + (tail - head) * start, head + (tail - head) * end]
                colors += [LOW_CONFIDENCE_BONE_COLOR] * 2
        else:
            vertices += [head, tail]
            colors += [BONE_COLOR] * 2
    return np.array(vertices), np.array(colors, dtype=np.float32)


def octahedron_mesh(head: np.ndarray, tail: np.ndarray, lateral: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A Blender-style tapered octahedral bone from head to tail: (6 vertices,
    8 triangular faces). The four waist vertices sit _OCTAHEDRON_WAIST along
    the bone, two of them along `lateral` (the bone's roll axis, projected
    perpendicular to the bone), so the shape turns with the bone's roll."""
    head = np.asarray(head, dtype=float)
    axis = np.asarray(tail, dtype=float) - head
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        return np.repeat(head[None], 6, axis=0), np.zeros((0, 3), dtype=np.int32)
    direction = axis / length
    side = np.asarray(lateral, dtype=float) - direction * np.dot(lateral, direction)
    if np.linalg.norm(side) < 1e-9:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        side = fallback - direction * np.dot(fallback, direction)
    side /= np.linalg.norm(side)
    other = np.cross(direction, side)
    waist = head + axis * _OCTAHEDRON_WAIST
    r = length * _OCTAHEDRON_HALF_WIDTH
    vertices = np.array([head, waist + side * r, waist + other * r, waist - side * r, waist - other * r, head + axis])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int32)
    return vertices, faces


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
        self.setCameraPosition(distance=3.0, elevation=_DEFAULT_ELEVATION, azimuth=_DEFAULT_AZIMUTH)
        self.opts["center"] = Vector(0, 0, 0)

        # The grid is the floor: the app stands the skeleton on z = 0.
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.1, 0.1)
        self.addItem(grid)
        # A small XYZ gizmo at the origin (display axes: X red = subject's
        # left, Y green = behind the subject, Z blue = up).
        self._axes = gl.GLAxisItem()
        self._axes.setSize(0.2, 0.2, 0.2)
        self.addItem(self._axes)

        self.show_labels = False
        self._bone_lines: gl.GLLinePlotItem | None = None
        self._bone_bodies: gl.GLMeshItem | None = None
        self._joints: gl.GLScatterPlotItem | None = None
        self._ghost: gl.GLScatterPlotItem | None = None
        self._labels: list[gl.GLTextItem] = []
        self._last_frame: tuple[np.ndarray, float] | None = None

    def clear_skeleton(self) -> None:
        for name in ("_bone_lines", "_bone_bodies", "_joints", "_ghost"):
            item = getattr(self, name)
            if item is not None:
                self.removeItem(item)
                setattr(self, name, None)
        for label in self._labels:
            self.removeItem(label)
        self._labels = []

    def set_camera_preset(self, name: str) -> None:
        """Front, side or top view (or the default three-quarter one), framed
        on the current skeleton."""
        elevation, azimuth = CAMERA_PRESETS[name]
        if self._last_frame is not None:
            center, distance = self._last_frame
            self.opts["center"] = Vector(*center)
            self.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        else:
            self.setCameraPosition(elevation=elevation, azimuth=azimuth)

    def frame_skeleton(self) -> None:
        """Re-centres and re-fits the camera on the skeleton, keeping its angle."""
        if self._last_frame is not None:
            center, distance = self._last_frame
            self.opts["center"] = Vector(*center)
            self.setCameraPosition(distance=distance)

    def show_skeleton(
        self,
        joint_positions: dict[str, np.ndarray],
        confidence: dict[str, float] | None = None,
        ghost_points: dict[str, np.ndarray] | None = None,
        lateral_axes: dict[str, np.ndarray] | None = None,
    ) -> None:
        """joint_positions: '<bone>.head'/'<bone>.tail' points in the engine Y-up
        frame (converted to the view's Z-up frame here -- see _engine_to_display):
        the app passes retarget.skeleton.forward_kinematics() of the exported
        rotations, so the preview shows what the export says.

        ghost_points: optional points in the same layout (the raw landmark
        joints), drawn as faint dots behind the skeleton.

        confidence: optional per-bone score from retarget.skeleton.bone_confidence().
        A bone below LOW_CONFIDENCE_THRESHOLD is drawn dashed in
        LOW_CONFIDENCE_BONE_COLOR -- the same convention as photo_view.py.

        lateral_axes: optional bone name -> the bone's roll axis (engine frame,
        see skeleton.bone_lateral_axes); with it every bone also gets an
        octahedral body that turns with its roll.
        """
        self.clear_skeleton()

        vertices, colors = _bone_segments(joint_positions, confidence)
        self._bone_lines = gl.GLLinePlotItem(
            pos=_engine_to_display(vertices).astype(np.float32), color=colors, width=3, mode="lines"
        )
        self.addItem(self._bone_lines)

        if lateral_axes:
            mesh_vertices, mesh_faces, face_colors = [], [], []
            for bone in BONES:
                lateral = lateral_axes.get(bone.name)
                if lateral is None:
                    continue
                verts, faces = octahedron_mesh(
                    joint_positions[f"{bone.name}.head"], joint_positions[f"{bone.name}.tail"], lateral
                )
                mesh_faces.append(faces + 6 * len(mesh_vertices))
                mesh_vertices.append(verts)
                color = LOW_CONFIDENCE_BODY_COLOR if _is_low_confidence(confidence, bone.name) else BONE_BODY_COLOR
                face_colors += [color] * len(faces)
            if mesh_vertices:
                self._bone_bodies = gl.GLMeshItem(
                    vertexes=_engine_to_display(np.concatenate(mesh_vertices)).astype(np.float32),
                    faces=np.concatenate(mesh_faces),
                    faceColors=np.array(face_colors, dtype=np.float32),
                    smooth=False,
                    drawEdges=False,
                    glOptions="translucent",
                )
                self.addItem(self._bone_bodies)

        joints = _engine_to_display(np.array(list(joint_positions.values()))).astype(np.float32)
        self._joints = gl.GLScatterPlotItem(pos=joints, color=JOINT_COLOR, size=7)
        self.addItem(self._joints)

        if ghost_points:
            ghost = _engine_to_display(np.array(list(ghost_points.values()))).astype(np.float32)
            self._ghost = gl.GLScatterPlotItem(pos=ghost, color=GHOST_COLOR, size=6)
            self.addItem(self._ghost)

        if self.show_labels:
            for bone in BONES:
                middle = (np.asarray(joint_positions[f"{bone.name}.head"]) + joint_positions[f"{bone.name}.tail"]) / 2
                label = gl.GLTextItem(pos=_engine_to_display(middle[None])[0], text=bone.name, color=LABEL_COLOR)
                self._labels.append(label)
                self.addItem(label)

        # Reframe on every update rather than only once -- a wildly different pose
        # (crouching vs. arms flung wide) can change the skeleton's bounding box
        # enough that the old distance/center would clip it. Passing only `distance`
        # (not elevation/azimuth) leaves the user's own camera rotation alone.
        self._last_frame = _compute_camera_frame(joints)
        self.frame_skeleton()

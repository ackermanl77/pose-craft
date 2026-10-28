"""A minimal 3D skeleton viewer built on pyqtgraph's OpenGL view, so the app can
show a solved pose next to the source photo before the user exports it.

The skeleton is drawn standing on the grid (the caller passes floor-relative
points, see main_window._preview_positions), as tapered octahedral bones --
Blender's own bone shape, which shows each bone's roll -- with line segments
on top; a low-confidence bone's segment is dashed, so it reads without
relying on colour. A small XYZ gizmo, front/side/top camera presets and
optional bone-name labels help judge depth.

F9: hovering a bone shows its name, local Euler angles (about its own bone
axes, as in Blender) and confidence in a small label; and the photo can be
shown as a backdrop plane behind the skeleton, placed with the person's own
weak-perspective fit (skeleton.image_to_world_scale) so the photo's body
lines up with the raw landmarks. Finger bones (B11) are drawn when present.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Transform3D, Vector
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QVector4D
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from pose_craft.export.blender_pose import low_confidence_threshold
from pose_craft.retarget.skeleton import ALL_BONE_NAMES, BONES, bone_local_euler_deg, image_to_world_scale

# A hover within this many pixels of a bone's drawn segment labels it.
HOVER_MAX_DISTANCE_PX = 12.0
# The backdrop texture's longest side (a photo needn't be uploaded at full size).
BACKDROP_MAX_SIDE = 768

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


# F16: the backdrop's default opacity, and the field of view of "View from the
# photo's camera". The backdrop comes from a weak-perspective fit (an
# orthographic camera), so that view looks from far away with a narrow field
# of view, where perspective all but vanishes and the photo lines up with the
# skeleton the way the fit sees it. The other presets use pyqtgraph's default.
DEFAULT_BACKDROP_OPACITY = 200 / 255
PHOTO_CAMERA_FOV = 6.0
DEFAULT_FOV = 60.0
_PHOTO_CAMERA_MARGIN = 1.08


def photo_camera_view(
    corners_engine: np.ndarray, view_aspect: float, fov_degrees: float = PHOTO_CAMERA_FOV
) -> tuple[np.ndarray, float]:
    """(display-frame centre, distance) that frames the backdrop between
    `corners_engine` straight on (the "front" direction: the backdrop is a
    constant-depth plane facing the camera) in a view `view_aspect` (width /
    height) wide whose *horizontal* field of view is `fov_degrees`
    (pyqtgraph's convention), the whole photo visible with a small margin."""
    corners = _engine_to_display(np.asarray(corners_engine, dtype=float))
    center = corners.mean(axis=0)
    half_width = 0.5 * float(np.linalg.norm(corners[1] - corners[0])) * _PHOTO_CAMERA_MARGIN
    half_height = 0.5 * float(np.linalg.norm(corners[3] - corners[0])) * _PHOTO_CAMERA_MARGIN
    tangent = np.tan(np.radians(fov_degrees) / 2.0)
    distance = max(half_width / tangent, half_height * max(view_aspect, 1e-6) / tangent)
    return center, float(distance)


# UI3: the axis gizmo's colours (the display axes, as the origin's GLAxisItem draws them) and the
# zoom step of the overlay's - / + buttons.
AXIS_COLORS = {"X": "#ff5555", "Y": "#55dd55", "Z": "#5599ff"}
ZOOM_STEP = 1.25


def axis_gizmo_directions(elevation_deg: float, azimuth_deg: float) -> dict[str, tuple[float, float, float]]:
    """Where the display axes X, Y, Z point on screen for pyqtgraph's orbit camera at
    `elevation_deg` / `azimuth_deg`: (right, down, toward the viewer) for each unit axis. The
    camera sits at centre + distance * (cos el cos az, cos el sin az, sin el), looking at the centre
    with Z up."""
    elevation, azimuth = np.radians(elevation_deg), np.radians(azimuth_deg)
    toward_camera = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(toward_camera, right)
    return {name: (float(axis @ right), float(-(axis @ up)), float(axis @ toward_camera))
            for name, axis in zip("XYZ", np.eye(3))}


class _AxisGizmo(QWidget):
    """The corner axis indicator (UI3): which way X, Y and Z point from the current camera."""

    SIZE = 64

    def __init__(self, viewport: SkeletonViewport):
        super().__init__(viewport)
        self._viewport = viewport
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAccessibleName("Axis orientation")
        self._painted_for: tuple[float, float] | None = None

    def camera_angles(self) -> tuple[float, float]:
        return float(self._viewport.opts["elevation"]), float(self._viewport.opts["azimuth"])

    def refresh_if_moved(self) -> None:
        if self.camera_angles() != self._painted_for:
            self.update()

    def paintEvent(self, event) -> None:
        self._painted_for = self.camera_angles()
        directions = axis_gizmo_directions(*self._painted_for)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        centre = QPointF(self.SIZE / 2, self.SIZE / 2)
        reach = self.SIZE / 2 - 12
        # Axes pointing away from the viewer first, so the nearer ones are drawn on top.
        for name, (dx, dy, depth) in sorted(directions.items(), key=lambda item: item[1][2]):
            colour = QColor(AXIS_COLORS[name])
            if depth < 0:
                colour.setAlpha(140)
            tip = QPointF(centre.x() + dx * reach, centre.y() + dy * reach)
            painter.setPen(QPen(colour, 2))
            painter.drawLine(centre, tip)
            painter.drawText(QPointF(tip.x() + dx * 6 - 4, tip.y() + dy * 6 + 4), name)
        painter.end()


class _NavigationBar(QWidget):
    """View presets, fit and zoom over the preview's corner (UI3), like Blender's viewport buttons."""

    def __init__(self, viewport: SkeletonViewport):
        super().__init__(viewport)
        self.setObjectName("viewportNavigation")
        self.setAccessibleName("3D preview navigation")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setStyleSheet(
            "#viewportNavigation { background: rgba(20, 20, 20, 190); border-radius: 4px; }"
            "#viewportNavigation QToolButton { color: #eee; padding: 2px 6px; border: none; }"
            "#viewportNavigation QToolButton:hover { background: rgba(255, 255, 255, 40); }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.buttons: dict[str, QToolButton] = {}
        for key, text, tip, slot in (
            ("front", "Front", "Front view (1)", lambda: viewport.set_camera_preset("front")),
            ("side", "Side", "Side view (3)", lambda: viewport.set_camera_preset("side")),
            ("top", "Top", "Top view (7)", lambda: viewport.set_camera_preset("top")),
            ("default", "3/4", "The default three-quarter view (0)", lambda: viewport.set_camera_preset("default")),
            ("fit", "Fit", "Frame the skeleton (F)", viewport.frame_skeleton),
            ("zoom_out", "\u2212", "Zoom out", lambda: viewport.zoom(ZOOM_STEP)),
            ("zoom_in", "+", "Zoom in", lambda: viewport.zoom(1 / ZOOM_STEP)),
        ):
            button = QToolButton(self)
            button.setText(text)
            button.setToolTip(tip)
            button.setAccessibleName(tip.split(" (")[0])
            button.setFocusPolicy(Qt.TabFocus)
            button.clicked.connect(slot)
            layout.addWidget(button)
            self.buttons[key] = button


def _engine_to_display(points: np.ndarray) -> np.ndarray:
    """(N, 3) engine-frame (Y-up) points -> pyqtgraph's Z-up display frame."""
    return np.asarray(points, dtype=np.float64) @ _ENGINE_TO_DISPLAY.T


def _drawn_bones(joint_positions: dict[str, np.ndarray]) -> list[str]:
    """Every bone (body first, then fingers) whose head and tail are given."""
    return [n for n in ALL_BONE_NAMES if f"{n}.head" in joint_positions and f"{n}.tail" in joint_positions]


def _is_low_confidence(confidence: dict[str, float] | None, bone_name: str) -> bool:
    return confidence is not None and confidence.get(bone_name, 1.0) < low_confidence_threshold()


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
    for name in _drawn_bones(joint_positions):
        head = np.asarray(joint_positions[f"{name}.head"], dtype=float)
        tail = np.asarray(joint_positions[f"{name}.tail"], dtype=float)
        if _is_low_confidence(confidence, name):
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


def photo_backdrop_corners(
    landmarks: np.ndarray,
    image_xy: np.ndarray,
    aspect: float,
    visibility: np.ndarray | None = None,
    depth_margin: float = 0.25,
) -> np.ndarray | None:
    """The photo's four corners (top-left, top-right, bottom-right,
    bottom-left), engine frame, in the same space as the raw landmarks: the
    weak-perspective fit maps every image point onto the X/Y of the world
    point it shows, and the plane sits `depth_margin` (of the person's
    height) behind the person's deepest landmark. None when no fit exists."""
    scale, world_center, image_center = image_to_world_scale(landmarks, image_xy, aspect, visibility)
    if scale <= 1e-9:
        return None
    landmarks = np.asarray(landmarks, dtype=float)
    height = float(np.ptp(landmarks[:, 1])) or 1.0
    z = float(landmarks[:, 2].min()) - depth_margin * height
    corners = []
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        u, v = x * aspect, y
        corners.append([
            (u - image_center[0]) / scale + world_center[0],
            -(v - image_center[1]) / scale + world_center[1],
            z,
        ])
    return np.array(corners)


def backdrop_transform(corners_engine: np.ndarray, width: int, height: int) -> Transform3D:
    """The GLImageItem transform drawing a width x height image (its data
    indexed [x, y], y = image row) between the given engine-frame corners."""
    tl, tr, _br, bl = _engine_to_display(corners_engine)
    ex, ey = (tr - tl) / width, (bl - tl) / height
    normal = np.cross(ex, ey)
    normal = normal / (np.linalg.norm(normal) or 1.0)
    return Transform3D(
        ex[0], ey[0], normal[0], tl[0],
        ex[1], ey[1], normal[1], tl[1],
        ex[2], ey[2], normal[2], tl[2],
        0.0, 0.0, 0.0, 1.0,
    )


def _segment_distance_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denominator = float(np.dot(ab, ab))
    t = 0.0 if denominator < 1e-18 else float(np.clip(np.dot(point - a, ab) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def hover_text(name: str, rotation, confidence: float | None) -> str:
    """The hover label: bone name, local Euler angles about its own axes, confidence."""
    text = name
    if rotation is not None:
        x, y, z = bone_local_euler_deg(name, rotation)
        text += f"\nX {x:+.1f}°  Y {y:+.1f}°  Z {z:+.1f}°"
    if confidence is not None:
        text += f"\nconfidence {confidence:.0%}"
    return text


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
        self.hover_enabled = True
        self._backdrop: gl.GLImageItem | None = None
        self._backdrop_rgba: np.ndarray | None = None
        self._backdrop_corners: np.ndarray | None = None
        self.backdrop_opacity = DEFAULT_BACKDROP_OPACITY
        self._joint_positions: dict[str, np.ndarray] = {}
        self._rotations: dict | None = None
        self._confidence: dict[str, float] | None = None
        self._hover_label = QLabel(self)
        self._hover_label.setStyleSheet("background:rgba(20,20,20,210); color:#eee; padding:3px; border-radius:3px;")
        self._hover_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hover_label.hide()
        # UI3: on-screen navigation, as in Blender -- the same views as the View menu, plus zoom.
        self.navigation = _NavigationBar(self)
        self.axis_gizmo = _AxisGizmo(self)
        self.setMouseTracking(True)
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
        self._joint_positions = {}
        self._hover_label.hide()

    # -- photo backdrop (F9) -----------------------------------------------------

    def set_backdrop(self, image_rgb: np.ndarray | None, corners_engine: np.ndarray | None = None) -> None:
        """Shows `image_rgb` (H, W, 3 uint8) between the engine-frame corners
        (see photo_backdrop_corners), or removes the backdrop with None."""
        if self._backdrop is not None:
            self.removeItem(self._backdrop)
            self._backdrop = None
        self._backdrop_rgba = None
        self._backdrop_corners = None
        if image_rgb is None or corners_engine is None:
            return
        import cv2

        h, w = image_rgb.shape[:2]
        if max(h, w) > BACKDROP_MAX_SIDE:
            factor = BACKDROP_MAX_SIDE / max(h, w)
            image_rgb = cv2.resize(image_rgb, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA)
            h, w = image_rgb.shape[:2]
        rgba = np.empty((w, h, 4), dtype=np.ubyte)
        rgba[..., :3] = np.transpose(image_rgb, (1, 0, 2))
        rgba[..., 3] = round(255 * self.backdrop_opacity)
        self._backdrop_rgba = rgba
        self._backdrop_corners = np.asarray(corners_engine, dtype=float)
        self._backdrop = gl.GLImageItem(rgba, glOptions="translucent")
        self._backdrop.setTransform(backdrop_transform(corners_engine, w, h))
        self.addItem(self._backdrop)

    def has_backdrop(self) -> bool:
        return self._backdrop is not None

    def set_backdrop_opacity(self, opacity: float) -> None:
        """F16: 0 (invisible) .. 1 (opaque); kept for the next backdrop too."""
        self.backdrop_opacity = float(min(1.0, max(0.0, opacity)))
        if self._backdrop is not None and self._backdrop_rgba is not None:
            self._backdrop_rgba[..., 3] = round(255 * self.backdrop_opacity)
            self._backdrop.setData(self._backdrop_rgba)

    def view_from_photo_camera(self) -> bool:
        """F16: look at the backdrop straight on from (nearly) the camera the
        pose was fitted with, so the photo and the skeleton overlay. False
        when there's no backdrop to align to."""
        if self._backdrop_corners is None:
            return False
        center, distance = photo_camera_view(self._backdrop_corners, max(1, self.width()) / max(1, self.height()))
        elevation, azimuth = CAMERA_PRESETS["front"]
        self.opts["center"] = Vector(*center)
        self.opts["fov"] = PHOTO_CAMERA_FOV
        self.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        return True

    # -- hover label (F9) --------------------------------------------------------

    def project_to_screen(self, points_engine: np.ndarray) -> np.ndarray:
        """(N, 3) engine-frame points -> (N, 2) widget pixel positions with the
        current camera (NaN for points behind it)."""
        viewport = (0, 0, max(1, self.width()), max(1, self.height()))
        matrix = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
        out = []
        for point in _engine_to_display(np.atleast_2d(points_engine)):
            clip = matrix.map(QVector4D(float(point[0]), float(point[1]), float(point[2]), 1.0))
            if clip.w() <= 1e-9:
                out.append([np.nan, np.nan])
                continue
            ndc_x, ndc_y = clip.x() / clip.w(), clip.y() / clip.w()
            out.append([(ndc_x + 1.0) / 2.0 * viewport[2], (1.0 - ndc_y) / 2.0 * viewport[3]])
        return np.array(out)

    def bone_at(self, x: float, y: float) -> str | None:
        """The drawn bone nearest the widget pixel (x, y), within HOVER_MAX_DISTANCE_PX."""
        names = _drawn_bones(self._joint_positions)
        if not names:
            return None
        heads = self.project_to_screen(np.array([self._joint_positions[f"{n}.head"] for n in names]))
        tails = self.project_to_screen(np.array([self._joint_positions[f"{n}.tail"] for n in names]))
        point = np.array([x, y])
        best, best_distance = None, HOVER_MAX_DISTANCE_PX
        for name, head, tail in zip(names, heads, tails):
            if np.isnan(head).any() or np.isnan(tail).any():
                continue
            distance = _segment_distance_2d(point, head, tail)
            if distance <= best_distance:
                best, best_distance = name, distance
        return best

    def hover_at(self, x: float, y: float) -> str | None:
        """Shows (or hides) the hover label for the bone under (x, y); returns its name."""
        name = self.bone_at(x, y) if self.hover_enabled else None
        if name is None:
            self._hover_label.hide()
            return None
        rotation = (self._rotations or {}).get(name)
        confidence = None if self._confidence is None else self._confidence.get(name)
        self._hover_label.setText(hover_text(name, rotation, confidence))
        self._hover_label.adjustSize()
        left = min(int(x) + 14, max(0, self.width() - self._hover_label.width()))
        top = min(int(y) + 14, max(0, self.height() - self._hover_label.height()))
        self._hover_label.move(left, top)
        self._hover_label.show()
        self._hover_label.raise_()
        return name

    def hover_label_text(self) -> str:
        return "" if self._hover_label.isHidden() else self._hover_label.text()

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons() == Qt.NoButton:
            pos = ev.position() if hasattr(ev, "position") else QPointF(ev.pos())
            self.hover_at(pos.x(), pos.y())
            return
        self._hover_label.hide()
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._hover_label.hide()
        super().leaveEvent(ev)

    def zoom(self, factor: float) -> None:
        """Moves the camera `factor` times as far from the centre (below 1 zooms in)."""
        distance = float(self.opts["distance"]) * factor
        self.setCameraPosition(distance=max(_MIN_CAMERA_DISTANCE * 0.25, distance))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.navigation.adjustSize()
        self.navigation.move(max(0, self.width() - self.navigation.width() - 6), 6)
        self.axis_gizmo.move(6, max(0, self.height() - self.axis_gizmo.height() - 6))

    def paintGL(self, *args, **kwargs) -> None:
        super().paintGL(*args, **kwargs)
        self.axis_gizmo.refresh_if_moved()  # only when the camera turned: no repaint loop

    def set_camera_preset(self, name: str) -> None:
        """Front, side or top view (or the default three-quarter one), framed
        on the current skeleton."""
        elevation, azimuth = CAMERA_PRESETS[name]
        self.opts["fov"] = DEFAULT_FOV  # back from the photo camera's narrow one
        if self._last_frame is not None:
            center, distance = self._last_frame
            self.opts["center"] = Vector(*center)
            self.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        else:
            self.setCameraPosition(elevation=elevation, azimuth=azimuth)

    def frame_skeleton(self) -> None:
        """Re-centres and re-fits the camera on the skeleton, keeping its angle."""
        if self._last_frame is not None:
            self.opts["fov"] = DEFAULT_FOV
            center, distance = self._last_frame
            self.opts["center"] = Vector(*center)
            self.setCameraPosition(distance=distance)

    def show_skeleton(
        self,
        joint_positions: dict[str, np.ndarray],
        confidence: dict[str, float] | None = None,
        ghost_points: dict[str, np.ndarray] | None = None,
        lateral_axes: dict[str, np.ndarray] | None = None,
        rotations: dict | None = None,
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

        rotations: optional bone name -> local Rotation, for the hover label's
        Euler angles (F9).
        """
        self.clear_skeleton()
        self._joint_positions = dict(joint_positions)
        self._rotations = rotations
        self._confidence = confidence

        vertices, colors = _bone_segments(joint_positions, confidence)
        self._bone_lines = gl.GLLinePlotItem(
            pos=_engine_to_display(vertices).astype(np.float32), color=colors, width=3, mode="lines"
        )
        self.addItem(self._bone_lines)

        if lateral_axes:
            mesh_vertices, mesh_faces, face_colors = [], [], []
            for name in _drawn_bones(joint_positions):
                lateral = lateral_axes.get(name)
                if lateral is None:
                    continue
                verts, faces = octahedron_mesh(joint_positions[f"{name}.head"], joint_positions[f"{name}.tail"], lateral)
                mesh_faces.append(faces + 6 * len(mesh_vertices))
                mesh_vertices.append(verts)
                color = LOW_CONFIDENCE_BODY_COLOR if _is_low_confidence(confidence, name) else BONE_BODY_COLOR
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

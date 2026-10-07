"""The photo display widget: keeps the full-resolution QPixmap around so it can
rescale on window resize (rather than blurring a pixmap that was already
downscaled once), turns a left click into a normalized (0..1, 0..1)
image-space point so the main window can pick whichever detected person is
nearest -- PoseForge lets you click the person directly in the photo, this is
the same idea -- and a right click into a context-menu request (manual
correction of the bone under it), and draws the solved 2D skeleton for every
detected person directly on top of the photo, so detection quality (and which
person is selected) is visible in context instead of only in the separate 3D
viewport.

Confidence is encoded by line style as well as colour (F3): a low-confidence
bone is dashed, so it reads for colour-blind users and on busy photos too.
Joints are neutral white -- they carry no confidence meaning.

F4: the selected person's main joints (shoulders, elbows, wrists, hips,
knees, ankles) are drawn as rings that can be dragged; releasing a drag
emits joint_dragged(landmark index, x, y) and the main window re-solves the
limb (as an undo step). A press that barely moves is still a plain click.
Finger bones (B11) are drawn thinner when a person has them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from pose_craft.export.blender_pose import low_confidence_threshold
from pose_craft.retarget.skeleton import ALL_BONE_NAMES, FINGER_BONE_NAMES

SELECTED_BONE_COLOR = QColor(77, 204, 255)  # matches viewport_3d.BONE_COLOR
SELECTED_LOW_CONFIDENCE_COLOR = QColor(255, 100, 60)  # matches viewport_3d.LOW_CONFIDENCE_BONE_COLOR
SELECTED_JOINT_COLOR = QColor(255, 255, 255)  # matches viewport_3d.JOINT_COLOR
UNSELECTED_COLOR = QColor(190, 190, 190, 150)  # dim/translucent so the selected person pops
HOVER_COLOR = QColor(255, 214, 90)
HOVER_PERSON_COLOR = QColor(130, 220, 255, 220)
DRAG_HANDLE_COLOR = QColor(255, 220, 90)
# A press within this many pixels of a handle grabs it; a drag shorter than
# DRAG_MIN_PIXELS is a click.
DRAG_GRAB_PIXELS = 10.0
DRAG_MIN_PIXELS = 3.0
_FINGER_BONES = set(FINGER_BONE_NAMES)


@dataclass
class PhotoOverlayPerson:
    """One detected person's overlay data, in the photo's own 2D image space.

    Deliberately independent of the mirror checkbox: mirroring relabels which
    *exported bone name* a side's data feeds (see main_window._selected_landmarks),
    it doesn't move anyone's arm in the photo. Drawing raw, unmirrored positions
    keeps the overlay honestly showing "what MediaPipe saw here", and pairs each
    bone with the confidence of the real landmarks that produced its position
    (also unmirrored) rather than a mirror-swapped value that wouldn't match.
    """

    joints_2d: dict[str, np.ndarray]
    confidence: dict[str, float]
    selected: bool
    label: str = ""  # how the hover tooltip names this person ("Person 2"); empty: by position
    out_of_frame: frozenset = frozenset()  # bones reaching outside the photo (B10)


def bone_pen_style(selected: bool, confidence: float, out_of_frame: bool = False) -> tuple[QColor, float, Qt.PenStyle]:
    """(colour, width, line style) for one bone of the overlay: the selected
    person's bones in the bone colour, solid, or -- below
    LOW_CONFIDENCE_THRESHOLD -- in the warning colour and dashed; everyone
    else thin and dim."""
    if not selected:
        return UNSELECTED_COLOR, 1.5, Qt.SolidLine
    if out_of_frame:  # B10: cut off by the frame -- a guess, not just hidden
        return SELECTED_LOW_CONFIDENCE_COLOR, 3.0, Qt.DotLine
    if confidence < low_confidence_threshold():
        return SELECTED_LOW_CONFIDENCE_COLOR, 3.0, Qt.DashLine
    return SELECTED_BONE_COLOR, 3.0, Qt.SolidLine


def legend_entries() -> list[tuple[QColor, Qt.PenStyle, str]]:
    """The legend under the photo, drawn from the same constants and styles the
    overlay uses (it used to be emoji whose colours matched nothing drawn)."""
    return [
        (SELECTED_BONE_COLOR, Qt.SolidLine, "Bone"),
        (SELECTED_LOW_CONFIDENCE_COLOR, Qt.DashLine, "Low confidence"),
        (SELECTED_LOW_CONFIDENCE_COLOR, Qt.DotLine, "Out of frame"),
        (UNSELECTED_COLOR, Qt.SolidLine, "Other people"),
    ]


def legend_swatch(color: QColor, style: Qt.PenStyle, width: int = 28, height: int = 12) -> QPixmap:
    """A small line sample for the legend, in the given colour and style."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        pen = QPen(color)
        pen.setWidthF(3.0)
        pen.setStyle(style)
        painter.setPen(pen)
        painter.drawLine(QPointF(1, height / 2), QPointF(width - 1, height / 2))
    finally:
        painter.end()
    return pixmap


class PhotoView(QLabel):
    clicked_normalized = Signal(float, float)
    # Right click: normalized image point plus the global screen position for a menu.
    context_requested = Signal(float, float, QPoint)
    # F4: a joint handle dragged to a new normalized point (landmark index, x, y).
    joint_dragged = Signal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setStyleSheet("background:#222; color:#aaa; border:1px solid #444;")
        self.setText("No photo loaded — open one, or drop an image file here")
        self.setWordWrap(True)
        self._source_pixmap: QPixmap | None = None
        self._overlay_people: list[PhotoOverlayPerson] = []
        self._handles: dict[int, np.ndarray] = {}
        self._drag: dict | None = None  # {"joint", "start" (widget px), "point" (normalized)}
        self._hover_person: int | None = None
        self._hover_bone: str | None = None
        self._hover_joint: str | None = None
        self._hover_pos: QPointF | None = None  # last mouse position over the widget, for re-hit-testing
        # Hover highlighting (F5): the bone/joint/person under the mouse is
        # highlighted and named in a tooltip, so mouse tracking is on.
        self.setMouseTracking(True)

    # -- joint dragging (F4) ---------------------------------------------------

    def set_drag_handles(self, handles: dict) -> None:
        """landmark index -> normalized (x, y) of the selected person's
        draggable joints; {} turns dragging off."""
        self._handles = {int(k): np.asarray(v, dtype=float) for k, v in handles.items()}
        if self._drag is not None and self._drag["joint"] not in self._handles:
            self._drag = None
        self.update()

    def drag_handles(self) -> dict[int, np.ndarray]:
        return dict(self._handles)

    def _widget_point(self, normalized_xy) -> QPointF | None:
        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return None
        pixmap, offset_x, offset_y = geometry
        return QPointF(offset_x + float(normalized_xy[0]) * pixmap.width(), offset_y + float(normalized_xy[1]) * pixmap.height())

    def handle_at(self, pos) -> int | None:
        """The drag handle within DRAG_GRAB_PIXELS of widget position `pos`."""
        best, best_distance = None, DRAG_GRAB_PIXELS
        for joint, xy in self._handles.items():
            point = self._widget_point(xy)
            if point is None:
                return None
            distance = float(np.hypot(point.x() - pos.x(), point.y() - pos.y()))
            if distance <= best_distance:
                best, best_distance = joint, distance
        return best

    def begin_drag(self, pos) -> bool:
        joint = self.handle_at(pos)
        if joint is None:
            return False
        self._drag = {"joint": joint, "start": QPointF(pos), "point": self._handles[joint].copy()}
        return True

    def drag_to(self, pos) -> None:
        if self._drag is None:
            return
        point = self._clamped_normalized(pos)
        if point is not None:
            self._drag["point"] = np.array(point)
            self.update()

    def end_drag(self, pos) -> bool:
        """Finishes a drag: emits joint_dragged and returns True, or returns
        False when the press barely moved (the caller treats it as a click)."""
        drag, self._drag = self._drag, None
        if drag is None:
            return False
        moved = float(np.hypot(pos.x() - drag["start"].x(), pos.y() - drag["start"].y()))
        self.update()
        if moved < DRAG_MIN_PIXELS:
            return False
        self.drag_to(pos)
        point = self._clamped_normalized(pos)
        if point is None:
            return False
        self.joint_dragged.emit(drag["joint"], float(point[0]), float(point[1]))
        return True

    def _clamped_normalized(self, pos) -> tuple[float, float] | None:
        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return None
        pixmap, offset_x, offset_y = geometry
        x = min(max((pos.x() - offset_x) / pixmap.width(), 0.0), 1.0)
        y = min(max((pos.y() - offset_y) / pixmap.height(), 0.0), 1.0)
        return x, y

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap()

    def clear_photo(self) -> None:
        self._source_pixmap = None
        self._overlay_people = []
        self._clear_hover()
        self.clear()
        self.setText("No photo loaded")

    def set_skeleton_overlay(self, people: list[PhotoOverlayPerson]) -> None:
        """people: every detected person to draw, in whatever order -- pass []
        to clear. Triggers a repaint; the actual drawing happens in paintEvent.
        """
        self._overlay_people = people
        # A redraw while the mouse rests (an edit, the next video frame) keeps
        # highlighting whatever is under it now, not a stale index.
        if self._hover_pos is not None:
            self._set_hover(self._hover_pos)
        else:
            self._clear_hover()
        self.update()

    def hover_info(self) -> tuple[int | None, str | None, str | None]:
        """(person index, bone, joint key) under the mouse; None for each
        that isn't there."""
        return self._hover_person, self._hover_bone, self._hover_joint

    def hover_text(self) -> str:
        """What the tooltip says about the hovered person/bone/joint ("" for nothing)."""
        if self._hover_person is None or not (0 <= self._hover_person < len(self._overlay_people)):
            return ""
        person = self._overlay_people[self._hover_person]
        name = person.label or f"Person {self._hover_person + 1}"
        if self._hover_joint is not None:
            return f"{name}: {self._hover_joint}"
        if self._hover_bone is not None:
            return f"{name}: {self._hover_bone}"
        return name

    def _clear_hover(self) -> None:
        self._hover_person = self._hover_bone = self._hover_joint = None
        self.setToolTip("")

    def _apply_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _displayed_pixmap_geometry(self) -> tuple[QPixmap, float, float] | None:
        """Returns (pixmap, offset_x, offset_y) for the currently displayed
        (possibly letterboxed) pixmap, or None if there's nothing to show --
        shared by click hit-testing and overlay drawing so they can't drift
        out of sync with each other.
        """
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        offset_x = (self.width() - pixmap.width()) / 2.0
        offset_y = (self.height() - pixmap.height()) / 2.0
        return pixmap, offset_x, offset_y

    def normalized_point(self, pos) -> tuple[float, float] | None:
        """A widget position as a normalized image point, or None outside the
        displayed photo (the letterbox margin, or no photo)."""
        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return None
        pixmap, offset_x, offset_y = geometry
        x_in_pixmap = pos.x() - offset_x
        y_in_pixmap = pos.y() - offset_y
        if not (0 <= x_in_pixmap <= pixmap.width() and 0 <= y_in_pixmap <= pixmap.height()):
            return None
        return x_in_pixmap / pixmap.width(), y_in_pixmap / pixmap.height()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        pos = event.position() if hasattr(event, "position") else event.pos()
        if event.button() == Qt.LeftButton and self.begin_drag(pos):
            return  # a click on a joint handle starts a drag (or is a click on release)
        point = self.normalized_point(pos)
        if point is None:
            return  # click landed in the label's letterbox margin, not on the photo
        if event.button() == Qt.LeftButton:
            self.clicked_normalized.emit(*point)
        elif event.button() == Qt.RightButton:
            global_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.context_requested.emit(point[0], point[1], global_pos)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        pos = event.position() if hasattr(event, "position") else event.pos()
        if self._drag is not None:
            self.drag_to(pos)
            return
        self._set_hover(pos)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._drag is None or event.button() != Qt.LeftButton:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        start = self._drag["start"]
        if not self.end_drag(pos):
            point = self.normalized_point(start)
            if point is not None:
                self.clicked_normalized.emit(*point)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hover_pos = None
        if self._hover_person is not None or self._hover_bone is not None or self._hover_joint is not None:
            self._clear_hover()
            self.update()

    def _set_hover(self, pos) -> None:
        self._hover_pos = QPointF(pos)
        person, bone, joint = self._hover_hit(pos)
        if (person, bone, joint) != (self._hover_person, self._hover_bone, self._hover_joint):
            self._hover_person, self._hover_bone, self._hover_joint = person, bone, joint
            self.setToolTip(self.hover_text())
            self.update()

    def _hover_hit(self, pos) -> tuple[int | None, str | None, str | None]:
        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return None, None, None
        pixmap, offset_x, offset_y = geometry
        width, height = pixmap.width(), pixmap.height()

        def to_widget_point(normalized_xy: np.ndarray) -> QPointF:
            return QPointF(offset_x + float(normalized_xy[0]) * width, offset_y + float(normalized_xy[1]) * height)

        best_bone_person, best_bone, best_bone_distance = None, None, 12.0
        best_joint_person, best_joint, best_joint_distance = None, None, 9.0
        for index, person in enumerate(self._overlay_people):
            for key, xy in person.joints_2d.items():
                point = to_widget_point(xy)
                distance = float(np.hypot(point.x() - pos.x(), point.y() - pos.y()))
                if distance <= best_joint_distance:
                    best_joint_person, best_joint, best_joint_distance = index, key, distance
            for name in ALL_BONE_NAMES:
                head_key, tail_key = f"{name}.head", f"{name}.tail"
                if head_key not in person.joints_2d or tail_key not in person.joints_2d:
                    continue
                head = to_widget_point(person.joints_2d[head_key])
                tail = to_widget_point(person.joints_2d[tail_key])
                segment = np.array([tail.x() - head.x(), tail.y() - head.y()], dtype=float)
                length_sq = float(np.dot(segment, segment))
                if length_sq <= 1e-12:
                    continue
                rel = np.array([pos.x() - head.x(), pos.y() - head.y()], dtype=float)
                t = float(np.clip(np.dot(rel, segment) / length_sq, 0.0, 1.0))
                nearest = np.array([head.x(), head.y()], dtype=float) + t * segment
                distance = float(np.hypot(*(nearest - np.array([pos.x(), pos.y()], dtype=float))))
                if distance <= best_bone_distance:
                    best_bone_person, best_bone, best_bone_distance = index, name, distance
        if best_joint is not None:
            return best_joint_person, best_bone if best_bone_person == best_joint_person else None, best_joint
        if best_bone is not None:
            return best_bone_person, best_bone, None
        return None, None, None

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._overlay_people:
            return

        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return
        pixmap, offset_x, offset_y = geometry

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            # Unselected people first, so the selected one's lines/joints always
            # render on top even where two people's overlays overlap.
            for index, person in sorted(enumerate(self._overlay_people), key=lambda item: item[1].selected):
                self._draw_person(painter, person, index, offset_x, offset_y, pixmap.width(), pixmap.height())
            self._draw_handles(painter)
        finally:
            painter.end()

    def _draw_handles(self, painter: QPainter) -> None:
        if not self._handles:
            return
        pen = QPen(DRAG_HANDLE_COLOR)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for joint, xy in self._handles.items():
            if self._drag is not None and self._drag["joint"] == joint:
                xy = self._drag["point"]
            point = self._widget_point(xy)
            if point is not None:
                painter.drawEllipse(QRectF(point.x() - 6, point.y() - 6, 12, 12))

    def _draw_person(
        self, painter: QPainter, person: PhotoOverlayPerson, person_index: int, offset_x: float, offset_y: float, width: int, height: int
    ) -> None:
        def to_widget_point(normalized_xy: np.ndarray) -> QPointF:
            return QPointF(offset_x + float(normalized_xy[0]) * width, offset_y + float(normalized_xy[1]) * height)

        for name in ALL_BONE_NAMES:
            if f"{name}.head" not in person.joints_2d:
                continue
            head = to_widget_point(person.joints_2d[f"{name}.head"])
            tail = to_widget_point(person.joints_2d[f"{name}.tail"])
            color, pen_width, style = bone_pen_style(person.selected, person.confidence.get(name, 1.0), name in person.out_of_frame)
            if person_index == self._hover_person and not person.selected:
                color, pen_width = HOVER_PERSON_COLOR, max(2.5, pen_width)
            if person_index == self._hover_person and name == self._hover_bone:
                color, pen_width = HOVER_COLOR, max(4.0, pen_width + 1.0)
            if name in _FINGER_BONES:
                pen_width = max(1.0, pen_width / 2.0)
            pen = QPen(color)
            pen.setWidthF(pen_width)
            pen.setStyle(style)
            painter.setPen(pen)
            painter.drawLine(head, tail)

        painter.setPen(Qt.NoPen)
        joint_brush = SELECTED_JOINT_COLOR if person.selected else UNSELECTED_COLOR
        if person_index == self._hover_person and not person.selected:
            joint_brush = HOVER_PERSON_COLOR
        painter.setBrush(joint_brush)
        radius = 3.5 if person.selected else 2.5
        for key, position in person.joints_2d.items():
            point = to_widget_point(position)
            r = radius / 2.0 if key.rsplit(".", 1)[0] in _FINGER_BONES else radius
            if person_index == self._hover_person and key == self._hover_joint:
                painter.setBrush(HOVER_COLOR)
                r = max(r, 5.0)
            else:
                painter.setBrush(joint_brush)
            painter.drawEllipse(point, r, r)

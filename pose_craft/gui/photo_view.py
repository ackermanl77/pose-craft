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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD
from pose_craft.retarget.skeleton import BONES

SELECTED_BONE_COLOR = QColor(77, 204, 255)  # matches viewport_3d.BONE_COLOR
SELECTED_LOW_CONFIDENCE_COLOR = QColor(255, 100, 60)  # matches viewport_3d.LOW_CONFIDENCE_BONE_COLOR
SELECTED_JOINT_COLOR = QColor(255, 255, 255)  # matches viewport_3d.JOINT_COLOR
UNSELECTED_COLOR = QColor(190, 190, 190, 150)  # dim/translucent so the selected person pops


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


def bone_pen_style(selected: bool, confidence: float) -> tuple[QColor, float, Qt.PenStyle]:
    """(colour, width, line style) for one bone of the overlay: the selected
    person's bones in the bone colour, solid, or -- below
    LOW_CONFIDENCE_THRESHOLD -- in the warning colour and dashed; everyone
    else thin and dim."""
    if not selected:
        return UNSELECTED_COLOR, 1.5, Qt.SolidLine
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return SELECTED_LOW_CONFIDENCE_COLOR, 3.0, Qt.DashLine
    return SELECTED_BONE_COLOR, 3.0, Qt.SolidLine


def legend_entries() -> list[tuple[QColor, Qt.PenStyle, str]]:
    """The legend under the photo, drawn from the same constants and styles the
    overlay uses (it used to be emoji whose colours matched nothing drawn)."""
    return [
        (SELECTED_BONE_COLOR, Qt.SolidLine, "Bone"),
        (SELECTED_LOW_CONFIDENCE_COLOR, Qt.DashLine, "Low confidence"),
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setStyleSheet("background:#222; color:#aaa; border:1px solid #444;")
        self.setText("No photo loaded — open one, or drop an image file here")
        self.setWordWrap(True)
        self._source_pixmap: QPixmap | None = None
        self._overlay_people: list[PhotoOverlayPerson] = []

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap()

    def clear_photo(self) -> None:
        self._source_pixmap = None
        self._overlay_people = []
        self.clear()
        self.setText("No photo loaded")

    def set_skeleton_overlay(self, people: list[PhotoOverlayPerson]) -> None:
        """people: every detected person to draw, in whatever order -- pass []
        to clear. Triggers a repaint; the actual drawing happens in paintEvent.
        """
        self._overlay_people = people
        self.update()

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
        point = self.normalized_point(pos)
        if point is None:
            return  # click landed in the label's letterbox margin, not on the photo
        if event.button() == Qt.LeftButton:
            self.clicked_normalized.emit(*point)
        elif event.button() == Qt.RightButton:
            global_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self.context_requested.emit(point[0], point[1], global_pos)

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
            for person in sorted(self._overlay_people, key=lambda p: p.selected):
                self._draw_person(painter, person, offset_x, offset_y, pixmap.width(), pixmap.height())
        finally:
            painter.end()

    def _draw_person(
        self, painter: QPainter, person: PhotoOverlayPerson, offset_x: float, offset_y: float, width: int, height: int
    ) -> None:
        def to_widget_point(normalized_xy: np.ndarray) -> QPointF:
            return QPointF(offset_x + float(normalized_xy[0]) * width, offset_y + float(normalized_xy[1]) * height)

        for bone in BONES:
            head = to_widget_point(person.joints_2d[f"{bone.name}.head"])
            tail = to_widget_point(person.joints_2d[f"{bone.name}.tail"])
            color, pen_width, style = bone_pen_style(person.selected, person.confidence.get(bone.name, 1.0))
            pen = QPen(color)
            pen.setWidthF(pen_width)
            pen.setStyle(style)
            painter.setPen(pen)
            painter.drawLine(head, tail)

        painter.setPen(Qt.NoPen)
        painter.setBrush(SELECTED_JOINT_COLOR if person.selected else UNSELECTED_COLOR)
        radius = 3.5 if person.selected else 2.5
        for position in person.joints_2d.values():
            point = to_widget_point(position)
            painter.drawEllipse(point, radius, radius)

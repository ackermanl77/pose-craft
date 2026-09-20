"""The photo display widget: keeps the full-resolution QPixmap around so it can
rescale on window resize (rather than blurring a pixmap that was already
downscaled once), turns a click into a normalized (0..1, 0..1) image-space
point so the main window can pick whichever detected person is nearest --
PoseForge lets you click the person directly in the photo, this is the same
idea -- and draws the solved 2D skeleton for every detected person directly
on top of the photo, so detection quality (and which person is selected) is
visible in context instead of only in the separate 3D viewport.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel

from pose_craft.export.blender_pose import LOW_CONFIDENCE_THRESHOLD
from pose_craft.retarget.skeleton import BONES

_SELECTED_BONE_COLOR = QColor(77, 204, 255)  # matches viewport_3d.BONE_COLOR
_SELECTED_LOW_CONFIDENCE_COLOR = QColor(255, 100, 60)
_SELECTED_JOINT_COLOR = QColor(255, 153, 26)  # matches viewport_3d.JOINT_COLOR
_UNSELECTED_COLOR = QColor(190, 190, 190, 150)  # dim/translucent so the selected person pops


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


class PhotoView(QLabel):
    clicked_normalized = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 480)
        self.setStyleSheet("background:#222; color:#aaa; border:1px solid #444;")
        self.setText("No photo loaded")
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

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        geometry = self._displayed_pixmap_geometry()
        if geometry is None:
            return
        pixmap, offset_x, offset_y = geometry

        pos = event.position() if hasattr(event, "position") else event.pos()
        x_in_pixmap = pos.x() - offset_x
        y_in_pixmap = pos.y() - offset_y

        if not (0 <= x_in_pixmap <= pixmap.width() and 0 <= y_in_pixmap <= pixmap.height()):
            return  # click landed in the label's letterbox margin, not on the photo

        self.clicked_normalized.emit(x_in_pixmap / pixmap.width(), y_in_pixmap / pixmap.height())

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

            if not person.selected:
                color = _UNSELECTED_COLOR
            elif person.confidence.get(bone.name, 1.0) < LOW_CONFIDENCE_THRESHOLD:
                color = _SELECTED_LOW_CONFIDENCE_COLOR
            else:
                color = _SELECTED_BONE_COLOR

            pen = QPen(color)
            pen.setWidthF(3.0 if person.selected else 1.5)
            painter.setPen(pen)
            painter.drawLine(head, tail)

        painter.setPen(Qt.NoPen)
        painter.setBrush(_SELECTED_JOINT_COLOR if person.selected else _UNSELECTED_COLOR)
        radius = 4.0 if person.selected else 2.5
        for position in person.joints_2d.values():
            point = to_widget_point(position)
            painter.drawEllipse(point, radius, radius)

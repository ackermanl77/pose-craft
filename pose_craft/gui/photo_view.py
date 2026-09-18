"""The photo display widget: keeps the full-resolution QPixmap around so it can
rescale on window resize (rather than blurring a pixmap that was already
downscaled once), and turns a click into a normalized (0..1, 0..1) image-space
point so the main window can pick whichever detected person is nearest --
PoseForge lets you click the person directly in the photo, this is the same idea.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel


class PhotoView(QLabel):
    clicked_normalized = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 480)
        self.setStyleSheet("background:#222; color:#aaa; border:1px solid #444;")
        self.setText("No photo loaded")
        self._source_pixmap: QPixmap | None = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap()

    def clear_photo(self) -> None:
        self._source_pixmap = None
        self.clear()
        self.setText("No photo loaded")

    def _apply_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        scaled = self._source_pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        offset_x = (self.width() - pixmap.width()) / 2.0
        offset_y = (self.height() - pixmap.height()) / 2.0
        pos = event.position() if hasattr(event, "position") else event.pos()
        x_in_pixmap = pos.x() - offset_x
        y_in_pixmap = pos.y() - offset_y

        if not (0 <= x_in_pixmap <= pixmap.width() and 0 <= y_in_pixmap <= pixmap.height()):
            return  # click landed in the label's letterbox margin, not on the photo

        self.clicked_normalized.emit(x_in_pixmap / pixmap.width(), y_in_pixmap / pixmap.height())

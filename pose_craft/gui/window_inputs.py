"""Opening photos and videos: the file dialogs, drag and drop, the command line and the
recent-files menu (C4: split out of main_window.py). A mixin of MainWindow; load_video() and the
solve lifecycle stay in main_window.py with the worker threads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from pose_craft.imaging import assess_photo_quality, imread_unicode

PHOTO_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

_PHOTO_FILTER = "Images (" + " ".join(f"*{s}" for s in PHOTO_SUFFIXES) + ")"

_VIDEO_FILTER = "Videos (" + " ".join(f"*{s}" for s in VIDEO_SUFFIXES) + ")"

MAX_RECENT_FILES = 8

def _update_recent_files(recent: list[str], path: str, limit: int = MAX_RECENT_FILES) -> list[str]:
    """`path` moved to the front of the recent-files list (case-insensitively
    unique, as Windows paths are), trimmed to `limit`."""
    key = str(Path(path)).lower()
    kept = [p for p in recent if str(Path(p)).lower() != key]
    return [str(Path(path))] + kept[: limit - 1]


class InputsMixin:
    def _recent_files(self) -> list[str]:
        value = self._settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in (value or [])]

    def _remember_file(self, path: str) -> None:
        self._settings.setValue("recent_files", _update_recent_files(self._recent_files(), path))
        self._settings.setValue("last_dir", str(Path(path).parent))
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent_files()
        for path in recent:
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.open_path(p))
        self.recent_menu.setEnabled(bool(recent))

    def _last_dir(self) -> str:
        return str(self._settings.value("last_dir", ""))

    def dragEnterEvent(self, event) -> None:
        if self._dropped_path(event.mimeData()) is not None and not self._worker_running():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = self._dropped_path(event.mimeData())
        if path is not None:
            event.acceptProposedAction()
            self.open_path(path)

    @staticmethod
    def _dropped_path(mime) -> str | None:
        """The first dropped local file that's a photo or a video, if any."""
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in PHOTO_SUFFIXES + VIDEO_SUFFIXES:
                return url.toLocalFile()
        return None

    def open_path(self, path: str) -> None:
        """Opens a photo or a video by extension (drag-and-drop, recent files)."""
        if Path(path).suffix.lower() in VIDEO_SUFFIXES:
            self.load_video(path)
        else:
            self.load_photo(path)

    def _on_load_photo(self) -> None:
        # Guards against the same race closeEvent guards against: loading a new
        # photo while a solve for the OLD photo is still running would let that
        # solve's results land on the new photo once it finishes, and risks two
        # threads calling the shared PoseEstimator concurrently. load_btn is
        # disabled for the duration of a solve (see _set_busy) so this
        # shouldn't be reachable from the UI, but checking here too costs nothing.
        if self._worker_running():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Photo", self._last_dir(), _PHOTO_FILTER)
        if path:
            self.load_photo(path)

    def load_photo(self, path: str) -> bool:
        if self._worker_running():
            return False
        image_bgr = imread_unicode(path)
        if image_bgr is None:
            QMessageBox.warning(self, "Load Photo", f"Could not read image:\n{path}")
            return False

        self._clear_video()
        self._image_bgr = image_bgr
        self._source_path = path
        self._set_photo_preview(image_bgr)
        self._clear_results()
        self.solve_btn.setEnabled(True)
        self.solve_action.setEnabled(True)
        self._remember_file(path)
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

        quality_warnings = assess_photo_quality(image_bgr)
        self.banner_label.setText("\n".join(quality_warnings))
        self.banner.setVisible(bool(quality_warnings))
        return True

    def _set_photo_preview(self, image_bgr: np.ndarray) -> None:
        self._preview_bgr = image_bgr
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.photo_view.set_source_pixmap(QPixmap.fromImage(qimage))

    def _on_load_video(self) -> None:
        if self._worker_running():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Video", self._last_dir(), _VIDEO_FILTER)
        if path:
            self.load_video(path)

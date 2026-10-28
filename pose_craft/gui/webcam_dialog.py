"""File > Record from Webcam... (F14): pick a camera and a length, count down, record with a live
preview, and hand the clip back to the main window, which tracks it like any opened video."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from pose_craft.estimation import webcam

PREVIEW_SIZE = (480, 270)


class RecordWorker(QThread):
    """Runs webcam.record_clip off the UI thread, sending scaled preview frames."""

    frame = Signal(QImage)
    finished_ok = Signal(object)  # webcam.RecordedClip
    failed = Signal(str)

    def __init__(self, source, path: Path, seconds: float, parent=None):
        super().__init__(parent)
        self._source = source
        self._path = path
        self._seconds = seconds
        self._stop = threading.Event()

    def stop(self) -> None:
        """Ends the recording early, keeping what was recorded."""
        self._stop.set()

    def _preview(self, bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888)
        self.frame.emit(image.scaled(*PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run(self) -> None:
        try:
            clip = webcam.record_clip(self._source, self._path, self._seconds, on_frame=self._preview,
                                      should_cancel=self._stop.is_set)
        except Exception as exc:  # noqa: BLE001 - shown in the dialog
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(clip)


class WebcamDialog(QDialog):
    """`sources` are the camera indices to offer (probed when None); a test passes a video file."""

    def __init__(self, parent=None, settings=None, sources=None, out_dir: Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Record from Webcam")
        self._settings = settings
        self._out_dir = Path(out_dir) if out_dir is not None else webcam.recordings_dir()
        self._worker: RecordWorker | None = None
        self._countdown_left = 0
        self._recording_path: Path | None = None
        self.clip: webcam.RecordedClip | None = None

        sources = webcam.list_cameras() if sources is None else list(sources)
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Camera")
        for source in sources:
            self.camera_combo.addItem(f"Camera {source}" if isinstance(source, int) else Path(str(source)).name, source)
        saved = self._setting("webcam/camera", None)
        if saved is not None:
            index = self.camera_combo.findText(f"Camera {saved}")
            if index >= 0:
                self.camera_combo.setCurrentIndex(index)

        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(1, 300)
        self.seconds_spin.setSuffix(" s")
        self.seconds_spin.setValue(int(self._setting("webcam/seconds", 10)))
        self.seconds_spin.setToolTip("The longest the recording runs; Stop ends it sooner.")
        self.countdown_spin = QSpinBox()
        self.countdown_spin.setRange(0, 10)
        self.countdown_spin.setSuffix(" s")
        self.countdown_spin.setValue(int(self._setting("webcam/countdown", 3)))
        self.countdown_spin.setToolTip("Time to step back into view before the recording starts.")

        self.preview = QLabel("The camera's picture appears here while recording.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(*PREVIEW_SIZE)
        self.status = QLabel()
        self.status.setWordWrap(True)

        self.record_btn = QPushButton("&Record")
        self.record_btn.clicked.connect(self._on_record)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton(self.record_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("&Camera:", self.camera_combo)
        form.addRow("&Length:", self.seconds_spin)
        form.addRow("Count&down:", self.countdown_spin)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.preview)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        if not sources:
            self.record_btn.setEnabled(False)
            self.status.setText("No camera found. Plug one in and open this again.")

    def _setting(self, key, default):
        if self._settings is None:
            return default
        value = self._settings.value(key, default)
        return default if value is None else value

    def _on_record(self) -> None:
        if self._worker is not None:  # the button reads Stop while recording
            self._worker.stop()
            return
        if self._timer.isActive():
            return
        if self._settings is not None:
            source = self.camera_combo.currentData()
            if isinstance(source, int):
                self._settings.setValue("webcam/camera", source)
            self._settings.setValue("webcam/seconds", self.seconds_spin.value())
            self._settings.setValue("webcam/countdown", self.countdown_spin.value())
        for widget in (self.camera_combo, self.seconds_spin, self.countdown_spin):
            widget.setEnabled(False)
        self._countdown_left = self.countdown_spin.value()
        if self._countdown_left:
            self.record_btn.setEnabled(False)
            self.status.setText(f"Recording in {self._countdown_left}...")
            self._timer.start()
        else:
            self._start()

    def _tick(self) -> None:
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.status.setText(f"Recording in {self._countdown_left}...")
            return
        self._timer.stop()
        self._start()

    def _start(self) -> None:
        self._recording_path = self._out_dir / time.strftime("webcam_%Y%m%d_%H%M%S.mp4")
        self._worker = RecordWorker(self.camera_combo.currentData(), self._recording_path, float(self.seconds_spin.value()), self)
        self._worker.frame.connect(lambda image: self.preview.setPixmap(QPixmap.fromImage(image)))
        self._worker.finished_ok.connect(self._on_recorded)
        self._worker.failed.connect(self._on_failed)
        self.record_btn.setText("&Stop")
        self.record_btn.setEnabled(True)
        self.status.setText("Recording... press Stop to end it sooner.")
        self._worker.start()

    def _on_recorded(self, clip) -> None:
        self._worker.wait()
        self._worker = None
        self.clip = clip
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._worker.wait()
        self._worker = None
        self.record_btn.setText("&Record")
        for widget in (self.camera_combo, self.seconds_spin, self.countdown_spin):
            widget.setEnabled(True)
        self.status.setText(f"Recording failed: {message}")

    def reject(self) -> None:
        self._timer.stop()
        if self._worker is not None:  # Cancel mid-recording: stop and throw the clip away
            self._worker.finished_ok.disconnect()
            self._worker.failed.disconnect()
            self._worker.stop()
            self._worker.wait()
            self._worker = None
            self._recording_path.unlink(missing_ok=True)
        super().reject()

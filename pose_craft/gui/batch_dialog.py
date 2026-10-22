"""File > Batch Process Folder (B13): solve every photo in a folder with the
current detection settings and write one pose file per person, on a worker
thread with a progress bar and Cancel. The work itself is pose_craft.batch,
shared with `python -m pose_craft.cli`."""
from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from pose_craft.batch import FORMATS, BatchResult, solve_folder
from pose_craft.estimation.landmarker import PoseEstimator

log = logging.getLogger(__name__)


@dataclasses.dataclass
class BatchSettings:
    """The main window's detection settings a batch runs with."""

    variant: str = "full"
    use_gpu: bool = True
    max_people: int = 4
    group_photo: bool = False
    hands: bool = False
    detection_threshold: float = 0.5
    skeleton: str = "posecraft"  # B22: the Export group's skeleton, for BVH/GLB


class BatchWorker(QThread):
    progress = Signal(int, int, str)  # photo number, photo count, photo name
    finished_ok = Signal(object)  # BatchResult
    failed = Signal(str)

    def __init__(self, input_dir: Path, output_dir: Path, fmt: str, mirror: bool, settings: BatchSettings, parent=None):
        super().__init__(parent)
        self._input_dir, self._output_dir, self._fmt, self._mirror = input_dir, output_dir, fmt, mirror
        self._settings = settings
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        settings = self._settings
        try:
            # Its own estimator: the detector session is shared (P5), and a
            # batch never races the window's own solves for one session.
            with PoseEstimator.for_variant(settings.variant, max_people=settings.max_people, use_gpu=settings.use_gpu,
                                           should_cancel=self._cancel.is_set) as estimator:
                estimator.group_photo = settings.group_photo
                estimator.min_detection_confidence = settings.detection_threshold
                if settings.hands:
                    estimator.enable_hands(should_cancel=self._cancel.is_set)
                    estimator.hands = True
                result = solve_folder(
                    self._input_dir, self._output_dir, estimator, fmt=self._fmt, mirror=self._mirror,
                    on_progress=lambda number, total, photo: self.progress.emit(number, total, photo.name),
                    should_cancel=self._cancel.is_set, skeleton=settings.skeleton,
                )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the dialog
            log.exception("batch failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BatchDialog(QDialog):
    def __init__(self, settings: BatchSettings, start_dir: str = "", mirror: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Process Folder")
        self._settings = settings
        self._worker: BatchWorker | None = None
        self.result: BatchResult | None = None
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.input_edit = QLineEdit(start_dir)
        self.input_edit.setAccessibleName("Photo folder")
        self.output_edit = QLineEdit(str(Path(start_dir) / "poses") if start_dir else "")
        self.output_edit.setAccessibleName("Output folder")
        for row, (label, edit) in enumerate((("Photos in", self.input_edit), ("Poses to", self.output_edit))):
            browse = QPushButton("Browse...")
            browse.clicked.connect(lambda _checked=False, e=edit, title=label: self._browse(e, title))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(browse, row, 2)
        self.format_combo = QComboBox()
        self.format_combo.addItems([fmt.upper() for fmt in FORMATS])
        self.format_combo.setAccessibleName("Format")
        grid.addWidget(QLabel("Format"), 2, 0)
        grid.addWidget(self.format_combo, 2, 1)
        self.mirror_checkbox = QCheckBox("Mirror left/right")
        self.mirror_checkbox.setChecked(mirror)
        grid.addWidget(self.mirror_checkbox, 3, 1)
        layout.addLayout(grid)
        layout.addWidget(QLabel(
            f"Uses the current detection settings: {settings.variant} model, {'GPU if available' if settings.use_gpu else 'CPU'}, "
            f"up to {settings.max_people} people{', group photo' if settings.group_photo else ''}"
            f"{', hands' if settings.hands else ''}, detection threshold {settings.detection_threshold:.2f}. "
            "Writes <photo>_person<N> files, replacing existing ones."
        ))
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("batchLog")
        self.log_view.setMinimumSize(520, 140)
        layout.addWidget(self.log_view)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start_btn = QPushButton("&Start")
        self.start_btn.clicked.connect(self.start)
        self.close_btn = QPushButton("&Close")
        self.close_btn.clicked.connect(self._on_close_clicked)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def _browse(self, edit: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, title, edit.text())
        if folder:
            edit.setText(folder)

    def start(self) -> None:
        input_dir, output_dir = Path(self.input_edit.text().strip()), Path(self.output_edit.text().strip())
        if not self.input_edit.text().strip() or not input_dir.is_dir():
            self.log_view.setPlainText("Choose a folder of photos first.")
            return
        if not self.output_edit.text().strip():
            self.log_view.setPlainText("Choose where the poses go.")
            return
        self.log_view.clear()
        self.start_btn.setEnabled(False)
        self.close_btn.setText("&Cancel")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self._worker = BatchWorker(input_dir, output_dir, FORMATS[self.format_combo.currentIndex()],
                                   self.mirror_checkbox.isChecked(), self._settings, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, number: int, total: int, name: str) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(number - 1)
        self.log_view.appendPlainText(f"Solving {name} ({number}/{total})...")

    def _on_finished(self, result: BatchResult) -> None:
        self.result = result
        for outcome in result.outcomes:
            detail = outcome.error or ", ".join(path.name for path in outcome.written)
            self.log_view.appendPlainText(f"{outcome.photo.name}: {detail}")
        self.log_view.appendPlainText(result.summary())
        self._done()

    def _on_failed(self, message: str) -> None:
        self.log_view.appendPlainText(f"Batch failed: {message}")
        self._done()

    def _done(self) -> None:
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.close_btn.setText("&Close")

    def running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _on_close_clicked(self) -> None:
        if self.running():
            self._worker.cancel()
            self.log_view.appendPlainText("Cancelling after the current photo...")
            return
        self.accept()

    def done(self, code: int) -> None:  # noqa: D401 -- QDialog.done override
        """Closing waits for a running batch to stop (it checks between photos)."""
        if self.running():
            self._worker.cancel()
            self._worker.wait()
        super().done(code)

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pose_craft.estimation.landmarker import DetectedPerson, detect_poses, ensure_model_downloaded
from pose_craft.export.blender_pose import save_pose_file
from pose_craft.gui.viewport_3d import SkeletonViewport
from pose_craft.retarget.skeleton import joint_positions, mirror_landmarks, solve_bone_rotations


class SolveWorker(QThread):
    """Runs model download + pose detection off the UI thread -- the first run in
    particular can take a while while the landmarker model downloads.
    """

    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self._image_bgr = image_bgr

    def run(self) -> None:
        try:
            model_path = ensure_model_downloaded("full")
            people = detect_poses(self._image_bgr, model_path)
            self.finished_ok.emit(people)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PoseCraft - Photo to Pose")
        self.resize(1200, 720)

        self._image_bgr: np.ndarray | None = None
        self._people: list[DetectedPerson] = []
        self._worker: SolveWorker | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        root.addLayout(left, stretch=1)

        self.photo_label = QLabel("No photo loaded")
        self.photo_label.setAlignment(Qt.AlignCenter)
        self.photo_label.setMinimumSize(360, 480)
        self.photo_label.setStyleSheet("background:#222; color:#aaa; border:1px solid #444;")
        left.addWidget(self.photo_label)

        load_btn = QPushButton("Load Photo...")
        load_btn.clicked.connect(self._on_load_photo)
        left.addWidget(load_btn)

        self.solve_btn = QPushButton("Solve Pose")
        self.solve_btn.clicked.connect(self._on_solve_pose)
        self.solve_btn.setEnabled(False)
        left.addWidget(self.solve_btn)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self.person_combo = QComboBox()
        self.person_combo.currentIndexChanged.connect(self._on_person_changed)
        person_row.addWidget(self.person_combo)
        left.addLayout(person_row)

        self.mirror_checkbox = QCheckBox("Mirror left/right (body orientation fix)")
        self.mirror_checkbox.stateChanged.connect(self._on_person_changed)
        left.addWidget(self.mirror_checkbox)

        self.export_btn = QPushButton("Export Pose JSON...")
        self.export_btn.clicked.connect(self._on_export_pose)
        self.export_btn.setEnabled(False)
        left.addWidget(self.export_btn)

        left.addStretch(1)

        right = QVBoxLayout()
        root.addLayout(right, stretch=2)
        right.addWidget(QLabel("3D preview"))
        self.viewport = SkeletonViewport()
        right.addWidget(self.viewport, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _on_load_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            QMessageBox.warning(self, "Load Photo", f"Could not read image:\n{path}")
            return

        self._image_bgr = image_bgr
        self._set_photo_preview(image_bgr)
        self.solve_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.person_combo.clear()
        self.viewport.clear_skeleton()
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

    def _set_photo_preview(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.photo_label.width(), self.photo_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.photo_label.setPixmap(pixmap)

    def _on_solve_pose(self) -> None:
        if self._image_bgr is None:
            return

        self.solve_btn.setEnabled(False)
        self.statusBar().showMessage("Solving pose (downloading model on first run may take a moment)...")

        self._worker = SolveWorker(self._image_bgr)
        self._worker.finished_ok.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.start()

    def _on_solve_finished(self, people: list[DetectedPerson]) -> None:
        self.solve_btn.setEnabled(True)
        self._people = people

        if not people:
            self.statusBar().showMessage("No person detected in this photo")
            QMessageBox.information(self, "Solve Pose", "No person was detected in this photo.")
            return

        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        for i in range(len(people)):
            self.person_combo.addItem(f"Person {i + 1}")
        self.person_combo.blockSignals(False)
        self.person_combo.setCurrentIndex(0)

        self.statusBar().showMessage(f"Solved {len(people)} pose(s)")
        self._update_preview_for_selected_person()

    def _on_solve_failed(self, message: str) -> None:
        self.solve_btn.setEnabled(True)
        self.statusBar().showMessage("Solve failed")
        QMessageBox.critical(self, "Solve Pose", f"Pose solving failed:\n{message}")

    def _on_person_changed(self) -> None:
        self._update_preview_for_selected_person()

    def _update_preview_for_selected_person(self) -> None:
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            self.export_btn.setEnabled(False)
            return

        landmarks = self._people[index].landmarks
        if self.mirror_checkbox.isChecked():
            landmarks = mirror_landmarks(landmarks)

        self.viewport.show_skeleton(joint_positions(landmarks))
        self.export_btn.setEnabled(True)

    def _on_export_pose(self) -> None:
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Pose JSON", "pose.json", "Pose JSON (*.json)")
        if not path:
            return

        landmarks = self._people[index].landmarks
        if self.mirror_checkbox.isChecked():
            landmarks = mirror_landmarks(landmarks)

        rotations = solve_bone_rotations(landmarks)
        save_pose_file(rotations, path, source_note="Exported from PoseCraft")
        self.statusBar().showMessage(f"Exported pose to {path}")

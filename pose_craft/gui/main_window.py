from __future__ import annotations

from pathlib import Path

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

from pose_craft.estimation.landmarker import DetectedPerson, PoseEstimator, ensure_model_downloaded
from pose_craft.export.blender_pose import build_pose_document, save_pose_file
from pose_craft.export.live_link import send_pose_document
from pose_craft.gui.photo_view import PhotoView
from pose_craft.gui.viewport_3d import SkeletonViewport
from pose_craft.imaging import imread_unicode
from pose_craft.retarget.skeleton import bone_confidence, joint_positions, mirror_landmarks, solve_bone_rotations


class SolveWorker(QThread):
    """Runs model download + pose detection off the UI thread -- the first run in
    particular can take a while while the landmarker model downloads. Reuses an
    already-created PoseEstimator when the caller has one (see MainWindow._estimator)
    so the model only has to load once per app session, not once per click.
    """

    finished_ok = Signal(list, object)  # list[DetectedPerson], PoseEstimator
    failed = Signal(str)

    def __init__(self, image_bgr: np.ndarray, estimator: PoseEstimator | None, parent=None):
        super().__init__(parent)
        self._image_bgr = image_bgr
        self._estimator = estimator

    def run(self) -> None:
        try:
            estimator = self._estimator
            if estimator is None:
                model_path = ensure_model_downloaded("full")
                estimator = PoseEstimator(model_path)
            people = estimator.detect(self._image_bgr)
            self.finished_ok.emit(people, estimator)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.failed.emit(str(exc))


def _nearest_person_index(people: list[DetectedPerson], x: float, y: float) -> int:
    distances = [((p.centroid_xy()[0] - x) ** 2 + (p.centroid_xy()[1] - y) ** 2) for p in people]
    return int(np.argmin(distances))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PoseCraft - Photo to Pose")
        self.resize(1200, 720)

        self._image_bgr: np.ndarray | None = None
        self._people: list[DetectedPerson] = []
        self._worker: SolveWorker | None = None
        self._estimator: PoseEstimator | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        root.addLayout(left, stretch=1)

        self.photo_view = PhotoView()
        self.photo_view.clicked_normalized.connect(self._on_photo_clicked)
        left.addWidget(self.photo_view)

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
        left.addWidget(QLabel("(or click the person directly in the photo)"))

        self.mirror_checkbox = QCheckBox("Mirror left/right (body orientation fix)")
        self.mirror_checkbox.stateChanged.connect(self._on_person_changed)
        left.addWidget(self.mirror_checkbox)

        self.export_btn = QPushButton("Export Pose JSON...")
        self.export_btn.clicked.connect(self._on_export_pose)
        self.export_btn.setEnabled(False)
        left.addWidget(self.export_btn)

        self.export_all_btn = QPushButton("Export All People...")
        self.export_all_btn.clicked.connect(self._on_export_all)
        self.export_all_btn.setEnabled(False)
        left.addWidget(self.export_all_btn)

        self.send_to_blender_btn = QPushButton("Send to Blender (Live Link)")
        self.send_to_blender_btn.clicked.connect(self._on_send_to_blender)
        self.send_to_blender_btn.setEnabled(False)
        left.addWidget(self.send_to_blender_btn)

        left.addStretch(1)

        right = QVBoxLayout()
        root.addLayout(right, stretch=2)
        right.addWidget(QLabel("3D preview"))
        self.viewport = SkeletonViewport()
        right.addWidget(self.viewport, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def closeEvent(self, event) -> None:
        if self._estimator is not None:
            self._estimator.close()
        super().closeEvent(event)

    def _on_load_photo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Photo", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return

        image_bgr = imread_unicode(path)
        if image_bgr is None:
            QMessageBox.warning(self, "Load Photo", f"Could not read image:\n{path}")
            return

        self._image_bgr = image_bgr
        self._set_photo_preview(image_bgr)
        self.solve_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)
        self.send_to_blender_btn.setEnabled(False)
        self.person_combo.clear()
        self._people = []
        self.viewport.clear_skeleton()
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

    def _set_photo_preview(self, image_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.photo_view.set_source_pixmap(QPixmap.fromImage(qimage))

    def _on_solve_pose(self) -> None:
        if self._image_bgr is None:
            return

        self.solve_btn.setEnabled(False)
        first_run_note = "" if self._estimator is not None else " (downloading model on first run may take a moment)"
        self.statusBar().showMessage(f"Solving pose...{first_run_note}")

        self._worker = SolveWorker(self._image_bgr, self._estimator)
        self._worker.finished_ok.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.start()

    def _on_solve_finished(self, people: list[DetectedPerson], estimator: PoseEstimator) -> None:
        self.solve_btn.setEnabled(True)
        self._people = people
        self._estimator = estimator

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

        self.export_all_btn.setEnabled(True)
        self.statusBar().showMessage(f"Solved {len(people)} pose(s)")
        self._update_preview_for_selected_person()

    def _on_solve_failed(self, message: str) -> None:
        self.solve_btn.setEnabled(True)
        self.statusBar().showMessage("Solve failed")
        QMessageBox.critical(self, "Solve Pose", f"Pose solving failed:\n{message}")

    def _on_person_changed(self) -> None:
        self._update_preview_for_selected_person()

    def _on_photo_clicked(self, x: float, y: float) -> None:
        if not self._people:
            return
        index = _nearest_person_index(self._people, x, y)
        self.person_combo.setCurrentIndex(index)  # triggers _on_person_changed via the signal

    def _selected_landmarks(self) -> np.ndarray | None:
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return None
        landmarks = self._people[index].landmarks
        if self.mirror_checkbox.isChecked():
            landmarks = mirror_landmarks(landmarks)
        return landmarks

    def _update_preview_for_selected_person(self) -> None:
        landmarks = self._selected_landmarks()
        if landmarks is None:
            self.export_btn.setEnabled(False)
            self.send_to_blender_btn.setEnabled(False)
            return

        self.viewport.show_skeleton(joint_positions(landmarks))
        self.export_btn.setEnabled(True)
        self.send_to_blender_btn.setEnabled(True)

    def _current_person_document_inputs(self):
        index = self.person_combo.currentIndex()
        landmarks = self._selected_landmarks()
        if landmarks is None:
            return None
        visibility = self._people[index].visibility
        rotations = solve_bone_rotations(landmarks)
        confidence = bone_confidence(visibility)
        return rotations, confidence

    def _on_export_pose(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, confidence = inputs

        path, _ = QFileDialog.getSaveFileName(self, "Export Pose JSON", "pose.json", "Pose JSON (*.json)")
        if not path:
            return

        save_pose_file(rotations, path, source_note="Exported from PoseCraft", bone_confidence=confidence)
        self.statusBar().showMessage(f"Exported pose to {path}")

    def _on_export_all(self) -> None:
        if not self._people:
            return

        directory = QFileDialog.getExistingDirectory(self, "Export All Poses To Folder")
        if not directory:
            return

        for index, person in enumerate(self._people):
            rotations = solve_bone_rotations(person.landmarks)
            confidence = bone_confidence(person.visibility)
            path = Path(directory) / f"pose_person{index + 1}.json"
            save_pose_file(rotations, path, source_note="Exported from PoseCraft (batch)", bone_confidence=confidence)

        self.statusBar().showMessage(f"Exported {len(self._people)} pose(s) to {directory}")

    def _on_send_to_blender(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, confidence = inputs
        document = build_pose_document(rotations, source_note="Sent live from PoseCraft", bone_confidence=confidence)
        try:
            send_pose_document(document)
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Send to Blender",
                "Could not reach Blender's Live Link.\n\n"
                "In Blender, open the PoseCraft sidebar tab and click 'Start Live Link', "
                "then try again.\n\n"
                f"Details: {exc}",
            )
            return

        self.statusBar().showMessage("Pose sent to Blender")

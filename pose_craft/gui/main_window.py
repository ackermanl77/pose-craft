from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pose_craft.estimation.landmarker import (
    DetectedPerson,
    PoseEstimator,
    ensure_model_downloaded,
    model_cache_path,
)
from pose_craft.export import live_link
from pose_craft.export.blender_pose import build_pose_document, save_pose_file
from pose_craft.export.live_link import send_pose_document
from pose_craft.gui.photo_view import PhotoOverlayPerson, PhotoView
from pose_craft.gui.viewport_3d import SkeletonViewport
from pose_craft.imaging import PHOTO_GUIDANCE, assess_photo_quality, imread_unicode
from pose_craft.retarget.skeleton import (
    bone_confidence,
    image_joint_positions_2d,
    joint_positions,
    mirror_landmarks,
    solve_bone_rotations,
)


class SolveWorker(QThread):
    """Runs model download + pose detection off the UI thread -- the first run in
    particular can take a while while the landmarker model downloads. Reuses an
    already-created PoseEstimator when the caller has one and its max_people
    already matches (see MainWindow._estimator) so the model only has to load
    once per app session, not once per click.
    """

    finished_ok = Signal(list, object)  # list[DetectedPerson], PoseEstimator
    failed = Signal(str)
    progress = Signal(int, int)  # downloaded_bytes, total_bytes -- only during a real download

    def __init__(
        self,
        image_bgr: np.ndarray,
        estimator: PoseEstimator | None,
        max_people: int,
        model_variant: str,
        parent=None,
    ):
        super().__init__(parent)
        self._image_bgr = image_bgr
        self._estimator = estimator
        self._max_people = max_people
        self._model_variant = model_variant

    def run(self) -> None:
        estimator = self._estimator
        newly_created = estimator is None
        try:
            if estimator is None:
                model_path = ensure_model_downloaded(
                    self._model_variant, on_progress=lambda downloaded, total: self.progress.emit(downloaded, total)
                )
                estimator = PoseEstimator(model_path, max_people=self._max_people)
            people = estimator.detect(self._image_bgr)
            self.finished_ok.emit(people, estimator)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            if newly_created and estimator is not None:
                estimator.close()
            self.failed.emit(str(exc))


class SendToBlenderWorker(QThread):
    """Runs the Live Link socket connection off the UI thread. A synchronous call
    here would freeze the whole window for up to send_pose_document's timeout
    (3s by default) whenever Blender's Live Link isn't running yet -- a likely
    first-time scenario, not an edge case.
    """

    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, document: dict, host: str, port: int, parent=None):
        super().__init__(parent)
        self._document = document
        self._host = host
        self._port = port

    def run(self) -> None:
        try:
            send_pose_document(self._document, host=self._host, port=self._port)
            self.finished_ok.emit()
        except OSError as exc:
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
        self._send_worker: SendToBlenderWorker | None = None
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

        left.addWidget(self._build_detect_group())
        left.addWidget(self._build_person_group())
        left.addWidget(self._build_export_group())
        left.addWidget(self._build_live_link_group())
        left.addStretch(1)

        right = QVBoxLayout()
        root.addLayout(right, stretch=2)
        right.addWidget(QLabel("3D preview"))
        self.viewport = SkeletonViewport()
        right.addWidget(self.viewport, stretch=1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_detect_group(self) -> QGroupBox:
        group = QGroupBox("1. Load && Detect")
        layout = QVBoxLayout(group)

        self.load_btn = QPushButton("Load Photo...")
        self.load_btn.clicked.connect(self._on_load_photo)
        layout.addWidget(self.load_btn)

        tips_label = QLabel("For best results:\n" + "\n".join(f"- {tip}" for tip in PHOTO_GUIDANCE))
        tips_label.setWordWrap(True)
        tips_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(tips_label)

        max_people_row = QHBoxLayout()
        max_people_row.addWidget(QLabel("Max people to detect:"))
        self.max_people_spin = QSpinBox()
        self.max_people_spin.setRange(1, 8)
        self.max_people_spin.setValue(4)
        self.max_people_spin.setToolTip(
            "A person beyond this count in a group photo won't be detected at all. "
            "Changing this after a solve rebuilds the pose model on the next Solve Pose."
        )
        max_people_row.addWidget(self.max_people_spin)
        layout.addLayout(max_people_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model quality:"))
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(["lite", "full", "heavy"])
        self.model_variant_combo.setCurrentText("full")
        self.model_variant_combo.setToolTip(
            "lite is fastest and best for low-end machines; heavy is the most accurate "
            "but slowest. Changing this after a solve rebuilds the pose model (and downloads "
            "it on first use) on the next Solve Pose."
        )
        model_row.addWidget(self.model_variant_combo)
        layout.addLayout(model_row)

        self.solve_btn = QPushButton("Solve Pose")
        self.solve_btn.clicked.connect(self._on_solve_pose)
        self.solve_btn.setEnabled(False)
        layout.addWidget(self.solve_btn)

        return group

    def _build_person_group(self) -> QGroupBox:
        group = QGroupBox("2. Select Person")
        layout = QVBoxLayout(group)

        person_row = QHBoxLayout()
        person_row.addWidget(QLabel("Person:"))
        self.person_combo = QComboBox()
        self.person_combo.currentIndexChanged.connect(self._on_person_changed)
        person_row.addWidget(self.person_combo)
        layout.addLayout(person_row)
        layout.addWidget(QLabel("(or click the person directly in the photo)"))

        self.mirror_checkbox = QCheckBox("Mirror left/right (body orientation fix)")
        self.mirror_checkbox.stateChanged.connect(self._on_person_changed)
        layout.addWidget(self.mirror_checkbox)

        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("3. Export")
        layout = QVBoxLayout(group)

        self.export_btn = QPushButton("Export Pose JSON...")
        self.export_btn.clicked.connect(self._on_export_pose)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        self.export_all_btn = QPushButton("Export All People...")
        self.export_all_btn.clicked.connect(self._on_export_all)
        self.export_all_btn.setEnabled(False)
        layout.addWidget(self.export_all_btn)

        return group

    def _build_live_link_group(self) -> QGroupBox:
        group = QGroupBox("4. Blender Live Link")
        layout = QVBoxLayout(group)

        live_link_row = QHBoxLayout()
        live_link_row.addWidget(QLabel("Blender host:"))
        self.live_link_host_edit = QLineEdit(live_link.DEFAULT_HOST)
        self.live_link_host_edit.setMaximumWidth(100)
        self.live_link_host_edit.setToolTip(
            "The Blender add-on only ever listens on 127.0.0.1 (localhost) -- change this "
            "only if PoseCraft and Blender already talk over some other configured "
            "loopback/tunnel, not to reach Blender on a different machine."
        )
        live_link_row.addWidget(self.live_link_host_edit)
        live_link_row.addWidget(QLabel("port:"))
        self.live_link_port_spin = QSpinBox()
        self.live_link_port_spin.setRange(1, 65535)
        self.live_link_port_spin.setValue(live_link.DEFAULT_PORT)
        self.live_link_port_spin.setToolTip(
            "Must match the port set in Blender's PoseCraft panel before Live Link is started there."
        )
        live_link_row.addWidget(self.live_link_port_spin)
        layout.addLayout(live_link_row)

        self.send_to_blender_btn = QPushButton("Send to Blender (Live Link)")
        self.send_to_blender_btn.clicked.connect(self._on_send_to_blender)
        self.send_to_blender_btn.setEnabled(False)
        layout.addWidget(self.send_to_blender_btn)

        return group

    def closeEvent(self, event) -> None:
        # Closing the shared PoseEstimator while a background thread is still
        # calling into it (or destroying a QThread object while it's still
        # running) is unsafe -- block briefly and let any in-flight work finish
        # first. Solves and Live Link sends are both short (seconds at most),
        # so this is a bounded wait, not an indefinite freeze.
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        if self._send_worker is not None and self._send_worker.isRunning():
            self._send_worker.wait()
        if self._estimator is not None:
            self._estimator.close()
        super().closeEvent(event)

    def _on_load_photo(self) -> None:
        # Guards against the same race closeEvent guards against: loading a new
        # photo while a solve for the OLD photo is still running would let that
        # solve's results land on the new photo once it finishes, and risks two
        # threads calling the shared PoseEstimator concurrently. load_btn is
        # disabled for the duration of a solve (see _on_solve_pose) so this
        # shouldn't be reachable from the UI, but checking here too costs nothing.
        if self._worker is not None and self._worker.isRunning():
            return

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
        self.photo_view.set_skeleton_overlay([])
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

        quality_warnings = assess_photo_quality(image_bgr)
        if quality_warnings:
            QMessageBox.warning(self, "Photo Quality", "\n\n".join(quality_warnings))

    def _set_photo_preview(self, image_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.photo_view.set_source_pixmap(QPixmap.fromImage(qimage))

    def _on_solve_pose(self) -> None:
        if self._image_bgr is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return  # defense in depth -- solve_btn being disabled should already prevent this

        max_people = self.max_people_spin.value()
        model_variant = self.model_variant_combo.currentText()
        if self._estimator is not None and (
            self._estimator.max_people != max_people or self._estimator.model_path != model_cache_path(model_variant)
        ):
            # num_poses and the model weights are both baked into the MediaPipe
            # graph at creation time -- changing either means the cached
            # estimator can't be reused as-is.
            self._estimator.close()
            self._estimator = None

        self.solve_btn.setEnabled(False)
        self.load_btn.setEnabled(False)  # a new photo mid-solve would let this solve's results land on it
        self.max_people_spin.setEnabled(False)
        self.model_variant_combo.setEnabled(False)
        first_run_note = "" if self._estimator is not None else " (downloading model on first run may take a moment)"
        self.statusBar().showMessage(f"Solving pose...{first_run_note}")

        self._worker = SolveWorker(self._image_bgr, self._estimator, max_people, model_variant)
        self._worker.finished_ok.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.progress.connect(self._on_solve_progress)
        self._worker.start()

    def _on_solve_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.statusBar().showMessage(f"Downloading pose model... {downloaded * 100 // total}%")
        else:
            self.statusBar().showMessage(f"Downloading pose model... {downloaded // 1024} KB")

    def _on_solve_finished(self, people: list[DetectedPerson], estimator: PoseEstimator) -> None:
        self.solve_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.max_people_spin.setEnabled(True)
        self.model_variant_combo.setEnabled(True)
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
        status = f"Solved {len(people)} pose(s)"
        if len(people) >= estimator.max_people:
            status += (
                f" -- reached the max-people limit ({estimator.max_people}); "
                "there may be more people in the photo than were detected"
            )
        self.statusBar().showMessage(status)
        self._update_preview_for_selected_person()

    def _on_solve_failed(self, message: str) -> None:
        self.solve_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.max_people_spin.setEnabled(True)
        self.model_variant_combo.setEnabled(True)
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
            self.photo_view.set_skeleton_overlay([])
            return

        self.viewport.show_skeleton(joint_positions(landmarks))
        self.export_btn.setEnabled(True)
        self.send_to_blender_btn.setEnabled(True)

        selected_index = self.person_combo.currentIndex()
        self.photo_view.set_skeleton_overlay([
            PhotoOverlayPerson(
                joints_2d=image_joint_positions_2d(person.image_landmarks_xy),
                confidence=bone_confidence(person.visibility),
                selected=(i == selected_index),
            )
            for i, person in enumerate(self._people)
        ])

    @staticmethod
    def _document_inputs_for(landmarks: np.ndarray, visibility: np.ndarray, mirror: bool) -> tuple[dict, dict]:
        """Shared by _current_person_document_inputs (the selected person) and
        _on_export_all (every detected person): mirror_landmarks just swaps
        left/right rows -- it doesn't care that `visibility` holds a score per
        landmark instead of a 3D position, and it needs to be swapped the same
        way `landmarks` is, or the confidence for e.g. a mirrored thigh.L would
        report the ORIGINAL left leg's visibility instead of the real
        (now swapped-in) right leg's.
        """
        if mirror:
            landmarks = mirror_landmarks(landmarks)
            visibility = mirror_landmarks(visibility)
        rotations = solve_bone_rotations(landmarks)
        confidence = bone_confidence(visibility)
        return rotations, confidence

    def _current_person_document_inputs(self):
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return None
        person = self._people[index]
        return self._document_inputs_for(person.landmarks, person.visibility, self.mirror_checkbox.isChecked())

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

        # Mirror is a property of the camera/photo orientation, not of one person, so
        # it applies to every person in a group photo the same way single-person
        # export already applies it via _selected_landmarks().
        mirror = self.mirror_checkbox.isChecked()

        for index, person in enumerate(self._people):
            rotations, confidence = self._document_inputs_for(person.landmarks, person.visibility, mirror)
            path = Path(directory) / f"pose_person{index + 1}.json"
            save_pose_file(rotations, path, source_note="Exported from PoseCraft (batch)", bone_confidence=confidence)

        self.statusBar().showMessage(f"Exported {len(self._people)} pose(s) to {directory}")

    def _on_send_to_blender(self) -> None:
        if self._send_worker is not None and self._send_worker.isRunning():
            return

        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, confidence = inputs
        document = build_pose_document(rotations, source_note="Sent live from PoseCraft", bone_confidence=confidence)

        host = self.live_link_host_edit.text().strip() or live_link.DEFAULT_HOST
        port = self.live_link_port_spin.value()

        self.send_to_blender_btn.setEnabled(False)
        self.statusBar().showMessage(f"Sending pose to Blender at {host}:{port}...")

        self._send_worker = SendToBlenderWorker(document, host, port)
        self._send_worker.finished_ok.connect(self._on_send_to_blender_finished)
        self._send_worker.failed.connect(self._on_send_to_blender_failed)
        self._send_worker.start()

    def _on_send_to_blender_finished(self) -> None:
        self.send_to_blender_btn.setEnabled(True)
        self.statusBar().showMessage("Pose sent to Blender")

    def _on_send_to_blender_failed(self, message: str) -> None:
        self.send_to_blender_btn.setEnabled(True)
        self.statusBar().showMessage("Send to Blender failed")
        QMessageBox.warning(
            self,
            "Send to Blender",
            "Could not reach Blender's Live Link.\n\n"
            "In Blender, open the PoseCraft sidebar tab and click 'Start Live Link', "
            "then try again.\n\n"
            f"Details: {message}",
        )

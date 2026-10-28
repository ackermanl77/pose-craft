"""Exporting to files: pose JSON, BVH/glTF (single poses and video animations), DAZ Studio
presets and "Export All" (C4: split out of main_window.py). A mixin of MainWindow; Live Link
(sending to Blender) stays in main_window.py with its worker thread."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from pose_craft.estimation.landmarker import (
    DetectedPerson,
)
from pose_craft.export.blender_pose import (
    save_pose_file,
)
from pose_craft.export.bvh_export import floor_root_position, save_bvh_animation, save_bvh_file
from pose_craft.export.skeleton_presets import DEFAULT_SKELETON
from pose_craft.gui.library_dialog import PoseLibraryDialog
from pose_craft.pose_library import LibraryError, PoseLibrary
from pose_craft.posing import (
    PersonEdits,
    default_export_name,
    document_inputs,
    out_of_frame_bones,
    root_for,
)

log = logging.getLogger("pose_craft.gui.main_window")


# BVH, binary glTF or FBX (F11), chosen by the file name's extension in the save dialog. FBX is
# offered when Blender is installed (export/fbx_export.py converts the glTF with it).
SKELETON_FILTERS = "BVH (*.bvh);;glTF binary (*.glb)"
FBX_FILTER = ";;FBX through Blender (*.fbx)"


def save_skeleton_animation(frames, path, *, frame_time, root_positions, skeleton=DEFAULT_SKELETON, blender=None):
    suffix = str(path).lower()
    if suffix.endswith(".glb"):
        from pose_craft.export.gltf_export import save_gltf_animation

        save_gltf_animation(frames, path, frame_time=frame_time, root_positions=root_positions, skeleton=skeleton)
    elif suffix.endswith(".fbx"):
        from pose_craft.export.fbx_export import save_fbx_animation

        save_fbx_animation(frames, path, blender=blender, frame_time=frame_time, root_positions=root_positions,
                           skeleton=skeleton)
    else:
        save_bvh_animation(frames, path, frame_time=frame_time, root_positions=root_positions, skeleton=skeleton)


class ExportMixin:
    # -- skeleton presets (B22) and FBX through Blender (F11) -------------------------------------

    def _skeleton(self) -> str:
        return self.skeleton_combo.currentData() or DEFAULT_SKELETON

    def _blender_path(self):
        from pose_craft.export.fbx_export import find_blender

        return find_blender(str(self._settings.value("blender/path", "") or ""))

    def _skeleton_filters(self) -> str:
        return SKELETON_FILTERS + (FBX_FILTER if self._blender_path() is not None else "")

    def _save_skeleton_pose(self, rotations, path: str) -> None:
        """One pose as BVH, glTF or FBX by the extension, on the chosen skeleton, standing on the floor (B6)."""
        save_skeleton_animation([rotations], path, frame_time=1.0 / 30.0, root_positions=[floor_root_position(rotations)],
                                skeleton=self._skeleton(), blender=self._blender_path())

    # -- pose library (F15) ------------------------------------------------------------------

    def _on_add_to_library(self) -> None:
        """Saves the selected person -- with their edits, the mirror switch and the
        orientation fix -- to the pose library, with a thumbnail of the photo."""
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return
        default = Path(self._source_path).stem if self._source_path else "Pose"
        if len(self._people) > 1:
            default += f" person {index + 1}"
        name, ok = QInputDialog.getText(self, "Add to Pose Library", "Name:", text=default)
        if not ok:
            return
        try:
            entry = PoseLibrary().add_person(
                self._people[index], name=name, edits=self._person_edits(index), aspect=self._photo_aspect(),
                orientation=self.orientation_combo.currentText(), mirror=self.mirror_checkbox.isChecked(),
                source=Path(self._source_path).name if self._source_path else "", image_bgr=self._preview_bgr,
            )
        except (LibraryError, OSError) as exc:
            QMessageBox.warning(self, "Add to Pose Library", f"Could not save the pose:\n{exc}")
            return
        self.statusBar().showMessage(f"Added \"{entry.name}\" to the pose library", 5000)

    def _on_open_library(self) -> None:
        self._library_dialog = PoseLibraryDialog(PoseLibrary(), send=self._start_live_link_send, parent=self)
        self._library_dialog.open()

    def _on_export_animation(self) -> None:
        track = self._current_track()
        if track is None:
            return
        _indices, frames, roots = self._video_track_frames(track)
        if not frames:
            return
        stem = Path(self._video.path).stem
        name = f"{stem}.bvh" if len(self._video.track_ids()) == 1 else default_export_name(self._video.path, track, ".bvh")
        path, _ = QFileDialog.getSaveFileName(self, "Export Animation BVH", str(Path(self._last_dir()) / name),
                                              self._skeleton_filters())
        if not path:
            return
        try:
            save_skeleton_animation(frames, path, frame_time=1.0 / self._video.fps, root_positions=roots,
                                    skeleton=self._skeleton(), blender=self._blender_path())
        except OSError as exc:
            self._report_export_error("Export Animation BVH", path, exc)
            return
        except RuntimeError as exc:  # Blender couldn't convert to FBX
            QMessageBox.warning(self, "Export FBX", str(exc))
            return
        self.statusBar().showMessage(f"Exported {len(frames)} frames of Person {track + 1} to {path}")

    def _on_export_all_tracks(self) -> None:
        """B12: one animated BVH per tracked person."""
        if self._video is None or not self._video.track_ids():
            return
        directory = QFileDialog.getExistingDirectory(self, "Export All People To Folder", self._last_dir())
        if not directory:
            return
        tracks = self._video.track_ids()
        paths = {track: Path(directory) / default_export_name(self._video.path, track, ".bvh") for track in tracks}
        existing = [p for p in paths.values() if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return
        for track, path in paths.items():
            _indices, frames, roots = self._video_track_frames(track)
            try:
                save_bvh_animation(frames, path, frame_time=1.0 / self._video.fps, root_positions=roots,
                                   skeleton=self._skeleton())
            except OSError as exc:
                self._report_export_error("Export All People", str(path), exc)
                return
        self.statusBar().showMessage(f"Exported {len(tracks)} animation(s) to {directory}")

    @staticmethod
    def _document_inputs_for(
        person: DetectedPerson,
        mirror: bool,
        orientation_correction=None,
        edits: PersonEdits | None = None,
        aspect: float = 1.0,
    ) -> tuple[dict, dict]:
        """Shared by _current_person_document_inputs (the selected person) and
        _on_export_all (every detected person): mirror_landmarks swaps the
        left/right rows (and, for 3D positions only, reflects X) -- a 1-D
        `visibility` array gets just the row swap, and it needs to be swapped
        the same way `landmarks` is, or the confidence for e.g. a mirrored thigh.L would
        report the ORIGINAL left leg's visibility instead of the real
        (now swapped-in) right leg's.

        orientation_correction (a scipy Rotation, or None for no change) is the
        same body-orientation fix applied live in the 3D preview -- see
        skeleton.apply_orientation_correction -- composed onto the root bone's
        rotation so every export (JSON, BVH, Live Link) agrees with what's shown
        on screen.

        edits: the person's manual corrections (F4) -- moved joints and depth
        flips on the raw landmarks before mirroring (they name what was
        clicked in the photo), bone adjustments on the solved rotations.
        aspect: the photo's width / height, for turning a joint drag into 3D.

        The person's hand landmarks (B11), when the hands were solved, add
        finger bones; they're mirrored and flipped along with the body.
        """
        return document_inputs(person, mirror, orientation_correction, edits, aspect)

    def _current_person_document_inputs(self):
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return None
        return self._person_document_inputs(index)

    def _mirrored_face(self, face) -> dict | None:
        """A FaceLandmarks' expression weights as exported (F12): Left/Right swapped with the mirror."""
        if face is None:
            return None
        from pose_craft.export.blender_pose import mirror_blendshapes

        return mirror_blendshapes(face.blendshapes) if self.mirror_checkbox.isChecked() else dict(face.blendshapes)

    def _face_for(self, index: int) -> dict | None:
        """Person `index`'s expression weights (F12), or None when no face was read."""
        if not 0 <= index < len(self._people):
            return None
        return self._mirrored_face(getattr(self._people[index], "face", None))

    def _person_document_inputs(self, index: int):
        return self._document_inputs_for(
            self._people[index],
            self.mirror_checkbox.isChecked(),
            self._orientation_correction(),
            self._person_edits(index),
            self._photo_aspect(),
        )

    def _out_of_frame_for(self, index: int) -> set[str]:
        """Person `index`'s bones reaching outside the photo (B10), named as exported."""
        return out_of_frame_bones(self._edited(index), self.mirror_checkbox.isChecked())

    def _root_for(self, index: int, rotations: dict) -> tuple[float, float]:
        """(hip_height, leg_length) in meters for person `index`'s export (B6)."""
        return root_for(self._edited(index), self.mirror_checkbox.isChecked(), rotations)

    def _export_path(self, title: str, name: str, file_filter: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(self._last_dir()) / name), file_filter)
        if path:
            self._settings.setValue("last_dir", str(Path(path).parent))
        return path

    def _on_export_pose(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, confidence = inputs
        index = self.person_combo.currentIndex()

        path = self._export_path(
            "Export Pose JSON", default_export_name(self._source_path, index, ".json"), "Pose JSON (*.json)"
        )
        if not path:
            return

        hip_height, leg_length = self._root_for(index, rotations)
        try:
            save_pose_file(
                rotations, path, source_note="Exported from PoseCraft", bone_confidence=confidence,
                hip_height=hip_height, leg_length=leg_length, out_of_frame=self._out_of_frame_for(index), face_blendshapes=self._face_for(index),
            )
        except OSError as exc:
            self._report_export_error("Export Pose JSON", path, exc)
            return
        self.statusBar().showMessage(f"Exported pose to {path}")

    def _on_export_all(self) -> None:
        if not self._people:
            return

        directory = QFileDialog.getExistingDirectory(self, "Export All Poses To Folder", self._last_dir())
        if not directory:
            return

        # Mirror and the orientation fix are both properties of the camera/photo,
        # not of one person, so they apply to every person in a group photo the
        # same way single-person export already applies them via
        # _current_person_document_inputs().
        mirror = self.mirror_checkbox.isChecked()
        orientation_correction = self._orientation_correction()
        paths = [Path(directory) / default_export_name(self._source_path, i, ".json") for i in range(len(self._people))]
        existing = [p for p in paths if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return

        for index, (person, path) in enumerate(zip(self._people, paths)):
            rotations, confidence = self._document_inputs_for(
                person, mirror, orientation_correction, self._person_edits(index), self._photo_aspect()
            )
            hip_height, leg_length = self._root_for(index, rotations)
            try:
                save_pose_file(
                    rotations, path, source_note="Exported from PoseCraft (batch)", bone_confidence=confidence,
                    hip_height=hip_height, leg_length=leg_length, out_of_frame=self._out_of_frame_for(index), face_blendshapes=self._face_for(index),
                )
            except OSError as exc:
                self._report_export_error("Export All People", str(path), exc)
                return

        self.statusBar().showMessage(f"Exported {len(self._people)} pose(s) to {directory}")

    def _on_export_bvh(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, _confidence = inputs

        path = self._export_path(
            "Export BVH", default_export_name(self._source_path, self.person_combo.currentIndex(), ".bvh"),
            self._skeleton_filters(),
        )
        if not path:
            return

        try:
            # Standing on the floor (B6): a crouch is exported lowered. .glb: glTF, .fbx: FBX (F11).
            self._save_skeleton_pose(rotations, path)
        except OSError as exc:
            self._report_export_error("Export BVH", path, exc)
            return
        except RuntimeError as exc:  # Blender couldn't convert to FBX
            QMessageBox.warning(self, "Export FBX", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(path).suffix.lstrip('.').upper()} to {path}")

    def _on_export_daz(self) -> None:
        from pose_craft.export.daz_pose import (
            DazFigureNotFound,
            channels_past_limits,
            load_daz_figure,
            save_daz_pose_file,
        )

        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, _confidence = inputs
        index = self.person_combo.currentIndex()
        try:
            figure = load_daz_figure()
        except (DazFigureNotFound, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export DAZ Pose", str(exc))
            return
        path = self._export_path(
            "Export DAZ Pose (Genesis 9)", default_export_name(self._source_path, index, ".duf"),
            "DAZ Studio pose preset (*.duf)",
        )
        if not path:
            return
        hip_height, leg_length = self._root_for(index, rotations)
        try:
            save_daz_pose_file(rotations, path, figure, hip_height=hip_height, leg_length=leg_length)
        except OSError as exc:
            self._report_export_error("Export DAZ Pose", path, exc)
            return
        past = channels_past_limits(rotations, figure)
        note = ""
        if past:
            bones = sorted({bone for bone, _axis, _value, _limit in past})
            note = f" -- beyond Genesis 9's joint limits (DAZ will clamp): {', '.join(bones)}"
        self.statusBar().showMessage(f"Exported DAZ pose to {path}{note}")

    def _on_export_all_bvh(self) -> None:
        if not self._people or self._video is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export All BVHs To Folder", self._last_dir())
        if not directory:
            return
        paths = [Path(directory) / default_export_name(self._source_path, i, ".bvh") for i in range(len(self._people))]
        existing = [p for p in paths if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People (BVH)", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return
        for index, (person, path) in enumerate(zip(self._people, paths)):
            rotations, _confidence = self._document_inputs_for(
                person, self.mirror_checkbox.isChecked(), self._orientation_correction(), self._person_edits(index),
                self._photo_aspect(),
            )
            try:
                save_bvh_file(rotations, path, root_position=floor_root_position(rotations), skeleton=self._skeleton())
            except OSError as exc:
                self._report_export_error("Export All People (BVH)", str(path), exc)
                return
        self.statusBar().showMessage(f"Exported {len(paths)} BVH file(s) to {directory}")

    def _report_export_error(self, title: str, path: str, exc: OSError) -> None:
        """A failed write (read-only folder, locked file, OneDrive placeholder)
        used to raise inside the Qt slot, which the windowed build has no
        console to show -- the export silently did nothing."""
        log.warning("%s failed for %s: %s", title, path, exc)
        self.statusBar().showMessage(f"{title} failed")
        QMessageBox.warning(self, title, f"Could not write:\n{path}\n\n{exc}")

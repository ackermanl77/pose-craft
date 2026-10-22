"""Pose > Pose Library... (report item F15): browse the saved poses as
thumbnails, filter them by text or tag, rename and tag them, and export,
send to Blender, mirror, blend or delete them (pose_craft/pose_library.py).

The window passes a `send` callback (its Live Link worker) so a library pose
goes to Blender exactly like the current pose does."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from pose_craft.export.live_link import LIBRARY_FORMAT_NAME
from pose_craft.pose_library import LibraryError, PoseLibrary, read_thumbnail

ALL_TAGS = "All tags"
EXPORT_FILTERS = "Pose JSON (*.json);;BVH (*.bvh);;glTF binary (*.glb);;DAZ Studio pose preset (*.duf)"


def _icon(path: Path) -> QIcon:
    image = read_thumbnail(path)
    if image is None:
        return QIcon()
    rgb = image[:, :, ::-1].copy()
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QIcon(QPixmap.fromImage(qimage.copy()))


class BlendDialog(QDialog):
    """Pick how far from the first pose to the second the blend goes."""

    def __init__(self, first: str, second: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blend Poses")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.setAccessibleName("Blend amount")
        self.value_label = QLabel()
        self.slider.valueChanged.connect(self._show_value)
        self._first, self._second = first, second
        self._show_value(50)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name (optional)")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.value_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.name_edit)
        layout.addWidget(buttons)

    def _show_value(self, value: int) -> None:
        self.value_label.setText(f"{100 - value}% {self._first}  +  {value}% {self._second}")

    def amount(self) -> float:
        return self.slider.value() / 100


class PoseLibraryDialog(QDialog):
    def __init__(self, library: PoseLibrary, send=None, parent=None):
        super().__init__(parent)
        self.library = library
        self._send = send
        self.setWindowTitle("Pose Library")
        self.resize(820, 560)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search names, tags and sources")
        self.search_edit.setAccessibleName("Search the pose library")
        self.search_edit.textChanged.connect(self.refresh)
        self.tag_combo = QComboBox()
        self.tag_combo.setAccessibleName("Filter by tag")
        self.tag_combo.currentTextChanged.connect(self.refresh)
        filters = QHBoxLayout()
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(self.tag_combo)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(128, 128))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.setAccessibleName("Saved poses")
        self.list.itemSelectionChanged.connect(self._on_selection)

        self.name_edit = QLineEdit()
        self.name_edit.setAccessibleName("Pose name")
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("comma, separated, tags")
        self.tags_edit.setAccessibleName("Pose tags")
        self.source_label = QLabel()
        self.save_details_btn = QPushButton("Save Name && Tags")
        self.save_details_btn.clicked.connect(self._on_save_details)
        details = QFormLayout()
        details.addRow("&Name", self.name_edit)
        details.addRow("&Tags", self.tags_edit)
        details.addRow("Source", self.source_label)
        details.addRow("", self.save_details_btn)

        self.export_btn = QPushButton("&Export...")
        self.export_btn.clicked.connect(self._on_export)
        self.send_btn = QPushButton("Send to &Blender")
        self.send_btn.clicked.connect(self._on_send)
        self.send_btn.setVisible(send is not None)
        # F17: every pose (or the selected ones) as Pose Library assets in Blender.
        self.send_library_btn = QPushButton("Send Library to Blender")
        self.send_library_btn.setToolTip(
            "Make Blender Pose Library assets (with thumbnails) from the selected poses, or all of them, on the "
            "Live Link target armature -- then blend and flip them in Blender's Pose Library."
        )
        self.send_library_btn.clicked.connect(self._on_send_library)
        self.send_library_btn.setVisible(send is not None)
        self.mirror_btn = QPushButton("&Mirror Copy")
        self.mirror_btn.clicked.connect(self._on_mirror)
        self.blend_btn = QPushButton("B&lend Two...")
        self.blend_btn.setToolTip("Select two poses (Ctrl+click) to mix them")
        self.blend_btn.clicked.connect(self._on_blend)
        self.delete_btn = QPushButton("&Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        actions = QHBoxLayout()
        for button in (self.export_btn, self.send_btn, self.send_library_btn, self.mirror_btn, self.blend_btn,
                       self.delete_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        actions.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addLayout(filters)
        layout.addWidget(self.list, 1)
        layout.addLayout(details)
        layout.addLayout(actions)
        self.refresh()

    # -- listing ---------------------------------------------------------------

    def refresh(self) -> None:
        tags = self.library.tags()
        current = self.tag_combo.currentText()
        self.tag_combo.blockSignals(True)
        self.tag_combo.clear()
        self.tag_combo.addItems([ALL_TAGS, *tags])
        self.tag_combo.setCurrentText(current if current in tags else ALL_TAGS)
        self.tag_combo.blockSignals(False)
        tag = "" if self.tag_combo.currentText() == ALL_TAGS else self.tag_combo.currentText()
        selected = set(self.selected_ids())
        self.list.clear()
        for entry in self.library.entries(self.search_edit.text(), tag):
            item = QListWidgetItem(_icon(self.library.thumbnail_path(entry.id)), entry.name)
            item.setData(Qt.UserRole, entry.id)
            item.setToolTip(", ".join(entry.tags) or entry.source)
            self.list.addItem(item)
            item.setSelected(entry.id in selected)
        self._on_selection()

    def selected_ids(self) -> list[str]:
        return [item.data(Qt.UserRole) for item in self.list.selectedItems()]

    def select(self, entry_id: str) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setSelected(item.data(Qt.UserRole) == entry_id)

    def _on_selection(self) -> None:
        ids = self.selected_ids()
        one = len(ids) == 1
        for button in (self.export_btn, self.send_btn, self.mirror_btn, self.delete_btn, self.save_details_btn):
            button.setEnabled(one if button is not self.delete_btn else bool(ids))
        self.blend_btn.setEnabled(len(ids) == 2)
        self.name_edit.setEnabled(one)
        self.tags_edit.setEnabled(one)
        if one:
            entry = self.library.get(ids[0])
            self.name_edit.setText(entry.name)
            self.tags_edit.setText(", ".join(entry.tags))
            mirrored = " (mirrored)" if entry.mirror else ""
            self.source_label.setText((entry.source or "-") + mirrored)
            self.mirror_btn.setEnabled(entry.kind == "person")
        else:
            self.name_edit.clear()
            self.tags_edit.clear()
            self.source_label.setText(f"{len(ids)} selected" if ids else "")

    # -- actions ---------------------------------------------------------------

    def _on_save_details(self) -> None:
        (entry_id,) = self.selected_ids()
        self.library.update(entry_id, name=self.name_edit.text(), tags=self.tags_edit.text())
        self.refresh()
        self.select(entry_id)

    def _on_export(self) -> None:
        (entry_id,) = self.selected_ids()
        entry = self.library.get(entry_id)
        path, chosen = QFileDialog.getSaveFileName(self, "Export Library Pose", f"{entry.name}.json", EXPORT_FILTERS)
        if not path:
            return
        if not Path(path).suffix:  # take the extension from the chosen filter
            path += chosen[chosen.rfind("*.") + 1: chosen.rfind(")")]
        try:
            self.library.export(entry_id, path)
        except (LibraryError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export Library Pose", f"Could not export {Path(path).name}:\n{exc}")
            return
        self.export_btn.setToolTip(f"Exported to {path}")

    def _on_send(self) -> None:
        (entry_id,) = self.selected_ids()
        if self._send is not None:
            self._send(self.library.live_link_document(entry_id))

    def _on_send_library(self) -> None:
        if self._send is None:
            return
        try:
            bundle = self.library.export_blender_bundle(
                Path(tempfile.gettempdir()) / "PoseCraft library for Blender", self.selected_ids() or None
            )
        except (LibraryError, OSError) as exc:
            QMessageBox.warning(self, "Send Library to Blender", str(exc))
            return
        self._send({"format": LIBRARY_FORMAT_NAME, "path": str(bundle)})

    def _on_mirror(self) -> None:
        (entry_id,) = self.selected_ids()
        copy = self.library.mirrored_copy(entry_id)
        self.refresh()
        self.select(copy.id)

    def _on_blend(self) -> None:
        first, second = (self.library.get(i) for i in self.selected_ids())
        dialog = BlendDialog(first.name, second.name, self)
        if dialog.exec() != QDialog.Accepted:
            return
        entry = self.library.blend(first.id, second.id, dialog.amount(), name=dialog.name_edit.text(),
                                   tags=sorted(set(first.tags) & set(second.tags)))
        self.refresh()
        self.select(entry.id)

    def _on_delete(self) -> None:
        ids = self.selected_ids()
        names = ", ".join(self.library.get(i).name for i in ids)
        if QMessageBox.question(self, "Delete Poses", f"Delete {names} from the library?") != QMessageBox.Yes:
            return
        for entry_id in ids:
            self.library.delete(entry_id)
        self.refresh()

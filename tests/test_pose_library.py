"""The pose library (report item F15): save, re-export, mirror, blend, tag."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pose_craft import pose_library as pl
from pose_craft.export.blender_pose import save_pose_file
from pose_craft.posing import PersonEdits, document_inputs, edited_person, root_for
from pose_craft.retarget.skeleton import ORIENTATION_CORRECTIONS

sys.path.insert(0, str(Path(__file__).parent))
from test_gui_flows import _person  # noqa: E402
from test_retarget_frames import _articulated_pose  # noqa: E402

EDITS = PersonEdits(flipped=frozenset({"forearm.L"}), overrides=(("spine", (5.0, -3.0, 2.0)),),
                    moved=((15, (0.42, 0.61)),))


def _angle(a, b) -> float:
    return float(np.degrees((a.inv() * b).magnitude()))


@pytest.fixture
def library(tmp_path):
    return pl.PoseLibrary(tmp_path / "Imágenes" / "poses")  # non-ASCII on purpose


@pytest.fixture
def photo():
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :320] = (30, 120, 220)
    return image


def test_a_saved_person_exports_exactly_like_the_window_did(library, photo):
    person = _person(_articulated_pose())
    entry = library.add_person(person, name="  Warrior ", tags="yoga, Standing,yoga", edits=EDITS, aspect=4 / 3,
                               orientation="Rotate 90° Left", source="IMG_1.jpg", image_bgr=photo)
    assert (entry.name, entry.tags, entry.kind) == ("Warrior", ["yoga", "Standing"], "person")
    assert library.thumbnail_path(entry.id).is_file()
    assert pl.read_thumbnail(library.thumbnail_path(entry.id)).shape == (256, 256, 3)

    inputs = library.export_inputs(entry.id)
    rotations, confidence = document_inputs(person, False, ORIENTATION_CORRECTIONS["Rotate 90° Left"], EDITS, 4 / 3)
    assert set(inputs.rotations) == set(rotations) and inputs.confidence == pytest.approx(confidence)
    assert max(_angle(inputs.rotations[b], rotations[b]) for b in rotations) < 1e-6
    hip, leg = root_for(edited_person(person, EDITS, 4 / 3), False, rotations)
    assert (inputs.hip_height, inputs.leg_length) == pytest.approx((hip, leg))


def test_every_export_format_is_written(library, tmp_path):
    entry = library.add_person(_person(_articulated_pose()), name="Pose")
    for suffix in (".json", ".bvh", ".glb"):
        assert library.export(entry.id, tmp_path / f"out{suffix}").stat().st_size > 100
    reference = tmp_path / "reference.json"
    inputs = library.export_inputs(entry.id)
    save_pose_file(inputs.rotations, reference, source_note="PoseCraft library: Pose", bone_confidence=inputs.confidence,
                   hip_height=inputs.hip_height, leg_length=inputs.leg_length, out_of_frame=inputs.out_of_frame)
    assert json.loads((tmp_path / "out.json").read_text()) == json.loads(reference.read_text())
    with pytest.raises(pl.LibraryError, match="export as one of"):
        library.export(entry.id, tmp_path / "out.fbx")
    document = library.live_link_document(entry.id)
    assert document["format"] == "pose_craft_pose" and document["bones"]


def test_a_mirrored_copy_is_the_mirror_switch_flipped(library, photo):
    person = _person(_articulated_pose())
    entry = library.add_person(person, name="Lunge", tags=["legs"], image_bgr=photo)
    mirrored = library.mirrored_copy(entry.id)
    assert (mirrored.name, mirrored.mirror, mirrored.tags) == ("Lunge (mirrored)", True, ["legs"])
    expected, _ = document_inputs(person, True, ORIENTATION_CORRECTIONS["Default"], None, 1.0)
    got = library.export_inputs(mirrored.id).rotations
    assert max(_angle(got[b], expected[b]) for b in expected) < 1e-6
    thumb, mirrored_thumb = (pl.read_thumbnail(library.thumbnail_path(i)) for i in (entry.id, mirrored.id))
    np.testing.assert_array_equal(mirrored_thumb, thumb[:, ::-1])


def test_blending_slerps_each_bone_between_the_two_poses(library):
    a = library.add_person(_person(_articulated_pose()), name="A", tags=["x", "shared"])
    b = library.mirrored_copy(a.id)
    library.update(b.id, tags=["shared", "y"])
    ra, rb = library.export_inputs(a.id).rotations, library.export_inputs(b.id).rotations
    start, end, middle = (library.blend(a.id, b.id, t, tags=["shared"]) for t in (0.0, 1.0, 0.25))
    for blend, reference in ((start, ra), (end, rb)):
        got = library.export_inputs(blend.id).rotations
        assert max(_angle(got[bone], reference[bone]) for bone in reference) < 1e-6
    got = library.export_inputs(middle.id).rotations
    for bone in ra:
        total = _angle(ra[bone], rb[bone])
        assert _angle(ra[bone], got[bone]) == pytest.approx(0.25 * total, abs=1e-6)
    assert library.get(middle.id).name == "A / A (mirrored) 25%"
    with pytest.raises(pl.LibraryError, match="only poses from a photo can be mirrored"):
        library.mirrored_copy(middle.id)


def test_listing_filters_by_text_and_tag_and_skips_broken_entries(library):
    first = library.add_person(_person(_articulated_pose()), name="Tree pose", tags=["yoga"])
    library.add_person(_person(_articulated_pose()), name="Jump", tags=["sport"], source="beach.jpg")
    (library.root / "not-an-entry").mkdir()
    (library.root / "half-written").mkdir()
    (library.root / "half-written" / "meta.json").write_text("{ broken")
    assert [e.name for e in library.entries()] == ["Jump", "Tree pose"]  # newest first
    assert [e.name for e in library.entries(tag="YOGA")] == ["Tree pose"]
    assert [e.name for e in library.entries("beach")] == ["Jump"]
    assert library.tags() == ["sport", "yoga"]
    library.delete(first.id)
    assert [e.name for e in library.entries()] == ["Jump"]
    with pytest.raises(pl.LibraryError):
        library.get("../escape")


def test_an_entry_from_a_newer_version_is_refused(library):
    entry = library.add_person(_person(_articulated_pose()), name="Future")
    path = library.root / entry.id / "pose.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps(dict(data, version=pl.VERSION + 1)))
    with pytest.raises(pl.LibraryError, match="newer PoseCraft"):
        library.export_inputs(entry.id)


def test_the_library_folder_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("POSECRAFT_LIBRARY_DIR", str(tmp_path / "lib"))
    assert pl.PoseLibrary().root == tmp_path / "lib"
    monkeypatch.delenv("POSECRAFT_LIBRARY_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert pl.default_library_dir() == tmp_path / "local" / "PoseCraft" / "pose_library"


# --- in the window ----------------------------------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402
from test_gui_important import _window_with_photo  # noqa: E402

from pose_craft.gui import library_dialog, window_export  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def test_the_window_adds_poses_and_the_library_dialog_works_with_them(qapp, settings, monkeypatch, tmp_path):
    window = _window_with_photo(settings, [_person(_articulated_pose()), _person(_articulated_pose())])
    sent = []
    try:
        assert window.add_to_library_action.isEnabled()
        window.person_combo.setCurrentIndex(1)
        window.mirror_checkbox.setChecked(True)
        monkeypatch.setattr(window_export.QInputDialog, "getText", lambda *a, **k: ("Second", True))
        window._on_add_to_library()
        (entry,) = pl.PoseLibrary().entries()
        assert (entry.name, entry.mirror) == ("Second", True)

        monkeypatch.setattr(window, "_start_live_link_send", sent.append)
        window._on_open_library()
        dialog = window._library_dialog
        assert dialog.list.count() == 1
        dialog.select(entry.id)
        dialog.name_edit.setText("Second pose")
        dialog.tags_edit.setText("test, demo")
        dialog._on_save_details()
        assert pl.PoseLibrary().get(entry.id).tags == ["test", "demo"]
        dialog._on_send()
        assert sent and sent[0]["format"] == "pose_craft_pose"

        dialog._on_mirror()
        assert dialog.list.count() == 2
        ids = [dialog.list.item(r).data(library_dialog.Qt.UserRole) for r in range(2)]
        for row in range(2):
            dialog.list.item(row).setSelected(True)
        assert dialog.blend_btn.isEnabled()
        monkeypatch.setattr(library_dialog.BlendDialog, "exec", lambda self: QDialog.Accepted)
        dialog._on_blend()
        assert dialog.list.count() == 3 and len(dialog.selected_ids()) == 1

        out = tmp_path / "exported"
        monkeypatch.setattr(library_dialog.QFileDialog, "getSaveFileName",
                            lambda *a, **k: (str(out), "BVH (*.bvh)"))
        dialog._on_export()
        assert (tmp_path / "exported.bvh").stat().st_size > 100

        dialog.tag_combo.setCurrentText("demo")
        assert dialog.list.count() == 3  # the blend keeps the tags its two poses share
        dialog.search_edit.setText("mirrored 50%")
        assert dialog.list.count() == 1  # only the blend's name has both words
        dialog.search_edit.clear()
        dialog.tag_combo.setCurrentText(library_dialog.ALL_TAGS)
        monkeypatch.setattr(library_dialog.QMessageBox, "question", lambda *a, **k: library_dialog.QMessageBox.Yes)
        dialog.select(ids[0])
        dialog._on_delete()
        assert dialog.list.count() == 2
        dialog.close()
    finally:
        window.close()

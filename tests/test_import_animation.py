"""Importing an animation from another tool (F18): any BVH skeleton becomes a tracked
person whose re-solved bones point where the file's did."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export.bvh_export import build_bvh_animation, floor_root_position
from pose_craft.import_animation import ImportError_, import_bvh, landmarks_from_clip, parse_bvh
from pose_craft.retarget import skeleton as sk

BODY_BONES = ("upper_arm.L", "forearm.L", "upper_arm.R", "forearm.R", "thigh.L", "shin.L", "thigh.R", "shin.R", "spine")


def _pose(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pose = {name: Rotation.from_rotvec(rng.normal(scale=0.35, size=3)) for name in sk.ALL_BONE_NAMES if "." not in name or name.split(".")[0] in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")}
    pose["spine"] = Rotation.from_euler("y", 30, degrees=True) * pose.get("spine", Rotation.identity())
    return pose


def _directions(rotations: dict) -> dict:
    world = sk.world_rotations_from_local(rotations)
    return {name: world[name].apply(sk.bone_rest_direction(name)) for name in BODY_BONES}


@pytest.mark.parametrize("skeleton", ["posecraft", "mixamo", "ue5", "humanik"])
def test_an_exported_bvh_imports_back_to_the_same_limb_directions(tmp_path, skeleton):
    poses = [_pose(1), _pose(2)]
    text = build_bvh_animation(poses, root_positions=[floor_root_position(p) for p in poses], skeleton=skeleton)
    path = tmp_path / f"{skeleton}.bvh"
    path.write_text(text)
    video = import_bvh(path)
    assert len(video.frames) == 2 and video.fps == pytest.approx(30.0) and video.track_ids() == [0]
    for pose, frame in zip(poses, video.frames):
        solved = sk.solve_bone_rotations(frame.people[0].landmarks)
        expected, got = _directions(pose), _directions(solved)
        for bone in BODY_BONES:
            angle = np.degrees(np.arccos(np.clip(np.dot(expected[bone], got[bone]) / np.linalg.norm(got[bone]), -1, 1)))
            assert angle < 2.0, (skeleton, bone, angle)


def test_a_z_up_file_facing_away_imports_the_same():
    """Re-express a Mixamo export in a Z-up, facing -Y frame: the rest pose sets the axes."""
    pose = _pose(3)
    clip = parse_bvh(build_bvh_animation([pose], skeleton="mixamo"))
    turn = Rotation.from_euler("x", 90, degrees=True) * Rotation.from_euler("y", 180, degrees=True)
    for joint in clip.joints:
        joint.offset = turn.apply(joint.offset)
        if joint.end_site is not None:
            joint.end_site = turn.apply(joint.end_site)
    # Every joint's rotation re-expressed in the new axes: r' = T r T^-1.
    cursor = 0
    for joint in clip.joints:
        rotation_at = cursor + (3 if len(joint.channels) == 6 else 0)
        assert joint.channels[-3:] == ["Zrotation", "Xrotation", "Yrotation"]
        rotation = Rotation.from_euler("ZXY", clip.frames[0][rotation_at:rotation_at + 3], degrees=True)
        clip.frames[0][rotation_at:rotation_at + 3] = (turn * rotation * turn.inv()).as_euler("ZXY", degrees=True)
        cursor += len(joint.channels)
    original = landmarks_from_clip(parse_bvh(build_bvh_animation([pose], skeleton="mixamo")))[0]
    turned = landmarks_from_clip(clip)[0]
    assert np.allclose(original, turned, atol=1e-6)


def test_files_that_cant_be_imported_say_why(tmp_path):
    with pytest.raises(ImportError_, match="not a BVH"):
        parse_bvh("hello")
    text = build_bvh_animation([{}]).replace("hand.L", "paw.L")
    with pytest.raises(ImportError_, match="hand L"):
        landmarks_from_clip(parse_bvh(text))


def test_the_window_imports_an_animation_and_exports_it_on_another_skeleton(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from pose_craft.gui import main_window as mw

    QApplication.instance() or QApplication([])
    poses = [_pose(4), _pose(5), _pose(6)]
    source = tmp_path / "sam3d.bvh"
    source.write_text(build_bvh_animation(poses, skeleton="ue5", frame_time=1 / 24))
    window = mw.MainWindow(settings=QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    try:
        assert window.import_animation(str(source))
        assert not window.video_group.isHidden() and window.frame_slider.maximum() == 2
        assert window.person_combo.count() == 1 and window.export_btn.isEnabled()
        window.skeleton_combo.setCurrentIndex(window.skeleton_combo.findData("mixamo"))
        out = tmp_path / "retargeted.bvh"
        monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._on_export_animation()
        text = out.read_text(encoding="utf-8")
        assert "ROOT mixamorig:Hips" in text and "Frames: 3" in text and "Frame Time: 0.041667" in text
        window._save_settings()
        assert window._settings.value("export/skeleton") == "mixamo"
    finally:
        window.close()

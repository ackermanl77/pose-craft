"""DAZ Studio pose presets for Genesis 9 (export/daz_pose.py), on a small stand-in figure written by
the test: no DAZ content ships with the repository. tests/test_real_daz.py checks the result in a
real DAZ Studio against the installed Genesis 9."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import daz_pose
from pose_craft.retarget import skeleton as sk


def _node(node_id, parent, order, center, end, orientation=(0, 0, 0), limits=None):
    rotation = [{"id": axis, "value": 0, **({"min": limits[axis][0], "max": limits[axis][1], "clamped": True}
                                           if limits and axis in limits else {})} for axis in "xyz"]
    return {
        "id": node_id, "type": "bone", "parent": f"#{parent}", "rotation_order": order,
        "center_point": [{"id": a, "value": v} for a, v in zip("xyz", center)],
        "end_point": [{"id": a, "value": v} for a, v in zip("xyz", end)],
        "orientation": [{"id": a, "value": v} for a, v in zip("xyz", orientation)],
        "rotation": rotation,
    }


def _figure_file(tmp_path, compressed=False):
    """A Genesis 9-shaped skeleton: an A-pose left arm with oriented joints, legs, a spine chain."""
    nodes = [
        {"id": "Genesis9", "type": "figure", "name": "Genesis9"},
        _node("l_forearm", "l_upperarm", "XZY", (33, 116.6, -3.9), (51.2, 100.1, 2.2), (1.16, -13.92, -42.16),
              {"z": (0, 0)}),  # out of order on purpose: parents come first after loading
        _node("hip", "Genesis9", "YZX", (0, 97.1, 0.5), (0, 80.9, 0.2)),
        _node("pelvis", "hip", "YZX", (0, 99, -0.3), (0, 80, -0.3)),
        _node("l_thigh", "pelvis", "YZX", (7.6, 88.7, -0.4), (9.3, 48.2, -1.8), (1.92, 4.95, 2.48)),
        _node("l_shin", "l_thigh", "YZX", (9.3, 48.2, -1.8), (10.9, 6.4, -4.7), (4.01, 6.82, 2.48)),
        _node("spine1", "hip", "YZX", (0, 98.9, -0.8), (0, 106.7, -0.8)),
        _node("l_shoulder", "spine1", "XYZ", (3.5, 135.5, -3.6), (13.9, 137.2, -4.3), (0.05, 3.51, 1.75)),
        _node("l_upperarm", "l_shoulder", "XYZ", (16.2, 134.7, -4.4), (33, 116.6, -3.9), (1.23, -1.15, -47.92),
              {"z": (-40, 90)}),
        _node("neck1", "spine1", "YZX", (0, 140, -3.5), (0, 146, -2)),
        _node("head", "neck1", "YZX", (0, 152, -1), (0, 170, -1)),
    ]
    path = tmp_path / "Genesis9.dsf"
    payload = json.dumps({"file_version": "0.6.0.0", "node_library": nodes}).encode()
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    return path


def _reconstruct_world(channels, figure):
    """World rotations back from channel values, with DAZ's measured convention (module docstring)."""
    world = {}
    for name, bone in figure.bones.items():
        values = dict(zip("XYZ", channels[name]))
        euler = Rotation.from_euler(bone.rotation_order.lower(), [values[a] for a in bone.rotation_order], degrees=True)
        local = bone.orientation * euler * bone.orientation.inv()
        world[name] = (world[bone.parent] if bone.parent else Rotation.identity()) * local
    return world


def test_the_figure_loads_parents_first_from_a_plain_or_gzipped_dsf(tmp_path):
    for compressed in (False, True):
        figure = daz_pose.load_daz_figure(_figure_file(tmp_path, compressed))
        names = list(figure.bones)
        assert names.index("l_upperarm") < names.index("l_forearm")
        assert figure.bones["hip"].parent is None and figure.bones["l_forearm"].parent == "l_upperarm"
        assert figure.bones["l_forearm"].limits == {"z": (0.0, 0.0)}
        assert figure.bones["l_upperarm"].rotation_order == "XYZ"


def test_the_rest_pose_is_all_zero_channels(tmp_path):
    figure = daz_pose.load_daz_figure(_figure_file(tmp_path))
    channels = daz_pose.daz_rotation_channels({}, figure)
    # The torso keeps its rest; limbs are aimed at PoseCraft's T-pose rest (arms out, legs straight
    # down), so even the rest pose lifts Genesis 9's A-pose arms and straightens its splayed legs.
    for name in ("hip", "pelvis", "spine1", "l_shoulder"):
        assert channels[name] == pytest.approx((0, 0, 0), abs=1e-6), name
    world = daz_pose.posed_world_rotations({}, figure)
    arm = world["l_upperarm"].apply(figure.bones["l_upperarm"].end - figure.bones["l_upperarm"].origin)
    assert np.allclose(arm / np.linalg.norm(arm), [1, 0, 0], atol=1e-6)  # aimed along PoseCraft's T-pose arm


def test_channels_reproduce_the_posed_world_rotations(tmp_path):
    figure = daz_pose.load_daz_figure(_figure_file(tmp_path))
    pose = {
        "spine": Rotation.from_euler("x", 12, degrees=True),
        "head": Rotation.from_euler("y", 20, degrees=True),
        "upper_arm.L": Rotation.from_euler("yz", [-40, -20], degrees=True),
        "forearm.L": Rotation.from_euler("y", -50, degrees=True),
        "thigh.L": Rotation.from_euler("x", -35, degrees=True),
        "shin.L": Rotation.from_euler("x", 45, degrees=True),
    }
    solution = daz_pose.solve_daz_pose(pose, figure)
    channels = solution.channels
    for name, rotation in _reconstruct_world(channels, figure).items():
        assert np.degrees((rotation.inv() * solution.world[name]).magnitude()) < 1e-6, name
    expected = daz_pose.posed_world_rotations(pose, figure)  # aimed targets (no locked axis in these bones)
    # Each aimed bone points where PoseCraft's bone points.
    pc_world = sk.world_rotations_from_local(pose)
    for pc_name, daz_name in (("thigh.L", "l_thigh"), ("shin.L", "l_shin"), ("head", "head")):
        bone = figure.bones[daz_name]
        direction = expected[daz_name].apply(bone.end - bone.origin)
        target = pc_world[pc_name].apply(sk.bone_rest_direction(pc_name))
        assert np.allclose(direction / np.linalg.norm(direction), target, atol=1e-6)
    assert channels["pelvis"] == (0.0, 0.0, 0.0) and channels["spine1"] == (0.0, 0.0, 0.0)


def test_the_document_is_a_pose_preset_setting_every_bone(tmp_path):
    figure = daz_pose.load_daz_figure(_figure_file(tmp_path))
    document = daz_pose.build_daz_pose_document({"head": Rotation.from_euler("y", 15, degrees=True)}, figure,
                                                hip_height=0.8, leg_length=0.95, name="My pose")
    assert document["asset_info"]["type"] == "preset_pose" and document["file_version"] == "0.6.0.0"
    urls = [a["url"] for a in document["scene"]["animations"]]
    assert len(urls) == 3 * len(figure.bones) + 1
    assert "name://@selection/head:?rotation/y/value" in urls
    hip = next(a for a in document["scene"]["animations"] if a["url"].endswith("hip:?translation/y/value"))
    legs = sum(np.linalg.norm(figure.bones[n].end - figure.bones[n].origin) for n in ("l_thigh", "l_shin"))
    assert hip["keys"][0][1] == pytest.approx(0.8 / 0.95 * legs - 97.1, abs=1e-3)


def test_channels_past_the_joint_limits_are_reported(tmp_path):
    figure = daz_pose.load_daz_figure(_figure_file(tmp_path))
    # The upper arm's Z range is -40..90 in the stand-in figure: swinging it far forward overshoots.
    past = daz_pose.channels_past_limits({"upper_arm.L": Rotation.from_euler("y", -100, degrees=True)}, figure)
    assert any(bone == "l_upperarm" for bone, _axis, _value, _limit in past)
    # The forearm's locked Z axis is part of the rig, not a pose problem: never reported.
    assert not any(bone == "l_forearm" for bone, _axis, _value, _limit in
                   daz_pose.channels_past_limits({"forearm.L": Rotation.from_euler("y", -60, degrees=True)}, figure))
    solution = daz_pose.solve_daz_pose({"forearm.L": Rotation.from_euler("y", -60, degrees=True)}, figure)
    assert solution.channels["l_forearm"][2] == 0.0  # clamped to its 0..0 Z, as DAZ will


def test_a_missing_genesis9_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(daz_pose.LIBRARY_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(daz_pose.DazFigureNotFound, match="Genesis 9"):
        daz_pose.find_genesis9_figure([tmp_path / "nowhere"])
    library = tmp_path / "lib"
    target = library / daz_pose.GENESIS9_DSF
    target.parent.mkdir(parents=True)
    target.write_bytes(_figure_file(tmp_path).read_bytes())
    assert daz_pose.find_genesis9_figure([tmp_path / "nowhere", library]) == target


def test_batch_writes_duf_presets(tmp_path, monkeypatch):
    from pose_craft import batch

    assert "duf" in batch.FORMATS
    figure_path = _figure_file(tmp_path)
    monkeypatch.setattr(daz_pose, "find_genesis9_figure", lambda libraries=None: figure_path)
    rotations = {"head": Rotation.from_euler("y", 10, degrees=True)}
    out = daz_pose.save_daz_pose_file(rotations, tmp_path / "p.duf", hip_height=0.9, leg_length=0.95)
    assert json.loads(out.read_text())["asset_info"]["type"] == "preset_pose"

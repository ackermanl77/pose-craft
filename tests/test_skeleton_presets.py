"""Skeleton presets for the BVH/glTF exports (B22): Mixamo, UE5 Mannequin and
HumanIK names, hierarchy and rest pose, with the posed joints where PoseCraft's
own skeleton puts them."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export.bvh_export import build_bvh_animation
from pose_craft.export.gltf_export import build_gltf_animation
from pose_craft.export.skeleton_presets import (
    PRESETS,
    SKELETON_CHOICES,
    convert_root_position,
    convert_rotations,
    posecraft_rig,
    preset_rig,
)
from pose_craft.retarget.skeleton import BONES, FINGER_BONES


def _world_heads(rig, rotations, root_position=(0.0, 0.0, 0.0)):
    """Forward kinematics the way BVH/glTF readers do it: each joint's head is
    its parent's head plus the parent's world rotation applied to its OFFSET."""
    world_rotation, heads = {}, {}
    for name in rig.order():
        parent = rig.parent[name]
        if parent is None:
            heads[name] = np.asarray(root_position, float)
            world_rotation[name] = rotations.get(name, Rotation.identity())
        else:
            heads[name] = heads[parent] + world_rotation[parent].apply(rig.offset[name])
            world_rotation[name] = world_rotation[parent] * rotations.get(name, Rotation.identity())
    return heads, world_rotation


def _random_pose(seed: int, with_fingers: bool) -> dict:
    rng = np.random.default_rng(seed)
    names = [bone.name for bone in BONES] + ([finger.name for finger in FINGER_BONES] if with_fingers else [])
    return {name: Rotation.from_rotvec(rng.normal(scale=0.5, size=3)) for name in names}


@pytest.mark.parametrize("key", list(PRESETS))
@pytest.mark.parametrize("with_fingers", [False, True])
def test_every_mapped_joint_lands_where_posecraft_puts_it(key, with_fingers):
    preset = PRESETS[key]
    source = posecraft_rig(with_fingers)
    rig = preset_rig(key, with_fingers)
    for seed in range(3):
        pose = _random_pose(seed, with_fingers)
        root = (12.0, 95.0, -4.0)
        expected_heads, expected_rotations = _world_heads(source, pose, root)
        heads, rotations = _world_heads(rig, convert_rotations(key, pose), convert_root_position(key, root))
        for bone in source.order():
            if bone == "spine":
                target = preset.hips
            elif bone == "neck":
                target = preset.neck_chain[0]
            else:
                target = preset.names[bone]
            assert np.allclose(heads[target], expected_heads[bone], atol=1e-6), (key, bone)
            # Each bone points the same way: its tip lands where PoseCraft's does.
            if bone not in ("spine", "neck"):
                assert np.allclose(rotations[target].apply(rig.tip[target]),
                                   expected_rotations[bone].apply(source.tip[bone]), atol=1e-6), (key, bone)


def test_the_presets_use_their_skeletons_names_and_rest_poses():
    mixamo = preset_rig("mixamo")
    assert mixamo.root == "mixamorig:Hips"
    assert mixamo.parent["mixamorig:LeftArm"] == "mixamorig:LeftShoulder"
    assert mixamo.parent["mixamorig:LeftShoulder"] == "mixamorig:Spine2"
    assert mixamo.parent["mixamorig:LeftUpLeg"] == "mixamorig:Hips"
    ue5 = preset_rig("ue5", with_fingers=True)
    assert ue5.root == "root" and ue5.parent["pelvis"] == "root"
    assert [ue5.parent[n] for n in ("spine_02", "neck_01", "head", "upperarm_l", "calf_r", "index_01_l")] == [
        "spine_01", "spine_05", "neck_02", "clavicle_l", "thigh_r", "hand_l"]
    # UE5 rests in an A-pose: the left forearm hangs 45 degrees below the shoulder.
    forearm = ue5.offset["lowerarm_l"]
    assert np.degrees(np.arctan2(-forearm[1], forearm[0])) == pytest.approx(45.0)
    humanik = preset_rig("humanik")
    assert humanik.root == "Hips" and "LeftForeArm" in humanik.parent
    assert [key for key, _label in SKELETON_CHOICES] == ["posecraft", "mixamo", "ue5", "humanik"]


def test_bvh_and_glb_exports_on_a_preset():
    pose = _random_pose(7, with_fingers=False)
    bvh = build_bvh_animation([pose, pose], root_positions=[(0, 95, 0), (1, 94, 0)], skeleton="ue5")
    assert bvh.startswith("HIERARCHY\nROOT root\n")
    assert "JOINT pelvis" in bvh and "JOINT clavicle_l" in bvh and "JOINT spine_05" in bvh
    motion = bvh.split("MOTION\n")[1].splitlines()
    channels = bvh.count("CHANNELS 3") * 3 + 6
    assert len(motion[2].split()) == channels
    glb = build_gltf_animation([pose], root_positions=[(0, 95, 0)], skeleton="mixamo")
    length = struct.unpack("<I", glb[12:16])[0]
    document = json.loads(glb[20:20 + length])
    names = [node["name"] for node in document["nodes"]]
    assert "mixamorig:Hips" in names and "mixamorig:LeftHand" in names
    assert names[document["skins"][0]["skeleton"]] == "mixamorig:Hips"
    with pytest.raises(ValueError):
        build_bvh_animation([pose], skeleton="nope")

"""glTF (.glb) export (F11). Checked separately with the Khronos glTF-Validator 2.0.0-dev.3.10:
0 errors, 0 warnings, 0 infos for an animated file and a pose with fingers."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export.gltf_export import CM, build_gltf_animation, build_gltf_pose
from pose_craft.retarget.skeleton import BONES, FINGER_BONES, forward_kinematics


def _parse(glb: bytes):
    magic, version, total = struct.unpack_from("<4sII", glb, 0)
    assert (magic, version, total) == (b"glTF", 2, len(glb))
    json_len, json_type = struct.unpack_from("<I4s", glb, 12)
    assert json_type == b"JSON"
    doc = json.loads(glb[20:20 + json_len])
    bin_len, bin_type = struct.unpack_from("<I4s", glb, 20 + json_len)
    assert bin_type == b"BIN\x00" and bin_len == doc["buffers"][0]["byteLength"]
    blob = glb[28 + json_len:28 + json_len + bin_len]

    def accessor(i):
        a = doc["accessors"][i]
        view = doc["bufferViews"][a["bufferView"]]
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        dtype = {5126: np.float32, 5123: np.uint16}[a["componentType"]]
        data = np.frombuffer(blob, dtype, a["count"] * width, view["byteOffset"])
        return data.reshape(a["count"], width) if width > 1 else data

    return doc, accessor


def _frames(n=6, fingers=False):
    names = [b.name for b in BONES] + ([f.name for f in FINGER_BONES] if fingers else [])
    rng = np.random.default_rng(4)
    return [{name: Rotation.from_rotvec(rng.normal(0, 0.4, 3)) for name in names} for _ in range(n)]


def _world_positions(doc, accessor, frame):
    """Forward kinematics over the glTF nodes, with frame ``frame`` of the animation applied."""
    nodes = doc["nodes"]
    rotation = {}
    translation = {}
    for channel in doc["animations"][0]["channels"]:
        values = accessor(doc["animations"][0]["samplers"][channel["sampler"]]["output"])[frame]
        (rotation if channel["target"]["path"] == "rotation" else translation)[channel["target"]["node"]] = values
    out = {}

    def visit(i, parent_rot, parent_pos):
        node = nodes[i]
        local_t = np.array(translation.get(i, node["translation"]), float)
        rot = parent_rot * Rotation.from_quat(rotation.get(i, node["rotation"]))
        pos = parent_pos + parent_rot.apply(local_t)
        out[node["name"]] = pos
        for child in node.get("children", []):
            visit(child, rot, pos)

    visit(doc["skins"][0]["skeleton"], Rotation.identity(), np.zeros(3))
    return out


@pytest.mark.parametrize("fingers", [False, True])
def test_the_glb_skeleton_poses_exactly_like_posecraft_own_forward_kinematics(fingers):
    frames = _frames(fingers=fingers)
    roots = [(5.0 * i, 90.0, -3.0) for i in range(len(frames))]
    doc, accessor = _parse(build_gltf_animation(frames, frame_time=0.04, root_positions=roots))
    assert len(doc["skins"][0]["joints"]) == len(BONES) + (len(FINGER_BONES) if fingers else 0)
    for frame in (0, 3, 5):
        glb = _world_positions(doc, accessor, frame)
        expected = forward_kinematics(frames[frame], root_position=np.array(roots[frame]))
        for name, position in expected.items():
            if name in glb:
                assert np.allclose(glb[name], np.asarray(position) * CM, atol=1e-5), (frame, name)
    times = accessor(doc["animations"][0]["samplers"][0]["input"])
    assert np.allclose(times, np.arange(len(frames)) * 0.04)


def test_quaternions_stay_in_one_hemisphere_and_the_skin_binds_the_rest_pose():
    flip = Rotation.from_quat([0, 0, 0.0, 1])
    frames = [{BONES[1].name: flip}, {BONES[1].name: Rotation.from_quat([0, 0, 0.01, -1])}]  # same rotation, flipped sign
    doc, accessor = _parse(build_gltf_animation(frames))
    joint = [n["name"] for n in doc["nodes"]].index(BONES[1].name)
    channel = next(c for c in doc["animations"][0]["channels"] if c["target"] == {"node": joint, "path": "rotation"})
    quats = accessor(doc["animations"][0]["samplers"][channel["sampler"]]["output"])
    assert np.dot(quats[0], quats[1]) > 0  # the short way round
    inverse = accessor(doc["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    assert np.allclose(inverse[:, :3, :3], np.eye(3))  # rest pose: translations only
    mesh = doc["meshes"][0]["primitives"][0]
    assert accessor(mesh["attributes"]["WEIGHTS_0"])[:, 0].min() == 1.0


def test_a_single_pose_and_errors():
    doc, _accessor = _parse(build_gltf_pose(_frames(1)[0], root_position=(0, 92, 0), name="Pose"))
    assert doc["animations"][0]["name"] == "Pose" and doc["scenes"][0]["name"] == "Pose"
    with pytest.raises(ValueError):
        build_gltf_animation([])
    with pytest.raises(ValueError):
        build_gltf_animation(_frames(2), root_positions=[(0, 0, 0)])

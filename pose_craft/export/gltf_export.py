"""Writes a binary glTF 2.0 (.glb) skinned skeleton with the pose or animation (F11): the open
format game engines (Unity, Unreal, Godot), three.js / Babylon.js and Blender all import, beside
the existing BVH, Blender JSON/Live Link and DAZ paths.

It is the BVH exporter's skeleton in a different container, so nothing is re-derived:

- Frame: glTF is Y-up, right-handed, with the asset facing +Z -- exactly the engine frame
  ``retarget/skeleton.py`` solves in (and the BVH exporter's own convention), so rotations and
  rest offsets are copied unchanged. Units are metres (the skeleton is in centimetres).
- Joints: one node per bone, translation = its rest head relative to its parent's (the BVH
  OFFSET), rotation = the solved parent-relative rotation; fingers when they were solved.
- Skin: the joints plus inverse bind matrices of the rest (T) pose, and a small "bone stick" mesh
  -- one thin box per bone, fully weighted to it -- so importers build a real armature and show
  something. Animation: a rotation channel per joint and a translation channel on the root (the
  floor-standing root positions), LINEAR, one key per frame.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.export.bvh_export import (
    _CHILDREN,
    _CHILDREN_WITH_FINGERS,
    _ROOT_BONE,
    _bone_vector,
    _channel_order,
    _offset,
)
from pose_craft.retarget.skeleton import FINGER_BONES, rest_head_positions

CM = 0.01  # skeleton centimetres -> glTF metres
STICK_HALF_WIDTH_M = 0.012
_FLOAT, _USHORT, _UINT = 5126, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


class _Bin:
    """The GLB's binary chunk: buffer views and accessors appended in order."""

    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, component: int, kind: str, target: int | None = None, minmax: bool = False) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        raw = np.ascontiguousarray(array).tobytes()
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        self.data.extend(raw)
        count = int(array.shape[0])
        accessor = {"bufferView": len(self.views) - 1, "componentType": component, "count": count, "type": kind}
        if minmax:
            flat = array.reshape(count, -1)
            accessor["min"] = [float(v) for v in flat.min(axis=0)]
            accessor["max"] = [float(v) for v in flat.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _stick(head: np.ndarray, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """8 vertices / 12 triangles of a thin box from ``head`` along ``vector`` (metres)."""
    length = float(np.linalg.norm(vector)) or 1e-3
    axis = vector / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    corners = []
    for end in (head, head + vector):
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corners.append(end + STICK_HALF_WIDTH_M * (su * u + sv * v))
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array(corners, np.float32), np.array(faces, np.uint16)


def build_gltf_animation(
    frames: list[dict[str, Rotation]],
    *,
    frame_time: float = 1.0 / 30.0,
    root_positions: list[tuple[float, float, float]] | None = None,
    name: str = "PoseCraft",
) -> bytes:
    """A .glb of ``frames`` (bone name -> local Rotation, engine frame; missing bones identity) --
    the same input as ``bvh_export.build_bvh_animation``. One frame makes a static pose (still
    written as a one-key animation, so importers apply it)."""
    if not frames:
        raise ValueError("a glTF animation needs at least one frame")
    if root_positions is not None and len(root_positions) != len(frames):
        raise ValueError(f"{len(root_positions)} root positions for {len(frames)} frames")
    finger_names = {finger.name for finger in FINGER_BONES}
    with_fingers = any(finger_names & set(rotations) for rotations in frames)
    children_of = _CHILDREN_WITH_FINGERS if with_fingers else _CHILDREN
    joints = _channel_order(children_of)
    index = {joint: i for i, joint in enumerate(joints)}
    roots = [tuple(p) for p in (root_positions or [(0.0, 0.0, 0.0)] * len(frames))]
    heads = rest_head_positions()

    nodes: list[dict] = []
    for joint in joints:
        translation = (np.array(roots[0]) if joint == _ROOT_BONE else _offset(joint)) * CM
        node = {"name": joint, "translation": [float(v) for v in translation],
                "rotation": [float(v) for v in frames[0].get(joint, Rotation.identity()).as_quat()]}
        kids = [index[child] for child in children_of[joint]]
        if kids:
            node["children"] = kids
        nodes.append(node)

    binary = _Bin()
    # The bone-stick mesh, bound to the rest pose.
    positions: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    joint_ids: list[np.ndarray] = []
    for joint in joints:
        head = (heads[joint] if joint in heads else np.zeros(3)) * CM
        corners, faces = _stick(head, _bone_vector(joint) * CM)
        indices.append(faces + len(positions) * 8)
        positions.append(corners)
        joint_ids.append(np.full((8, 4), [index[joint], 0, 0, 0], np.uint16))
    position_accessor = binary.add(np.concatenate(positions), _FLOAT, "VEC3", _ARRAY_BUFFER, minmax=True)
    joints_accessor = binary.add(np.concatenate(joint_ids), _USHORT, "VEC4", _ARRAY_BUFFER)
    weights = np.zeros((8 * len(joints), 4), np.float32)
    weights[:, 0] = 1.0
    weights_accessor = binary.add(weights, _FLOAT, "VEC4", _ARRAY_BUFFER)
    index_accessor = binary.add(np.concatenate(indices).astype(np.uint16).reshape(-1), _USHORT, "SCALAR", _ELEMENT_ARRAY_BUFFER)
    inverse_binds = []
    for joint in joints:
        head = (heads[joint] if joint in heads else np.zeros(3)) * CM
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = -head  # the rest pose has no rotations: the inverse bind is a translation
        inverse_binds.append(matrix.T.reshape(-1))  # glTF matrices are column-major
    inverse_bind_accessor = binary.add(np.array(inverse_binds, np.float32), _FLOAT, "MAT4")

    # The animation.
    times = binary.add(np.arange(len(frames), dtype=np.float32) * np.float32(frame_time), _FLOAT, "SCALAR", minmax=True)
    samplers, channels = [], []
    for joint in joints:
        quats = np.array([frame.get(joint, Rotation.identity()).as_quat() for frame in frames], np.float32)
        # Keep consecutive quaternions in the same hemisphere so interpolation takes the short way.
        for i in range(1, len(quats)):
            if float(np.dot(quats[i], quats[i - 1])) < 0:
                quats[i] = -quats[i]
        samplers.append({"input": times, "output": binary.add(quats, _FLOAT, "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": index[joint], "path": "rotation"}})
    root_translations = np.array(roots, np.float32) * np.float32(CM)
    samplers.append({"input": times, "output": binary.add(root_translations, _FLOAT, "VEC3"), "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": index[_ROOT_BONE], "path": "translation"}})

    mesh_node = len(nodes)
    nodes.append({"name": f"{name} bones", "mesh": 0, "skin": 0})
    buffers: list[dict] = [{"byteLength": 0}]
    document = {
        "asset": {"version": "2.0", "generator": "PoseCraft"},
        "scene": 0,
        "scenes": [{"name": name, "nodes": [index[_ROOT_BONE], mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": f"{name} bones", "primitives": [{
            "attributes": {"POSITION": position_accessor, "JOINTS_0": joints_accessor, "WEIGHTS_0": weights_accessor},
            "indices": index_accessor, "mode": 4,
        }]}],
        "skins": [{"name": name, "joints": list(range(len(joints))), "skeleton": index[_ROOT_BONE],
                   "inverseBindMatrices": inverse_bind_accessor}],
        "animations": [{"name": name, "samplers": samplers, "channels": channels}],
        "buffers": buffers,
        "bufferViews": binary.views,
        "accessors": binary.accessors,
    }
    while len(binary.data) % 4:
        binary.data.append(0)
    buffers[0]["byteLength"] = len(binary.data)
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(binary.data)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
            + struct.pack("<I4s", len(binary.data), b"BIN\x00") + bytes(binary.data))


def build_gltf_pose(bone_rotations: dict[str, Rotation], *, root_position=(0.0, 0.0, 0.0), name: str = "PoseCraft") -> bytes:
    return build_gltf_animation([bone_rotations], root_positions=[tuple(root_position)], name=name)


def save_gltf_animation(frames, path: str | Path, *, frame_time: float = 1.0 / 30.0, root_positions=None) -> None:
    Path(path).write_bytes(build_gltf_animation(frames, frame_time=frame_time, root_positions=root_positions,
                                                name=Path(path).stem))


def save_gltf_pose(bone_rotations, path: str | Path, *, root_position=(0.0, 0.0, 0.0)) -> None:
    Path(path).write_bytes(build_gltf_pose(bone_rotations, root_position=root_position, name=Path(path).stem))

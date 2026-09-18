"""Serializes solved bone rotations (engine frame: Y-up) into a JSON pose file
using Blender's axis convention (Z-up), plus the companion add-on in
blender_addon/import_pose.py that applies that JSON onto an armature's pose bones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FORMAT_NAME = "pose_craft_pose"
FORMAT_VERSION = 1

# Engine (Y-up, X-right, Z-toward-camera) -> Blender (Z-up, X-right, Y-into-scene).
# A pure 90-degree rotation about X, so quaternions convert by conjugation, not just relabeling.
_ENGINE_TO_BLENDER_AXES = Rotation.from_matrix([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def _rotation_to_blender(rotation: Rotation) -> Rotation:
    return _ENGINE_TO_BLENDER_AXES * rotation * _ENGINE_TO_BLENDER_AXES.inv()


def build_pose_document(bone_rotations: dict[str, Rotation], *, source_note: str = "") -> dict:
    """bone_rotations: bone name -> local (parent-relative) Rotation, engine frame."""
    bones = {}
    for name, rotation in bone_rotations.items():
        blender_rotation = _rotation_to_blender(rotation)
        x, y, z, w = blender_rotation.as_quat()  # scipy returns (x, y, z, w)
        bones[name] = {"quaternion_wxyz": [w, x, y, z]}

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "up_axis": "Z",
        "source_note": source_note,
        "bones": bones,
    }


def save_pose_file(bone_rotations: dict[str, Rotation], path: str | Path, *, source_note: str = "") -> None:
    document = build_pose_document(bone_rotations, source_note=source_note)
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_pose_document(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != FORMAT_NAME:
        raise ValueError(f"not a {FORMAT_NAME} file: {path}")
    return document

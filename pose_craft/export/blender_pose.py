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

# Below this MediaPipe visibility score, a bone is flagged "low_confidence" in the
# exported file -- a hint (surfaced by the Blender add-on) that this joint was
# likely occluded/hidden in the source photo and may need manual correction,
# the same category of case PoseForge's own docs call out.
LOW_CONFIDENCE_THRESHOLD = 0.5

# Engine (Y-up, +X = subject's left, +Z = the way the subject faces, i.e. toward
# the camera) -> Blender's rig convention (Z-up, .L bones on +X, the character
# facing -Y): engine +X -> Blender +X, +Y -> +Z, +Z -> -Y. A pure 90-degree
# rotation about X (both frames right-handed), so quaternions convert by
# conjugation, not just relabeling.
_ENGINE_TO_BLENDER_AXES = Rotation.from_matrix([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def _rotation_to_blender(rotation: Rotation) -> Rotation:
    return _ENGINE_TO_BLENDER_AXES * rotation * _ENGINE_TO_BLENDER_AXES.inv()


def build_pose_document(
    bone_rotations: dict[str, Rotation],
    *,
    source_note: str = "",
    bone_confidence: dict[str, float] | None = None,
) -> dict:
    """bone_rotations: bone name -> local (parent-relative) Rotation, engine frame.

    bone_confidence: optional bone name -> MediaPipe visibility score (see
    retarget.skeleton.bone_confidence), used to add "confidence"/"low_confidence"
    per bone. Bones missing from this dict (or when it's omitted entirely) simply
    don't get those keys -- callers that don't have visibility data are unaffected.
    """
    bones = {}
    for name, rotation in bone_rotations.items():
        blender_rotation = _rotation_to_blender(rotation)
        x, y, z, w = blender_rotation.as_quat()  # scipy returns (x, y, z, w)
        entry = {"quaternion_wxyz": [w, x, y, z]}
        if bone_confidence is not None and name in bone_confidence:
            confidence = bone_confidence[name]
            entry["confidence"] = confidence
            entry["low_confidence"] = confidence < LOW_CONFIDENCE_THRESHOLD
        bones[name] = entry

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "up_axis": "Z",
        "source_note": source_note,
        "bones": bones,
    }


def save_pose_file(
    bone_rotations: dict[str, Rotation],
    path: str | Path,
    *,
    source_note: str = "",
    bone_confidence: dict[str, float] | None = None,
) -> None:
    document = build_pose_document(bone_rotations, source_note=source_note, bone_confidence=bone_confidence)
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_pose_document(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != FORMAT_NAME:
        raise ValueError(f"not a {FORMAT_NAME} file: {path}")
    return document

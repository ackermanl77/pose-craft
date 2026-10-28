"""Serializes solved bone rotations (engine frame: Y-up) into a JSON pose file
using Blender's axis convention (Z-up), plus the companion add-on in
blender_addon/import_pose.py that applies that JSON onto an armature's pose bones.

Per bone, format version 2 writes:
- "quaternion_wxyz": the parent-relative (local) rotation, as in version 1.
- "world_quaternion_wxyz": the bone's armature-space rotation relative to
  PoseCraft's T-pose rest skeleton (its parents' rotations composed in).
- "rest_direction": the bone's rest (head -> tail) direction in that skeleton.
The last two let the add-on pose a rig whose rest pose or hierarchy differs
from PoseCraft's (an A-pose, a hips bone above the thighs, clavicles): it aims
each rig bone at world_rotation * rest_direction, measured against the rig's
own rest direction and the actual posed parent -- see the add-on's
rig_pose_basis(). Version 1 readers only need "quaternion_wxyz", but reject
the version number, which is the intended "update the add-on" signal.

Optionally (still version 2; readers ignore unknown keys) a top-level "root":
{"hip_height": m, "leg_length": m} -- how high the hip center sits above the
floor in this pose, and the subject's thigh + shin length, both in the
subject's own meters. hip_height / leg_length is the rig-independent ratio a
reader scales by its own leg length to place the hips (a crouch is well under
1). The add-on's "Keep feet on the floor" option instead measures the floor
on the rig itself.

Face (F12, only when the app's "Face expressions" option found one): a top-level
"face": {"blendshapes": {name: 0..1}} with ARKit's blendshape names (jawOpen,
mouthSmileLeft, eyeBlinkRight ...), which the add-on sets on the rig's matching
shape keys. Still version 2: an older add-on ignores it.

Finger bones (B11, only when the app's "Hands and fingers" option solved
them): the same three keys per bone, named as in Rigify (f_index.01.L ...
thumb.03.R, see skeleton.FINGER_BONE_NAMES). Still version 2: a body-only
document is unchanged, and an older add-on reports the finger bones as
unmatched rather than failing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.retarget.skeleton import (
    BONE_LENGTHS_CM,
    bone_rest_direction,
    hip_height_above_floor,
    measured_proportions,
    world_rotations_from_local,
)

FORMAT_NAME = "pose_craft_pose"
FORMAT_VERSION = 2
# A whole tracked video person sent over Live Link (build_animation_document);
# mirrored by the add-on's ANIMATION_FORMAT_NAME / MAX_SUPPORTED_ANIMATION_VERSION.
ANIMATION_FORMAT_NAME = "pose_craft_animation"
ANIMATION_FORMAT_VERSION = 1

# Below this MediaPipe visibility score, a bone is flagged "low_confidence" in the
# exported file -- a hint (surfaced by the Blender add-on) that this joint was
# likely occluded/hidden in the source photo and may need manual correction,
# the same category of case PoseForge's own docs call out.
LOW_CONFIDENCE_THRESHOLD = 0.5
_low_confidence_threshold = LOW_CONFIDENCE_THRESHOLD


def low_confidence_threshold() -> float:
    """The visibility below which a bone counts as low-confidence -- in the export,
    the photo overlay, the 3D view and the person list alike. Defaults to
    LOW_CONFIDENCE_THRESHOLD; the GUI's Advanced settings change it (B10)."""
    return _low_confidence_threshold


def set_low_confidence_threshold(value: float) -> None:
    global _low_confidence_threshold
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"the low-confidence threshold must be within 0-1, got {value}")
    _low_confidence_threshold = float(value)

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


def _wxyz(rotation: Rotation) -> list[float]:
    x, y, z, w = rotation.as_quat()  # scipy returns (x, y, z, w)
    return [float(w), float(x), float(y), float(z)]


def build_pose_document(
    bone_rotations: dict[str, Rotation],
    *,
    source_note: str = "",
    bone_confidence: dict[str, float] | None = None,
    hip_height: float | None = None,
    leg_length: float | None = None,
    out_of_frame: set[str] | None = None,
    face_blendshapes: dict[str, float] | None = None,
) -> dict:
    """bone_rotations: bone name -> local (parent-relative) Rotation, engine frame.

    face_blendshapes: optional ARKit-named expression weights (F12), written as "face".

    out_of_frame: bones depending on a landmark outside the photo (see
    retarget.skeleton.bones_out_of_frame), written as "out_of_frame": true (B10)
    -- a guess for a cut-off limb, distinct from "low_confidence" (hidden).

    hip_height / leg_length: optional, written as the "root" entry (see the
    module docstring); see root_measurements() for computing them.

    bone_confidence: optional bone name -> MediaPipe visibility score (see
    retarget.skeleton.bone_confidence), used to add "confidence"/"low_confidence"
    per bone. Bones missing from this dict (or when it's omitted entirely) simply
    don't get those keys -- callers that don't have visibility data are unaffected.
    """
    world_rotations = world_rotations_from_local(bone_rotations)
    bones = {}
    for name, rotation in bone_rotations.items():
        entry: dict[str, Any] = {"quaternion_wxyz": _wxyz(_rotation_to_blender(rotation))}
        if name in world_rotations:
            entry["world_quaternion_wxyz"] = _wxyz(_rotation_to_blender(world_rotations[name]))
            entry["rest_direction"] = [float(v) for v in _ENGINE_TO_BLENDER_AXES.apply(bone_rest_direction(name))]
        if bone_confidence is not None and name in bone_confidence:
            confidence = bone_confidence[name]
            entry["confidence"] = confidence
            entry["low_confidence"] = confidence < low_confidence_threshold()
        if out_of_frame and name in out_of_frame:
            entry["out_of_frame"] = True
        bones[name] = entry

    document: dict[str, Any] = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "up_axis": "Z",
        "source_note": source_note,
        "bones": bones,
    }
    if hip_height is not None:
        document["root"] = {"hip_height": float(hip_height)}
        if leg_length is not None:
            document["root"]["leg_length"] = float(leg_length)
    if face_blendshapes:
        document["face"] = {"blendshapes": {name: round(float(value), 4) for name, value in face_blendshapes.items()}}
    return document


def mirror_blendshapes(blendshapes: dict[str, float] | None) -> dict[str, float] | None:
    """The expression of the mirrored pose (F12): ...Left and ...Right weights swap."""
    if not blendshapes:
        return blendshapes

    def mirrored(name: str) -> str:
        if name.endswith("Left"):
            return name[:-4] + "Right"
        if name.endswith("Right"):
            return name[:-5] + "Left"
        return name

    return {mirrored(name): value for name, value in blendshapes.items()}


def build_animation_document(
    frames: list[tuple[int, dict[str, Rotation]]],
    fps: float,
    *,
    source: str = "",
    root_positions_cm: list | None = None,
    faces: list | None = None,
) -> dict:
    """A tracked person's solved video frames for Live Link's "Send Animation
    to Blender" (add-on 0.7.0+): {"format": ANIMATION_FORMAT_NAME, "version":
    ANIMATION_FORMAT_VERSION, "fps", "source", "frames": [{"frame_index",
    "pose"}]}, each pose an ordinary build_pose_document() so the add-on
    retargets every frame exactly like a single pose. frames: (clip frame
    index, local rotations), indices increasing -- gaps are frames the
    tracker lost, which the add-on leaves unkeyed. live_link.send_animation_document
    splits it into chunks under the add-on's payload cap.

    root_positions_cm: the root position per frame (estimation.video.
    animation_root_positions, cm on the BVH rest skeleton), for root motion:
    each frame then carries "root_offset", its horizontal travel from the
    first frame in Blender axes, and the document "root_leg_length", that
    skeleton's leg length in the same units -- the add-on scales the travel by
    its rig's own leg length (add-on 0.8.0+; an older add-on ignores both).

    faces: optional expression weights per frame (F12; None where no face was found), carried
    in each frame's pose as "face" (add-on 0.10.0+ keys the rig's shape keys from it)."""
    indices = [int(index) for index, _rotations in frames]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("animation frame indices must increase")
    document: dict[str, Any] = {
        "format": ANIMATION_FORMAT_NAME,
        "version": ANIMATION_FORMAT_VERSION,
        "fps": float(fps),
        "source": source,
        "frames": [
            {"frame_index": index, "pose": build_pose_document(rotations, face_blendshapes=face)}
            for index, (_frame, rotations), face in zip(indices, frames, faces or [None] * len(frames))
        ],
    }
    if faces is not None and len(faces) != len(frames):
        raise ValueError("faces needs one entry per frame")
    if root_positions_cm is not None:
        if len(root_positions_cm) != len(frames):
            raise ValueError("root_positions_cm needs one position per frame")
        first = np.asarray(root_positions_cm[0], dtype=float)
        for entry, position in zip(document["frames"], root_positions_cm):
            travel = np.asarray(position, dtype=float) - first
            travel[1] = 0.0  # horizontal only (engine Y is up): "Keep feet on the floor" sets the height
            entry["root_offset"] = [float(v) for v in _ENGINE_TO_BLENDER_AXES.apply(travel)]
        document["root_leg_length"] = float(BONE_LENGTHS_CM["thigh.L"] + BONE_LENGTHS_CM["shin.L"])
    return document


def root_measurements(
    bone_rotations: dict[str, Rotation], landmarks: np.ndarray, hands: dict | None = None
) -> tuple[float, float]:
    """(hip_height, leg_length) for build_pose_document, in the landmarks' own
    units (meters for BlazePose's world landmarks): the posed skeleton built
    with the subject's measured proportions, its hip center's height above its
    lowest joint, and the mean thigh + shin length. hands: the hand landmarks
    the rotations were solved with, if any (fingers can touch the floor)."""
    lengths, shoulder_half, hip_half = measured_proportions(landmarks, hands)
    hip_height = hip_height_above_floor(bone_rotations, lengths, shoulder_half, hip_half)
    leg_length = (lengths["thigh.L"] + lengths["shin.L"] + lengths["thigh.R"] + lengths["shin.R"]) / 2.0
    return hip_height, leg_length


def save_pose_file(
    bone_rotations: dict[str, Rotation],
    path: str | Path,
    *,
    source_note: str = "",
    bone_confidence: dict[str, float] | None = None,
    hip_height: float | None = None,
    leg_length: float | None = None,
    out_of_frame: set[str] | None = None,
    face_blendshapes: dict[str, float] | None = None,
) -> None:
    document = build_pose_document(
        bone_rotations,
        source_note=source_note,
        bone_confidence=bone_confidence,
        hip_height=hip_height,
        leg_length=leg_length,
        out_of_frame=out_of_frame,
        face_blendshapes=face_blendshapes,
    )
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_pose_document(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != FORMAT_NAME:
        raise ValueError(f"not a {FORMAT_NAME} file: {path}")
    return document

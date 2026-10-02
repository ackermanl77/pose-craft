"""Writes a standard BVH (Biovision Hierarchy) file from the same 15-bone
hierarchy, rest directions, and locally-solved (parent-relative) rotations
that retarget/skeleton.py and export/blender_pose.py already use -- a
standard-interchange alternative alongside the existing JSON/Live Link
Blender path, for any tool (motion-capture viewers, game engines, other DCC
apps) that reads BVH instead of PoseCraft's own JSON format. Additive: the
JSON/Live Link path is unchanged.

Axis convention: BVH has no single universal up-axis/forward convention
across tools, unlike Blender, which this project already targets via a
documented axis conversion (blender_pose.py's _ENGINE_TO_BLENDER_AXES). This
module picks Y-up, right-handed, +X = the subject's left (image right),
+Z = the way the subject faces (toward the camera) -- identical to the engine
frame retarget/skeleton.py already works in, and the usual "Y-up, character
faces +Z" layout of mocap BVH files -- as
*this exporter's own* BVH convention, and documents that choice here as the
one place to change it if a downstream BVH consumer expects a different one
(the same role _world_to_engine()'s docstring plays for the input
side). Because of that choice _ENGINE_TO_BVH_AXES is the identity rotation,
but it is still expressed as an explicit, named conversion (mirroring
blender_pose.py's _rotation_to_blender pattern) rather than just skipped, so
a future change to a different target convention only has to edit this one
constant plus the rest-skeleton offsets (skeleton.rest_head_positions) if that also needs
handedness flipped.

Rest skeleton: a JOINT's OFFSET is where its head sits relative to its
parent's head in the T-pose rest skeleton (BVH's own definition), NOT its own
bone vector -- so a chain child's OFFSET is its parent's bone vector, and the
shoulders/hips additionally sit SHOULDER_HALF_WIDTH_CM / HIP_HALF_WIDTH_CM to
either side of the spine's tail/head. A leaf joint's End Site OFFSET is its
own bone vector (where its tip is). With that layout, BVH forward kinematics
(rest offsets + the MOTION rotations) reproduces the solved skeleton's joint
positions for a subject with these proportions -- see
tests/test_axis_conventions.py.

Rotation channel order: BVH also has no single standard for the order its
three rotation channels are applied in. This module always writes
"Zrotation Xrotation Yrotation" and always converts each bone's rotation to
matching intrinsic Z, X, Y Euler angles (scipy's uppercase 'ZXY' convention,
which composes exactly the way BVH's channel order is meant to be read: first
rotate about local Z, then about the new local X, then about the newest local
Y), so the CHANNELS line and the MOTION values paired with it are always
self-consistent.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.retarget.skeleton import (  # noqa: F401 -- the size constants are re-exported
    BONE_LENGTHS_CM,
    BONES,
    HIP_HALF_WIDTH_CM,
    SHOULDER_HALF_WIDTH_CM,
    BoneSpec,
    hip_height_above_floor,
    rest_head_positions,
)

BVH_ROTATION_ORDER = "ZXY"  # intrinsic Z, X, Y -- see module docstring

# See module docstring: this exporter's own BVH frame is defined to be
# identical to the engine frame, so this conversion is the identity -- but
# named and applied the same way blender_pose.py's axis conversion is, so a
# future change to a different BVH convention is a one-line edit here.
_ENGINE_TO_BVH_AXES = Rotation.identity()


def _rotation_to_bvh(rotation: Rotation) -> Rotation:
    return _ENGINE_TO_BVH_AXES * rotation * _ENGINE_TO_BVH_AXES.inv()


# The rest skeleton's proportions (BONE_LENGTHS_CM, SHOULDER_HALF_WIDTH_CM,
# HIP_HALF_WIDTH_CM) and where the arms and legs attach live in
# retarget/skeleton.py, shared with its forward_kinematics() preview, and are
# re-exported here under the names this module has always used.
_BONES_BY_NAME: dict[str, BoneSpec] = {b.name: b for b in BONES}

_CHILDREN: dict[str, list[str]] = {b.name: [] for b in BONES}
for _bone in BONES:
    if _bone.parent is not None:
        _CHILDREN[_bone.parent].append(_bone.name)

_ROOT_BONE = next(b.name for b in BONES if b.parent is None)


def _bone_vector(bone_name: str) -> np.ndarray:
    """The bone's own rest vector (head -> tail): rest_direction scaled to its length."""
    bone = _BONES_BY_NAME[bone_name]
    direction = bone.rest_direction / np.linalg.norm(bone.rest_direction)
    return direction * BONE_LENGTHS_CM[bone_name]


_REST_HEADS = rest_head_positions()


def _offset(bone_name: str) -> np.ndarray:
    """BVH OFFSET for a JOINT/ROOT: its rest head relative to its parent's rest
    head (zero for the root, which is placed by the root position channels)."""
    parent = _BONES_BY_NAME[bone_name].parent
    if parent is None:
        return np.zeros(3)
    return _REST_HEADS[bone_name] - _REST_HEADS[parent]


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _write_joint(lines: list[str], name: str, indent: int, is_root: bool) -> None:
    pad = "  " * indent
    keyword = "ROOT" if is_root else "JOINT"
    lines.append(f"{pad}{keyword} {name}")
    lines.append(f"{pad}{{")
    offset = _offset(name)
    lines.append(f"{pad}  OFFSET {_fmt(offset[0])} {_fmt(offset[1])} {_fmt(offset[2])}")
    if is_root:
        lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")

    children = _CHILDREN[name]
    if children:
        for child in children:
            _write_joint(lines, child, indent + 1, is_root=False)
    else:
        # A leaf JOINT (hand/foot/head -- no further bone in our 15-bone set)
        # still needs a terminating End Site so BVH readers know where its own
        # tip is: the bone's own rest vector, relative to the joint's head.
        tip = _bone_vector(name)
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {_fmt(tip[0])} {_fmt(tip[1])} {_fmt(tip[2])}")
        lines.append(f"{pad}  }}")

    lines.append(f"{pad}}}")


def _channel_order() -> list[str]:
    """The joint traversal order MOTION values must be written in -- must match
    the HIERARCHY block's own depth-first traversal exactly, which it does by
    construction: both walk _CHILDREN starting from _ROOT_BONE."""
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for child in _CHILDREN[name]:
            visit(child)

    visit(_ROOT_BONE)
    return order


def floor_root_position(bone_rotations: dict[str, Rotation]) -> tuple[float, float, float]:
    """A ROOT position that stands the pose on the floor (Y = 0) with this
    file's rest skeleton: the hip center lifted by its height above the
    lowest joint (skeleton.hip_height_above_floor), so a crouch is exported
    lowered rather than with its hips at the origin."""
    return (0.0, hip_height_above_floor(bone_rotations), 0.0)


def _motion_line(bone_rotations: dict[str, Rotation], root_position, order: list[str]) -> str:
    values: list[str] = [_fmt(v) for v in root_position]
    for name in order:
        rotation = bone_rotations.get(name, Rotation.identity())
        bvh_rotation = _rotation_to_bvh(rotation)
        with warnings.catch_warnings():
            # At exactly +/-90 degrees about X (e.g. a thigh raised straight
            # forward) ZXY Euler angles hit gimbal lock; scipy warns and zeroes
            # the third angle, but the returned angles still compose back to the
            # exact same rotation, so the warning is just noise here.
            warnings.simplefilter("ignore", UserWarning)
            z, x, y = bvh_rotation.as_euler(BVH_ROTATION_ORDER, degrees=True)
        values.extend([_fmt(z), _fmt(x), _fmt(y)])
    return " ".join(values)


def build_bvh_animation(
    frames: list[dict[str, Rotation]],
    *,
    frame_time: float = 1.0 / 30.0,
    root_positions: list[tuple[float, float, float]] | None = None,
) -> str:
    """A multi-frame BVH: the same HIERARCHY as build_bvh_document, and one
    MOTION line per entry of `frames` (each a bone name -> local Rotation dict,
    missing bones identity). root_positions: one per frame (default: all at
    the origin)."""
    if not frames:
        raise ValueError("a BVH needs at least one frame")
    if root_positions is not None and len(root_positions) != len(frames):
        raise ValueError(f"{len(root_positions)} root positions for {len(frames)} frames")
    hierarchy_lines: list[str] = ["HIERARCHY"]
    _write_joint(hierarchy_lines, _ROOT_BONE, indent=0, is_root=True)

    order = _channel_order()
    positions = root_positions if root_positions is not None else [(0.0, 0.0, 0.0)] * len(frames)
    lines = hierarchy_lines + [
        "MOTION",
        f"Frames: {len(frames)}",
        f"Frame Time: {frame_time:.6f}",
    ]
    lines += [_motion_line(rotations, position, order) for rotations, position in zip(frames, positions)]
    return "\n".join(lines) + "\n"


def build_bvh_document(
    bone_rotations: dict[str, Rotation],
    *,
    frame_time: float = 1.0 / 30.0,
    root_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """bone_rotations: bone name -> local (parent-relative) Rotation, engine
    frame -- the same dict solve_bone_rotations() returns and
    export.blender_pose.build_pose_document() also consumes. A bone missing
    from the dict defaults to identity, so a partial dict still produces a
    structurally valid file instead of a KeyError.

    Produces a single-frame (static pose) BVH: a HIERARCHY block for all 15
    bones plus their End Sites, and a MOTION block with exactly one frame --
    one set of Euler angles per joint, in the same traversal/channel order the
    HIERARCHY block declares. Pass root_position=floor_root_position(...) to
    stand the pose on the floor (the app does).
    """
    return build_bvh_animation([bone_rotations], frame_time=frame_time, root_positions=[root_position])


def save_bvh_file(
    bone_rotations: dict[str, Rotation],
    path: str | Path,
    *,
    frame_time: float = 1.0 / 30.0,
    root_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    Path(path).write_text(
        build_bvh_document(bone_rotations, frame_time=frame_time, root_position=root_position),
        encoding="utf-8",
    )


def save_bvh_animation(
    frames: list[dict[str, Rotation]],
    path: str | Path,
    *,
    frame_time: float = 1.0 / 30.0,
    root_positions: list[tuple[float, float, float]] | None = None,
) -> None:
    Path(path).write_text(
        build_bvh_animation(frames, frame_time=frame_time, root_positions=root_positions), encoding="utf-8"
    )

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
module picks Y-up, right-handed, +X-right, +Z-toward-camera/forward --
identical to the engine frame retarget/skeleton.py already works in -- as
*this exporter's own* BVH convention, and documents that choice here as the
one place to change it if a downstream BVH consumer expects a different one
(the same role _mediapipe_world_to_engine()'s docstring plays for the input
side). Because of that choice _ENGINE_TO_BVH_AXES is the identity rotation,
but it is still expressed as an explicit, named conversion (mirroring
blender_pose.py's _rotation_to_blender pattern) rather than just skipped, so
a future change to a different target convention only has to edit this one
constant plus its counterpart in _CHILDREN/BONE_LENGTHS_CM if that also needs
handedness flipped.

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

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.retarget.skeleton import BONES, BoneSpec

BVH_ROTATION_ORDER = "ZXY"  # intrinsic Z, X, Y -- see module docstring

# See module docstring: this exporter's own BVH frame is defined to be
# identical to the engine frame, so this conversion is the identity -- but
# named and applied the same way blender_pose.py's axis conversion is, so a
# future change to a different BVH convention is a one-line edit here.
_ENGINE_TO_BVH_AXES = Rotation.identity()


def _rotation_to_bvh(rotation: Rotation) -> Rotation:
    return _ENGINE_TO_BVH_AXES * rotation * _ENGINE_TO_BVH_AXES.inv()


# Typical adult body-segment lengths (an arbitrary but consistent unit,
# nominally centimeters), used only to give each BVH JOINT's static OFFSET a
# plausible non-zero length along its rest_direction. retarget/skeleton.py
# itself has no notion of bone length -- it only ever compares *directions*,
# since it solves rotations from whatever real landmark positions a specific
# photo produced -- so BVH's required OFFSET has to come from somewhere else.
# This is a "reasonable" generic proportion table, not a calibrated
# anthropometric one tied to any specific subject.
BONE_LENGTHS_CM: dict[str, float] = {
    "spine": 45.0,
    "neck": 10.0,
    "head": 20.0,
    "upper_arm.L": 30.0,
    "upper_arm.R": 30.0,
    "forearm.L": 25.0,
    "forearm.R": 25.0,
    "hand.L": 18.0,
    "hand.R": 18.0,
    "thigh.L": 45.0,
    "thigh.R": 45.0,
    "shin.L": 40.0,
    "shin.R": 40.0,
    "foot.L": 20.0,
    "foot.R": 20.0,
}

_BONES_BY_NAME: dict[str, BoneSpec] = {b.name: b for b in BONES}

_CHILDREN: dict[str, list[str]] = {b.name: [] for b in BONES}
for _bone in BONES:
    if _bone.parent is not None:
        _CHILDREN[_bone.parent].append(_bone.name)

_ROOT_BONE = next(b.name for b in BONES if b.parent is None)


def _offset(bone_name: str) -> np.ndarray:
    bone = _BONES_BY_NAME[bone_name]
    direction = bone.rest_direction / np.linalg.norm(bone.rest_direction)
    return direction * BONE_LENGTHS_CM[bone_name]


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
        # tip is, reusing the same rest_direction/length convention as OFFSET.
        lines.append(f"{pad}  End Site")
        lines.append(f"{pad}  {{")
        lines.append(f"{pad}    OFFSET {_fmt(offset[0])} {_fmt(offset[1])} {_fmt(offset[2])}")
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
    HIERARCHY block declares.
    """
    hierarchy_lines: list[str] = ["HIERARCHY"]
    _write_joint(hierarchy_lines, _ROOT_BONE, indent=0, is_root=True)

    order = _channel_order()
    motion_values: list[str] = [_fmt(v) for v in root_position]
    for name in order:
        rotation = bone_rotations.get(name, Rotation.identity())
        bvh_rotation = _rotation_to_bvh(rotation)
        z, x, y = bvh_rotation.as_euler(BVH_ROTATION_ORDER, degrees=True)
        motion_values.extend([_fmt(z), _fmt(x), _fmt(y)])

    lines = hierarchy_lines + [
        "MOTION",
        "Frames: 1",
        f"Frame Time: {frame_time:.6f}",
        " ".join(motion_values),
    ]
    return "\n".join(lines) + "\n"


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

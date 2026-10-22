"""Skeleton presets for the BVH and glTF exports (B22): the same pose on a
Mixamo, UE5 Mannequin or HumanIK skeleton -- their bone names, hierarchy and
rest pose -- so Unity, Unreal and MotionBuilder users can retarget without
going through Blender.

A :class:`Rig` is a skeleton the writers can emit: joints in depth-first
order, each with its parent, its rest OFFSET (head relative to the parent's
head, cm, engine frame: Y up, +X the subject's left, +Z forward) and its tip
vector (the bone stick / BVH End Site). ``posecraft_rig()`` is PoseCraft's own
15-bone (+ fingers) skeleton, exactly what the exporters always wrote.

A preset is built from it, keeping every joint where it was so the posed
skeleton matches PoseCraft's own (tests/test_skeleton_presets.py checks the
joint positions):

- **Hips:** a root joint at the hip centre carries the root translation and
  the torso rotation; the legs hang from it. PoseCraft's one spine bone becomes
  the preset's spine chain (Spine/Spine1/Spine2, spine_01..05), straight and
  unrotated, so every joint above lands where it was.
- **Neck:** the first neck bone gets the neck rotation, the others (UE5's
  neck_02) none.
- **Clavicles** (LeftShoulder, clavicle_l...): unrotated bones from near the
  spine's top to the shoulder joint.
- **A-pose rest** (UE5): the arm chains rest ``arm_down_deg`` below the
  horizontal. Every arm offset is turned by that rest rotation A, so the
  upper arm's rotation becomes r A^-1 and every bone below it A r A^-1 --
  the same posed arm on a different rest pose.
- **A floor root** (UE5's "root"): a joint at the floor below the pelvis that
  takes the horizontal motion; the pelvis sits at the standing hip height.

All presets stay Y-up in centimetres like PoseCraft's own files (FBX/glTF
importers convert to Unreal's Z-up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.retarget.skeleton import (
    BONE_LENGTHS_CM,
    BONES,
    FINGER_BONES,
    bone_length,
    bone_rest_direction,
    hip_height_above_floor,
    rest_head_positions,
)

DEFAULT_SKELETON = "posecraft"


@dataclass
class Rig:
    root: str
    parent: dict[str, str | None]
    children: dict[str, list[str]]
    offset: dict[str, np.ndarray]
    tip: dict[str, np.ndarray]

    def order(self) -> list[str]:
        """Depth-first joint order (the BVH HIERARCHY / MOTION order)."""
        order: list[str] = []

        def visit(name: str) -> None:
            order.append(name)
            for child in self.children[name]:
                visit(child)

        visit(self.root)
        return order

    def rest_heads(self) -> dict[str, np.ndarray]:
        heads: dict[str, np.ndarray] = {}
        for name in self.order():
            parent = self.parent[name]
            heads[name] = (heads[parent] if parent else np.zeros(3)) + self.offset[name]
        return heads


@dataclass(frozen=True)
class SkeletonPreset:
    key: str
    label: str
    names: dict  # PoseCraft bone -> preset bone (every core bone; fingers optional)
    hips: str
    spine_chain: tuple
    neck_chain: tuple
    clavicles: dict = field(default_factory=dict)  # "L"/"R" -> name
    arm_down_deg: float = 0.0
    floor_root: str | None = None


def _bone_vector(name: str) -> np.ndarray:
    direction = bone_rest_direction(name)
    return direction / np.linalg.norm(direction) * bone_length(BONE_LENGTHS_CM, name)


def posecraft_rig(with_fingers: bool = False) -> Rig:
    """PoseCraft's own skeleton, exactly as the exporters always wrote it."""
    heads = rest_head_positions()
    bones = list(BONES) + (list(FINGER_BONES) if with_fingers else [])
    parent = {bone.name: bone.parent for bone in bones}
    children: dict[str, list[str]] = {bone.name: [] for bone in bones}
    for bone in bones:
        if bone.parent is not None:
            children[bone.parent].append(bone.name)
    offset = {}
    for name, parent_name in parent.items():
        offset[name] = heads[name] - heads[parent_name] if parent_name else np.zeros(3)
    root = next(bone.name for bone in BONES if bone.parent is None)
    return Rig(root, parent, children, offset, {name: _bone_vector(name) for name in parent})


_SIDES = {"L": "Left", "R": "Right"}
_FINGERS_HUMANIK = {"thumb": "Thumb", "f_index": "Index", "f_middle": "Middle", "f_ring": "Ring", "f_pinky": "Pinky"}
_FINGERS_UE = {"thumb": "thumb", "f_index": "index", "f_middle": "middle", "f_ring": "ring", "f_pinky": "pinky"}


def _humanik_names(prefix: str = "") -> dict:
    names = {
        "spine": "Spine", "neck": "Neck", "head": "Head",
        "upper_arm.L": "LeftArm", "forearm.L": "LeftForeArm", "hand.L": "LeftHand",
        "upper_arm.R": "RightArm", "forearm.R": "RightForeArm", "hand.R": "RightHand",
        "thigh.L": "LeftUpLeg", "shin.L": "LeftLeg", "foot.L": "LeftFoot",
        "thigh.R": "RightUpLeg", "shin.R": "RightLeg", "foot.R": "RightFoot",
    }
    for side, word in _SIDES.items():
        for finger, finger_word in _FINGERS_HUMANIK.items():
            for segment in (1, 2, 3):
                names[f"{finger}.0{segment}.{side}"] = f"{word}Hand{finger_word}{segment}"
    return {key: prefix + value for key, value in names.items()}


def _ue5_names() -> dict:
    names = {"spine": "spine_01", "neck": "neck_01", "head": "head"}
    for side, suffix in (("L", "l"), ("R", "r")):
        names.update({
            f"upper_arm.{side}": f"upperarm_{suffix}", f"forearm.{side}": f"lowerarm_{suffix}", f"hand.{side}": f"hand_{suffix}",
            f"thigh.{side}": f"thigh_{suffix}", f"shin.{side}": f"calf_{suffix}", f"foot.{side}": f"foot_{suffix}",
        })
        for finger, finger_word in _FINGERS_UE.items():
            for segment in (1, 2, 3):
                names[f"{finger}.0{segment}.{side}"] = f"{finger_word}_0{segment}_{suffix}"
    return names


_MIXAMO = _humanik_names("mixamorig:")
PRESETS: dict[str, SkeletonPreset] = {
    "mixamo": SkeletonPreset(
        "mixamo", "Mixamo", _MIXAMO, hips="mixamorig:Hips",
        spine_chain=("mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2"), neck_chain=("mixamorig:Neck",),
        clavicles={"L": "mixamorig:LeftShoulder", "R": "mixamorig:RightShoulder"},
    ),
    "ue5": SkeletonPreset(
        "ue5", "Unreal Engine 5 Mannequin", _ue5_names(), hips="pelvis",
        spine_chain=("spine_01", "spine_02", "spine_03", "spine_04", "spine_05"), neck_chain=("neck_01", "neck_02"),
        clavicles={"L": "clavicle_l", "R": "clavicle_r"}, arm_down_deg=45.0, floor_root="root",
    ),
    "humanik": SkeletonPreset(
        "humanik", "HumanIK (MotionBuilder, Maya)", _humanik_names(), hips="Hips",
        spine_chain=("Spine", "Spine1", "Spine2"), neck_chain=("Neck",),
        clavicles={"L": "LeftShoulder", "R": "RightShoulder"},
    ),
}
SKELETON_CHOICES = [(DEFAULT_SKELETON, "PoseCraft (Rigify names)")] + [(key, preset.label) for key, preset in PRESETS.items()]

_CLAVICLE_START = 0.15  # where a clavicle starts, as a fraction of the way from the spine top to the shoulder
_UP = np.array([0.0, 1.0, 0.0])


def _arm_rest_rotation(side: str, degrees: float) -> Rotation:
    """A: the A-pose's turn of the arm chain about the forward (Z) axis -- the
    left arm (+X) down is a negative turn, the right arm (-X) a positive one."""
    return Rotation.from_rotvec([0.0, 0.0, (-1 if side == "L" else 1) * np.radians(degrees)])


def _arm_side(posecraft_name: str) -> str | None:
    """'L'/'R' for a bone in an arm chain (upper arm, forearm, hand, fingers)."""
    base, _, side = posecraft_name.rpartition(".")
    if side not in ("L", "R"):
        return None
    if base.split(".")[0] in ("upper_arm", "forearm", "hand", *_FINGERS_UE):
        return side
    return None


def preset_rig(key: str, with_fingers: bool = False) -> Rig:
    preset = PRESETS[key]
    source = posecraft_rig(with_fingers)
    heads = source.rest_heads()
    name = lambda bone: preset.names[bone]  # noqa: E731
    parent: dict[str, str | None] = {}
    children: dict[str, list[str]] = {}
    offset: dict[str, np.ndarray] = {}
    tip: dict[str, np.ndarray] = {}
    world_head: dict[str, np.ndarray] = {}

    def add(joint: str, parent_joint: str | None, head: np.ndarray, tip_vector: np.ndarray) -> None:
        parent[joint] = parent_joint
        children[joint] = []
        if parent_joint is not None:
            children[parent_joint].append(joint)
        world_head[joint] = head
        offset[joint] = head - world_head[parent_joint] if parent_joint is not None else head.copy()
        tip[joint] = tip_vector

    spine_root = source.root
    spine_vector = source.tip[spine_root]
    standing_hip_height = hip_height_above_floor({})
    if preset.floor_root:
        add(preset.floor_root, None, np.array([0.0, -standing_hip_height, 0.0]), np.array([0.0, 0.0, 10.0]))
    segments = len(preset.spine_chain) + 1  # the hips bone takes the first segment
    step = spine_vector / segments
    add(preset.hips, preset.floor_root, np.zeros(3), step)
    previous = preset.hips
    for index, spine_joint in enumerate(preset.spine_chain, start=1):
        add(spine_joint, previous, step * index, step)
        previous = spine_joint
    spine_top_joint, spine_top = previous, spine_vector

    neck_vector = source.tip["neck"]
    neck_step = neck_vector / len(preset.neck_chain)
    neck_previous = spine_top_joint
    for index, neck_joint in enumerate(preset.neck_chain):
        add(neck_joint, neck_previous, heads["neck"] + neck_step * index, neck_step)
        neck_previous = neck_joint

    arm_rest = {side: _arm_rest_rotation(side, preset.arm_down_deg) for side in ("L", "R")}

    def visit(bone: str, parent_joint: str) -> None:
        side = _arm_side(bone)
        if side is not None and preset.arm_down_deg:
            shoulder = heads[f"upper_arm.{side}"]
            head = shoulder + arm_rest[side].apply(heads[bone] - shoulder)
            tip_vector = arm_rest[side].apply(source.tip[bone])
        else:
            head, tip_vector = heads[bone], source.tip[bone]
        add(name(bone), parent_joint, head, tip_vector)
        for child in source.children[bone]:
            visit(child, name(bone))

    add(name("head"), preset.neck_chain[-1], heads["head"], source.tip["head"])
    for side in ("L", "R"):
        arm_parent = spine_top_joint
        if side in preset.clavicles:
            shoulder = heads[f"upper_arm.{side}"]
            start = spine_top + (shoulder - spine_top) * _CLAVICLE_START
            add(preset.clavicles[side], spine_top_joint, start, shoulder - start)
            arm_parent = preset.clavicles[side]
        visit(f"upper_arm.{side}", arm_parent)
        visit(f"thigh.{side}", preset.hips)
    return Rig(preset.floor_root or preset.hips, parent, children, offset, tip)


def convert_rotations(key: str, rotations: dict[str, Rotation]) -> dict[str, Rotation]:
    """PoseCraft's local rotations -> the preset's, for its rig (see the module docstring)."""
    preset = PRESETS[key]
    identity = Rotation.identity()
    result: dict[str, Rotation] = {preset.hips: rotations.get("spine", identity)}
    for joint in (*preset.spine_chain, *preset.neck_chain[1:], *preset.clavicles.values()):
        result[joint] = identity
    if preset.floor_root:
        result[preset.floor_root] = identity
    result[preset.neck_chain[0]] = rotations.get("neck", identity)
    arm_rest = {side: _arm_rest_rotation(side, preset.arm_down_deg) for side in ("L", "R")}
    for bone, target in preset.names.items():
        if bone in ("spine", "neck"):
            continue
        rotation = rotations.get(bone, identity)
        side = _arm_side(bone)
        if side is not None and preset.arm_down_deg:
            rest = arm_rest[side]
            rotation = rotation * rest.inv() if bone.startswith("upper_arm") else rest * rotation * rest.inv()
        result[target] = rotation
    return result


def convert_root_position(key: str, root_position) -> tuple[float, float, float]:
    """The root channel's translation: the hips' for most presets; with a floor
    root, the horizontal motion plus the drop below the standing hip height."""
    preset = PRESETS[key]
    x, y, z = (float(v) for v in root_position)
    if preset.floor_root:
        return (x, y - hip_height_above_floor({}), z)
    return (x, y, z)


def check_skeleton(skeleton: str) -> str:
    if skeleton != DEFAULT_SKELETON and skeleton not in PRESETS:
        raise ValueError(f"unknown skeleton {skeleton!r} (choose from {', '.join(k for k, _ in SKELETON_CHOICES)})")
    return skeleton


def rig_for(skeleton: str, with_fingers: bool) -> Rig:
    check_skeleton(skeleton)
    return posecraft_rig(with_fingers) if skeleton == DEFAULT_SKELETON else preset_rig(skeleton, with_fingers)


def frames_for(skeleton: str, frames: list[dict[str, Rotation]], root_positions):
    """(frames, root positions) on `skeleton`'s rig (unchanged for PoseCraft's own)."""
    if check_skeleton(skeleton) == DEFAULT_SKELETON:
        return frames, root_positions
    converted = [convert_rotations(skeleton, rotations) for rotations in frames]
    roots = [convert_root_position(skeleton, position) for position in root_positions]
    return converted, roots

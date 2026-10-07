"""Maps MediaPipe's 33 pose landmarks onto a generic humanoid bone hierarchy and
computes per-bone rotations by comparing each bone's current direction (from
landmark positions) against its rest-pose direction (a T-pose).

Coordinate convention used throughout this module (the "engine" frame):
right-handed, Y-up, +X toward image right, +Z toward the camera. For a subject
facing the camera that makes +X the subject's own LEFT and +Z the direction
they face -- hence LEFT_X = +X and FORWARD = +Z below, the same body-relative
layout as a Blender rig (.L bones on +X; see export/blender_pose.py).
Landmark-space vectors must already be in this frame before reaching
`solve_bone_rotations` -- see `estimation/landmarker.py` for the MediaPipe
world-landmark conversion.

Every bone gets a full frame, not just a direction: the direction a bone
points (head -> tail) plus a second, "secondary" axis that fixes its roll
about that direction. A direction alone leaves the roll free, and the old
shortest-arc swing picked an arbitrary one -- a whole body turned 60 degrees
solved to a root rotation of 0, a head turn to 0, and a bent elbow to a
twisted, off-hinge forearm. The secondary axis per bone (see
`_REST_SECONDARY` for each one's rest value):

- spine: the average of the hip line and the shoulder line (left minus right)
- head: the ear line; the neck takes NECK_YAW_SHARE of the head's twist
- upper arm / thigh: the elbow / knee bend plane's normal, blended toward the
  roll carried from the torso as the limb straightens (see FULL_BEND_DEG)
- forearm / shin: no roll of their own beyond the hinge, so the bend is a pure
  rotation about the hinge axis; the forearm also takes FOREARM_TWIST_SHARE of
  the hand's pronation
- hand: the index/pinky knuckle line
- foot: the sole (heel -> toe) and the ankle-heel-toe plane, falling back to
  the ankle -> toe line and FOOT_REST_PITCH_DEG when the heel is hidden

A bone whose secondary axis can't be measured (a straight arm, coincident
knuckles) keeps the roll carried from its parent with no twist of its own
(`_carried`), which depends only on the pose, never on the camera angle: a
rigid rotation of the whole body changes the root and nothing else. The
per-bone confidence score (see `bone_confidence`) is about landmark
visibility, not about how well a bone's roll was measured.

Fingers (B11, optional): with a hand's 21 MediaPipe hand landmarks (see
`FINGER_BONES`), the hand bone is measured from the palm (wrist -> middle
knuckle, rolled by the index-pinky knuckle line) and each hand gets 15 finger
bones in Rigify's names (thumb.01-03, f_index/f_middle/f_ring/f_pinky.01-03).
The first segment of each finger is framed by the knuckle line (the thumb by
the back-of-hand normal), the other two are pure hinges. Without hand
landmarks nothing changes: the 15 body bones, solved exactly as before.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# MediaPipe BlazePose 33-landmark indices.
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

NUM_LANDMARKS = 33


def _mid(landmarks: np.ndarray, a: int, b: int) -> np.ndarray:
    return (landmarks[a] + landmarks[b]) / 2.0


@dataclass(frozen=True)
class BoneSpec:
    name: str
    parent: str | None
    # Rest-pose direction (head -> tail) in the engine frame, T-pose, unit length.
    rest_direction: np.ndarray
    # Functions that derive the bone's head/tail joint positions from a (33, 3) landmark array.
    head_fn: Callable[[np.ndarray], np.ndarray]
    tail_fn: Callable[[np.ndarray], np.ndarray]
    # Raw landmark indices this bone's position depends on, for confidence scoring
    # (see bone_confidence) -- listed explicitly rather than derived from head_fn/tail_fn
    # since some of those (e.g. the head bone's extrapolated top-of-skull point) do
    # arithmetic that has no meaningful "visibility" of its own.
    visibility_landmarks: tuple[int, ...]


def _hip_center(lm: np.ndarray) -> np.ndarray:
    return _mid(lm, LEFT_HIP, RIGHT_HIP)


def _shoulder_center(lm: np.ndarray) -> np.ndarray:
    return _mid(lm, LEFT_SHOULDER, RIGHT_SHOULDER)


# Skull-top height above the ear midpoint, as a multiple of the ear-to-ear
# distance -- only sets the head bone's length, which the solve ignores.
_HEAD_TOP_HEIGHT_PER_EAR_WIDTH = 0.8


def _head_top(lm: np.ndarray) -> np.ndarray:
    """A head-top point along the head's own "up" axis, since MediaPipe has no
    skull landmark: perpendicular to both the ear-to-ear line and the face's
    forward direction (ear midpoint -> eye midpoint, roughly the Frankfort
    plane, which is level on a level head). The old version extrapolated back
    along the nose->ear line; the nose sits below and in front of the ears, so
    that tilted a level head ~20 degrees backward on export.
    """
    ear_center = _mid(lm, LEFT_EAR, RIGHT_EAR)
    lateral = lm[LEFT_EAR] - lm[RIGHT_EAR]  # toward the subject's left (+X at rest)
    forward = _mid(lm, LEFT_EYE, RIGHT_EYE) - ear_center  # the way the face points (+Z at rest)
    up = np.cross(forward, lateral)  # +Z x +X = +Y at rest
    up_norm = np.linalg.norm(up)
    ear_width = np.linalg.norm(lateral)
    if up_norm < 1e-9 or ear_width < 1e-9:
        # Degenerate (coincident ears/eyes): fall back to straight up.
        return ear_center + np.array([0.0, 0.15, 0.0])
    return ear_center + up / up_norm * ear_width * _HEAD_TOP_HEIGHT_PER_EAR_WIDTH


UP = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, -1.0, 0.0])
LEFT_X = np.array([1.0, 0.0, 0.0])
RIGHT_X = np.array([-1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])

# The foot bone runs ankle -> toe, which slopes down on a foot standing flat
# (ankle ~8 cm up, toe ~17 cm ahead: 25 degrees), so that's its rest direction.
# The foot's rotation itself is measured from the sole (heel -> toe), which is
# level on a flat foot, so a flat foot solves to identity whatever the
# subject's own ankle height; this pitch only matters for the bone's drawn
# direction and for the fallback when the heel is hidden.
FOOT_REST_PITCH_DEG = 25.0
FOOT_REST_DIRECTION = np.array(
    [0.0, -np.sin(np.radians(FOOT_REST_PITCH_DEG)), np.cos(np.radians(FOOT_REST_PITCH_DEG))]
)

BONES: list[BoneSpec] = [
    BoneSpec("spine", None, UP, _hip_center, _shoulder_center,
             (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER)),
    BoneSpec("neck", "spine", UP, _shoulder_center, lambda lm: _mid(lm, LEFT_EAR, RIGHT_EAR),
             (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR)),
    BoneSpec("head", "neck", UP, lambda lm: _mid(lm, LEFT_EAR, RIGHT_EAR), _head_top,
             (LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE)),
    BoneSpec("upper_arm.L", "spine", LEFT_X, lambda lm: lm[LEFT_SHOULDER], lambda lm: lm[LEFT_ELBOW],
             (LEFT_SHOULDER, LEFT_ELBOW)),
    BoneSpec("forearm.L", "upper_arm.L", LEFT_X, lambda lm: lm[LEFT_ELBOW], lambda lm: lm[LEFT_WRIST],
             (LEFT_ELBOW, LEFT_WRIST)),
    BoneSpec("hand.L", "forearm.L", LEFT_X, lambda lm: lm[LEFT_WRIST], lambda lm: _mid(lm, LEFT_INDEX, LEFT_PINKY),
             (LEFT_WRIST, LEFT_INDEX, LEFT_PINKY)),
    BoneSpec("upper_arm.R", "spine", RIGHT_X, lambda lm: lm[RIGHT_SHOULDER], lambda lm: lm[RIGHT_ELBOW],
             (RIGHT_SHOULDER, RIGHT_ELBOW)),
    BoneSpec("forearm.R", "upper_arm.R", RIGHT_X, lambda lm: lm[RIGHT_ELBOW], lambda lm: lm[RIGHT_WRIST],
             (RIGHT_ELBOW, RIGHT_WRIST)),
    BoneSpec("hand.R", "forearm.R", RIGHT_X, lambda lm: lm[RIGHT_WRIST], lambda lm: _mid(lm, RIGHT_INDEX, RIGHT_PINKY),
             (RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY)),
    BoneSpec("thigh.L", "spine", DOWN, lambda lm: lm[LEFT_HIP], lambda lm: lm[LEFT_KNEE],
             (LEFT_HIP, LEFT_KNEE)),
    BoneSpec("shin.L", "thigh.L", DOWN, lambda lm: lm[LEFT_KNEE], lambda lm: lm[LEFT_ANKLE],
             (LEFT_KNEE, LEFT_ANKLE)),
    BoneSpec("foot.L", "shin.L", FOOT_REST_DIRECTION, lambda lm: lm[LEFT_ANKLE], lambda lm: lm[LEFT_FOOT_INDEX],
             (LEFT_ANKLE, LEFT_FOOT_INDEX)),
    BoneSpec("thigh.R", "spine", DOWN, lambda lm: lm[RIGHT_HIP], lambda lm: lm[RIGHT_KNEE],
             (RIGHT_HIP, RIGHT_KNEE)),
    BoneSpec("shin.R", "thigh.R", DOWN, lambda lm: lm[RIGHT_KNEE], lambda lm: lm[RIGHT_ANKLE],
             (RIGHT_KNEE, RIGHT_ANKLE)),
    BoneSpec("foot.R", "shin.R", FOOT_REST_DIRECTION, lambda lm: lm[RIGHT_ANKLE], lambda lm: lm[RIGHT_FOOT_INDEX],
             (RIGHT_ANKLE, RIGHT_FOOT_INDEX)),
]

BONE_NAMES = [b.name for b in BONES]
_BONES_BY_NAME: dict[str, BoneSpec] = {b.name: b for b in BONES}

# -- hand landmarks and finger bones (B11) -------------------------------------

# MediaPipe's 21 hand landmarks: 0 wrist; thumb 1-4 (CMC, MCP, IP, tip); index
# 5-8 (MCP, PIP, DIP, tip); middle 9-12; ring 13-16; pinky 17-20.
NUM_HAND_LANDMARKS = 21
HAND_WRIST = 0
HAND_INDEX_MCP, HAND_MIDDLE_MCP, HAND_PINKY_MCP = 5, 9, 17
HAND_SIDES = ("L", "R")

FINGERS = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")
_FINGER_FIRST_LANDMARK = {"thumb": 1, "f_index": 5, "f_middle": 9, "f_ring": 13, "f_pinky": 17}


@dataclass(frozen=True)
class FingerBoneSpec:
    """One finger segment: head/tail are hand-landmark indices (not pose ones)."""

    name: str
    parent: str
    side: str
    finger: str
    segment: int  # 1 (knuckle) .. 3 (tip)
    rest_direction: np.ndarray
    head_index: int
    tail_index: int


def _finger_rest_direction(finger: str, side: str) -> np.ndarray:
    """Rest-pose finger directions on a palm-down T-pose hand: the four fingers
    straight along the arm, the thumb 45 degrees forward (toward the index side,
    +Z) -- conventions, like the rest of the rest skeleton."""
    sign = 1.0 if side == "L" else -1.0
    if finger == "thumb":
        return np.array([sign, 0.0, 1.0]) / np.sqrt(2.0)
    return np.array([sign, 0.0, 0.0])


FINGER_BONES: list[FingerBoneSpec] = [
    FingerBoneSpec(
        name=f"{finger}.0{segment}.{side}",
        parent=f"hand.{side}" if segment == 1 else f"{finger}.0{segment - 1}.{side}",
        side=side,
        finger=finger,
        segment=segment,
        rest_direction=_finger_rest_direction(finger, side),
        head_index=_FINGER_FIRST_LANDMARK[finger] + segment - 1,
        tail_index=_FINGER_FIRST_LANDMARK[finger] + segment,
    )
    for side in HAND_SIDES
    for finger in FINGERS
    for segment in (1, 2, 3)
]
FINGER_BONE_NAMES = [b.name for b in FINGER_BONES]
_FINGERS_BY_NAME: dict[str, FingerBoneSpec] = {b.name: b for b in FINGER_BONES}
ALL_BONE_NAMES = BONE_NAMES + FINGER_BONE_NAMES


def bone_parent(name: str) -> str | None:
    """The parent of any bone, body or finger."""
    if name in _BONES_BY_NAME:
        return _BONES_BY_NAME[name].parent
    return _FINGERS_BY_NAME[name].parent


def bone_rest_direction(name: str) -> np.ndarray:
    """The rest (head -> tail) direction of any bone, body or finger."""
    if name in _BONES_BY_NAME:
        return _BONES_BY_NAME[name].rest_direction
    return _FINGERS_BY_NAME[name].rest_direction


def _check_hands(hands: dict | None, columns: int) -> dict[str, np.ndarray]:
    """Validates an optional {"L"/"R": (21, columns)} hand-landmark dict."""
    checked: dict[str, np.ndarray] = {}
    for side, points in (hands or {}).items():
        if points is None:
            continue
        if side not in HAND_SIDES:
            raise ValueError(f"hand side must be 'L' or 'R', got {side!r}")
        points = np.asarray(points, dtype=float)
        if points.shape != (NUM_HAND_LANDMARKS, columns):
            raise ValueError(f"expected ({NUM_HAND_LANDMARKS}, {columns}) hand landmarks, got {points.shape}")
        checked[side] = points
    return checked

# Each bone's rest-pose secondary axis (perpendicular to its rest_direction):
# the zero-roll reference that the measured secondary axis is compared against.
# - spine/neck/head: the subject's left (hip, shoulder and ear lines).
# - upper arms and forearms: the elbow bend normal cross(upper, forearm) for a
#   forearm bent forward (+Z) from a T-pose: -Y on the left, +Y on the right.
# - thighs and shins: the knee bend normal for a shin bent backward from a
#   straight leg, cross(-Y, -Z) = +X on both sides.
# - hands: index minus pinky knuckle on a palm-down T-pose hand, +Z.
# - feet: the ankle-heel-toe plane's normal on a flat foot, +X.
# The exact directions are conventions, not anatomical claims; what matters
# is that the rest pose and every solved pose use the same ones.
_REST_SECONDARY: dict[str, np.ndarray] = {
    "spine": LEFT_X, "neck": LEFT_X, "head": LEFT_X,
    "upper_arm.L": DOWN, "forearm.L": DOWN, "hand.L": FORWARD,
    "upper_arm.R": UP, "forearm.R": UP, "hand.R": FORWARD,
    "thigh.L": LEFT_X, "shin.L": LEFT_X, "foot.L": LEFT_X,
    "thigh.R": LEFT_X, "shin.R": LEFT_X, "foot.R": LEFT_X,
}
# Fingers: the knuckle line (index minus pinky knuckle, +Z on a palm-down hand,
# the axis a finger curls about); the thumb: the back-of-hand normal (+Y).
for _finger_bone in FINGER_BONES:
    _REST_SECONDARY[_finger_bone.name] = UP if _finger_bone.finger == "thumb" else FORWARD
del _finger_bone

# How much of the head's twist about the neck goes into the neck bone; the head
# takes the rest. Real necks share a head turn roughly this way, and it avoids
# the candy-wrapper skinning of putting it all on one joint.
NECK_YAW_SHARE = 0.4
# How much of the hand's pronation/supination (twist about the forearm) goes
# into the forearm bone, which is where it anatomically happens; the hand
# keeps the rest.
FOREARM_TWIST_SHARE = 0.5
# At or above this elbow/knee bend, the bend plane alone sets the upper arm's or
# thigh's roll. Below it, the roll blends linearly (by sin of the bend) toward
# the roll carried from the torso, so a nearly straight limb, whose bend plane
# is mostly noise, neither flips nor jitters.
FULL_BEND_DEG = 30.0
# Below this visibility the heel is ignored and the foot falls back to the
# ankle -> toe line with FOOT_REST_PITCH_DEG.
HEEL_MIN_VISIBILITY = 0.5


def _finger_points(hands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for bone in FINGER_BONES:
        if bone.side in hands:
            points[f"{bone.name}.head"] = hands[bone.side][bone.head_index]
            points[f"{bone.name}.tail"] = hands[bone.side][bone.tail_index]
    return points


def joint_positions(landmarks: np.ndarray, hands: dict | None = None) -> dict[str, np.ndarray]:
    """Returns every bone's head/tail joint position, keyed as '<bone>.head' / '<bone>.tail'.

    Used by the 3D viewport to draw the skeleton without duplicating the BONES table.
    hands: optional {"L"/"R": (21, 3)} hand landmarks in the same frame; adds
    the finger bones of each hand given.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {landmarks.shape}")
    points: dict[str, np.ndarray] = {}
    for bone in BONES:
        points[f"{bone.name}.head"] = bone.head_fn(landmarks)
        points[f"{bone.name}.tail"] = bone.tail_fn(landmarks)
    points.update(_finger_points(_check_hands(hands, 3)))
    return points


def _head_top_2d(lm: np.ndarray) -> np.ndarray:
    """2D-image-space stand-in for _head_top, for image_joint_positions_2d below.
    _head_top needs a 3D cross product (the head's up axis is perpendicular to
    the ear line and the face direction); image coordinates only have two axes,
    and image y increases downward rather than up, so the photo overlay keeps
    this simpler nose/ear extrapolation -- it only decides where the drawn head
    line ends, not anything that's exported.
    """
    ear_center = _mid(lm, LEFT_EAR, RIGHT_EAR)
    return ear_center + (ear_center - lm[NOSE]) * 0.6 + np.array([0.0, -0.05])


# Bones whose 3D tail_fn isn't dimension-agnostic (only "head", via _head_top's
# cross product) need a 2D-specific override here for image_joint_positions_2d.
_BONE_2D_TAIL_OVERRIDES = {"head": _head_top_2d}


def image_joint_positions_2d(image_landmarks_xy: np.ndarray, hands_xy: dict | None = None) -> dict[str, np.ndarray]:
    """2D-image-space analogue of joint_positions(), for drawing the solved
    skeleton directly on the source photo (gui/photo_view.py) rather than only
    in the 3D viewport. Reuses BONES' head/tail landmark lookups where they're
    dimension-agnostic (plain indexing or midpoint averaging, which work the
    same regardless of how many columns `landmarks` has) -- only the head
    bone's extrapolated top-of-skull point needs a 2D-specific version, since
    the 3D one uses a cross product. hands_xy: optional {"L"/"R": (21, 2)}
    normalized hand landmarks, adding each given hand's finger bones.
    """
    if image_landmarks_xy.shape != (NUM_LANDMARKS, 2):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 2) landmarks, got {image_landmarks_xy.shape}")
    points: dict[str, np.ndarray] = {}
    for bone in BONES:
        points[f"{bone.name}.head"] = bone.head_fn(image_landmarks_xy)
        tail_fn = _BONE_2D_TAIL_OVERRIDES.get(bone.name, bone.tail_fn)
        points[f"{bone.name}.tail"] = tail_fn(image_landmarks_xy)
    points.update(_finger_points(_check_hands(hands_xy, 2)))
    return points


def bone_confidence(visibility: np.ndarray, hand_presence: dict[str, float] | None = None) -> dict[str, float]:
    """Per-bone confidence from MediaPipe's per-landmark visibility scores: the
    lowest visibility among the raw landmarks a bone's position depends on.

    A bone touching a hidden/occluded landmark (PoseForge's "hidden limb leaves
    too little to work with" case) ends up with a low score here, which
    export/blender_pose.py surfaces as "low_confidence" per bone so the export
    itself flags what's likely to need manual correction in Blender.

    hand_presence: optional {"L"/"R": hand-landmark model presence score}; each
    given hand's finger bones score min(presence, the pose wrist's visibility).
    """
    if visibility.shape != (NUM_LANDMARKS,):
        raise ValueError(f"expected ({NUM_LANDMARKS},) visibility, got {visibility.shape}")
    scores = {bone.name: float(min(visibility[i] for i in bone.visibility_landmarks)) for bone in BONES}
    wrists = {"L": LEFT_WRIST, "R": RIGHT_WRIST}
    for bone in FINGER_BONES:
        if hand_presence and bone.side in hand_presence:
            scores[bone.name] = float(min(hand_presence[bone.side], visibility[wrists[bone.side]]))
    return scores


# A landmark whose presence (B10) is below this is treated as outside the frame.
OUT_OF_FRAME_PRESENCE = 0.5


def bones_out_of_frame(landmark_presence: np.ndarray | None, threshold: float = OUT_OF_FRAME_PRESENCE) -> set[str]:
    """Body bones depending on a landmark MediaPipe places outside the frame (its
    per-landmark presence below `threshold`) -- e.g. the feet of a person cut off
    at the knees. Kept apart from bone_confidence's "low visibility", which
    also covers joints inside the frame that are merely hidden."""
    if landmark_presence is None:
        return set()
    if landmark_presence.shape != (NUM_LANDMARKS,):
        raise ValueError(f"expected ({NUM_LANDMARKS},) presence, got {landmark_presence.shape}")
    return {bone.name for bone in BONES if min(landmark_presence[i] for i in bone.visibility_landmarks) < threshold}


def _swing_rotation(rest_direction: np.ndarray, current_direction: np.ndarray) -> Rotation:
    """Shortest-arc rotation that takes `rest_direction` onto `current_direction` (both unit vectors)."""
    rest = rest_direction / np.linalg.norm(rest_direction)
    current = current_direction / np.linalg.norm(current_direction)

    dot = float(np.clip(np.dot(rest, current), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return Rotation.identity()
    if dot < -1.0 + 1e-9:
        # 180-degree rotation: pick any axis perpendicular to `rest`.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(rest[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(rest, fallback)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi)

    axis = np.cross(rest, current)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return Rotation.from_rotvec(axis * angle)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def _frame_rotation(
    rest_primary: np.ndarray, rest_secondary: np.ndarray, primary: np.ndarray, secondary: np.ndarray
) -> Rotation | None:
    """The rotation taking the rest frame (rest_primary, rest_secondary) onto the
    current one: `primary` exactly, and `secondary` after removing its component
    along `primary` (Gram-Schmidt). None when either current vector is
    degenerate, so the caller can fall back to a carried roll."""
    p = _unit(np.asarray(primary, dtype=float))
    if p is None:
        return None
    s = _unit(secondary - p * np.dot(secondary, p))
    if s is None:
        return None
    p0 = rest_primary / np.linalg.norm(rest_primary)
    s0 = rest_secondary - p0 * np.dot(rest_secondary, p0)
    s0 /= np.linalg.norm(s0)
    current = np.column_stack([p, s, np.cross(p, s)])
    rest = np.column_stack([p0, s0, np.cross(p0, s0)])
    return Rotation.from_matrix(current @ rest.T)


def _carried(parent_world: Rotation, rest_direction: np.ndarray, direction: np.ndarray) -> Rotation:
    """A bone's world rotation with no twist of its own relative to its parent:
    the parent's frame, swung (shortest arc) from where it points the bone's
    rest direction onto `direction`. Used wherever a bone's roll can't be
    measured, instead of a shortest arc from the world rest pose, which made
    the result depend on the camera angle."""
    if _unit(np.asarray(direction, dtype=float)) is None:
        return parent_world
    return _swing_rotation(parent_world.apply(rest_direction), direction) * parent_world


def _signed_angle_about(axis: np.ndarray, reference: np.ndarray, measured: np.ndarray) -> float | None:
    """Signed angle (radians, right-handed about `axis`) from `reference` to
    `measured`, both projected onto the plane perpendicular to `axis`."""
    a = _unit(np.asarray(axis, dtype=float))
    if a is None:
        return None
    r = _unit(reference - a * np.dot(reference, a))
    m = _unit(measured - a * np.dot(measured, a))
    if r is None or m is None:
        return None
    return float(np.arctan2(np.dot(np.cross(r, m), a), np.dot(r, m)))


def _twist_angle_about(rotation: Rotation, axis: np.ndarray) -> float:
    """The twist part of a swing-twist decomposition of `rotation` about the
    unit `axis`, as a signed angle in (-pi, pi]."""
    x, y, z, w = rotation.as_quat()
    angle = 2.0 * np.arctan2(float(np.dot([x, y, z], axis)), w)
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _bend_secondary(upper_direction: np.ndarray, lower_direction: np.ndarray, carried_secondary: np.ndarray) -> np.ndarray:
    """An upper arm's or thigh's secondary axis: the normal of the elbow/knee
    bend plane, cross(upper, lower), blended toward the roll carried from the
    torso as the limb straightens. The blend weight is sin(bend) /
    sin(FULL_BEND_DEG), capped at 1, so any bend of FULL_BEND_DEG or more is
    all bend plane, and a straight limb is all carried roll."""
    upper = _unit(np.asarray(upper_direction, dtype=float))
    lower = _unit(np.asarray(lower_direction, dtype=float))
    if upper is None or lower is None:
        return carried_secondary
    normal = np.cross(upper, lower)
    sin_bend = float(np.linalg.norm(normal))
    weight = min(1.0, sin_bend / np.sin(np.radians(FULL_BEND_DEG)))
    if sin_bend < 1e-9:
        return carried_secondary
    return weight * normal / sin_bend + (1.0 - weight) * carried_secondary


def _torso_attachment(spine_world: Rotation, spine_direction: np.ndarray, line: np.ndarray) -> Rotation:
    """The frame the arms (shoulder line) or legs (hip line) hang from: the spine
    direction with that line as its lateral axis. The single spine bone takes
    the average of both lines, so when the shoulders are twisted against the
    hips, a straight limb's carried roll follows its own girdle, not the
    average. Falls back to the spine's frame if the line is degenerate."""
    frame = _frame_rotation(UP, LEFT_X, spine_direction, line)
    return frame if frame is not None else spine_world


def _heel_is_usable(
    ankle: np.ndarray, heel: np.ndarray, toe: np.ndarray, heel_visibility: float | None
) -> bool:
    """Whether the heel landmark can define the sole: visible enough, no farther
    from the ankle than the toe is, and spanning a real triangle with the ankle
    and toe, with the sole within 60 degrees of the ankle -> toe line."""
    if heel_visibility is not None and heel_visibility < HEEL_MIN_VISIBILITY:
        return False
    foot_length = np.linalg.norm(toe - ankle)
    sole = toe - heel
    if foot_length < 1e-9 or np.linalg.norm(sole) < 1e-9 or np.linalg.norm(heel - ankle) > foot_length:
        return False
    cos_sole = np.dot(sole, toe - ankle) / (np.linalg.norm(sole) * foot_length)
    if cos_sole < np.cos(np.radians(60.0)):
        return False
    to_ankle = ankle - heel
    if np.linalg.norm(to_ankle) < 1e-9:
        return False
    sin_spread = np.linalg.norm(np.cross(to_ankle, sole)) / (np.linalg.norm(to_ankle) * np.linalg.norm(sole))
    return sin_spread > np.sin(np.radians(10.0))


# Per-side landmark indices the solver needs beyond the BONES table.
_SIDE_LANDMARKS: dict[str, dict[str, int]] = {
    "L": {"INDEX": LEFT_INDEX, "PINKY": LEFT_PINKY, "ANKLE": LEFT_ANKLE, "HEEL": LEFT_HEEL, "FOOT_INDEX": LEFT_FOOT_INDEX},
    "R": {"INDEX": RIGHT_INDEX, "PINKY": RIGHT_PINKY, "ANKLE": RIGHT_ANKLE, "HEEL": RIGHT_HEEL, "FOOT_INDEX": RIGHT_FOOT_INDEX},
}


_LEFT_RIGHT_PAIRS = [
    (LEFT_EYE_INNER, RIGHT_EYE_INNER), (LEFT_EYE, RIGHT_EYE), (LEFT_EYE_OUTER, RIGHT_EYE_OUTER),
    (LEFT_EAR, RIGHT_EAR), (MOUTH_LEFT, MOUTH_RIGHT),
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_ELBOW, RIGHT_ELBOW), (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_PINKY, RIGHT_PINKY), (LEFT_INDEX, RIGHT_INDEX), (LEFT_THUMB, RIGHT_THUMB),
    (LEFT_HIP, RIGHT_HIP), (LEFT_KNEE, RIGHT_KNEE), (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL), (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
]


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mirrors a pose left/right (a reflection through the body's midline, like
    Blender's "paste X-flipped pose"): swaps the left/right landmark rows AND
    reflects the lateral coordinate -- X -> -X for (33, 3) engine-frame
    landmarks, x -> 1 - x for (33, 2) normalized image landmarks. Used as the
    "body orientation" fix when the estimator's left/right labeling comes out
    swapped -- see PoseForge's equivalent control.

    Swapping rows alone (this function's old behavior) left the old left
    shoulder's position labeled as the right shoulder while it still sat on the
    +X side, so every arm/leg pointed across the body -- a T-pose came out with
    both arms rotated 180 degrees. A 1-D per-landmark array (e.g. visibility)
    has no coordinate to reflect, so only its rows are swapped.
    """
    mirrored = np.array(landmarks, copy=True)
    for left, right in _LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]
    if mirrored.ndim == 2 and mirrored.shape[1] == 3:
        mirrored[:, 0] = -mirrored[:, 0]
    elif mirrored.ndim == 2 and mirrored.shape[1] == 2:
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
    return mirrored


def mirror_hands(hands: dict | None) -> dict:
    """mirror_landmarks for a {"L"/"R": hand landmarks} dict: the sides swap
    and the lateral coordinate is reflected the same way (X -> -X for (21, 3),
    x -> 1 - x for (21, 2); a scalar per side, e.g. a presence score, just
    swaps)."""
    mirrored = {}
    for side, value in (hands or {}).items():
        other = "R" if side == "L" else "L"
        array = np.array(value, dtype=float, copy=True)
        if array.ndim == 2 and array.shape[1] == 3:
            array[:, 0] = -array[:, 0]
        elif array.ndim == 2 and array.shape[1] == 2:
            array[:, 0] = 1.0 - array[:, 0]
        mirrored[other] = array if array.ndim else float(array)
    return mirrored


# The one bone with parent=None -- see apply_orientation_correction.
ROOT_BONE = BONES[0].name

# Global orientation-correction candidates for the "body orientation / floor
# contact" GUI control (gui/main_window.py's orientation_combo, next to the
# mirror checkbox. _world_to_engine() in estimation/landmarker.py follows
# MediaPipe's documented axis convention, but a monocular estimate can still
# come out upside-down or facing 90 degrees off -- these are one-click fixes
# for that, the same "small, live-preview" shape the mirror checkbox already
# provides for a left/right swap. All rotations are in the engine frame (Y-up,
# +X image right / subject's left, +Z toward camera); "Rotate 90° Left" turns
# the subject toward their own left (+Z -> +X).
ORIENTATION_CORRECTIONS: dict[str, Rotation] = {
    "Default": Rotation.identity(),
    "Flip Upside-Down": Rotation.from_euler("z", 180, degrees=True),
    "Rotate 90° Left": Rotation.from_euler("y", 90, degrees=True),
    "Rotate 90° Right": Rotation.from_euler("y", -90, degrees=True),
}


def apply_orientation_correction(
    local_rotations: dict[str, Rotation], correction: Rotation, root_bone: str = ROOT_BONE
) -> dict[str, Rotation]:
    """Composes a global re-orientation `correction` onto the already-solved
    root bone's LOCAL rotation.

    `root_bone` (parent=None in BONES) has no parent to be relative to, so its
    local rotation already IS its world rotation -- left-multiplying just that
    one entry by `correction` rigidly re-orients the whole solved skeleton
    without touching any other bone's parent-relative angle (a global rotation
    of a rigid hierarchy leaves every child's relative-to-parent angle
    unchanged; only the root's absolute orientation changes). Returns a new
    dict; `local_rotations` is left untouched. Meant to be applied right after
    solve_bone_rotations(), before the result reaches the 3D viewport preview
    or build_pose_document()/save_pose_file() -- see gui/main_window.py.
    """
    if root_bone not in local_rotations:
        raise KeyError(f"unknown root bone {root_bone!r}")
    corrected = dict(local_rotations)
    corrected[root_bone] = correction * local_rotations[root_bone]
    return corrected


def rotate_joint_positions(
    points: dict[str, np.ndarray], correction: Rotation, pivot: np.ndarray
) -> dict[str, np.ndarray]:
    """Rigidly rotates every point in a joint_positions()-style dict about
    `pivot` by `correction` -- the position-space equivalent of
    apply_orientation_correction, for the 3D viewport preview.

    joint_positions() derives every point directly from raw landmark
    positions rather than by forward-kinematics from solved rotations, so
    there is no rotations dict for the viewport to consume in the first
    place; rotating the already-computed positions about a fixed pivot (the
    hip center, i.e. points["spine.head"]) produces the same rigid
    re-orientation visually that apply_orientation_correction produces for
    the exported rotations.
    """
    return {name: pivot + correction.apply(point - pivot) for name, point in points.items()}


def _solve_fingers(hand_world: Rotation, hand: np.ndarray, side: str) -> dict[str, Rotation]:
    """World rotations of one hand's 15 finger bones from its 21 landmarks: the
    first segment framed by the knuckle line (the thumb by the back-of-hand
    normal, which points +Y on a palm-down hand), the next two as hinges
    carried from their parent -- a finger joint is close to a 1-DOF hinge."""
    # The palm's own axes (from the hand bone's frame, so they're orthogonal to
    # the palm direction): a finger curls about the lateral one exactly.
    knuckles = hand_world.apply(FORWARD)
    back = hand_world.apply(UP)
    world: dict[str, Rotation] = {}
    for bone in FINGER_BONES:
        if bone.side != side:
            continue
        parent_world = hand_world if bone.segment == 1 else world[bone.parent]
        direction = hand[bone.tail_index] - hand[bone.head_index]
        carried = _carried(parent_world, bone.rest_direction, direction)
        if bone.segment == 1:
            secondary = back if bone.finger == "thumb" else knuckles
            frame = _frame_rotation(bone.rest_direction, _REST_SECONDARY[bone.name], direction, secondary)
            world[bone.name] = frame if frame is not None else carried
        else:
            world[bone.name] = carried
    return world


def _solve_world_rotations(
    landmarks: np.ndarray, visibility: np.ndarray | None, hands: dict[str, np.ndarray] | None = None
) -> dict[str, Rotation]:
    """Every bone's world rotation (engine frame, relative to the T-pose rest
    skeleton), parents before children. See the module docstring for the
    secondary axis each bone uses."""
    lm = landmarks
    hands = hands or {}

    def direction(name: str) -> np.ndarray:
        bone = _BONES_BY_NAME[name]
        return bone.tail_fn(lm) - bone.head_fn(lm)

    def frame_or(name: str, fallback: Rotation, secondary: np.ndarray | None) -> Rotation:
        if secondary is None:
            return fallback
        frame = _frame_rotation(_BONES_BY_NAME[name].rest_direction, _REST_SECONDARY[name], direction(name), secondary)
        return frame if frame is not None else fallback

    world: dict[str, Rotation] = {}

    # Torso: one spine bone from the hip center to the shoulder center, rolled
    # by the average of the hip and shoulder lines.
    hip_line = lm[LEFT_HIP] - lm[RIGHT_HIP]
    shoulder_line = lm[LEFT_SHOULDER] - lm[RIGHT_SHOULDER]
    lines = [v for v in (_unit(hip_line), _unit(shoulder_line)) if v is not None]
    spine_direction = direction("spine")
    world["spine"] = frame_or(
        "spine", _carried(Rotation.identity(), UP, spine_direction), np.sum(lines, axis=0) if lines else None
    )

    # Head: a full frame from the ear line. The neck carries NECK_YAW_SHARE of
    # the head's twist about the neck's own axis.
    ear_line = lm[LEFT_EAR] - lm[RIGHT_EAR]
    head_frame = _frame_rotation(UP, LEFT_X, direction("head"), ear_line)
    neck_direction = direction("neck")
    neck = _carried(world["spine"], UP, neck_direction)
    neck_axis = _unit(neck_direction)
    if head_frame is not None and neck_axis is not None:
        twist = _signed_angle_about(neck_direction, neck.apply(LEFT_X), head_frame.apply(LEFT_X))
        if twist is not None:
            neck = Rotation.from_rotvec(neck_axis * NECK_YAW_SHARE * twist) * neck
    world["neck"] = neck
    world["head"] = head_frame if head_frame is not None else _carried(neck, UP, direction("head"))

    for suffix in ("L", "R"):
        upper, forearm, hand = f"upper_arm.{suffix}", f"forearm.{suffix}", f"hand.{suffix}"
        thigh, shin, foot = f"thigh.{suffix}", f"shin.{suffix}", f"foot.{suffix}"

        # Arm: the upper arm rolls to the elbow's bend plane, the forearm is a
        # pure hinge on it, then the hand's twist is shared with the forearm.
        shoulders = _torso_attachment(world["spine"], spine_direction, shoulder_line)
        upper_rest = _BONES_BY_NAME[upper].rest_direction
        upper_carried = _carried(shoulders, upper_rest, direction(upper))
        secondary = _bend_secondary(direction(upper), direction(forearm), upper_carried.apply(_REST_SECONDARY[upper]))
        world[upper] = frame_or(upper, upper_carried, secondary)

        forearm_rest = _BONES_BY_NAME[forearm].rest_direction
        forearm_hinged = _carried(world[upper], forearm_rest, direction(forearm))
        hand_rest = _BONES_BY_NAME[hand].rest_direction
        if suffix in hands:
            # B11: the hand model's palm is far better than the pose model's
            # two knuckle points -- wrist -> middle knuckle, rolled by the
            # index-pinky knuckle line.
            palm = hands[suffix]
            hand_direction = palm[HAND_MIDDLE_MCP] - palm[HAND_WRIST]
            knuckles = palm[HAND_INDEX_MCP] - palm[HAND_PINKY_MCP]
            frame = _frame_rotation(hand_rest, _REST_SECONDARY[hand], hand_direction, knuckles)
            world[hand] = frame if frame is not None else _carried(forearm_hinged, hand_rest, hand_direction)
        else:
            knuckles = lm[_SIDE_LANDMARKS[suffix]["INDEX"]] - lm[_SIDE_LANDMARKS[suffix]["PINKY"]]
            world[hand] = frame_or(hand, _carried(forearm_hinged, hand_rest, direction(hand)), knuckles)
        pronation = _twist_angle_about(forearm_hinged.inv() * world[hand], forearm_rest)
        world[forearm] = forearm_hinged * Rotation.from_rotvec(forearm_rest * FOREARM_TWIST_SHARE * pronation)
        if suffix in hands:
            world.update(_solve_fingers(world[hand], hands[suffix], suffix))

        # Leg: the same bend-plane roll and hinge, from the knee.
        hips = _torso_attachment(world["spine"], spine_direction, hip_line)
        thigh_carried = _carried(hips, DOWN, direction(thigh))
        secondary = _bend_secondary(direction(thigh), direction(shin), thigh_carried.apply(_REST_SECONDARY[thigh]))
        world[thigh] = frame_or(thigh, thigh_carried, secondary)
        world[shin] = _carried(world[thigh], DOWN, direction(shin))

        # Foot: the sole frame when the heel is usable, else the ankle -> toe
        # line at the rest pitch with the shin's roll.
        ankle = lm[_SIDE_LANDMARKS[suffix]["ANKLE"]]
        heel_index = _SIDE_LANDMARKS[suffix]["HEEL"]
        heel = lm[heel_index]
        toe = lm[_SIDE_LANDMARKS[suffix]["FOOT_INDEX"]]
        foot_fallback = _carried(world[shin], FOOT_REST_DIRECTION, direction(foot))
        heel_visibility = None if visibility is None else float(visibility[heel_index])
        sole_frame = None
        if _heel_is_usable(ankle, heel, toe, heel_visibility):
            sole_frame = _frame_rotation(FORWARD, LEFT_X, toe - heel, np.cross(ankle - heel, toe - heel))
        world[foot] = sole_frame if sole_frame is not None else foot_fallback

    return world


def solve_bone_rotations(
    landmarks: np.ndarray, visibility: np.ndarray | None = None, hands: dict | None = None
) -> dict[str, Rotation]:
    """Computes each bone's LOCAL (parent-relative) rotation from a (33, 3)
    landmark array in the engine frame (see module docstring): a full frame per
    bone, direction plus roll. Returns bone name -> scipy Rotation; the root
    bone's local rotation is its world rotation.

    visibility: optional (33,) per-landmark visibility. Only the heels use it
    today: a heel below HEEL_MIN_VISIBILITY is ignored and that foot falls back
    to its ankle -> toe line (see FOOT_REST_PITCH_DEG). Without it, an
    implausibly placed heel still triggers the same fallback.

    hands: optional {"L"/"R": (21, 3)} hand landmarks in the engine frame (B11;
    only their directions matter, so they needn't be anchored at the wrist).
    Each given hand also gets its 15 finger bones, and its hand bone is
    measured from the palm. Without it the result is exactly the 15 body bones.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {landmarks.shape}")
    if visibility is not None and np.shape(visibility) != (NUM_LANDMARKS,):
        raise ValueError(f"expected ({NUM_LANDMARKS},) visibility, got {np.shape(visibility)}")

    world_rotations = _solve_world_rotations(np.asarray(landmarks, dtype=float), visibility, _check_hands(hands, 3))
    local_rotations: dict[str, Rotation] = {}
    for bone in BONES:
        if bone.parent is None:
            local_rotations[bone.name] = world_rotations[bone.name]
        else:
            local_rotations[bone.name] = world_rotations[bone.parent].inv() * world_rotations[bone.name]
    for finger in FINGER_BONES:
        if finger.name in world_rotations:
            local_rotations[finger.name] = world_rotations[finger.parent].inv() * world_rotations[finger.name]
    return local_rotations


def world_rotations_from_local(local_rotations: dict[str, Rotation]) -> dict[str, Rotation]:
    """Composes parent-relative rotations back into world (T-pose-relative)
    rotations down the BONES hierarchy. A body bone missing from the dict
    counts as identity; finger bones are included only when given."""
    world: dict[str, Rotation] = {}
    for bone in BONES:
        local = local_rotations.get(bone.name, Rotation.identity())
        world[bone.name] = local if bone.parent is None else world[bone.parent] * local
    for finger in FINGER_BONES:
        if finger.name in local_rotations:
            world[finger.name] = world[finger.parent] * local_rotations[finger.name]
    return world


# -- rest skeleton and forward kinematics --------------------------------------

# Typical adult body-segment lengths (nominally centimeters), for a rest
# skeleton with plausible proportions: the BVH export's OFFSETs and End Sites
# (export/bvh_export.py) and the default forward-kinematics preview. The solve
# itself only compares directions; this is a generic proportion table, not a
# calibrated anthropometric one. See measured_proportions() for a subject's own.
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

# Half the shoulder/hip width: how far upper_arm.L/R's heads sit to either
# side of the spine's tail (shoulder center), and thigh.L/R's to either side of
# its head (hip center), in the rest skeleton.
SHOULDER_HALF_WIDTH_CM = 18.0
HIP_HALF_WIDTH_CM = 10.0

# Bones whose head doesn't sit at their parent's tail: (which end of the
# parent it's measured from, which half width, lateral direction). Every other
# bone continues its parent's chain from the parent's tail.
_ATTACHMENTS: dict[str, tuple[str, str, np.ndarray]] = {
    "upper_arm.L": ("tail", "shoulder", LEFT_X),
    "upper_arm.R": ("tail", "shoulder", RIGHT_X),
    "thigh.L": ("head", "hip", LEFT_X),
    "thigh.R": ("head", "hip", RIGHT_X),
}


# Finger segment lengths and where each finger's first segment starts (its
# knuckle, or the thumb's base), relative to the wrist in the rest hand's own
# frame (left hand: +X along the hand, +Z toward the index side, +Y the back of
# the hand; the right hand mirrors X). Typical adult proportions, in cm, for a
# palm (wrist -> middle knuckle) of REST_PALM_LENGTH_CM; a measured hand scales
# them by its own palm length (the "palm.L"/"palm.R" entries measured_proportions
# adds).
REST_PALM_LENGTH_CM = 9.3
FINGER_LENGTHS_CM: dict[str, tuple[float, float, float]] = {
    "thumb": (3.8, 3.0, 2.6),
    "f_index": (4.0, 2.4, 2.0),
    "f_middle": (4.4, 2.8, 2.2),
    "f_ring": (4.1, 2.6, 2.1),
    "f_pinky": (3.2, 2.0, 1.9),
}
FINGER_BASE_OFFSETS_CM: dict[str, tuple[float, float, float]] = {
    "thumb": (2.0, -1.0, 2.0),
    "f_index": (9.0, 0.0, 2.0),
    "f_middle": (9.3, 0.0, 0.0),  # on the hand bone's own axis (wrist -> middle knuckle)
    "f_ring": (8.8, 0.0, -1.8),
    "f_pinky": (8.0, 0.0, -3.6),
}


def _palm_scale(lengths: dict[str, float], side: str) -> float:
    """Measured palm length / the rest one (1 for the cm rest skeleton)."""
    return float(lengths.get(f"palm.{side}", REST_PALM_LENGTH_CM)) / REST_PALM_LENGTH_CM


def bone_length(lengths: dict[str, float], name: str) -> float:
    """A bone's length from `lengths`, with fingers falling back to
    FINGER_LENGTHS_CM scaled to the measured palm (if any)."""
    if name in lengths:
        return float(lengths[name])
    finger = _FINGERS_BY_NAME[name]
    return FINGER_LENGTHS_CM[finger.finger][finger.segment - 1] * _palm_scale(lengths, finger.side)


def _finger_base_offset(lengths: dict[str, float], finger: FingerBoneSpec) -> np.ndarray:
    """Rest offset of a finger's first segment's head from the wrist."""
    offset = np.array(FINGER_BASE_OFFSETS_CM[finger.finger], dtype=float)
    if finger.side == "R":
        offset[0] = -offset[0]
    return offset * _palm_scale(lengths, finger.side)


def rest_head_positions(
    bone_lengths: dict[str, float] | None = None,
    shoulder_half_width: float = SHOULDER_HALF_WIDTH_CM,
    hip_half_width: float = HIP_HALF_WIDTH_CM,
) -> dict[str, np.ndarray]:
    """Every bone's head position in the rest (T-pose) skeleton, root at the
    origin, finger bones included. BONES lists every parent before its
    children, so one pass suffices."""
    lengths = BONE_LENGTHS_CM if bone_lengths is None else bone_lengths
    half_widths = {"shoulder": shoulder_half_width, "hip": hip_half_width}
    heads: dict[str, np.ndarray] = {}
    for bone in BONES:
        if bone.parent is None:
            heads[bone.name] = np.zeros(3)
            continue
        parent = _BONES_BY_NAME[bone.parent]
        end, width, lateral_direction = _ATTACHMENTS.get(bone.name, ("tail", None, np.zeros(3)))
        anchor = heads[parent.name]
        if end == "tail":
            anchor = anchor + parent.rest_direction * lengths[parent.name]
        lateral = lateral_direction * half_widths[width] if width is not None else np.zeros(3)
        heads[bone.name] = anchor + lateral
    for finger in FINGER_BONES:
        if finger.segment == 1:
            heads[finger.name] = heads[finger.parent] + _finger_base_offset(lengths, finger)
        else:
            finger_parent = _FINGERS_BY_NAME[finger.parent]
            heads[finger.name] = heads[finger_parent.name] + finger_parent.rest_direction * bone_length(lengths, finger_parent.name)
    return heads


def measured_proportions(
    landmarks: np.ndarray, hands: dict | None = None
) -> tuple[dict[str, float], float, float]:
    """A subject's own (bone_lengths, shoulder_half_width, hip_half_width), in
    the landmarks' units, for forward_kinematics(): with these, the FK skeleton
    lands on the landmarks wherever the solved rotations reproduce them.
    hands: optional {"L"/"R": (21, 3)} hand landmarks, adding each given hand's
    finger lengths and its palm length ("palm.L"/"palm.R", which scales the
    rest hand's knuckle positions)."""
    points = joint_positions(landmarks)
    lengths = {
        bone.name: float(np.linalg.norm(points[f"{bone.name}.tail"] - points[f"{bone.name}.head"])) for bone in BONES
    }
    for side, hand in _check_hands(hands, 3).items():
        lengths[f"palm.{side}"] = float(np.linalg.norm(hand[HAND_MIDDLE_MCP] - hand[HAND_WRIST]))
        for finger in FINGER_BONES:
            if finger.side == side:
                lengths[finger.name] = float(np.linalg.norm(hand[finger.tail_index] - hand[finger.head_index]))
    shoulder_half = float(np.linalg.norm(landmarks[LEFT_SHOULDER] - landmarks[RIGHT_SHOULDER])) / 2.0
    hip_half = float(np.linalg.norm(landmarks[LEFT_HIP] - landmarks[RIGHT_HIP])) / 2.0
    return lengths, shoulder_half, hip_half


def forward_kinematics(
    local_rotations: dict[str, Rotation],
    bone_lengths: dict[str, float] | None = None,
    shoulder_half_width: float = SHOULDER_HALF_WIDTH_CM,
    hip_half_width: float = HIP_HALF_WIDTH_CM,
    root_position: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Poses the rest skeleton with parent-relative rotations (what every export
    writes) and returns joint_positions()-style '<bone>.head'/'<bone>.tail'
    points, engine frame. This is the same evaluation a BVH reader or Blender
    does, so drawing it shows what the exported rotations actually say --
    unlike joint_positions(), which reads the landmarks directly. A bone
    missing from `local_rotations` counts as identity."""
    lengths = BONE_LENGTHS_CM if bone_lengths is None else bone_lengths
    rest_heads = rest_head_positions(lengths, shoulder_half_width, hip_half_width)
    world = world_rotations_from_local(local_rotations)
    origin = np.zeros(3) if root_position is None else np.asarray(root_position, dtype=float)
    points: dict[str, np.ndarray] = {}
    for bone in BONES:
        if bone.parent is None:
            head = origin
        else:
            offset = rest_heads[bone.name] - rest_heads[bone.parent]
            head = points[f"{bone.parent}.head"] + world[bone.parent].apply(offset)
        points[f"{bone.name}.head"] = head
        points[f"{bone.name}.tail"] = head + world[bone.name].apply(bone.rest_direction * lengths[bone.name])
    for finger in FINGER_BONES:
        if finger.name not in world:
            continue
        offset = rest_heads[finger.name] - rest_heads[finger.parent]
        head = points[f"{finger.parent}.head"] + world[finger.parent].apply(offset)
        points[f"{finger.name}.head"] = head
        points[f"{finger.name}.tail"] = head + world[finger.name].apply(
            finger.rest_direction * bone_length(lengths, finger.name)
        )
    return points


def floor_height(points: dict[str, np.ndarray]) -> float:
    """The height (Y) of the floor a posed skeleton stands on: its lowest joint
    point. Usually a heel or toe; a hand or knee when the pose rests on it (a
    crawl, a kneel), which is then what touches the floor."""
    return float(min(point[1] for point in points.values()))


def hip_height_above_floor(
    local_rotations: dict[str, Rotation],
    bone_lengths: dict[str, float] | None = None,
    shoulder_half_width: float = SHOULDER_HALF_WIDTH_CM,
    hip_half_width: float = HIP_HALF_WIDTH_CM,
) -> float:
    """How high the root (the hip center) sits above the floor in this pose:
    forward_kinematics with the root at the origin, minus floor_height. Same
    units as the bone lengths. A standing pose gives about the leg length, a
    crouch much less -- the world landmarks are hip-centred, so without this
    every pose stands with its hips at one fixed height."""
    points = forward_kinematics(local_rotations, bone_lengths, shoulder_half_width, hip_half_width)
    return -floor_height(points)


def bone_lateral_axes(local_rotations: dict[str, Rotation]) -> dict[str, np.ndarray]:
    """Each bone's posed roll axis (its rest secondary axis, see _REST_SECONDARY,
    carried by its world rotation), engine frame -- what the 3D preview turns
    each bone's octahedral body by, so roll is visible."""
    world = world_rotations_from_local(local_rotations)
    return {name: rotation.apply(_REST_SECONDARY[name]) for name, rotation in world.items()}


def _bone_axes_basis(bone_name: str) -> Rotation:
    """The bone's rest frame as Blender-style bone axes: Y along the bone, X its
    secondary axis (see _REST_SECONDARY), Z completing a right-handed frame."""
    y = bone_rest_direction(bone_name) / np.linalg.norm(bone_rest_direction(bone_name))
    x = _REST_SECONDARY[bone_name] - y * np.dot(_REST_SECONDARY[bone_name], y)
    x = x / np.linalg.norm(x)
    return Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))


def bone_local_euler_deg(bone_name: str, local_rotation: Rotation) -> tuple[float, float, float]:
    """A bone's local (parent-relative) rotation as X, Y, Z Euler degrees about
    its own bone axes (Y along the bone, as in Blender) -- the inverse of
    bone_axes_rotation, for the 3D view's hover label (F9)."""
    basis = _bone_axes_basis(bone_name)
    x, y, z = (basis.inv() * local_rotation * basis).as_euler("xyz", degrees=True)
    return float(x), float(y), float(z)


# -- manual correction ---------------------------------------------------------

# The landmarks downstream of each limb bone's head: what a depth flip of that
# bone moves (everything the bone and its children are built from, beyond its
# own head joint).
_LIMB_DOWNSTREAM: dict[str, tuple[int, ...]] = {}
for _side, _s in (("L", "LEFT"), ("R", "RIGHT")):
    _hand = tuple(globals()[f"{_s}_{n}"] for n in ("PINKY", "INDEX", "THUMB"))
    _foot = tuple(globals()[f"{_s}_{n}"] for n in ("HEEL", "FOOT_INDEX"))
    _LIMB_DOWNSTREAM[f"upper_arm.{_side}"] = (globals()[f"{_s}_ELBOW"], globals()[f"{_s}_WRIST"]) + _hand
    _LIMB_DOWNSTREAM[f"forearm.{_side}"] = (globals()[f"{_s}_WRIST"],) + _hand
    _LIMB_DOWNSTREAM[f"hand.{_side}"] = _hand
    _LIMB_DOWNSTREAM[f"thigh.{_side}"] = (globals()[f"{_s}_KNEE"], globals()[f"{_s}_ANKLE"]) + _foot
    _LIMB_DOWNSTREAM[f"shin.{_side}"] = (globals()[f"{_s}_ANKLE"],) + _foot
    _LIMB_DOWNSTREAM[f"foot.{_side}"] = (globals()[f"{_s}_FOOT_INDEX"],)
del _side, _s, _hand, _foot

FLIPPABLE_BONES = tuple(_LIMB_DOWNSTREAM)


def flip_limb_depth(landmarks: np.ndarray, bone_name: str) -> np.ndarray:
    """Reflects a limb front-to-back about its own head joint: every landmark
    downstream of `bone_name`'s head gets z -> 2 * z_head - z (engine +Z
    points at the camera, so that's the plane through the joint facing the
    camera). A photo can't tell a forearm reaching toward the camera from one
    reaching away by the same angle -- monocular estimation's most common
    single-limb error -- and this is the one-click fix: the limb's image
    position is unchanged, only its depth. Returns a new array."""
    if bone_name not in _LIMB_DOWNSTREAM:
        raise ValueError(f"{bone_name!r} isn't a limb bone (one of {FLIPPABLE_BONES})")
    flipped = np.array(landmarks, dtype=float, copy=True)
    head_z = _BONES_BY_NAME[bone_name].head_fn(flipped)[2]
    rows = list(_LIMB_DOWNSTREAM[bone_name])
    flipped[rows, 2] = 2.0 * head_z - flipped[rows, 2]
    return flipped


def flip_hand_depth(landmarks: np.ndarray, hands: dict | None, bone_name: str) -> dict:
    """The hand-landmark part of flip_limb_depth: a flip of an arm bone also
    reflects that side's hand landmarks (B11) about the same plane through the
    bone's head. `landmarks` are the pose landmarks before the flip. Returns a
    new dict; other hands (and leg flips) leave it unchanged."""
    if bone_name not in _LIMB_DOWNSTREAM:
        raise ValueError(f"{bone_name!r} isn't a limb bone (one of {FLIPPABLE_BONES})")
    flipped = {side: np.array(points, dtype=float, copy=True) for side, points in (hands or {}).items()}
    side = bone_name.rsplit(".", 1)[1]
    if bone_name.split(".")[0] in ("upper_arm", "forearm", "hand") and side in flipped:
        head_z = _BONES_BY_NAME[bone_name].head_fn(np.asarray(landmarks, dtype=float))[2]
        flipped[side][:, 2] = 2.0 * head_z - flipped[side][:, 2]
    return flipped


# Joints that can be dragged on the photo (F4), each with the joint its bone
# hangs from (None: no single parent bone -- a shoulder or hip just moves), and
# the landmarks that move rigidly with it (a wrist carries the hand, an ankle
# the foot).
DRAGGABLE_JOINTS: dict[int, int | None] = {
    LEFT_SHOULDER: None, RIGHT_SHOULDER: None,
    LEFT_ELBOW: LEFT_SHOULDER, RIGHT_ELBOW: RIGHT_SHOULDER,
    LEFT_WRIST: LEFT_ELBOW, RIGHT_WRIST: RIGHT_ELBOW,
    LEFT_HIP: None, RIGHT_HIP: None,
    LEFT_KNEE: LEFT_HIP, RIGHT_KNEE: RIGHT_HIP,
    LEFT_ANKLE: LEFT_KNEE, RIGHT_ANKLE: RIGHT_KNEE,
}
_JOINT_FOLLOWERS: dict[int, tuple[int, ...]] = {
    LEFT_WRIST: (LEFT_PINKY, LEFT_INDEX, LEFT_THUMB),
    RIGHT_WRIST: (RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB),
    LEFT_ANKLE: (LEFT_HEEL, LEFT_FOOT_INDEX),
    RIGHT_ANKLE: (RIGHT_HEEL, RIGHT_FOOT_INDEX),
}
_WRIST_HAND_SIDE = {LEFT_WRIST: "L", RIGHT_WRIST: "R"}
JOINT_LABELS: dict[int, str] = {
    LEFT_SHOULDER: "left shoulder", RIGHT_SHOULDER: "right shoulder",
    LEFT_ELBOW: "left elbow", RIGHT_ELBOW: "right elbow",
    LEFT_WRIST: "left wrist", RIGHT_WRIST: "right wrist",
    LEFT_HIP: "left hip", RIGHT_HIP: "right hip",
    LEFT_KNEE: "left knee", RIGHT_KNEE: "right knee",
    LEFT_ANKLE: "left ankle", RIGHT_ANKLE: "right ankle",
}


def image_to_world_scale(
    landmarks: np.ndarray, image_xy: np.ndarray, aspect: float = 1.0, visibility: np.ndarray | None = None
) -> tuple[float, np.ndarray, np.ndarray]:
    """A weak-perspective camera fitted to one person: (scale, world_center,
    image_center) with u = aspect * x, v = y (image units of the photo's
    height) ~ image_center + scale * (X, -Y) - scale * world_center(X, -Y).
    The world landmarks are camera-aligned (the crop rotation is undone), so
    only a scale and an offset separate them from the image. Least squares,
    weighted by visibility. Used to turn a 2D drag into a 3D move (F4) and to
    place the photo behind the 3D preview (F9)."""
    world = np.asarray(landmarks, dtype=float)[:, :2] * np.array([1.0, -1.0])
    image = np.asarray(image_xy, dtype=float) * np.array([aspect, 1.0])
    weights = np.ones(len(world)) if visibility is None else np.clip(np.asarray(visibility, dtype=float), 0.05, 1.0)
    weights = weights / weights.sum()
    world_center = (world * weights[:, None]).sum(axis=0)
    image_center = (image * weights[:, None]).sum(axis=0)
    w, i = world - world_center, image - image_center
    denominator = float((weights[:, None] * w * w).sum())
    scale = float((weights[:, None] * w * i).sum()) / denominator if denominator > 1e-12 else 0.0
    return scale, world_center * np.array([1.0, -1.0]), image_center


def move_joint(
    landmarks: np.ndarray,
    image_xy: np.ndarray,
    joint: int,
    new_xy,
    aspect: float = 1.0,
    visibility: np.ndarray | None = None,
    hands: dict | None = None,
    hands_xy: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Moves one joint to a new normalized image position (F4: a 2D drag on the
    photo) and lifts the move into 3D: the image displacement becomes a world
    one through image_to_world_scale; a joint with a parent bone keeps that
    bone's length and which side of its parent it lies in depth (the bone
    swings to the new image direction), others keep their depth. The joint's
    followers (hand or foot landmarks, and the hand landmarks of a wrist) move
    with it. Returns new (landmarks, image_xy, hands, hands_xy)."""
    if joint not in DRAGGABLE_JOINTS:
        raise ValueError(f"landmark {joint} can't be dragged (one of {sorted(DRAGGABLE_JOINTS)})")
    landmarks = np.array(landmarks, dtype=float, copy=True)
    image_xy = np.array(image_xy, dtype=float, copy=True)
    hands = {side: np.array(points, dtype=float, copy=True) for side, points in (hands or {}).items()}
    hands_xy = {side: np.array(points, dtype=float, copy=True) for side, points in (hands_xy or {}).items()}
    new_xy = np.asarray(new_xy, dtype=float)
    scale, _world_center, _image_center = image_to_world_scale(landmarks, image_xy, aspect, visibility)
    if scale <= 1e-9:
        return landmarks, image_xy, hands, hands_xy

    delta_image = new_xy - image_xy[joint]
    planar_delta = np.array([delta_image[0] * aspect, -delta_image[1]]) / scale
    old = landmarks[joint].copy()
    parent = DRAGGABLE_JOINTS[joint]
    if parent is None:
        new = old + np.array([planar_delta[0], planar_delta[1], 0.0])
    else:
        bone = old - landmarks[parent]
        length = float(np.linalg.norm(bone))
        planar = bone[:2] + planar_delta
        planar_length = float(np.linalg.norm(planar))
        depth = 0.0
        if planar_length < length:
            depth = np.sqrt(length * length - planar_length * planar_length) * (1.0 if bone[2] >= 0 else -1.0)
        new = landmarks[parent] + np.array([planar[0], planar[1], depth])
    shift = new - old
    landmarks[joint] = new
    image_xy[joint] = new_xy
    for follower in _JOINT_FOLLOWERS.get(joint, ()):
        landmarks[follower] += shift
        image_xy[follower] += delta_image
    side = _WRIST_HAND_SIDE.get(joint)
    if side in hands:
        hands[side] += shift
    if side in hands_xy:
        hands_xy[side] += delta_image
    return landmarks, image_xy, hands, hands_xy


def bone_axes_rotation(bone_name: str, angles_deg: tuple[float, float, float]) -> Rotation:
    """A manual adjustment of `bone_name` as a rotation in its own rest frame,
    from (X, Y, Z) degrees about Blender-style bone axes: Y runs along the bone
    (a twist), X is its secondary axis (see _REST_SECONDARY) and Z completes
    the right-handed frame -- so "Y" twists any bone, whichever way the engine
    frame's own axes happen to point along it."""
    basis = _bone_axes_basis(bone_name)
    return basis * Rotation.from_euler("xyz", angles_deg, degrees=True) * basis.inv()


def apply_bone_overrides(
    local_rotations: dict[str, Rotation], overrides: dict[str, Rotation]
) -> dict[str, Rotation]:
    """Composes manual per-bone adjustments (see bone_axes_rotation) onto solved
    local rotations, each in the bone's own posed frame (local * override), so
    an axis means the bone's own axis wherever it points. Its children follow,
    as in any rig. Returns a new dict."""
    adjusted = dict(local_rotations)
    for name, override in overrides.items():
        adjusted[name] = adjusted.get(name, Rotation.identity()) * override
    return adjusted

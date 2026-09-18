"""Maps MediaPipe's 33 pose landmarks onto a generic humanoid bone hierarchy and
computes per-bone rotations by comparing each bone's current direction (from
landmark positions) against its rest-pose direction (a T-pose).

Coordinate convention used throughout this module (the "engine" frame):
right-handed, Y-up, +X to the subject's right, +Z toward the camera.
Landmark-space vectors must already be in this frame before reaching
`solve_bone_rotations` -- see `estimation/landmarker.py` for the MediaPipe
world-landmark conversion.

This only solves the "swing" (the direction a bone points); twist around a
bone's own axis is not recoverable from joint positions alone and is left at
identity. This mirrors PoseForge's own disclaimer that a photo-derived pose is
an estimate, not an exact copy -- fine limbs and hand/foot rotation will often
need manual correction in the target application.
"""
from __future__ import annotations

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
    head_fn: "callable[[np.ndarray], np.ndarray]"
    tail_fn: "callable[[np.ndarray], np.ndarray]"
    # Raw landmark indices this bone's position depends on, for confidence scoring
    # (see bone_confidence) -- listed explicitly rather than derived from head_fn/tail_fn
    # since some of those (e.g. the head bone's extrapolated top-of-skull point) do
    # arithmetic that has no meaningful "visibility" of its own.
    visibility_landmarks: tuple[int, ...]


def _hip_center(lm: np.ndarray) -> np.ndarray:
    return _mid(lm, LEFT_HIP, RIGHT_HIP)


def _shoulder_center(lm: np.ndarray) -> np.ndarray:
    return _mid(lm, LEFT_SHOULDER, RIGHT_SHOULDER)


def _head_top(lm: np.ndarray) -> np.ndarray:
    ear_center = _mid(lm, LEFT_EAR, RIGHT_EAR)
    # Extrapolate a head-top point above the nose/ear line since MediaPipe has no skull landmark.
    return ear_center + (ear_center - lm[NOSE]) * 0.6 + np.array([0.0, 0.15, 0.0])


UP = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, -1.0, 0.0])
LEFT_X = np.array([1.0, 0.0, 0.0])
RIGHT_X = np.array([-1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, -1.0])

BONES: list[BoneSpec] = [
    BoneSpec("spine", None, UP, _hip_center, _shoulder_center,
             (LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER)),
    BoneSpec("neck", "spine", UP, _shoulder_center, lambda lm: _mid(lm, LEFT_EAR, RIGHT_EAR),
             (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_EAR, RIGHT_EAR)),
    BoneSpec("head", "neck", UP, lambda lm: _mid(lm, LEFT_EAR, RIGHT_EAR), _head_top,
             (LEFT_EAR, RIGHT_EAR, NOSE)),
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
    BoneSpec("thigh.L", None, DOWN, lambda lm: lm[LEFT_HIP], lambda lm: lm[LEFT_KNEE],
             (LEFT_HIP, LEFT_KNEE)),
    BoneSpec("shin.L", "thigh.L", DOWN, lambda lm: lm[LEFT_KNEE], lambda lm: lm[LEFT_ANKLE],
             (LEFT_KNEE, LEFT_ANKLE)),
    BoneSpec("foot.L", "shin.L", FORWARD, lambda lm: lm[LEFT_ANKLE], lambda lm: lm[LEFT_FOOT_INDEX],
             (LEFT_ANKLE, LEFT_FOOT_INDEX)),
    BoneSpec("thigh.R", None, DOWN, lambda lm: lm[RIGHT_HIP], lambda lm: lm[RIGHT_KNEE],
             (RIGHT_HIP, RIGHT_KNEE)),
    BoneSpec("shin.R", "thigh.R", DOWN, lambda lm: lm[RIGHT_KNEE], lambda lm: lm[RIGHT_ANKLE],
             (RIGHT_KNEE, RIGHT_ANKLE)),
    BoneSpec("foot.R", "shin.R", FORWARD, lambda lm: lm[RIGHT_ANKLE], lambda lm: lm[RIGHT_FOOT_INDEX],
             (RIGHT_ANKLE, RIGHT_FOOT_INDEX)),
]

BONE_NAMES = [b.name for b in BONES]


def joint_positions(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Returns every bone's head/tail joint position, keyed as '<bone>.head' / '<bone>.tail'.

    Used by the 3D viewport to draw the skeleton without duplicating the BONES table.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {landmarks.shape}")
    points: dict[str, np.ndarray] = {}
    for bone in BONES:
        points[f"{bone.name}.head"] = bone.head_fn(landmarks)
        points[f"{bone.name}.tail"] = bone.tail_fn(landmarks)
    return points


def bone_confidence(visibility: np.ndarray) -> dict[str, float]:
    """Per-bone confidence from MediaPipe's per-landmark visibility scores: the
    lowest visibility among the raw landmarks a bone's position depends on.

    A bone touching a hidden/occluded landmark (PoseForge's "hidden limb leaves
    too little to work with" case) ends up with a low score here, which
    export/blender_pose.py surfaces as "low_confidence" per bone so the export
    itself flags what's likely to need manual correction in Blender.
    """
    if visibility.shape != (NUM_LANDMARKS,):
        raise ValueError(f"expected ({NUM_LANDMARKS},) visibility, got {visibility.shape}")
    return {bone.name: float(min(visibility[i] for i in bone.visibility_landmarks)) for bone in BONES}


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


_LEFT_RIGHT_PAIRS = [
    (LEFT_EYE_INNER, RIGHT_EYE_INNER), (LEFT_EYE, RIGHT_EYE), (LEFT_EYE_OUTER, RIGHT_EYE_OUTER),
    (LEFT_EAR, RIGHT_EAR), (MOUTH_LEFT, MOUTH_RIGHT),
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_ELBOW, RIGHT_ELBOW), (LEFT_WRIST, RIGHT_WRIST),
    (LEFT_PINKY, RIGHT_PINKY), (LEFT_INDEX, RIGHT_INDEX), (LEFT_THUMB, RIGHT_THUMB),
    (LEFT_HIP, RIGHT_HIP), (LEFT_KNEE, RIGHT_KNEE), (LEFT_ANKLE, RIGHT_ANKLE),
    (LEFT_HEEL, RIGHT_HEEL), (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
]


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Swaps left/right landmark rows. Used as the "body orientation" fix when the
    estimator's facing guess comes out mirrored -- see PoseForge's equivalent control.
    """
    mirrored = landmarks.copy()
    for left, right in _LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]
    return mirrored


def solve_bone_rotations(landmarks: np.ndarray) -> dict[str, Rotation]:
    """Computes each bone's world-space swing rotation from a (33, 3) landmark array
    in the engine frame (see module docstring). Returns bone name -> scipy Rotation.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {landmarks.shape}")

    world_rotations: dict[str, Rotation] = {}
    local_rotations: dict[str, Rotation] = {}
    for bone in BONES:
        head = bone.head_fn(landmarks)
        tail = bone.tail_fn(landmarks)
        direction = tail - head
        if np.linalg.norm(direction) < 1e-6:
            world_rotations[bone.name] = Rotation.identity()
        else:
            world_rotations[bone.name] = _swing_rotation(bone.rest_direction, direction)

        if bone.parent is None:
            local_rotations[bone.name] = world_rotations[bone.name]
        else:
            parent_world = world_rotations[bone.parent]
            local_rotations[bone.name] = parent_world.inv() * world_rotations[bone.name]

    return local_rotations

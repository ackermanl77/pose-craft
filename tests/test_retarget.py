from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.retarget import skeleton as sk


def _t_pose_landmarks() -> np.ndarray:
    """A synthetic 33-landmark array whose bone directions exactly match BONES'
    rest directions, so every solved rotation should come out ~identity.

    Left/right placement here only needs to be self-consistent with skeleton.py's
    LEFT_X/RIGHT_X constants -- it is not a claim about MediaPipe's real axis
    convention (see skeleton.py's module docstring for that caveat).
    """
    lm = np.zeros((sk.NUM_LANDMARKS, 3))

    lm[sk.LEFT_HIP] = [0.1, 0.0, 0.0]
    lm[sk.RIGHT_HIP] = [-0.1, 0.0, 0.0]
    lm[sk.LEFT_SHOULDER] = [0.2, 0.5, 0.0]
    lm[sk.RIGHT_SHOULDER] = [-0.2, 0.5, 0.0]

    lm[sk.LEFT_EAR] = [0.05, 0.65, 0.0]
    lm[sk.RIGHT_EAR] = [-0.05, 0.65, 0.0]
    lm[sk.NOSE] = [0.0, 0.60, 0.0]

    lm[sk.LEFT_ELBOW] = lm[sk.LEFT_SHOULDER] + [0.25, 0.0, 0.0]
    lm[sk.LEFT_WRIST] = lm[sk.LEFT_ELBOW] + [0.25, 0.0, 0.0]
    lm[sk.LEFT_INDEX] = lm[sk.LEFT_WRIST] + [0.1, 0.0, 0.0]
    lm[sk.LEFT_PINKY] = lm[sk.LEFT_WRIST] + [0.1, 0.0, 0.0]

    lm[sk.RIGHT_ELBOW] = lm[sk.RIGHT_SHOULDER] + [-0.25, 0.0, 0.0]
    lm[sk.RIGHT_WRIST] = lm[sk.RIGHT_ELBOW] + [-0.25, 0.0, 0.0]
    lm[sk.RIGHT_INDEX] = lm[sk.RIGHT_WRIST] + [-0.1, 0.0, 0.0]
    lm[sk.RIGHT_PINKY] = lm[sk.RIGHT_WRIST] + [-0.1, 0.0, 0.0]

    lm[sk.LEFT_KNEE] = lm[sk.LEFT_HIP] + [0.0, -0.5, 0.0]
    lm[sk.LEFT_ANKLE] = lm[sk.LEFT_KNEE] + [0.0, -0.5, 0.0]
    lm[sk.LEFT_FOOT_INDEX] = lm[sk.LEFT_ANKLE] + [0.0, 0.0, -0.2]

    lm[sk.RIGHT_KNEE] = lm[sk.RIGHT_HIP] + [0.0, -0.5, 0.0]
    lm[sk.RIGHT_ANKLE] = lm[sk.RIGHT_KNEE] + [0.0, -0.5, 0.0]
    lm[sk.RIGHT_FOOT_INDEX] = lm[sk.RIGHT_ANKLE] + [0.0, 0.0, -0.2]

    return lm


def test_t_pose_yields_near_identity_rotations():
    landmarks = _t_pose_landmarks()
    rotations = sk.solve_bone_rotations(landmarks)

    assert set(rotations.keys()) == set(sk.BONE_NAMES)
    for name, rot in rotations.items():
        angle = np.linalg.norm(rot.as_rotvec())
        assert angle < 1e-6, f"{name} expected ~identity, got rotvec angle {angle}"


def test_raised_arm_matches_expected_swing():
    landmarks = _t_pose_landmarks()
    # Raise the left arm from pointing sideways (+X) to pointing up (+Y).
    landmarks[sk.LEFT_ELBOW] = landmarks[sk.LEFT_SHOULDER] + [0.0, 0.25, 0.0]

    rotations = sk.solve_bone_rotations(landmarks)
    expected = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])

    # upper_arm.L's parent (spine) is untouched, so local == world here.
    relative = rotations["upper_arm.L"] * expected.inv()
    assert np.linalg.norm(relative.as_rotvec()) < 1e-6


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sk.solve_bone_rotations(np.zeros((10, 3)))


def test_mirror_landmarks_swaps_left_and_right():
    landmarks = _t_pose_landmarks()
    mirrored = sk.mirror_landmarks(landmarks)

    assert np.array_equal(mirrored[sk.LEFT_SHOULDER], landmarks[sk.RIGHT_SHOULDER])
    assert np.array_equal(mirrored[sk.RIGHT_HIP], landmarks[sk.LEFT_HIP])
    # Mirroring twice returns the original.
    assert np.array_equal(sk.mirror_landmarks(mirrored), landmarks)

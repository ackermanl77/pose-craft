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
    # index/pinky spread proportional to (0, 0, 1) == skeleton._TWIST_REST_LATERAL,
    # i.e. "zero twist" by this project's own convention -- see test_hand_twist_*.
    lm[sk.LEFT_INDEX] = lm[sk.LEFT_WRIST] + [0.1, 0.0, 0.02]
    lm[sk.LEFT_PINKY] = lm[sk.LEFT_WRIST] + [0.1, 0.0, -0.02]

    lm[sk.RIGHT_ELBOW] = lm[sk.RIGHT_SHOULDER] + [-0.25, 0.0, 0.0]
    lm[sk.RIGHT_WRIST] = lm[sk.RIGHT_ELBOW] + [-0.25, 0.0, 0.0]
    lm[sk.RIGHT_INDEX] = lm[sk.RIGHT_WRIST] + [-0.1, 0.0, 0.02]
    lm[sk.RIGHT_PINKY] = lm[sk.RIGHT_WRIST] + [-0.1, 0.0, -0.02]

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


def test_leaning_torso_keeps_thigh_local_rotation_near_identity():
    """A whole-body rigid lean (torso and legs rotate together, no articulation
    at the hip) should leave thigh.L/R's *local* (parent-relative) rotation near
    identity -- it only changed because the parent (spine) did. Regression test
    for thigh.L/R being parented to "spine" in BONES, matching upper_arm.L/R:
    with parent=None instead, this local rotation incorrectly equaled the raw
    world rotation, i.e. any torso lean falsely read as a hip kick on export.
    """
    landmarks = _t_pose_landmarks()
    lean = Rotation.from_rotvec([np.pi / 6, 0.0, 0.0])  # 30 degree rigid lean
    leaned = lean.apply(landmarks)

    rotations = sk.solve_bone_rotations(leaned)

    assert np.linalg.norm(rotations["spine"].as_rotvec()) > 0.1  # the lean itself is captured
    for name in ("thigh.L", "thigh.R"):
        angle = np.linalg.norm(rotations[name].as_rotvec())
        assert angle < 1e-6, f"{name} expected ~identity local rotation, got rotvec angle {angle}"


def _sample_image_landmarks_2d() -> np.ndarray:
    lm = np.zeros((sk.NUM_LANDMARKS, 2))
    lm[sk.LEFT_HIP] = [0.55, 0.5]
    lm[sk.RIGHT_HIP] = [0.45, 0.5]
    lm[sk.LEFT_SHOULDER] = [0.6, 0.3]
    lm[sk.RIGHT_SHOULDER] = [0.4, 0.3]
    lm[sk.LEFT_EAR] = [0.52, 0.18]
    lm[sk.RIGHT_EAR] = [0.48, 0.18]
    lm[sk.NOSE] = [0.5, 0.2]
    lm[sk.LEFT_ELBOW] = [0.7, 0.35]
    lm[sk.RIGHT_ELBOW] = [0.3, 0.35]
    lm[sk.LEFT_WRIST] = [0.8, 0.4]
    lm[sk.RIGHT_WRIST] = [0.2, 0.4]
    lm[sk.LEFT_INDEX] = [0.82, 0.42]
    lm[sk.LEFT_PINKY] = [0.81, 0.41]
    lm[sk.RIGHT_INDEX] = [0.18, 0.42]
    lm[sk.RIGHT_PINKY] = [0.19, 0.41]
    lm[sk.LEFT_KNEE] = [0.56, 0.7]
    lm[sk.RIGHT_KNEE] = [0.44, 0.7]
    lm[sk.LEFT_ANKLE] = [0.57, 0.9]
    lm[sk.RIGHT_ANKLE] = [0.43, 0.9]
    lm[sk.LEFT_FOOT_INDEX] = [0.6, 0.95]
    lm[sk.RIGHT_FOOT_INDEX] = [0.4, 0.95]
    return lm


def test_image_joint_positions_2d_returns_every_bone_head_and_tail():
    joints = sk.image_joint_positions_2d(_sample_image_landmarks_2d())

    assert set(joints.keys()) == {f"{name}.{part}" for name in sk.BONE_NAMES for part in ("head", "tail")}
    for value in joints.values():
        assert value.shape == (2,)

    # A plain indexed bone should read straight through, unlike the head bone.
    landmarks = _sample_image_landmarks_2d()
    assert np.array_equal(joints["upper_arm.L.head"], landmarks[sk.LEFT_SHOULDER])
    assert np.array_equal(joints["upper_arm.L.tail"], landmarks[sk.LEFT_ELBOW])


def test_image_joint_positions_2d_head_bone_does_not_crash():
    # The one bone whose 3D tail_fn (_head_top) isn't dimension-agnostic -- this
    # is the regression case _BONE_2D_TAIL_OVERRIDES exists for.
    joints = sk.image_joint_positions_2d(_sample_image_landmarks_2d())
    assert joints["head.tail"].shape == (2,)


def test_image_joint_positions_2d_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sk.image_joint_positions_2d(np.zeros((sk.NUM_LANDMARKS, 3)))


def test_hand_twist_is_solved_from_index_pinky_spread():
    landmarks = _t_pose_landmarks()
    # Rotate just the index/pinky spread by +90 degrees around the hand's own
    # (unrotated, +X) axis -- (0, 0, 1) rest lateral -> (0, -1, 0) current lateral --
    # simulating a wrist twisted 90 degrees with the rest of the arm unchanged.
    landmarks[sk.LEFT_INDEX] = landmarks[sk.LEFT_WRIST] + [0.1, -0.02, 0.0]
    landmarks[sk.LEFT_PINKY] = landmarks[sk.LEFT_WRIST] + [0.1, 0.02, 0.0]

    rotations = sk.solve_bone_rotations(landmarks)
    expected = Rotation.from_rotvec([np.pi / 2, 0.0, 0.0])

    # forearm.L (hand.L's parent) is untouched, so local == world here.
    relative = rotations["hand.L"] * expected.inv()
    assert np.linalg.norm(relative.as_rotvec()) < 1e-6


def test_hand_twist_is_identity_when_index_and_pinky_coincide():
    landmarks = _t_pose_landmarks()
    # Collapse both onto their own midpoint: zero lateral spread (degenerate twist
    # reference) while keeping hand.L's tail -- and therefore its swing -- unchanged.
    midpoint = (landmarks[sk.LEFT_INDEX] + landmarks[sk.LEFT_PINKY]) / 2.0
    landmarks[sk.LEFT_INDEX] = midpoint
    landmarks[sk.LEFT_PINKY] = midpoint

    rotations = sk.solve_bone_rotations(landmarks)
    assert np.linalg.norm(rotations["hand.L"].as_rotvec()) < 1e-9


def test_other_bones_have_no_twist_solving():
    assert "forearm.L" not in sk._TWIST_CURRENT_LATERAL_FN
    assert "upper_arm.L" not in sk._TWIST_CURRENT_LATERAL_FN
    assert "thigh.L" not in sk._TWIST_CURRENT_LATERAL_FN


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sk.solve_bone_rotations(np.zeros((10, 3)))


def test_bone_confidence_takes_the_minimum_contributing_landmark():
    visibility = np.full(sk.NUM_LANDMARKS, 0.9)
    visibility[sk.LEFT_WRIST] = 0.1  # occluded hand

    confidence = sk.bone_confidence(visibility)

    assert set(confidence.keys()) == set(sk.BONE_NAMES)
    assert confidence["forearm.L"] == pytest.approx(0.1)  # depends on LEFT_WRIST
    assert confidence["hand.L"] == pytest.approx(0.1)  # depends on LEFT_WRIST
    assert confidence["upper_arm.L"] == pytest.approx(0.9)  # does not depend on LEFT_WRIST
    assert confidence["thigh.R"] == pytest.approx(0.9)  # unrelated side entirely


def test_bone_confidence_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sk.bone_confidence(np.zeros(10))


def test_mirror_landmarks_swaps_left_and_right():
    landmarks = _t_pose_landmarks()
    mirrored = sk.mirror_landmarks(landmarks)

    assert np.array_equal(mirrored[sk.LEFT_SHOULDER], landmarks[sk.RIGHT_SHOULDER])
    assert np.array_equal(mirrored[sk.RIGHT_HIP], landmarks[sk.LEFT_HIP])
    # Mirroring twice returns the original.
    assert np.array_equal(sk.mirror_landmarks(mirrored), landmarks)


def test_orientation_corrections_include_default_identity():
    assert "Default" in sk.ORIENTATION_CORRECTIONS
    assert np.linalg.norm(sk.ORIENTATION_CORRECTIONS["Default"].as_rotvec()) < 1e-9


def test_apply_orientation_correction_only_changes_the_root_bone():
    landmarks = _t_pose_landmarks()
    rotations = sk.solve_bone_rotations(landmarks)

    corrected = sk.apply_orientation_correction(rotations, sk.ORIENTATION_CORRECTIONS["Flip Upside-Down"])

    # Every non-root bone's LOCAL (parent-relative) rotation is untouched -- a
    # global re-orientation of the whole rigid skeleton doesn't change any
    # joint's angle relative to its own parent.
    for name in sk.BONE_NAMES:
        if name == sk.ROOT_BONE:
            continue
        assert np.allclose(corrected[name].as_quat(), rotations[name].as_quat())

    # The root bone's rotation is exactly the correction composed on top of
    # the original solved root rotation.
    expected_root = sk.ORIENTATION_CORRECTIONS["Flip Upside-Down"] * rotations[sk.ROOT_BONE]
    relative = corrected[sk.ROOT_BONE] * expected_root.inv()
    assert np.linalg.norm(relative.as_rotvec()) < 1e-9

    # The original dict passed in is never mutated.
    assert np.linalg.norm(rotations[sk.ROOT_BONE].as_rotvec()) < 1e-9


def test_apply_orientation_correction_with_default_is_a_no_op():
    landmarks = _t_pose_landmarks()
    rotations = sk.solve_bone_rotations(landmarks)

    corrected = sk.apply_orientation_correction(rotations, sk.ORIENTATION_CORRECTIONS["Default"])

    for name in sk.BONE_NAMES:
        assert np.allclose(corrected[name].as_quat(), rotations[name].as_quat())


def test_apply_orientation_correction_rejects_unknown_root_bone():
    with pytest.raises(KeyError):
        sk.apply_orientation_correction({}, Rotation.identity(), root_bone="not_a_bone")


def test_rotate_joint_positions_rotates_rigidly_about_the_pivot():
    points = {
        "spine.head": np.array([0.0, 0.0, 0.0]),
        "spine.tail": np.array([0.0, 1.0, 0.0]),
        "upper_arm.L.head": np.array([0.2, 0.5, 0.0]),
    }
    pivot = points["spine.head"]
    correction = Rotation.from_euler("z", 180, degrees=True)  # "Flip Upside-Down"

    rotated = sk.rotate_joint_positions(points, correction, pivot)

    # The pivot itself never moves.
    assert np.allclose(rotated["spine.head"], pivot)
    # A point straight up from the pivot ends up straight down (180 about Z
    # flips both X and Y).
    assert np.allclose(rotated["spine.tail"], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(rotated["upper_arm.L.head"], [-0.2, -0.5, 0.0], atol=1e-9)

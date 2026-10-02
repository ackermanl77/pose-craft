"""Full per-bone frames (body yaw, head turns, elbow/knee hinges, flat feet).

The old solver built every bone's rotation from its direction alone (a
shortest-arc swing), which dropped rotations the 33 landmarks do measure: a
whole body turned 60 degrees solved to a root rotation of 0, a 45-degree head
turn to 0, a 90-degree elbow to a 120-degree rotation with a large twist, and a
flat foot to 25 degrees toe-down. These tests pin each case to a measurable
target. All positions are in the engine frame (Y-up, +X the subject's left,
+Z toward the camera).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.retarget import skeleton as sk

X, Y, Z = np.eye(3)


def _angle_deg(rotation: Rotation) -> float:
    return float(np.degrees(np.linalg.norm(rotation.as_rotvec())))


def _set_flat_foot(lm: np.ndarray, side: str, forward=Z) -> None:
    """A foot resting flat: the heel and toe on the floor, the ankle 8 cm above
    it and 4 cm in front of the heel, the toe 17 cm ahead of the ankle."""
    ankle = lm[getattr(sk, f"{side}_ANKLE")]
    floor = ankle - 0.08 * Y
    lm[getattr(sk, f"{side}_HEEL")] = floor - 0.04 * np.asarray(forward)
    lm[getattr(sk, f"{side}_FOOT_INDEX")] = floor + 0.17 * np.asarray(forward)


def _standing_t_pose() -> np.ndarray:
    lm = np.zeros((sk.NUM_LANDMARKS, 3))
    lm[sk.LEFT_HIP], lm[sk.RIGHT_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
    lm[sk.LEFT_SHOULDER], lm[sk.RIGHT_SHOULDER] = [0.2, 0.5, 0.0], [-0.2, 0.5, 0.0]
    lm[sk.LEFT_EAR], lm[sk.RIGHT_EAR] = [0.07, 0.65, 0.0], [-0.07, 0.65, 0.0]
    lm[sk.LEFT_EYE], lm[sk.RIGHT_EYE] = [0.03, 0.65, 0.08], [-0.03, 0.65, 0.08]
    lm[sk.NOSE] = [0.0, 0.62, 0.1]
    for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
        shoulder = lm[getattr(sk, f"{side}_SHOULDER")]
        lm[getattr(sk, f"{side}_ELBOW")] = shoulder + [sign * 0.25, 0.0, 0.0]
        wrist = shoulder + [sign * 0.5, 0.0, 0.0]
        lm[getattr(sk, f"{side}_WRIST")] = wrist
        # Palm down: the index knuckle sits in front (+Z) of the pinky.
        lm[getattr(sk, f"{side}_INDEX")] = wrist + [sign * 0.1, 0.0, 0.02]
        lm[getattr(sk, f"{side}_PINKY")] = wrist + [sign * 0.1, 0.0, -0.02]
        hip = lm[getattr(sk, f"{side}_HIP")]
        lm[getattr(sk, f"{side}_KNEE")] = hip + [0.0, -0.45, 0.0]
        lm[getattr(sk, f"{side}_ANKLE")] = hip + [0.0, -0.85, 0.0]
        _set_flat_foot(lm, side)
    return lm


def _articulated_pose() -> np.ndarray:
    """Bent elbows and knees, a turned head, a raised foot: every bone has a
    non-trivial frame, so a rigid rotation exercises all of them."""
    lm = _standing_t_pose()
    lm[sk.LEFT_ELBOW] = lm[sk.LEFT_SHOULDER] + [0.15, -0.2, 0.05]
    lm[sk.LEFT_WRIST] = lm[sk.LEFT_ELBOW] + [0.05, 0.05, 0.22]
    lm[sk.LEFT_INDEX] = lm[sk.LEFT_WRIST] + [0.03, 0.02, 0.09]
    lm[sk.LEFT_PINKY] = lm[sk.LEFT_WRIST] + [-0.01, -0.01, 0.09]
    lm[sk.RIGHT_ELBOW] = lm[sk.RIGHT_SHOULDER] + [-0.1, -0.22, -0.05]
    lm[sk.RIGHT_WRIST] = lm[sk.RIGHT_ELBOW] + [0.02, 0.2, 0.1]
    lm[sk.RIGHT_INDEX] = lm[sk.RIGHT_WRIST] + [0.0, 0.09, 0.03]
    lm[sk.RIGHT_PINKY] = lm[sk.RIGHT_WRIST] + [-0.03, 0.08, 0.0]
    lm[sk.LEFT_KNEE] = lm[sk.LEFT_HIP] + [0.02, -0.3, 0.3]
    lm[sk.LEFT_ANKLE] = lm[sk.LEFT_KNEE] + [0.0, -0.4, -0.05]
    _set_flat_foot(lm, "LEFT", forward=np.array([0.3, 0.0, 1.0]) / np.hypot(0.3, 1.0))
    lm[sk.RIGHT_KNEE] = lm[sk.RIGHT_HIP] + [-0.03, -0.44, 0.05]
    lm[sk.RIGHT_ANKLE] = lm[sk.RIGHT_KNEE] + [0.0, -0.38, -0.12]
    _set_flat_foot(lm, "RIGHT")
    ear_center = (lm[sk.LEFT_EAR] + lm[sk.RIGHT_EAR]) / 2.0
    turn = Rotation.from_euler("y", 30, degrees=True)
    for index in (sk.LEFT_EAR, sk.RIGHT_EAR, sk.LEFT_EYE, sk.RIGHT_EYE, sk.NOSE):
        lm[index] = ear_center + turn.apply(lm[index] - ear_center)
    return lm


def _rigidly_rotated(lm: np.ndarray, rotation: Rotation) -> np.ndarray:
    pivot = (lm[sk.LEFT_HIP] + lm[sk.RIGHT_HIP]) / 2.0
    return pivot + rotation.apply(lm - pivot)


def test_standing_t_pose_with_flat_feet_is_identity():
    for name, rot in sk.solve_bone_rotations(_standing_t_pose()).items():
        assert _angle_deg(rot) < 1e-6, name


# --- B1: the root carries body yaw ---------------------------------------------


def test_body_turned_60_degrees_puts_the_whole_turn_on_the_root():
    turn = Rotation.from_euler("y", 60, degrees=True)
    rotations = sk.solve_bone_rotations(_rigidly_rotated(_standing_t_pose(), turn))

    assert _angle_deg(rotations["spine"] * turn.inv()) < 1e-6
    for name in sk.BONE_NAMES:
        if name != "spine":
            assert _angle_deg(rotations[name]) < 1e-6, name


@pytest.mark.parametrize("seed", range(50))
def test_a_rigid_rotation_changes_only_the_root(seed):
    rng = np.random.default_rng(seed)
    rotation = Rotation.random(random_state=rng)
    pose = _articulated_pose()

    original = sk.solve_bone_rotations(pose)
    rotated = sk.solve_bone_rotations(_rigidly_rotated(pose, rotation))

    assert _angle_deg(rotated["spine"] * (rotation * original["spine"]).inv()) < 1e-6
    for name in sk.BONE_NAMES:
        if name != "spine":
            assert _angle_deg(rotated[name] * original[name].inv()) < 1e-6, name


def test_shoulders_twisted_against_the_hips_split_the_twist_on_the_spine():
    lm = _standing_t_pose()
    twist = Rotation.from_euler("y", 40, degrees=True)
    shoulder_center = (lm[sk.LEFT_SHOULDER] + lm[sk.RIGHT_SHOULDER]) / 2.0
    upper_body = [
        sk.LEFT_SHOULDER, sk.RIGHT_SHOULDER, sk.LEFT_ELBOW, sk.RIGHT_ELBOW, sk.LEFT_WRIST, sk.RIGHT_WRIST,
        sk.LEFT_INDEX, sk.RIGHT_INDEX, sk.LEFT_PINKY, sk.RIGHT_PINKY,
        sk.LEFT_EAR, sk.RIGHT_EAR, sk.LEFT_EYE, sk.RIGHT_EYE, sk.NOSE,
    ]
    for index in upper_body:
        lm[index] = shoulder_center + twist.apply(lm[index] - shoulder_center)

    rotations = sk.solve_bone_rotations(lm)

    # One spine bone for the whole torso: it takes the average of the hip and
    # shoulder lines, and the arms and legs each take the remaining half.
    assert _angle_deg(rotations["spine"] * Rotation.from_euler("y", 20, degrees=True).inv()) < 1e-6
    assert abs(_angle_deg(rotations["upper_arm.L"]) - 20.0) < 1e-6
    assert abs(_angle_deg(rotations["thigh.L"]) - 20.0) < 1e-6


# --- B2: head turns ---------------------------------------------------------


def test_head_turned_45_degrees_splits_the_yaw_between_neck_and_head():
    lm = _standing_t_pose()
    ear_center = (lm[sk.LEFT_EAR] + lm[sk.RIGHT_EAR]) / 2.0
    turn = Rotation.from_euler("y", 45, degrees=True)
    for index in (sk.LEFT_EAR, sk.RIGHT_EAR, sk.LEFT_EYE, sk.RIGHT_EYE, sk.NOSE):
        lm[index] = ear_center + turn.apply(lm[index] - ear_center)

    rotations = sk.solve_bone_rotations(lm)

    head_world = rotations["spine"] * rotations["neck"] * rotations["head"]
    assert _angle_deg(head_world * turn.inv()) < 1e-6
    neck_yaw = np.degrees(rotations["neck"].as_rotvec()[1])
    head_yaw = np.degrees(rotations["head"].as_rotvec()[1])
    assert neck_yaw + head_yaw == pytest.approx(45.0, abs=1e-6)
    assert neck_yaw == pytest.approx(45.0 * sk.NECK_YAW_SHARE, abs=1e-6)


# --- B3: limb roll and hinges -----------------------------------------------


def test_90_degree_elbow_is_a_pure_hinge_on_the_forearm():
    """Upper arm hanging down, forearm pointing forward."""
    lm = _standing_t_pose()
    lm[sk.LEFT_ELBOW] = lm[sk.LEFT_SHOULDER] + [0.0, -0.25, 0.0]
    lm[sk.LEFT_WRIST] = lm[sk.LEFT_ELBOW] + [0.0, 0.0, 0.25]
    # Thumb side up: the index knuckle above the pinky, like a handshake.
    lm[sk.LEFT_INDEX] = lm[sk.LEFT_WRIST] + [0.0, 0.02, 0.1]
    lm[sk.LEFT_PINKY] = lm[sk.LEFT_WRIST] + [0.0, -0.02, 0.1]

    rotations = sk.solve_bone_rotations(lm)
    forearm = rotations["forearm.L"]

    assert _angle_deg(forearm) == pytest.approx(90.0, abs=1e-6)
    axis = forearm.as_rotvec() / np.linalg.norm(forearm.as_rotvec())
    assert abs(np.dot(axis, sk.LEFT_X)) < 1e-6  # perpendicular to the bone: no twist
    # The upper arm is a plain 90-degree swing down, with no roll.
    assert _angle_deg(rotations["upper_arm.L"] * Rotation.from_euler("z", -90, degrees=True).inv()) < 1e-6


def test_upper_arm_roll_follows_the_elbow_plane():
    """Same arm straight out, forearm bent up instead of forward: the upper arm
    has to roll 90 degrees about its own axis so the elbow bends that way."""
    lm = _standing_t_pose()
    lm[sk.LEFT_WRIST] = lm[sk.LEFT_ELBOW] + [0.0, 0.25, 0.0]
    # The hand carried along by the bend, untwisted: palm toward the head.
    lm[sk.LEFT_INDEX] = lm[sk.LEFT_WRIST] + [-0.02, 0.1, 0.0]
    lm[sk.LEFT_PINKY] = lm[sk.LEFT_WRIST] + [0.02, 0.1, 0.0]

    rotations = sk.solve_bone_rotations(lm)
    upper = rotations["upper_arm.L"]

    assert _angle_deg(upper) == pytest.approx(90.0, abs=1e-6)
    assert abs(abs(np.dot(upper.as_rotvec() / np.linalg.norm(upper.as_rotvec()), sk.LEFT_X)) - 1.0) < 1e-6
    forearm = rotations["forearm.L"]
    assert _angle_deg(forearm) == pytest.approx(90.0, abs=1e-6)
    assert abs(np.dot(forearm.as_rotvec(), sk.LEFT_X)) < 1e-6


def test_90_degree_knee_is_a_pure_hinge_on_the_shin():
    lm = _standing_t_pose()
    lm[sk.LEFT_KNEE] = lm[sk.LEFT_HIP] + [0.0, 0.0, 0.45]  # thigh forward
    lm[sk.LEFT_ANKLE] = lm[sk.LEFT_KNEE] + [0.0, -0.4, 0.0]  # shin straight down
    _set_flat_foot(lm, "LEFT")

    rotations = sk.solve_bone_rotations(lm)
    shin = rotations["shin.L"]

    assert _angle_deg(shin) == pytest.approx(90.0, abs=1e-6)
    assert abs(np.dot(shin.as_rotvec(), Y)) < 1e-6  # hinge axis is lateral, not along the shin
    assert _angle_deg(rotations["thigh.L"] * Rotation.from_euler("x", -90, degrees=True).inv()) < 1e-6
    # The foot is flat on the floor again, so its world rotation is identity.
    foot_world = rotations["spine"] * rotations["thigh.L"] * rotations["shin.L"] * rotations["foot.L"]
    assert _angle_deg(foot_world) < 1e-6


def test_hand_twist_is_shared_between_forearm_and_hand():
    lm = _standing_t_pose()
    twist = Rotation.from_euler("x", 90, degrees=True)
    wrist = lm[sk.LEFT_WRIST]
    for index in (sk.LEFT_INDEX, sk.LEFT_PINKY):
        lm[index] = wrist + twist.apply(lm[index] - wrist)

    rotations = sk.solve_bone_rotations(lm)

    assert _angle_deg(rotations["forearm.L"] * rotations["hand.L"] * twist.inv()) < 1e-6
    share = Rotation.from_euler("x", 90 * sk.FOREARM_TWIST_SHARE, degrees=True)
    assert _angle_deg(rotations["forearm.L"] * share.inv()) < 1e-6


# --- B4: feet ---------------------------------------------------------------


def test_flat_foot_has_no_pitch():
    """Ankle 8 cm up, toe 17 cm ahead of it: this used to export 25 degrees toe-down."""
    rotations = sk.solve_bone_rotations(_standing_t_pose())
    assert _angle_deg(rotations["foot.L"]) < 1e-6
    assert _angle_deg(rotations["foot.R"]) < 1e-6


def test_foot_pointing_down_pitches_by_the_sole_angle():
    lm = _standing_t_pose()
    # Tiptoe: the sole tilted 40 degrees toe-down about the ankle.
    tilt = Rotation.from_euler("x", 40, degrees=True)  # +Z (forward) tips toward -Y
    ankle = lm[sk.LEFT_ANKLE]
    for index in (sk.LEFT_HEEL, sk.LEFT_FOOT_INDEX):
        lm[index] = ankle + tilt.apply(lm[index] - ankle)

    foot = sk.solve_bone_rotations(lm)["foot.L"]
    assert _angle_deg(foot * tilt.inv()) < 1e-6


def test_foot_roll_comes_from_the_ankle_heel_toe_plane():
    lm = _standing_t_pose()
    heel = lm[sk.LEFT_HEEL]
    toe = lm[sk.LEFT_FOOT_INDEX]
    sole_axis = (toe - heel) / np.linalg.norm(toe - heel)
    roll = Rotation.from_rotvec(sole_axis * np.radians(20))
    lm[sk.LEFT_ANKLE] = heel + roll.apply(lm[sk.LEFT_ANKLE] - heel)
    lm[sk.LEFT_KNEE] = heel + roll.apply(lm[sk.LEFT_KNEE] - heel)

    rotations = sk.solve_bone_rotations(lm)
    foot_world = rotations["spine"] * rotations["thigh.L"] * rotations["shin.L"] * rotations["foot.L"]
    assert _angle_deg(foot_world * roll.inv()) < 1e-6


def test_hidden_heel_falls_back_to_the_rest_pitch():
    lm = _standing_t_pose()
    visibility = np.ones(sk.NUM_LANDMARKS)
    visibility[sk.LEFT_HEEL] = 0.1
    lm[sk.LEFT_HEEL] = lm[sk.LEFT_FOOT_INDEX] + [0.0, 0.3, 0.0]  # a wild guess for a hidden heel

    rotations = sk.solve_bone_rotations(lm, visibility=visibility)

    # Ankle 8 cm above a toe 17 cm ahead is atan(8/17) = 25.2 degrees down,
    # within a fraction of a degree of the assumed rest pitch.
    assert _angle_deg(rotations["foot.L"]) < 1.0


def test_implausible_heel_falls_back_without_visibility():
    lm = _standing_t_pose()
    lm[sk.LEFT_HEEL] = [0.0, 0.0, 0.0]  # nowhere near the foot
    assert _angle_deg(sk.solve_bone_rotations(lm)["foot.L"]) < 1.0

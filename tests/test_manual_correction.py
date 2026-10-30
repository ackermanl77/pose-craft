"""Manual correction math (F4): per-limb depth flips and per-bone adjustments."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _articulated_pose  # noqa: E402


def test_a_depth_flip_reflects_the_limb_about_its_own_joint():
    """A forearm reaching toward the camera becomes one reaching away by the
    same angle -- the same image, the other depth -- and nothing else moves."""
    lm = _articulated_pose()
    flipped = sk.flip_limb_depth(lm, "forearm.L")

    elbow = lm[sk.LEFT_ELBOW]
    for index in (sk.LEFT_WRIST, sk.LEFT_INDEX, sk.LEFT_PINKY, sk.LEFT_THUMB):
        np.testing.assert_allclose(flipped[index][:2], lm[index][:2])  # same image position
        assert flipped[index][2] - elbow[2] == pytest.approx(-(lm[index][2] - elbow[2]))
    untouched = [i for i in range(sk.NUM_LANDMARKS) if i not in (sk.LEFT_WRIST, sk.LEFT_INDEX, sk.LEFT_PINKY, sk.LEFT_THUMB)]
    np.testing.assert_array_equal(flipped[untouched], lm[untouched])
    np.testing.assert_allclose(sk.flip_limb_depth(flipped, "forearm.L"), lm)  # flipping twice undoes it

    # The bone length is preserved, and the solved forearm now points the other way in depth.
    for points in (lm, flipped):
        assert np.linalg.norm(points[sk.LEFT_WRIST] - points[sk.LEFT_ELBOW]) == pytest.approx(
            np.linalg.norm(lm[sk.LEFT_WRIST] - lm[sk.LEFT_ELBOW])
        )
    before = sk.forward_kinematics(sk.solve_bone_rotations(lm))
    after = sk.forward_kinematics(sk.solve_bone_rotations(flipped))
    depth_before = (before["forearm.L.tail"] - before["forearm.L.head"])[2]
    depth_after = (after["forearm.L.tail"] - after["forearm.L.head"])[2]
    assert np.sign(depth_before) == -np.sign(depth_after)


def test_a_thigh_flip_moves_the_whole_leg():
    lm = _articulated_pose()
    flipped = sk.flip_limb_depth(lm, "thigh.R")
    hip_z = lm[sk.RIGHT_HIP][2]
    for name in ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX"):
        index = getattr(sk, f"RIGHT_{name}")
        assert flipped[index][2] == pytest.approx(2 * hip_z - lm[index][2])
    np.testing.assert_array_equal(flipped[sk.LEFT_KNEE], lm[sk.LEFT_KNEE])


def test_only_limbs_can_be_flipped():
    assert set(sk.FLIPPABLE_BONES) == {n for n in sk.BONE_NAMES if n.split(".")[0] in ("upper_arm", "forearm", "hand", "thigh", "shin", "foot")}
    with pytest.raises(ValueError):
        sk.flip_limb_depth(_articulated_pose(), "spine")


def test_bone_overrides_compose_in_the_bones_own_frame():
    rotations = sk.solve_bone_rotations(_articulated_pose())
    twist = sk.bone_axes_rotation("forearm.L", (0.0, 25.0, 0.0))
    adjusted = sk.apply_bone_overrides(rotations, {"forearm.L": twist})
    np.testing.assert_allclose((rotations["forearm.L"].inv() * adjusted["forearm.L"]).as_rotvec(), twist.as_rotvec(), atol=1e-12)
    assert all(adjusted[n] is rotations[n] for n in sk.BONE_NAMES if n != "forearm.L")


@pytest.mark.parametrize("bone", sk.BONE_NAMES)
def test_bone_y_is_a_twist_along_the_bone_and_x_z_bend_it(bone):
    """Blender's convention for every bone, whatever engine axis it runs along:
    Y twists it in place, X and Z swing where it points."""
    rotations = sk.solve_bone_rotations(_articulated_pose())
    rest = sk._BONES_BY_NAME[bone].rest_direction

    def pointing(angles):
        adjusted = sk.apply_bone_overrides(rotations, {bone: sk.bone_axes_rotation(bone, angles)})
        return sk.world_rotations_from_local(adjusted)[bone].apply(rest)

    np.testing.assert_allclose(pointing((0, 40, 0)), pointing((0, 0, 0)), atol=1e-12)
    for angles in ((40, 0, 0), (0, 0, 40)):
        cos = np.dot(pointing(angles), pointing((0, 0, 0))) / np.linalg.norm(rest) ** 2
        assert cos == pytest.approx(np.cos(np.radians(40)))


def test_lateral_axes_follow_each_bones_roll():
    rest = sk.bone_lateral_axes({})
    np.testing.assert_allclose(rest["spine"], sk.LEFT_X)
    rolled = sk.bone_lateral_axes({"spine": Rotation.from_euler("y", 90, degrees=True)})
    np.testing.assert_allclose(rolled["spine"], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(rolled["head"], [0.0, 0.0, -1.0], atol=1e-12)  # children carry it


def test_a_mirrored_adjustment_is_the_reflection_on_the_opposite_bone():
    """UI7: paste mirrored -- the opposite bone, and angles whose rotation is the original's
    reflected through the midline (X -> -X), whatever the two rest frames are."""
    flip = np.diag([-1.0, 1.0, 1.0])
    for bone in ("forearm.L", "thigh.R", "foot.L", "spine", "head"):
        other, angles = sk.mirrored_bone_angles(bone, (10.0, -15.0, 30.0))
        assert other == sk.opposite_bone(bone)
        expected = flip @ sk.bone_axes_rotation(bone, (10.0, -15.0, 30.0)).as_matrix() @ flip
        np.testing.assert_allclose(sk.bone_axes_rotation(other, angles).as_matrix(), expected, atol=1e-3)
    assert sk.opposite_bone("upper_arm.R") == "upper_arm.L" and sk.opposite_bone("neck") == "neck"
    assert sk.mirrored_bone_angles("forearm.L", (10.0, -15.0, 30.0)) == ("forearm.R", (10.0, 15.0, -30.0))


def test_turning_a_posed_bone_about_a_world_axis_gives_the_adjustment_that_does_it():
    """UI6: rotating in the 3D view. The new adjustment, applied like any other, turns the bone's
    world frame by exactly that angle about that world axis (head fixed), whatever its parents do."""
    locals_ = {name: Rotation.from_euler("xyz", np.random.default_rng(i).uniform(-40, 40, 3), degrees=True)
               for i, name in enumerate(sk.BONE_NAMES)}
    bone, start = "forearm.L", (10.0, -5.0, 20.0)
    before = sk.apply_bone_overrides(locals_, {bone: sk.bone_axes_rotation(bone, start)})
    world_before = sk.world_rotations_from_local(before)[bone]
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    angles = sk.rotated_bone_angles(bone, start, world_before, axis, 35.0)
    after = sk.apply_bone_overrides(locals_, {bone: sk.bone_axes_rotation(bone, angles)})
    world_after = sk.world_rotations_from_local(after)[bone]
    expected = Rotation.from_rotvec(axis * np.radians(35.0)) * world_before
    np.testing.assert_allclose(world_after.as_matrix(), expected.as_matrix(), atol=5e-3)  # angles are rounded to 0.1°


def test_turning_about_the_bone_s_own_axis_adds_to_that_spin_box():
    world = sk.world_rotations_from_local({})["upper_arm.R"]
    for index, axis in enumerate("XYZ"):
        start = [0.0, 0.0, 0.0]
        angles = sk.rotated_bone_angles("upper_arm.R", tuple(start), world, sk.bone_axis_in_world("upper_arm.R", world, axis), 30.0)
        start[index] = 30.0
        assert angles == pytest.approx(tuple(start), abs=0.05)

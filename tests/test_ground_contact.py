"""Ground contact (B6): the world landmarks are hip-centred, so every export used
to put the hips at one fixed height -- a crouch exported standing in mid-air,
and the BVH root sat at the origin with the legs through the floor. These
tests pin the hip height to what the pose actually implies, and check the
multi-frame BVH writer the video path uses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import blender_pose
from pose_craft.export import bvh_export as bvh
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_axis_conventions import _bvh_forward_kinematics  # noqa: E402
from test_retarget_frames import _standing_t_pose  # noqa: E402


def _crouched_pose() -> np.ndarray:
    """The standing T-pose with both thighs swung forward to horizontal and the
    shins hanging straight down: the hips drop by about a thigh length."""
    lm = _standing_t_pose()
    for side in ("LEFT", "RIGHT"):
        hip = lm[getattr(sk, f"{side}_HIP")]
        knee = hip + [0.0, 0.0, 0.45]
        offset = knee - lm[getattr(sk, f"{side}_KNEE")]
        for name in ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX"):
            lm[getattr(sk, f"{side}_{name}")] += offset
    return lm


def test_a_crouch_sits_a_thigh_length_lower_than_standing():
    standing = sk.solve_bone_rotations(_standing_t_pose())
    crouched = sk.solve_bone_rotations(_crouched_pose())

    stand_height = sk.hip_height_above_floor(standing)
    crouch_height = sk.hip_height_above_floor(crouched)

    lengths = sk.BONE_LENGTHS_CM
    foot_drop = lengths["foot.L"] * np.sin(np.radians(sk.FOOT_REST_PITCH_DEG))
    assert stand_height == pytest.approx(lengths["thigh.L"] + lengths["shin.L"] + foot_drop, abs=0.5)
    assert stand_height - crouch_height == pytest.approx(lengths["thigh.L"], abs=0.5)


def test_floor_height_is_the_lowest_joint():
    points = {"a.head": np.array([0.0, 2.0, 0.0]), "a.tail": np.array([1.0, -3.0, 2.0])}
    assert sk.floor_height(points) == -3.0


def test_bvh_with_the_floor_root_position_stands_on_y_zero():
    for landmarks in (_standing_t_pose(), _crouched_pose()):
        rotations = sk.solve_bone_rotations(landmarks)
        document = bvh.build_bvh_document(rotations, root_position=bvh.floor_root_position(rotations))
        fk = _bvh_forward_kinematics(document)
        assert min(point[1] for point in fk.values()) == pytest.approx(0.0, abs=1e-4)


def test_pose_document_root_entry_reports_a_lowered_crouch():
    entries = []
    for landmarks in (_standing_t_pose(), _crouched_pose()):
        rotations = sk.solve_bone_rotations(landmarks)
        hip_height, leg_length = blender_pose.root_measurements(rotations, landmarks)
        document = blender_pose.build_pose_document(rotations, hip_height=hip_height, leg_length=leg_length)
        entries.append(document["root"])
    standing, crouched = entries
    assert standing["leg_length"] == pytest.approx(0.85)  # this fixture's thigh + shin, in meters
    assert standing["hip_height"] / standing["leg_length"] > 1.0  # straight legs plus the foot
    assert crouched["hip_height"] / crouched["leg_length"] < 0.7


def test_pose_document_has_no_root_entry_unless_asked():
    assert "root" not in blender_pose.build_pose_document({})


def test_multi_frame_bvh_writes_one_motion_line_per_frame():
    frames = [{"spine": Rotation.from_euler("y", angle, degrees=True)} for angle in (0, 30, 60)]
    document = bvh.build_bvh_animation(frames, frame_time=0.04, root_positions=[(0, 90, 0), (0, 80, 0), (0, 70, 0)])
    lines = document.strip().splitlines()
    motion = lines.index("MOTION")
    assert lines[motion + 1] == "Frames: 3"
    assert lines[motion + 2] == "Frame Time: 0.040000"
    rows = [[float(v) for v in line.split()] for line in lines[motion + 3 :]]
    assert len(rows) == 3 and all(len(row) == 6 + 3 * (len(sk.BONES) - 1) for row in rows)
    assert [row[1] for row in rows] == [90, 80, 70]
    for row, angle in zip(rows, (0, 30, 60)):
        spine = Rotation.from_euler(bvh.BVH_ROTATION_ORDER, row[3:6], degrees=True)
        assert np.degrees(spine.magnitude()) == pytest.approx(angle, abs=1e-4)


def test_single_frame_bvh_is_the_one_frame_animation():
    rotations = sk.solve_bone_rotations(_crouched_pose())
    assert bvh.build_bvh_document(rotations) == bvh.build_bvh_animation([rotations])


def test_bvh_animation_rejects_mismatched_root_positions():
    with pytest.raises(ValueError):
        bvh.build_bvh_animation([{}, {}], root_positions=[(0, 0, 0)])
    with pytest.raises(ValueError):
        bvh.build_bvh_animation([])

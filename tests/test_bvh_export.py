from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import bvh_export as bvh
from pose_craft.retarget import skeleton as sk

# The BONES table happens to already be authored in depth-first pre-order
# (spine, neck, head, upper_arm.L, forearm.L, hand.L, upper_arm.R, ...), which
# is also the exact order build_bvh_document() writes both the HIERARCHY
# block and the single MOTION frame's values in (see bvh_export._channel_order,
# which walks the same parent->children map the HIERARCHY writer does). So
# sk.BONE_NAMES doubles as the channel order for these tests.
_LEAF_BONES = {"head", "hand.L", "hand.R", "foot.L", "foot.R"}


def _motion_value_index(bone_name: str) -> int:
    """Index into the flat MOTION values list of the first (Z) rotation value
    for `bone_name` -- 3 leading root-position values, then 3 rotation values
    per bone in sk.BONE_NAMES order (the root's own rotation is channels 3-5)."""
    i = sk.BONE_NAMES.index(bone_name)
    return 3 + i * 3


def _parse_motion_values(document: str) -> list[float]:
    lines = document.splitlines()
    motion_index = lines.index("MOTION")
    assert lines[motion_index + 1] == "Frames: 1"
    assert lines[motion_index + 2].startswith("Frame Time:")
    return [float(v) for v in lines[motion_index + 3].split()]


def test_hierarchy_starts_correctly_and_has_balanced_braces():
    document = bvh.build_bvh_document({})
    lines = document.splitlines()

    assert lines[0] == "HIERARCHY"
    assert lines[1].startswith("ROOT spine")
    assert document.count("{") == document.count("}")
    assert document.count("{") > 0


def test_hierarchy_has_one_root_and_fourteen_joints_and_five_end_sites():
    document = bvh.build_bvh_document({})

    # 1 ROOT + 14 JOINT lines for the 15 bones total (root counted separately).
    assert sum(1 for line in document.splitlines() if line.strip().startswith("ROOT ")) == 1
    assert sum(1 for line in document.splitlines() if line.strip().startswith("JOINT ")) == len(sk.BONE_NAMES) - 1
    # One End Site per leaf bone (a bone with no children in BONES).
    assert document.count("End Site") == len(_LEAF_BONES)


def test_every_bone_name_appears_exactly_once_as_a_joint_or_root():
    document = bvh.build_bvh_document({})
    lines = document.splitlines()
    named = [
        line.strip().split(" ", 1)[1]
        for line in lines
        if line.strip().startswith("ROOT ") or line.strip().startswith("JOINT ")
    ]
    assert sorted(named) == sorted(sk.BONE_NAMES)


def test_motion_block_has_one_frame_and_the_right_channel_count():
    document = bvh.build_bvh_document({})
    values = _parse_motion_values(document)

    # 6 channels for the root (position + rotation) + 3 rotation channels for
    # each of the other 14 joints.
    expected_count = 6 + (len(sk.BONE_NAMES) - 1) * 3
    assert len(values) == expected_count


def test_identity_rotations_produce_zero_motion_values():
    document = bvh.build_bvh_document({name: Rotation.identity() for name in sk.BONE_NAMES})
    values = _parse_motion_values(document)
    assert np.allclose(values, 0.0, atol=1e-9)


def test_missing_bones_default_to_identity_without_raising():
    document = bvh.build_bvh_document({"spine": Rotation.from_euler("z", 30, degrees=True)})
    values = _parse_motion_values(document)
    assert len(values) == 6 + (len(sk.BONE_NAMES) - 1) * 3

    # Every bone other than spine was defaulted to identity -> zero rotation values.
    for name in sk.BONE_NAMES:
        if name == "spine":
            continue
        idx = _motion_value_index(name)
        assert np.allclose(values[idx : idx + 3], 0.0, atol=1e-9)


def test_known_rotation_round_trips_through_bvh_euler_values():
    """Assign a known, non-trivial rotation to two different bones (root and a
    non-root joint), then parse the written MOTION values back into a Rotation
    using the same 'ZXY' convention build_bvh_document() writes, and confirm it
    reconstructs the original -- this is the axis/order round trip a real BVH
    reader depends on.
    """
    spine_rotation = Rotation.from_euler("ZXY", [30.0, -15.0, 60.0], degrees=True)
    arm_rotation = Rotation.from_euler("ZXY", [-45.0, 10.0, 5.0], degrees=True)

    document = bvh.build_bvh_document({"spine": spine_rotation, "upper_arm.L": arm_rotation})
    values = _parse_motion_values(document)

    for bone_name, expected in (("spine", spine_rotation), ("upper_arm.L", arm_rotation)):
        idx = _motion_value_index(bone_name)
        z, x, y = values[idx : idx + 3]
        recovered = Rotation.from_euler(bvh.BVH_ROTATION_ORDER, [z, x, y], degrees=True)
        relative = recovered * expected.inv()
        assert np.linalg.norm(relative.as_rotvec()) < 1e-6, bone_name


def _offsets_by_joint(document: str) -> dict[str, np.ndarray]:
    """First OFFSET after each ROOT/JOINT line -- the joint's own OFFSET."""
    offsets: dict[str, np.ndarray] = {}
    current = None
    for line in document.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            current = stripped.split(" ", 1)[1]
        elif stripped.startswith("OFFSET") and current is not None:
            offsets[current] = np.array([float(v) for v in stripped.split()[1:]])
            current = None
    return offsets


def test_joint_offsets_are_head_relative_to_the_parents_head():
    """Regression test: every JOINT's OFFSET used to be its OWN bone vector
    (shifting the whole skeleton one level) -- BVH defines it as the joint's
    head relative to its parent's head."""
    offsets = _offsets_by_joint(bvh.build_bvh_document({}))
    L = bvh.BONE_LENGTHS_CM

    np.testing.assert_allclose(offsets["spine"], [0.0, 0.0, 0.0])  # placed by the root position channels
    np.testing.assert_allclose(offsets["neck"], [0.0, L["spine"], 0.0])  # the spine's own vector
    np.testing.assert_allclose(offsets["head"], [0.0, L["neck"], 0.0])
    np.testing.assert_allclose(offsets["upper_arm.L"], [bvh.SHOULDER_HALF_WIDTH_CM, L["spine"], 0.0])
    np.testing.assert_allclose(offsets["upper_arm.R"], [-bvh.SHOULDER_HALF_WIDTH_CM, L["spine"], 0.0])
    np.testing.assert_allclose(offsets["forearm.L"], [L["upper_arm.L"], 0.0, 0.0])
    np.testing.assert_allclose(offsets["thigh.L"], [bvh.HIP_HALF_WIDTH_CM, 0.0, 0.0])
    np.testing.assert_allclose(offsets["shin.R"], [0.0, -L["thigh.R"], 0.0])
    np.testing.assert_allclose(offsets["foot.R"], [0.0, -L["shin.R"], 0.0])


def test_save_bvh_file_writes_a_readable_bvh_file(tmp_path):
    path = tmp_path / "pose.bvh"
    bvh.save_bvh_file({"spine": Rotation.from_euler("y", 45, degrees=True)}, path)

    text = path.read_text(encoding="utf-8")
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text
    assert "Frames: 1" in text

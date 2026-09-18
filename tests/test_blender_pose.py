from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import blender_pose as bp


def test_identity_stays_identity():
    doc = bp.build_pose_document({"spine": Rotation.identity()})
    w, x, y, z = doc["bones"]["spine"]["quaternion_wxyz"]
    assert np.allclose([w, x, y, z], [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_engine_y_axis_becomes_blender_z_axis():
    # A 90-degree turn around the engine's up axis (Y) should become a 90-degree
    # turn around Blender's up axis (Z), since Y_engine maps to Z_blender.
    engine_rotation = Rotation.from_rotvec([0.0, np.pi / 2, 0.0])
    doc = bp.build_pose_document({"spine": engine_rotation})
    w, x, y, z = doc["bones"]["spine"]["quaternion_wxyz"]

    blender_rotation = Rotation.from_quat([x, y, z, w])
    rotvec = blender_rotation.as_rotvec()
    assert np.allclose(rotvec, [0.0, 0.0, np.pi / 2], atol=1e-9)


def test_low_confidence_flag_is_added_when_below_threshold():
    doc = bp.build_pose_document(
        {"upper_arm.L": Rotation.identity(), "thigh.R": Rotation.identity()},
        bone_confidence={"upper_arm.L": 0.2, "thigh.R": 0.95},
    )
    assert doc["bones"]["upper_arm.L"]["low_confidence"] is True
    assert doc["bones"]["upper_arm.L"]["confidence"] == pytest.approx(0.2)
    assert doc["bones"]["thigh.R"]["low_confidence"] is False


def test_no_confidence_keys_when_not_provided():
    doc = bp.build_pose_document({"upper_arm.L": Rotation.identity()})
    assert "confidence" not in doc["bones"]["upper_arm.L"]
    assert "low_confidence" not in doc["bones"]["upper_arm.L"]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "pose.json"
    bp.save_pose_file({"upper_arm.L": Rotation.from_rotvec([0.1, 0.2, 0.3])}, path, source_note="unit test")

    document = bp.load_pose_document(path)
    assert document["format"] == bp.FORMAT_NAME
    assert "upper_arm.L" in document["bones"]
    assert document["source_note"] == "unit test"

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == document

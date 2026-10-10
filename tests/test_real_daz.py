"""The DAZ Studio export in a real DAZ Studio (opt-in: set POSECRAFT_DAZ to DAZStudio.exe; needs
Genesis 9 in the DAZ content library). A PoseCraft pose is written as a .duf pose preset, DAZ Studio
runs headless (tests/real_daz_apply_pose.dsa) to load Genesis 9, apply the preset and dump every
bone's world rotation, and each must match what export/daz_pose.py intended. Checked with DAZ
Studio 6.25 on 2026-10-09 (about 30 s, most of it DAZ starting up)."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import daz_pose

DAZ = os.environ.get("POSECRAFT_DAZ", "")
SCRIPT = Path(__file__).with_name("real_daz_apply_pose.dsa")
pytestmark = pytest.mark.skipif(not (DAZ and Path(DAZ).is_file()), reason="set POSECRAFT_DAZ to DAZStudio.exe")

# Within Genesis 9's joint limits, so DAZ applies every channel as written.
POSE = {
    "spine": Rotation.from_euler("x", 10, degrees=True),
    "head": Rotation.from_euler("y", 15, degrees=True),
    "upper_arm.L": Rotation.from_euler("yz", [-40, -20], degrees=True),
    "upper_arm.R": Rotation.from_euler("z", 30, degrees=True),
    "thigh.L": Rotation.from_euler("x", -35, degrees=True),
    "shin.L": Rotation.from_euler("x", 45, degrees=True),
    "thigh.R": Rotation.from_euler("z", -10, degrees=True),
}


def _daz_rotation(wxyz):
    # DAZ Script's DzQuat reads back as the conjugate of the usual (w, x, y, z) convention.
    w, x, y, z = wxyz
    return Rotation.from_quat([-x, -y, -z, w])


def test_a_pose_preset_poses_genesis9_as_intended(tmp_path):
    figure = daz_pose.load_daz_figure()
    figure_scene = figure.path.parents[4] / "People" / "Genesis 9" / "Genesis 9.duf"
    preset = daz_pose.save_daz_pose_file(POSE, tmp_path / "pose.duf", figure, hip_height=0.85, leg_length=0.95)
    out = tmp_path / "applied.json"
    subprocess.run(
        [DAZ, "-headless", "-noPrompt", "-instanceName", "posecraft_test", "-scriptArg", str(preset),
         "-scriptArg", str(out), "-scriptArg", str(figure_scene), str(SCRIPT)],
        timeout=600, capture_output=True,
    )
    applied = json.loads(out.read_text(encoding="utf-8"))
    solution = daz_pose.solve_daz_pose(POSE, figure)
    assert solution.clamped == []
    expected = solution.world
    # Bones DAZ drives itself are left out: the twist bones and metatarsals follow the main joints
    # through Genesis 9's own formulas, and the eyes keep their default look. None is a parent of a
    # posed bone, so they don't change the pose.
    driven = ("twist", "metatarsal", "_eye")
    errors = {
        name: float(np.degrees((_daz_rotation(applied["bones"][name]["ws_rot"]).inv() * expected[name]).magnitude()))
        for name in figure.bones
        if name in applied["bones"] and not any(part in name for part in driven)
    }
    worst = max(errors, key=errors.get)
    assert errors[worst] < 0.01, (worst, errors[worst])
    assert applied["hip_y"] == pytest.approx(daz_pose.hip_height_offset(figure, 0.85, 0.95), abs=1e-3)
    # And the posed bones point where PoseCraft's do (a locked axis can cost a few degrees).
    targets = daz_pose.posed_world_rotations(POSE, figure)
    for name in ("l_upperarm", "l_forearm", "l_thigh", "l_shin", "r_upperarm", "head"):
        bone = figure.bones[name]
        rest = (bone.end - bone.origin) / np.linalg.norm(bone.end - bone.origin)
        aim = np.degrees(np.arccos(np.clip(np.dot(_daz_rotation(applied["bones"][name]["ws_rot"]).apply(rest),
                                                   targets[name].apply(rest)), -1, 1)))
        assert aim < 6.0, (name, aim)

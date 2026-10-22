"""FBX export through Blender (F11).

The real conversion is opt-in like tests/test_real_blender.py: it runs when
POSECRAFT_BLENDER points at blender.exe, imports both the glTF PoseCraft wrote
and the FBX Blender made from it, and checks every bone is at the same place
in every frame."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import fbx_export
from pose_craft.export.gltf_export import build_gltf_animation

BLENDER = os.environ.get("POSECRAFT_BLENDER", "")

POSE = {
    "spine": Rotation.from_euler("y", 25, degrees=True),
    "upper_arm.L": Rotation.from_euler("z", -40, degrees=True),
    "forearm.L": Rotation.from_euler("y", 30, degrees=True),
    "shin.R": Rotation.from_euler("x", 50, degrees=True),
}


def test_blender_is_found_in_the_environment_then_program_files(tmp_path):
    exe = tmp_path / "custom" / "blender.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert fbx_export.find_blender(environ={"POSECRAFT_BLENDER": str(exe)}) == exe
    for version in ("Blender 4.2", "Blender 5.2", "Blender 5.10"):
        folder = tmp_path / "pf" / "Blender Foundation" / version
        folder.mkdir(parents=True)
        (folder / "blender.exe").write_bytes(b"")
    found = fbx_export.find_blender(environ={}, program_dirs=[str(tmp_path / "pf")])
    assert found.parent.name == "Blender 5.10"


def test_a_failed_conversion_reports_blenders_message(tmp_path):
    def runner(command, **kwargs):
        assert command[1:3] == ["--background", "--factory-startup"]
        assert "BLENDER_USER_CONFIG" in kwargs["env"]
        return SimpleNamespace(returncode=1, stdout="", stderr="Error: glTF import failed\n")

    with pytest.raises(RuntimeError, match="glTF import failed"):
        fbx_export.save_fbx_pose(POSE, tmp_path / "x.fbx", blender="blender", runner=runner)


_COMPARE = r"""
import json, sys
import bpy
glb, fbx, out = sys.argv[sys.argv.index("--") + 1:][:3]

def positions(path, importer):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30  # the glTF's keys are 1/30 s apart; the FBX carries its own rate
    importer(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    action = arm.animation_data.action
    start, end = (int(v) for v in action.frame_range)
    result = {}
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        result[frame - start] = {b.name: list(arm.matrix_world @ b.head) for b in arm.pose.bones}
    return result

data = {"glb": positions(glb, bpy.ops.import_scene.gltf), "fbx": positions(fbx, bpy.ops.import_scene.fbx)}
open(out, "w").write(json.dumps(data))
"""


@pytest.mark.skipif(not (BLENDER and Path(BLENDER).is_file()), reason="set POSECRAFT_BLENDER to a blender executable")
@pytest.mark.parametrize("skeleton", ["posecraft", "mixamo", "ue5"])
def test_the_fbx_holds_the_same_animation_as_the_gltf(tmp_path, skeleton):
    frames = [POSE, {"thigh.L": Rotation.from_euler("x", -60, degrees=True)}]
    roots = [(0.0, 95.0, 0.0), (5.0, 90.0, 10.0)]
    glb = tmp_path / "pose.glb"
    glb.write_bytes(build_gltf_animation(frames, root_positions=roots, skeleton=skeleton))
    fbx = tmp_path / "pose.fbx"
    fbx_export.save_fbx_animation(frames, fbx, blender=BLENDER, root_positions=roots, skeleton=skeleton)
    out = tmp_path / "positions.json"
    run = subprocess.run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr",
                          _COMPARE, "--", str(glb), str(fbx), str(out)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["glb"].keys() == data["fbx"].keys() and len(data["glb"]) == 2
    for frame, bones in data["glb"].items():
        assert set(bones) == set(data["fbx"][frame])
        for name, head in bones.items():
            assert all(abs(a - b) < 1e-3 for a, b in zip(head, data["fbx"][frame][name])), (skeleton, frame, name)

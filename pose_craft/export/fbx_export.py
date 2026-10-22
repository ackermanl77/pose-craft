"""FBX export through Blender (F11): PoseCraft writes its glTF (export/gltf_export.py, on the
chosen skeleton preset) and a headless Blender imports it and saves it as FBX with Blender's
own exporter -- no FBX SDK, and the file is what Unity, Unreal, Maya and MotionBuilder expect
from Blender.

Blender is found at, in order: the POSECRAFT_BLENDER environment variable, the "blender/path"
setting, then the newest ``Blender Foundation\\Blender *\\blender.exe`` under Program Files
(``blender`` on PATH elsewhere). It runs with --background --factory-startup and its user
folders pointed at a temporary folder, so the user's Blender preferences and add-ons are never
loaded or changed. A pose takes a few seconds (Blender's start-up).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pose_craft.export.gltf_export import build_gltf_animation
from pose_craft.export.skeleton_presets import DEFAULT_SKELETON

BLENDER_TIMEOUT_S = 300

# Run inside Blender: import the .glb at the animation's own frame rate (the importer turns glTF
# seconds into frames at the scene's fps), set the scene range to the animation (the FBX exporter
# bakes the scene range), export everything as FBX (armature + mesh + animation).
_CONVERT_SCRIPT = r"""
import math
import sys
import bpy

glb, fbx, fps = sys.argv[sys.argv.index("--") + 1:][:3]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps, scene.render.fps_base = int(fps), 1.0
bpy.ops.import_scene.gltf(filepath=glb)
ranges = [action.frame_range for action in bpy.data.actions]
if ranges:
    scene.frame_start = int(math.floor(min(r[0] for r in ranges)))
    scene.frame_end = max(scene.frame_start, int(math.ceil(max(r[1] for r in ranges))))
bpy.ops.export_scene.fbx(
    filepath=fbx, use_selection=False, object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
    bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
    bake_anim_simplify_factor=0.0, armature_nodetype="NULL", primary_bone_axis="Y", secondary_bone_axis="X",
)
print("POSECRAFT_FBX_OK", fbx)
"""


class BlenderNotFound(RuntimeError):
    pass


def _version_key(path: Path) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", path.parent.name)) or (0,)


def find_blender(configured: str | None = None, environ=None, program_dirs=None) -> Path | None:
    environ = os.environ if environ is None else environ
    for candidate in (environ.get("POSECRAFT_BLENDER"), configured):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    if sys.platform == "win32" or program_dirs is not None:
        roots = program_dirs if program_dirs is not None else [environ.get("ProgramFiles", r"C:\Program Files")]
        found = [p for root in roots if root for p in Path(root, "Blender Foundation").glob("Blender*/blender.exe")]
        if found:
            return max(found, key=_version_key)
    on_path = shutil.which("blender")
    return Path(on_path) if on_path else None


def glb_to_fbx(glb_path: str | Path, fbx_path: str | Path, blender: str | Path, runner=subprocess.run,
               fps: int = 30) -> None:
    """Converts with a headless Blender; raises RuntimeError with Blender's own message on failure."""
    with tempfile.TemporaryDirectory(prefix="posecraft_blender_") as user:
        user_dir = Path(user)
        env = dict(
            os.environ,
            BLENDER_USER_RESOURCES=str(user_dir), BLENDER_USER_CONFIG=str(user_dir / "config"),
            BLENDER_USER_SCRIPTS=str(user_dir / "scripts"), BLENDER_USER_EXTENSIONS=str(user_dir / "extensions"),
            BLENDER_USER_DATAFILES=str(user_dir / "datafiles"),
        )
        command = [str(blender), "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python-expr", _CONVERT_SCRIPT, "--", str(glb_path), str(fbx_path), str(int(fps))]
        run = runner(command, env=env, capture_output=True, text=True, timeout=BLENDER_TIMEOUT_S)
    if run.returncode != 0 or "POSECRAFT_FBX_OK" not in (run.stdout or "") or not Path(fbx_path).is_file():
        tail = "\n".join(((run.stderr or "") + (run.stdout or "")).strip().splitlines()[-6:])
        raise RuntimeError(f"Blender could not write the FBX (exit code {run.returncode}):\n{tail}")


def save_fbx_animation(frames, path: str | Path, *, blender: str | Path, frame_time: float = 1.0 / 30.0,
                       root_positions=None, skeleton: str = DEFAULT_SKELETON, runner=subprocess.run) -> None:
    """``frames`` (as bvh_export.build_bvh_animation) as an FBX, via a glTF and Blender."""
    with tempfile.TemporaryDirectory(prefix="posecraft_fbx_") as work:
        glb = Path(work) / f"{Path(path).stem}.glb"
        glb.write_bytes(build_gltf_animation(frames, frame_time=frame_time, root_positions=root_positions,
                                             name=Path(path).stem, skeleton=skeleton))
        glb_to_fbx(glb, path, blender, runner=runner, fps=max(1, round(1.0 / frame_time)))


def save_fbx_pose(bone_rotations, path: str | Path, *, blender: str | Path, root_position=(0.0, 0.0, 0.0),
                  skeleton: str = DEFAULT_SKELETON, runner=subprocess.run) -> None:
    save_fbx_animation([bone_rotations], path, blender=blender, root_positions=[tuple(root_position)],
                       skeleton=skeleton, runner=runner)

"""F17: the pose library becomes Blender Pose Library assets.

The bundle (PoseLibrary.export_blender_bundle) is checked here; the asset creation
runs in a REAL headless Blender when POSECRAFT_BLENDER points at blender.exe (like
tests/test_real_blender.py): the add-on is installed into a temporary user folder,
a PoseCraft-shaped armature is built, and create_library_assets() must make one
asset per pose, tagged, with a preview, leaving the armature's own pose alone."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pose_craft import pose_library as pl
from pose_craft.export import live_link
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_gui_flows import _person  # noqa: E402
from test_retarget_frames import _articulated_pose  # noqa: E402

BLENDER = os.environ.get("POSECRAFT_BLENDER", "")
ADDON = Path(__file__).resolve().parents[1] / "pose_craft" / "export" / "blender_addon" / "import_pose.py"
E2B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _library(tmp_path) -> pl.PoseLibrary:
    library = pl.PoseLibrary(tmp_path / "library")
    photo = np.zeros((480, 640, 3), np.uint8)
    photo[:, :320] = (30, 120, 220)
    library.add_person(_person(_articulated_pose()), name="Warrior", tags="yoga, standing", image_bgr=photo)
    mirrored = library.mirrored_copy(library.entries()[0].id)
    library.update(mirrored.id, name="Warrior mirrored")
    return library


def test_the_bundle_holds_every_pose_and_its_thumbnail(tmp_path):
    library = _library(tmp_path)
    bundle = library.export_blender_bundle(tmp_path / "bundle")
    data = json.loads(bundle.read_text(encoding="utf-8"))
    assert data["format"] == live_link.LIBRARY_FORMAT_NAME == "pose_craft_library"
    names = sorted(pose["name"] for pose in data["poses"])
    assert names == ["Warrior", "Warrior mirrored"]
    first = next(pose for pose in data["poses"] if pose["name"] == "Warrior")
    assert first["tags"] == ["yoga", "standing"] and first["document"]["format"] == "pose_craft_pose"
    assert (bundle.parent / first["thumbnail"]).is_file()
    only = library.export_blender_bundle(tmp_path / "one", [library.entries()[0].id])
    assert len(json.loads(only.read_text(encoding="utf-8"))["poses"]) == 1
    with pytest.raises(pl.LibraryError):
        pl.PoseLibrary(tmp_path / "empty").export_blender_bundle(tmp_path / "none")


def test_the_live_link_library_request_carries_only_the_path(monkeypatch):
    sent = {}
    monkeypatch.setattr(live_link, "_exchange", lambda request, host, port, timeout: sent.update(request) or {"ok": True})
    live_link.send_library_request("C:/x/posecraft_library.json")
    assert sent == {"format": "pose_craft_library", "path": "C:/x/posecraft_library.json",
                    "live_link_protocol": live_link.LIBRARY_REQUEST_PROTOCOL}
    assert live_link.LIBRARY_REQUEST_PROTOCOL == live_link.PROTOCOL_VERSION == 4


_SCRIPT = r"""
import json, sys
from pathlib import Path
import addon_utils, bpy
from mathutils import Vector

addon_path, rig_path, bundle, out = sys.argv[sys.argv.index("--") + 1:][:4]
bpy.ops.preferences.addon_install(filepath=addon_path, overwrite=True)
assert addon_utils.enable("import_pose", default_set=True) is not None
addon = sys.modules["import_pose"]
bpy.ops.wm.read_homefile(use_empty=True)
spec = json.loads(Path(rig_path).read_text())
obj = bpy.data.objects.new("PC", bpy.data.armatures.new("PC"))
bpy.context.scene.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode="EDIT")
for name, v in spec.items():
    eb = obj.data.edit_bones.new(name)
    eb.head, eb.tail = Vector(v["head"]), Vector(v["tail"])
for name, v in spec.items():
    if v["parent"]:
        obj.data.edit_bones[name].parent = obj.data.edit_bones[v["parent"]]
bpy.ops.object.mode_set(mode="POSE")
obj.pose.bones["head"].rotation_quaternion = (0.9, 0.1, 0.0, 0.0)
before = list(obj.pose.bones["head"].rotation_quaternion)
result = addon.create_library_assets(obj, bundle, bpy.context.scene)
assets = [a for a in bpy.data.actions if a.asset_data is not None]
out_data = {
    "result": result,
    "assets": sorted(a.name for a in assets),
    "tags": sorted({t.name for a in assets for t in a.asset_data.tags}),
    "previews": [a.preview is not None for a in assets],
    "pose_kept": [round(x - y, 6) for x, y in zip(obj.pose.bones["head"].rotation_quaternion, before)],
}
Path(out).write_text(json.dumps(out_data))
"""


@pytest.mark.skipif(not (BLENDER and Path(BLENDER).is_file()), reason="set POSECRAFT_BLENDER to a blender executable")
def test_blender_makes_one_pose_asset_per_library_pose(tmp_path):
    bundle = _library(tmp_path).export_blender_bundle(tmp_path / "bundle")
    heads = sk.rest_head_positions()
    rig = {}
    for name in sk.ALL_BONE_NAMES:
        head = E2B @ heads[name] / 100.0
        tail = head + E2B @ sk.bone_rest_direction(name) * sk.bone_length(sk.BONE_LENGTHS_CM, name) / 100.0
        rig[name] = {"head": head.tolist(), "tail": tail.tolist(), "parent": sk.bone_parent(name)}
    rig_path = tmp_path / "rig.json"
    rig_path.write_text(json.dumps(rig))
    user = tmp_path / "blender_user"
    env = dict(os.environ, BLENDER_USER_RESOURCES=str(user), BLENDER_USER_CONFIG=str(user / "config"),
               BLENDER_USER_SCRIPTS=str(user / "scripts"), BLENDER_USER_EXTENSIONS=str(user / "extensions"),
               BLENDER_USER_DATAFILES=str(user / "datafiles"))
    out = tmp_path / "out.json"
    run = subprocess.run([BLENDER, "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr",
                          _SCRIPT, "--", str(ADDON), str(rig_path), str(bundle), str(out)],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, (run.stdout + run.stderr)[-3000:]
    data = json.loads(out.read_text())
    assert data["result"]["created"] == 2 and data["result"]["failed"] == []
    assert data["assets"] == ["Warrior", "Warrior mirrored"]
    assert {"PoseCraft", "yoga", "standing"} <= set(data["tags"])
    assert all(data["previews"])
    assert data["pose_kept"] == [0.0, 0.0, 0.0, 0.0]

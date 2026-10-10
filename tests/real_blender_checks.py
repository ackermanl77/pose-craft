"""Runs INSIDE a real Blender (headless), driven by tests/test_real_blender.py:

    blender --background --factory-startup --python-exit-code 1 --python real_blender_checks.py -- <work dir>

The work dir holds the documents the test generated with the app's own code
(pose_body.json, pose_hands.json, pose_crouch.json, anim.json, pose.bvh,
rig_posecraft.json); this writes results.json there. Not collected by pytest
(no test_ prefix) and only importable inside Blender.

The add-on is installed into the user script folder the test points
BLENDER_USER_RESOURCES/... at, never the real one. Live Link is exercised
for real: the server runs, the app's own live_link client sends from a
thread, and this script pumps the server's bpy.app.timers callback (timers
don't fire while a --background script runs).
"""
from __future__ import annotations

import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

import addon_utils
import bpy
from mathutils import Quaternion, Vector

WORK = Path(sys.argv[sys.argv.index("--") + 1])
REPO = Path(__file__).resolve().parents[1]
ADDON = REPO / "pose_craft" / "export" / "blender_addon" / "import_pose.py"
sys.path.insert(0, str(REPO))
from pose_craft.export import live_link  # noqa: E402 -- stdlib only, importable in Blender's Python

results: dict = {"blender": bpy.app.version_string}


def doc(name):
    return json.loads((WORK / name).read_text(encoding="utf-8"))


def reset():
    bpy.ops.wm.read_homefile(use_empty=True)


def armature(name, spec):
    """spec: bone -> (head, tail, parent)."""
    obj = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit = obj.data.edit_bones
    for bone, (head, tail, _parent) in spec.items():
        eb = edit.new(bone)
        eb.head, eb.tail = Vector(head), Vector(tail)
    for bone, (_head, _tail, parent) in spec.items():
        if parent:
            edit[bone].parent = edit[parent]
    bpy.ops.object.mode_set(mode="POSE")
    return obj


def max_error(obj, document, mapping):
    """Largest angle (degrees) between a measured rig bone (or chord: a
    (first, last) pair) and the document's target direction."""
    bpy.context.view_layer.update()
    worst = 0.0
    for pc, rig_bone in mapping.items():
        data = document["bones"].get(pc)
        if data is None:
            continue
        target = Quaternion(data["world_quaternion_wxyz"]) @ Vector(data["rest_direction"])
        first, last = rig_bone if isinstance(rig_bone, tuple) else (rig_bone, rig_bone)
        pose = obj.pose.bones
        posed = obj.matrix_world.to_3x3() @ (pose[last].tail - pose[first].head)
        worst = max(worst, math.degrees(posed.angle(target)))
    return worst


def key_count(obj):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return 0, []
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
        curves = list(bag.fcurves) if bag else []
    except ImportError:  # Blender < 4.4
        curves = list(ad.action.fcurves)
    frames = sorted({kp.co[0] for fc in curves for kp in fc.keyframe_points})
    return sum(len(fc.keyframe_points) for fc in curves), frames


# --- 1. install / register ------------------------------------------------------------
bpy.ops.preferences.addon_install(filepath=str(ADDON), overwrite=True)
assert addon_utils.enable("import_pose", default_set=True) is not None
addon = sys.modules["import_pose"]
results["registered"] = hasattr(bpy.types, "POSECRAFT_PT_panel") and hasattr(bpy.types.Scene, "posecraft_target")

rig_spec = doc("rig_posecraft.json")
body, hands = doc("pose_body.json"), doc("pose_hands.json")
body_names = list(body["bones"])
identity = {name: name for name in rig_spec}

# --- 2a. PoseCraft-shaped rig: both solves, rotation modes, keys, floor ----------------------
for match in (True, False):
    reset()
    obj = armature("PC", {n: (v["head"], v["tail"], v["parent"]) for n, v in rig_spec.items()})
    before = max_error(obj, hands, identity)  # the rest pose is far from the targets
    applied, unmatched, low = addon.apply_pose_document(obj, hands, match_rig_rest=match)
    results[f"simple_match_{match}"] = {"applied": len(applied), "unmatched": unmatched,
                                        "max_err": max_error(obj, hands, identity), "before": before}
reset()
obj = armature("PC", {n: (v["head"], v["tail"], v["parent"]) for n, v in rig_spec.items() if n in body_names})
for pose_bone in obj.pose.bones:
    pose_bone.rotation_mode = "ZXY"
scene = bpy.context.scene
scene.posecraft_insert_keyframes, scene.posecraft_frame_step = True, 5
scene.frame_set(1)
operator = [bpy.ops.posecraft.load_pose(filepath=str(WORK / name)) for name in ("pose_body.json", "pose_crouch.json")]
keys, frames = key_count(obj)
results["euler_rig"] = {
    "operator": [sorted(r) for r in operator], "modes_kept": all(pb.rotation_mode == "ZXY" for pb in obj.pose.bones),
    "keys": keys, "key_frames": frames, "frame_after": scene.frame_current,
    "max_err": max_error(obj, doc("pose_crouch.json"), {n: n for n in body_names}),
}
reset()
obj = armature("PC", {n: (v["head"], v["tail"], v["parent"]) for n, v in rig_spec.items() if n in body_names})
addon.apply_pose_document(obj, doc("pose_crouch.json"), floor_contact=True)
bpy.context.view_layer.update()
results["floor"] = {
    "rest_min": min(min(b.head_local.z, b.tail_local.z) for b in obj.data.bones),
    "posed_min": min(min(pb.head.z, pb.tail.z) for pb in obj.pose.bones),
}

# --- 2b. Mixamo-style A-pose rig (hips, clavicles, 3 spine bones, fingers) -------------------
P, spec = "mixamorig:", {}


def add(name, head, tail, parent):
    spec[P + name] = (tuple(head), tuple(tail), P + parent if parent else None)


add("Hips", (0, 0, 1.0), (0, 0, 1.1), None)
add("Spine", (0, 0, 1.1), (0, 0.01, 1.22), "Hips")
add("Spine1", (0, 0.01, 1.22), (0, 0, 1.34), "Spine")
add("Spine2", (0, 0, 1.34), (0, 0.02, 1.48), "Spine1")
add("Neck", (0, 0.02, 1.48), (0, 0, 1.58), "Spine2")
add("Head", (0, 0, 1.58), (0, 0, 1.78), "Neck")
for side, sx in (("Left", 1), ("Right", -1)):
    shoulder, down = Vector((sx * 0.17, 0.03, 1.42)), Vector((sx, 0, -1)).normalized()
    elbow, wrist = shoulder + down * 0.28, shoulder + down * 0.53
    add(f"{side}Shoulder", (sx * 0.03, 0.02, 1.44), shoulder, "Spine2")
    add(f"{side}Arm", shoulder, elbow, f"{side}Shoulder")
    add(f"{side}ForeArm", elbow, wrist, f"{side}Arm")
    add(f"{side}Hand", wrist, wrist + down * 0.09, f"{side}ForeArm")
    for i, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
        point = wrist + down * (0.03 if i == 0 else 0.09) + Vector((0, -0.03 + 0.015 * i, 0))
        direction = (down + Vector((0, -0.6, 0))).normalized() if i == 0 else down
        parent = f"{side}Hand"
        for segment in (1, 2, 3):
            add(f"{side}Hand{finger}{segment}", point, point + direction * 0.025, parent)
            parent, point = f"{side}Hand{finger}{segment}", point + direction * 0.025
    hip, knee, ankle = Vector((sx * 0.09, 0, 0.98)), Vector((sx * 0.1, -0.01, 0.53)), Vector((sx * 0.1, 0.02, 0.08))
    add(f"{side}UpLeg", hip, knee, "Hips")
    add(f"{side}Leg", knee, ankle, f"{side}UpLeg")
    add(f"{side}Foot", ankle, (sx * 0.1, -0.12, 0.02), f"{side}Leg")
reset()
obj = armature("Mixamo", spec)
obj.location = (0.3, 0.2, 0.0)
applied, unmatched, _low = addon.apply_pose_document(obj, hands)
mapping = {pc: P + name for pc, name in {**addon.MIXAMO_BONE_NAMES, **addon.MIXAMO_FINGER_BONE_NAMES}.items()}
mapping["spine"] = P + "Spine2"  # the chain's last bone carries the full rotation
results["mixamo"] = {"applied": len(applied), "unmatched": unmatched, "max_err": max_error(obj, hands, mapping)}

# --- 2c. a generated Rigify rig ----------------------------------------------------------------
results["rigify_available"] = addon_utils.enable("rigify", default_set=True) is not None
if results["rigify_available"]:
    reset()
    bpy.ops.object.armature_human_metarig_add()
    metarig = bpy.context.object
    bpy.ops.pose.rigify_generate()
    rig = bpy.context.object
    metarig.hide_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    applied, unmatched, _low = addon.apply_pose_document(rig, hands, pose_controls=True)
    org = {"spine": "ORG-spine.003", "neck": ("ORG-spine.004", "ORG-spine.005"), "head": "ORG-spine.006"}
    org.update({name: "ORG-" + name for name in rig_spec if name not in org})
    results["rigify"] = {
        "applied": len(applied), "unmatched": unmatched, "spine_chain": [n for n in applied if n.startswith("spine_fk")],
        "ik_fk": [rig.pose.bones[h]["IK_FK"] for h in ("upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R")],
        "max_err": max_error(rig, hands, org),
    }

# --- 3. the app's BVH through Blender's own importer --------------------------------------------
results["bvh_available"] = addon_utils.enable("io_anim_bvh", default_set=True) is not None
if results["bvh_available"]:
    reset()
    bpy.ops.import_anim.bvh(filepath=str(WORK / "pose.bvh"), global_scale=0.01, update_scene_fps=True)
    obj = bpy.context.object
    bpy.context.scene.frame_set(1)
    limbs = {n: n for n in body_names if n not in ("spine",)}
    results["bvh"] = {
        "bones": len(obj.data.bones), "keys": key_count(obj)[0],
        "root_z": (obj.matrix_world @ obj.pose.bones["spine"].head).z,
        "lowest_z": min(min((obj.matrix_world @ pb.head).z, (obj.matrix_world @ pb.tail).z) for pb in obj.pose.bones),
        "max_err": max_error(obj, body, limbs),
    }

# --- 4. Live Link: a real server, the app's own client --------------------------------------------
reset()
obj = armature("PC", {n: (v["head"], v["tail"], v["parent"]) for n, v in rig_spec.items() if n in body_names})
scene = bpy.context.scene
scene.render.fps = 30
probe = socket.socket()
probe.bind(("127.0.0.1", 0))
port = probe.getsockname()[1]
probe.close()
scene.posecraft_live_link_port = port
bpy.ops.posecraft.toggle_live_link()
bpy.ops.ed.undo_push(message="baseline")
other = bpy.data.objects.new("Other", None)  # selecting something else must not matter
scene.collection.objects.link(other)
bpy.context.view_layer.objects.active = other


def send(function, document):
    box = {}

    def run():
        try:
            box["reply"] = function(document, port=port)
        except Exception as exc:  # noqa: BLE001 - reported to the test
            box["error"] = repr(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 60.0
    while thread.is_alive() and time.monotonic() < deadline:
        addon._poll_live_link_server()  # what bpy.app.timers would call
        time.sleep(0.01)
    thread.join(5.0)
    return box


pose_reply = send(live_link.send_pose_document, body)
pose_err = max_error(bpy.data.objects["PC"], body, {n: n for n in body_names})
scene.frame_set(1)
anim = doc("anim.json")
anim_reply = send(live_link.send_animation_document, anim)
obj = bpy.data.objects["PC"]
keys, frames = key_count(obj)
first = anim["frames"][0]["frame_index"]
anim_err = 0.0
for frame in anim["frames"]:
    scene.frame_set(1 + frame["frame_index"] - first)
    anim_err = max(anim_err, max_error(obj, frame["pose"], {n: n for n in body_names}))
scene.frame_set(1)
bpy.ops.ed.undo()  # one step: the whole animation
undone_keys = key_count(bpy.data.objects["PC"])[0]
results["live_link"] = {
    "pose_reply": pose_reply, "pose_err": pose_err, "anim_reply": anim_reply, "anim_err": anim_err,
    "keys": keys, "key_frames": [frames[0], frames[-1], len(frames)] if frames else None,
    "keys_after_one_undo": undone_keys,
    "timer_while_running": bpy.app.timers.is_registered(addon._poll_live_link_server),
}

# --- 4b. A 30 fps clip into a 24 fps scene, with root motion (add-on 0.8.0) ----------------------
reset()
obj = armature("PC", {n: (v["head"], v["tail"], v["parent"]) for n, v in rig_spec.items() if n in body_names})
scene = bpy.context.scene
scene.render.fps = 24
scene.posecraft_target = obj
scene.posecraft_root_motion = True
scene.frame_set(1)
resampled_reply = send(live_link.send_animation_document, doc("anim24.json"))
obj = bpy.data.objects["PC"]
_keys, resampled_frames = key_count(obj)
root_bone = next(pb for pb in obj.pose.bones if pb.parent is None)
scene.frame_set(1)
start_head = (obj.matrix_world @ root_bone.head).copy()
scene.frame_set(36)
end_head = obj.matrix_world @ root_bone.head
results["resampled"] = {
    "reply": resampled_reply.get("reply"), "error": resampled_reply.get("error"),
    "key_frames": [resampled_frames[0], resampled_frames[-1], len(resampled_frames)] if resampled_frames else None,
    "travel": [end_head[i] - start_head[i] for i in range(3)],
}

# --- 1 again: unregister stops the server ------------------------------------------------------
addon_utils.disable("import_pose", default_set=True)
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", port))
    port_freed = True
except OSError:
    port_freed = False
finally:
    probe.close()
results["unregistered"] = {
    "classes_gone": not hasattr(bpy.types, "POSECRAFT_PT_panel") and not hasattr(bpy.types.Scene, "posecraft_target"),
    "port_freed": port_freed,
    "timer_gone": not bpy.app.timers.is_registered(addon._poll_live_link_server),
}
(WORK / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

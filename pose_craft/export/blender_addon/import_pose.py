"""PoseCraft Pose Importer -- a Blender add-on.

Install: Blender > Edit > Preferences > Add-ons > Install..., pick this file, enable
"PoseCraft Pose Importer". A "PoseCraft" tab then appears in the 3D viewport sidebar (N-panel).

Usage: select an armature, enter Pose Mode, open the PoseCraft tab, click
"Load Pose JSON" and pick a .json file exported by the PoseCraft app.

Bone name matching is heuristic: for each bone name in the pose file, this tries
an exact match against the armature's pose bones first, then common Rigify-style
prefixes/suffixes (DEF-, ORG-, ".fk", ".ik"). Bones it can't match are reported
in the Blender status bar / console rather than silently skipped -- expect to
need manual correction for some limbs, the same caveat PoseForge gives for DAZ.
"""
from __future__ import annotations

import json

import bpy
from mathutils import Quaternion

bl_info = {
    "name": "PoseCraft Pose Importer",
    "author": "PoseCraft",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PoseCraft",
    "description": "Applies a PoseCraft .json pose export to an armature's pose bones",
    "category": "Animation",
}

NAME_CANDIDATES_TEMPLATES = [
    "{name}",
    "DEF-{name}",
    "ORG-{name}",
    "{name}_fk",
    "{name}.fk",
    "{name}_ik",
    "{name}.ik",
]


def _resolve_bone_name(pose_bones, exported_name: str) -> str | None:
    for template in NAME_CANDIDATES_TEMPLATES:
        candidate = template.format(name=exported_name)
        if candidate in pose_bones:
            return candidate
    return None


def apply_pose_document(armature_obj: bpy.types.Object, document: dict) -> tuple[list[str], list[str]]:
    """Returns (applied_bone_names, unmatched_bone_names)."""
    if document.get("format") != "pose_craft_pose":
        raise ValueError("Not a PoseCraft pose file")

    pose_bones = armature_obj.pose.bones
    applied, unmatched = [], []

    for exported_name, data in document["bones"].items():
        bone_name = _resolve_bone_name(pose_bones, exported_name)
        if bone_name is None:
            unmatched.append(exported_name)
            continue

        pose_bone = pose_bones[bone_name]
        pose_bone.rotation_mode = "QUATERNION"
        w, x, y, z = data["quaternion_wxyz"]
        pose_bone.rotation_quaternion = Quaternion((w, x, y, z))
        applied.append(bone_name)

    return applied, unmatched


class POSECRAFT_OT_load_pose(bpy.types.Operator):
    bl_idname = "posecraft.load_pose"
    bl_label = "Load Pose JSON"
    bl_description = "Apply a PoseCraft .json pose export to the active armature"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        armature_obj = context.active_object
        if armature_obj is None or armature_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature first")
            return {"CANCELLED"}

        with open(self.filepath, "r", encoding="utf-8") as f:
            document = json.load(f)

        try:
            applied, unmatched = apply_pose_document(armature_obj, document)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        message = f"Applied {len(applied)} bone(s)"
        if unmatched:
            message += f", {len(unmatched)} unmatched: {', '.join(unmatched)}"
            self.report({"WARNING"}, message)
        else:
            self.report({"INFO"}, message)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class POSECRAFT_PT_panel(bpy.types.Panel):
    bl_idname = "POSECRAFT_PT_panel"
    bl_label = "PoseCraft"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PoseCraft"

    def draw(self, context):
        layout = self.layout
        layout.operator(POSECRAFT_OT_load_pose.bl_idname, icon="ARMATURE_DATA")
        active = context.active_object
        if active is None or active.type != "ARMATURE":
            layout.label(text="Select an armature", icon="ERROR")


CLASSES = (POSECRAFT_OT_load_pose, POSECRAFT_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

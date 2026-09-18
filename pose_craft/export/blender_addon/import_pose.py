"""PoseCraft Pose Importer -- a Blender add-on.

Install: Blender > Edit > Preferences > Add-ons > Install..., pick this file, enable
"PoseCraft Pose Importer". A "PoseCraft" tab then appears in the 3D viewport sidebar (N-panel).

Two ways to get a pose onto an armature:
- **Load Pose JSON**: select an armature, click "Load Pose JSON", pick a .json
  file exported by the PoseCraft app.
- **Live Link**: click "Start Live Link" once (keeps an armature selected as the
  target), then in the PoseCraft app use "Send to Blender" after each solve --
  no file dialog needed for repeated tries. Live Link listens on
  127.0.0.1:8642 (see pose_craft/export/live_link.py on the app side, which
  must agree on this port since the add-on can't import that package).

Bone name matching is heuristic: for each bone name in the pose file, this tries
an exact match against the armature's pose bones first, then common Rigify-style
prefixes/suffixes (DEF-, ORG-, ".fk", ".ik"). Bones it can't match are reported
in the Blender status bar / console rather than silently skipped, and bones the
PoseCraft app flagged "low_confidence" (likely occluded in the source photo) are
called out separately -- expect to need manual correction for those, the same
caveat PoseForge gives for DAZ.

NOTE: this file has not been loaded into a real Blender instance or tested
against an actual Rigify-generated armature -- no Blender install exists in the
dev environment this was written in. See docs/PROJECT_STATUS.md's Known Issues.
"""
from __future__ import annotations

import json
import socket

import bpy
from mathutils import Quaternion

bl_info = {
    "name": "PoseCraft Pose Importer",
    "author": "PoseCraft",
    "version": (0, 2, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PoseCraft",
    "description": "Applies a PoseCraft .json pose export (file or Live Link) to an armature's pose bones",
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

LIVE_LINK_HOST = "127.0.0.1"
LIVE_LINK_PORT = 8642
_POLL_INTERVAL_SECONDS = 0.2

# Live Link server state. Blender add-ons that need to survive across operator
# calls conventionally keep this at module scope rather than on a class instance.
_server_socket: socket.socket | None = None
_live_link_status = "Not started"


def _resolve_bone_name(pose_bones, exported_name: str) -> str | None:
    for template in NAME_CANDIDATES_TEMPLATES:
        candidate = template.format(name=exported_name)
        if candidate in pose_bones:
            return candidate
    return None


def apply_pose_document(armature_obj: bpy.types.Object, document: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (applied_bone_names, unmatched_bone_names, low_confidence_bone_names)."""
    if document.get("format") != "pose_craft_pose":
        raise ValueError("Not a PoseCraft pose file")

    pose_bones = armature_obj.pose.bones
    applied, unmatched, low_confidence = [], [], []

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
        if data.get("low_confidence"):
            low_confidence.append(bone_name)

    return applied, unmatched, low_confidence


def _summarize(applied: list[str], unmatched: list[str], low_confidence: list[str]) -> tuple[str, str]:
    """Returns (message, report_level) for bpy.types.Operator.report / status display."""
    message = f"Applied {len(applied)} bone(s)"
    level = "INFO"
    if low_confidence:
        message += f"; {len(low_confidence)} low-confidence (check manually): {', '.join(low_confidence)}"
        level = "WARNING"
    if unmatched:
        message += f"; {len(unmatched)} unmatched: {', '.join(unmatched)}"
        level = "WARNING"
    return message, level


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
            applied, unmatched, low_confidence = apply_pose_document(armature_obj, document)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        message, level = _summarize(applied, unmatched, low_confidence)
        self.report({level}, message)
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before the expected data arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _poll_live_link_server():
    """`bpy.app.timers` callback: non-blocking accept, then a short blocking read
    of one length-prefixed JSON message (small and localhost-fast, so this won't
    noticeably stall Blender's UI). Returning None stops the timer; any other
    return value is the delay in seconds until the next call.
    """
    global _live_link_status

    if _server_socket is None:
        return None

    try:
        conn, _addr = _server_socket.accept()
    except BlockingIOError:
        return _POLL_INTERVAL_SECONDS
    except OSError as exc:
        _live_link_status = f"Server error: {exc}"
        return None

    try:
        conn.settimeout(2.0)
        header = _recv_exact(conn, 4)
        length = int.from_bytes(header, "big")
        payload = _recv_exact(conn, length)
        document = json.loads(payload.decode("utf-8"))

        armature_obj = bpy.context.view_layer.objects.active
        if armature_obj is None or armature_obj.type != "ARMATURE":
            _live_link_status = "Received a pose but no armature is selected"
        else:
            applied, unmatched, low_confidence = apply_pose_document(armature_obj, document)
            _live_link_status, _level = _summarize(applied, unmatched, low_confidence)
    except (OSError, ValueError) as exc:
        _live_link_status = f"Error applying received pose: {exc}"
    finally:
        conn.close()

    return _POLL_INTERVAL_SECONDS


def _start_live_link_server() -> str | None:
    """Returns an error message on failure, or None on success."""
    global _server_socket, _live_link_status

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((LIVE_LINK_HOST, LIVE_LINK_PORT))
    except OSError as exc:
        server_socket.close()
        return f"Could not bind {LIVE_LINK_HOST}:{LIVE_LINK_PORT}: {exc}"

    server_socket.listen(1)
    server_socket.setblocking(False)
    _server_socket = server_socket
    _live_link_status = f"Listening on {LIVE_LINK_HOST}:{LIVE_LINK_PORT}"
    bpy.app.timers.register(_poll_live_link_server, first_interval=_POLL_INTERVAL_SECONDS, persistent=True)
    return None


def _stop_live_link_server() -> None:
    global _server_socket, _live_link_status
    if _server_socket is not None:
        _server_socket.close()
        _server_socket = None
    _live_link_status = "Stopped"


class POSECRAFT_OT_toggle_live_link(bpy.types.Operator):
    bl_idname = "posecraft.toggle_live_link"
    bl_label = "Start/Stop Live Link"
    bl_description = "Listen for poses sent from the PoseCraft app's 'Send to Blender' button"

    def execute(self, context):
        if _server_socket is None:
            error = _start_live_link_server()
            if error:
                self.report({"ERROR"}, error)
                return {"CANCELLED"}
            self.report({"INFO"}, _live_link_status)
        else:
            _stop_live_link_server()
            self.report({"INFO"}, "Live Link stopped")
        return {"FINISHED"}


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

        layout.separator()
        is_running = _server_socket is not None
        layout.operator(
            POSECRAFT_OT_toggle_live_link.bl_idname,
            text="Stop Live Link" if is_running else "Start Live Link",
            icon="PAUSE" if is_running else "PLAY",
        )
        layout.label(text=_live_link_status)


CLASSES = (POSECRAFT_OT_load_pose, POSECRAFT_OT_toggle_live_link, POSECRAFT_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    _stop_live_link_server()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

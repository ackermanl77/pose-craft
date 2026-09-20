"""PoseCraft Pose Importer -- a Blender add-on.

Install: Blender > Edit > Preferences > Add-ons > Install..., pick this file, enable
"PoseCraft Pose Importer". A "PoseCraft" tab then appears in the 3D viewport sidebar (N-panel).

Two ways to get a pose onto an armature:
- **Load Pose JSON**: select an armature, click "Load Pose JSON", pick a .json
  file exported by the PoseCraft app.
- **Live Link**: set the port (must match the PoseCraft app's "Blender host/port"
  fields -- the app defaults to 8642, same as this add-on's default), click
  "Start Live Link" once (keeps an armature selected as the target), then in
  the PoseCraft app use "Send to Blender" after each solve -- no file dialog
  needed for repeated tries. Always listens on 127.0.0.1 (localhost) only --
  there's no UI for binding to all interfaces, since that would expose a
  socket that applies arbitrary bone rotations on request to the network with
  no authentication, which needs to be a deliberate choice, not a default.
  Change the app's "Blender host" field only if PoseCraft and Blender are
  proven to already talk over some other configured loopback/tunnel; it does
  not make this add-on listen on a different interface.

Bone name matching, for either path above: PoseCraft's 15 bone names (spine,
neck, head, upper_arm.L/R, forearm.L/R, hand.L/R, thigh.L/R, shin.L/R,
foot.L/R -- see POSECRAFT_BONE_NAMES, which mirrors
pose_craft/retarget/skeleton.py::BONE_NAMES) don't necessarily match your
rig's own bone names. Two ways to bridge that, checked in this order:

1. **Manual Bone Mapping** (the "PoseCraft" panel's "Bone Mapping" box):
   click "Initialize/Refresh" once per armature to get one row per PoseCraft
   bone name, then either click "Auto-Guess Unset" to fill blanks with the
   heuristic below, or pick each target bone yourself from the searchable
   dropdown. Saved on the armature object (persists with the .blend file).
   An explicit row always wins, even if it points at a bone that's since been
   renamed or deleted -- that's reported as unmatched rather than silently
   falling back to a guess you didn't ask for.
2. **Automatic name-guessing** (used for any PoseCraft bone name with no
   manual mapping row, or a blank one): tries the exact name, then common
   Rigify-style prefixes/suffixes (DEF-, ORG-, ".fk", ".ik") -- see
   NAME_CANDIDATES_TEMPLATES.

Bones that still can't be resolved either way are reported in the Blender
status bar / console rather than silently skipped, and bones the PoseCraft
app flagged "low_confidence" (likely occluded in the source photo) are called
out separately -- expect to need manual correction for those, the same
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
    "version": (0, 3, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PoseCraft",
    "description": "Applies a PoseCraft .json pose export (file or Live Link) to an armature's pose bones",
    "category": "Animation",
}

# Mirrors pose_craft/retarget/skeleton.py::BONE_NAMES -- kept as a plain literal
# here since this add-on runs inside Blender's own Python and can't import that
# package. If the app's bone list ever changes, update this list too;
# "Initialize/Refresh" (POSECRAFT_OT_init_bone_mapping) will pick up the new
# set on next use, dropping stale rows and adding blank ones for new names.
POSECRAFT_BONE_NAMES = [
    "spine", "neck", "head",
    "upper_arm.L", "forearm.L", "hand.L",
    "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L",
    "thigh.R", "shin.R", "foot.R",
]

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
DEFAULT_LIVE_LINK_PORT = 8642
_POLL_INTERVAL_SECONDS = 0.2

# Live Link server state. Blender add-ons that need to survive across operator
# calls conventionally keep this at module scope rather than on a class instance.
_server_socket: socket.socket | None = None
_live_link_status = "Not started"


def _guess_bone_name(pose_bones, exported_name: str) -> str | None:
    """Tries common Rigify-style naming patterns against the armature's actual
    pose bones. This is the fallback `_resolve_bone_name` uses when no manual
    mapping row exists (or its row's target bone is blank) for a given
    PoseCraft bone name -- see POSECRAFT_BoneMapping.
    """
    for template in NAME_CANDIDATES_TEMPLATES:
        candidate = template.format(name=exported_name)
        if candidate in pose_bones:
            return candidate
    return None


def _resolve_bone_name(armature_obj: bpy.types.Object, exported_name: str) -> str | None:
    pose_bones = armature_obj.pose.bones

    for entry in armature_obj.posecraft_bone_mapping:
        if entry.posecraft_name == exported_name and entry.target_bone:
            # An explicit manual mapping exists -- honor it exactly. If the named
            # bone no longer exists (renamed/deleted since the mapping was set),
            # report unmatched rather than silently falling back to a guess the
            # user didn't ask for.
            return entry.target_bone if entry.target_bone in pose_bones else None

    return _guess_bone_name(pose_bones, exported_name)


def apply_pose_document(armature_obj: bpy.types.Object, document: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (applied_bone_names, unmatched_bone_names, low_confidence_bone_names)."""
    if document.get("format") != "pose_craft_pose":
        raise ValueError("Not a PoseCraft pose file")

    pose_bones = armature_obj.pose.bones
    applied, unmatched, low_confidence = [], [], []

    for exported_name, data in document["bones"].items():
        bone_name = _resolve_bone_name(armature_obj, exported_name)
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


class POSECRAFT_BoneMapping(bpy.types.PropertyGroup):
    """One row of the manual bone-mapping list: a fixed PoseCraft bone name
    (see POSECRAFT_BONE_NAMES) paired with the target bone the user picked on
    their own armature. Stored on the armature Object itself (posecraft_bone_mapping,
    registered in register() below) so it persists with the .blend file.
    """

    posecraft_name: bpy.props.StringProperty(name="PoseCraft Bone")
    target_bone: bpy.props.StringProperty(name="Target Bone")


class POSECRAFT_UL_bone_mapping(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.posecraft_name)
        # `data` here is the armature Object passed to template_list() below, so
        # data.data is its Armature datablock -- prop_search's standard "pick an
        # existing bone by name, with autocomplete" idiom.
        row.prop_search(item, "target_bone", data.data, "bones", text="", icon="BONE_DATA")


class POSECRAFT_OT_init_bone_mapping(bpy.types.Operator):
    bl_idname = "posecraft.init_bone_mapping"
    bl_label = "Initialize/Refresh Bone Mapping"
    bl_description = (
        "Set up one row per PoseCraft bone name for this armature, keeping any bone "
        "you've already assigned and adding blank rows for the rest"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature_obj = context.active_object
        if armature_obj is None or armature_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature first")
            return {"CANCELLED"}

        existing = {entry.posecraft_name: entry.target_bone for entry in armature_obj.posecraft_bone_mapping}
        armature_obj.posecraft_bone_mapping.clear()
        for name in POSECRAFT_BONE_NAMES:
            entry = armature_obj.posecraft_bone_mapping.add()
            entry.posecraft_name = name
            entry.target_bone = existing.get(name, "")

        self.report({"INFO"}, f"{len(POSECRAFT_BONE_NAMES)} bone mapping row(s) ready")
        return {"FINISHED"}


class POSECRAFT_OT_auto_guess_bone_mapping(bpy.types.Operator):
    bl_idname = "posecraft.auto_guess_bone_mapping"
    bl_label = "Auto-Guess Unset Bone Mapping"
    bl_description = "Fill in any blank mapping row using the same name-guessing Load Pose JSON falls back on"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature_obj = context.active_object
        if armature_obj is None or armature_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature first")
            return {"CANCELLED"}

        pose_bones = armature_obj.pose.bones
        filled = 0
        for entry in armature_obj.posecraft_bone_mapping:
            if entry.target_bone:
                continue
            guess = _guess_bone_name(pose_bones, entry.posecraft_name)
            if guess is not None:
                entry.target_bone = guess
                filled += 1

        self.report({"INFO"}, f"Filled {filled} row(s) from name-guessing")
        return {"FINISHED"}


class POSECRAFT_OT_clear_bone_mapping(bpy.types.Operator):
    bl_idname = "posecraft.clear_bone_mapping"
    bl_label = "Clear Bone Mapping"
    bl_description = "Blank every assigned target bone (keeps the row list itself)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        armature_obj = context.active_object
        if armature_obj is None or armature_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature first")
            return {"CANCELLED"}

        for entry in armature_obj.posecraft_bone_mapping:
            entry.target_bone = ""

        self.report({"INFO"}, "Cleared bone mapping")
        return {"FINISHED"}


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


def _start_live_link_server(port: int) -> str | None:
    """Returns an error message on failure, or None on success."""
    global _server_socket, _live_link_status

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((LIVE_LINK_HOST, port))
    except OSError as exc:
        server_socket.close()
        return f"Could not bind {LIVE_LINK_HOST}:{port}: {exc}"

    server_socket.listen(1)
    server_socket.setblocking(False)
    _server_socket = server_socket
    _live_link_status = f"Listening on {LIVE_LINK_HOST}:{port}"
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
            error = _start_live_link_server(context.scene.posecraft_live_link_port)
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
        else:
            box = layout.box()
            box.label(text="Bone Mapping (optional)", icon="BONE_DATA")
            row = box.row(align=True)
            row.operator(POSECRAFT_OT_init_bone_mapping.bl_idname, text="Initialize/Refresh")
            row.operator(POSECRAFT_OT_auto_guess_bone_mapping.bl_idname, text="Auto-Guess Unset")

            if len(active.posecraft_bone_mapping) > 0:
                box.template_list(
                    "POSECRAFT_UL_bone_mapping", "",
                    active, "posecraft_bone_mapping",
                    active, "posecraft_bone_mapping_active_index",
                    rows=6,
                )
                box.operator(POSECRAFT_OT_clear_bone_mapping.bl_idname, text="Clear All", icon="X")
            else:
                box.label(text="No manual mapping set -- using name-guessing only")

        layout.separator()
        is_running = _server_socket is not None
        row = layout.row()
        row.enabled = not is_running  # num_poses-style: the port is fixed once the socket is bound
        row.prop(context.scene, "posecraft_live_link_port")
        layout.operator(
            POSECRAFT_OT_toggle_live_link.bl_idname,
            text="Stop Live Link" if is_running else "Start Live Link",
            icon="PAUSE" if is_running else "PLAY",
        )
        layout.label(text=_live_link_status)


CLASSES = (
    POSECRAFT_OT_load_pose,
    POSECRAFT_BoneMapping,
    POSECRAFT_UL_bone_mapping,
    POSECRAFT_OT_init_bone_mapping,
    POSECRAFT_OT_auto_guess_bone_mapping,
    POSECRAFT_OT_clear_bone_mapping,
    POSECRAFT_OT_toggle_live_link,
    POSECRAFT_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.posecraft_live_link_port = bpy.props.IntProperty(
        name="Live Link Port",
        description="Must match the PoseCraft app's 'port' field next to its Send to Blender button",
        default=DEFAULT_LIVE_LINK_PORT,
        min=1,
        max=65535,
    )
    bpy.types.Object.posecraft_bone_mapping = bpy.props.CollectionProperty(type=POSECRAFT_BoneMapping)
    bpy.types.Object.posecraft_bone_mapping_active_index = bpy.props.IntProperty()


def unregister():
    _stop_live_link_server()
    del bpy.types.Object.posecraft_bone_mapping_active_index
    del bpy.types.Object.posecraft_bone_mapping
    del bpy.types.Scene.posecraft_live_link_port
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

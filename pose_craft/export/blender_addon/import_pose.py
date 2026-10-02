"""PoseCraft Pose Importer -- a Blender add-on.

Install: Blender > Edit > Preferences > Add-ons > Install..., pick this file, enable
"PoseCraft Pose Importer". A "PoseCraft" tab then appears in the 3D viewport sidebar (N-panel).

Two ways to get a pose onto an armature:
- **Load Pose JSON**: select an armature, click "Load Pose JSON", pick a .json
  file exported by the PoseCraft app.
- **Live Link**: set the port (must match the PoseCraft app's "Blender host/port"
  fields -- the app defaults to 8642, same as this add-on's default), select
  the armature and click "Start Live Link" once (it becomes the "Live Link
  target", so selecting other objects afterwards doesn't matter), then in
  the PoseCraft app use "Send to Blender" after each solve -- no file dialog
  needed for repeated tries. Each received pose is its own undo step. Reads
  never block Blender's UI: each timer tick takes whatever bytes have arrived,
  and a connection that hasn't delivered its whole message within 2 seconds
  is dropped. Always listens on 127.0.0.1 (localhost) only --
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
   manual mapping row, or a blank one): tries the exact name, then Rigify's
   control bones (upper_arm_fk.L, then upper_arm_ik.L), then its DEF-/ORG-
   bones as a last resort, then Mixamo's names (mixamorig:LeftArm, ...) -- see
   bone_name_candidates().

Rotations, two ways:

- **Match rig rest pose** (the panel checkbox, on by default; needs a format
  version 2 file): each mapped bone is aimed at the world-space target the
  file gives (PoseCraft's world rotation times its rest direction), measured
  against the rig's OWN rest direction and the actual posed parent -- see
  rig_pose_basis(). A-pose arms, sloped foot bones, a hips bone above the
  thighs (Mixamo) and clavicles between spine and arms all work; bones with no
  PoseCraft counterpart stay as they are and their children still land on
  target.
- Off (or a version 1 file): each bone's parent-relative rotation is converted
  into the bone's rest frame with armature_delta_to_basis() and applied as-is,
  which assumes the rig's rest pose is a T-pose shaped like PoseCraft's.

A mapping row can name a chain: "Extra Bones" lists more rig bones continuing
after its target (a Mixamo "Spine1, Spine2"), and the rotation is spread over
the chain (Match rig rest pose only). Mixamo spines are chained automatically.

Options in the panel: "Keep feet on the floor" moves the rig's root so the
pose stands on the rig's own floor (a crouch comes out lowered); "Insert
keyframes" (or Blender's auto-keying) keys every applied bone at the current
frame, and "Then advance" steps the frame forward for the next pose. Each
bone keeps its own rotation mode (Euler, axis-angle or quaternion).

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
import math
import socket
import time

import bpy
from mathutils import Quaternion, Vector

bl_info = {
    "name": "PoseCraft Pose Importer",
    "author": "PoseCraft",
    "version": (0, 5, 0),
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

# Mirrors pose_craft/export/blender_pose.py::FORMAT_VERSION, for the same
# reason POSECRAFT_BONE_NAMES mirrors BONE_NAMES above -- this add-on can't
# import that package. A document whose version is newer than this add-on
# understands is rejected rather than silently (mis)applied.
MAX_SUPPORTED_FORMAT_VERSION = 2

# Name-guessing templates, tried in order. {name} is the PoseCraft bone name
# ("upper_arm.L"); {base}/{side} split it into "upper_arm" / ".L" (side is ""
# for center bones like "spine"), because Rigify puts its _fk/_ik tag BEFORE
# the side suffix: upper_arm_fk.L, not upper_arm.L_fk. Control bones come
# first: a generated Rigify rig's DEF- and ORG- bones are driven by
# constraints from those controls, so posing them directly has no visible
# effect (or is overwritten on the next depsgraph update) -- they're only a
# last resort, for deform-only rigs that have nothing else.
NAME_CANDIDATES_TEMPLATES = [
    "{name}",
    "{base}_fk{side}",
    "{base}_ik{side}",
    "DEF-{name}",
    "ORG-{name}",
]

# Mixamo's names for PoseCraft's bones (a Mixamo "Left" bone is the
# character's own left, same as PoseCraft's .L), tried with each of
# MIXAMO_PREFIXES -- Mixamo exports use "mixamorig:", a second character
# imported into the same file often gets "mixamorig1:", and some pipelines
# strip the prefix entirely.
MIXAMO_BONE_NAMES = {
    "spine": "Spine", "neck": "Neck", "head": "Head",
    "upper_arm.L": "LeftArm", "forearm.L": "LeftForeArm", "hand.L": "LeftHand",
    "upper_arm.R": "RightArm", "forearm.R": "RightForeArm", "hand.R": "RightHand",
    "thigh.L": "LeftUpLeg", "shin.L": "LeftLeg", "foot.L": "LeftFoot",
    "thigh.R": "RightUpLeg", "shin.R": "RightLeg", "foot.R": "RightFoot",
}
MIXAMO_PREFIXES = ["mixamorig:", "mixamorig1:", ""]
# Mixamo's spine is three bones (Spine, Spine1, Spine2); PoseCraft's one spine
# bone is spread over all of them (see distribute_rotation).
MIXAMO_SPINE_CHAIN_EXTRAS = ["Spine1", "Spine2"]

LIVE_LINK_HOST = "127.0.0.1"
DEFAULT_LIVE_LINK_PORT = 8642
# Mirrors pose_craft/export/live_link.py::PROTOCOL_VERSION. Version 2: every
# request gets a length-prefixed JSON reply saying what was applied (see
# build_live_link_reply). A request without the field is from an older app.
LIVE_LINK_PROTOCOL_VERSION = 2
_POLL_INTERVAL_SECONDS = 0.1
# A 15-bone document is about 4 KB; guards against a bogus or hostile length
# header making the server buffer an unbounded read.
MAX_LIVE_LINK_PAYLOAD_BYTES = 262144  # 256 KB (a plain literal: the tests read it from the source)
# A connection has this long, from accept, to deliver its whole message before
# it's dropped. Reads never block: each timer tick takes whatever has arrived.
LIVE_LINK_MESSAGE_DEADLINE_SECONDS = 2.0

# Live Link server state. Blender add-ons that need to survive across operator
# calls conventionally keep this at module scope rather than on a class instance.
_server_socket: socket.socket | None = None
_live_link_status = "Not started"
# Accepted connections still delivering their message: socket -> LiveLinkReader.
_connections: dict = {}


def bone_name_candidates(exported_name: str) -> list[str]:
    """Every rig bone name `exported_name` might go by, most-preferred first:
    NAME_CANDIDATES_TEMPLATES, then MIXAMO_BONE_NAMES with each prefix. Pure
    (no bpy), so tests/test_blender_addon_source.py can run it outside Blender.
    """
    base, dot, side_letter = exported_name.rpartition(".")
    if dot and side_letter in ("L", "R"):
        side = "." + side_letter
    else:
        base, side = exported_name, ""
    candidates = [template.format(name=exported_name, base=base, side=side) for template in NAME_CANDIDATES_TEMPLATES]
    mixamo_name = MIXAMO_BONE_NAMES.get(exported_name)
    if mixamo_name is not None:
        candidates += [prefix + mixamo_name for prefix in MIXAMO_PREFIXES]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def armature_delta_to_basis(rest_rotation, delta):
    """Converts an exported rotation (parent-relative, expressed in ARMATURE
    axes) into a pose bone's rotation_quaternion (expressed in the bone's own
    rest frame): R^-1 @ delta @ R, with R = bone.matrix_local's rotation.

    Why: Blender evaluates a pose bone as
    pose = parent_pose @ (parent_rest^-1 @ rest) @ basis. Writing each pose as
    a world-space delta on top of its rest (pose = D @ rest) turns that into
    D_bone = D_parent @ (R @ basis @ R^-1), so setting basis = R^-1 @ delta @ R
    gives D_bone = D_parent @ delta -- exactly the parent-relative delta the
    file stores. Writing `delta` straight into rotation_quaternion (the old
    behavior) rotated about the wrong axes for any bone whose rest frame isn't
    aligned with the armature's (i.e. nearly every bone: a bone's local Y runs
    along its length). Duck-typed on `@` and `.inverted()`, so it works on
    mathutils Quaternions and on the test suite's stand-in.
    """
    return rest_rotation.inverted() @ delta @ rest_rotation


def bones_parents_first(parents: dict) -> list:
    """Bone names ordered so every parent comes before its children (a
    Blender armature's bone collection makes no such promise). `parents` maps
    name -> parent name or None. Pure (no bpy)."""
    depth = {}
    for name in parents:
        chain, current = [], name
        while current is not None and current not in depth and current not in chain:
            chain.append(current)
            current = parents.get(current)
        base = depth.get(current, -1) if current is not None else -1
        for offset, bone in enumerate(reversed(chain)):
            depth[bone] = base + 1 + offset
    names = list(parents)
    return sorted(names, key=lambda n: (depth[n], names.index(n)))


def rig_pose_basis(order, parents, rest, basis, deltas, inherit_rotation=None):
    """The rig-aware solve: new basis rotations (rotation_quaternion) for the
    bones in `deltas`, so each one's armature-space pose is exactly
    delta @ rest, whatever the rig's hierarchy.

    order: bone names, parents first (bones_parents_first). parents: name ->
    parent name or None. rest: name -> armature-space rest rotation
    (bone.matrix_local). basis: name -> current basis rotation for bones that
    keep theirs (missing = identity). deltas: name -> desired armature-space
    rotation delta for the mapped bones (PoseCraft's world rotation, after
    aligning the rig's rest direction onto PoseCraft's).

    Walks the hierarchy evaluating each bone's posed rotation the way Blender
    does (pose = parent_pose @ parent_rest^-1 @ rest @ basis), so a mapped bone
    is solved against its actual posed parent -- mapped or not (a Mixamo Hips,
    a clavicle) -- instead of assuming its parent is PoseCraft's parent.
    inherit_rotation: name -> the bone's "Inherit Rotation" flag (missing =
    True). A bone with it off ignores its parent's pose rotation (Blender
    evaluates it from its own rest orientation), so it's solved that way too.
    Duck-typed on `@` and `.inverted()` like armature_delta_to_basis.
    """
    inherit_rotation = inherit_rotation or {}
    pose, result = {}, {}
    for name in order:
        parent = parents.get(name)
        if parent is None or not inherit_rotation.get(name, True):
            chain = rest[name]
        else:
            chain = pose[parent] @ rest[parent].inverted() @ rest[name]
        if name in deltas:
            target = deltas[name] @ rest[name]
            result[name] = chain.inverted() @ target
            pose[name] = target
        else:
            current = basis.get(name)
            pose[name] = chain if current is None else chain @ current
    return result


def distribute_rotation(identity, delta, count: int) -> list:
    """Splits an armature-space rotation delta over a chain of `count` rig
    bones mapped to one PoseCraft bone (a Mixamo Spine/Spine1/Spine2): bone k
    (0 = the chain's base) gets slerp(identity, delta, (k + 1) / count), so the
    bend is spread evenly and the chain's last bone -- where the neck and arms
    attach -- ends at the full rotation, exactly as a single mapped bone
    would. Duck-typed on mathutils' Quaternion.slerp(other, factor)."""
    return [identity.slerp(delta, (k + 1) / count) for k in range(count)]


def parse_bone_chain(target_bone: str, extra_bones: str) -> list:
    """A mapping row's rig bones, base first: its target bone, then the
    comma-separated "extra bones" continuing the chain (blanks dropped)."""
    names = [target_bone.strip()] if target_bone and target_bone.strip() else []
    names += [name.strip() for name in (extra_bones or "").split(",") if name.strip()]
    return names


def mixamo_chain_extras(exported_name: str, base_bone: str, existing) -> list:
    """The extra spine bones a Mixamo rig chains after a guessed base bone
    (Spine -> Spine1, Spine2), when they exist in `existing` -- so a Mixamo
    spine bends along its three bones without a manual mapping."""
    if exported_name != "spine":
        return []
    for prefix in MIXAMO_PREFIXES:
        if base_bone == prefix + MIXAMO_BONE_NAMES["spine"]:
            return [prefix + extra for extra in MIXAMO_SPINE_CHAIN_EXTRAS if prefix + extra in existing]
    return []


def top_ancestor(name: str, parents: dict) -> str:
    """The root of `name`'s hierarchy (the bone to move to move the whole rig)."""
    seen = set()
    while parents.get(name) is not None and name not in seen:
        seen.add(name)
        name = parents[name]
    return name


def floor_contact_shift(rest_heights, posed_heights) -> float:
    """How far to raise the rig (armature-space Z) so its lowest posed point
    sits where its lowest rest point did -- the floor the rig was built
    standing on. Negative lowers a crouch's hips; zero for an empty list."""
    rest_heights, posed_heights = list(rest_heights), list(posed_heights)
    if not rest_heights or not posed_heights:
        return 0.0
    return min(rest_heights) - min(posed_heights)


def armature_translation_to_location(rest_rotation, translation):
    """An armature-space translation for a root pose bone, as the change to its
    `location` (which Blender applies in the bone's rest frame, before its
    basis rotation): R^-1 @ t. Duck-typed on `@` and `.inverted()`."""
    return rest_rotation.inverted() @ translation


ROTATION_MODES_EULER = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


def rotation_data_path(rotation_mode: str) -> str:
    """The pose bone property that holds its rotation in `rotation_mode`."""
    if rotation_mode in ROTATION_MODES_EULER:
        return "rotation_euler"
    if rotation_mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_quaternion"


def rotation_channel(rotation_mode: str, quaternion, previous_euler=None):
    """(data path, value) to write a basis rotation in the pose bone's OWN
    rotation mode instead of switching it to quaternions -- which silently
    broke Euler rigs and any existing Euler F-curves on those bones. Euler
    values are made compatible with `previous_euler` (no 360-degree flips
    between keys). Duck-typed on mathutils' Quaternion.to_euler(order,
    compatible) and .to_axis_angle()."""
    data_path = rotation_data_path(rotation_mode)
    if data_path == "rotation_euler":
        if previous_euler is None:
            return data_path, quaternion.to_euler(rotation_mode)
        return data_path, quaternion.to_euler(rotation_mode, previous_euler)
    if data_path == "rotation_axis_angle":
        axis, angle = quaternion.to_axis_angle()
        return data_path, (angle, axis[0], axis[1], axis[2])
    return data_path, quaternion


def pick_live_link_target(captured, active):
    """The armature a Live Link pose goes to: the one captured when Live Link
    started (so selecting a light or a mesh afterwards doesn't break every
    later send), else the active object if it's an armature, else None."""
    for candidate in (captured, active):
        if candidate is not None and getattr(candidate, "type", None) == "ARMATURE":
            return candidate
    return None


class LiveLinkReader:
    """Incremental framing for one Live Link connection, so the server never
    blocks Blender's UI thread: feed() whatever bytes a non-blocking recv
    returned; it returns the decoded document once the 4-byte length header
    and that many payload bytes have arrived (None until then). Raises
    ValueError for an oversized length or a payload that isn't JSON. A
    connection still incomplete at `deadline` (time.monotonic()) is expired()
    and gets dropped -- a stalled or stray client costs nothing. Pure (no bpy).
    """

    def __init__(self, deadline: float, max_bytes=None):
        self.deadline = deadline
        self.max_bytes = MAX_LIVE_LINK_PAYLOAD_BYTES if max_bytes is None else max_bytes
        self._buffer = bytearray()
        self._length = None

    def feed(self, data: bytes):
        self._buffer += data
        if self._length is None and len(self._buffer) >= 4:
            self._length = int.from_bytes(self._buffer[:4], "big")
            del self._buffer[:4]
            if self._length > self.max_bytes:
                raise ValueError(f"payload length {self._length} exceeds max of {self.max_bytes} bytes")
        if self._length is None or len(self._buffer) < self._length:
            return None
        try:
            return json.loads(bytes(self._buffer[: self._length]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed pose payload: {exc}") from None

    def expired(self, now: float) -> bool:
        return now > self.deadline


def build_live_link_reply(applied=(), requested=0, unmatched=(), low_confidence=(), armature=None, error=None) -> dict:
    """The reply the Live Link server sends back after each pose (protocol 2),
    read by the app's send_pose_document(). Pure (no bpy)."""
    return {
        "ok": error is None,
        "applied": list(applied),
        "requested": int(requested),
        "unmatched": list(unmatched),
        "low_confidence": list(low_confidence),
        "armature": armature,
        "addon_version": list(bl_info["version"]),
        "protocol": LIVE_LINK_PROTOCOL_VERSION,
        "error": error,
    }


def encode_live_link_message(message: dict) -> bytes:
    """4-byte big-endian length prefix + UTF-8 JSON, the Live Link framing."""
    payload = json.dumps(message).encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload


def live_link_request_error(document) -> str | None:
    """A reason to refuse a Live Link request before applying it, or None."""
    protocol = document.get("live_link_protocol") if isinstance(document, dict) else None
    if isinstance(protocol, int) and protocol > LIVE_LINK_PROTOCOL_VERSION:
        return (
            f"The PoseCraft app speaks Live Link protocol {protocol}, this add-on only up to "
            f"{LIVE_LINK_PROTOCOL_VERSION}; update the PoseCraft add-on."
        )
    return None


def _guess_bone_name(pose_bones, exported_name: str) -> str | None:
    """Tries common Rigify/Mixamo naming patterns (bone_name_candidates)
    against the armature's actual pose bones. This is the fallback
    `_resolve_bone_chain` uses when no manual mapping row exists (or its row's
    target bone is blank) for a given PoseCraft bone name -- see
    POSECRAFT_BoneMapping.
    """
    for candidate in bone_name_candidates(exported_name):
        if candidate in pose_bones:
            return candidate
    return None


def _resolve_bone_chain(armature_obj: bpy.types.Object, exported_name: str) -> list[str] | None:
    """The rig bones `exported_name` drives, base first (usually one; a spine
    chain can be several), or None if it can't be resolved."""
    pose_bones = armature_obj.pose.bones

    for entry in armature_obj.posecraft_bone_mapping:
        if entry.posecraft_name == exported_name and entry.target_bone:
            # An explicit manual mapping exists -- honor it exactly. If a named
            # bone no longer exists (renamed/deleted since the mapping was set),
            # report unmatched rather than silently falling back to a guess the
            # user didn't ask for.
            chain = parse_bone_chain(entry.target_bone, entry.extra_bones)
            return chain if all(name in pose_bones for name in chain) else None

    guess = _guess_bone_name(pose_bones, exported_name)
    if guess is None:
        return None
    return [guess] + mixamo_chain_extras(exported_name, guess, pose_bones)


def _validate_quaternion(exported_name: str, data: object) -> tuple[float, float, float, float]:
    if not isinstance(data, dict):
        raise ValueError(f"bone '{exported_name}': expected an object, got {type(data).__name__}")
    values = data.get("quaternion_wxyz")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"bone '{exported_name}': 'quaternion_wxyz' must be a 4-element list")
    try:
        w, x, y, z = (float(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"bone '{exported_name}': 'quaternion_wxyz' must contain 4 numbers") from None
    if not all(math.isfinite(v) for v in (w, x, y, z)):
        raise ValueError(f"bone '{exported_name}': 'quaternion_wxyz' must be finite, got {values}")
    return w, x, y, z


def _validate_vector(exported_name: str, data: dict, key: str, size: int) -> tuple[float, ...] | None:
    """An optional numeric list field of a bone entry (version 2 world targets):
    None when absent, ValueError when present but malformed."""
    values = data.get(key)
    if values is None:
        return None
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"bone '{exported_name}': '{key}' must be a {size}-element list")
    try:
        numbers = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"bone '{exported_name}': '{key}' must contain numbers") from None
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"bone '{exported_name}': '{key}' must be finite")
    return numbers


def _write_rotation(pose_bone, quaternion) -> str:
    """Sets a pose bone's basis rotation in its own rotation mode (see
    rotation_channel) and returns the data path written, for keyframing."""
    data_path, value = rotation_channel(pose_bone.rotation_mode, quaternion, pose_bone.rotation_euler.copy())
    setattr(pose_bone, data_path, value)
    return data_path


def _keep_feet_on_floor(armature_obj: bpy.types.Object, bone_names: list[str]) -> None:
    """Moves the rig's root bone up or down so the lowest point of the posed
    bones sits where the lowest point of their rest pose did (B6): a crouch
    or a sit comes out lowered instead of hanging at standing height."""
    if not bone_names:
        return
    bpy.context.view_layer.update()  # evaluate the new rotations
    pose_bones, bones = armature_obj.pose.bones, armature_obj.data.bones
    rest_z = [z for name in bone_names for z in (bones[name].head_local.z, bones[name].tail_local.z)]
    posed_z = [z for name in bone_names for z in (pose_bones[name].head.z, pose_bones[name].tail.z)]
    shift = floor_contact_shift(rest_z, posed_z)
    parents = {bone.name: (bone.parent.name if bone.parent else None) for bone in bones}
    root = pose_bones[top_ancestor(bone_names[0], parents)]
    rest_rotation = root.bone.matrix_local.to_quaternion()
    root.location = root.location + armature_translation_to_location(rest_rotation, Vector((0.0, 0.0, shift)))


def apply_pose_document(
    armature_obj: bpy.types.Object, document: dict, match_rig_rest: bool = True, floor_contact: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Returns (applied_bone_names, unmatched_bone_names, low_confidence_bone_names).

    match_rig_rest: use the rig-aware solve (rig_pose_basis) when the document
    carries version 2 world targets for every bone; otherwise (or when False)
    apply each parent-relative rotation in the bone's rest frame (a chain's
    extra bones then stay as they are).
    floor_contact: afterwards, move the rig's root so the pose stands on the
    rig's own floor (_keep_feet_on_floor)."""
    if not isinstance(document, dict) or document.get("format") != "pose_craft_pose":
        raise ValueError("Not a PoseCraft pose file")

    version = document.get("version")
    if not isinstance(version, int) or version > MAX_SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported pose format version {version!r} (this add-on understands up to "
            f"version {MAX_SUPPORTED_FORMAT_VERSION}); update the PoseCraft add-on."
        )

    bones = document.get("bones")
    if not isinstance(bones, dict):
        raise ValueError("Malformed pose file: 'bones' must be an object")

    # Validate every bone's data before applying any of it, so a single bad
    # entry aborts cleanly instead of leaving the armature half-posed.
    quaternions = {name: _validate_quaternion(name, data) for name, data in bones.items()}
    world_targets = {}
    for name, data in bones.items():
        world = _validate_vector(name, data, "world_quaternion_wxyz", 4)
        rest_direction = _validate_vector(name, data, "rest_direction", 3)
        if world is not None and rest_direction is not None:
            world_targets[name] = (world, rest_direction)

    pose_bones = armature_obj.pose.bones
    applied, unmatched, low_confidence = [], [], []
    resolved = {}
    for exported_name in quaternions:
        chain = _resolve_bone_chain(armature_obj, exported_name)
        if chain is None:
            unmatched.append(exported_name)
        else:
            resolved[exported_name] = chain

    use_rig_rest = match_rig_rest and bool(resolved) and all(name in world_targets for name in resolved)
    if use_rig_rest:
        bones_data = armature_obj.data.bones
        deltas = {}
        for exported_name, chain in resolved.items():
            world, rest_direction = world_targets[exported_name]
            shares = distribute_rotation(Quaternion(), Quaternion(world), len(chain))
            for bone_name, share in zip(chain, shares):
                rig_rest_direction = (bones_data[bone_name].matrix_local.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized()
                alignment = rig_rest_direction.rotation_difference(Vector(rest_direction).normalized())
                deltas[bone_name] = share @ alignment
        parents = {bone.name: (bone.parent.name if bone.parent else None) for bone in bones_data}
        rest = {bone.name: bone.matrix_local.to_quaternion() for bone in bones_data}
        basis = {pose_bone.name: pose_bone.matrix_basis.to_quaternion() for pose_bone in pose_bones}
        inherit = {bone.name: bone.use_inherit_rotation for bone in bones_data}
        new_basis = rig_pose_basis(bones_parents_first(parents), parents, rest, basis, deltas, inherit)
    else:
        new_basis = {}
        for exported_name, chain in resolved.items():
            rest_rotation = pose_bones[chain[0]].bone.matrix_local.to_quaternion()
            new_basis[chain[0]] = armature_delta_to_basis(rest_rotation, Quaternion(quaternions[exported_name]))

    for exported_name, chain in resolved.items():
        for bone_name in chain:
            if bone_name not in new_basis:
                continue
            _write_rotation(pose_bones[bone_name], new_basis[bone_name])
            applied.append(bone_name)
            if bones[exported_name].get("low_confidence"):
                low_confidence.append(bone_name)

    if floor_contact:
        _keep_feet_on_floor(armature_obj, applied)

    return applied, unmatched, low_confidence


def _keyframe_applied(armature_obj: bpy.types.Object, bone_names: list[str], scene) -> None:
    """B18: keys each applied bone's rotation (and the root's location, which
    floor contact may have moved) at the current frame, then advances the
    frame by the panel's step -- a sequence of photos becomes blocked-out
    animation."""
    pose_bones = armature_obj.pose.bones
    frame = scene.frame_current
    for name in bone_names:
        pose_bone = pose_bones[name]
        pose_bone.keyframe_insert(rotation_data_path(pose_bone.rotation_mode), frame=frame, group=name)
    if bone_names and scene.posecraft_floor_contact:
        parents = {bone.name: (bone.parent.name if bone.parent else None) for bone in armature_obj.data.bones}
        root = top_ancestor(bone_names[0], parents)
        pose_bones[root].keyframe_insert("location", frame=frame, group=root)
    if scene.posecraft_frame_step:
        scene.frame_set(frame + scene.posecraft_frame_step)


def _should_keyframe(scene) -> bool:
    return bool(scene.posecraft_insert_keyframes or scene.tool_settings.use_keyframe_insert_auto)


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

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                document = json.load(f)
        except (OSError, ValueError) as exc:  # ValueError covers JSONDecodeError/UnicodeDecodeError
            self.report({"ERROR"}, f"Could not read pose file: {exc}")
            return {"CANCELLED"}

        try:
            applied, unmatched, low_confidence = apply_pose_document(
                armature_obj, document, context.scene.posecraft_match_rig_rest, context.scene.posecraft_floor_contact
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        if _should_keyframe(context.scene):
            _keyframe_applied(armature_obj, applied, context.scene)

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
    extra_bones: bpy.props.StringProperty(
        name="Extra Bones",
        description=(
            "Optional: more rig bones continuing the chain after the target bone, comma-separated "
            "(e.g. 'Spine1, Spine2'). The rotation is spread evenly over the whole chain"
        ),
    )


class POSECRAFT_UL_bone_mapping(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.posecraft_name)
        # `data` here is the armature Object passed to template_list() below, so
        # data.data is its Armature datablock -- prop_search's standard "pick an
        # existing bone by name, with autocomplete" idiom.
        row.prop_search(item, "target_bone", data.data, "bones", text="", icon="BONE_DATA")
        row.prop(item, "extra_bones", text="")


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

        existing = {
            entry.posecraft_name: (entry.target_bone, entry.extra_bones) for entry in armature_obj.posecraft_bone_mapping
        }
        armature_obj.posecraft_bone_mapping.clear()
        for name in POSECRAFT_BONE_NAMES:
            entry = armature_obj.posecraft_bone_mapping.add()
            entry.posecraft_name = name
            entry.target_bone, entry.extra_bones = existing.get(name, ("", ""))

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
                if not entry.extra_bones:
                    entry.extra_bones = ", ".join(mixamo_chain_extras(entry.posecraft_name, guess, pose_bones))
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
            entry.extra_bones = ""

        self.report({"INFO"}, "Cleared bone mapping")
        return {"FINISHED"}


def _tag_view3d_redraw() -> None:
    """B20: the panel's status line is drawn from a module global, which Blender
    doesn't know changed -- redraw the 3D views so it shows the last pose."""
    window_manager = bpy.context.window_manager
    for window in getattr(window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _apply_live_link_document(document) -> dict:
    """Applies one received document to the Live Link target (main thread) and
    returns the reply for the app."""
    global _live_link_status

    refusal = live_link_request_error(document)
    scene = bpy.context.scene
    armature_obj = pick_live_link_target(scene.posecraft_target, bpy.context.view_layer.objects.active)
    if refusal is not None:
        _live_link_status = refusal
        return build_live_link_reply(error=refusal)
    if armature_obj is None:
        _live_link_status = "Received a pose but no target armature is set or selected"
        return build_live_link_reply(error="No target armature in Blender: pick one in the PoseCraft panel")
    applied, unmatched, low_confidence = apply_pose_document(
        armature_obj, document, scene.posecraft_match_rig_rest, scene.posecraft_floor_contact
    )
    if _should_keyframe(scene):
        _keyframe_applied(armature_obj, applied, scene)
    _live_link_status, _level = _summarize(applied, unmatched, low_confidence)
    try:
        # B18: each received pose is its own Ctrl+Z step, like Load Pose JSON.
        bpy.ops.ed.undo_push(message="PoseCraft pose")
    except RuntimeError:
        pass  # no undo context from this timer; the pose is still applied
    requested = len(document.get("bones") or {}) if isinstance(document, dict) else 0
    return build_live_link_reply(applied, requested, unmatched, low_confidence, armature_obj.name)


def _send_reply_and_close(conn: socket.socket, reply: dict) -> None:
    try:
        # A reply is a few hundred bytes, far under any socket buffer; the
        # short timeout only bounds a peer that stopped reading.
        conn.settimeout(0.5)
        conn.sendall(encode_live_link_message(reply))
    except OSError:
        pass  # an app from before replies existed has already closed its end
    finally:
        conn.close()


def _drop_connection(conn: socket.socket) -> None:
    _connections.pop(conn, None)
    try:
        conn.close()
    except OSError:
        pass


def _poll_live_link_server():
    """`bpy.app.timers` callback on Blender's main thread, so nothing here may
    block (B16): accepts any waiting connections, then takes whatever bytes
    each one has delivered (non-blocking recv into a LiveLinkReader) and
    applies a document once it's complete. A connection that hasn't finished
    its message within LIVE_LINK_MESSAGE_DEADLINE_SECONDS is dropped.
    Returning None stops the timer; any other value is the delay until the
    next call.
    """
    global _live_link_status

    if _server_socket is None:
        return None

    while True:
        try:
            conn, _addr = _server_socket.accept()
        except BlockingIOError:
            break
        except OSError as exc:
            _live_link_status = f"Server error: {exc}"
            _tag_view3d_redraw()
            return None
        conn.setblocking(False)
        _connections[conn] = LiveLinkReader(time.monotonic() + LIVE_LINK_MESSAGE_DEADLINE_SECONDS)

    now = time.monotonic()
    for conn, reader in list(_connections.items()):
        document, closed = None, False
        try:
            while document is None:
                try:
                    data = conn.recv(65536)
                except BlockingIOError:
                    break
                if not data:
                    closed = True
                    break
                document = reader.feed(data)
        except (OSError, ValueError) as exc:
            _live_link_status = f"Dropped a Live Link message: {exc}"
            _connections.pop(conn, None)
            _send_reply_and_close(conn, build_live_link_reply(error=str(exc)))
            _tag_view3d_redraw()
            continue

        if document is not None:
            _connections.pop(conn, None)
            try:
                reply = _apply_live_link_document(document)
            except Exception as exc:  # noqa: BLE001 - a bpy.app.timers callback must never
                # raise: per Blender's contract an uncaught exception here silently
                # unregisters the timer, so Live Link would stop accepting connections
                # with no visible error while the panel still claims it's listening.
                _live_link_status = f"Error applying received pose: {exc}"
                reply = build_live_link_reply(error=f"Error applying the pose: {exc}")
            _send_reply_and_close(conn, reply)
            _tag_view3d_redraw()
        elif closed or reader.expired(now):
            _drop_connection(conn)

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

    server_socket.listen(4)
    server_socket.setblocking(False)
    _server_socket = server_socket
    _live_link_status = f"Listening on {LIVE_LINK_HOST}:{port}"
    bpy.app.timers.register(_poll_live_link_server, first_interval=_POLL_INTERVAL_SECONDS, persistent=True)
    return None


def _stop_live_link_server() -> None:
    global _server_socket, _live_link_status
    for conn in list(_connections):
        _drop_connection(conn)
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
            # B17: capture the target now, so later selections don't change it.
            active = context.active_object
            if active is not None and active.type == "ARMATURE":
                context.scene.posecraft_target = active
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

        scene = context.scene
        layout.prop(scene, "posecraft_match_rig_rest")
        layout.prop(scene, "posecraft_floor_contact")
        row = layout.row(align=True)
        row.prop(scene, "posecraft_insert_keyframes")
        row.prop(scene, "posecraft_frame_step")

        layout.separator()
        is_running = _server_socket is not None
        layout.prop(scene, "posecraft_target")
        row = layout.row()
        row.enabled = not is_running  # num_poses-style: the port is fixed once the socket is bound
        row.prop(scene, "posecraft_live_link_port")
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
    bpy.types.Scene.posecraft_match_rig_rest = bpy.props.BoolProperty(
        name="Match rig rest pose",
        description=(
            "Aim each bone using this rig's own rest pose and hierarchy (A-pose rigs, hips and clavicle "
            "bones, sloped feet). Off: apply PoseCraft's parent-relative rotations as-is, which assumes "
            "a T-pose rig shaped like PoseCraft's skeleton"
        ),
        default=True,
    )
    bpy.types.Scene.posecraft_floor_contact = bpy.props.BoolProperty(
        name="Keep feet on the floor",
        description=(
            "After posing, move the rig's root bone so the lowest posed bone sits where the rest pose's "
            "lowest point was: crouches and sits come out lowered instead of hanging at standing height"
        ),
        default=False,
    )
    bpy.types.Scene.posecraft_insert_keyframes = bpy.props.BoolProperty(
        name="Insert keyframes",
        description="Key every applied bone at the current frame (also done whenever auto-keying is on)",
        default=False,
    )
    bpy.types.Scene.posecraft_frame_step = bpy.props.IntProperty(
        name="Then advance",
        description="Frames to step forward after each keyed pose (0: stay on the current frame)",
        default=0,
        min=0,
        max=1000,
    )
    bpy.types.Scene.posecraft_target = bpy.props.PointerProperty(
        name="Live Link target",
        description="The armature received poses go to (set from the active armature when Live Link starts)",
        type=bpy.types.Object,
        poll=lambda _scene, obj: obj.type == "ARMATURE",
    )
    bpy.types.Object.posecraft_bone_mapping = bpy.props.CollectionProperty(type=POSECRAFT_BoneMapping)
    bpy.types.Object.posecraft_bone_mapping_active_index = bpy.props.IntProperty()


def unregister():
    _stop_live_link_server()
    del bpy.types.Object.posecraft_bone_mapping_active_index
    del bpy.types.Object.posecraft_bone_mapping
    del bpy.types.Scene.posecraft_target
    del bpy.types.Scene.posecraft_frame_step
    del bpy.types.Scene.posecraft_insert_keyframes
    del bpy.types.Scene.posecraft_floor_contact
    del bpy.types.Scene.posecraft_match_rig_rest
    del bpy.types.Scene.posecraft_live_link_port
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

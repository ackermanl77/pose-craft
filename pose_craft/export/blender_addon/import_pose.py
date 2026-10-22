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
  against the rig's OWN rest direction and the parent's pose as Blender
  evaluated it, constraints included -- see rig_pose_basis_evaluated() (and
  rig_pose_basis(), the same solve on a simulated hierarchy). A-pose arms, sloped foot bones, a hips bone above the
  thighs (Mixamo) and clavicles between spine and arms all work; bones with no
  PoseCraft counterpart stay as they are and their children still land on
  target.
- Off (or a version 1 file): each bone's parent-relative rotation is converted
  into the bone's rest frame with armature_delta_to_basis() and applied as-is,
  which assumes the rig's rest pose is a T-pose shaped like PoseCraft's.

A mapping row can name a chain: "Extra Bones" lists more rig bones continuing
after its target (a Mixamo "Spine1, Spine2"), and the rotation is spread over
the chain (Match rig rest pose only). Mixamo and Rigify spines are chained
automatically (Spine/Spine1/Spine2; spine_fk/.001/.002/.003).

Rigify (and other constraint-driven rigs), "Pose rig controls" on by default:
a generated Rigify rig deforms with DEF- bones that copy ORG- bones, which in
turn copy the FK or IK controls -- posing a DEF-/ORG- bone directly does
nothing. A mapped bone whose rotation is overridden by a constraint is
redirected to the bone that actually drives it (constraint_driver follows
Copy Transforms/Copy Rotation/Child Of/Armature targets, preferring FK
controls; else a control with the bone's unprefixed name), the limbs whose FK
controls get posed are switched to FK (Rigify's IK_FK property on
upper_arm_parent.L/thigh_parent.L, see rigify_fk_switches), and every bone is
solved against its parent's pose as Blender actually evaluated it,
constraints included (rig_pose_basis_evaluated) -- so the MCH- bones Rigify
puts between the torso and the limbs don't throw the limbs off.

Fingers: a file from the app's "Hands and fingers" option also has 15
finger bones per hand, named as in Rigify (thumb.01.L .. f_pinky.03.R, see
POSECRAFT_FINGER_BONE_NAMES), which match a Rigify rig's finger controls by
name and a Mixamo rig's LeftHandThumb1.. bones through
MIXAMO_FINGER_BONE_NAMES. Bone Mapping rows exist for them too.

Options in the panel: "Keep feet on the floor" moves the rig's root so the
pose stands on the rig's own floor (a crouch comes out lowered); "Insert
keyframes" (or Blender's auto-keying) keys every applied bone at the current
frame, and "Then advance" steps the frame forward for the next pose. Each
bone keeps its own rotation mode (Euler, axis-angle or quaternion).

Animation (Live Link, add-on 0.7.0+): the app's "Send Animation to Blender"
sends a tracked video person as a "pose_craft_animation" document -- one
ordinary pose document per solved video frame plus the clip's fps. Every
frame goes through the same retarget path as a single pose (bone mapping,
Match rig rest pose, Rigify controls, Keep feet on the floor) and is keyed
into the armature's action, whatever "Insert keyframes" says; the first
frame lands on Blender's current frame. The clip is resampled to the scene's
frame rate (add-on 0.8.0): every Blender frame the clip covers gets one key,
the pose at that exact moment of the clip, blended (slerp) from the two
solved frames around it -- so a 30 fps clip in a 24 fps scene keeps its
timing without keying any frame twice, and a 24 fps clip in a 60 fps scene
gets a key on every frame. A frame inside a stretch the tracker lost gets
no key (Blender interpolates over the gap), quaternion keys are kept in one
hemisphere so they don't spin the long way, and the current frame is
restored afterwards. With "Root motion" on, the app's per-frame root travel
(from planted feet) moves the rig's root bone, scaled from the subject's leg
length to the rig's; "Keep feet on the floor" still sets the height. Long clips arrive in
several requests ("chunks", each under MAX_LIVE_LINK_PAYLOAD_BYTES and at
most a few dozen frames, so no single timer tick blocks Blender for long);
the whole animation is one undo step, pushed after the last chunk.

Bones that still can't be resolved either way are reported in the Blender
status bar / console rather than silently skipped, and bones the PoseCraft
app flagged "low_confidence" (likely occluded in the source photo) are called
out separately -- expect to need manual correction for those, the same
caveat PoseForge gives for DAZ.

Checked headless in a real Blender 5.2.2 (tests/test_real_blender.py, opt-in
via POSECRAFT_BLENDER): install/register/unregister, Load Pose JSON on a
PoseCraft-shaped, a Mixamo-style and a generated Rigify rig (fingers, FK
switch, FK spine), rotation modes, keyframes, Keep feet on the floor, and a
real Live Link server receiving a pose and a chunked animation. The panel
itself (drawing, the file browser) needs Blender's UI and wasn't checked.
See docs/PROJECT_STATUS.md.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import socket
import time

import bpy
from mathutils import Quaternion, Vector

bl_info = {
    "name": "PoseCraft Pose Importer",
    "author": "PoseCraft",
    "version": (0, 9, 0),
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

# Mirrors pose_craft/retarget/skeleton.py::FINGER_BONE_NAMES (B11): present in
# a file only when the app solved the hands. Rigify's own finger names.
POSECRAFT_FINGER_BONE_NAMES = [
    "thumb.01.L", "thumb.02.L", "thumb.03.L",
    "f_index.01.L", "f_index.02.L", "f_index.03.L",
    "f_middle.01.L", "f_middle.02.L", "f_middle.03.L",
    "f_ring.01.L", "f_ring.02.L", "f_ring.03.L",
    "f_pinky.01.L", "f_pinky.02.L", "f_pinky.03.L",
    "thumb.01.R", "thumb.02.R", "thumb.03.R",
    "f_index.01.R", "f_index.02.R", "f_index.03.R",
    "f_middle.01.R", "f_middle.02.R", "f_middle.03.R",
    "f_ring.01.R", "f_ring.02.R", "f_ring.03.R",
    "f_pinky.01.R", "f_pinky.02.R", "f_pinky.03.R",
]

# Mirrors pose_craft/export/blender_pose.py::FORMAT_VERSION, for the same
# reason POSECRAFT_BONE_NAMES mirrors BONE_NAMES above -- this add-on can't
# import that package. A document whose version is newer than this add-on
# understands is rejected rather than silently (mis)applied.
MAX_SUPPORTED_FORMAT_VERSION = 2
# Mirrors pose_craft/export/blender_pose.py::ANIMATION_FORMAT_NAME/_VERSION:
# a Live Link request carrying a whole tracked video person (see
# parse_animation_document).
ANIMATION_FORMAT_NAME = "pose_craft_animation"
MAX_SUPPORTED_ANIMATION_VERSION = 1

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
MIXAMO_FINGER_BONE_NAMES = {
    "thumb.01.L": "LeftHandThumb1", "thumb.02.L": "LeftHandThumb2", "thumb.03.L": "LeftHandThumb3",
    "f_index.01.L": "LeftHandIndex1", "f_index.02.L": "LeftHandIndex2", "f_index.03.L": "LeftHandIndex3",
    "f_middle.01.L": "LeftHandMiddle1", "f_middle.02.L": "LeftHandMiddle2", "f_middle.03.L": "LeftHandMiddle3",
    "f_ring.01.L": "LeftHandRing1", "f_ring.02.L": "LeftHandRing2", "f_ring.03.L": "LeftHandRing3",
    "f_pinky.01.L": "LeftHandPinky1", "f_pinky.02.L": "LeftHandPinky2", "f_pinky.03.L": "LeftHandPinky3",
    "thumb.01.R": "RightHandThumb1", "thumb.02.R": "RightHandThumb2", "thumb.03.R": "RightHandThumb3",
    "f_index.01.R": "RightHandIndex1", "f_index.02.R": "RightHandIndex2", "f_index.03.R": "RightHandIndex3",
    "f_middle.01.R": "RightHandMiddle1", "f_middle.02.R": "RightHandMiddle2", "f_middle.03.R": "RightHandMiddle3",
    "f_ring.01.R": "RightHandRing1", "f_ring.02.R": "RightHandRing2", "f_ring.03.R": "RightHandRing3",
    "f_pinky.01.R": "RightHandPinky1", "f_pinky.02.R": "RightHandPinky2", "f_pinky.03.R": "RightHandPinky3",
}
MIXAMO_PREFIXES = ["mixamorig:", "mixamorig1:", ""]
# Mixamo's spine is three bones (Spine, Spine1, Spine2); PoseCraft's one spine
# bone is spread over all of them (see distribute_rotation).
MIXAMO_SPINE_CHAIN_EXTRAS = ["Spine1", "Spine2"]
# Rigify's FK spine controls above spine_fk (a generated rig's torso).
RIGIFY_SPINE_CHAIN_EXTRAS = ["spine_fk.001", "spine_fk.002", "spine_fk.003"]

# Constraint types that override a bone's own rotation: a bone carrying one
# (unmuted) is driven by something else, and posing it has no effect.
OVERRIDING_CONSTRAINTS = [
    "COPY_TRANSFORMS", "COPY_ROTATION", "CHILD_OF", "ARMATURE",
    "DAMPED_TRACK", "STRETCH_TO", "TRACK_TO", "LOCKED_TRACK", "IK",
]
# The subset whose target is the bone it copies its rotation from -- what
# constraint_driver follows back to the control that drives a DEF-/ORG- bone.
FOLLOWED_CONSTRAINTS = ["COPY_TRANSFORMS", "COPY_ROTATION", "CHILD_OF", "ARMATURE"]
RIGIFY_PREFIXES = ["DEF-", "ORG-", "MCH-"]
# Rigify limbs: which chain's master control holds each FK control's IK/FK
# switch (upper_arm_parent.L holds IK_FK for upper_arm/forearm/hand_fk.L).
RIGIFY_LIMB_ROOTS = {
    "upper_arm": "upper_arm", "forearm": "upper_arm", "hand": "upper_arm",
    "thigh": "thigh", "shin": "thigh", "foot": "thigh",
}
RIGIFY_IK_FK_PROPERTY = "IK_FK"

LIVE_LINK_HOST = "127.0.0.1"
DEFAULT_LIVE_LINK_PORT = 8642
# Mirrors pose_craft/export/live_link.py::PROTOCOL_VERSION. Version 2: every
# request gets a length-prefixed JSON reply saying what was applied (see
# build_live_link_reply). A request without the field is from an older app.
# Version 3: animation requests (pose_craft_animation, chunked). A request
# carries the lowest protocol that understands it -- single poses still say 2,
# so they keep working with a 0.4-0.6 add-on, while an animation says 3 and an
# older add-on refuses it with "update the add-on" instead of misreading it.
LIVE_LINK_PROTOCOL_VERSION = 4
# Version 4 (F17): {"format": "pose_craft_library", "path": <bundle JSON>} -- the app's
# pose library, made into Pose Library assets (create_library_assets).
LIBRARY_FORMAT_NAME = "pose_craft_library"
_POLL_INTERVAL_SECONDS = 0.1
# A 15-bone document is about 4 KB; guards against a bogus or hostile length
# header making the server buffer an unbounded read. Animations are not an
# exception: the app splits them into chunks under this cap (live_link.py's
# ANIMATION_CHUNK_MAX_BYTES) rather than this cap being raised.
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
    mixamo_name = MIXAMO_BONE_NAMES.get(exported_name) or MIXAMO_FINGER_BONE_NAMES.get(exported_name)
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


def rigify_chain_extras(exported_name: str, base_bone: str, existing) -> list:
    """Rigify's FK spine above a guessed spine_fk (spine_fk.001-.003, when
    they exist): the torso bends along it like a Mixamo spine."""
    if exported_name != "spine" or base_bone != "spine_fk":
        return []
    return [name for name in RIGIFY_SPINE_CHAIN_EXTRAS if name in existing]


def is_constraint_driven(name: str, constraints: dict) -> bool:
    """Whether `name` has an unmuted constraint that overrides its rotation.
    `constraints`: bone name -> list of {"type", "subtarget", "influence",
    "mute"} for constraints targeting this armature's bones. Pure (no bpy).
    A constraint at influence 0 still counts: Rigify drives influences."""
    return any(c.get("type") in OVERRIDING_CONSTRAINTS and not c.get("mute") for c in constraints.get(name, ()))


def constraint_driver(name: str, constraints: dict, max_depth: int = 8) -> str:
    """The bone whose rotation `name` follows through Copy Transforms / Copy
    Rotation / Child Of / Armature constraints (several hops: DEF- -> ORG- ->
    the control), preferring FK targets (Rigify's ORG- bones copy both the
    FK and the IK chain), then non-MCH ones, then the higher influence. `name`
    itself when nothing is followed. Pure (no bpy)."""
    seen, current = {name}, name
    for _hop in range(max_depth):
        options = [
            c for c in constraints.get(current, ())
            if c.get("type") in FOLLOWED_CONSTRAINTS and c.get("subtarget") and not c.get("mute")
        ]
        if not options:
            break

        def preference(item):
            index, constraint = item
            target = constraint["subtarget"]
            return (
                0 if "_fk" in target.lower() else 1,
                1 if target.startswith("MCH-") else 0,
                -float(constraint.get("influence", 1.0)),
                index,
            )

        _index, best = min(enumerate(options), key=preference)
        if best["subtarget"] in seen:
            break
        current = best["subtarget"]
        seen.add(current)
    return current


def control_bone_for(name: str, constraints: dict, existing) -> str:
    """The bone to pose so that `name` moves: `name` itself when nothing
    overrides its rotation; else the free bone its constraints follow
    (constraint_driver); else a free control named like it without Rigify's
    DEF-/ORG-/MCH- prefix (upper_arm_fk.L for DEF-upper_arm.L). Pure (no bpy)."""
    if not is_constraint_driven(name, constraints):
        return name
    driver = constraint_driver(name, constraints)
    if driver != name and driver in existing and not is_constraint_driven(driver, constraints):
        return driver
    for prefix in RIGIFY_PREFIXES:
        if name.startswith(prefix):
            for candidate in bone_name_candidates(name[len(prefix):]):
                if any(candidate.startswith(p) for p in RIGIFY_PREFIXES):
                    continue
                if candidate in existing and not is_constraint_driven(candidate, constraints):
                    return candidate
    return driver if driver in existing else name


def rigify_fk_switches(controls, properties: dict) -> list:
    """(holder bone, property) pairs to set to 1.0 so the Rigify limbs whose FK
    controls are in `controls` follow FK instead of IK: the limb's master
    control (upper_arm_parent.L / thigh_parent.L, or an MCH- one) carrying
    an IK_FK property. `properties`: bone name -> its custom property names.
    Pure (no bpy)."""
    switches = []
    for control in controls:
        base, dot, side = control.rpartition(".")
        side = dot + side if dot and side in ("L", "R") else ""
        stem = base if side else control
        if not stem.endswith("_fk"):
            continue
        root = RIGIFY_LIMB_ROOTS.get(stem[: -len("_fk")])
        if root is None:
            continue
        for holder in (f"{root}_parent{side}", f"MCH-{root}_parent{side}"):
            if RIGIFY_IK_FK_PROPERTY in properties.get(holder, ()):
                if (holder, RIGIFY_IK_FK_PROPERTY) not in switches:
                    switches.append((holder, RIGIFY_IK_FK_PROPERTY))
                break
    return switches


def rig_pose_basis_evaluated(order, parents, rest, deltas, evaluate, write, inherit_rotation=None, passes=2):
    """rig_pose_basis against the rig as Blender actually evaluates it: each
    mapped bone is solved from its parent's evaluated pose (constraints and
    drivers included -- Rigify's MCH- bones between the torso and the limbs
    follow other bones), so it lands on its target whatever drives the
    parent. Level by level down the hierarchy: `evaluate(names)` returns
    {name: posed armature-space rotation} after the writes so far (Blender:
    view_layer.update(), then pose_bone.matrix), `write(name, basis)` applies
    a basis rotation. A second pass catches a parent driven by a bone solved
    after it (a constraint across branches). Returns the bases written.
    Duck-typed on `@` and `.inverted()`."""
    inherit_rotation = inherit_rotation or {}
    depth = {}
    for name in order:
        parent = parents.get(name)
        depth[name] = 0 if parent is None else depth.get(parent, -1) + 1
    mapped = [name for name in order if name in deltas]
    result = {}
    for _pass in range(max(1, passes)):
        for level in sorted({depth[name] for name in mapped}):
            bones = [name for name in mapped if depth[name] == level]
            needed = [parents[n] for n in bones if parents.get(n) is not None and inherit_rotation.get(n, True)]
            posed = evaluate(sorted(set(needed))) if needed else {}
            for name in bones:
                parent = parents.get(name)
                if parent is None or not inherit_rotation.get(name, True):
                    chain = rest[name]
                else:
                    chain = posed[parent] @ rest[parent].inverted() @ rest[name]
                basis = chain.inverted() @ (deltas[name] @ rest[name])
                write(name, basis)
                result[name] = basis
    return result


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


def build_live_link_reply(
    applied=(), requested=0, unmatched=(), low_confidence=(), armature=None, error=None, animation=None, library=None
) -> dict:
    """The reply the Live Link server sends back after each pose (protocol 2),
    read by the app's send_pose_document(). For an animation chunk (protocol
    3), `animation` adds "animation_frames" (frames keyed by this request),
    "start_frame"/"end_frame" (the Blender frames they span) and
    "chunk_index"/"chunk_count". Pure (no bpy)."""
    reply = {
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
    if animation:
        reply.update(animation)
    if library is not None:
        reply["library"] = library
    return reply


def read_library_bundle(path: str) -> list:
    """The poses of a PoseCraft library bundle (pose_library.export_blender_bundle):
    [(name, tags, pose document, thumbnail path or None)]. Raises ValueError for
    anything that isn't one. Pure (no bpy)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read the PoseCraft library: {exc}") from None
    if not isinstance(bundle, dict) or bundle.get("format") != LIBRARY_FORMAT_NAME or not isinstance(bundle.get("poses"), list):
        raise ValueError("Not a PoseCraft library bundle")
    folder = os.path.dirname(os.path.abspath(path))
    poses = []
    for item in bundle["poses"]:
        if not isinstance(item, dict) or not isinstance(item.get("document"), dict):
            raise ValueError("Malformed pose in the PoseCraft library")
        name = str(item.get("name") or "PoseCraft pose")[:60]
        tags = [str(tag) for tag in item.get("tags") or [] if isinstance(tag, str)]
        thumbnail = item.get("thumbnail")
        thumbnail_path = None
        if isinstance(thumbnail, str) and os.path.basename(thumbnail) == thumbnail:
            candidate = os.path.join(folder, thumbnail)
            thumbnail_path = candidate if os.path.isfile(candidate) else None
        poses.append((name, tags, item["document"], thumbnail_path))
    return poses


def _make_pose_asset(armature_obj, name: str, bone_names, frame):
    """One Pose Library asset (an Action marked as asset) from the armature's
    current pose of `bone_names`, through Blender's own pose_library code."""
    from pose_library import pose_creation

    fields = {f.name for f in dataclasses.fields(pose_creation.PoseCreationParams)}
    values = {
        "armature_ob": armature_obj, "src_action": None, "src_action_slot": None,
        "src_frame_nr": frame, "bone_names": frozenset(bone_names), "new_asset_name": name,
    }
    params = pose_creation.PoseCreationParams(**{key: value for key, value in values.items() if key in fields})
    return pose_creation.create_pose_asset(params)


def create_library_assets(armature_obj, bundle_path: str, scene) -> dict:
    """F17: every pose of the bundle applied to `armature_obj` and stored as a
    Pose Library asset (with the app's thumbnail as its preview and its tags as
    asset tags); the armature's pose is put back afterwards. Returns
    {"created": n, "failed": [names], "assets": [names]}."""
    poses = read_library_bundle(bundle_path)
    pose_bones = armature_obj.pose.bones
    saved = {pb.name: (pb.rotation_mode, tuple(pb.rotation_quaternion), tuple(pb.rotation_euler), tuple(pb.location))
             for pb in pose_bones}
    created, failed = [], []
    try:
        for name, tags, document, thumbnail in poses:
            for pb in pose_bones:
                pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.location = (0.0, 0.0, 0.0)
            try:
                applied, _unmatched, _low = apply_pose_document(
                    armature_obj, document, scene.posecraft_match_rig_rest, scene.posecraft_floor_contact,
                    scene.posecraft_pose_controls,
                )
                bpy.context.view_layer.update()
                action = _make_pose_asset(armature_obj, name, applied, scene.frame_current) if applied else None
            except (ValueError, RuntimeError, ImportError) as exc:
                print(f"PoseCraft: pose asset '{name}' failed: {exc}")
                action = None
            if action is None:
                failed.append(name)
                continue
            for tag in ["PoseCraft", *tags]:
                if tag not in {t.name for t in action.asset_data.tags}:
                    action.asset_data.tags.new(tag)
            action.asset_data.description = "Made by PoseCraft"
            if thumbnail:
                try:
                    with bpy.context.temp_override(id=action):
                        bpy.ops.ed.lib_id_load_custom_preview(filepath=thumbnail)
                except (RuntimeError, AttributeError, TypeError):
                    pass  # keeps the generated preview
            created.append(action.name)
    finally:
        for pb in pose_bones:
            mode, quaternion, euler, location = saved[pb.name]
            pb.rotation_quaternion, pb.rotation_euler, pb.location = quaternion, euler, location
    return {"created": len(created), "failed": failed, "assets": created}


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


def _armature_constraints(armature_obj: bpy.types.Object) -> dict:
    """Every pose bone's constraints that target this armature's bones, in
    the plain-dict form the pure helpers take (see is_constraint_driven)."""
    table = {}
    for pose_bone in armature_obj.pose.bones:
        entries = []
        for constraint in pose_bone.constraints:
            mute = bool(getattr(constraint, "mute", False)) or not getattr(constraint, "enabled", True)
            if constraint.type == "ARMATURE":
                for target in constraint.targets:
                    if target.target is armature_obj and target.subtarget:
                        entries.append({"type": "ARMATURE", "subtarget": target.subtarget,
                                        "influence": target.weight, "mute": mute})
                continue
            subtarget = getattr(constraint, "subtarget", "") if getattr(constraint, "target", None) is armature_obj else ""
            entries.append({"type": constraint.type, "subtarget": subtarget,
                            "influence": constraint.influence, "mute": mute})
        if entries:
            table[pose_bone.name] = entries
    return table


def _switch_rigify_limbs_to_fk(armature_obj: bpy.types.Object, controls) -> list:
    """Sets Rigify's IK_FK switch to FK for the limbs whose FK controls get
    posed (rigify_fk_switches); returns the holder bones changed."""
    pose_bones = armature_obj.pose.bones
    properties = {pb.name: set(pb.keys()) for pb in pose_bones if pb.keys()}
    changed = []
    for holder, prop in rigify_fk_switches(controls, properties):
        if pose_bones[holder][prop] != 1.0:
            pose_bones[holder][prop] = 1.0
            changed.append(holder)
    return changed


def _resolve_bone_chain(
    armature_obj: bpy.types.Object, exported_name: str, constraints: dict | None = None
) -> list[str] | None:
    """The rig bones `exported_name` drives, base first (usually one; a spine
    chain can be several), or None if it can't be resolved. With
    `constraints` (see _armature_constraints), a constraint-driven bone is
    replaced by the control that drives it (control_bone_for)."""
    chain = _resolve_bone_chain_names(armature_obj, exported_name)
    if chain is None or constraints is None:
        return chain
    redirected = []
    for name in chain:
        control = control_bone_for(name, constraints, armature_obj.pose.bones)
        if control not in redirected:
            redirected.append(control)
    return redirected


def _resolve_bone_chain_names(armature_obj: bpy.types.Object, exported_name: str) -> list[str] | None:
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
    return (
        [guess]
        + mixamo_chain_extras(exported_name, guess, pose_bones)
        + rigify_chain_extras(exported_name, guess, pose_bones)
    )


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


def validate_pose_document(document) -> tuple[dict, dict, dict]:
    """(bones, quaternions, world_targets) of a pose document, every bone
    checked before anything is applied, so a single bad entry aborts cleanly
    instead of leaving the armature half-posed. Raises ValueError. Pure (no
    bpy)."""
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

    quaternions = {name: _validate_quaternion(name, data) for name, data in bones.items()}
    world_targets = {}
    for name, data in bones.items():
        world = _validate_vector(name, data, "world_quaternion_wxyz", 4)
        rest_direction = _validate_vector(name, data, "rest_direction", 3)
        if world is not None and rest_direction is not None:
            world_targets[name] = (world, rest_direction)
    return bones, quaternions, world_targets


def is_animation_document(document) -> bool:
    return isinstance(document, dict) and document.get("format") == ANIMATION_FORMAT_NAME


def _plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_animation_document(document) -> dict:
    """Checks a whole pose_craft_animation request (or one chunk of it) before
    anything is keyed and returns {"fps", "frames": [(frame_index, pose)],
    "first_frame_index", "start_frame", "chunk_index", "chunk_count"}.

    frames: the clip's solved frames, frame_index strictly increasing (gaps are
    frames the tracker lost). root_offsets / root_leg_length: the optional
    root motion (each frame's travel from the clip's first frame, in Blender
    axes, and the leg length it was measured with). first_frame_index: the clip frame that lands on
    start_frame -- the first frame of the WHOLE animation, so every chunk is
    spaced from the same anchor. start_frame: None on the first chunk (the
    add-on uses Blender's current frame and replies with it); later chunks
    repeat what the first reply said. Raises ValueError. Pure (no bpy)."""
    if not is_animation_document(document):
        raise ValueError("Not a PoseCraft animation")
    version = document.get("version")
    if not _plain_int(version) or version > MAX_SUPPORTED_ANIMATION_VERSION:
        raise ValueError(
            f"Unsupported animation format version {version!r} (this add-on understands up to "
            f"version {MAX_SUPPORTED_ANIMATION_VERSION}); update the PoseCraft add-on."
        )
    fps = document.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Malformed animation: 'fps' must be a positive number, got {fps!r}")
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Malformed animation: 'frames' must be a non-empty list")
    parsed = []
    root_offsets = {}
    for position, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed animation: frame {position} must be an object")
        frame_index = entry.get("frame_index")
        if not _plain_int(frame_index) or frame_index < 0:
            raise ValueError(f"Malformed animation: frame {position} needs a non-negative 'frame_index'")
        if parsed and frame_index <= parsed[-1][0]:
            raise ValueError("Malformed animation: 'frame_index' must increase from frame to frame")
        try:
            validate_pose_document(entry.get("pose"))
        except ValueError as exc:
            raise ValueError(f"Animation frame {frame_index}: {exc}") from None
        parsed.append((frame_index, entry["pose"]))
        offset = entry.get("root_offset")
        if offset is not None:
            if (not isinstance(offset, (list, tuple)) or len(offset) != 3
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in offset)):
                raise ValueError(f"Malformed animation: frame {frame_index}'s 'root_offset' must be 3 finite numbers")
            root_offsets[frame_index] = [float(v) for v in offset]
    leg_length = document.get("root_leg_length")
    if leg_length is not None and (isinstance(leg_length, bool) or not isinstance(leg_length, (int, float))
                                   or not math.isfinite(leg_length) or leg_length <= 0):
        raise ValueError("Malformed animation: 'root_leg_length' must be a positive number")
    first = document.get("first_frame_index", parsed[0][0])
    if not _plain_int(first) or first > parsed[0][0]:
        raise ValueError("Malformed animation: 'first_frame_index' must be an integer no later than the first frame")
    start_frame = document.get("start_frame")
    if start_frame is not None and not _plain_int(start_frame):
        raise ValueError("Malformed animation: 'start_frame' must be an integer")
    chunk = document.get("chunk") or {"index": 0, "count": 1}
    index, count = (chunk.get("index"), chunk.get("count")) if isinstance(chunk, dict) else (None, None)
    if not (_plain_int(index) and _plain_int(count) and 0 <= index < count):
        raise ValueError("Malformed animation: 'chunk' must be {'index': i, 'count': n} with 0 <= i < n")
    if index > 0 and start_frame is None:
        raise ValueError("Malformed animation: a later chunk must repeat the first chunk's 'start_frame'")
    return {
        "fps": float(fps),
        "frames": parsed,
        "first_frame_index": first,
        "start_frame": start_frame,
        "chunk_index": index,
        "chunk_count": count,
        "root_offsets": root_offsets,  # clip frame index -> armature-space travel (root motion)
        "root_leg_length": float(leg_length) if leg_length is not None else None,
    }


def animation_frame_numbers(frame_indices, first_frame_index: int, start_frame: int, clip_fps: float, scene_fps: float):
    """The Blender frame each clip frame is keyed on: start_frame for
    first_frame_index, the rest at the clip's timing in the scene's fps
    (rounded to whole frames). Pure (no bpy)."""
    scale = float(scene_fps) / float(clip_fps)
    return [int(start_frame + math.floor((index - first_frame_index) * scale + 0.5)) for index in frame_indices]


def slerp_wxyz(a, b, t: float) -> list:
    """Spherical interpolation between quaternions a and b (w, x, y, z) at t
    in [0, 1], the short way round, normalized. Pure (no bpy)."""
    a = tuple(float(v) for v in a)
    b = keep_quaternion_hemisphere(a, b)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    if dot > 0.9995:  # nearly the same rotation: a plain blend is exact enough and stable
        q = tuple(x + t * (y - x) for x, y in zip(a, b))
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        weight_a, weight_b = math.sin((1.0 - t) * theta) / sin_theta, math.sin(t * theta) / sin_theta
        q = tuple(weight_a * x + weight_b * y for x, y in zip(a, b))
    norm = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / norm for v in q]


def interpolate_pose_documents(a: dict, b: dict, t: float) -> dict:
    """The pose a fraction t of the way from pose document a to b: each bone's
    local and world rotation slerped, "low_confidence" kept if either has it.
    Everything else (rest directions, a bone only one of them has) comes from
    the nearer pose. Pure (no bpy)."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    nearer = a if t < 0.5 else b
    bones = {}
    for name, entry in nearer["bones"].items():
        bone_a, bone_b = a["bones"].get(name), b["bones"].get(name)
        if bone_a is None or bone_b is None:
            bones[name] = entry
            continue
        merged = dict(entry)
        merged["quaternion_wxyz"] = slerp_wxyz(bone_a["quaternion_wxyz"], bone_b["quaternion_wxyz"], t)
        if "world_quaternion_wxyz" in bone_a and "world_quaternion_wxyz" in bone_b:
            merged["world_quaternion_wxyz"] = slerp_wxyz(bone_a["world_quaternion_wxyz"], bone_b["world_quaternion_wxyz"], t)
        if bone_a.get("low_confidence") or bone_b.get("low_confidence"):
            merged["low_confidence"] = True
        bones[name] = merged
    return dict(nearer, bones=bones)


def resample_animation(samples, first_frame_index: int, start_frame: int, clip_fps: float, scene_fps: float,
                       include_first: bool = True) -> list:
    """The Blender keys for solved clip samples [(clip_frame_index, pose,
    root_offset or None)], indices increasing: one (blender_frame, pose,
    root_offset) per whole Blender frame from the first sample's time to the
    last one's (first_frame_index lands on start_frame; time scales by
    scene_fps / clip_fps), the pose blended between the two samples around
    that time. A frame inside a gap (its two samples more than one clip frame
    apart: frames the tracker lost) gets no key unless a sample lands on it.
    include_first=False starts strictly after the first sample's time -- that
    sample is the previous chunk's last one, already keyed up to there.
    Pure (no bpy)."""
    if not samples:
        return []
    scale = float(scene_fps) / float(clip_fps)
    times = [start_frame + (index - first_frame_index) * scale for index, _pose, _offset in samples]
    eps = 1e-6
    first = math.ceil(times[0] - eps) if include_first else math.floor(times[0] + eps) + 1
    last = math.floor(times[-1] + eps)
    keys = []
    j = 0
    for frame in range(first, last + 1):
        while j + 1 < len(times) and times[j + 1] <= frame + eps:
            j += 1
        index_a, pose_a, offset_a = samples[j]
        if abs(times[j] - frame) <= eps:
            keys.append((frame, pose_a, offset_a))
            continue
        if j + 1 >= len(samples) or samples[j + 1][0] - index_a > 1:
            continue  # beyond the last sample, or inside a tracking gap
        index_b, pose_b, offset_b = samples[j + 1]
        t = (frame - times[j]) / (times[j + 1] - times[j])
        if offset_a is not None and offset_b is not None:
            offset = [x + t * (y - x) for x, y in zip(offset_a, offset_b)]
        else:
            offset = offset_a if t < 0.5 else offset_b
        keys.append((frame, interpolate_pose_documents(pose_a, pose_b, t), offset))
    return keys


def keep_quaternion_hemisphere(previous, current) -> tuple:
    """`current` (w, x, y, z), negated when it points away from `previous`:
    the same rotation, but Blender interpolates quaternion F-curves component
    by component, so a sign flip between two keys would spin the bone the long
    way round in between. Pure (no bpy)."""
    current = tuple(float(v) for v in current)
    if previous is not None and sum(float(p) * c for p, c in zip(previous, current)) < 0.0:
        return tuple(-c for c in current)
    return current


def _write_rotation(pose_bone, quaternion) -> str:
    """Sets a pose bone's basis rotation in its own rotation mode (see
    rotation_channel) and returns the data path written, for keyframing."""
    data_path, value = rotation_channel(pose_bone.rotation_mode, quaternion, pose_bone.rotation_euler.copy())
    if data_path == "rotation_quaternion":
        # Same rotation, but on the current key's side (animation: no long-way spins between keys).
        value = Quaternion(keep_quaternion_hemisphere(pose_bone.rotation_quaternion, value))
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
    armature_obj: bpy.types.Object,
    document: dict,
    match_rig_rest: bool = True,
    floor_contact: bool = False,
    pose_controls: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (applied_bone_names, unmatched_bone_names, low_confidence_bone_names).

    match_rig_rest: use the rig-aware solve when the document carries version
    2 world targets for every bone -- each bone solved against its parent's
    evaluated pose (rig_pose_basis_evaluated); otherwise (or when False)
    apply each parent-relative rotation in the bone's rest frame (a chain's
    extra bones then stay as they are).
    floor_contact: afterwards, move the rig's root so the pose stands on the
    rig's own floor (_keep_feet_on_floor).
    pose_controls: redirect constraint-driven bones (Rigify DEF-/ORG-) to the
    controls that drive them, and switch the Rigify limbs being posed to FK."""
    bones, quaternions, world_targets = validate_pose_document(document)

    pose_bones = armature_obj.pose.bones
    applied, unmatched, low_confidence = [], [], []
    resolved = {}
    constraints = _armature_constraints(armature_obj) if pose_controls else None
    for exported_name in quaternions:
        chain = _resolve_bone_chain(armature_obj, exported_name, constraints)
        if chain is None:
            unmatched.append(exported_name)
        else:
            resolved[exported_name] = chain
    if pose_controls:
        switched = _switch_rigify_limbs_to_fk(armature_obj, [name for chain in resolved.values() for name in chain])
        if switched:
            print(f"PoseCraft: switched Rigify limb(s) to FK: {', '.join(switched)}")

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
        inherit = {bone.name: bone.use_inherit_rotation for bone in bones_data}

        def evaluate(names):
            bpy.context.view_layer.update()
            return {name: pose_bones[name].matrix.to_quaternion() for name in names}

        def write(name, basis):
            _write_rotation(pose_bones[name], basis)

        new_basis = rig_pose_basis_evaluated(
            bones_parents_first(parents), parents, rest, deltas, evaluate, write, inherit
        )
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


def _keyframe_applied(
    armature_obj: bpy.types.Object, bone_names: list[str], scene, frame=None, advance: bool = True,
    root_location: bool = False,
) -> None:
    """B18: keys each applied bone's rotation (and the root's location, which
    floor contact may have moved) at the current frame (or `frame`), then
    advances the frame by the panel's step -- a sequence of photos becomes
    blocked-out animation. An animation request keys its own frames and
    passes advance=False."""
    pose_bones = armature_obj.pose.bones
    frame = scene.frame_current if frame is None else frame
    for name in bone_names:
        pose_bone = pose_bones[name]
        pose_bone.keyframe_insert(rotation_data_path(pose_bone.rotation_mode), frame=frame, group=name)
    if bone_names and (scene.posecraft_floor_contact or root_location):
        parents = {bone.name: (bone.parent.name if bone.parent else None) for bone in armature_obj.data.bones}
        root = top_ancestor(bone_names[0], parents)
        pose_bones[root].keyframe_insert("location", frame=frame, group=root)
    if advance and scene.posecraft_frame_step:
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
                armature_obj, document, context.scene.posecraft_match_rig_rest, context.scene.posecraft_floor_contact,
                context.scene.posecraft_pose_controls,
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


class POSECRAFT_OT_import_library(bpy.types.Operator):
    bl_idname = "posecraft.import_library"
    bl_label = "Import PoseCraft Library as Pose Assets"
    bl_description = ("Make a Pose Library asset (with its thumbnail) from every pose of a PoseCraft library "
                      "bundle (posecraft_library.json), on the active armature")
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        armature_obj = context.active_object
        if armature_obj is None or armature_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an armature first")
            return {"CANCELLED"}
        try:
            result = create_library_assets(armature_obj, self.filepath, context.scene)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        level = "WARNING" if result["failed"] else "INFO"
        self.report({level}, f"Made {result['created']} pose asset(s)"
                    + (f"; failed: {', '.join(result['failed'])}" if result["failed"] else ""))
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
        names = POSECRAFT_BONE_NAMES + POSECRAFT_FINGER_BONE_NAMES
        for name in names:
            entry = armature_obj.posecraft_bone_mapping.add()
            entry.posecraft_name = name
            entry.target_bone, entry.extra_bones = existing.get(name, ("", ""))

        self.report({"INFO"}, f"{len(names)} bone mapping row(s) ready")
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
                    extras = mixamo_chain_extras(entry.posecraft_name, guess, pose_bones)
                    extras += rigify_chain_extras(entry.posecraft_name, guess, pose_bones)
                    entry.extra_bones = ", ".join(extras)
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


def _scene_fps(scene) -> float:
    render = scene.render
    return float(render.fps) / float(render.fps_base or 1.0)


# The animation being received in chunks: its identity, the previous chunk's
# last solved frame (the next chunk blends from it) and the root motion's
# starting point. Reset by every first chunk.
_animation_chunk_state = {}


def _rig_leg_length(armature_obj, constraints=None):
    """The rig's thigh + shin length (armature units) through the same bone
    resolution a pose uses, or None when the rig has no matching legs."""
    lengths = []
    for exported_name in ("thigh.L", "shin.L"):
        chain = _resolve_bone_chain(armature_obj, exported_name, constraints)
        if chain is None:
            return None
        lengths.append(sum(armature_obj.data.bones[name].length for name in chain))
    return sum(lengths) or None


def _apply_root_motion(armature_obj, bone_names, offset, leg_length, state, floor_contact: bool) -> bool:
    """Moves the rig's root bone to the animation's starting location plus
    `offset` (the subject's travel in Blender axes) scaled by rig leg length /
    `leg_length`, then (with floor contact) sets its height as for a single
    pose. Returns whether the root moved."""
    if not bone_names:
        return False
    if "rig_leg_length" not in state:
        state["rig_leg_length"] = _rig_leg_length(armature_obj)
    rig_leg = state["rig_leg_length"]
    if not rig_leg:
        return False
    parents = {bone.name: (bone.parent.name if bone.parent else None) for bone in armature_obj.data.bones}
    root = armature_obj.pose.bones[top_ancestor(bone_names[0], parents)]
    if "base_location" not in state:
        state["base_location"] = root.location.copy()
    travel = Vector(offset) * (rig_leg / leg_length)
    root.location = state["base_location"] + armature_translation_to_location(root.bone.matrix_local.to_quaternion(), travel)
    if floor_contact:
        _keep_feet_on_floor(armature_obj, bone_names)
    return True


def _apply_animation_document(armature_obj, document: dict, scene) -> dict:
    """Keys one animation request (or chunk) on `armature_obj`: the clip is
    resampled to the scene's frame rate (resample_animation, blending from the
    previous chunk's last frame), each key's pose goes through
    apply_pose_document -- the same retarget path as a single pose -- at its
    own Blender frame, plus the root motion when it's on. Everything is
    validated first (parse_animation_document), and the scene's current frame
    is restored afterwards. Returns the reply's animation fields plus the
    union of applied/unmatched/low-confidence bones. (No bpy annotations:
    tests/test_blender_animation.py runs it against a simulated scene.)"""
    global _animation_chunk_state
    animation = parse_animation_document(document)
    original_frame = scene.frame_current
    start_frame = original_frame if animation["start_frame"] is None else animation["start_frame"]
    identity = (animation["first_frame_index"], start_frame, animation["fps"], getattr(armature_obj, "name", None))
    if animation["chunk_index"] == 0 or _animation_chunk_state.get("identity") != identity:
        _animation_chunk_state = {"identity": identity}
    state = _animation_chunk_state
    samples = [(index, pose, animation["root_offsets"].get(index)) for index, pose in animation["frames"]]
    tail = state.get("tail") if animation["chunk_index"] > 0 else None
    keys = resample_animation(
        ([tail] if tail else []) + samples, animation["first_frame_index"], start_frame,
        animation["fps"], _scene_fps(scene), include_first=tail is None,
    )
    state["tail"] = samples[-1]
    root_motion = bool(getattr(scene, "posecraft_root_motion", False)) and animation["root_leg_length"] is not None
    applied_all, unmatched_all, low_all, requested = [], [], [], 0
    try:
        for frame, pose, offset in keys:
            scene.frame_set(frame)  # evaluate the rig (and any keys already there) at this frame
            move_root = root_motion and offset is not None
            applied, unmatched, low_confidence = apply_pose_document(
                armature_obj, pose, scene.posecraft_match_rig_rest, scene.posecraft_floor_contact and not move_root,
                scene.posecraft_pose_controls,
            )
            moved = move_root and _apply_root_motion(
                armature_obj, applied, offset, animation["root_leg_length"], state, scene.posecraft_floor_contact
            )
            _keyframe_applied(armature_obj, applied, scene, frame=frame, advance=False, root_location=moved)
            for target, names in ((applied_all, applied), (unmatched_all, unmatched), (low_all, low_confidence)):
                target.extend(name for name in names if name not in target)
            requested = max(requested, len(pose["bones"]))
    finally:
        scene.frame_set(original_frame)
    return {
        "applied": applied_all,
        "unmatched": unmatched_all,
        "low_confidence": low_all,
        "requested": requested,
        "animation": {
            "animation_frames": len(keys),  # Blender frames keyed by this request
            "start_frame": start_frame,
            "end_frame": keys[-1][0] if keys else None,
            "chunk_index": animation["chunk_index"],
            "chunk_count": animation["chunk_count"],
        },
    }


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
    if isinstance(document, dict) and document.get("format") == LIBRARY_FORMAT_NAME:
        try:
            result = create_library_assets(armature_obj, str(document.get("path") or ""), scene)
        except ValueError as exc:
            _live_link_status = str(exc)
            return build_live_link_reply(error=str(exc))
        _live_link_status = f"Made {result['created']} Pose Library asset(s) on {armature_obj.name}"
        try:
            bpy.ops.ed.undo_push(message="PoseCraft library")
        except RuntimeError:
            pass
        return build_live_link_reply(result["assets"], result["created"] + len(result["failed"]),
                                     armature=armature_obj.name, library=result)
    if is_animation_document(document):
        result = _apply_animation_document(armature_obj, document, scene)
        info = result["animation"]
        summary, _level = _summarize(result["applied"], result["unmatched"], result["low_confidence"])
        _live_link_status = (
            f"Keyed {info['animation_frames']} frame(s) at {info['start_frame']}-{info['end_frame']} "
            f"(part {info['chunk_index'] + 1}/{info['chunk_count']}); {summary}"
        )
        if info["chunk_index"] == info["chunk_count"] - 1:
            try:
                # One Ctrl+Z step for the whole animation: nothing is pushed
                # between its chunks, so undoing returns to before the first.
                bpy.ops.ed.undo_push(message="PoseCraft animation")
            except RuntimeError:
                pass
        return build_live_link_reply(
            result["applied"], result["requested"], result["unmatched"], result["low_confidence"],
            armature_obj.name, animation=info,
        )
    applied, unmatched, low_confidence = apply_pose_document(
        armature_obj, document, scene.posecraft_match_rig_rest, scene.posecraft_floor_contact,
        scene.posecraft_pose_controls,
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
    # Checked in Blender 5.2: without this the persistent timer stayed
    # registered after Stop and after the add-on was disabled, until its next
    # tick noticed the closed socket.
    if bpy.app.timers.is_registered(_poll_live_link_server):
        bpy.app.timers.unregister(_poll_live_link_server)
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
        layout.operator(POSECRAFT_OT_import_library.bl_idname, icon="ASSET_MANAGER")

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
        layout.prop(scene, "posecraft_pose_controls")
        layout.prop(scene, "posecraft_floor_contact")
        layout.prop(scene, "posecraft_root_motion")
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
    POSECRAFT_OT_import_library,
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
    bpy.types.Scene.posecraft_pose_controls = bpy.props.BoolProperty(
        name="Pose rig controls",
        description=(
            "For constraint-driven rigs such as Rigify: pose the control that drives a mapped DEF-/ORG- bone "
            "instead of the bone itself, and switch the Rigify arms and legs being posed to FK"
        ),
        default=True,
    )
    bpy.types.Scene.posecraft_root_motion = bpy.props.BoolProperty(
        name="Root motion (animations)",
        description=(
            "When keying a Live Link animation, move the rig's root with the subject's travel (from planted "
            "feet), scaled to the rig's leg length. Off: the rig animates in place"
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
    del bpy.types.Scene.posecraft_root_motion
    del bpy.types.Scene.posecraft_pose_controls
    del bpy.types.Scene.posecraft_match_rig_rest
    del bpy.types.Scene.posecraft_live_link_port
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

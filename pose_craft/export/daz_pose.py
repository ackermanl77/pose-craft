"""DAZ Studio pose presets (.duf) for Genesis 9.

A pose preset is a DSON file (JSON, `"type": "preset_pose"`) whose "animations"
set each bone's rotation channels on the selected figure:
"name://@selection/l_upperarm:?rotation/x/value" -> degrees. Like DAZ's own
presets it sets every bone of the figure (bones PoseCraft doesn't pose are
zeroed) and the hip's height, so loading it replaces the previous pose.

The channel values depend on the figure's own joints, so they are computed
from the user's installed Genesis 9 figure file (data/DAZ 3D/Genesis 9/Base/
Genesis9.dsf in the DAZ content library -- read at export time, never shipped
with PoseCraft): each bone's parent, rotation order, orientation and rest
head/tail. Measured in DAZ Studio 6.25 (tests/test_real_daz.py): a bone's
rotation is O * E * O^-1, where O is its orientation (the .dsf's x/y/z
orientation as extrinsic XYZ Euler degrees) and E the x/y/z channels applied
as extrinsic rotations in the bone's rotation order (e.g. "YZX"); every
Genesis 9 bone's rest world rotation is the identity, and a bone's world
rotation is its parent's times its own. PoseCraft's engine axes are DAZ's
(Y up, the figure facing +Z, its left on +X), so no axis change is needed.

Retargeting (GENESIS9_BONE_MAP): PoseCraft's single torso bone ("spine")
drives Genesis 9's hip -- the legs hang from it as they hang from PoseCraft's
torso -- and pelvis, spine1-4, the collars, neck2 and the twist bones stay at
rest. Every other mapped bone is aimed: its posed world rotation is
PoseCraft's world rotation times the swing that takes the Genesis 9 rest
direction onto PoseCraft's T-pose rest direction, which absorbs Genesis 9's
A-pose arms. Channels are solved parents first within the figure's joint
limits, the children making up for what a parent couldn't do (solve_daz_pose).
"""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.retarget.skeleton import bone_rest_direction, world_rotations_from_local

GENESIS9_DSF = Path("data") / "DAZ 3D" / "Genesis 9" / "Base" / "Genesis9.dsf"
LIBRARY_ENV = "POSECRAFT_DAZ_LIBRARY"  # one or more content library folders (os.pathsep-separated)

GENESIS9_BONE_MAP: dict[str, str] = {
    "spine": "hip",
    "neck": "neck1",
    "head": "head",
    **{f"{part}.{side}": f"{side.lower()}_{daz}" for side in ("L", "R") for part, daz in (
        ("upper_arm", "upperarm"), ("forearm", "forearm"), ("hand", "hand"),
        ("thigh", "thigh"), ("shin", "shin"), ("foot", "foot"),
    )},
    **{f"{finger}.0{i}.{side}": f"{side.lower()}_{daz}{i}" for side in ("L", "R") for finger, daz in (
        ("thumb", "thumb"), ("f_index", "index"), ("f_middle", "mid"), ("f_ring", "ring"), ("f_pinky", "pinky"),
    ) for i in (1, 2, 3)},
}
# Mapped bones whose world rotation is applied as is instead of aimed: the root, whose rest frame is
# the world's in both skeletons (its Genesis 9 head -> tail even points down, into the pelvis).
_UNAIMED = {"spine"}


class DazFigureNotFound(FileNotFoundError):
    """No Genesis 9 figure file in the DAZ content libraries looked at."""


@dataclass(frozen=True)
class DazBone:
    name: str
    parent: str | None
    rotation_order: str  # e.g. "YZX"
    orientation: Rotation
    origin: np.ndarray  # rest head, cm
    end: np.ndarray  # rest tail, cm
    # (min, max) degrees per clamped rotation channel ("x"/"y"/"z"); DAZ clamps a loaded value to it.
    limits: dict[str, tuple[float, float]] | None = None


@dataclass(frozen=True)
class DazFigure:
    name: str
    bones: dict[str, DazBone]  # parents before children
    path: Path


def daz_library_folders() -> list[Path]:
    """Where DAZ content libraries usually are: $POSECRAFT_DAZ_LIBRARY, then DAZ Studio's defaults."""
    folders = [Path(p) for p in os.environ.get(LIBRARY_ENV, "").split(os.pathsep) if p.strip()]
    public = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    folders += [public / "Documents" / "My DAZ 3D Library", Path.home() / "Documents" / "DAZ 3D" / "Studio" / "My Library"]
    return folders


def find_genesis9_figure(libraries: list[Path] | None = None) -> Path:
    for folder in libraries if libraries is not None else daz_library_folders():
        candidate = Path(folder) / GENESIS9_DSF
        if candidate.is_file():
            return candidate
    raise DazFigureNotFound(
        "Genesis 9 wasn't found in the DAZ content library (looked for "
        f"{GENESIS9_DSF.as_posix()} in {', '.join(str(f) for f in (libraries or daz_library_folders()))}). "
        f"Install Genesis 9 Starter Essentials, or set {LIBRARY_ENV} to your content library folder."
    )


def _read_dson(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # DAZ saves compressed files too
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _xyz(entries) -> np.ndarray:
    values = {entry["id"]: float(entry.get("value", 0.0)) for entry in entries or ()}
    return np.array([values.get("x", 0.0), values.get("y", 0.0), values.get("z", 0.0)])


def _limits(entries) -> dict[str, tuple[float, float]]:
    limits = {}
    for entry in entries or ():
        if entry.get("clamped") and "min" in entry and "max" in entry:
            limits[entry["id"]] = (float(entry["min"]), float(entry["max"]))
    return limits


def load_daz_figure(path: Path | None = None) -> DazFigure:
    """A figure's skeleton from its .dsf (default: the installed Genesis 9)."""
    path = Path(path) if path is not None else find_genesis9_figure()
    data = _read_dson(path)
    nodes = data.get("node_library") or []
    figure = next((n for n in nodes if n.get("type") == "figure"), None)
    if figure is None:
        raise ValueError(f"{path} has no figure in its node library")
    bones: dict[str, DazBone] = {}
    pending = [n for n in nodes if n.get("type") == "bone"]
    while pending:  # parents first, whatever the file's order
        progressed = False
        for node in list(pending):
            parent = str(node.get("parent", "")).lstrip("#") or None
            if parent == figure["id"]:
                parent = None
            if parent is not None and parent not in bones:
                continue
            bones[node["id"]] = DazBone(
                name=node["id"], parent=parent, rotation_order=str(node.get("rotation_order", "XYZ")).upper(),
                orientation=Rotation.from_euler("xyz", _xyz(node.get("orientation")), degrees=True),
                origin=_xyz(node.get("center_point")), end=_xyz(node.get("end_point")),
                limits=_limits(node.get("rotation")),
            )
            pending.remove(node)
            progressed = True
        if not progressed:
            raise ValueError(f"{path}: bones whose parent isn't in the figure: {[n['id'] for n in pending][:5]}")
    return DazFigure(name=str(figure.get("name") or figure["id"]), bones=bones, path=path)


def _swing(source: np.ndarray, target: np.ndarray) -> Rotation:
    """The shortest-arc rotation taking direction `source` onto `target`."""
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    axis = np.cross(source, target)
    sin, cos = np.linalg.norm(axis), float(np.dot(source, target))
    if sin < 1e-9:
        if cos > 0:
            return Rotation.identity()
        perpendicular = np.cross(source, [1.0, 0.0, 0.0] if abs(source[0]) < 0.9 else [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(np.pi * perpendicular / np.linalg.norm(perpendicular))
    return Rotation.from_rotvec(axis / sin * np.arctan2(sin, cos))


def posed_world_rotations(bone_rotations: dict[str, Rotation], figure: DazFigure,
                          bone_map: dict[str, str] = GENESIS9_BONE_MAP) -> dict[str, Rotation]:
    """Every figure bone's posed world rotation (relative to its rest): mapped bones from PoseCraft
    (see the module docstring), the rest following their parent."""
    pc_world = world_rotations_from_local(bone_rotations)
    targets: dict[str, Rotation] = {}
    for pc_name, daz_name in bone_map.items():
        if pc_name not in pc_world or daz_name not in figure.bones:
            continue
        if pc_name in _UNAIMED:
            targets[daz_name] = pc_world[pc_name]
            continue
        bone = figure.bones[daz_name]
        targets[daz_name] = pc_world[pc_name] * _swing(bone.end - bone.origin, bone_rest_direction(pc_name))
    world: dict[str, Rotation] = {}
    for name, bone in figure.bones.items():
        parent_world = world[bone.parent] if bone.parent is not None else Rotation.identity()
        world[name] = targets.get(name, parent_world)
    return world


@dataclass(frozen=True)
class DazPoseSolution:
    channels: dict[str, tuple[float, float, float]]  # (x, y, z) degrees per bone, within the joint limits
    world: dict[str, Rotation]  # what those channels pose each bone to (DAZ's own result)
    clamped: list[tuple[str, str, float, float]]  # (bone, axis, wanted, limit) beyond a joint's range


def solve_daz_pose(bone_rotations: dict[str, Rotation], figure: DazFigure,
                   bone_map: dict[str, str] = GENESIS9_BONE_MAP, tolerance: float = 0.5) -> DazPoseSolution:
    """Channel values for every bone, solved parents first the way DAZ applies them: each bone's
    channels are clamped to its joint limits (as DAZ does on load) and its children are solved
    against where it really ends up, so a child makes up for what its parent couldn't do (Genesis 9's
    forearm can't rotate about Z at all; the hand takes that over). `clamped` lists the channels
    that wanted more than their range by over `tolerance` degrees -- axes a joint can't turn at all
    (min == max) aren't listed: they're part of the rig, not a pose problem."""
    targets = posed_world_rotations(bone_rotations, figure, bone_map)
    channels: dict[str, tuple[float, float, float]] = {}
    world: dict[str, Rotation] = {}
    clamped: list[tuple[str, str, float, float]] = []
    for name, bone in figure.bones.items():
        parent_world = world[bone.parent] if bone.parent is not None else Rotation.identity()
        local = parent_world.inv() * targets[name]
        euler = (bone.orientation.inv() * local * bone.orientation).as_euler(bone.rotation_order.lower(), degrees=True)
        values = {axis: 0.0 if abs(v) < 1e-9 else float(v) for axis, v in zip(bone.rotation_order, euler)}
        for axis, (low, high) in (bone.limits or {}).items():
            key = axis.upper()
            wanted = values[key]
            values[key] = min(max(wanted, low), high)
            if low != high and abs(values[key] - wanted) > tolerance:
                clamped.append((name, axis, wanted, values[key]))
        channels[name] = (values["X"], values["Y"], values["Z"])
        applied = Rotation.from_euler(bone.rotation_order.lower(), [values[a] for a in bone.rotation_order], degrees=True)
        world[name] = parent_world * bone.orientation * applied * bone.orientation.inv()
    return DazPoseSolution(channels=channels, world=world, clamped=clamped)


def daz_rotation_channels(bone_rotations: dict[str, Rotation], figure: DazFigure,
                          bone_map: dict[str, str] = GENESIS9_BONE_MAP) -> dict[str, tuple[float, float, float]]:
    """(x, y, z) rotation channel values in degrees for every bone of `figure` (solve_daz_pose)."""
    return solve_daz_pose(bone_rotations, figure, bone_map).channels


def channels_past_limits(bone_rotations: dict[str, Rotation], figure: DazFigure) -> list[tuple[str, str, float, float]]:
    """(bone, axis, wanted, limit) for each channel the pose would push past the figure's joint
    limits (see solve_daz_pose): that part of the pose comes out short in DAZ."""
    return solve_daz_pose(bone_rotations, figure).clamped


def hip_height_offset(figure: DazFigure, hip_height: float | None, leg_length: float | None) -> float:
    """The hip's Y translation (cm) that puts it at the subject's hip-height-to-leg-length ratio on
    this figure -- a crouch comes out lowered. 0 without the subject's measurements."""
    if not hip_height or not leg_length or "hip" not in figure.bones:
        return 0.0
    if "l_thigh" not in figure.bones or "l_shin" not in figure.bones:
        return 0.0
    figure_leg = sum(float(np.linalg.norm(figure.bones[n].end - figure.bones[n].origin)) for n in ("l_thigh", "l_shin"))
    return float(hip_height / leg_length * figure_leg - figure.bones["hip"].origin[1])


def build_daz_pose_document(
    bone_rotations: dict[str, Rotation],
    figure: DazFigure,
    *,
    hip_height: float | None = None,
    leg_length: float | None = None,
    name: str = "PoseCraft pose",
) -> dict:
    """The .duf pose preset for `figure` (see the module docstring)."""
    animations = []
    for bone, values in daz_rotation_channels(bone_rotations, figure).items():
        for axis, value in zip("xyz", values):
            animations.append({"url": f"name://@selection/{bone}:?rotation/{axis}/value", "keys": [[0, round(value, 4)]]})
    animations.append({
        "url": "name://@selection/hip:?translation/y/value",
        "keys": [[0, round(hip_height_offset(figure, hip_height, leg_length), 4)]],
    })
    return {
        "file_version": "0.6.0.0",
        "asset_info": {
            "id": f"/PoseCraft/{name}.duf".replace(" ", "%20"),
            "type": "preset_pose",
            "contributor": {"author": "PoseCraft", "email": "", "website": ""},
            "revision": "1.0",
            "modified": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "scene": {"animations": animations},
    }


def save_daz_pose_file(
    bone_rotations: dict[str, Rotation],
    path: str | Path,
    figure: DazFigure | None = None,
    *,
    hip_height: float | None = None,
    leg_length: float | None = None,
) -> Path:
    """Writes a Genesis 9 pose preset to `path` (figure: default the installed Genesis 9)."""
    figure = figure if figure is not None else load_daz_figure()
    path = Path(path)
    document = build_daz_pose_document(bone_rotations, figure, hip_height=hip_height, leg_length=leg_length,
                                       name=path.stem)
    path.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return path

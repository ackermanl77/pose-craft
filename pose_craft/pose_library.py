"""The pose library (report item F15): estimated poses kept for later, with a
thumbnail, a name and tags, re-exported, mirrored or blended without the
original photo.

Each entry is a folder in the library (POSECRAFT_LIBRARY_DIR, else
%LOCALAPPDATA%\\PoseCraft\\pose_library, ~/.local/share/PoseCraft/pose_library
elsewhere) holding `meta.json`, `pose.json` and `thumb.png`. There are two
kinds:

- "person": a solved person exactly as the window had it -- the detected
  landmarks (body and hands, image positions, visibility, presence), the
  manual edits (F4), the photo's aspect, the orientation fix and the mirror
  switch. Every export re-runs the same posing.document_inputs()/root_for()
  the window's exports use, so a library export equals exporting from the
  photo, and "Mirror" is just the mirror switch flipped.
- "blend": a mix of two entries -- each bone's rotation slerped between
  theirs, hip height and leg length interpolated -- stored as rotations.

No Qt here: the window's library dialog and the tests use it the same way.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pose_craft.estimation.landmarker import DetectedPerson, HandLandmarks
from pose_craft.posing import PersonEdits, document_inputs, edited_person, out_of_frame_bones, root_for
from pose_craft.retarget.skeleton import ORIENTATION_CORRECTIONS

FORMAT = "posecraft_library_entry"
VERSION = 1
THUMB_SIDE = 256
EXPORT_FORMATS = (".json", ".bvh", ".glb", ".duf")


class LibraryError(ValueError):
    """A missing or unreadable entry, or a request the entry can't satisfy."""


@dataclasses.dataclass
class PoseEntry:
    id: str
    name: str
    tags: list[str]
    created: float
    kind: str  # "person" or "blend"
    source: str = ""  # the photo's file name, or "blend of A and B"
    mirror: bool = False
    orientation: str = "Default"

    def matches(self, text: str = "", tag: str = "") -> bool:
        if tag and tag.lower() not in (t.lower() for t in self.tags):
            return False
        haystack = " ".join([self.name, self.source, *self.tags]).lower()
        return all(word in haystack for word in text.lower().split())


@dataclasses.dataclass
class ExportInputs:
    """What every exporter takes, for one library entry."""

    rotations: dict[str, Rotation]
    confidence: dict[str, float]
    hip_height: float | None
    leg_length: float | None
    out_of_frame: set[str]


def default_library_dir() -> Path:
    override = os.environ.get("POSECRAFT_LIBRARY_DIR")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PoseCraft" / "pose_library"
    return Path.home() / ".local" / "share" / "PoseCraft" / "pose_library"


def clean_tags(tags) -> list[str]:
    """Tags from a list or a comma-separated string: trimmed, unique (case-insensitively), in order."""
    if isinstance(tags, str):
        tags = tags.split(",")
    seen, result = set(), []
    for tag in tags or []:
        tag = " ".join(str(tag).split())
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            result.append(tag)
    return result


# ------------------------------------------------------------- serialization


def _array(value) -> list | None:
    return None if value is None else np.asarray(value, dtype=float).tolist()


def person_to_json(person: DetectedPerson) -> dict:
    return {
        "landmarks": _array(person.landmarks),
        "image_xy": _array(person.image_landmarks_xy),
        "visibility": _array(person.visibility),
        "landmark_presence": _array(person.landmark_presence),
        "hands": {
            side: {"world": _array(hand.world), "image_xy": _array(hand.image_xy), "presence": float(hand.presence)}
            for side, hand in person.hands.items()
        },
    }


def person_from_json(data: dict) -> DetectedPerson:
    presence = data.get("landmark_presence")
    return DetectedPerson(
        landmarks=np.asarray(data["landmarks"], dtype=float),
        image_landmarks_xy=np.asarray(data["image_xy"], dtype=float),
        visibility=np.asarray(data["visibility"], dtype=float),
        hands={
            side: HandLandmarks(np.asarray(h["world"], dtype=float), np.asarray(h["image_xy"], dtype=float),
                                float(h["presence"]))
            for side, h in (data.get("hands") or {}).items()
        },
        landmark_presence=None if presence is None else np.asarray(presence, dtype=float),
    )


def edits_to_json(edits: PersonEdits | None) -> dict:
    edits = edits or PersonEdits()
    return {
        "flipped": sorted(edits.flipped),
        "overrides": [[name, list(angles)] for name, angles in edits.overrides],
        "moved": [[index, list(xy)] for index, xy in edits.moved],
    }


def edits_from_json(data: dict | None) -> PersonEdits:
    data = data or {}
    return PersonEdits(
        flipped=frozenset(data.get("flipped", [])),
        overrides=tuple((str(name), tuple(float(a) for a in angles)) for name, angles in data.get("overrides", [])),
        moved=tuple((int(index), tuple(float(v) for v in xy)) for index, xy in data.get("moved", [])),
    )


def _rotations_to_json(rotations: dict[str, Rotation]) -> dict:
    return {name: rotation.as_quat().tolist() for name, rotation in rotations.items()}


def _rotations_from_json(data: dict) -> dict[str, Rotation]:
    return {name: Rotation.from_quat(quat) for name, quat in data.items()}


# ---------------------------------------------------------------- thumbnails


def person_thumbnail(image_bgr: np.ndarray, image_xy: np.ndarray, side: int = THUMB_SIDE) -> np.ndarray:
    """A square crop of the photo around the person (their landmarks' box plus
    a margin), scaled to `side`, padded with dark grey where the crop leaves
    the photo."""
    h, w = image_bgr.shape[:2]
    xy = np.asarray(image_xy, dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
    else:
        x0, y0 = xy.min(axis=0) * (w, h)
        x1, y1 = xy.max(axis=0) * (w, h)
    size = max(x1 - x0, y1 - y0) * 1.25 + 8
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - size / 2)), int(round(cy - size / 2))
    size = int(round(size))
    canvas = np.full((size, size, 3), 40, dtype=np.uint8)
    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + size), min(h, top + size)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = image_bgr[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA)


def _write_png(path: Path, image_bgr: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image_bgr)
    if not ok:
        raise LibraryError("could not encode the thumbnail")
    path.write_bytes(data.tobytes())  # not cv2.imwrite: it fails on non-ASCII paths


def read_thumbnail(path: Path) -> np.ndarray | None:
    try:
        return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


# ------------------------------------------------------------------- library


class PoseLibrary:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else default_library_dir()

    def _folder(self, entry_id: str) -> Path:
        if not entry_id or not all(c.isalnum() or c == "-" for c in entry_id):
            raise LibraryError(f"bad library entry id {entry_id!r}")
        return self.root / entry_id

    def thumbnail_path(self, entry_id: str) -> Path:
        return self._folder(entry_id) / "thumb.png"

    def _write(self, entry: PoseEntry, pose: dict, thumbnail: np.ndarray | None) -> PoseEntry:
        folder = self._folder(entry.id)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "pose.json").write_text(json.dumps({"format": FORMAT, "version": VERSION, **pose}), encoding="utf-8")
        self._write_meta(entry)
        if thumbnail is not None:
            _write_png(folder / "thumb.png", thumbnail)
        return entry

    def _write_meta(self, entry: PoseEntry) -> None:
        path = self._folder(entry.id) / "meta.json"
        tmp = path.with_name("meta.json.tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(entry), indent=1), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def add_person(
        self,
        person: DetectedPerson,
        *,
        name: str,
        tags=(),
        edits: PersonEdits | None = None,
        aspect: float = 1.0,
        orientation: str = "Default",
        mirror: bool = False,
        source: str = "",
        image_bgr: np.ndarray | None = None,
    ) -> PoseEntry:
        if orientation not in ORIENTATION_CORRECTIONS:
            raise LibraryError(f"unknown orientation {orientation!r}")
        entry = PoseEntry(self._new_id(), name.strip() or "Pose", clean_tags(tags), time.time(), "person",
                          source, bool(mirror), orientation)
        thumbnail = None
        if image_bgr is not None:
            thumbnail = person_thumbnail(image_bgr, edited_person(person, edits, aspect).image_xy)
            if mirror:
                thumbnail = cv2.flip(thumbnail, 1)
        pose = {"person": person_to_json(person), "edits": edits_to_json(edits), "aspect": float(aspect)}
        return self._write(entry, pose, thumbnail)

    def entries(self, text: str = "", tag: str = "") -> list[PoseEntry]:
        """Every readable entry, newest first, optionally filtered."""
        found = []
        if not self.root.is_dir():
            return found
        for folder in self.root.iterdir():
            try:
                entry = PoseEntry(**json.loads((folder / "meta.json").read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue  # not an entry (or a half-written one): skip it
            if entry.matches(text, tag):
                found.append(entry)
        return sorted(found, key=lambda e: e.created, reverse=True)

    def tags(self) -> list[str]:
        return sorted({tag for entry in self.entries() for tag in entry.tags}, key=str.lower)

    def get(self, entry_id: str) -> PoseEntry:
        try:
            return PoseEntry(**json.loads((self._folder(entry_id) / "meta.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise LibraryError(f"no library entry {entry_id}") from exc

    def _pose(self, entry_id: str) -> dict:
        try:
            data = json.loads((self._folder(entry_id) / "pose.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LibraryError(f"library entry {entry_id} can't be read") from exc
        if data.get("format") != FORMAT or data.get("version", 0) > VERSION:
            raise LibraryError(f"library entry {entry_id} is from a newer PoseCraft")
        return data

    def update(self, entry_id: str, *, name: str | None = None, tags=None) -> PoseEntry:
        entry = self.get(entry_id)
        if name is not None:
            entry.name = name.strip() or entry.name
        if tags is not None:
            entry.tags = clean_tags(tags)
        self._write_meta(entry)
        return entry

    def delete(self, entry_id: str) -> None:
        folder = self._folder(entry_id)
        if folder.is_dir():
            shutil.rmtree(folder)

    def export_inputs(self, entry_id: str) -> ExportInputs:
        entry, pose = self.get(entry_id), self._pose(entry_id)
        if entry.kind == "blend":
            return ExportInputs(_rotations_from_json(pose["rotations"]), dict(pose.get("confidence") or {}),
                                pose.get("hip_height"), pose.get("leg_length"), set(pose.get("out_of_frame") or []))
        person = person_from_json(pose["person"])
        edits = edits_from_json(pose.get("edits"))
        aspect = float(pose.get("aspect", 1.0))
        rotations, confidence = document_inputs(person, entry.mirror, ORIENTATION_CORRECTIONS[entry.orientation],
                                                edits, aspect)
        edited = edited_person(person, edits, aspect)
        hip_height, leg_length = root_for(edited, entry.mirror, rotations)
        return ExportInputs(rotations, confidence, hip_height, leg_length, out_of_frame_bones(edited, entry.mirror))

    def mirrored_copy(self, entry_id: str) -> PoseEntry:
        """A new entry: the same person with the mirror switch flipped."""
        entry, pose = self.get(entry_id), self._pose(entry_id)
        if entry.kind != "person":
            raise LibraryError("only poses from a photo can be mirrored (mirror the poses before blending them)")
        copy = dataclasses.replace(entry, id=self._new_id(), name=f"{entry.name} (mirrored)", created=time.time(),
                                   mirror=not entry.mirror)
        thumb = read_thumbnail(self.thumbnail_path(entry_id))
        pose = {k: v for k, v in pose.items() if k not in ("format", "version")}
        return self._write(copy, pose, None if thumb is None else cv2.flip(thumb, 1))

    def blend(self, first_id: str, second_id: str, amount: float, *, name: str = "", tags=()) -> PoseEntry:
        """A new entry `amount` of the way from the first pose to the second (0..1)."""
        amount = float(min(1.0, max(0.0, amount)))
        a, b = self.export_inputs(first_id), self.export_inputs(second_id)
        rotations = {}
        for bone in sorted(set(a.rotations) | set(b.rotations)):
            if bone in a.rotations and bone in b.rotations:
                pair = Rotation.from_quat(np.stack([a.rotations[bone].as_quat(), b.rotations[bone].as_quat()]))
                rotations[bone] = Slerp([0.0, 1.0], pair)([amount])[0]
            else:  # a finger bone only one of them has
                rotations[bone] = a.rotations[bone] if bone in a.rotations else b.rotations[bone]

        def mix(x, y):
            return None if x is None or y is None else float(x + (y - x) * amount)

        confidence = {bone: min(a.confidence.get(bone, 1.0), b.confidence.get(bone, 1.0)) for bone in rotations}
        first, second = self.get(first_id), self.get(second_id)
        entry = PoseEntry(self._new_id(), name.strip() or f"{first.name} / {second.name} {round(amount * 100)}%",
                          clean_tags(tags), time.time(), "blend", f"blend of {first.name} and {second.name}")
        pose = {
            "rotations": _rotations_to_json(rotations),
            "confidence": confidence,
            "hip_height": mix(a.hip_height, b.hip_height),
            "leg_length": mix(a.leg_length, b.leg_length),
            "out_of_frame": sorted(a.out_of_frame | b.out_of_frame),
            "blend": {"first": first_id, "second": second_id, "amount": amount},
        }
        thumbs = [read_thumbnail(self.thumbnail_path(i)) for i in (first_id, second_id)]
        thumbnail = None
        if all(t is not None for t in thumbs):
            half = THUMB_SIDE // 2
            thumbnail = np.hstack([cv2.resize(t, (half, THUMB_SIDE)) for t in thumbs])
        return self._write(entry, pose, thumbnail)

    def live_link_document(self, entry_id: str) -> dict:
        """The entry as the pose document Live Link sends to Blender."""
        from pose_craft.export.blender_pose import build_pose_document

        inputs = self.export_inputs(entry_id)
        return build_pose_document(
            inputs.rotations, source_note=f"PoseCraft library: {self.get(entry_id).name}",
            bone_confidence=inputs.confidence, hip_height=inputs.hip_height, leg_length=inputs.leg_length,
            out_of_frame=inputs.out_of_frame,
        )

    def export(self, entry_id: str, path: str | Path, figure=None) -> Path:
        """Writes the entry as a pose JSON (.json), BVH (.bvh), binary glTF
        (.glb) or DAZ Studio preset (.duf, Genesis 9; `figure` from
        daz_pose.load_daz_figure()) -- the same writers and arguments as the
        window's exports."""
        from pose_craft.export.blender_pose import save_pose_file
        from pose_craft.export.bvh_export import floor_root_position, save_bvh_file

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in EXPORT_FORMATS:
            raise LibraryError(f"export as one of {', '.join(EXPORT_FORMATS)}, not {suffix or 'no extension'}")
        inputs = self.export_inputs(entry_id)
        if suffix == ".json":
            save_pose_file(inputs.rotations, path, source_note=f"PoseCraft library: {self.get(entry_id).name}",
                           bone_confidence=inputs.confidence, hip_height=inputs.hip_height,
                           leg_length=inputs.leg_length, out_of_frame=inputs.out_of_frame)
        elif suffix == ".bvh":
            save_bvh_file(inputs.rotations, path, root_position=floor_root_position(inputs.rotations))
        elif suffix == ".glb":
            from pose_craft.export.gltf_export import save_gltf_pose

            save_gltf_pose(inputs.rotations, path, root_position=floor_root_position(inputs.rotations))
        else:
            from pose_craft.export.daz_pose import load_daz_figure, save_daz_pose_file

            save_daz_pose_file(inputs.rotations, path, figure or load_daz_figure(),
                               hip_height=inputs.hip_height, leg_length=inputs.leg_length)
        return path

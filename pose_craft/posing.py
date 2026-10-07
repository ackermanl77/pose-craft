"""Turning a detected person into what gets exported, without Qt (B13): manual
corrections (PersonEdits, applied by edited_person), the solved bone rotations
and per-bone confidence (document_inputs), the bones reaching outside the photo
(out_of_frame_bones) and the root measurements (root_for). Shared by the GUI
(gui/main_window.py) and the folder batch (pose_craft/batch.py, pose_craft/cli.py),
so a batch export matches a manual one exactly."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.estimation.landmarker import DetectedPerson
from pose_craft.export.blender_pose import root_measurements
from pose_craft.retarget.skeleton import (
    apply_bone_overrides,
    apply_orientation_correction,
    bone_axes_rotation,
    bone_confidence,
    bones_out_of_frame,
    flip_hand_depth,
    flip_limb_depth,
    mirror_hands,
    mirror_landmarks,
    move_joint,
    solve_bone_rotations,
)


def default_export_name(source: str | None, person_index: int, suffix: str) -> str:
    """"<photo stem>_person<N><suffix>", or "pose_person<N>" with no photo."""
    stem = Path(source).stem if source else "pose"
    return f"{stem}_person{person_index + 1}{suffix}"


@dataclasses.dataclass
class EditedPerson:
    """A person after their manual edits (moved joints, then depth flips),
    before mirroring: what every solve, export and overlay uses."""

    landmarks: np.ndarray
    image_xy: np.ndarray
    visibility: np.ndarray
    hands: dict  # "L"/"R" -> (21, 3) engine-frame hand landmarks (B11)
    hands_xy: dict  # "L"/"R" -> (21, 2) normalized
    hand_presence: dict  # "L"/"R" -> score
    # (33,) per-landmark presence (B10), a dragged joint counted as in frame; None when unknown.
    landmark_presence: np.ndarray | None = None

    def out_of_frame(self) -> set[str]:
        """Body bones depending on a landmark outside the photo (B10)."""
        return bones_out_of_frame(self.landmark_presence)


def edited_person(person: DetectedPerson, edits: PersonEdits | None, aspect: float = 1.0) -> EditedPerson:
    """Applies a person's edits: each moved joint in landmark order (a parent
    joint before its child, so a moved elbow then wrist keeps the forearm's
    length from the moved elbow), then the depth flips (hands included)."""
    landmarks = np.asarray(person.landmarks, dtype=float)
    image_xy = np.asarray(person.image_landmarks_xy, dtype=float)
    hands = person.hand_world() if hasattr(person, "hand_world") else {}
    hands_xy = person.hand_image_xy() if hasattr(person, "hand_image_xy") else {}
    if edits is not None:
        for joint, xy in edits.moved:
            landmarks, image_xy, hands, hands_xy = move_joint(
                landmarks, image_xy, joint, xy, aspect, person.visibility, hands, hands_xy
            )
        for bone in sorted(edits.flipped):
            hands = flip_hand_depth(landmarks, hands, bone)
            landmarks = flip_limb_depth(landmarks, bone)
    presence = person.hand_presence() if hasattr(person, "hand_presence") else {}
    landmark_presence = getattr(person, "landmark_presence", None)
    if landmark_presence is not None and edits is not None and edits.moved:
        landmark_presence = np.array(landmark_presence, dtype=float)
        for joint, _xy in edits.moved:
            landmark_presence[joint] = 1.0  # placed by hand inside the photo
    return EditedPerson(landmarks, image_xy, np.asarray(person.visibility), hands, hands_xy, presence, landmark_presence)


@dataclasses.dataclass(frozen=True)
class PersonEdits:
    """Manual corrections for one person (F4): limbs whose depth is flipped
    (raw, unmirrored bone names -- what the user clicked in the photo), and
    per-bone rotation adjustments in degrees about the bone's own X/Y/Z axes
    (exported bone names, as in the "Adjust bone" picker), and joints dragged
    on the photo to a new image position (raw landmark indices; photos only)."""

    flipped: frozenset = frozenset()
    overrides: tuple = ()  # ((bone name, (x, y, z) degrees), ...), sorted
    moved: tuple = ()  # ((landmark index, (x, y) normalized), ...), sorted by index

    def override_rotations(self) -> dict[str, Rotation]:
        return {name: bone_axes_rotation(name, angles) for name, angles in self.overrides}

    def with_flip_toggled(self, bone: str) -> PersonEdits:
        return dataclasses.replace(self, flipped=self.flipped ^ {bone})

    def with_override(self, bone: str, angles: tuple[float, float, float] | None) -> PersonEdits:
        entries = {name: value for name, value in self.overrides if name != bone}
        if angles is not None and any(abs(a) > 1e-9 for a in angles):
            entries[bone] = tuple(float(a) for a in angles)
        return dataclasses.replace(self, overrides=tuple(sorted(entries.items())))

    def override_for(self, bone: str) -> tuple[float, float, float]:
        return dict(self.overrides).get(bone, (0.0, 0.0, 0.0))

    def with_joint_moved(self, joint: int, xy: tuple[float, float] | None) -> PersonEdits:
        entries = {index: value for index, value in self.moved if index != joint}
        if xy is not None:
            entries[int(joint)] = (float(xy[0]), float(xy[1]))
        return dataclasses.replace(self, moved=tuple(sorted(entries.items())))

    def is_empty(self) -> bool:
        return not self.flipped and not self.overrides and not self.moved


def document_inputs(
    person: DetectedPerson,
    mirror: bool,
    orientation_correction=None,
    edits: PersonEdits | None = None,
    aspect: float = 1.0,
) -> tuple[dict, dict]:
    """(bone rotations, bone confidence) for exporting `person`.

    mirror_landmarks swaps the left/right rows (and, for 3D positions only,
    reflects X) -- a 1-D `visibility` array gets just the row swap, and it needs
    to be swapped the same way `landmarks` is, or the confidence for e.g. a
    mirrored thigh.L would report the ORIGINAL left leg's visibility instead of
    the real (now swapped-in) right leg's.

    orientation_correction (a scipy Rotation, or None for no change) is the
    same body-orientation fix applied live in the 3D preview -- see
    skeleton.apply_orientation_correction -- composed onto the root bone's
    rotation so every export (JSON, BVH, Live Link) agrees with what's shown.

    edits: the person's manual corrections (F4) -- moved joints and depth flips
    on the raw landmarks before mirroring (they name what was clicked in the
    photo), bone adjustments on the solved rotations. aspect: the photo's
    width / height, for turning a joint drag into 3D. Hand landmarks (B11),
    when solved, add finger bones; they're mirrored and flipped with the body.
    """
    edited = edited_person(person, edits, aspect)
    landmarks, visibility, hands, presence = edited.landmarks, edited.visibility, edited.hands, edited.hand_presence
    if mirror:
        landmarks = mirror_landmarks(landmarks)
        visibility = mirror_landmarks(visibility)
        hands = mirror_hands(hands)
        presence = mirror_hands(presence)
    rotations = solve_bone_rotations(landmarks, visibility=visibility, hands=hands)
    if edits is not None and edits.overrides:
        rotations = apply_bone_overrides(rotations, edits.override_rotations())
    if orientation_correction is not None:
        rotations = apply_orientation_correction(rotations, orientation_correction)
    confidence = bone_confidence(visibility, presence)
    return rotations, confidence


def out_of_frame_bones(edited: EditedPerson, mirror: bool) -> set[str]:
    """The bones reaching outside the photo (B10), named as exported: with the
    mirror, the presence rows are swapped the way visibility is."""
    presence = edited.landmark_presence
    if presence is None:
        return set()
    return bones_out_of_frame(mirror_landmarks(presence) if mirror else presence)


def root_for(edited: EditedPerson, mirror: bool, rotations: dict) -> tuple[float, float]:
    """(hip_height, leg_length) in meters for the export's "root" (B6)."""
    landmarks, hands = edited.landmarks, edited.hands
    if mirror:
        landmarks, hands = mirror_landmarks(landmarks), mirror_hands(hands)
    return root_measurements(rotations, landmarks, hands)

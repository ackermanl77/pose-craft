"""Folder batch processing (B13): solve every photo in a folder and write one pose
per detected person, named "<photo stem>_person<N>.json" (or .bvh) -- the same
names and the same export a manual "Export All People" gives. Used by
`python -m pose_craft.cli` and the GUI's File > Batch Process Folder.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from pose_craft.estimation.landmarker import PoseEstimator
from pose_craft.export.blender_pose import save_pose_file
from pose_craft.export.bvh_export import floor_root_position, save_bvh_file
from pose_craft.imaging import imread_unicode
from pose_craft.posing import default_export_name, document_inputs, edited_person, out_of_frame_bones, root_for

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
FORMATS = ("json", "bvh")


@dataclasses.dataclass
class PhotoOutcome:
    photo: Path
    written: list[Path] = dataclasses.field(default_factory=list)
    error: str | None = None  # why nothing was written (unreadable, nobody found, ...)


@dataclasses.dataclass
class BatchResult:
    outcomes: list[PhotoOutcome] = dataclasses.field(default_factory=list)
    cancelled: bool = False

    @property
    def written(self) -> list[Path]:
        return [path for outcome in self.outcomes for path in outcome.written]

    def summary(self) -> str:
        failed = [outcome for outcome in self.outcomes if outcome.error]
        text = f"{len(self.written)} pose(s) from {len(self.outcomes) - len(failed)} of {len(self.outcomes)} photo(s)"
        if self.cancelled:
            text += " (cancelled)"
        return text


def photos_in(folder: Path) -> list[Path]:
    """The photos directly in `folder`, sorted by name (no recursion)."""
    return sorted(path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in PHOTO_SUFFIXES)


def solve_folder(
    input_dir: Path,
    output_dir: Path,
    estimator: PoseEstimator,
    *,
    fmt: str = "json",
    mirror: bool = False,
    overwrite: bool = True,
    on_progress: Callable[[int, int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> BatchResult:
    """Solve every photo in `input_dir` with `estimator` (its max_people,
    group_photo, hands and thresholds apply) and write each person's pose into
    `output_dir`. A photo that can't be read or has nobody in it is recorded in
    its outcome and skipped; `overwrite=False` leaves existing files alone."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    photos = photos_in(Path(input_dir))
    result = BatchResult()
    for number, photo in enumerate(photos, start=1):
        if should_cancel is not None and should_cancel():
            result.cancelled = True
            break
        if on_progress is not None:
            on_progress(number, len(photos), photo)
        outcome = PhotoOutcome(photo)
        result.outcomes.append(outcome)
        image = imread_unicode(photo)
        if image is None:
            outcome.error = "couldn't be read as an image"
            continue
        people = estimator.detect(image)
        if not people:
            outcome.error = "no person detected"
            continue
        aspect = image.shape[1] / image.shape[0]
        for index, person in enumerate(people):
            target = output_dir / default_export_name(str(photo), index, f".{fmt}")
            if target.exists() and not overwrite:
                continue
            rotations, confidence = document_inputs(person, mirror, aspect=aspect)
            try:
                if fmt == "bvh":
                    save_bvh_file(rotations, target, root_position=floor_root_position(rotations))
                else:
                    edited = edited_person(person, None, aspect)
                    hip_height, leg_length = root_for(edited, mirror, rotations)
                    save_pose_file(
                        rotations, target, source_note=f"Exported from PoseCraft (batch: {photo.name})",
                        bone_confidence=confidence, hip_height=hip_height, leg_length=leg_length,
                        out_of_frame=out_of_frame_bones(edited, mirror),
                    )
            except OSError as error:
                outcome.error = f"couldn't write {target.name}: {error}"
                break
            outcome.written.append(target)
    return result

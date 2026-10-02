"""Regenerates the real-photo golden outputs for tests/test_real_photo_golden.py.

For every photo in the golden folder (tests/data/golden by default, or the
POSECRAFT_GOLDEN_DIR environment variable / --dir), runs the whole pipeline --
downscale, BlazePose "full" on the CPU, engine-frame conversion, retargeting --
and writes <photo stem>.golden.json next to it: per person, the 2D landmarks
(pixels), world landmarks (meters, engine frame) and solved local bone
rotations (rotation vectors, radians).

Only commit photos you have the right to redistribute (CC0 or your own), and
regenerate goldens with this script, never by hand -- then review the diff: a
golden change is a behavior change.

    python scripts/generate_goldens.py [--dir PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
MODEL_VARIANT = "full"


def golden_dir() -> Path:
    return Path(os.environ.get("POSECRAFT_GOLDEN_DIR") or ROOT / "tests" / "data" / "golden")


def solve_photo(photo: Path, estimator) -> dict:
    """The pipeline output a golden file stores, for one photo."""
    from pose_craft.imaging import imread_unicode
    from pose_craft.retarget.skeleton import solve_bone_rotations

    image = imread_unicode(str(photo))
    if image is None:
        raise OSError(f"could not read {photo}")
    h, w = image.shape[:2]
    people = []
    for person in estimator.detect(image):
        rotations = solve_bone_rotations(person.landmarks, visibility=person.visibility)
        people.append({
            "image_xy_px": (person.image_landmarks_xy * [w, h]).round(3).tolist(),
            "world_m": person.landmarks.round(5).tolist(),
            "visibility": person.visibility.round(4).tolist(),
            "rotvec": {name: [round(float(v), 6) for v in rot.as_rotvec()] for name, rot in rotations.items()},
        })
    return {"photo": photo.name, "model": MODEL_VARIANT, "device": "CPU", "people": people}


def cpu_estimator():
    """A CPU PoseEstimator for the cached "full" model, or None if it isn't cached."""
    from pose_craft.estimation.landmarker import PoseEstimator, model_cache_path

    model = model_cache_path(MODEL_VARIANT)
    if not model.exists():
        return None
    return PoseEstimator(model, max_people=4, use_gpu=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()
    folder = args.dir or golden_dir()
    estimator = cpu_estimator()
    if estimator is None:
        print("The 'full' model isn't cached; run a solve in the app once (or ensure_model_downloaded).")
        return 1
    photos = sorted(p for p in folder.glob("*") if p.suffix.lower() in PHOTO_SUFFIXES)
    for photo in photos:
        result = solve_photo(photo, estimator)
        photo.with_suffix(".golden.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"{photo.name}: {len(result['people'])} person(s)")
    if not photos:
        print(f"no photos in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

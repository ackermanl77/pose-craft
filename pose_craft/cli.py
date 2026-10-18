"""Command line (B13): solve a folder of photos without the GUI.

    python -m pose_craft.cli --input photos/ --output poses/ [--format json|bvh|duf|glb]
        [--mirror] [--model lite|full|heavy] [--cpu] [--max-people 4]
        [--group-photo] [--hands] [--detection-threshold 0.5] [--no-overwrite]

Writes "<photo stem>_person<N>.json" (or .bvh) per detected person -- the same
files "Export All People" writes -- and prints a line per photo. Exit code 0
when every photo gave at least one pose, 1 when some didn't, 2 on bad input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pose_craft.batch import FORMATS, solve_folder
from pose_craft.estimation.landmarker import MODEL_URLS, PoseEstimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pose_craft.cli", description="Solve every photo in a folder into Blender poses.")
    parser.add_argument("--input", type=Path, required=True, help="folder of photos (not searched recursively)")
    parser.add_argument("--output", type=Path, required=True, help="folder for the pose files (created if missing)")
    parser.add_argument(
        "--format", choices=FORMATS, default="json",
        help="pose JSON for the Blender add-on (default), BVH, or duf: a DAZ Studio pose preset for Genesis 9 "
        "(needs Genesis 9 in the DAZ content library)",
    )
    parser.add_argument("--mirror", action="store_true", help="swap left and right, as the GUI's Mirror option")
    parser.add_argument("--model", choices=list(MODEL_URLS), default="full", help="pose model quality (default: full)")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    parser.add_argument("--max-people", type=int, default=4, help="people to detect per photo (default: 4)")
    parser.add_argument("--group-photo", action="store_true", help="also search tiles for small faces (slower)")
    parser.add_argument("--hands", action="store_true", help="also solve hands and fingers")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="person detector threshold, 0.1-0.9 (default: 0.5)")
    parser.add_argument("--no-overwrite", action="store_true", help="keep pose files that already exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_dir():
        print(f"error: {args.input} is not a folder", file=sys.stderr)
        return 2
    if not 0.1 <= args.detection_threshold <= 0.9:
        print("error: --detection-threshold must be within 0.1-0.9", file=sys.stderr)
        return 2
    with PoseEstimator.for_variant(args.model, max_people=max(1, args.max_people), use_gpu=not args.cpu) as estimator:
        estimator.group_photo = args.group_photo
        estimator.min_detection_confidence = args.detection_threshold
        if args.hands:
            estimator.enable_hands()
            estimator.hands = True
        print(f"pose model: {args.model} on {estimator.device}")
        result = solve_folder(
            args.input, args.output, estimator, fmt=args.format, mirror=args.mirror, overwrite=not args.no_overwrite,
        )
    for outcome in result.outcomes:
        detail = outcome.error or ", ".join(path.name for path in outcome.written) or "kept existing files"
        print(f"{outcome.photo.name}: {detail}")
    print(result.summary())
    return 0 if result.outcomes and not any(outcome.error for outcome in result.outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Pre-converts the pose models for a frozen build (K3).

Downloads each model variant's pinned .task bundle (verifying its SHA-256, see
landmarker.MODEL_SHA256), converts it to ONNX with the project's own
converter, and writes the results to an output folder (default: build/models)
with a manifest.json that pose_craft.estimation.landmarker.bundled_onnx_paths()
reads. The pose detector is byte-identical across lite/full/heavy, so it's
stored once (12 MB instead of 36 MB). The hand landmarker's 21-point model
(B11, the "Hands and fingers" option) is converted too, under the manifest's
"hand" key (--no-hands leaves it out).

PoseCraft.spec bundles this folder when it holds a manifest, and then leaves
the onnx/tflite converter packages out of the build: the frozen app needs no
download and no network on first run.

    python scripts/convert_models.py [--out build/models] [--variants lite full heavy] [--no-hands]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def convert(out_dir: Path, variants: list[str], hands: bool = True) -> dict:
    """Writes the models and manifest into out_dir and returns the manifest."""
    from pose_craft.estimation.landmarker import (
        BUNDLED_MANIFEST,
        ensure_hand_model_downloaded,
        ensure_hand_onnx_model,
        ensure_model_downloaded,
        ensure_onnx_models,
        file_sha256,
    )
    from pose_craft.estimation.tflite_to_onnx import CONVERTER_VERSION

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"converter_version": CONVERTER_VERSION, "variants": {}, "sha256": {}}
    detectors: dict[str, str] = {}  # sha256 -> file name, so identical detectors are stored once
    for variant in variants:
        task = ensure_model_downloaded(variant)
        paths = ensure_onnx_models(task)
        detector_hash = file_sha256(paths["detector"])
        if detector_hash not in detectors:
            name = "pose_detector.onnx" if not detectors else f"pose_detector_{variant}.onnx"
            shutil.copyfile(paths["detector"], out_dir / name)
            detectors[detector_hash] = name
            manifest["sha256"][name] = detector_hash
        landmarks_name = f"pose_landmarks_{variant}.onnx"
        shutil.copyfile(paths["landmarks"], out_dir / landmarks_name)
        manifest["sha256"][landmarks_name] = file_sha256(out_dir / landmarks_name)
        manifest["variants"][variant] = {"detector": detectors[detector_hash], "landmarks": landmarks_name}
    if hands:
        hand_name = "hand_landmarks.onnx"
        shutil.copyfile(ensure_hand_onnx_model(ensure_hand_model_downloaded()), out_dir / hand_name)
        manifest["sha256"][hand_name] = file_sha256(out_dir / hand_name)
        manifest["hand"] = {"landmarks": hand_name}
    (out_dir / BUNDLED_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    from pose_craft.estimation.landmarker import MODEL_URLS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "models")
    parser.add_argument("--variants", nargs="+", default=list(MODEL_URLS), choices=list(MODEL_URLS))
    parser.add_argument("--no-hands", action="store_true", help="leave out the hand landmark model")
    args = parser.parse_args()
    manifest = convert(args.out, args.variants, hands=not args.no_hands)
    for name, digest in manifest["sha256"].items():
        print(f"{digest}  {name}")
    print(f"wrote {len(manifest['variants'])} variant(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

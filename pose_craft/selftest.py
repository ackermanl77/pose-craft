"""``--self-test``: check a build really solves poses, without showing a window.

Run as ``PoseCraft.exe --self-test`` (or ``python -m pose_craft.app --self-test``)
before publishing a release. Checks:

- ONNX Runtime and its execution providers;
- the bundled pre-converted models (every variant plus the hand model) -- in a
  source install, the cached "full" model instead;
- a real solve of a bundled CC0 photo (tests/data/golden/yoga_tree.jpg) on the
  CPU, compared with its golden output (the tolerances of
  tests/test_real_photo_golden.py), then the same photo on the GPU compared with
  the CPU (a missing GPU is reported, not a failure), and the hand model;
- pose JSON (read back), BVH and glTF exports of that pose;
- the main window, built on the "offscreen" platform with throwaway settings.

Nothing is downloaded and the user's settings and logs are not touched. Exit
code 0 means every check passed. The windowed build has no console, so the
report is also written to ``%TEMP%/PoseCraft-selftest.txt`` (path printed at
the end).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pose_craft import __version__

SELF_TEST_FLAG = "--self-test"
REPORT_NAME = "PoseCraft-selftest.txt"
PHOTO = "yoga_tree.jpg"
MODEL_VARIANT = "full"  # the variant the golden output was written with
MAX_2D_PX = 2.0  # same as tests/test_real_photo_golden.py
MAX_WORLD_MM = 15.0
MAX_GPU_PX = 1.0  # GPU vs CPU (measured within 0.02 px on DirectML)


def wants_self_test(argv) -> bool:
    return SELF_TEST_FLAG in argv


def photo_dir() -> Path:
    """The self-test photo and its golden output: <bundle>/selftest in a frozen
    build (PoseCraft.spec), tests/data/golden in the source tree."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "selftest"
    return Path(__file__).resolve().parent.parent / "tests" / "data" / "golden"


def _check_models(log: Callable[[str], None]) -> bool:
    from pose_craft.estimation.landmarker import (
        MODEL_URLS,
        bundled_hand_onnx_path,
        bundled_model_dir,
        bundled_onnx_paths,
        is_model_available,
    )

    if bundled_model_dir() is None:
        if not is_model_available(MODEL_VARIANT):
            log(f"FAIL  no bundled models and the '{MODEL_VARIANT}' model isn't cached (run one solve in the app)")
            return False
        log(f"info  source install: using the cached '{MODEL_VARIANT}' model")
        return True
    missing = [variant for variant in MODEL_URLS if bundled_onnx_paths(variant) is None]
    if bundled_hand_onnx_path() is None:
        missing.append("hand")
    if missing:
        log(f"FAIL  bundled models missing: {', '.join(missing)} (in {bundled_model_dir()})")
        return False
    log(f"ok    bundled models: {', '.join(MODEL_URLS)} + hand")
    return True


def _visible_errors(person, want: dict, w: int, h: int) -> tuple[float, float]:
    visible = np.asarray(want["visibility"]) >= 0.5
    err_2d = np.linalg.norm(person.image_landmarks_xy * [w, h] - np.asarray(want["image_xy_px"]), axis=1)[visible]
    err_world = np.linalg.norm(person.landmarks - np.asarray(want["world_m"]), axis=1)[visible] * 1000
    return float(err_2d.max(initial=0)), float(err_world.max(initial=0))


def _check_solve(workdir: Path, log: Callable[[str], None]) -> bool:
    from pose_craft.estimation.landmarker import PoseEstimator, is_hand_model_available
    from pose_craft.export.blender_pose import load_pose_document, save_pose_file
    from pose_craft.export.bvh_export import save_bvh_file
    from pose_craft.export.gltf_export import save_gltf_pose
    from pose_craft.imaging import imread_unicode
    from pose_craft.retarget.skeleton import solve_bone_rotations

    photo = photo_dir() / PHOTO
    golden_path = photo.with_name(Path(PHOTO).stem + ".golden.json")
    image = imread_unicode(str(photo))
    if image is None or not golden_path.is_file():
        log(f"FAIL  self-test photo or its golden output not found in {photo.parent}")
        return False
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    h, w = image.shape[:2]

    ok = True
    with PoseEstimator.for_variant(MODEL_VARIANT, max_people=4, use_gpu=False) as cpu:
        people = cpu.detect(image)
        if len(people) != len(golden["people"]):
            log(f"FAIL  CPU solve: {len(people)} people, expected {len(golden['people'])}")
            return False
        err_2d, err_world = _visible_errors(people[0], golden["people"][0], w, h)
        if err_2d >= MAX_2D_PX or err_world >= MAX_WORLD_MM:
            log(f"FAIL  CPU solve differs from the golden output: {err_2d:.2f} px, {err_world:.1f} mm")
            ok = False
        else:
            log(f"ok    CPU solve matches the golden output (max {err_2d:.2f} px, {err_world:.1f} mm)")
        if is_hand_model_available():
            cpu.enable_hands()
            cpu.hands = True
            hands = sum(len(p.hands) for p in cpu.detect(image))
            log(f"ok    hand model loaded ({hands} hand(s) found in the photo)")
        else:
            log("info  hand model not cached (source install): skipped")
    reference = people[0]

    with PoseEstimator.for_variant(MODEL_VARIANT, max_people=4, use_gpu=True) as gpu:
        if gpu.device == "CPU":
            log(f"info  GPU not used: {gpu.gpu_fallback_reason or 'no GPU runtime in this build'}")
        else:
            gpu_people = gpu.detect(image)
            if not gpu_people:
                log(f"FAIL  {gpu.device} solve found nobody")
                ok = False
            else:
                diff = float(np.abs(gpu_people[0].image_landmarks_xy - reference.image_landmarks_xy).max() * max(w, h))
                if diff >= MAX_GPU_PX:
                    log(f"FAIL  {gpu.device} solve differs from the CPU by {diff:.3f} px")
                    ok = False
                else:
                    log(f"ok    {gpu.device} solve matches the CPU (max {diff:.3f} px)")

    rotations = solve_bone_rotations(reference.landmarks, visibility=reference.visibility)
    json_path, bvh_path, glb_path = workdir / "pose.json", workdir / "pose.bvh", workdir / "pose.glb"
    save_pose_file(rotations, json_path, source_note="self-test")
    save_bvh_file(rotations, bvh_path)
    save_gltf_pose(rotations, glb_path)
    document = load_pose_document(json_path)
    problems = [p.name for p in (json_path, bvh_path, glb_path) if not p.is_file() or p.stat().st_size == 0]
    if not bvh_path.read_text(encoding="utf-8").startswith("HIERARCHY"):
        problems.append("pose.bvh (no HIERARCHY)")
    if glb_path.read_bytes()[:4] != b"glTF":
        problems.append("pose.glb (not binary glTF)")
    if problems or not document:
        log(f"FAIL  exports: {', '.join(problems) or 'the pose JSON did not read back'}")
        ok = False
    else:
        log(f"ok    exports: pose JSON ({len(rotations)} bones, read back), BVH, glTF")
    return ok


def _check_window(workdir: Path, log: Callable[[str], None]) -> bool:
    from PySide6.QtCore import QSettings

    from pose_craft.gui.main_window import MainWindow

    settings = QSettings(str(workdir / "settings.ini"), QSettings.IniFormat)
    window = MainWindow(settings=settings, warm_up=False)
    try:
        menus = [action.text() for action in window.menuBar().actions()]
        if len(menus) < 3:
            log(f"FAIL  main window menus: {menus}")
            return False
        log(f"ok    main window: {len(menus)} menus")
        return True
    finally:
        window.close()
        window.deleteLater()


def run_self_test(out: Callable[[str], None] = print) -> int:
    lines: list[str] = []

    def log(line: str) -> None:
        lines.append(line)
        try:
            out(line)
        except Exception:  # noqa: BLE001 -- a windowed build may have no usable stdout
            pass

    ok = True
    log(f"PoseCraft {__version__} self-test")
    log(f"Python {sys.version.split()[0]} ({'frozen' if getattr(sys, 'frozen', False) else 'source'})")
    workdir = Path(tempfile.mkdtemp(prefix="posecraft_selftest_"))
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([sys.argv[0], "-platform", "offscreen"])
        import onnxruntime

        log(f"ok    ONNX Runtime {onnxruntime.__version__}: {', '.join(onnxruntime.get_available_providers())}")
        for check in (_check_models, _check_solve, _check_window):
            try:
                ok = (check(log) if check is _check_models else check(workdir, log)) and ok
            except Exception as exc:  # noqa: BLE001 -- report, don't crash
                ok = False
                log(f"FAIL  {type(exc).__name__}: {exc}")
                log(traceback.format_exc().strip())
            if check is _check_models and not ok:
                break  # nothing to solve with
        app.processEvents()
    except Exception as exc:  # noqa: BLE001
        ok = False
        log(f"FAIL  {type(exc).__name__}: {exc}")
        log(traceback.format_exc().strip())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log("RESULT: PASS" if ok else "RESULT: FAIL")
    report = os.path.join(tempfile.gettempdir(), REPORT_NAME)
    try:
        with open(report, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        try:
            out(f"(report written to {report})")
        except Exception:  # noqa: BLE001
            pass
    except OSError:
        pass
    return 0 if ok else 1

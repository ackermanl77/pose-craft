"""Real-photo end-to-end regression test: the whole pipeline (downscale,
BlazePose on the CPU, axis conversion, retargeting) against golden outputs
written by scripts/generate_goldens.py.

Skipped unless both exist: golden photos (tests/data/golden, or the folder
named by POSECRAFT_GOLDEN_DIR) and the cached "full" model
(~/.cache/pose_craft/models). The committed photos are public domain or CC0
(tests/data/golden/SOURCES.md); only photos that may be redistributed belong
there.

test_goldens_put_up_left_and_facing_where_the_photos_show_them needs no
model: it reads the committed goldens and checks the engine axes against what
the photos show -- the head above the hips above the ankles, and the person's
left on +X and nose toward +Z when they face the camera (the opposite when
they walk away) -- i.e. the "Y up, faces +Z, left +X" convention end to end.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_goldens.py"
_spec = importlib.util.spec_from_file_location("generate_goldens", _SCRIPT)
goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(goldens)

MAX_2D_PX = 2.0
MAX_WORLD_MM = 15.0
MAX_ROTATION_DEG = 5.0

_GOLDEN_FILES = sorted(goldens.golden_dir().glob("*.golden.json")) if goldens.golden_dir().exists() else []


@pytest.fixture(scope="module")
def estimator():
    pytest.importorskip("onnxruntime")
    result = goldens.cpu_estimator()
    if result is None:
        pytest.skip("the 'full' pose model isn't cached")
    return result


@pytest.mark.skipif(not _GOLDEN_FILES, reason="no golden photos (see scripts/generate_goldens.py)")
@pytest.mark.parametrize("golden_path", _GOLDEN_FILES, ids=lambda p: p.stem)
def test_real_photo_matches_its_golden_output(golden_path, estimator):
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    photo = golden_path.with_name(expected["photo"])
    actual = goldens.solve_photo(photo, estimator)

    assert len(actual["people"]) == len(expected["people"])
    for want, got in zip(expected["people"], actual["people"]):
        visible = np.asarray(want["visibility"]) >= 0.5
        err_2d = np.linalg.norm(np.subtract(got["image_xy_px"], want["image_xy_px"]), axis=1)[visible]
        err_world = np.linalg.norm(np.subtract(got["world_m"], want["world_m"]), axis=1)[visible] * 1000
        assert err_2d.max(initial=0) < MAX_2D_PX
        assert err_world.max(initial=0) < MAX_WORLD_MM
        for bone, rotvec in want["rotvec"].items():
            diff = Rotation.from_rotvec(got["rotvec"][bone]) * Rotation.from_rotvec(rotvec).inv()
            assert np.degrees(diff.magnitude()) < MAX_ROTATION_DEG, bone


# What each committed photo shows: which way its people face.
FACING = {
    "race_1948": "camera", "runner_1953": "camera", "yoga_tree": "camera",
    "walking_pair": "away", "arms_raised_from_behind": "away",
}
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 0, 11, 12, 23, 24, 27, 28


@pytest.mark.skipif(not _GOLDEN_FILES, reason="no golden photos (see scripts/generate_goldens.py)")
def test_goldens_put_up_left_and_facing_where_the_photos_show_them():
    committed = {path.name.removesuffix(".golden.json") for path in _GOLDEN_FILES}
    if not set(FACING) <= committed:
        pytest.skip("not the committed golden set (POSECRAFT_GOLDEN_DIR points elsewhere)")
    checked_full_body = 0
    for path in _GOLDEN_FILES:
        facing = FACING[path.name.removesuffix(".golden.json")]
        for person in json.loads(path.read_text(encoding="utf-8"))["people"]:
            world, visible = np.asarray(person["world_m"]), np.asarray(person["visibility"])
            left_minus_right = world[L_SHOULDER, 0] - world[R_SHOULDER, 0]
            assert (left_minus_right > 0.05) if facing == "camera" else (left_minus_right < -0.05), path.name
            if min(visible[L_ANKLE], visible[R_ANKLE]) < 0.5:
                continue  # legs out of the frame (arms_raised_from_behind): depth and height unreliable
            checked_full_body += 1
            hips = world[[L_HIP, R_HIP]].mean(axis=0)
            ankles = world[[L_ANKLE, R_ANKLE]].mean(axis=0)
            assert world[NOSE, 1] - hips[1] > 0.4 and hips[1] - ankles[1] > 0.3, path.name  # Y up
            assert (world[NOSE, 2] > hips[2]) if facing == "camera" else (world[NOSE, 2] < hips[2]), path.name
    assert checked_full_body >= 4  # 7 people; 3 have an ankle hidden or out of the frame

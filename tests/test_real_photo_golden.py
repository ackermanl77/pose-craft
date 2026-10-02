"""Real-photo end-to-end regression test: the whole pipeline (downscale,
BlazePose on the CPU, axis conversion, retargeting) against golden outputs
written by scripts/generate_goldens.py.

Skipped unless both exist: golden photos (tests/data/golden, or the folder
named by POSECRAFT_GOLDEN_DIR) and the cached "full" model
(~/.cache/pose_craft/models). No photos are committed yet -- only photos
that may be redistributed (CC0 or the owner's own) belong there.
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

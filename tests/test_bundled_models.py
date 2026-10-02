"""Pre-converted, bundled models (K3): a frozen build ships the ONNX models
made at build time, so it needs no download, no network on first run, and no
onnx/tflite converter at runtime."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pose_craft.estimation import landmarker

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "convert_models.py"
_spec = importlib.util.spec_from_file_location("convert_models", _SCRIPT)
convert_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_models)


@pytest.fixture
def fake_conversion(tmp_path, monkeypatch):
    """ensure_model_downloaded/ensure_onnx_models stand-ins: every variant has
    the same detector bytes and its own landmark model."""
    source = tmp_path / "cache"
    source.mkdir()

    def fake_download(variant, **_kwargs):
        return source / f"pose_landmarker_{variant}.task"

    def fake_onnx(task):
        variant = task.stem.split("_")[-1]
        paths = {"detector": source / f"{variant}.detector.onnx", "landmarks": source / f"{variant}.landmarks.onnx"}
        paths["detector"].write_bytes(b"same detector")
        paths["landmarks"].write_bytes(f"landmarks {variant}".encode())
        return paths

    monkeypatch.setattr(landmarker, "ensure_model_downloaded", fake_download)
    monkeypatch.setattr(landmarker, "ensure_onnx_models", fake_onnx)
    return tmp_path / "models"


def test_conversion_stores_the_shared_detector_once(fake_conversion):
    manifest = convert_models.convert(fake_conversion, ["lite", "full", "heavy"])

    assert {entry["detector"] for entry in manifest["variants"].values()} == {"pose_detector.onnx"}
    assert sorted(p.name for p in fake_conversion.glob("*.onnx")) == [
        "pose_detector.onnx", "pose_landmarks_full.onnx", "pose_landmarks_heavy.onnx", "pose_landmarks_lite.onnx",
    ]
    on_disk = json.loads((fake_conversion / landmarker.BUNDLED_MANIFEST).read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_bundled_models_are_used_without_any_download(fake_conversion, monkeypatch):
    convert_models.convert(fake_conversion, ["lite", "full"])
    monkeypatch.setenv("POSECRAFT_MODEL_DIR", str(fake_conversion))

    def no_network(*_args, **_kwargs):
        raise AssertionError("a bundled model must not be downloaded")

    monkeypatch.setattr(landmarker, "ensure_model_downloaded", no_network)
    paths = landmarker.resolve_onnx_models("full")
    assert paths == {
        "detector": fake_conversion / "pose_detector.onnx",
        "landmarks": fake_conversion / "pose_landmarks_full.onnx",
    }
    assert landmarker.is_model_available("full")
    assert landmarker.bundled_onnx_paths("heavy") is None  # not in this bundle: falls back to downloading


def test_a_broken_bundle_is_ignored(tmp_path, monkeypatch):
    (tmp_path / landmarker.BUNDLED_MANIFEST).write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("POSECRAFT_MODEL_DIR", str(tmp_path))
    assert landmarker.bundled_onnx_paths("full") is None
    (tmp_path / landmarker.BUNDLED_MANIFEST).write_text(
        json.dumps({"variants": {"full": {"detector": "missing.onnx", "landmarks": "missing.onnx"}}}), encoding="utf-8"
    )
    assert landmarker.bundled_onnx_paths("full") is None


def test_no_bundle_outside_a_frozen_build(monkeypatch):
    monkeypatch.delenv("POSECRAFT_MODEL_DIR", raising=False)
    assert landmarker.bundled_model_dir() is None


def test_an_estimator_runs_from_a_real_converted_bundle(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    if not landmarker.is_model_available("lite"):
        pytest.skip("the 'lite' pose model isn't cached")
    convert_models.convert(tmp_path, ["lite"])
    monkeypatch.setenv("POSECRAFT_MODEL_DIR", str(tmp_path))

    estimator = landmarker.PoseEstimator.for_variant("lite", use_gpu=False)
    estimator.warm_up()
    assert estimator.variant == "lite" and estimator.model_path is None
    assert estimator.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []

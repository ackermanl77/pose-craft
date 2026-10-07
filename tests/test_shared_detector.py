"""P5: the pose detector, byte-identical in the lite/full/heavy bundles, is cached
on disk once (by content) and its session is shared across variants."""
from __future__ import annotations

from pathlib import Path

import pytest

from pose_craft.estimation import landmarker, onnx_pose, tflite_to_onnx


@pytest.fixture
def fake_converter(monkeypatch):
    calls = []

    def convert(task_path, paths, *_members):
        calls.append(Path(task_path).name)
        paths["detector"].write_bytes(b"the one detector")
        paths["landmarks"].write_bytes(f"landmarks {Path(task_path).stem}".encode())
        return paths

    monkeypatch.setattr(tflite_to_onnx, "convert_task_bundle", convert)
    return calls


def test_three_variants_share_one_detector_file(tmp_path, fake_converter):
    results = {variant: landmarker.ensure_onnx_models(tmp_path / f"pose_landmarker_{variant}.task") for variant in ("lite", "full", "heavy")}
    detectors = {paths["detector"] for paths in results.values()}
    assert len(detectors) == 1 and next(iter(detectors)).name.startswith("pose_detector.")
    assert sorted(path.name for path in tmp_path.glob("*.onnx") if "detector" in path.name) == [next(iter(detectors)).name]
    assert len({paths["landmarks"] for paths in results.values()}) == 3


def test_a_second_call_reuses_the_cache_without_converting(tmp_path, fake_converter):
    task = tmp_path / "pose_landmarker_full.task"
    first = landmarker.ensure_onnx_models(task)
    assert landmarker.ensure_onnx_models(task) == first
    assert fake_converter == ["pose_landmarker_full.task"]


def test_a_per_variant_detector_from_an_older_version_is_folded_in(tmp_path, fake_converter):
    task = tmp_path / "pose_landmarker_lite.task"
    legacy = landmarker.onnx_model_paths(task)
    legacy["detector"].write_bytes(b"the one detector")
    legacy["landmarks"].write_bytes(b"landmarks")
    paths = landmarker.ensure_onnx_models(task)
    assert fake_converter == []  # nothing re-converted
    assert not legacy["detector"].exists() and paths["detector"].read_bytes() == b"the one detector"


def test_detector_sessions_are_created_once_per_file_and_device(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(onnx_pose, "create_session", lambda path, use_gpu=True: created.append((path, use_gpu)) or object())
    monkeypatch.setattr(onnx_pose, "_detector_sessions", {})
    detector = tmp_path / "pose_detector.abc.onnx"
    first = onnx_pose.shared_detector_session(detector, True)
    assert onnx_pose.shared_detector_session(detector, True) is first
    assert onnx_pose.shared_detector_session(detector, False) is not first
    assert len(created) == 2

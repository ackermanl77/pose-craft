from pathlib import Path

import numpy as np

from pose_craft.estimation import landmarker
from pose_craft.estimation.onnx_pose import PoseResult
from pose_craft.estimation.tflite_to_onnx import CONVERTER_VERSION


def test_world_to_engine_flips_y_and_z():
    """MediaPipe's world frame is +y down, smaller z = closer to the camera;
    the engine frame is +Y up, +Z toward the camera. Negating both keeps the
    frame right-handed (a rotation, not a reflection)."""
    world = np.array([[0.1, 0.2, 0.3], [-0.4, -0.5, 0.6]])
    np.testing.assert_allclose(landmarker._world_to_engine(world), [[0.1, -0.2, -0.3], [-0.4, 0.5, -0.6]])
    np.testing.assert_allclose(np.linalg.det(np.diag([1.0, -1.0, -1.0])), 1.0)


def test_onnx_model_paths_sit_next_to_the_task_bundle_and_carry_the_converter_version(tmp_path):
    paths = landmarker.onnx_model_paths(tmp_path / "pose_landmarker_full.task")
    assert paths == {
        "detector": tmp_path / f"pose_landmarker_full.v{CONVERTER_VERSION}.detector.onnx",
        "landmarks": tmp_path / f"pose_landmarker_full.v{CONVERTER_VERSION}.landmarks.onnx",
    }


def test_ensure_onnx_models_converts_once_then_reuses_the_cache(tmp_path, monkeypatch):
    calls = []

    def fake_convert(task_path: Path, dest_paths: dict[str, Path]) -> dict[str, Path]:
        calls.append(task_path)
        for path in dest_paths.values():
            path.write_bytes(b"onnx")
        return dest_paths

    monkeypatch.setattr("pose_craft.estimation.tflite_to_onnx.convert_task_bundle", fake_convert)
    task = tmp_path / "pose_landmarker_lite.task"

    first = landmarker.ensure_onnx_models(task)
    second = landmarker.ensure_onnx_models(task)

    assert calls == [task]
    assert first == second


class _FakeBlazePose:
    device = "GPU"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def detect(self, image_rgb, max_people, min_detection_confidence, min_presence_confidence):
        self.calls.append((image_rgb.copy(), max_people))
        return self.results


def _result(center_x: float) -> PoseResult:
    xy = np.full((33, 2), center_x)
    world = np.zeros((33, 3))
    world[:, 1] = 0.5  # MediaPipe world frame: +y is down
    return PoseResult(image_xy=xy, visibility=np.ones(33), world=world, presence=0.9)


def test_pose_estimator_converts_results_and_orders_people_left_to_right():
    estimator = landmarker.PoseEstimator.__new__(landmarker.PoseEstimator)
    estimator._pose = _FakeBlazePose([_result(0.8), _result(0.2)])
    estimator.max_people = 3
    image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    image_bgr[..., 0] = 255  # blue in BGR

    people = estimator.detect(image_bgr)

    assert [round(p.centroid_xy()[0], 2) for p in people] == [0.2, 0.8]
    np.testing.assert_allclose(people[0].landmarks[:, 1], -0.5)  # engine frame is Y-up
    sent_rgb, max_people = estimator._pose.calls[0]
    assert max_people == 3
    assert sent_rgb[0, 0].tolist() == [0, 0, 255]  # BGR -> RGB before inference
    assert estimator.device == "GPU"

"""GPU vs CPU parity (T5): the same model on a GPU provider must give the CPU's
landmarks. CUDA's default TF32 math once moved landmarks up to 12 px without
any test noticing.

Opt-in, because creating GPU sessions is slow (seconds for a cold CUDA start)
and hosted CI has no GPU: run before a release with

    set POSECRAFT_GPU_TESTS=1
    python -m pytest -q -m gpu

Needs the cached "full" model and a GPU provider (CUDA or DirectML); each
available one is compared against the CPU. Real photos in the golden folder
(see scripts/generate_goldens.py) are compared through the whole pipeline;
a synthetic image always runs through the raw models.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not os.environ.get("POSECRAFT_GPU_TESTS"), reason="set POSECRAFT_GPU_TESTS=1 to run"),
]

ort = pytest.importorskip("onnxruntime")

from pose_craft.estimation import landmarker, onnx_pose  # noqa: E402

MAX_LANDMARK_DIFF_PX = 0.5


def _gpu_preferences() -> list[str]:
    available = ort.get_available_providers()
    return [key for key, provider in onnx_pose._GPU_PROVIDERS.items() if provider in available]


# Raw landmark-model outputs (x/y/z in 256 px crop pixels, visibility and
# presence logits) on the synthetic card: CPU vs GPU differ by under 0.001 in
# fp32, but by 0.3-0.9 with CUDA's TF32 math (measured, RTX 5080).
MAX_RAW_OUTPUT_DIFF = 0.05


@pytest.fixture(scope="module")
def models():
    if not landmarker.is_model_available("full"):
        pytest.skip("the 'full' pose model isn't cached")
    return landmarker.resolve_onnx_models("full")


def _synthetic_image(size: int) -> np.ndarray:
    """A smooth, deterministic test card (gradients and discs): structure the
    networks respond to, unlike white noise."""
    ys, xs = np.mgrid[0:size, 0:size] / size
    image = np.stack([xs, ys, 0.5 + 0.5 * np.sin(6 * xs + 4 * ys)], axis=-1)
    for cx, cy, r in ((0.5, 0.3, 0.12), (0.4, 0.65, 0.2), (0.62, 0.62, 0.15)):
        image[(xs - cx) ** 2 + (ys - cy) ** 2 < r * r] *= 0.3
    return (image * 255).astype(np.uint8)


@pytest.mark.parametrize("variant", list(landmarker.MODEL_URLS))
@pytest.mark.parametrize("gpu", _gpu_preferences() or ["none"])
def test_gpu_landmark_model_matches_the_cpu(variant, gpu):
    if gpu == "none":
        pytest.skip("no GPU execution provider in this onnxruntime build")
    if not landmarker.is_model_available(variant):
        pytest.skip(f"the {variant!r} pose model isn't cached")
    path = landmarker.resolve_onnx_models(variant)["landmarks"]
    cpu = onnx_pose.create_session(path, use_gpu=False)
    gpu_session = onnx_pose.create_session(path, use_gpu=True, preference=(gpu,))
    assert gpu_session.get_providers()[0] == onnx_pose._GPU_PROVIDERS[gpu]

    tensor = (_synthetic_image(onnx_pose.LANDMARK_INPUT_SIZE).astype(np.float32) / 255.0)[None]
    name = cpu.get_inputs()[0].name
    reference = cpu.run(["Identity"], {name: tensor})[0]
    result = gpu_session.run(["Identity"], {name: tensor})[0]
    assert np.abs(result - reference).max() < MAX_RAW_OUTPUT_DIFF


@pytest.mark.parametrize("gpu", _gpu_preferences() or ["none"])
def test_gpu_pipeline_matches_the_cpu_on_real_photos(models, gpu):
    if gpu == "none":
        pytest.skip("no GPU execution provider in this onnxruntime build")
    import importlib.util
    from pathlib import Path

    from pose_craft.imaging import imread_unicode

    script = Path(__file__).resolve().parent.parent / "scripts" / "generate_goldens.py"
    spec = importlib.util.spec_from_file_location("generate_goldens", script)
    goldens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(goldens)
    folder = goldens.golden_dir()
    photos = sorted(p for p in folder.glob("*") if p.suffix.lower() in goldens.PHOTO_SUFFIXES) if folder.exists() else []
    if not photos:
        pytest.skip("no golden photos (see scripts/generate_goldens.py)")

    cpu = onnx_pose.BlazePose(models["detector"], models["landmarks"], use_gpu=False)
    gpu_pose = onnx_pose.BlazePose.__new__(onnx_pose.BlazePose)
    gpu_pose._detector = onnx_pose.create_session(models["detector"], True, (gpu,))
    gpu_pose._landmarks = onnx_pose.create_session(models["landmarks"], True, (gpu,))
    gpu_pose._detector_input = gpu_pose._detector.get_inputs()[0].name
    gpu_pose._landmarks_input = gpu_pose._landmarks.get_inputs()[0].name
    gpu_pose._landmarks_outputs = [o.name for o in gpu_pose._landmarks.get_outputs()]

    for photo in photos:
        image = landmarker.prepare_input_rgb(imread_unicode(str(photo)))
        h, w = image.shape[:2]
        expected = cpu.detect(image, max_people=4)
        actual = gpu_pose.detect(image, max_people=4)
        assert len(actual) == len(expected), photo.name
        for want, got in zip(expected, actual):
            diff = np.abs((got.image_xy - want.image_xy) * [w, h]).max()
            assert diff < MAX_LANDMARK_DIFF_PX, f"{photo.name}: {diff:.3f} px"

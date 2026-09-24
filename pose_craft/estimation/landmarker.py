"""Photo -> one 33-landmark 3D skeleton per detected person, converted into the
engine's Y-up coordinate frame (see retarget/skeleton.py's module docstring).

Uses MediaPipe's Pose Landmarker models (BlazePose), run on ONNX Runtime rather
than the mediapipe package: MediaPipe's Windows build is CPU-only, while ONNX
Runtime's CUDA provider runs the same models on an NVIDIA GPU. See
onnx_pose.py for the re-implemented pipeline and tflite_to_onnx.py for the
model conversion.

The pose-landmarker model isn't bundled with this repo (it's a multi-MB binary).
`ensure_model_downloaded` fetches MediaPipe's official `.task` bundle to a local
cache directory the first time the app runs, the same "downloads what it needs
on first run" behavior PoseForge describes for its own weights; it's converted
to ONNX next to it on first use (`ensure_onnx_models`).
"""
from __future__ import annotations

import dataclasses
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "pose_craft" / "models"

# Official MediaPipe-hosted model assets. "full" trades a bit of speed for accuracy;
# swap to the "lite" URL below on low-end GPUs/CPUs.
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# A real .task model is several MB; anything smaller almost certainly means a
# truncated download or an HTML error page saved under the expected filename.
MIN_MODEL_BYTES = 1_000_000


def model_cache_path(variant: str) -> Path:
    """Where `variant`'s model file lives (or would be downloaded to), without
    triggering a download -- lets a caller (e.g. the GUI, deciding whether a
    cached PoseEstimator needs rebuilding for a newly-selected variant) check
    which model a given variant name maps to without any network access.
    """
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}, expected one of {list(MODEL_URLS)}")
    return CACHE_DIR / f"pose_landmarker_{variant}.task"


def ensure_model_downloaded(variant: str = "full", on_progress: Callable[[int, int], None] | None = None) -> Path:
    """on_progress(downloaded_bytes, total_bytes) is called periodically during an
    actual download (never for an already-cached model) so a caller on a UI thread
    can show progress instead of an indefinite "please wait" -- total_bytes may be
    0 if the server didn't send a Content-Length header.
    """
    dest = model_cache_path(variant)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= MIN_MODEL_BYTES:
        return dest

    # Download to a temp file in the same directory (so the final os.replace is an
    # atomic same-filesystem rename) and only publish it at `dest` once the transfer
    # has actually finished -- an interrupted download must never leave a file behind
    # that a future `dest.exists()` check would mistake for a complete, valid model.
    fd, tmp_name = tempfile.mkstemp(dir=CACHE_DIR, prefix=f".{dest.name}.", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        import os

        os.close(fd)

        # urlretrieve's reporthook must be omitted entirely (not passed as None) when
        # there's no callback, so callers/tests that stub out urlretrieve with a plain
        # (url, filename) signature keep working unchanged.
        retrieve_kwargs = {}
        if on_progress is not None:
            def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                on_progress(min(downloaded, total_size) if total_size > 0 else downloaded, total_size)

            retrieve_kwargs["reporthook"] = _reporthook

        urllib.request.urlretrieve(MODEL_URLS[variant], tmp_path, **retrieve_kwargs)
        if tmp_path.stat().st_size < MIN_MODEL_BYTES:
            raise OSError(f"downloaded model for variant {variant!r} looks truncated ({tmp_path.stat().st_size} bytes)")
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest


@dataclasses.dataclass
class DetectedPerson:
    landmarks: np.ndarray  # (33, 3) in the engine's Y-up, +Z-toward-camera frame
    image_landmarks_xy: np.ndarray  # (33, 2) normalized [0, 1] image-space x/y, for click-to-select
    visibility: np.ndarray  # (33,) BlazePose's per-landmark visibility score, 0..1

    def centroid_xy(self) -> tuple[float, float]:
        return float(np.mean(self.image_landmarks_xy[:, 0])), float(np.mean(self.image_landmarks_xy[:, 1]))


def _world_to_engine(world_landmarks: np.ndarray) -> np.ndarray:
    """Converts BlazePose world landmarks ((33, 3) meters, origin at the hip
    midpoint, in MediaPipe's world frame) to the engine's right-handed Y-up,
    +Z-toward-camera frame (see retarget/skeleton.py's module docstring).

    MediaPipe's world frame is image-aligned: +x toward image right, +y DOWN
    (like image rows), and z is depth with smaller values closer to the camera
    (the same documented convention as its normalized image landmarks) -- a
    right-handed camera frame looking along +z. Negating both Y and Z is a
    180-degree rotation about X, so the result stays right-handed: +X image
    right, +Y up, +Z toward the camera. Flipping Y alone (this function's old
    mapping) made the frame left-handed, which mirrored every pose front-to-back
    -- a knee raised toward the camera exported *behind* the rig.
    """
    points = np.array(world_landmarks, dtype=np.float64)
    points[:, 1] = -points[:, 1]
    points[:, 2] = -points[:, 2]
    return points


def onnx_model_paths(model_path: Path) -> dict[str, Path]:
    """Where the ONNX conversions of a downloaded `.task` bundle are cached (next
    to it). The converter version is part of the name so a converter fix
    invalidates stale conversions instead of silently reusing them."""
    from pose_craft.estimation.tflite_to_onnx import CONVERTER_VERSION

    stem = f"{model_path.stem}.v{CONVERTER_VERSION}"
    return {key: model_path.with_name(f"{stem}.{key}.onnx") for key in ("detector", "landmarks")}


def ensure_onnx_models(model_path: Path) -> dict[str, Path]:
    """Converts `model_path`'s TFLite graphs to ONNX once, reusing the cached
    conversion afterwards (converting takes well under a second, but there's no
    reason to redo it on every launch)."""
    from pose_craft.estimation.tflite_to_onnx import convert_task_bundle

    paths = onnx_model_paths(model_path)
    if all(path.exists() for path in paths.values()):
        return paths
    return convert_task_bundle(model_path, paths)


class PoseEstimator:
    """Owns one BlazePose model pair on ONNX Runtime -- on the GPU (CUDA) when
    `use_gpu` is set and a CUDA-capable ONNX Runtime + NVIDIA driver are
    available, otherwise on the CPU -- so repeated solves don't re-pay model-load
    cost on every call. Create one per (model variant, device) and reuse it;
    `max_people` can be changed between calls. `device` reports "GPU" or "CPU".
    """

    def __init__(self, model_path: Path, max_people: int = 4, use_gpu: bool = True):
        from pose_craft.estimation.onnx_pose import BlazePose

        paths = ensure_onnx_models(model_path)
        self._pose = BlazePose(paths["detector"], paths["landmarks"], use_gpu=use_gpu)
        self.model_path = model_path
        self.max_people = max_people
        self.use_gpu = use_gpu

    @property
    def device(self) -> str:
        return self._pose.device

    def close(self) -> None:
        # ONNX Runtime sessions release their (GPU) memory when garbage-collected;
        # dropping the reference is all "closing" needs to do.
        self._pose = None

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedPerson]:
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        results = self._pose.detect(
            image_bgr[:, :, ::-1],
            max_people=self.max_people,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
        )
        people = [
            DetectedPerson(
                landmarks=_world_to_engine(result.world),
                image_landmarks_xy=result.image_xy,
                visibility=result.visibility,
            )
            for result in results
        ]
        people.sort(key=lambda p: p.centroid_xy()[0])
        return people

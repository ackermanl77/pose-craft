"""Wraps MediaPipe's Pose Landmarker task: photo -> one 33-landmark 3D skeleton per
detected person, converted into the engine's Y-up coordinate frame (see
retarget/skeleton.py's module docstring).

The pose-landmarker model isn't bundled with this repo (it's a multi-MB binary).
`ensure_model_downloaded` fetches it to a local cache directory the first time
the app runs, the same "downloads what it needs on first run" behavior PoseForge
describes for its own weights.
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


def ensure_model_downloaded(variant: str = "full", on_progress: Callable[[int, int], None] | None = None) -> Path:
    """on_progress(downloaded_bytes, total_bytes) is called periodically during an
    actual download (never for an already-cached model) so a caller on a UI thread
    can show progress instead of an indefinite "please wait" -- total_bytes may be
    0 if the server didn't send a Content-Length header.
    """
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}, expected one of {list(MODEL_URLS)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"pose_landmarker_{variant}.task"
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
    landmarks: np.ndarray  # (33, 3) in the engine's Y-up frame
    image_landmarks_xy: np.ndarray  # (33, 2) normalized [0, 1] image-space x/y, for click-to-select
    visibility: np.ndarray  # (33,) MediaPipe's per-landmark visibility score, 0..1

    def centroid_xy(self) -> tuple[float, float]:
        return float(np.mean(self.image_landmarks_xy[:, 0])), float(np.mean(self.image_landmarks_xy[:, 1]))


def _mediapipe_world_to_engine(world_landmarks) -> np.ndarray:
    """Converts MediaPipe's world-landmark list to the engine's Y-up, +Z-toward-camera frame.

    MediaPipe documents world landmarks as real-world-scale meters with the origin
    at the hip midpoint, but is not fully explicit about axis direction across
    versions. This mapping (flip Y, keep X/Z) matches the commonly observed output
    where MediaPipe's raw Y increases downward like image coordinates. If an
    exported pose comes out upside-down or mirrored, this is the one function to fix.
    """
    points = np.array([[lm.x, -lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)
    return points


class PoseEstimator:
    """Owns one live MediaPipe `PoseLandmarker` instance so repeated solves don't
    re-pay model-load cost on every call. Create one per (model variant, max
    people) combination you need and reuse it; call `close()` when done with it
    (the app closes its shared instance on exit).
    """

    def __init__(self, model_path: Path, max_people: int = 4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        self._mp = mp
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=max_people,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self.model_path = model_path
        self.max_people = max_people

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedPerson]:
        image_rgb = image_bgr[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect(mp_image)

        people: list[DetectedPerson] = []
        for image_landmarks, world_landmarks in zip(result.pose_landmarks, result.pose_world_landmarks):
            engine_landmarks = _mediapipe_world_to_engine(world_landmarks)
            image_xy = np.array([[lm.x, lm.y] for lm in image_landmarks], dtype=np.float64)
            visibility = np.array([lm.visibility for lm in image_landmarks], dtype=np.float64)
            people.append(DetectedPerson(landmarks=engine_landmarks, image_landmarks_xy=image_xy, visibility=visibility))

        people.sort(key=lambda p: p.centroid_xy()[0])
        return people

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
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "pose_craft" / "models"

# Official MediaPipe-hosted model assets. "full" trades a bit of speed for accuracy;
# swap to the "lite" URL below on low-end GPUs/CPUs.
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_model_downloaded(variant: str = "full") -> Path:
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}, expected one of {list(MODEL_URLS)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"pose_landmarker_{variant}.task"
    if not dest.exists():
        urllib.request.urlretrieve(MODEL_URLS[variant], dest)
    return dest


@dataclasses.dataclass
class DetectedPerson:
    landmarks: np.ndarray  # (33, 3) in the engine's Y-up frame
    bbox_center_x: float  # normalized 0..1 image-space x of the person's landmark centroid, for sorting/picking


def _mediapipe_world_to_engine(world_landmarks) -> np.ndarray:
    """Converts MediaPipe's world-landmark list to the engine's Y-up, +Z-toward-camera frame.

    MediaPipe documents world landmarks as real-world-scale meters with the origin
    at the hip midpoint, but is not fully explicit about axis direction across
    versions. This mapping (flip Y, keep X/Z) matches the commonly observed output
    where MediaPipe's raw Y increases downward like image coordinates. If an
    exported pose comes out upside-down or mirrored, this is the one place to fix.
    """
    points = np.array([[lm.x, -lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)
    return points


def detect_poses(image_bgr: np.ndarray, model_path: Path, max_people: int = 4) -> list[DetectedPerson]:
    """Runs the pose landmarker on a single BGR image (as loaded by OpenCV) and
    returns one DetectedPerson per detected body, in the engine coordinate frame.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_poses=max_people,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    image_rgb = image_bgr[:, :, ::-1]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    people: list[DetectedPerson] = []
    for image_landmarks, world_landmarks in zip(result.pose_landmarks, result.pose_world_landmarks):
        engine_landmarks = _mediapipe_world_to_engine(world_landmarks)
        centroid_x = float(np.mean([lm.x for lm in image_landmarks]))
        people.append(DetectedPerson(landmarks=engine_landmarks, bbox_center_x=centroid_x))

    people.sort(key=lambda p: p.bbox_center_x)
    return people

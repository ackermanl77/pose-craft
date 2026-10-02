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
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "pose_craft" / "models"

# Official MediaPipe-hosted model assets, pinned to version 1 (not ".../latest/"),
# so every install runs the exact model the ONNX converter and the pipeline were
# validated on. "full" trades a bit of speed for accuracy; "lite" suits low-end
# machines.
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# Accepted SHA-256 per variant, checked after every download (before the file
# is published to the cache) and on a cached file before it's used. "full" has
# two: the ".../latest/" file earlier versions of this app downloaded differs
# from ".../1/" only in the zip entries' timestamps (both .tflite files have the
# same size and CRC), so an existing cache stays valid.
MODEL_SHA256 = {
    "lite": ("59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",),
    "full": (
        "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    ),
    "heavy": ("64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",),
}

# A real .task model is several MB; anything smaller almost certainly means a
# truncated download or an HTML error page saved under the expected filename.
MIN_MODEL_BYTES = 1_000_000

# Per network operation (connect, each read), not for the whole download.
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_BYTES = 256 * 1024


def model_cache_path(variant: str) -> Path:
    """Where `variant`'s model file lives (or would be downloaded to), without
    triggering a download -- lets a caller (e.g. the GUI, deciding whether a
    cached PoseEstimator needs rebuilding for a newly-selected variant) check
    which model a given variant name maps to without any network access.
    """
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}, expected one of {list(MODEL_URLS)}")
    return CACHE_DIR / f"pose_landmarker_{variant}.task"


class DownloadCancelled(Exception):
    """Raised by ensure_model_downloaded when `should_cancel` returns True."""


class ModelChecksumError(OSError):
    """A downloaded model's SHA-256 isn't one of MODEL_SHA256's for its variant."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum_ok(variant: str, path: Path) -> bool:
    """Whether `path` hashes to an accepted value for `variant` (always true for
    a variant with no pinned hash)."""
    expected = MODEL_SHA256.get(variant)
    return not expected or file_sha256(path) in expected


def ensure_model_downloaded(
    variant: str = "full",
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """on_progress(downloaded_bytes, total_bytes) is called periodically during an
    actual download (never for an already-cached model) so a caller on a UI thread
    can show progress instead of an indefinite "please wait" -- total_bytes may be
    0 if the server didn't send a Content-Length header.

    should_cancel() is polled before every chunk; when it returns True the
    download stops with DownloadCancelled and leaves no partial file behind.

    The file must hash to one of MODEL_SHA256[variant] (ModelChecksumError
    otherwise, with nothing saved); a cached file that doesn't is downloaded
    again rather than trusted.
    """
    dest = model_cache_path(variant)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= MIN_MODEL_BYTES and _checksum_ok(variant, dest):
        return dest

    # Download to a temp file in the same directory (so the final os.replace is an
    # atomic same-filesystem rename) and only publish it at `dest` once the transfer
    # has actually finished -- an interrupted download must never leave a file behind
    # that a future `dest.exists()` check would mistake for a complete, valid model.
    fd, tmp_name = tempfile.mkstemp(dir=CACHE_DIR, prefix=f".{dest.name}.", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)

        # A streamed read with a socket timeout, not urlretrieve: urlretrieve has
        # no timeout, so a stalled connection used to hang the solve forever
        # (and closing the window with it). The timeout applies to each network
        # operation; `should_cancel` is checked before every chunk.
        with urllib.request.urlopen(MODEL_URLS[variant], timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(
            tmp_path, "wb"
        ) as out:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled(f"download of the {variant!r} model was cancelled")
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(min(downloaded, total) if total > 0 else downloaded, total)
        if tmp_path.stat().st_size < MIN_MODEL_BYTES:
            raise OSError(f"downloaded model for variant {variant!r} looks truncated ({tmp_path.stat().st_size} bytes)")
        if not _checksum_ok(variant, tmp_path):
            raise ModelChecksumError(
                f"the downloaded {variant!r} pose model doesn't match its pinned SHA-256 -- the download was "
                "corrupted or the file on the server changed; nothing was saved"
            )
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


# Pre-converted models (K3): a frozen build bundles the ONNX models made at
# build time by scripts/convert_models.py under <bundle>/models, with a
# manifest naming each variant's files (one detector shared by all three --
# it's byte-identical across them). With those, the app needs no download,
# no network on first run, and no onnx/tflite converter at runtime.
BUNDLED_MANIFEST = "manifest.json"


def bundled_model_dir() -> Path | None:
    """Where pre-converted models live: $POSECRAFT_MODEL_DIR if set, else
    <bundle>/models in a frozen build, else None (a dev install converts on
    first use)."""
    override = os.environ.get("POSECRAFT_MODEL_DIR")
    if override:
        return Path(override)
    bundle = getattr(sys, "_MEIPASS", None)
    return Path(bundle) / "models" if bundle else None


def bundled_onnx_paths(variant: str) -> dict[str, Path] | None:
    """`variant`'s pre-converted {"detector", "landmarks"} ONNX files from the
    bundled manifest, or None when there's no bundle or it lacks the variant."""
    directory = bundled_model_dir()
    if directory is None or not (directory / BUNDLED_MANIFEST).is_file():
        return None
    try:
        manifest = json.loads((directory / BUNDLED_MANIFEST).read_text(encoding="utf-8"))
        entry = manifest["variants"][variant]
        paths = {key: directory / entry[key] for key in ("detector", "landmarks")}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return paths if all(path.is_file() for path in paths.values()) else None


def is_model_available(variant: str) -> bool:
    """Whether `variant` can be loaded without any network access (bundled, or
    downloaded and verified) -- e.g. for a background warm-up at startup."""
    if bundled_onnx_paths(variant) is not None:
        return True
    path = model_cache_path(variant)
    return path.exists() and path.stat().st_size >= MIN_MODEL_BYTES and _checksum_ok(variant, path)


def resolve_onnx_models(
    variant: str,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """`variant`'s ONNX models: the bundled pre-converted ones when present,
    otherwise downloaded (ensure_model_downloaded) and converted once
    (ensure_onnx_models)."""
    bundled = bundled_onnx_paths(variant)
    if bundled is not None:
        return bundled
    return ensure_onnx_models(ensure_model_downloaded(variant, on_progress=on_progress, should_cancel=should_cancel))


# Photos with a longer side than this are shrunk (area-averaged) before
# inference. The models only ever see a 224 px detector input and 256 px
# person crops, which the pipeline warps with bilinear sampling -- straight
# from an 8000 px photo that reads 4 source pixels per output pixel out of
# ~36x36, so aliasing and sensor noise pass through almost untouched. Measured
# on 8x-upscaled noisy sample photos: about half the landmark error and ~6x
# faster. Normalized image coordinates don't depend on the size, so nothing
# downstream changes. 2048 keeps a person filling 1/8 of the frame at ~256 px.
MAX_INPUT_SIDE = 2048


def prepare_input_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """BGR photo -> contiguous RGB model input, downscaled with INTER_AREA when
    its longer side exceeds MAX_INPUT_SIDE (and converted after shrinking, so
    the full-size image is never copied)."""
    import cv2

    h, w = image_bgr.shape[:2]
    if max(h, w) > MAX_INPUT_SIDE:
        scale = MAX_INPUT_SIDE / max(h, w)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        image_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class PoseEstimator:
    """Owns one BlazePose model pair on ONNX Runtime -- on the GPU (DirectML or
    CUDA, whichever the installed onnxruntime build offers) when `use_gpu` is set
    and usable, otherwise on the CPU -- so repeated solves don't re-pay model-load
    cost on every call. Create one per (model variant, device) and reuse it;
    `max_people` can be changed between calls. `device` reports "GPU (DirectML)",
    "GPU (CUDA)" or "CPU".

    `group_photo` (also per call): search overlapping tiles too whenever the
    whole photo yields fewer than max_people, so small faces in a group shot
    are found (slower). Off, tiles are still tried when the whole photo yields
    nobody at all -- see onnx_pose.BlazePose.detect_rois.
    """

    group_photo: bool = False

    def __init__(
        self,
        model_path: Path | None = None,
        max_people: int = 4,
        use_gpu: bool = True,
        *,
        onnx_paths: dict[str, Path] | None = None,
        variant: str | None = None,
    ):
        """Either `model_path` (a downloaded .task bundle, converted on first
        use) or `onnx_paths` (already-converted {"detector", "landmarks"}
        files, see for_variant)."""
        from pose_craft.estimation.onnx_pose import BlazePose

        if onnx_paths is None:
            if model_path is None:
                raise ValueError("PoseEstimator needs a model_path or onnx_paths")
            onnx_paths = ensure_onnx_models(model_path)
        self._pose = BlazePose(onnx_paths["detector"], onnx_paths["landmarks"], use_gpu=use_gpu)
        self.model_path = model_path
        self.variant = variant
        self.max_people = max_people
        self.use_gpu = use_gpu

    @classmethod
    def for_variant(
        cls,
        variant: str,
        max_people: int = 4,
        use_gpu: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> "PoseEstimator":
        """The estimator for a model variant, from bundled pre-converted models
        when the build has them, else downloading/converting on first use."""
        paths = resolve_onnx_models(variant, on_progress=on_progress, should_cancel=should_cancel)
        bundled = bundled_onnx_paths(variant) is not None
        return cls(
            None if bundled else model_cache_path(variant),
            max_people=max_people,
            use_gpu=use_gpu,
            onnx_paths=paths,
            variant=variant,
        )

    def warm_up(self) -> None:
        """Runs one detector and one landmark inference on a blank image, so the
        first real solve doesn't pay the (GPU) session's first-run setup --
        measured at several seconds for a cold CUDA start."""
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        self._pose.warm_up()

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

    def track(self, image_rgb: np.ndarray, previous):
        """One video frame (already prepare_input_rgb'd) for one tracked person:
        see onnx_pose.BlazePose.track. Returns a PoseResult or None."""
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        return self._pose.track(image_rgb, previous, 0.5, 0.5)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedPerson]:
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        results = self._pose.detect(
            prepare_input_rgb(image_bgr),
            max_people=self.max_people,
            min_detection_confidence=0.5,
            min_presence_confidence=0.5,
            tiled=True if self.group_photo else None,
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

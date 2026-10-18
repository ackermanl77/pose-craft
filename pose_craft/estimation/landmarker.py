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
from collections.abc import Callable
from pathlib import Path

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

# The hand landmarker (B11, the "Hands and fingers" option), pinned and checked
# the same way. Only its 21-point landmark model is used (see onnx_pose's hand
# section); the bundle's palm detector is never converted.
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_SHA256 = ("fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",)
HAND_MODEL_FILENAME = "hand_landmarker.task"

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


def hand_model_cache_path() -> Path:
    """Where the hand landmarker bundle lives (or would be downloaded to)."""
    return CACHE_DIR / HAND_MODEL_FILENAME


def _cached_file_ok(path: Path, accepted: tuple[str, ...]) -> bool:
    return path.exists() and path.stat().st_size >= MIN_MODEL_BYTES and (not accepted or file_sha256(path) in accepted)


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
    accepted = tuple(MODEL_SHA256.get(variant) or ())
    return _download_verified(MODEL_URLS[variant], dest, accepted, f"{variant!r} pose", on_progress, should_cancel)


def ensure_hand_model_downloaded(
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """The hand landmarker bundle, downloaded once and SHA-256-checked exactly
    like the pose models (see ensure_model_downloaded)."""
    return _download_verified(
        HAND_MODEL_URL, hand_model_cache_path(), HAND_MODEL_SHA256, "hand", on_progress, should_cancel
    )


def _download_verified(
    url: str,
    dest: Path,
    accepted: tuple[str, ...],
    label: str,
    on_progress: Callable[[int, int], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> Path:
    """Downloads `url` to `dest` unless a cached copy hashing to one of
    `accepted` is there already (any hash, when `accepted` is empty)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _cached_file_ok(dest, accepted):
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
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(
            tmp_path, "wb"
        ) as out:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled(f"download of the {label} model was cancelled")
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(min(downloaded, total) if total > 0 else downloaded, total)
        if tmp_path.stat().st_size < MIN_MODEL_BYTES:
            raise OSError(f"downloaded {label} model looks truncated ({tmp_path.stat().st_size} bytes)")
        if accepted and file_sha256(tmp_path) not in accepted:
            raise ModelChecksumError(
                f"the downloaded {label} model doesn't match its pinned SHA-256 -- the download was "
                "corrupted or the file on the server changed; nothing was saved"
            )
        tmp_path.replace(dest)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest


@dataclasses.dataclass
class HandLandmarks:
    """One hand's 21 landmarks (B11)."""

    world: np.ndarray  # (21, 3) engine frame, meters, the wrist on the pose's wrist landmark
    image_xy: np.ndarray  # (21, 2) normalized image x/y
    presence: float  # the hand model's presence score, 0..1


@dataclasses.dataclass
class DetectedPerson:
    landmarks: np.ndarray  # (33, 3) in the engine's Y-up, +Z-toward-camera frame
    image_landmarks_xy: np.ndarray  # (33, 2) normalized [0, 1] image-space x/y, for click-to-select
    visibility: np.ndarray  # (33,) BlazePose's per-landmark visibility score, 0..1
    # "L"/"R" -> HandLandmarks, only with the "Hands and fingers" option (B11).
    hands: dict = dataclasses.field(default_factory=dict)
    # (33,) per-landmark presence, 0..1: inside the frame (B10). None when unknown.
    landmark_presence: np.ndarray | None = None

    def hand_world(self) -> dict[str, np.ndarray]:
        return {side: hand.world for side, hand in self.hands.items()}

    def hand_image_xy(self) -> dict[str, np.ndarray]:
        return {side: hand.image_xy for side, hand in self.hands.items()}

    def hand_presence(self) -> dict[str, float]:
        return {side: hand.presence for side, hand in self.hands.items()}

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


def hand_onnx_model_path(task_path: Path) -> Path:
    """Where the hand landmark model's ONNX conversion is cached (next to the bundle)."""
    from pose_craft.estimation.tflite_to_onnx import CONVERTER_VERSION

    return task_path.with_name(f"{task_path.stem}.v{CONVERTER_VERSION}.landmarks.onnx")


def ensure_hand_onnx_model(task_path: Path) -> Path:
    """Converts the hand bundle's landmark model to ONNX once (the palm
    detector is skipped), reusing the cached conversion afterwards."""
    from pose_craft.estimation.tflite_to_onnx import HAND_TASK_BUNDLE_MEMBERS, convert_task_bundle

    path = hand_onnx_model_path(task_path)
    if not path.exists():
        convert_task_bundle(task_path, {"landmarks": path}, HAND_TASK_BUNDLE_MEMBERS)
    return path


def shared_detector_path(directory: Path, detector_sha256: str) -> Path:
    """Where a converted pose detector is cached by content (P5): the lite, full
    and heavy bundles carry a byte-identical detector, so it's stored once."""
    from pose_craft.estimation.tflite_to_onnx import CONVERTER_VERSION

    return directory / f"pose_detector.{detector_sha256[:8]}.v{CONVERTER_VERSION}.onnx"


def _detector_pointer(paths: dict[str, Path]) -> Path:
    """A variant's small text file naming its shared detector (next to its landmarks)."""
    return paths["detector"].with_suffix(".txt")


def ensure_onnx_models(model_path: Path) -> dict[str, Path]:
    """Converts `model_path`'s TFLite graphs to ONNX once, reusing the cached
    conversion afterwards (converting takes well under a second, but there's no
    reason to redo it on every launch). The detector lands in a content-named
    shared file (shared_detector_path), so the three variants keep one copy; a
    per-variant detector left by an older version is folded into it."""
    from pose_craft.estimation.tflite_to_onnx import convert_task_bundle

    paths = onnx_model_paths(model_path)
    pointer = _detector_pointer(paths)
    if paths["landmarks"].exists() and pointer.exists():
        shared = paths["landmarks"].with_name(pointer.read_text(encoding="utf-8").strip())
        if shared.exists():
            return {"detector": shared, "landmarks": paths["landmarks"]}
    if not (paths["landmarks"].exists() and paths["detector"].exists()):
        convert_task_bundle(model_path, paths)
    shared = shared_detector_path(paths["detector"].parent, file_sha256(paths["detector"]))
    if shared.exists():
        paths["detector"].unlink()
    else:
        os.replace(paths["detector"], shared)
    pointer.write_text(shared.name, encoding="utf-8")
    return {"detector": shared, "landmarks": paths["landmarks"]}


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


def bundled_hand_onnx_path() -> Path | None:
    """The pre-converted hand landmark model from the bundled manifest (its
    "hand" entry), or None."""
    directory = bundled_model_dir()
    if directory is None or not (directory / BUNDLED_MANIFEST).is_file():
        return None
    try:
        manifest = json.loads((directory / BUNDLED_MANIFEST).read_text(encoding="utf-8"))
        path = directory / manifest["hand"]["landmarks"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return path if path.is_file() else None


def is_hand_model_available() -> bool:
    """Whether the hand model can be loaded without any network access."""
    return bundled_hand_onnx_path() is not None or _cached_file_ok(hand_model_cache_path(), HAND_MODEL_SHA256)


def resolve_hand_onnx_model(
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """The hand landmark model: bundled when present, else downloaded and
    converted once."""
    bundled = bundled_hand_onnx_path()
    if bundled is not None:
        return bundled
    return ensure_hand_onnx_model(ensure_hand_model_downloaded(on_progress=on_progress, should_cancel=should_cancel))


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


# Pose landmarks the hand crop is derived from, per side: wrist, index, pinky.
_HAND_POSE_LANDMARKS = {"L": (15, 19, 17), "R": (16, 20, 18)}
# A hand whose pose landmarks are less visible than this (out of frame,
# hidden) isn't searched: the crop would be a guess.
HAND_MIN_POSE_VISIBILITY = 0.5


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
    # B11, per call like group_photo: also run the hand landmark model on each
    # person's hands (needs enable_hands() first, which loads the model).
    hands: bool = False
    # B10, per call too: a person is reported when the detector scores them at
    # least min_detection_confidence and the landmark model's presence clears
    # min_presence_confidence. Lower them for hard photos (small, dark, unusual
    # poses); higher values drop doubtful detections.
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5

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
        self._hand_model = None

    @classmethod
    def for_variant(
        cls,
        variant: str,
        max_people: int = 4,
        use_gpu: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PoseEstimator:
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

    def enable_hands(
        self,
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Loads the hand landmark model (B11) on this estimator's device, once
        -- downloading and converting it first in a source install."""
        if self._hand_model is None:
            from pose_craft.estimation.onnx_pose import HandLandmarker

            path = resolve_hand_onnx_model(on_progress=on_progress, should_cancel=should_cancel)
            self._hand_model = HandLandmarker(path, use_gpu=self.use_gpu)

    @property
    def hands_ready(self) -> bool:
        """True once enable_hands() has loaded the hand model."""
        return self._hand_model is not None

    def find_hands(self, image_rgb: np.ndarray, person: DetectedPerson, result) -> None:
        """The hands of one already-solved person (a video frame's, B21); see _find_hands."""
        if self._hand_model is not None:
            self._find_hands(image_rgb, person, result)

    def _find_hands(self, image_rgb: np.ndarray, person: DetectedPerson, result) -> None:
        """Runs the hand model on both of `person`'s hands, from the pose's
        wrist/index/pinky landmarks (a hand whose landmarks are barely visible
        is skipped), storing what it finds in person.hands."""
        for side, (wrist, index, pinky) in _HAND_POSE_LANDMARKS.items():
            if min(result.visibility[i] for i in (wrist, index, pinky)) < HAND_MIN_POSE_VISIBILITY:
                continue
            hand = self._hand_model.detect(
                image_rgb, result.image_xy[wrist], result.image_xy[index], result.image_xy[pinky]
            )
            if hand is None:
                continue
            world = _world_to_engine(hand.world)
            world += person.landmarks[wrist] - world[0]  # put the hand's wrist on the pose's
            person.hands[side] = HandLandmarks(world=world, image_xy=hand.image_xy, presence=hand.presence)

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

    @property
    def gpu_fallback_reason(self) -> str | None:
        """Why the GPU requested with `use_gpu` isn't used: no hardware DirectX
        12 GPU, no GPU runtime in this build, or a GPU whose results didn't
        match the CPU's (onnx_pose.gpu_session_problem). None when it is used."""
        return self._pose.gpu_fallback_reason

    def close(self) -> None:
        # ONNX Runtime sessions release their (GPU) memory when garbage-collected;
        # dropping the reference is all "closing" needs to do.
        self._pose = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def new_tracker(self, max_people: int):
        """A video.MultiPersonTracker (B12) on this estimator's models: stable
        IDs for up to max_people people through a video."""
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        from pose_craft.estimation.video import MultiPersonTracker

        return MultiPersonTracker(self._pose, max_people)

    def track(self, image_rgb: np.ndarray, previous):
        """One video frame (already prepare_input_rgb'd) for one tracked person:
        see onnx_pose.BlazePose.track. Returns a PoseResult or None."""
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        return self._pose.track(image_rgb, previous, self.min_detection_confidence, self.min_presence_confidence)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedPerson]:
        if self._pose is None:
            raise RuntimeError("PoseEstimator is closed")
        image_rgb = prepare_input_rgb(image_bgr)
        results = self._pose.detect(
            image_rgb,
            max_people=self.max_people,
            min_detection_confidence=self.min_detection_confidence,
            min_presence_confidence=self.min_presence_confidence,
            tiled=True if self.group_photo else None,
        )
        find_hands = self.hands and self._hand_model is not None
        people = []
        for result in results:
            person = DetectedPerson(
                landmarks=_world_to_engine(result.world),
                image_landmarks_xy=result.image_xy,
                visibility=result.visibility,
                landmark_presence=result.landmark_presence,
            )
            if find_hands:
                self._find_hands(image_rgb, person, result)
            people.append(person)
        people.sort(key=lambda p: p.centroid_xy()[0])
        return people

"""BlazePose (MediaPipe's Pose Landmarker) re-implemented on ONNX Runtime.

MediaPipe's Windows wheels are built CPU-only, so the pose models themselves are
converted to ONNX (see tflite_to_onnx.py) and the calculator graph MediaPipe wraps
around them is re-implemented here in NumPy/OpenCV, following MediaPipe's own
sources calculator by calculator:

  pose_detector_graph.cc
    ImageToTensor (224x224 letterbox, zero border, [-1, 1]) -> detector ->
    SsdAnchors + TensorsToDetections -> weighted NonMaxSuppression ->
    DetectionLetterboxRemoval -> AlignmentPointsRects -> RectTransformation
  pose_landmarks_detector_graph.cc (once per detected person)
    ImageToTensor (256x256 rotated crop, replicate border, [0, 1]) -> landmarks
    model -> presence gate -> TensorsToLandmarks -> RefineLandmarksFromHeatmap ->
    LandmarkProjection / WorldLandmarkProjection

IMAGE running mode only (what this app uses): no tracking or smoothing between
calls, exactly as MediaPipe's own IMAGE mode behaves.

One addition MediaPipe doesn't have: group-photo tiling (`detect(..., tiled=)`).
The detector always sees the whole photo letterboxed into 224x224, and its
detections are driven by the face, so in a wide group shot each face can be a
few pixels across and nobody is found. Tiling runs the same detector on
overlapping tiles too (see tile_regions / merge_tile_detections); the landmark
model still runs on the full image.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import math
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DETECTOR_INPUT_SIZE = 224
LANDMARK_INPUT_SIZE = 256
NUM_LANDMARKS = 39  # 33 body landmarks + 6 auxiliary ones (ROI helpers), as the model outputs
NUM_BODY_LANDMARKS = 33
MIN_SUPPRESSION_THRESHOLD = 0.3  # PoseLandmarkerOptions' default
HEATMAP_KERNEL_SIZE = 7
HEATMAP_MIN_CONFIDENCE = 0.5  # RefineLandmarksFromHeatmapCalculatorOptions' default

CUDA_PROVIDER = "CUDAExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

# GPU execution providers in the order they're tried. DirectML first: on an
# RTX 5080 it ran the pipeline in ~7 ms against CUDA's 10-17 ms and matched the
# CPU to 0.03 px, and it also runs on AMD/Intel GPUs. The onnxruntime-directml
# and onnxruntime-gpu wheels install the same `onnxruntime` module, so one
# environment normally has only one of the two; the order matters only when a
# custom build offers both.
GPU_PREFERENCE = ("directml", "cuda")
_GPU_PROVIDERS = {"directml": DML_PROVIDER, "cuda": CUDA_PROVIDER}

# CUDA provider options. use_tf32=0: ONNX Runtime enables TF32 math by default
# on Ampere and newer GPUs, which moved landmarks up to 12 px (lite), 3 px
# (full) and 4 px (heavy) from the CPU result; with it off they match to
# ~0.04 px. The heavy model gets slower (measured ~16-23 ms -> ~38-40 ms per
# solve); DirectML avoids that trade-off.
CUDA_PROVIDER_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC", "use_tf32": "0"}


# -- detector -----------------------------------------------------------------


def ssd_anchor_centers() -> np.ndarray:
    """(2254, 2) anchor centers, normalized -- SsdAnchorsCalculator with the pose
    detector's options (5 layers, strides 8/16/32/32/32, one aspect ratio plus the
    interpolated-scale anchor, fixed anchor size). With fixed_anchor_size every
    anchor is 1x1, so only the centers matter."""
    strides = [8, 16, 32, 32, 32]
    centers = []
    layer = 0
    while layer < len(strides):
        # Consecutive layers sharing a stride share one feature map; each such layer
        # contributes 2 anchors per cell (aspect ratio 1.0 + interpolated scale).
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
        anchors_per_cell = 2 * (last - layer)
        size = math.ceil(DETECTOR_INPUT_SIZE / strides[layer])
        ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        cell = np.stack([(xs + 0.5) / size, (ys + 0.5) / size], axis=-1).reshape(-1, 2)
        centers.append(np.repeat(cell, anchors_per_cell, axis=0))
        layer = last
    return np.concatenate(centers).astype(np.float32)


_ANCHORS = ssd_anchor_centers()


@dataclasses.dataclass
class Detection:
    score: float
    box: np.ndarray  # (xmin, ymin, xmax, ymax), normalized to the letterboxed input
    keypoints: np.ndarray  # (4, 2) normalized x/y


def decode_detections(raw_boxes: np.ndarray, raw_scores: np.ndarray, min_score: float) -> list[Detection]:
    """TensorsToDetectionsCalculator: raw (2254, 12) regressors (XYWH order, since
    reverse_output_order is set) + (2254,) logits -> scored boxes/keypoints."""
    # float64: exp(100) overflows float32 (harmlessly, to a 0 score, but noisily).
    scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores.astype(np.float64), -100.0, 100.0)))
    keep = np.nonzero(scores >= min_score)[0]
    scale = float(DETECTOR_INPUT_SIZE)
    detections = []
    for i in keep:
        ax, ay = _ANCHORS[i]
        r = raw_boxes[i]
        cx, cy = r[0] / scale + ax, r[1] / scale + ay
        w, h = r[2] / scale, r[3] / scale
        keypoints = r[4:12].reshape(4, 2) / scale + _ANCHORS[i]
        detections.append(
            Detection(float(scores[i]), np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]), keypoints)
        )
    return detections


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def weighted_nms(detections: list[Detection], threshold: float = MIN_SUPPRESSION_THRESHOLD) -> list[Detection]:
    """NonMaxSuppressionCalculator's WEIGHTED algorithm: every box overlapping the
    current best one by more than `threshold` IoU is folded into it, with box and
    keypoints averaged by score (the best box keeps its own score)."""
    remaining = sorted(detections, key=lambda d: d.score, reverse=True)
    output = []
    while remaining:
        best = remaining[0]
        candidates, rest = [], []
        for det in remaining:
            (candidates if _iou(det.box, best.box) > threshold else rest).append(det)
        weights = np.array([d.score for d in candidates])
        total = weights.sum()
        box = np.sum([d.box * w for d, w in zip(candidates, weights)], axis=0) / total
        keypoints = np.sum([d.keypoints * w for d, w in zip(candidates, weights)], axis=0) / total
        output.append(Detection(best.score, box, keypoints))
        remaining = rest
    return output


def remove_letterbox(det: Detection, pad: tuple[float, float], width: int, height: int) -> Detection:
    """DetectionLetterboxRemoval, then scaled to pixels: a detection in the
    letterboxed detector input's normalized coordinates -> pixel coordinates of
    the image the detector saw (box and keypoints alike)."""
    pad_x, pad_y = pad
    scale = np.array([width / (1.0 - 2 * pad_x), height / (1.0 - 2 * pad_y)])
    offset = np.array([pad_x, pad_y])
    keypoints = (det.keypoints - offset) * scale
    corners = (det.box.reshape(2, 2) - offset) * scale
    return Detection(det.score, corners.reshape(4), keypoints)


# Group-photo tiling: a GRID x GRID layout of tiles, each overlapping its
# neighbours by TILE_OVERLAP of its own size, so a face on a tile border is
# whole in at least one tile. Measured on a 2x2 collage of a two-person photo
# (8 people): the whole-image pass found 1, the tiles 7.
TILE_GRID = 2
TILE_OVERLAP = 0.25


def tile_regions(width: int, height: int, grid: int = TILE_GRID, overlap: float = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) pixel rectangles covering the image in a grid x grid
    layout of equal, overlapping tiles with the image's aspect ratio."""
    regions = []
    tile_w = width / (1.0 + (grid - 1) * (1.0 - overlap))
    tile_h = height / (1.0 + (grid - 1) * (1.0 - overlap))
    for row in range(grid):
        for col in range(grid):
            x0 = col * tile_w * (1.0 - overlap)
            y0 = row * tile_h * (1.0 - overlap)
            regions.append((round(x0), round(y0), min(width, round(x0 + tile_w)), min(height, round(y0 + tile_h))))
    return regions


def merge_tile_detections(
    primary: list[Detection], tiled: list[Detection], threshold: float = MIN_SUPPRESSION_THRESHOLD
) -> list[Detection]:
    """Adds the people the tiles found to the whole-image detections (all in the
    same pixel coordinates). The tiles' own duplicates (one face seen by two
    overlapping tiles) are merged with weighted NMS; a tile detection that
    overlaps a whole-image one is dropped rather than averaged into it, since a
    tile can cut a large person off and the whole-image detection is the better
    estimate of their body. Sorted by score, best first."""
    kept = [
        det for det in weighted_nms(tiled, threshold)
        if all(_iou(det.box, other.box) <= threshold for other in primary)
    ]
    return sorted(primary + kept, key=lambda d: d.score, reverse=True)


@dataclasses.dataclass
class NormRect:
    """MediaPipe's NormalizedRect: center/size normalized to the image, rotation
    in radians (clockwise in image coordinates)."""

    x_center: float
    y_center: float
    width: float
    height: float
    rotation: float


def _normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def detection_to_roi(keypoints: np.ndarray, image_w: int, image_h: int) -> NormRect:
    """AlignmentPointsRects (keypoint 0 = hip center, keypoint 1 = a point whose
    distance from it sets the body's scale; rotated so that vector points up) then
    RectTransformation (scale 1.25, square on the long side, in pixels)."""
    x0, y0 = keypoints[0, 0] * image_w, keypoints[0, 1] * image_h
    x1, y1 = keypoints[1, 0] * image_w, keypoints[1, 1] * image_h
    # The alignment rect is already square in pixels, so square_long is a no-op.
    box_size = 2.0 * math.hypot(x1 - x0, y1 - y0)
    rotation = _normalize_radians(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    return NormRect(
        x_center=x0 / image_w,
        y_center=y0 / image_h,
        width=box_size / image_w * 1.25,
        height=box_size / image_h * 1.25,
        rotation=rotation,
    )


# -- image -> tensor ----------------------------------------------------------


def _warp_rotated_rect(image_rgb: np.ndarray, center, size, rotation: float, out_size: int, border: int) -> np.ndarray:
    """ImageToTensorConverter (OpenCV backend): maps the rotated rect's corners onto
    the output square with a perspective warp, bilinear filtering."""
    box = cv2.boxPoints(((float(center[0]), float(center[1])), (float(size[0]), float(size[1])), math.degrees(rotation)))
    dst = np.array([[0, out_size], [0, 0], [out_size, 0], [out_size, out_size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
    return cv2.warpPerspective(image_rgb, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def detector_input(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Whole image, letterboxed into 224x224 with zero padding, scaled to [-1, 1].
    Returns the tensor and the (horizontal, vertical) padding fraction per side."""
    h, w = image_rgb.shape[:2]
    roi_aspect = h / w
    if 1.0 > roi_aspect:
        size = (w, w)
        pad = (0.0, (1.0 - roi_aspect) / 2.0)
    else:
        size = (h, h)
        pad = ((1.0 - 1.0 / roi_aspect) / 2.0, 0.0)
    crop = _warp_rotated_rect(image_rgb, (w / 2, h / 2), size, 0.0, DETECTOR_INPUT_SIZE, cv2.BORDER_CONSTANT)
    tensor = crop.astype(np.float32) * (2.0 / 255.0) - 1.0
    return tensor[None], pad


def landmark_input(image_rgb: np.ndarray, roi: NormRect, size: int = LANDMARK_INPUT_SIZE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    crop = _warp_rotated_rect(
        image_rgb,
        (roi.x_center * w, roi.y_center * h),
        (roi.width * w, roi.height * h),
        roi.rotation,
        size,
        cv2.BORDER_REPLICATE,
    )
    return (crop.astype(np.float32) / 255.0)[None]


# -- landmarks ----------------------------------------------------------------


def refine_from_heatmap(xy: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """RefineLandmarksFromHeatmapCalculator: replaces each landmark's crop-space x/y
    with the confidence-weighted centroid of a 7x7 heatmap window around it, when the
    window's peak confidence is high enough."""
    hm_h, hm_w, _ = heatmap.shape
    offset = (HEATMAP_KERNEL_SIZE - 1) // 2
    refined = xy.copy()
    for i in range(xy.shape[0]):
        # int() truncates toward zero, matching the C++ float->int conversion.
        col, row = int(xy[i, 0] * hm_w), int(xy[i, 1] * hm_h)
        if not (0 <= col < hm_w and 0 <= row < hm_h):
            continue
        r0, r1 = max(0, row - offset), min(hm_h, row + offset + 1)
        c0, c1 = max(0, col - offset), min(hm_w, col + offset + 1)
        confidence = 1.0 / (1.0 + np.exp(-heatmap[r0:r1, c0:c1, i]))
        total = confidence.sum()
        if confidence.max() >= HEATMAP_MIN_CONFIDENCE and total > 0:
            rows, cols = np.mgrid[r0:r1, c0:c1]
            refined[i, 0] = (cols * confidence).sum() / hm_w / total
            refined[i, 1] = (rows * confidence).sum() / hm_h / total
    return refined


def project_to_image(xy: np.ndarray, roi: NormRect) -> np.ndarray:
    """LandmarkProjectionCalculator (NORM_RECT form): crop-normalized -> image-normalized."""
    x, y = xy[:, 0] - 0.5, xy[:, 1] - 0.5
    cos, sin = math.cos(roi.rotation), math.sin(roi.rotation)
    return np.stack(
        [(cos * x - sin * y) * roi.width + roi.x_center, (sin * x + cos * y) * roi.height + roi.y_center], axis=1
    )


def rotate_world(world: np.ndarray, rotation: float) -> np.ndarray:
    """WorldLandmarkProjectionCalculator: undo the crop's rotation in x/y."""
    cos, sin = math.cos(rotation), math.sin(rotation)
    out = world.copy()
    out[:, 0] = cos * world[:, 0] - sin * world[:, 1]
    out[:, 1] = sin * world[:, 0] + cos * world[:, 1]
    return out


@dataclasses.dataclass
class PoseResult:
    image_xy: np.ndarray  # (33, 2) normalized image coordinates
    visibility: np.ndarray  # (33,)
    world: np.ndarray  # (33, 3) meters, MediaPipe's world frame (origin at hip center, +y down)
    presence: float
    # The model's two ROI helper landmarks (hip center, body-scale point), image
    # coordinates: where to look for this person in the next video frame.
    aux_xy: np.ndarray | None = None

    def next_roi(self, image_w: int, image_h: int) -> NormRect | None:
        """MediaPipe VIDEO mode's PoseLandmarksToRoi: the next frame's crop for
        this person, from the auxiliary landmarks -- the same alignment-point
        rule the detector's keypoints go through (detection_to_roi)."""
        if self.aux_xy is None:
            return None
        return detection_to_roi(self.aux_xy[:2], image_w, image_h)


# -- sessions -----------------------------------------------------------------


_dlls_preloaded = False


def _preload_cuda_dlls() -> None:
    """Loads CUDA/cuDNN from the pip-installed nvidia-* packages so the CUDA
    execution provider can find them without a system-wide CUDA toolkit install.
    The PyInstaller build keeps those packages' site-packages layout under
    _internal/ (see PoseCraft.spec), so the same lookup works frozen. Harmless
    without them -- the CUDA provider then fails to initialize and ONNX Runtime
    runs on the CPU."""
    global _dlls_preloaded
    if _dlls_preloaded:
        return
    _dlls_preloaded = True
    import onnxruntime as ort

    if CUDA_PROVIDER not in ort.get_available_providers() or not hasattr(ort, "preload_dlls"):
        return
    try:
        # directory="" -> only the nvidia site-packages layout, then default DLL search
        # (never a co-installed PyTorch's copies, which may be a different CUDA major).
        ort.preload_dlls(directory="")
    except Exception:  # noqa: BLE001 -- missing CUDA libs just means "run on CPU"
        log.info("could not preload CUDA DLLs; ONNX Runtime will fall back to the CPU", exc_info=True)


@dataclasses.dataclass(frozen=True)
class GpuAdapter:
    """A DXGI adapter DirectML can run on. `index` is its IDXGIFactory1::EnumAdapters1
    position, which is what the DirectML provider's `device_id` option selects."""

    index: int
    name: str
    vendor_id: int
    dedicated_memory_mb: int


_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_MICROSOFT_VENDOR_ID = 0x1414  # Microsoft Basic Render Driver (WARP) and other software adapters
_D3D_FEATURE_LEVEL_11_0 = 0xB000  # the minimum DirectML needs
_IID_IDXGI_FACTORY1 = "770aae78-f26f-4dba-a829-253c83d1b387"
_IID_ID3D12_DEVICE = "189819f1-1db6-4b57-be54-1821339b85f7"


def directml_adapters() -> list[GpuAdapter]:
    """The hardware GPUs DirectML can use: DXGI adapters that aren't software
    renderers and can create a Direct3D 12 device (feature level 11_0), in DXGI
    order. Microsoft Basic Render Driver -- what a PC without a working GPU
    driver has -- is left out: DirectML on it runs on the CPU in software,
    slower than the CPU provider (and with onnxruntime-directml 1.24.4 it fails
    to initialize at all). Empty off Windows or if DXGI/D3D12 can't be loaded."""
    if sys.platform != "win32":
        return []
    try:
        return _enumerate_dxgi_adapters()
    except Exception:  # noqa: BLE001 -- no usable GPU means "run on the CPU"
        log.info("could not enumerate DXGI adapters; DirectML won't be used", exc_info=True)
        return []


def _enumerate_dxgi_adapters() -> list[GpuAdapter]:
    import ctypes
    import uuid
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

    def guid(text: str) -> _GUID:
        u = uuid.UUID(text)
        g = _GUID(u.fields[0], u.fields[1], u.fields[2])
        g.d4[:] = u.bytes[8:]
        return g

    class _LUID(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.LONG)]

    class _DESC1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
        _fields_ = [
            ("description", ctypes.c_wchar * 128), ("vendor_id", ctypes.c_uint), ("device_id", ctypes.c_uint),
            ("subsys_id", ctypes.c_uint), ("revision", ctypes.c_uint), ("dedicated_video_memory", ctypes.c_size_t),
            ("dedicated_system_memory", ctypes.c_size_t), ("shared_system_memory", ctypes.c_size_t),
            ("luid", _LUID), ("flags", ctypes.c_uint),
        ]

    def method(obj, index, *argtypes):
        """COM vtable slot `index` of `obj` (IUnknown: 0-2, IDXGIObject: 3-6,
        IDXGIFactory/IDXGIAdapter: 7-..., EnumAdapters1 = 12, GetDesc1 = 10)."""
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])

    dxgi = ctypes.WinDLL("dxgi")
    d3d12 = ctypes.WinDLL("d3d12")
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(guid(_IID_IDXGI_FACTORY1)), ctypes.byref(factory)) != 0:
        return []
    adapters = []
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if method(factory, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(factory, index, ctypes.byref(adapter)) != 0:
                break  # DXGI_ERROR_NOT_FOUND: no more adapters
            try:
                desc = _DESC1()
                method(adapter, 10, ctypes.POINTER(_DESC1))(adapter, ctypes.byref(desc))
                software = desc.flags & _DXGI_ADAPTER_FLAG_SOFTWARE or desc.vendor_id == _MICROSOFT_VENDOR_ID
                # ppDevice=NULL only tests support: S_FALSE (1) when a device could be created.
                d3d12_ok = d3d12.D3D12CreateDevice(adapter, _D3D_FEATURE_LEVEL_11_0, ctypes.byref(guid(_IID_ID3D12_DEVICE)), None) in (0, 1)
                if not software and d3d12_ok:
                    adapters.append(GpuAdapter(index, desc.description, desc.vendor_id, desc.dedicated_video_memory >> 20))
            finally:
                method(adapter, 2)(adapter)  # Release
            index += 1
    finally:
        method(factory, 2)(factory)
    return adapters


@functools.lru_cache(maxsize=1)
def best_directml_adapter() -> GpuAdapter | None:
    """The hardware GPU with the most dedicated memory (the discrete card on a
    laptop with integrated graphics too), or None when there is none."""
    adapters = directml_adapters()
    if not adapters:
        return None
    return max(adapters, key=lambda a: (a.dedicated_memory_mb, -a.index))


def select_providers(
    available: list[str],
    use_gpu: bool,
    preference: tuple[str, ...] = GPU_PREFERENCE,
    dml_device_id: int | None = None,
) -> tuple[list, bool]:
    """The provider list for an InferenceSession: the available GPU providers in
    `preference` order (each with its options), then the CPU as the fallback
    ONNX Runtime uses if a GPU provider fails to initialize. DirectML is only
    included with a `dml_device_id` -- the hardware GPU to run on (see
    best_directml_adapter); without one there's no GPU for it. Returns
    (providers, whether DirectML is among them -- it needs its own session options)."""
    providers: list = []
    uses_dml = False
    if use_gpu:
        for key in preference:
            name = _GPU_PROVIDERS[key]
            if name not in available:
                continue
            if name == DML_PROVIDER:
                if dml_device_id is None:
                    continue
                providers.append((name, {"device_id": dml_device_id}))
                uses_dml = True
            else:
                providers.append((name, dict(CUDA_PROVIDER_OPTIONS)))
    return providers + [CPU_PROVIDER], uses_dml


def create_session(model_path: Path, use_gpu: bool = True, preference: tuple[str, ...] = GPU_PREFERENCE):
    """An ONNX Runtime session on the first usable GPU provider (see
    select_providers) when requested, otherwise the CPU provider. Check
    `session.get_providers()[0]` for which won."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only -- keeps provider fallbacks from spamming stderr
    available = ort.get_available_providers()
    dml_device_id = None
    if use_gpu:
        _preload_cuda_dlls()
        if DML_PROVIDER in available and "directml" in preference:
            adapter = best_directml_adapter()
            dml_device_id = adapter.index if adapter else None
    providers, uses_dml = select_providers(available, use_gpu, preference, dml_device_id)
    if uses_dml:
        # Required by the DirectML provider (ONNX Runtime's DirectML docs):
        # no memory-pattern optimization and no parallel execution.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


# A GPU session must reproduce the CPU's outputs on a probe input to within this
# fraction of each output's range. Measured on an RTX 5080 and an AMD Radeon iGPU
# (DirectML): at most ~7e-4 (the heatmap), and landmarks within 0.004 px -- a
# broken driver/GPU gives garbage, NaNs or zeros, far outside it.
GPU_CHECK_RELATIVE_TOLERANCE = 1e-2

_gpu_check_cache: dict[tuple[str, str], str | None] = {}


def _probe_input(session) -> dict:
    """A smooth, deterministic image (gradients and a bright ellipse) for the
    model's NHWC float input -- well-conditioned, unlike noise."""
    shape = session.get_inputs()[0].shape
    height, width = (int(d) if isinstance(d, int) and d > 0 else LANDMARK_INPUT_SIZE for d in shape[1:3])
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    y /= height
    x /= width
    image = np.stack([x, y, 0.5 * (x + y)], axis=-1)
    image[((x - 0.5) / 0.2) ** 2 + ((y - 0.45) / 0.35) ** 2 < 1] = (0.9, 0.7, 0.6)
    return {session.get_inputs()[0].name: image[None]}


def gpu_session_problem(session, model_path: Path) -> str | None:
    """Why `session` (on a GPU provider) can't be trusted, or None when it
    reproduces the CPU provider's outputs on a probe input. A CPU session needs
    no check. Cached per (provider, model) for the process."""
    provider = session.get_providers()[0]
    if provider == CPU_PROVIDER:
        return None
    key = (provider, str(model_path))
    if key not in _gpu_check_cache:
        _gpu_check_cache[key] = _compare_with_cpu(session, model_path, provider)
    return _gpu_check_cache[key]


def _compare_with_cpu(session, model_path: Path, provider: str) -> str | None:
    label = {DML_PROVIDER: "DirectML", CUDA_PROVIDER: "CUDA"}.get(provider, provider)
    model = Path(model_path).stem
    try:
        inputs = _probe_input(session)
        gpu_outputs = session.run(None, inputs)
        cpu_outputs = create_session(model_path, use_gpu=False).run(None, inputs)
    except Exception as exc:  # noqa: BLE001 -- any failure means "don't use this GPU"
        return f"{label} failed to run {model}: {exc}"
    for gpu, cpu in zip(gpu_outputs, cpu_outputs):
        gpu = np.asarray(gpu, dtype=np.float64)
        cpu = np.asarray(cpu, dtype=np.float64)
        if gpu.shape != cpu.shape or not np.all(np.isfinite(gpu)):
            return f"{label} gave invalid results for {model}"
        scale = max(float(np.max(np.abs(cpu))), 1e-6) if cpu.size else 1.0
        error = float(np.max(np.abs(gpu - cpu))) / scale if cpu.size else 0.0
        if error > GPU_CHECK_RELATIVE_TOLERANCE:
            return f"{label} results for {model} differ from the CPU's ({error:.1%} of the output range)"
    return None


def no_gpu_reason() -> str:
    """Why a requested GPU isn't available, when ONNX Runtime ended up on the CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if DML_PROVIDER in available and best_directml_adapter() is None:
        return "no DirectX 12 GPU found (only a software renderer)"
    if DML_PROVIDER not in available and CUDA_PROVIDER not in available:
        return "this build has no GPU runtime"
    return "the GPU runtime could not start"


# Detector sessions shared by every BlazePose (P5): the lite/full/heavy variants use
# the same detector file, so switching the model quality rebuilds only the landmark
# session. Keyed by (resolved path, GPU requested); each has a lock because a
# DirectML session must not run two inferences at once.
_detector_sessions: dict[tuple[str, bool], tuple[object, threading.Lock]] = {}
_detector_sessions_lock = threading.Lock()


def shared_detector_session(detector_path: Path, use_gpu: bool) -> tuple[object, threading.Lock]:
    """The (session, run lock) for `detector_path`, created on first use."""
    key = (str(Path(detector_path).resolve()), bool(use_gpu))
    with _detector_sessions_lock:
        if key not in _detector_sessions:
            _detector_sessions[key] = (create_session(detector_path, use_gpu), threading.Lock())
        return _detector_sessions[key]


class BlazePose:
    """Two ONNX Runtime sessions (detector + landmark model) plus MediaPipe's
    pre/post-processing. Construct once and reuse -- session creation (and, on the
    GPU, CUDA initialization) is the expensive part."""

    # Replaced by the shared detector's own lock in __init__ (a default for
    # instances assembled without it, e.g. around a stand-in session).
    _detector_lock: threading.Lock = threading.Lock()

    def __init__(self, detector_path: Path, landmarks_path: Path, use_gpu: bool = True):
        self._detector, self._detector_lock = shared_detector_session(detector_path, use_gpu)
        self._landmarks = create_session(landmarks_path, use_gpu)
        #: Why a requested GPU isn't used (None when it is, or wasn't requested).
        self.gpu_fallback_reason: str | None = None
        if use_gpu:
            reason = gpu_session_problem(self._detector, detector_path) or gpu_session_problem(
                self._landmarks, landmarks_path
            )
            if reason:
                log.warning("not using the GPU: %s", reason)
                self.gpu_fallback_reason = reason
                self._detector, self._detector_lock = shared_detector_session(detector_path, use_gpu=False)
                self._landmarks = create_session(landmarks_path, use_gpu=False)
            elif self._detector.get_providers()[0] == CPU_PROVIDER:
                self.gpu_fallback_reason = no_gpu_reason()
        self._detector_input = self._detector.get_inputs()[0].name
        self._landmarks_input = self._landmarks.get_inputs()[0].name
        self._landmarks_outputs = [o.name for o in self._landmarks.get_outputs()]

    @property
    def device(self) -> str:
        """"GPU (DirectML)" or "GPU (CUDA)" when both models run on that provider,
        else "CPU"."""
        first = {s.get_providers()[0] for s in (self._detector, self._landmarks)}
        if first == {DML_PROVIDER}:
            return "GPU (DirectML)"
        if first == {CUDA_PROVIDER}:
            return "GPU (CUDA)"
        return "CPU"

    def warm_up(self) -> None:
        """One inference through each model on a blank image (nothing is found):
        a GPU provider does most of its setup on the first run."""
        blank = np.zeros((LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3), dtype=np.uint8)
        self._run_detector(blank, 0.5)
        self.landmarks_for_roi(blank, NormRect(0.5, 0.5, 1.0, 1.0, 0.0), min_presence=2.0)  # presence <= 1: no result

    def _run_detector(self, image_rgb: np.ndarray, min_detection_confidence: float) -> list[Detection]:
        """The detector on one (whole or tile) image, weighted-NMS'd, in that
        image's pixel coordinates."""
        h, w = image_rgb.shape[:2]
        tensor, pad = detector_input(image_rgb)
        with self._detector_lock:
            raw_boxes, raw_scores = self._detector.run(None, {self._detector_input: tensor})
        detections = weighted_nms(decode_detections(raw_boxes[0], raw_scores[0, :, 0], min_detection_confidence))
        return [remove_letterbox(det, pad, w, h) for det in detections]

    def detect_rois(
        self, image_rgb: np.ndarray, max_people: int, min_detection_confidence: float, tiled: bool | None = None
    ) -> list[NormRect]:
        """tiled: True also searches overlapping tiles (group photos) whenever the
        whole image yields fewer than max_people; None (the default) does that
        only when the whole image yields nobody at all; False never does."""
        h, w = image_rgb.shape[:2]
        detections = self._run_detector(image_rgb, min_detection_confidence)
        want_tiles = tiled if tiled is not None else not detections
        if want_tiles and len(detections) < max_people:
            from_tiles = []
            for x0, y0, x1, y1 in tile_regions(w, h):
                tile = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
                offset = np.array([x0, y0], dtype=np.float64)
                for det in self._run_detector(tile, min_detection_confidence):
                    from_tiles.append(
                        Detection(det.score, det.box + np.tile(offset, 2), det.keypoints + offset)
                    )
            detections = merge_tile_detections(detections, from_tiles)
        size = np.array([w, h], dtype=np.float64)
        return [detection_to_roi(det.keypoints / size, w, h) for det in detections[:max_people]]

    def landmarks_for_roi(self, image_rgb: np.ndarray, roi: NormRect, min_presence: float) -> PoseResult | None:
        outputs = dict(zip(self._landmarks_outputs, self._landmarks.run(None, {self._landmarks_input: landmark_input(image_rgb, roi)})))
        # Output names are the TFLite graph's own: Identity = landmarks, Identity_1 =
        # pose presence (already a sigmoid), Identity_3 = heatmap, Identity_4 = world.
        presence = float(outputs["Identity_1"].reshape(-1)[0])
        if presence < min_presence:
            return None
        raw = outputs["Identity"].reshape(NUM_LANDMARKS, 5)
        xy = raw[:, :2] / LANDMARK_INPUT_SIZE
        xy = refine_from_heatmap(xy, outputs["Identity_3"][0])
        visibility = 1.0 / (1.0 + np.exp(-raw[:, 3]))
        world = rotate_world(outputs["Identity_4"].reshape(NUM_LANDMARKS, 3), roi.rotation)
        body = slice(0, NUM_BODY_LANDMARKS)
        aux = slice(NUM_BODY_LANDMARKS, NUM_BODY_LANDMARKS + 2)
        return PoseResult(
            image_xy=project_to_image(xy[body], roi).astype(np.float64),
            visibility=visibility[body].astype(np.float64),
            world=world[body].astype(np.float64),
            presence=presence,
            aux_xy=project_to_image(xy[aux], roi).astype(np.float64),
        )

    def track(
        self,
        image_rgb: np.ndarray,
        previous: PoseResult | None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> PoseResult | None:
        """One video frame for one person, MediaPipe VIDEO-mode style: the
        landmark model runs on the crop the previous frame's landmarks predict,
        and the detector runs only when there's no previous frame or the person
        was lost there (presence below the threshold). Returns None when nobody
        is found."""
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        roi = previous.next_roi(w, h) if previous is not None else None
        if roi is not None:
            result = self.landmarks_for_roi(image_rgb, roi, min_presence_confidence)
            if result is not None:
                return result
        found = self.detect(image_rgb, 1, min_detection_confidence, min_presence_confidence)
        return found[0] if found else None

    def detect(
        self,
        image_rgb: np.ndarray,
        max_people: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        tiled: bool | None = None,
    ) -> list[PoseResult]:
        """tiled: see detect_rois. With None (auto), a photo whose whole-image
        detections all fail the landmark model's presence check is also retried
        with tiles -- the whole image can "find" a person-like blob in a group
        shot while every real face in it is too small."""
        image_rgb = np.ascontiguousarray(image_rgb)

        def solve(tiles: bool | None) -> list[PoseResult]:
            results = []
            for roi in self.detect_rois(image_rgb, max_people, min_detection_confidence, tiles):
                result = self.landmarks_for_roi(image_rgb, roi, min_presence_confidence)
                if result is not None:
                    results.append(result)
            return results

        results = solve(tiled)
        if not results and tiled is None:
            results = solve(True)
        return results


# -- hands (B11) ----------------------------------------------------------------
#
# MediaPipe Holistic's way of finding a hand: no palm detector, the crop comes
# from the pose's wrist, index and pinky landmarks (HandDetectionsFromPoseToRects
# + RectTransformation, scale 2.7, shift -0.1), then the 21-point hand landmark
# model runs on it. One refinement pass re-crops from the hand's own landmarks
# (HandLandmarksToRect + RectTransformation, scale 2.0, shift -0.1, MediaPipe's
# hand-tracking crop) and runs the model again; Holistic uses a separate
# re-crop model for that step, which this doesn't need.

HAND_LANDMARK_INPUT_SIZE = 224
NUM_HAND_LANDMARKS = 21
HAND_ROI_FROM_POSE_SCALE = 2.7
HAND_ROI_FROM_LANDMARKS_SCALE = 2.0
HAND_ROI_SHIFT_Y = -0.1
MIN_HAND_PRESENCE = 0.5
# The hand landmarks the re-crop box is fitted to (HandLandmarksToRect's
# partial set: the wrist and the two lower joints of each finger).
_HAND_RECROP_LANDMARKS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


def transform_rect(
    rect: NormRect, image_w: int, image_h: int, scale: float, shift_y: float = 0.0, square_long: bool = True
) -> NormRect:
    """RectTransformationCalculator for a NormalizedRect: shift along the rect's
    own (rotated) y axis by shift_y of its height, make it square on its long
    side in pixels, then scale it."""
    width, height, rotation = rect.width, rect.height, rect.rotation
    x_shift = -(image_h * height * shift_y * math.sin(rotation)) / image_w
    y_shift = (image_h * height * shift_y * math.cos(rotation)) / image_h
    if square_long:
        long_side = max(width * image_w, height * image_h)
        width, height = long_side / image_w, long_side / image_h
    return NormRect(rect.x_center + x_shift, rect.y_center + y_shift, width * scale, height * scale, rotation)


def hand_roi_from_pose(wrist_xy, index_xy, pinky_xy, image_w: int, image_h: int) -> NormRect | None:
    """The hand crop from one side's pose landmarks (normalized x/y): centred on
    an estimated middle knuckle ((2 * index + pinky) / 3), twice its distance
    from the wrist across, rotated so wrist -> knuckle points up, then
    transformed as MediaPipe Holistic does. None when the points coincide."""
    size = np.array([image_w, image_h], dtype=np.float64)
    wrist = np.asarray(wrist_xy, dtype=np.float64) * size
    middle = (2.0 * np.asarray(index_xy, dtype=np.float64) + np.asarray(pinky_xy, dtype=np.float64)) / 3.0 * size
    box_size = 2.0 * float(np.hypot(*(middle - wrist)))
    if box_size < 1e-6:
        return None
    rotation = _normalize_radians(math.pi / 2 - math.atan2(-(middle[1] - wrist[1]), middle[0] - wrist[0]))
    rect = NormRect(middle[0] / image_w, middle[1] / image_h, box_size / image_w, box_size / image_h, rotation)
    return transform_rect(rect, image_w, image_h, HAND_ROI_FROM_POSE_SCALE, HAND_ROI_SHIFT_Y)


def hand_roi_from_landmarks(hand_xy: np.ndarray, image_w: int, image_h: int) -> NormRect:
    """HandLandmarksToRectCalculator + its RectTransformation: the next crop for
    a hand from its own (21, 2) normalized landmarks."""
    xy = np.asarray(hand_xy, dtype=np.float64)
    x0, y0 = xy[0, 0] * image_w, xy[0, 1] * image_h
    x1 = ((xy[5, 0] + xy[13, 0]) / 2.0 + xy[9, 0]) / 2.0 * image_w
    y1 = ((xy[5, 1] + xy[13, 1]) / 2.0 + xy[9, 1]) / 2.0 * image_h
    rotation = _normalize_radians(math.pi / 2 - math.atan2(-(y1 - y0), x1 - x0))
    points = xy[_HAND_RECROP_LANDMARKS]
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    offsets = (points - center) * np.array([image_w, image_h])
    reverse = _normalize_radians(-rotation)
    cos_r, sin_r = math.cos(reverse), math.sin(reverse)
    projected = np.stack(
        [offsets[:, 0] * cos_r - offsets[:, 1] * sin_r, offsets[:, 0] * sin_r + offsets[:, 1] * cos_r], axis=1
    )
    low, high = projected.min(axis=0), projected.max(axis=0)
    mid = (low + high) / 2.0
    cos_f, sin_f = math.cos(rotation), math.sin(rotation)
    center_x = mid[0] * cos_f - mid[1] * sin_f + image_w * center[0]
    center_y = mid[0] * sin_f + mid[1] * cos_f + image_h * center[1]
    rect = NormRect(
        center_x / image_w, center_y / image_h, (high[0] - low[0]) / image_w, (high[1] - low[1]) / image_h, rotation
    )
    return transform_rect(rect, image_w, image_h, HAND_ROI_FROM_LANDMARKS_SCALE, HAND_ROI_SHIFT_Y)


@dataclasses.dataclass
class HandResult:
    image_xy: np.ndarray  # (21, 2) normalized image coordinates
    world: np.ndarray  # (21, 3) meters, MediaPipe's world frame (around the hand's centre, +y down)
    presence: float


class HandLandmarker:
    """MediaPipe's 21-point hand landmark model on ONNX Runtime, run on crops
    derived from the pose (see hand_roi_from_pose). Same device rules as
    BlazePose: a GPU session is kept only if it matches the CPU."""

    def __init__(self, model_path: Path, use_gpu: bool = True):
        self._session = create_session(model_path, use_gpu)
        reason = gpu_session_problem(self._session, model_path) if use_gpu else None
        if reason:
            log.warning("not using the GPU for the hand model: %s", reason)
            self._session = create_session(model_path, use_gpu=False)
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]

    def landmarks_for_roi(self, image_rgb: np.ndarray, roi: NormRect, min_presence: float = MIN_HAND_PRESENCE) -> HandResult | None:
        tensor = landmark_input(image_rgb, roi, HAND_LANDMARK_INPUT_SIZE)
        outputs = dict(zip(self._outputs, self._session.run(None, {self._input: tensor})))
        # Identity = landmarks (x, y, z in input pixels), Identity_1 = hand
        # presence (already a sigmoid), Identity_2 = handedness (unused: the
        # side comes from the pose), Identity_3 = world landmarks (meters).
        presence = float(outputs["Identity_1"].reshape(-1)[0])
        if presence < min_presence:
            return None
        raw = outputs["Identity"].reshape(NUM_HAND_LANDMARKS, 3)
        xy = raw[:, :2] / HAND_LANDMARK_INPUT_SIZE
        world = rotate_world(outputs["Identity_3"].reshape(NUM_HAND_LANDMARKS, 3), roi.rotation)
        return HandResult(
            image_xy=project_to_image(xy, roi).astype(np.float64), world=world.astype(np.float64), presence=presence
        )

    def detect(self, image_rgb: np.ndarray, wrist_xy, index_xy, pinky_xy, refine: bool = True) -> HandResult | None:
        """One hand from its pose landmarks (normalized x/y), or None when the
        model doesn't see a hand there. With `refine`, a second pass on the
        crop the first pass's own landmarks give (kept if it still sees the hand)."""
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        roi = hand_roi_from_pose(wrist_xy, index_xy, pinky_xy, w, h)
        if roi is None:
            return None
        result = self.landmarks_for_roi(image_rgb, roi)
        if result is not None and refine:
            refined = self.landmarks_for_roi(image_rgb, hand_roi_from_landmarks(result.image_xy, w, h))
            if refined is not None:
                result = refined
        return result

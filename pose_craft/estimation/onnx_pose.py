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
"""
from __future__ import annotations

import dataclasses
import logging
import math
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
CPU_PROVIDER = "CPUExecutionProvider"


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


def landmark_input(image_rgb: np.ndarray, roi: NormRect) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    crop = _warp_rotated_rect(
        image_rgb,
        (roi.x_center * w, roi.y_center * h),
        (roi.width * w, roi.height * h),
        roi.rotation,
        LANDMARK_INPUT_SIZE,
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


def create_session(model_path: Path, use_gpu: bool = True):
    """An ONNX Runtime session on the CUDA provider when requested and usable,
    otherwise the CPU provider. Check `session.get_providers()[0]` for which won."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # errors only -- keeps provider fallbacks from spamming stderr
    providers = [CPU_PROVIDER]
    if use_gpu:
        _preload_cuda_dlls()
        if CUDA_PROVIDER in ort.get_available_providers():
            providers = [(CUDA_PROVIDER, {"cudnn_conv_algo_search": "HEURISTIC"}), CPU_PROVIDER]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


class BlazePose:
    """Two ONNX Runtime sessions (detector + landmark model) plus MediaPipe's
    pre/post-processing. Construct once and reuse -- session creation (and, on the
    GPU, CUDA initialization) is the expensive part."""

    def __init__(self, detector_path: Path, landmarks_path: Path, use_gpu: bool = True):
        self._detector = create_session(detector_path, use_gpu)
        self._landmarks = create_session(landmarks_path, use_gpu)
        self._detector_input = self._detector.get_inputs()[0].name
        self._landmarks_input = self._landmarks.get_inputs()[0].name
        self._landmarks_outputs = [o.name for o in self._landmarks.get_outputs()]

    @property
    def device(self) -> str:
        """"GPU" when both models run on the CUDA provider, else "CPU"."""
        on_gpu = all(s.get_providers()[0] == CUDA_PROVIDER for s in (self._detector, self._landmarks))
        return "GPU" if on_gpu else "CPU"

    def detect_rois(self, image_rgb: np.ndarray, max_people: int, min_detection_confidence: float) -> list[NormRect]:
        h, w = image_rgb.shape[:2]
        tensor, (pad_x, pad_y) = detector_input(image_rgb)
        raw_boxes, raw_scores = self._detector.run(None, {self._detector_input: tensor})
        detections = weighted_nms(decode_detections(raw_boxes[0], raw_scores[0, :, 0], min_detection_confidence))
        rois = []
        for det in detections[:max_people]:
            # DetectionLetterboxRemoval: letterboxed-input coordinates -> image coordinates.
            keypoints = det.keypoints.copy()
            keypoints[:, 0] = (keypoints[:, 0] - pad_x) / (1.0 - 2 * pad_x)
            keypoints[:, 1] = (keypoints[:, 1] - pad_y) / (1.0 - 2 * pad_y)
            rois.append(detection_to_roi(keypoints, w, h))
        return rois

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
        return PoseResult(
            image_xy=project_to_image(xy[body], roi).astype(np.float64),
            visibility=visibility[body].astype(np.float64),
            world=world[body].astype(np.float64),
            presence=presence,
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        max_people: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> list[PoseResult]:
        image_rgb = np.ascontiguousarray(image_rgb)
        results = []
        for roi in self.detect_rois(image_rgb, max_people, min_detection_confidence):
            result = self.landmarks_for_roi(image_rgb, roi, min_presence_confidence)
            if result is not None:
                results.append(result)
        return results

import math

import numpy as np
import pytest

from pose_craft.estimation import onnx_pose
from pose_craft.estimation.onnx_pose import (
    BlazePose,
    Detection,
    NormRect,
    decode_detections,
    detection_to_roi,
    detector_input,
    merge_tile_detections,
    project_to_image,
    refine_from_heatmap,
    rotate_world,
    ssd_anchor_centers,
    tile_regions,
    weighted_nms,
)


def test_anchor_layout_matches_the_pose_detector_output_count():
    anchors = ssd_anchor_centers()
    # 28x28x2 (stride 8) + 14x14x2 (stride 16) + 7x7x6 (three stride-32 layers).
    assert anchors.shape == (2254, 2)
    np.testing.assert_allclose(anchors[0], [0.5 / 28, 0.5 / 28])
    np.testing.assert_allclose(anchors[1], anchors[0])  # two anchors per cell
    np.testing.assert_allclose(anchors[1568], [0.5 / 14, 0.5 / 14])
    np.testing.assert_allclose(anchors[-1], [6.5 / 7, 6.5 / 7])


def test_decode_detections_applies_anchor_offsets_and_score_threshold():
    raw_boxes = np.zeros((2254, 12), dtype=np.float32)
    raw_scores = np.full(2254, -10.0, dtype=np.float32)
    i = 100
    anchor = ssd_anchor_centers()[i]
    raw_boxes[i, :4] = [22.4, -11.2, 44.8, 89.6]  # dx, dy, w, h in 224-pixel units
    raw_boxes[i, 4:6] = [0.0, 22.4]  # keypoint 0: 0.1 below the anchor
    raw_scores[i] = 5.0

    detections = decode_detections(raw_boxes, raw_scores, min_score=0.5)

    assert len(detections) == 1
    det = detections[0]
    assert det.score == pytest.approx(1 / (1 + math.exp(-5.0)))
    cx, cy = anchor[0] + 0.1, anchor[1] - 0.05
    np.testing.assert_allclose(det.box, [cx - 0.1, cy - 0.2, cx + 0.1, cy + 0.2], atol=1e-6)
    np.testing.assert_allclose(det.keypoints[0], [anchor[0], anchor[1] + 0.1], atol=1e-6)


def test_weighted_nms_merges_overlaps_by_score_and_keeps_separate_people():
    kp = np.zeros((4, 2))
    a = Detection(0.9, np.array([0.1, 0.1, 0.3, 0.5]), kp + 0.2)
    b = Detection(0.6, np.array([0.12, 0.1, 0.32, 0.5]), kp + 0.3)  # overlaps a
    c = Detection(0.8, np.array([0.6, 0.1, 0.8, 0.5]), kp + 0.7)  # separate person

    merged = weighted_nms([b, c, a])

    assert [round(d.score, 2) for d in merged] == [0.9, 0.8]
    np.testing.assert_allclose(merged[0].box, (a.box * 0.9 + b.box * 0.6) / 1.5)
    np.testing.assert_allclose(merged[0].keypoints, (0.2 * 0.9 + 0.3 * 0.6) / 1.5)
    np.testing.assert_allclose(merged[1].box, c.box)


def test_detection_to_roi_for_an_upright_person_is_unrotated_and_scaled():
    # Hip center at (0.5, 0.6), scale keypoint straight above it -> body axis points up.
    keypoints = np.array([[0.5, 0.6], [0.5, 0.4], [0, 0], [0, 0]])
    roi = detection_to_roi(keypoints, 1000, 1000)
    assert roi.rotation == pytest.approx(0.0, abs=1e-9)
    assert (roi.x_center, roi.y_center) == pytest.approx((0.5, 0.6))
    assert roi.width == pytest.approx(2 * 0.2 * 1.25)
    assert roi.height == pytest.approx(2 * 0.2 * 1.25)


def test_detection_to_roi_stays_square_in_pixels_on_a_wide_photo():
    keypoints = np.array([[0.5, 0.6], [0.5, 0.4], [0, 0], [0, 0]])
    roi = detection_to_roi(keypoints, 2000, 1000)
    assert roi.width * 2000 == pytest.approx(roi.height * 1000)


def test_detection_to_roi_for_a_person_lying_sideways_rotates_90_degrees():
    # Body axis pointing to the right in the image.
    keypoints = np.array([[0.4, 0.5], [0.6, 0.5], [0, 0], [0, 0]])
    roi = detection_to_roi(keypoints, 1000, 1000)
    assert roi.rotation == pytest.approx(math.pi / 2)


def test_project_to_image_undoes_the_crop_rotation():
    roi = NormRect(0.5, 0.5, 0.5, 0.5, math.pi / 2)
    # The crop's "up" (y=0 edge centre) maps to the image's right side for a 90° ROI.
    projected = project_to_image(np.array([[0.5, 0.0], [0.5, 0.5]]), roi)
    np.testing.assert_allclose(projected, [[0.75, 0.5], [0.5, 0.5]], atol=1e-9)


def test_rotate_world_rotates_x_y_and_leaves_z():
    world = np.array([[1.0, 0.0, 0.3]])
    np.testing.assert_allclose(rotate_world(world, math.pi / 2), [[0.0, 1.0, 0.3]], atol=1e-9)


def test_refine_from_heatmap_moves_a_landmark_to_a_confident_peak():
    heatmap = np.full((64, 64, 2), -20.0, dtype=np.float32)
    heatmap[34, 30, 0] = 20.0  # confident peak two cells off landmark 0's own cell
    xy = np.array([[32 / 64, 32 / 64], [0.5, 0.5]])

    refined = refine_from_heatmap(xy, heatmap)

    assert refined[0, 0] == pytest.approx(30 / 64, abs=1e-3)
    assert refined[0, 1] == pytest.approx(34 / 64, abs=1e-3)
    np.testing.assert_array_equal(refined[1], xy[1])  # no confident peak -> unchanged


def test_refine_from_heatmap_skips_landmarks_outside_the_crop():
    heatmap = np.full((64, 64, 1), 20.0, dtype=np.float32)
    xy = np.array([[1.5, 0.5]])
    np.testing.assert_array_equal(refine_from_heatmap(xy, heatmap), xy)


def test_detector_input_letterboxes_a_wide_photo_with_zero_padding():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    tensor, (pad_x, pad_y) = detector_input(image)
    assert tensor.shape == (1, 224, 224, 3)
    assert (pad_x, pad_y) == pytest.approx((0.0, 0.25))
    assert tensor[0, 5, 112, 0] == pytest.approx(-1.0)  # top padding band (zero pixel)
    assert tensor[0, 112, 112, 0] == pytest.approx(1.0)  # photo content (white)


class _FakeSession:
    def __init__(self, outputs: dict[str, np.ndarray]):
        self._outputs = outputs

    def run(self, names, feed):
        return [self._outputs[name] for name in (names or self._outputs)]


def _fake_landmark_outputs(presence: float) -> dict[str, np.ndarray]:
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[:, 0] = 128.0  # crop center
    raw[:, 1] = 64.0  # a quarter of the way down the crop
    raw[:, 3] = 10.0  # confidently visible
    world = np.zeros((39, 3), dtype=np.float32)
    world[:, 0] = 0.25
    return {
        "Identity": raw.reshape(1, 195),
        "Identity_1": np.array([[presence]], dtype=np.float32),
        "Identity_2": np.zeros((1, 256, 256, 1), dtype=np.float32),
        "Identity_3": np.full((1, 64, 64, 39), -20.0, dtype=np.float32),
        "Identity_4": world.reshape(1, 117),
    }


def _pose_with_fake_landmark_model(presence: float) -> BlazePose:
    pose = BlazePose.__new__(BlazePose)
    outputs = _fake_landmark_outputs(presence)
    pose._landmarks = _FakeSession(outputs)
    pose._landmarks_input = "input_1"
    pose._landmarks_outputs = list(outputs)
    return pose


def test_landmarks_for_roi_projects_33_body_landmarks_into_the_photo():
    pose = _pose_with_fake_landmark_model(presence=0.9)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    roi = NormRect(0.5, 0.5, 0.4, 0.4, 0.0)

    result = pose.landmarks_for_roi(image, roi, min_presence=0.5)

    assert result.image_xy.shape == (33, 2)
    np.testing.assert_allclose(result.image_xy[0], [0.5, 0.5 - 0.25 * 0.4], atol=1e-6)
    assert result.visibility.min() > 0.99
    np.testing.assert_allclose(result.world[:, 0], 0.25)
    assert result.presence == pytest.approx(0.9)


def test_landmarks_for_roi_drops_a_crop_without_a_person():
    pose = _pose_with_fake_landmark_model(presence=0.2)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert pose.landmarks_for_roi(image, NormRect(0.5, 0.5, 0.4, 0.4, 0.0), min_presence=0.5) is None


def test_create_session_falls_back_to_cpu_when_gpu_not_requested(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "relu.onnx"
    onnx.save_model(model, str(path))

    session = onnx_pose.create_session(path, use_gpu=False)

    assert session.get_providers() == ["CPUExecutionProvider"]
    assert isinstance(session, ort.InferenceSession)


# --- GPU providers: DirectML, CUDA without TF32, CPU fallback ---------------------


def test_cuda_provider_options_disable_tf32():
    """Regression test: ONNX Runtime's CUDA provider defaults to TF32 math on
    Ampere and newer GPUs, which moved landmarks up to 12 px from the CPU result."""
    providers, _dml = onnx_pose.select_providers([onnx_pose.CUDA_PROVIDER, onnx_pose.CPU_PROVIDER], use_gpu=True)
    name, options = providers[0]
    assert name == onnx_pose.CUDA_PROVIDER
    assert str(options["use_tf32"]) == "0"
    assert providers[-1] == onnx_pose.CPU_PROVIDER


def test_directml_is_preferred_and_falls_back_to_the_cpu():
    available = [onnx_pose.DML_PROVIDER, onnx_pose.CUDA_PROVIDER, onnx_pose.CPU_PROVIDER]
    providers, dml = onnx_pose.select_providers(available, use_gpu=True)
    assert dml
    assert [p if isinstance(p, str) else p[0] for p in providers] == [
        onnx_pose.DML_PROVIDER, onnx_pose.CUDA_PROVIDER, onnx_pose.CPU_PROVIDER
    ]


def test_gpu_preference_can_force_cuda_or_directml():
    available = [onnx_pose.DML_PROVIDER, onnx_pose.CUDA_PROVIDER, onnx_pose.CPU_PROVIDER]
    providers, dml = onnx_pose.select_providers(available, use_gpu=True, preference=("cuda",))
    assert not dml and providers[0][0] == onnx_pose.CUDA_PROVIDER and len(providers) == 2


def test_cpu_only_when_gpu_not_requested_or_unavailable():
    assert onnx_pose.select_providers([onnx_pose.CUDA_PROVIDER], use_gpu=False) == ([onnx_pose.CPU_PROVIDER], False)
    assert onnx_pose.select_providers([onnx_pose.CPU_PROVIDER], use_gpu=True) == ([onnx_pose.CPU_PROVIDER], False)


@pytest.mark.parametrize(
    ("providers", "label"),
    [
        (["DmlExecutionProvider", "CPUExecutionProvider"], "GPU (DirectML)"),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "GPU (CUDA)"),
        (["CPUExecutionProvider"], "CPU"),
    ],
)
def test_device_label_names_the_provider(providers, label):
    class _Session:
        def get_providers(self):
            return providers

    pose = BlazePose.__new__(BlazePose)
    pose._detector = pose._landmarks = _Session()
    assert pose.device == label


# --- B8: group-photo tiling ------------------------------------------------------


class _BlobDetector:
    """A stand-in detector session that "finds a face" at the brightest blob of
    its 224x224 input, but only when the blob is at least MIN_SIZE input pixels
    across -- like BlazePose's face-driven detector, which misses faces that
    are only a few pixels wide in the letterboxed whole photo."""

    MIN_SIZE = 6

    def run(self, names, feed):
        tensor = next(iter(feed.values()))[0]
        bright = tensor[..., 0] > 0.5
        raw_boxes = np.zeros((1, 2254, 12), dtype=np.float32)
        raw_scores = np.full((1, 2254, 1), -50.0, dtype=np.float32)
        anchors = ssd_anchor_centers() * 224
        from scipy import ndimage

        labels, count = ndimage.label(bright)
        for k, found in enumerate(ndimage.find_objects(labels)):
            ys, xs = found
            size = max(ys.stop - ys.start, xs.stop - xs.start)
            if size < self.MIN_SIZE:
                continue
            cx, cy = (xs.start + xs.stop) / 2, (ys.start + ys.stop) / 2
            i = 1000 + k
            raw_boxes[0, i, :4] = [cx - anchors[i, 0], cy - anchors[i, 1], size, size]
            # keypoint 0 (hip center) 3 face sizes below the face, keypoint 1 (the
            # body-scale point) at the face, as on an upright person.
            raw_boxes[0, i, 4:8] = [cx - anchors[i, 0], cy + 3 * size - anchors[i, 1], cx - anchors[i, 0], cy - anchors[i, 1]]
            raw_scores[0, i, 0] = 5.0
        return [raw_boxes, raw_scores]


def _pose_with_blob_detector() -> BlazePose:
    pose = BlazePose.__new__(BlazePose)
    pose._detector = _BlobDetector()
    pose._detector_input = "input"
    return pose


def _group_photo(faces_xy, size, shape=(900, 1800)):
    image = np.zeros((*shape, 3), dtype=np.uint8)
    for x, y in faces_xy:
        image[y : y + size, x : x + size] = 255
    return image


def test_tile_regions_cover_the_image_with_overlap():
    regions = tile_regions(1800, 900)
    assert len(regions) == onnx_pose.TILE_GRID**2
    covered = np.zeros((900, 1800), dtype=int)
    for x0, y0, x1, y1 in regions:
        covered[y0:y1, x0:x1] += 1
        assert (x1 - x0) / (y1 - y0) == pytest.approx(2.0, rel=0.01)  # the image's own aspect
    assert covered.min() >= 1
    assert covered[:, 890:910].min() >= 2  # neighbouring tiles overlap across the seams


def test_small_faces_in_a_group_photo_are_found_by_tiling():
    """Regression test: the detector only sees the whole photo shrunk to 224 px,
    where these 30 px faces are under 4 px -- nobody was found. Tiles find all four,
    each mapped back to its own place in the photo."""
    faces = [(200, 150), (1500, 160), (300, 700), (1400, 650)]
    image = _group_photo(faces, size=30)
    pose = _pose_with_blob_detector()

    assert pose.detect_rois(image, max_people=8, min_detection_confidence=0.5, tiled=False) == []
    rois = pose.detect_rois(image, max_people=8, min_detection_confidence=0.5)  # auto: nobody found -> tiles

    assert len(rois) == 4
    h, w = image.shape[:2]
    for x, y in faces:
        # The ROI is centred on keypoint 0, three face sizes below the face.
        expected = ((x + 15) / w, (y + 15 + 3 * 30) / h)
        assert min(math.dist((r.x_center, r.y_center), expected) for r in rois) < 0.01


def test_tiles_do_not_duplicate_or_replace_a_person_the_whole_image_found():
    big = (800, 100)
    image = _group_photo([big], size=200)
    image[600:630, 100:130] = 255  # plus one small face only a tile can see
    pose = _pose_with_blob_detector()

    whole = pose.detect_rois(image, max_people=8, min_detection_confidence=0.5, tiled=False)
    both = pose.detect_rois(image, max_people=8, min_detection_confidence=0.5, tiled=True)

    assert len(whole) == 1 and len(both) == 2
    # The big person's ROI is the whole-image one, untouched by tile estimates.
    assert any(
        r.x_center == pytest.approx(whole[0].x_center) and r.width == pytest.approx(whole[0].width) for r in both
    )


def test_merge_tile_detections_keeps_whole_image_detections_and_merges_tile_duplicates():
    kp = np.zeros((4, 2))
    primary = [Detection(0.7, np.array([0.0, 0.0, 10.0, 10.0]), kp)]
    tiled = [
        Detection(0.9, np.array([1.0, 1.0, 11.0, 11.0]), kp + 1),  # same face as primary
        Detection(0.8, np.array([50.0, 50.0, 60.0, 60.0]), kp + 5),  # new person, seen by two tiles
        Detection(0.6, np.array([51.0, 50.0, 61.0, 60.0]), kp + 6),
    ]
    merged = merge_tile_detections(primary, tiled)
    assert [d.score for d in merged] == [0.8, 0.7]
    assert merged[1] is primary[0]


def test_auto_mode_retries_with_tiles_when_every_whole_image_detection_is_rejected():
    """On a 3x3 collage of a real two-person photo the whole image "found" one
    blob that the landmark model then rejected, so nobody was returned while
    tiling finds four people; auto mode must fall back to tiles then too."""
    image = _group_photo([(200, 150), (1500, 160)], size=30)
    image[300:400, 800:1200] = 255  # a big, non-face blob the whole-image detector accepts
    pose = _pose_with_blob_detector()

    class _PresenceBySize(_FakeSession):
        """Accepts only the small faces' ROIs, as the landmark model would reject a non-person."""

        def run(self, names, feed):
            crop = next(iter(feed.values()))[0]
            outputs = dict(self._outputs)
            # A face fills ~2% of its crop; the wide blob, 0.4% of its much larger one.
            outputs["Identity_1"] = np.array([[0.9 if crop.mean() > 0.01 else 0.1]], dtype=np.float32)
            return [outputs[name] for name in (names or outputs)]

    outputs = _fake_landmark_outputs(0.9)
    pose._landmarks = _PresenceBySize(outputs)
    pose._landmarks_input = "input_1"
    pose._landmarks_outputs = list(outputs)

    assert pose.detect(image, max_people=8, tiled=False) == []
    assert len(pose.detect(image, max_people=8)) == 2

"""Face expressions (F12): the face crops, the blendshape documents and their mirroring, the video
smoothing, the Blender add-on's shape-key matching -- and, when MediaPipe's face model is cached
(~/.cache/pose_craft/models, downloaded on the first "Face expressions" solve), its ONNX conversion
against LiteRT's outputs and the whole pose -> face pipeline against MediaPipe's own result on a
public-domain portrait (tests/data/face, see SOURCES.md)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.estimation import landmarker
from pose_craft.estimation.landmarker import DetectedPerson, FaceLandmarks
from pose_craft.estimation.onnx_pose import (
    FACE_BLENDSHAPE_LANDMARKS,
    FACE_BLENDSHAPE_NAMES,
    face_roi_from_landmarks,
    face_roi_from_pose,
)
from pose_craft.export import blender_pose

sys.path.insert(0, str(Path(__file__).parent))
from test_blender_addon_source import _load_pure_functions  # noqa: E402

DATA = Path(__file__).parent / "data" / "face"
ADDON = _load_pure_functions("face_blendshapes", "shape_key_match_name", "match_shape_keys",
                             "interpolate_pose_documents", "slerp_wxyz")


def test_mediapipes_constants_are_complete():
    assert len(FACE_BLENDSHAPE_LANDMARKS) == 146 and len(set(FACE_BLENDSHAPE_LANDMARKS)) == 146
    assert max(FACE_BLENDSHAPE_LANDMARKS) == 477
    assert len(FACE_BLENDSHAPE_NAMES) == 52 and FACE_BLENDSHAPE_NAMES[0] == "_neutral"


def _pose_head(tilt_degrees: float = 0.0, scale: float = 0.1) -> np.ndarray:
    """A (33, 2) pose with only a plausible head: nose, eyes, ears and mouth, tilted."""
    head = {0: (0.0, 0.0), 2: (0.25, -0.3), 5: (-0.25, -0.3), 7: (0.6, -0.2), 8: (-0.6, -0.2),
            9: (0.15, 0.35), 10: (-0.15, 0.35), 1: (0.15, -0.3), 3: (0.35, -0.3), 4: (-0.15, -0.3), 6: (-0.35, -0.3)}
    # Facing the camera: the subject's right eye (5) is on the image's left, as above.
    angle = math.radians(tilt_degrees)
    pose = np.full((33, 2), 0.5)
    for index, (x, y) in head.items():
        pose[index] = (0.5 + scale * (x * math.cos(angle) - y * math.sin(angle)),
                       0.4 + scale * (x * math.sin(angle) + y * math.cos(angle)))
    return pose


def test_the_face_crop_from_the_pose_is_levelled_on_the_eyes_and_covers_the_head():
    for tilt in (0.0, 20.0, -35.0):
        roi = face_roi_from_pose(_pose_head(tilt), 1000, 1000)
        assert math.isclose(roi.rotation, math.radians(tilt), abs_tol=1e-6)
        assert roi.width * 1000 >= 1.2 * 0.1 * 1000 * 1.9 - 1e-6  # ear to ear (1.2 * scale) widened 1.9x
        assert math.isclose(roi.width, roi.height)  # square in a square image
    pose = _pose_head()
    pose[5] = pose[2]  # coincident eyes: no crop
    assert face_roi_from_pose(pose, 640, 480) is None


def test_the_next_crop_from_the_faces_own_landmarks():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.4, 0.6, (478, 2))
    xy[33], xy[263] = (0.42, 0.45), (0.58, 0.45)  # level eye corners
    roi = face_roi_from_landmarks(xy, 800, 800)
    assert math.isclose(roi.rotation, 0.0, abs_tol=1e-9)
    low, high = xy.min(axis=0), xy.max(axis=0)
    assert math.isclose(roi.x_center, (low[0] + high[0]) / 2) and math.isclose(roi.y_center, (low[1] + high[1]) / 2)
    assert math.isclose(roi.width, max(high - low) * 1.5)


def test_the_face_goes_into_the_pose_and_animation_documents_and_mirrors():
    rotations = {"spine": Rotation.identity()}
    face = {"jawOpen": 0.31234, "mouthSmileLeft": 0.9, "eyeBlinkRight": 0.2}
    document = blender_pose.build_pose_document(rotations, face_blendshapes=face)
    assert document["face"] == {"blendshapes": {"jawOpen": 0.3123, "mouthSmileLeft": 0.9, "eyeBlinkRight": 0.2}}
    assert "face" not in blender_pose.build_pose_document(rotations)
    assert blender_pose.mirror_blendshapes(face) == {"jawOpen": 0.31234, "mouthSmileRight": 0.9, "eyeBlinkLeft": 0.2}
    animation = blender_pose.build_animation_document([(0, rotations), (1, rotations)], 30.0, faces=[face, None])
    assert animation["frames"][0]["pose"]["face"]["blendshapes"]["jawOpen"] == 0.3123
    assert "face" not in animation["frames"][1]["pose"]
    with pytest.raises(ValueError):
        blender_pose.build_animation_document([(0, rotations)], 30.0, faces=[face, face])


def test_video_smoothing_keeps_the_face_and_holds_it_briefly():
    from pose_craft.estimation.video import HAND_HOLD_SECONDS, smooth_people

    def person(face):
        return DetectedPerson(landmarks=np.zeros((33, 3)), image_landmarks_xy=np.full((33, 2), 0.5),
                              visibility=np.ones(33), face=face)

    open_jaw = FaceLandmarks(blendshapes={"jawOpen": 1.0, "eyeBlinkLeft": 0.0}, image_xy=np.zeros((478, 2)), presence=0.9)
    fps = 30.0
    hold = int(round(HAND_HOLD_SECONDS * fps))
    frames = [person(open_jaw)] * 5 + [person(None)] * (hold + 2)
    smoothed = smooth_people(frames, fps)
    assert all(0.0 <= p.face.blendshapes["jawOpen"] <= 1.0 for p in smoothed[:5])
    assert smoothed[4].face.blendshapes["jawOpen"] > 0.9
    assert all(p.face is not None for p in smoothed[5:5 + hold])  # a brief miss keeps the expression
    assert smoothed[-1].face is None


def test_the_addon_matches_shape_keys_by_arkit_name():
    weights = ADDON["face_blendshapes"]({"face": {"blendshapes": {"jawOpen": 1.4, "mouthSmileLeft": 0.5, "bad": "x"}}})
    assert weights == {"jawOpen": 1.0, "mouthSmileLeft": 0.5}  # clamped, malformed entries dropped
    assert ADDON["face_blendshapes"]({"bones": {}}) == {}
    keys = ["Basis", "JawOpen", "blendShape1.mouthSmileLeft", "mouth_smile_right", "Fcl_ALL_Joy"]
    assert ADDON["match_shape_keys"](keys, weights) == {"JawOpen": 1.0, "blendShape1.mouthSmileLeft": 0.5}


def test_the_addon_blends_faces_when_resampling():
    a = {"bones": {}, "face": {"blendshapes": {"jawOpen": 0.0, "eyeBlinkLeft": 1.0}}}
    b = {"bones": {}, "face": {"blendshapes": {"jawOpen": 1.0, "mouthPucker": 0.4}}}
    mid = ADDON["interpolate_pose_documents"](a, b, 0.25)["face"]["blendshapes"]
    assert math.isclose(mid["jawOpen"], 0.25)
    assert mid["eyeBlinkLeft"] == 1.0 and mid["mouthPucker"] == 0.0  # only one side has it: the nearer pose's


_face_model = pytest.mark.skipif(
    not landmarker.is_face_model_available(),
    reason="MediaPipe's face model isn't cached (a 'Face expressions' solve downloads it)",
)


@_face_model
def test_the_blendshape_model_converts_exactly():
    """The converted blendshape model (SUB, MUL, SUM, SQRT, TRANSPOSE, STRIDED_SLICE ... added to
    tflite_to_onnx for it) against LiteRT running the original TFLite model."""
    import onnxruntime as ort

    paths = landmarker.resolve_face_onnx_models()
    session = ort.InferenceSession(str(paths["blendshapes"]), providers=["CPUExecutionProvider"])
    for case in json.loads((DATA / "blendshapes_litert_reference.json").read_text())["cases"]:
        points = np.asarray(case["points"], dtype=np.float32)[None]
        scores = session.run(None, {session.get_inputs()[0].name: points})[0].reshape(-1)
        assert np.abs(scores - np.asarray(case["scores"])).max() < 1e-4


@_face_model
def test_a_real_portrait_reads_like_mediapipe():
    """The whole path -- BlazePose, the face crop from its head points, the landmark and blendshape
    models -- against MediaPipe's own face landmarker on a smiling NASA portrait."""
    import cv2

    reference = json.loads((DATA / "portrait_smiling.mediapipe.json").read_text())["blendshapes"]
    with landmarker.PoseEstimator.for_variant("full", max_people=1, use_gpu=False) as estimator:
        estimator.enable_face()
        estimator.face = True
        people = estimator.detect(cv2.imread(str(DATA / "portrait_smiling.jpg")))
    assert len(people) == 1 and people[0].face is not None
    ours = people[0].face.blendshapes
    assert set(ours) == set(reference)
    diffs = np.array([abs(ours[name] - reference[name]) for name in reference])
    assert diffs.mean() < 0.03 and diffs.max() < 0.2
    for name in ("mouthSmileLeft", "mouthSmileRight"):
        assert ours[name] > 0.8  # the photo's broad smile
    assert ours["eyeBlinkLeft"] < 0.3 and ours["jawOpen"] < 0.3

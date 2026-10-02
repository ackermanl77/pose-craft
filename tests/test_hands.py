"""Hands and fingers (B11): the pose-driven hand crop, the hand landmark pass,
15 finger bones per hand in the skeleton, and finger rotations in the JSON,
BVH and add-on name tables -- with body-only output unchanged when no hand
landmarks are given."""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.estimation import landmarker, onnx_pose
from pose_craft.estimation.landmarker import DetectedPerson, HandLandmarks
from pose_craft.export import bvh_export as bvh
from pose_craft.export.blender_pose import build_pose_document
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _articulated_pose, _rigidly_rotated, _standing_t_pose  # noqa: E402

ADDON_PATH = Path(__file__).resolve().parent.parent / "pose_craft" / "export" / "blender_addon" / "import_pose.py"


def _rest_hand(wrist: np.ndarray, side: str, scale: float = 0.01) -> np.ndarray:
    """A palm-down T-pose hand at `wrist` (engine frame, meters), laid out
    exactly like the rest skeleton's hand: every finger bone at rest."""
    hand = np.zeros((sk.NUM_HAND_LANDMARKS, 3))
    hand[0] = wrist
    sign = 1.0 if side == "L" else -1.0
    for finger in sk.FINGERS:
        base = np.array(sk.FINGER_BASE_OFFSETS_CM[finger]) * scale
        base[0] *= sign
        first = sk._FINGER_FIRST_LANDMARK[finger]
        point = wrist + base
        hand[first] = point
        direction = sk._finger_rest_direction(finger, side)
        for segment, length in enumerate(sk.FINGER_LENGTHS_CM[finger]):
            point = point + direction * length * scale
            hand[first + segment + 1] = point
    return hand


def _curl(hand: np.ndarray, finger: str, side: str, angle_deg: float) -> np.ndarray:
    """Curls a finger toward the palm (-Y) by `angle_deg` at each of its joints."""
    hand = hand.copy()
    first = sk._FINGER_FIRST_LANDMARK[finger]
    sign = 1.0 if side == "L" else -1.0
    for joint in range(first, first + 3):
        pivot = hand[joint].copy()
        # Curling toward the palm: rotate about the knuckle line (+Z), down for either hand.
        turn = Rotation.from_rotvec([0.0, 0.0, -sign * math.radians(angle_deg)])
        hand[joint + 1:first + 4] = pivot + turn.apply(hand[joint + 1:first + 4] - pivot)
    return hand


def _t_pose_with_hands():
    lm = _standing_t_pose()
    hands = {"L": _rest_hand(lm[sk.LEFT_WRIST], "L"), "R": _rest_hand(lm[sk.RIGHT_WRIST], "R")}
    return lm, hands


# --- the skeleton -------------------------------------------------------------------


def test_finger_bone_names_follow_rigify_and_hang_off_the_hands():
    assert len(sk.FINGER_BONE_NAMES) == 30
    assert "f_index.01.L" in sk.FINGER_BONE_NAMES and "thumb.03.R" in sk.FINGER_BONE_NAMES
    for bone in sk.FINGER_BONES:
        expected = f"hand.{bone.side}" if bone.segment == 1 else f"{bone.finger}.0{bone.segment - 1}.{bone.side}"
        assert bone.parent == expected
    assert sk.ALL_BONE_NAMES == sk.BONE_NAMES + sk.FINGER_BONE_NAMES


def test_without_hands_the_solve_is_exactly_the_body_only_one():
    lm = _articulated_pose()
    body = sk.solve_bone_rotations(lm)
    assert list(body) == sk.BONE_NAMES
    assert sk.solve_bone_rotations(lm, hands=None).keys() == body.keys()
    assert sk.solve_bone_rotations(lm, hands={}).keys() == body.keys()


def test_a_rest_hand_solves_to_straight_fingers():
    lm, hands = _t_pose_with_hands()
    rotations = sk.solve_bone_rotations(lm, hands=hands)
    assert set(sk.FINGER_BONE_NAMES) <= set(rotations)
    for name in sk.FINGER_BONE_NAMES + ["hand.L", "hand.R"]:
        assert rotations[name].magnitude() < 1e-6, name


@pytest.mark.parametrize("side", ["L", "R"])
def test_a_curled_finger_is_a_hinge_about_the_knuckle_line(side):
    lm, hands = _t_pose_with_hands()
    hands[side] = _curl(hands[side], "f_index", side, 30.0)
    rotations = sk.solve_bone_rotations(lm, hands=hands)
    for segment in (1, 2, 3):
        local = rotations[f"f_index.0{segment}.{side}"]
        assert local.magnitude() == pytest.approx(math.radians(30.0), abs=1e-6)
        axis = local.as_rotvec() / local.magnitude()
        np.testing.assert_allclose(np.abs(axis), [0.0, 0.0, 1.0], atol=1e-6)  # about the knuckle line
    assert rotations[f"f_middle.01.{side}"].magnitude() < 1e-6  # other fingers untouched
    # In bone axes (X = the knuckle line), a curl reads as a pure X rotation.
    x, y, z = sk.bone_local_euler_deg(f"f_index.02.{side}", rotations[f"f_index.02.{side}"])
    assert abs(x) == pytest.approx(30.0, abs=1e-4) and abs(y) < 1e-4 and abs(z) < 1e-4


def test_finger_rotations_are_invariant_to_a_rigid_turn_of_the_whole_body():
    lm, hands = _t_pose_with_hands()
    hands["L"] = _curl(hands["L"], "f_ring", "L", 40.0)
    turn = Rotation.from_euler("xyz", [20, -60, 15], degrees=True)
    turned_lm = _rigidly_rotated(lm, turn)
    pivot = sk._hip_center(lm)
    turned_hands = {s: pivot + turn.apply(h - pivot) for s, h in hands.items()}
    before = sk.solve_bone_rotations(lm, hands=hands)
    after = sk.solve_bone_rotations(turned_lm, hands=turned_hands)
    for name in sk.FINGER_BONE_NAMES:
        assert (before[name] * after[name].inv()).magnitude() < 1e-6, name


def test_forward_kinematics_with_measured_hands_lands_on_the_hand_landmarks():
    lm, hands = _t_pose_with_hands()
    hands["R"] = _curl(hands["R"], "f_middle", "R", 50.0)
    hands["L"] = _curl(hands["L"], "thumb", "L", 20.0)
    rotations = sk.solve_bone_rotations(lm, hands=hands)
    lengths, shoulder, hip = sk.measured_proportions(lm, hands)
    points = sk.forward_kinematics(rotations, lengths, shoulder, hip, root_position=sk._hip_center(lm))
    expected = sk.joint_positions(lm, hands)
    for name in sk.FINGER_BONE_NAMES:
        for end in ("head", "tail"):
            np.testing.assert_allclose(points[f"{name}.{end}"], expected[f"{name}.{end}"], atol=1e-6, err_msg=name)


def test_mirroring_hands_swaps_sides_and_reflects_x():
    lm, hands = _t_pose_with_hands()
    mirrored = sk.mirror_hands(hands)
    np.testing.assert_allclose(mirrored["R"][:, 0], -hands["L"][:, 0])
    np.testing.assert_allclose(mirrored["L"][:, 1:], hands["R"][:, 1:])
    assert sk.mirror_hands({"L": 0.9}) == {"R": 0.9}
    xy = {"L": np.full((21, 2), 0.25)}
    np.testing.assert_allclose(sk.mirror_hands(xy)["R"][:, 0], 0.75)
    # A mirrored pose with mirrored hands still solves straight fingers.
    rotations = sk.solve_bone_rotations(sk.mirror_landmarks(lm), hands=mirrored)
    assert all(rotations[n].magnitude() < 1e-6 for n in sk.FINGER_BONE_NAMES)


def test_an_arm_depth_flip_reflects_that_hand_too():
    lm, hands = _t_pose_with_hands()
    hands["L"][:, 2] += 0.05
    flipped = sk.flip_hand_depth(lm, hands, "forearm.L")
    elbow_z = lm[sk.LEFT_ELBOW][2]
    np.testing.assert_allclose(flipped["L"][:, 2], 2 * elbow_z - hands["L"][:, 2])
    np.testing.assert_allclose(flipped["R"], hands["R"])
    np.testing.assert_allclose(sk.flip_hand_depth(lm, hands, "shin.L")["L"], hands["L"])  # a leg flip: no change


def test_finger_confidence_comes_from_the_hand_presence_and_the_wrist():
    visibility = np.ones(sk.NUM_LANDMARKS)
    visibility[sk.RIGHT_WRIST] = 0.4
    scores = sk.bone_confidence(visibility, {"L": 0.8, "R": 0.95})
    assert scores["f_index.02.L"] == pytest.approx(0.8)
    assert scores["thumb.01.R"] == pytest.approx(0.4)
    assert "f_index.01.L" not in sk.bone_confidence(visibility)


# --- exports ------------------------------------------------------------------------------


def test_the_json_carries_finger_bones_only_when_solved():
    lm, hands = _t_pose_with_hands()
    hands["L"] = _curl(hands["L"], "f_pinky", "L", 25.0)
    with_fingers = build_pose_document(sk.solve_bone_rotations(lm, hands=hands))
    entry = with_fingers["bones"]["f_pinky.02.L"]
    assert {"quaternion_wxyz", "world_quaternion_wxyz", "rest_direction"} <= set(entry)
    np.testing.assert_allclose(entry["rest_direction"], [1.0, 0.0, 0.0], atol=1e-12)  # engine +X -> Blender +X
    body_only = build_pose_document(sk.solve_bone_rotations(lm))
    assert set(body_only["bones"]) == set(sk.BONE_NAMES)
    assert body_only["version"] == with_fingers["version"] == 2


def _bvh_joint_names(text: str) -> list[str]:
    return [line.split()[1] for line in text.splitlines() if line.strip().startswith(("ROOT", "JOINT"))]


def test_the_bvh_gets_finger_joints_only_when_solved_and_stays_consistent():
    lm, hands = _t_pose_with_hands()
    body_only = bvh.build_bvh_document(sk.solve_bone_rotations(lm))
    assert _bvh_joint_names(body_only) == bvh._channel_order()  # the 15-joint file, unchanged
    assert len(_bvh_joint_names(body_only)) == 15

    hands["R"] = _curl(hands["R"], "f_index", "R", 35.0)
    rotations = sk.solve_bone_rotations(lm, hands=hands)
    text = bvh.build_bvh_document(rotations)
    names = _bvh_joint_names(text)
    assert len(names) == 45 and set(sk.FINGER_BONE_NAMES) <= set(names)
    assert names.index("hand.R") < names.index("f_index.01.R") < names.index("f_index.03.R")
    motion = text.strip().splitlines()[-1].split()
    assert len(motion) == 6 + 3 * (len(names) - 1)
    # The finger's MOTION channels decode back to its local rotation.
    offset = 6 + 3 * (names.index("f_index.02.R") - 1)
    z, x, y = (float(v) for v in motion[offset : offset + 3])
    decoded = Rotation.from_euler(bvh.BVH_ROTATION_ORDER, [z, x, y], degrees=True)
    assert (decoded * rotations["f_index.02.R"].inv()).magnitude() < 1e-5


def test_bvh_finger_offsets_follow_the_rest_hand():
    names = _bvh_joint_names(bvh.build_bvh_document(sk.solve_bone_rotations(*_t_pose_with_hands()[:1], hands=_t_pose_with_hands()[1])))
    assert "thumb.01.L" in names
    np.testing.assert_allclose(bvh._offset("f_index.01.L"), sk.FINGER_BASE_OFFSETS_CM["f_index"])
    np.testing.assert_allclose(bvh._offset("f_index.02.L"), [sk.FINGER_LENGTHS_CM["f_index"][0], 0.0, 0.0])
    np.testing.assert_allclose(bvh._offset("f_index.01.R")[0], -sk.FINGER_BASE_OFFSETS_CM["f_index"][0])


# --- the hand crop and the hand landmark pass -----------------------------------------------


def test_hand_roi_from_pose_is_centred_past_the_knuckles_and_upright_for_a_raised_hand():
    # A hand pointing straight up in the image: wrist below, knuckles above.
    roi = onnx_pose.hand_roi_from_pose((0.5, 0.6), (0.49, 0.5), (0.52, 0.5), 1000, 1000)
    assert roi.rotation == pytest.approx(0.0, abs=0.05)
    middle_y = 0.5
    box = 2.0 * math.hypot(0.5 - 0.5, 0.6 - middle_y)  # ~2x wrist -> middle knuckle
    assert roi.width == pytest.approx(box * onnx_pose.HAND_ROI_FROM_POSE_SCALE, rel=0.02)
    # Shifted by -0.1 of its (pre-scale) height along its own up axis: toward the fingers.
    assert roi.y_center == pytest.approx(middle_y - 0.1 * box, abs=1e-3)
    # A hand pointing right in the image: rotated a quarter turn.
    right = onnx_pose.hand_roi_from_pose((0.4, 0.5), (0.5, 0.49), (0.5, 0.52), 1000, 1000)
    assert abs(right.rotation) == pytest.approx(math.pi / 2, abs=0.05)
    assert onnx_pose.hand_roi_from_pose((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), 100, 100) is None


def test_hand_roi_from_landmarks_covers_the_hand():
    xy = np.zeros((21, 2))
    xy[0] = [0.5, 0.7]
    for finger, x in zip((1, 5, 9, 13, 17), (0.42, 0.46, 0.5, 0.54, 0.58)):
        for k in range(4):
            xy[finger + k] = [x, 0.6 - 0.04 * k]
    roi = onnx_pose.hand_roi_from_landmarks(xy, 640, 480)
    assert roi.rotation == pytest.approx(0.0, abs=0.1)
    corners = onnx_pose.project_to_image(np.array([[0, 0], [1, 1]], dtype=float), roi)
    assert corners[0, 0] < xy[:, 0].min() and corners[1, 0] > xy[:, 0].max()
    assert corners[0, 1] < xy[:, 1].min() and corners[1, 1] > xy[:, 1].max()


class _FakeHandSession:
    """Stands in for the converted hand model: a fixed hand in the crop."""

    def __init__(self, presence=0.9):
        self.presence = presence
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        raw = np.zeros((21, 3), dtype=np.float32)
        raw[:, 0] = np.linspace(80, 150, 21)
        raw[:, 1] = 112.0
        world = np.zeros((21, 3), dtype=np.float32)
        world[:, 0] = np.linspace(-0.05, 0.05, 21)
        return [raw.reshape(1, 63), np.array([[self.presence]], np.float32), np.array([[0.5]], np.float32),
                world.reshape(1, 63)]


def _hand_landmarker(presence=0.9):
    model = onnx_pose.HandLandmarker.__new__(onnx_pose.HandLandmarker)
    model._session = _FakeHandSession(presence)
    model._input = "input_1"
    model._outputs = ["Identity", "Identity_1", "Identity_2", "Identity_3"]
    return model


def test_hand_landmarker_projects_crop_landmarks_into_the_image_and_refines_once():
    model = _hand_landmarker()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = model.detect(image, (0.5, 0.6), (0.49, 0.5), (0.52, 0.5))
    assert result is not None and result.image_xy.shape == (21, 2) and result.world.shape == (21, 3)
    assert model._session.calls == 2  # the pose crop, then the landmarks' own crop
    assert np.all((result.image_xy > 0) & (result.image_xy < 1))
    assert _hand_landmarker(presence=0.1).detect(image, (0.5, 0.6), (0.49, 0.5), (0.52, 0.5)) is None


def test_the_estimator_finds_both_hands_and_anchors_them_at_the_pose_wrists():
    estimator = landmarker.PoseEstimator.__new__(landmarker.PoseEstimator)
    estimator._hand_model = _hand_landmarker()
    lm = _standing_t_pose()
    visibility = np.ones(33)
    visibility[sk.RIGHT_INDEX] = 0.1  # the right hand is hidden: not searched
    result = onnx_pose.PoseResult(np.full((33, 2), 0.5) + np.linspace(0, 0.1, 33)[:, None], visibility, np.zeros((33, 3)), 0.9)
    person = DetectedPerson(lm, result.image_xy, visibility)
    estimator._find_hands(np.zeros((480, 640, 3), dtype=np.uint8), person, result)
    assert set(person.hands) == {"L"}
    hand = person.hands["L"]
    np.testing.assert_allclose(hand.world[0], lm[sk.LEFT_WRIST])
    assert person.hand_presence() == {"L": pytest.approx(0.9)}


def test_detect_without_the_hands_option_leaves_people_without_hands():
    estimator = landmarker.PoseEstimator.__new__(landmarker.PoseEstimator)
    estimator._hand_model = _hand_landmarker()
    estimator.max_people, estimator.hands = 1, False

    class _Pose:
        def detect(self, image, **kwargs):
            xy = np.full((33, 2), 0.5) + np.linspace(0, 0.1, 33)[:, None]
            return [onnx_pose.PoseResult(xy, np.ones(33), np.zeros((33, 3)), 0.9)]

    estimator._pose = _Pose()
    people = estimator.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert people[0].hands == {}
    estimator.hands = True
    assert estimator.detect(np.zeros((64, 64, 3), dtype=np.uint8))[0].hands  # found with the option on


def test_the_hand_model_is_pinned_like_the_pose_models():
    assert "/float16/1/" in landmarker.HAND_MODEL_URL and "/latest/" not in landmarker.HAND_MODEL_URL
    assert all(len(h) == 64 for h in landmarker.HAND_MODEL_SHA256)


def test_the_real_hand_model_runs_when_cached():
    pytest.importorskip("onnxruntime")
    if not landmarker.is_hand_model_available():
        pytest.skip("the hand model isn't cached or bundled")
    model = onnx_pose.HandLandmarker(landmarker.resolve_hand_onnx_model(), use_gpu=False)
    image = np.full((240, 320, 3), 127, dtype=np.uint8)
    roi = onnx_pose.NormRect(0.5, 0.5, 0.5, 0.6, 0.0)
    result = model.landmarks_for_roi(image, roi, min_presence=0.0)
    assert result.image_xy.shape == (21, 2) and np.all(np.isfinite(result.world))
    assert 0.0 <= result.presence <= 1.0
    assert model.landmarks_for_roi(image, roi) is None  # a blank grey image holds no hand


# --- the add-on's name tables ----------------------------------------------------------------


def _addon_constant(name: str):
    tree = ast.parse(ADDON_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_addon_finger_names_mirror_the_skeleton_and_map_to_mixamo():
    assert _addon_constant("POSECRAFT_FINGER_BONE_NAMES") == sk.FINGER_BONE_NAMES
    mixamo = _addon_constant("MIXAMO_FINGER_BONE_NAMES")
    assert set(mixamo) == set(sk.FINGER_BONE_NAMES)
    assert mixamo["f_index.01.L"] == "LeftHandIndex1" and mixamo["thumb.03.R"] == "RightHandThumb3"


def test_addon_guesses_rigify_and_mixamo_finger_bones():
    from test_blender_addon_source import _load_pure_functions

    candidates = _load_pure_functions("bone_name_candidates")["bone_name_candidates"]
    assert candidates("f_middle.02.L")[0] == "f_middle.02.L"  # Rigify's finger control, by name
    assert "mixamorig:LeftHandMiddle2" in candidates("f_middle.02.L")
    assert "DEF-thumb.01.R" in candidates("thumb.01.R")


def test_hand_landmarks_dataclass_round_trip():
    hand = HandLandmarks(np.zeros((21, 3)), np.zeros((21, 2)), 0.7)
    person = DetectedPerson(np.zeros((33, 3)), np.zeros((33, 2)), np.ones(33), hands={"R": hand})
    assert set(person.hand_world()) == {"R"} and person.hand_presence() == {"R": 0.7}
    assert json.dumps(person.hand_presence())

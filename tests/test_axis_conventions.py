"""End-to-end axis-convention tests: raw MediaPipe world landmarks ->
_world_to_engine -> solve_bone_rotations -> Blender JSON / BVH, checked by
where a limb actually ends up on the target rig rather than by comparing
quaternion components. These are the "semantic" regression net for the
depth-mirroring bug (a knee raised toward the camera used to export *behind*
the rig) and the related left/right and mirror-checkbox bugs.

MediaPipe's world frame, as documented for BlazePose: origin at the hip
midpoint, +x toward image right, +y DOWN (image-like), and z is depth with
smaller = closer to the camera. For a subject facing the camera, their own
left side is on image right (+x).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.estimation.landmarker import _world_to_engine
from pose_craft.export import blender_pose as bp
from pose_craft.export import bvh_export as bvh
from pose_craft.retarget import skeleton as sk

# Blender rig conventions: Z up, a character faces -Y, and its .L side is +X.
BLENDER_UP = np.array([0.0, 0.0, 1.0])
BLENDER_FRONT = np.array([0.0, -1.0, 0.0])
BLENDER_LEFT = np.array([1.0, 0.0, 0.0])

# Rest (T-pose) bone directions on a Blender rig, in armature space.
_BLENDER_REST = {
    "thigh.L": -BLENDER_UP,
    "shin.L": -BLENDER_UP,
    "upper_arm.L": BLENDER_LEFT,
    "upper_arm.R": -BLENDER_LEFT,
    "foot.L": BLENDER_FRONT,
}


def _mediapipe_standing_person() -> np.ndarray:
    """A subject facing the camera, arms straight out (T-pose), in RAW
    MediaPipe world coordinates (+y down, smaller z = closer to the camera)."""
    lm = np.zeros((sk.NUM_LANDMARKS, 3))
    lm[sk.LEFT_HIP], lm[sk.RIGHT_HIP] = [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]
    lm[sk.LEFT_SHOULDER], lm[sk.RIGHT_SHOULDER] = [0.2, -0.5, 0.0], [-0.2, -0.5, 0.0]
    lm[sk.LEFT_EAR], lm[sk.RIGHT_EAR] = [0.07, -0.65, 0.0], [-0.07, -0.65, 0.0]
    lm[sk.LEFT_EYE], lm[sk.RIGHT_EYE] = [0.03, -0.65, -0.08], [-0.03, -0.65, -0.08]
    lm[sk.NOSE] = [0.0, -0.62, -0.1]

    for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
        shoulder = lm[getattr(sk, f"{side}_SHOULDER")]
        lm[getattr(sk, f"{side}_ELBOW")] = shoulder + [sign * 0.25, 0.0, 0.0]
        lm[getattr(sk, f"{side}_WRIST")] = shoulder + [sign * 0.5, 0.0, 0.0]
        lm[getattr(sk, f"{side}_INDEX")] = shoulder + [sign * 0.6, 0.0, -0.02]
        lm[getattr(sk, f"{side}_PINKY")] = shoulder + [sign * 0.6, 0.0, 0.02]
        hip = lm[getattr(sk, f"{side}_HIP")]
        lm[getattr(sk, f"{side}_KNEE")] = hip + [0.0, 0.45, 0.0]
        lm[getattr(sk, f"{side}_ANKLE")] = hip + [0.0, 0.85, 0.0]
        # Toes point toward the camera: smaller z.
        lm[getattr(sk, f"{side}_FOOT_INDEX")] = hip + [0.0, 0.85, -0.2]
    return lm


def _blender_local_quaternions(raw_mediapipe: np.ndarray, mirror: bool = False) -> dict[str, Rotation]:
    landmarks = _world_to_engine(raw_mediapipe)
    if mirror:
        landmarks = sk.mirror_landmarks(landmarks)
    document = bp.build_pose_document(sk.solve_bone_rotations(landmarks))
    out = {}
    for name, entry in document["bones"].items():
        w, x, y, z = entry["quaternion_wxyz"]
        out[name] = Rotation.from_quat([x, y, z, w])
    return out


def _blender_world_direction(quats: dict[str, Rotation], bone: str) -> np.ndarray:
    """World (armature-space) direction of `bone` after posing, composing the
    exported parent-relative deltas down PoseCraft's own hierarchy."""
    spec = {b.name: b for b in sk.BONES}
    chain = []
    name = bone
    while name is not None:
        chain.append(name)
        name = spec[name].parent
    world = Rotation.identity()
    for name in reversed(chain):
        world = world * quats[name]
    return world.apply(_BLENDER_REST[bone])


def test_standing_person_exports_as_near_identity():
    quats = _blender_local_quaternions(_mediapipe_standing_person())
    for name, rot in quats.items():
        assert np.linalg.norm(rot.as_rotvec()) < 1e-6, name


def test_knee_raised_toward_camera_ends_up_in_front_of_the_blender_rig():
    raw = _mediapipe_standing_person()
    # Left thigh horizontal, pointing at the camera (smaller MediaPipe z).
    raw[sk.LEFT_KNEE] = raw[sk.LEFT_HIP] + [0.0, 0.0, -0.45]
    raw[sk.LEFT_ANKLE] = raw[sk.LEFT_KNEE] + [0.0, 0.4, 0.0]
    raw[sk.LEFT_FOOT_INDEX] = raw[sk.LEFT_ANKLE] + [0.0, 0.0, -0.2]

    thigh_dir = _blender_world_direction(_blender_local_quaternions(raw), "thigh.L")
    assert np.dot(thigh_dir, BLENDER_FRONT) > 0.99, thigh_dir


def test_knee_raised_toward_camera_ends_up_in_front_in_the_bvh():
    raw = _mediapipe_standing_person()
    raw[sk.LEFT_KNEE] = raw[sk.LEFT_HIP] + [0.0, 0.0, -0.45]
    raw[sk.LEFT_ANKLE] = raw[sk.LEFT_KNEE] + [0.0, 0.4, 0.0]

    rotations = sk.solve_bone_rotations(_world_to_engine(raw))
    positions = _bvh_forward_kinematics(bvh.build_bvh_document(rotations))
    thigh = positions["shin.L"] - positions["thigh.L"]
    # This exporter's BVH frame: Y up, the character faces +Z.
    assert thigh[2] > 0.99 * np.linalg.norm(thigh), thigh


def test_left_arm_raised_to_the_subjects_left_stays_on_the_left_side():
    raw = _mediapipe_standing_person()
    # Left arm up and out at 45 degrees, toward image right (+x) and up (-y).
    raw[sk.LEFT_ELBOW] = raw[sk.LEFT_SHOULDER] + [0.18, -0.18, 0.0]
    raw[sk.LEFT_WRIST] = raw[sk.LEFT_ELBOW] + [0.18, -0.18, 0.0]
    raw[sk.LEFT_INDEX] = raw[sk.LEFT_WRIST] + [0.07, -0.07, -0.01]
    raw[sk.LEFT_PINKY] = raw[sk.LEFT_WRIST] + [0.07, -0.07, 0.01]

    arm = _blender_world_direction(_blender_local_quaternions(raw), "upper_arm.L")
    expected = (BLENDER_LEFT + BLENDER_UP) / np.sqrt(2.0)
    assert np.dot(arm, expected) > 0.99, arm


def test_mirrored_t_pose_is_still_a_t_pose():
    quats = _blender_local_quaternions(_mediapipe_standing_person(), mirror=True)
    for name, rot in quats.items():
        assert np.linalg.norm(rot.as_rotvec()) < 1e-6, name


def test_mirror_moves_a_raised_left_arm_to_the_right_arm():
    raw = _mediapipe_standing_person()
    raw[sk.LEFT_ELBOW] = raw[sk.LEFT_SHOULDER] + [0.18, -0.18, 0.0]
    raw[sk.LEFT_WRIST] = raw[sk.LEFT_ELBOW] + [0.18, -0.18, 0.0]

    quats = _blender_local_quaternions(raw, mirror=True)
    right_arm = _blender_world_direction(quats, "upper_arm.R")
    left_arm = _blender_world_direction(quats, "upper_arm.L")
    assert np.dot(right_arm, (-BLENDER_LEFT + BLENDER_UP) / np.sqrt(2.0)) > 0.99, right_arm
    assert np.dot(left_arm, BLENDER_LEFT) > 0.99, left_arm  # the other arm is still straight out


def test_level_head_has_no_pitch():
    quats = _blender_local_quaternions(_mediapipe_standing_person())
    assert np.degrees(np.linalg.norm(quats["head"].as_rotvec())) < 1.0


def test_nodding_head_pitches_forward_by_the_nod_angle():
    raw = _mediapipe_standing_person()
    ear_center = (raw[sk.LEFT_EAR] + raw[sk.RIGHT_EAR]) / 2.0
    # 30-degree nod about the lateral (x) axis through the ears: in MediaPipe's
    # frame (+y down, -z toward camera), the eyes/nose move down and toward the camera.
    nod = Rotation.from_euler("x", 30, degrees=True)
    for index in (sk.LEFT_EYE, sk.RIGHT_EYE, sk.NOSE):
        raw[index] = ear_center + nod.apply(raw[index] - ear_center)

    quats = _blender_local_quaternions(raw)
    head_up = quats["head"].apply(BLENDER_UP)  # neck is upright, so local == world here
    assert abs(np.degrees(np.linalg.norm(quats["head"].as_rotvec())) - 30.0) < 1.0
    assert np.dot(head_up, BLENDER_FRONT) > 0.4  # tipped toward the front, not the back


# --- BVH forward kinematics ---------------------------------------------------


def _parse_bvh(document: str):
    """Minimal BVH reader: returns (joints, motion) where joints is a list of
    (name, parent, offset, end_site_offset or None) in file order."""
    tokens = document.split()
    joints = []
    stack: list[str] = []
    i = 0
    last_name = None
    pending_end = False
    while tokens[i] != "MOTION":
        tok = tokens[i]
        if tok in ("ROOT", "JOINT"):
            last_name = tokens[i + 1]
            joints.append([last_name, stack[-1] if stack else None, None, None])
            i += 2
        elif tok == "End":
            pending_end = True
            i += 2
        elif tok == "{":
            stack.append("__end__" if pending_end else last_name)
            i += 1
        elif tok == "}":
            if stack.pop() == "__end__":
                pending_end = False
            i += 1
        elif tok == "OFFSET":
            offset = np.array([float(v) for v in tokens[i + 1 : i + 4]])
            if pending_end:
                owner = next(j for j in reversed(joints) if j[0] == stack[-2])
                owner[3] = offset
            else:
                joints[-1][2] = offset
            i += 4
        elif tok == "CHANNELS":
            i += 2 + int(tokens[i + 1])
        else:
            i += 1
    frame_line = document.splitlines()[-1]
    motion = [float(v) for v in frame_line.split()]
    return joints, motion


def _bvh_forward_kinematics(document: str) -> dict[str, np.ndarray]:
    """Evaluates rest OFFSETs + the single MOTION frame into global positions:
    '<joint>' for each joint's own position, '<joint>.end' for End Sites."""
    joints, motion = _parse_bvh(document)
    root_pos = np.array(motion[:3])
    values = motion[3:]
    rot: dict[str, Rotation] = {}
    pos: dict[str, np.ndarray] = {}
    for k, (name, parent, offset, end) in enumerate(joints):
        z, x, y = values[3 * k : 3 * k + 3]
        local = Rotation.from_euler(bvh.BVH_ROTATION_ORDER, [z, x, y], degrees=True)
        if parent is None:
            pos[name] = root_pos + offset
            rot[name] = local
        else:
            pos[name] = pos[parent] + rot[parent].apply(offset)
            rot[name] = rot[parent] * local
        if end is not None:
            pos[f"{name}.end"] = pos[name] + rot[name].apply(end)
    return pos


def _posed_landmarks_matching_bvh_proportions() -> np.ndarray:
    """Engine-frame landmarks for a non-trivial pose, built with the exact
    segment lengths/attachment offsets of the BVH rest skeleton, so the BVH's
    forward kinematics should reproduce every joint position exactly."""
    L = bvh.BONE_LENGTHS_CM
    lean = Rotation.from_euler("x", 20, degrees=True)  # torso leans toward the camera
    up = lean.apply([0.0, 1.0, 0.0])
    lateral = lean.apply([1.0, 0.0, 0.0])

    def unit(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    lm = np.zeros((sk.NUM_LANDMARKS, 3))
    lm[sk.LEFT_HIP] = lateral * bvh.HIP_HALF_WIDTH_CM
    lm[sk.RIGHT_HIP] = -lateral * bvh.HIP_HALF_WIDTH_CM
    shoulder_center = up * L["spine"]
    lm[sk.LEFT_SHOULDER] = shoulder_center + lateral * bvh.SHOULDER_HALF_WIDTH_CM
    lm[sk.RIGHT_SHOULDER] = shoulder_center - lateral * bvh.SHOULDER_HALF_WIDTH_CM

    neck_dir = unit(lean.apply([0.0, 1.0, 0.3]))
    ear_center = shoulder_center + neck_dir * L["neck"]
    lm[sk.LEFT_EAR], lm[sk.RIGHT_EAR] = ear_center + lateral * 7, ear_center - lateral * 7
    forward = lean.apply([0.0, 0.0, 1.0])
    lm[sk.LEFT_EYE], lm[sk.RIGHT_EYE] = ear_center + forward * 8 + lateral * 3, ear_center + forward * 8 - lateral * 3
    lm[sk.NOSE] = ear_center + forward * 10 - up * 3

    def limb(root, upper_idx, dirs, lengths):
        point = root
        for idx, d, length in zip(upper_idx, dirs, lengths):
            point = point + unit(d) * length
            lm[idx] = point

    limb(lm[sk.LEFT_SHOULDER], [sk.LEFT_ELBOW, sk.LEFT_WRIST], [[1, 1, 0.5], [0.2, 1, 1]],
         [L["upper_arm.L"], L["forearm.L"]])
    limb(lm[sk.RIGHT_SHOULDER], [sk.RIGHT_ELBOW, sk.RIGHT_WRIST], [[-0.3, -1, 0], [-0.2, -1, 0.4]],
         [L["upper_arm.R"], L["forearm.R"]])
    for side, hand_dir in (("LEFT", unit([0.1, 1, 1])), ("RIGHT", unit([0, -1, 0.3]))):
        wrist = lm[getattr(sk, f"{side}_WRIST")]
        tip = wrist + hand_dir * L[f"hand.{side[0]}"]
        spread = unit(np.cross(hand_dir, [1.0, 0.0, 0.0])) * 2.0
        lm[getattr(sk, f"{side}_INDEX")], lm[getattr(sk, f"{side}_PINKY")] = tip + spread, tip - spread

    limb(lm[sk.LEFT_HIP], [sk.LEFT_KNEE, sk.LEFT_ANKLE, sk.LEFT_FOOT_INDEX],
         [[0, -0.3, 1], [0, -1, 0.1], [0, -0.2, 1]], [L["thigh.L"], L["shin.L"], L["foot.L"]])
    limb(lm[sk.RIGHT_HIP], [sk.RIGHT_KNEE, sk.RIGHT_ANKLE, sk.RIGHT_FOOT_INDEX],
         [[-0.1, -1, -0.2], [0, -1, -0.3], [0, 0, 1]], [L["thigh.R"], L["shin.R"], L["foot.R"]])
    return lm


def test_bvh_forward_kinematics_reproduces_the_solved_skeleton():
    landmarks = _posed_landmarks_matching_bvh_proportions()
    rotations = sk.solve_bone_rotations(landmarks)
    fk = _bvh_forward_kinematics(bvh.build_bvh_document(rotations))
    expected = sk.joint_positions(landmarks)

    children = {b.name: [c.name for c in sk.BONES if c.parent == b.name] for b in sk.BONES}
    for bone in sk.BONES:
        head = expected[f"{bone.name}.head"]
        np.testing.assert_allclose(fk[bone.name], head, atol=1e-4, err_msg=f"{bone.name} head")
        if not children[bone.name]:
            tail_fk = fk[f"{bone.name}.end"]
            if bone.name == "head":
                # The head's tail is a derived skull-top point, not a segment of
                # BONE_LENGTHS_CM's length -- only compare its direction.
                want = expected["head.tail"] - head
                got = tail_fk - head
                np.testing.assert_allclose(got / np.linalg.norm(got), want / np.linalg.norm(want), atol=1e-6)
            else:
                np.testing.assert_allclose(tail_fk, expected[f"{bone.name}.tail"], atol=1e-4,
                                           err_msg=f"{bone.name} tail")


def test_bvh_end_site_is_the_leaf_bones_own_vector():
    joints, _motion = _parse_bvh(bvh.build_bvh_document({}))
    by_name = {name: (offset, end) for name, _parent, offset, end in joints}
    for leaf in ("hand.L", "foot.R", "head"):
        spec = next(b for b in sk.BONES if b.name == leaf)
        np.testing.assert_allclose(by_name[leaf][1], spec.rest_direction * bvh.BONE_LENGTHS_CM[leaf], atol=1e-9)

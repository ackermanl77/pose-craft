from __future__ import annotations

import ast
from pathlib import Path

from pose_craft.export.blender_pose import FORMAT_VERSION
from pose_craft.retarget.skeleton import BONE_NAMES

ADDON_PATH = Path(__file__).resolve().parent.parent / "pose_craft" / "export" / "blender_addon" / "import_pose.py"


def _extract_top_level_constant(name: str) -> object:
    """Parses import_pose.py's source (rather than importing it -- it needs `bpy`,
    which isn't installed in this test environment) to pull out a module-level
    literal constant, so tests here can catch it drifting without needing Blender.
    """
    tree = ast.parse(ADDON_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {ADDON_PATH}")


def test_addon_bone_names_match_skeleton_bone_names():
    """POSECRAFT_BONE_NAMES is a hand-maintained mirror of skeleton.BONE_NAMES
    (the add-on can't import pose_craft, since it runs inside Blender's own
    Python) -- this is the regression guard for it drifting out of sync, which
    would otherwise silently break the manual bone-mapping UI's row list.
    """
    assert _extract_top_level_constant("POSECRAFT_BONE_NAMES") == BONE_NAMES


def test_addon_max_supported_format_version_matches_export_format_version():
    """MAX_SUPPORTED_FORMAT_VERSION is a hand-maintained mirror of
    blender_pose.FORMAT_VERSION, for the same reason POSECRAFT_BONE_NAMES
    mirrors BONE_NAMES -- catches the add-on silently falling behind (and
    rejecting every file/Live Link document) after a real format bump.
    """
    assert _extract_top_level_constant("MAX_SUPPORTED_FORMAT_VERSION") == FORMAT_VERSION


def test_addon_source_is_syntactically_valid():
    # Cheap sanity check that complements py_compile in dev: a parse failure here
    # would also break every other test in this file with a less obvious traceback.
    ast.parse(ADDON_PATH.read_text(encoding="utf-8"))


def _load_pure_functions(*names: str) -> dict:
    """Compiles just the named pure (bpy-free) functions and classes plus the
    module-level literal constants they use out of the add-on's source, since
    the module itself can't be imported without Blender."""
    import json
    import math
    import time

    tree = ast.parse(ADDON_PATH.read_text(encoding="utf-8"))
    namespace: dict = {"json": json, "math": math, "time": time}
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            for target in node.targets:
                namespace[target.id] = value
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {f.name for f in functions} == set(names)
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(ADDON_PATH), "exec"), namespace)
    return namespace


def test_rigify_candidates_put_the_fk_tag_before_the_side_suffix():
    candidates = _load_pure_functions("bone_name_candidates")["bone_name_candidates"]
    arm = candidates("upper_arm.L")
    assert "upper_arm_fk.L" in arm
    assert "upper_arm.L_fk" not in arm
    # Control bones are preferred over constraint-driven DEF-/ORG- bones.
    assert arm.index("upper_arm_fk.L") < arm.index("DEF-upper_arm.L") < arm.index("ORG-upper_arm.L")
    assert candidates("shin.R")[:2] == ["shin.R", "shin_fk.R"]
    assert candidates("spine")[:2] == ["spine", "spine_fk"]


def test_mixamo_candidates_cover_every_bone():
    candidates = _load_pure_functions("bone_name_candidates")["bone_name_candidates"]
    assert "mixamorig:LeftArm" in candidates("upper_arm.L")
    assert "mixamorig:RightUpLeg" in candidates("thigh.R")
    for name in BONE_NAMES:
        assert any(c.startswith("mixamorig:") for c in candidates(name)), name
        assert len(set(candidates(name))) == len(candidates(name))


class _Quat:
    """Minimal stand-in for mathutils.Quaternion (`@` and .inverted())."""

    def __init__(self, rotation):
        self.r = rotation

    def __matmul__(self, other):
        return _Quat(self.r * other.r)

    def inverted(self):
        return _Quat(self.r.inv())


def test_armature_delta_to_basis_reproduces_the_armature_space_delta_down_a_chain():
    """Simulates Blender's pose evaluation (pose = parent_pose @
    parent_rest^-1 @ rest @ basis, rotation part only) for a two-bone chain
    whose rest frames are NOT aligned with the armature axes, and checks that
    the converted basis rotations give each bone the world-space delta the
    exported parent-relative rotations describe."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    to_basis = _load_pure_functions("armature_delta_to_basis")["armature_delta_to_basis"]
    rest_parent = Rotation.from_euler("xyz", [10, -70, 35], degrees=True)
    rest_child = Rotation.from_euler("xyz", [-40, 20, 95], degrees=True)
    delta_parent = Rotation.from_rotvec([0.3, -0.2, 0.9])  # exported local (root: == world delta)
    delta_child = Rotation.from_rotvec([-0.5, 0.4, 0.1])  # exported, relative to the parent

    basis_parent = to_basis(_Quat(rest_parent), _Quat(delta_parent)).r
    basis_child = to_basis(_Quat(rest_child), _Quat(delta_child)).r
    pose_parent = rest_parent * basis_parent
    pose_child = pose_parent * rest_parent.inv() * rest_child * basis_child

    np.testing.assert_allclose((pose_parent * rest_parent.inv()).as_matrix(), delta_parent.as_matrix(), atol=1e-9)
    np.testing.assert_allclose(
        (pose_child * rest_child.inv()).as_matrix(), (delta_parent * delta_child).as_matrix(), atol=1e-9
    )


# --- B5: retargeting against the rig's own rest pose and hierarchy -------------------


def _rest_with_direction(direction, roll_deg):
    """A bone rest rotation whose local Y axis points along `direction`, rolled."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(y, d)
    if np.linalg.norm(axis) < 1e-9:
        swing = Rotation.identity() if d[1] > 0 else Rotation.from_rotvec([np.pi, 0.0, 0.0])
    else:
        swing = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(np.clip(np.dot(y, d), -1, 1)))
    return swing * Rotation.from_euler("y", roll_deg, degrees=True)


def _align(a, b):
    """Shortest-arc rotation taking a onto b (what mathutils' rotation_difference returns)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-12:
        return Rotation.identity()
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arccos(np.clip(np.dot(a, b), -1, 1)))


# A Mixamo-like rig in Blender armature axes (Z up, facing -Y, .L on +X): a Hips
# root above both the spine and the thighs, clavicles between the spine and the
# arms, A-pose arms 45 degrees down, and feet sloping 50 degrees down.
_S = 2 ** -0.5
_MIXAMO_LIKE_RIG = {
    # name: (parent, rest direction, roll in degrees, PoseCraft bone or None)
    "Hips": (None, (0, 0, 1), 0, None),
    "Spine": ("Hips", (0, 0, 1), 10, "spine"),
    "Neck": ("Spine", (0, -0.2, 1), -20, "neck"),
    "Head": ("Neck", (0, 0, 1), 5, "head"),
    "LeftShoulder": ("Spine", (1, 0, 0.1), 30, None),
    "LeftArm": ("LeftShoulder", (_S, 0, -_S), 40, "upper_arm.L"),
    "LeftForeArm": ("LeftArm", (_S, -0.1, -_S), -60, "forearm.L"),
    "LeftHand": ("LeftForeArm", (_S, 0, -_S), 15, "hand.L"),
    "RightShoulder": ("Spine", (-1, 0, 0.1), -30, None),
    "RightArm": ("RightShoulder", (-_S, 0, -_S), 70, "upper_arm.R"),
    "RightForeArm": ("RightArm", (-_S, -0.1, -_S), 0, "forearm.R"),
    "RightHand": ("RightForeArm", (-_S, 0, -_S), 90, "hand.R"),
    "LeftUpLeg": ("Hips", (0.05, 0, -1), 180, "thigh.L"),
    "LeftLeg": ("LeftUpLeg", (0, 0.05, -1), 170, "shin.L"),
    "LeftFoot": ("LeftLeg", (0, -0.64, -0.77), -10, "foot.L"),
    "RightUpLeg": ("Hips", (-0.05, 0, -1), 180, "thigh.R"),
    "RightLeg": ("RightUpLeg", (0, 0.05, -1), 190, "shin.R"),
    "RightFoot": ("RightLeg", (0, -0.64, -0.77), 20, "foot.R"),
}


def _posed_document():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from scipy.spatial.transform import Rotation
    from test_retarget_frames import _articulated_pose, _rigidly_rotated

    from pose_craft.export.blender_pose import build_pose_document
    from pose_craft.retarget import skeleton as sk

    landmarks = _rigidly_rotated(_articulated_pose(), Rotation.from_euler("xyz", [15, 50, -10], degrees=True))
    return build_pose_document(sk.solve_bone_rotations(landmarks))


def _quat_wxyz(values):
    from scipy.spatial.transform import Rotation

    w, x, y, z = values
    return Rotation.from_quat([x, y, z, w])


def _evaluate(rig, rest, basis):
    """Blender's pose evaluation, rotation part: pose = parent_pose @ parent_rest^-1 @ rest @ basis."""
    from scipy.spatial.transform import Rotation

    pose = {}
    for name, (parent, *_rest) in rig.items():
        chain = rest[name] if parent is None else pose[parent] * rest[parent].inv() * rest[name]
        pose[name] = chain * basis.get(name, Rotation.identity())
    return pose


def test_rig_pose_basis_aims_every_bone_of_an_a_pose_mixamo_like_rig():
    """Regression test for the A-pose/hierarchy limitation: applying the v1
    parent-relative deltas on this rig leaves the arms 45 degrees low and the
    thighs ignoring the spine; the rig-aware solve must point every mapped bone
    exactly where PoseCraft's skeleton points it."""
    import numpy as np

    solve = _load_pure_functions("rig_pose_basis")["rig_pose_basis"]
    document = _posed_document()
    rest = {name: _rest_with_direction(d, roll) for name, (_p, d, roll, _m) in _MIXAMO_LIKE_RIG.items()}
    parents = {name: parent for name, (parent, *_x) in _MIXAMO_LIKE_RIG.items()}
    y = np.array([0.0, 1.0, 0.0])

    targets, deltas = {}, {}
    for name, (_parent, _d, _roll, posecraft) in _MIXAMO_LIKE_RIG.items():
        if posecraft is None:
            continue
        entry = document["bones"][posecraft]
        world = _quat_wxyz(entry["world_quaternion_wxyz"])
        # What the add-on does: align the rig's rest direction onto PoseCraft's,
        # then apply PoseCraft's world rotation.
        deltas[name] = _Quat(world * _align(rest[name].apply(y), entry["rest_direction"]))
        targets[name] = world.apply(entry["rest_direction"])

    basis = solve(list(_MIXAMO_LIKE_RIG), parents, {n: _Quat(r) for n, r in rest.items()}, {}, deltas)
    pose = _evaluate(_MIXAMO_LIKE_RIG, rest, {n: q.r for n, q in basis.items()})

    for name, target in targets.items():
        np.testing.assert_allclose(pose[name].apply(y), target, atol=1e-9, err_msg=name)
    # Unmapped bones (hips, clavicles) stay at rest.
    for name in ("Hips", "LeftShoulder", "RightShoulder"):
        assert name not in basis

    # The v1 path on the same rig: the A-pose arm ends up far from its target.
    to_basis = _load_pure_functions("armature_delta_to_basis")["armature_delta_to_basis"]
    v1_basis = {}
    for name, (_parent, _d, _roll, posecraft) in _MIXAMO_LIKE_RIG.items():
        if posecraft is not None:
            local = _quat_wxyz(document["bones"][posecraft]["quaternion_wxyz"])
            v1_basis[name] = to_basis(_Quat(rest[name]), _Quat(local)).r
    v1_pose = _evaluate(_MIXAMO_LIKE_RIG, rest, v1_basis)
    v1_error = np.degrees(np.arccos(np.clip(np.dot(v1_pose["LeftArm"].apply(y), targets["LeftArm"]), -1, 1)))
    assert v1_error > 20


def test_rig_pose_basis_equals_the_v1_formula_on_a_posecraft_shaped_rig():
    import numpy as np

    from pose_craft.export.blender_pose import _ENGINE_TO_BLENDER_AXES
    from pose_craft.retarget import skeleton as sk

    loaded = _load_pure_functions("rig_pose_basis", "armature_delta_to_basis")
    document = _posed_document()
    rest = {
        b.name: _rest_with_direction(_ENGINE_TO_BLENDER_AXES.apply(b.rest_direction), 17 * i)
        for i, b in enumerate(sk.BONES)
    }
    parents = {b.name: b.parent for b in sk.BONES}
    deltas = {name: _Quat(_quat_wxyz(document["bones"][name]["world_quaternion_wxyz"])) for name in sk.BONE_NAMES}

    basis = loaded["rig_pose_basis"](sk.BONE_NAMES, parents, {n: _Quat(r) for n, r in rest.items()}, {}, deltas)

    for name in sk.BONE_NAMES:
        v1 = loaded["armature_delta_to_basis"](
            _Quat(rest[name]), _Quat(_quat_wxyz(document["bones"][name]["quaternion_wxyz"]))
        ).r
        assert (basis[name].r * v1.inv()).magnitude() < 1e-9, name


def test_bones_parents_first_orders_any_hierarchy():
    order = _load_pure_functions("bones_parents_first")["bones_parents_first"]
    parents = {"c": "b", "b": "a", "a": None, "d": "a"}
    result = order(parents)
    assert sorted(result) == sorted(parents)
    for name, parent in parents.items():
        if parent is not None:
            assert result.index(parent) < result.index(name)


# --- B15: Live Link request/response ------------------------------------------------


def test_live_link_protocol_and_port_match_the_app():
    from pose_craft.export import live_link

    assert _extract_top_level_constant("LIVE_LINK_PROTOCOL_VERSION") == live_link.PROTOCOL_VERSION
    assert _extract_top_level_constant("DEFAULT_LIVE_LINK_PORT") == live_link.DEFAULT_PORT


def test_live_link_reply_frame_is_what_the_app_client_reads():
    import json
    import socket
    import threading

    from pose_craft.export import live_link

    loaded = _load_pure_functions("build_live_link_reply", "encode_live_link_message")
    reply = loaded["build_live_link_reply"](
        applied=["LeftArm"], requested=15, unmatched=["foot.L"], low_confidence=[], armature="Armature"
    )
    frame = loaded["encode_live_link_message"](reply)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def _serve():
        conn, _addr = server.accept()
        with conn:
            length = int.from_bytes(conn.recv(4), "big")
            while length > 0:
                length -= len(conn.recv(length))
            conn.sendall(frame)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    received = live_link.send_pose_document({"format": "pose_craft_pose"}, port=server.getsockname()[1], timeout=2.0)
    thread.join(timeout=2.0)
    server.close()

    assert received == json.loads(json.dumps(reply))
    assert received["ok"] is True and received["requested"] == 15 and received["armature"] == "Armature"
    assert received["protocol"] == live_link.PROTOCOL_VERSION


def test_live_link_reply_for_an_error():
    loaded = _load_pure_functions("build_live_link_reply")
    reply = loaded["build_live_link_reply"](error="Select an armature in Blender")
    assert reply["ok"] is False and reply["error"] == "Select an armature in Blender"
    assert reply["applied"] == []


def test_live_link_request_protocol_check():
    check = _load_pure_functions("live_link_request_error")["live_link_request_error"]
    assert check({"format": "pose_craft_pose"}) is None  # an app from before the field existed
    assert check({"live_link_protocol": 2}) is None
    assert "update" in check({"live_link_protocol": 99}).lower()


# --- B16: the Live Link server never blocks Blender's main thread ----------------


def _frame(document) -> bytes:
    import json

    payload = json.dumps(document).encode("utf-8")
    return len(payload).to_bytes(4, "big") + payload


def test_live_link_reader_assembles_a_message_from_any_chunking():
    reader_cls = _load_pure_functions("LiveLinkReader")["LiveLinkReader"]
    document = {"format": "pose_craft_pose", "bones": {"spine": {"quaternion_wxyz": [1, 0, 0, 0]}}}
    frame = _frame(document)
    for chunk in (1, 3, 4, 7, len(frame)):
        reader = reader_cls(deadline=1e9)
        results = [reader.feed(frame[i : i + chunk]) for i in range(0, len(frame), chunk)]
        assert results[-1] == document
        assert all(r is None for r in results[:-1])


def test_live_link_reader_rejects_oversized_and_malformed_payloads():
    import pytest

    loaded = _load_pure_functions("LiveLinkReader")
    assert loaded["MAX_LIVE_LINK_PAYLOAD_BYTES"] <= 256 * 1024  # a 15-bone document is ~4 KB
    reader = loaded["LiveLinkReader"](deadline=1e9)
    with pytest.raises(ValueError):
        reader.feed((10 * 1024 * 1024).to_bytes(4, "big"))  # rejected from the header alone
    reader = loaded["LiveLinkReader"](deadline=1e9)
    with pytest.raises(ValueError):
        reader.feed((3).to_bytes(4, "big") + b"{x}")


def test_a_stalled_client_never_blocks_a_poll_and_expires():
    """Regression test: the timer callback did blocking recv()s with a 2 s
    timeout each, so a client that connected and stalled froze Blender's UI.
    Polled the way the server now does (non-blocking recv into a reader), each
    tick returns at once and the connection expires at its deadline."""
    import socket
    import time

    loaded = _load_pure_functions("LiveLinkReader")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname(), timeout=2)
    conn, _addr = server.accept()
    conn.setblocking(False)
    try:
        client.sendall(_frame({"format": "pose_craft_pose"})[:6])  # header + 2 bytes, then stall
        reader = loaded["LiveLinkReader"](deadline=time.monotonic() + 0.3)
        received = None
        expired = False
        for _tick in range(20):
            start = time.monotonic()
            try:
                while received is None:
                    data = conn.recv(65536)
                    if not data:
                        break
                    received = reader.feed(data)
            except BlockingIOError:
                pass
            assert time.monotonic() - start < 0.05  # the tick never waits on the socket
            if reader.expired(time.monotonic()):
                expired = True
                break
            time.sleep(0.05)
        assert received is None and expired
    finally:
        for sock in (conn, client, server):
            sock.close()


# --- B17: the Live Link target is captured, not "whatever is selected" -------------


def test_live_link_target_prefers_the_captured_armature():
    pick = _load_pure_functions("pick_live_link_target")["pick_live_link_target"]

    class _Obj:
        def __init__(self, kind):
            self.type = kind

    rig, other_rig, light = _Obj("ARMATURE"), _Obj("ARMATURE"), _Obj("LIGHT")
    assert pick(rig, light) is rig  # the user selected a light after starting Live Link
    assert pick(rig, other_rig) is rig
    assert pick(None, other_rig) is other_rig  # nothing captured: fall back to the active armature
    assert pick(None, light) is None
    assert pick(light, None) is None  # a stale non-armature pointer never wins


# --- B19: a bone keeps its rotation mode -------------------------------------------


class _MathQuat:
    """Stand-in for mathutils.Quaternion's conversions."""

    def __init__(self, rotation):
        self.r = rotation

    def to_euler(self, order, compatible=None):
        # mathutils' Euler order "XYZ" applies X first: scipy's extrinsic "xyz".
        return tuple(self.r.as_euler(order.lower()))

    def to_axis_angle(self):
        import numpy as np

        rotvec = self.r.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        return (tuple(rotvec / angle) if angle else (0.0, 1.0, 0.0)), angle


def test_rotation_is_written_in_the_bones_own_rotation_mode():
    import numpy as np
    from scipy.spatial.transform import Rotation

    loaded = _load_pure_functions("rotation_channel", "rotation_data_path")
    channel = loaded["rotation_channel"]
    rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    q = _MathQuat(rotation)

    path, value = channel("ZXY", q)
    assert path == "rotation_euler"
    np.testing.assert_allclose(Rotation.from_euler("zxy", value).as_matrix(), rotation.as_matrix(), atol=1e-9)
    path, value = channel("AXIS_ANGLE", q)
    assert path == "rotation_axis_angle" and len(value) == 4
    np.testing.assert_allclose(np.array(value[1:]) * value[0], rotation.as_rotvec(), atol=1e-9)
    assert channel("QUATERNION", q) == ("rotation_quaternion", q)
    assert loaded["rotation_data_path"]("YXZ") == "rotation_euler"


# --- B5: spine chains and "Inherit Rotation" off ------------------------------------


class _SlerpQuat(_Quat):
    def slerp(self, other, factor):
        from scipy.spatial.transform import Rotation, Slerp

        both = Rotation.concatenate([self.r, other.r])
        return _SlerpQuat(Slerp([0.0, 1.0], both)([factor])[0])


def test_distribute_rotation_spreads_a_bend_evenly_and_ends_at_the_full_rotation():
    import numpy as np
    from scipy.spatial.transform import Rotation

    distribute = _load_pure_functions("distribute_rotation")["distribute_rotation"]
    delta = Rotation.from_rotvec([0.0, 0.9, 0.3])
    shares = distribute(_SlerpQuat(Rotation.identity()), _SlerpQuat(delta), 3)
    assert len(shares) == 3
    assert (shares[-1].r * delta.inv()).magnitude() < 1e-9
    steps = [shares[0].r] + [shares[k].r * shares[k - 1].r.inv() for k in (1, 2)]
    np.testing.assert_allclose([s.magnitude() for s in steps], delta.magnitude() / 3, atol=1e-9)


def test_mapping_rows_and_mixamo_spines_name_a_chain():
    loaded = _load_pure_functions("parse_bone_chain", "mixamo_chain_extras")
    assert loaded["parse_bone_chain"]("Spine", " Spine1 ,Spine2, ") == ["Spine", "Spine1", "Spine2"]
    assert loaded["parse_bone_chain"]("Spine", "") == ["Spine"]
    assert loaded["parse_bone_chain"]("", "") == []
    extras = loaded["mixamo_chain_extras"]
    existing = {"mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2"}
    assert extras("spine", "mixamorig:Spine", existing) == ["mixamorig:Spine1", "mixamorig:Spine2"]
    assert extras("spine", "spine_fk", existing) == []
    assert extras("neck", "mixamorig:Neck", existing) == []


def _evaluate_with_inherit(rig, rest, basis, inherit):
    """Blender's pose evaluation, rotation part, honoring "Inherit Rotation" off
    (the bone's rotation then ignores its parent's pose)."""
    from scipy.spatial.transform import Rotation

    pose = {}
    for name, (parent, *_rest) in rig.items():
        if parent is None or not inherit.get(name, True):
            chain = rest[name]
        else:
            chain = pose[parent] * rest[parent].inv() * rest[name]
        pose[name] = chain * basis.get(name, Rotation.identity())
    return pose


def _mixamo_three_bone_spine_rig():
    """_MIXAMO_LIKE_RIG with Mixamo's real three-bone spine: the neck and
    clavicles hang from Spine2, at the top of the chain."""
    rig = {"Hips": _MIXAMO_LIKE_RIG["Hips"], "Spine": ("Hips", (0, 0, 1), 10, "spine")}
    rig["Spine1"] = ("Spine", (0, -0.1, 1), 0, None)
    rig["Spine2"] = ("Spine1", (0, 0.05, 1), -5, None)
    for name, entry in _MIXAMO_LIKE_RIG.items():
        if name in rig:
            continue
        rig[name] = ("Spine2",) + entry[1:] if entry[0] == "Spine" else entry
    return rig


def test_a_spine_chain_shares_the_bend_and_the_rest_of_the_rig_still_lands_on_target():
    """B5: the mapped spine used to take the whole bend while Spine1/Spine2
    followed rigidly. Spread over the chain, the bend accumulates up the
    chain, the top of the chain ends at the single spine bone's full rotation,
    and the neck, arms and legs still reach their targets."""
    import numpy as np

    loaded = _load_pure_functions("rig_pose_basis", "distribute_rotation")
    rig = _mixamo_three_bone_spine_rig()
    document = _posed_document()
    rest = {name: _rest_with_direction(d, roll) for name, (_p, d, roll, _m) in rig.items()}
    parents = {name: parent for name, (parent, *_x) in rig.items()}
    y = np.array([0.0, 1.0, 0.0])

    targets, deltas = {}, {}
    for name, (_parent, _d, _roll, posecraft) in rig.items():
        if posecraft is None:
            continue
        entry = document["bones"][posecraft]
        world = _quat_wxyz(entry["world_quaternion_wxyz"])
        targets[name] = world.apply(entry["rest_direction"])
        if posecraft == "spine":
            shares = loaded["distribute_rotation"](_SlerpQuat(_quat_wxyz([1, 0, 0, 0])), _SlerpQuat(world), 3)
            for bone, share in zip(("Spine", "Spine1", "Spine2"), shares):
                deltas[bone] = _Quat(share.r * _align(rest[bone].apply(y), entry["rest_direction"]))
        else:
            deltas[name] = _Quat(world * _align(rest[name].apply(y), entry["rest_direction"]))

    basis = loaded["rig_pose_basis"](list(rig), parents, {n: _Quat(r) for n, r in rest.items()}, {}, deltas)
    pose = _evaluate(rig, rest, {n: q.r for n, q in basis.items()})

    for name, target in targets.items():
        if name != "Spine":
            np.testing.assert_allclose(pose[name].apply(y), target, atol=1e-9, err_msg=name)
    np.testing.assert_allclose(pose["Spine2"].apply(y), targets["Spine"], atol=0.15)  # the top aims ~ like the spine
    world_spine = _quat_wxyz(document["bones"]["spine"]["world_quaternion_wxyz"])
    bends = [(pose[b] * rest[b].inv()).magnitude() for b in ("Spine", "Spine1", "Spine2")]
    assert bends[0] < bends[1] < bends[2]  # the bend accumulates up the chain
    assert abs(bends[2] - world_spine.magnitude()) < 0.2


def test_rig_pose_basis_honors_bones_that_dont_inherit_rotation():
    import numpy as np

    solve = _load_pure_functions("rig_pose_basis")["rig_pose_basis"]
    document = _posed_document()
    rest = {name: _rest_with_direction(d, roll) for name, (_p, d, roll, _m) in _MIXAMO_LIKE_RIG.items()}
    parents = {name: parent for name, (parent, *_x) in _MIXAMO_LIKE_RIG.items()}
    inherit = {"LeftForeArm": False, "Head": False}
    y = np.array([0.0, 1.0, 0.0])
    targets, deltas = {}, {}
    for name, (_parent, _d, _roll, posecraft) in _MIXAMO_LIKE_RIG.items():
        if posecraft is not None:
            entry = document["bones"][posecraft]
            world = _quat_wxyz(entry["world_quaternion_wxyz"])
            deltas[name] = _Quat(world * _align(rest[name].apply(y), entry["rest_direction"]))
            targets[name] = world.apply(entry["rest_direction"])

    basis = solve(list(_MIXAMO_LIKE_RIG), parents, {n: _Quat(r) for n, r in rest.items()}, {}, deltas, inherit)
    pose = _evaluate_with_inherit(_MIXAMO_LIKE_RIG, rest, {n: q.r for n, q in basis.items()}, inherit)
    for name, target in targets.items():
        np.testing.assert_allclose(pose[name].apply(y), target, atol=1e-9, err_msg=name)

    # Solved as if they inherited (the old behaviour), those bones miss.
    old = solve(list(_MIXAMO_LIKE_RIG), parents, {n: _Quat(r) for n, r in rest.items()}, {}, deltas)
    old_pose = _evaluate_with_inherit(_MIXAMO_LIKE_RIG, rest, {n: q.r for n, q in old.items()}, inherit)
    miss = np.degrees(np.arccos(np.clip(np.dot(old_pose["LeftForeArm"].apply(y), targets["LeftForeArm"]), -1, 1)))
    assert miss > 10


# --- B6: keep the feet on the rig's floor -------------------------------------------


def test_floor_contact_moves_the_root_by_the_drop_of_the_lowest_point():
    import numpy as np
    from scipy.spatial.transform import Rotation

    loaded = _load_pure_functions("floor_contact_shift", "top_ancestor", "armature_translation_to_location")
    shift = loaded["floor_contact_shift"]
    assert shift([0.0, 0.1, 0.9], [0.4, 0.5, 1.2]) == 0.0 - 0.4  # crouch: feet rose 0.4 above the floor
    assert shift([], [1.0]) == 0.0
    parents = {"Hips": None, "Spine": "Hips", "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg"}
    assert loaded["top_ancestor"]("LeftLeg", parents) == "Hips"
    assert loaded["top_ancestor"]("Hips", parents) == "Hips"

    class _RotQuat(_Quat):
        def inverted(self):
            return _RotQuat(self.r.inv())

        def __matmul__(self, vector):
            return self.r.apply(vector)

    rest = Rotation.from_euler("x", 90, degrees=True)  # a root bone lying along +Z (Y axis up)
    location = loaded["armature_translation_to_location"](_RotQuat(rest), np.array([0.0, 0.0, -0.4]))
    np.testing.assert_allclose(rest.apply(location), [0.0, 0.0, -0.4], atol=1e-12)  # moves the head down 0.4

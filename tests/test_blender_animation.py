"""B12 / Live Link animations (add-on 0.7.0, protocol 3): "Send Animation to
Blender" sends a tracked person as a pose_craft_animation document, split into
chunks under the add-on's payload cap; the add-on validates the whole chunk,
poses every frame through the same retarget path as a single pose, keys it at
the clip's timing from Blender's current frame and pushes one undo step after
the last chunk. Blender isn't installed, so -- like test_blender_addon_source.py
and test_rigify.py -- the add-on's functions run outside Blender, here with a
simulated scene and a stand-in for the bpy-heavy apply_pose_document."""
from __future__ import annotations

import json
import socket
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import blender_pose, live_link
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_blender_addon_source import _extract_top_level_constant, _load_pure_functions  # noqa: E402

_PARSE = (
    "validate_pose_document", "_validate_quaternion", "_validate_vector", "is_animation_document", "_plain_int",
    "parse_animation_document", "animation_frame_numbers", "resample_animation", "interpolate_pose_documents",
    "slerp_wxyz", "keep_quaternion_hemisphere",
)


def _rotations(seed: int, hands: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    names = list(sk.BONE_NAMES) + (list(sk.FINGER_BONE_NAMES) if hands else [])
    return {name: Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)) for name in names}


def _animation(indices, fps=30.0, hands=False) -> dict:
    return blender_pose.build_animation_document([(i, _rotations(i, hands)) for i in indices], fps, source="clip.mp4")


def test_the_addon_mirrors_the_animation_format():
    assert _extract_top_level_constant("ANIMATION_FORMAT_NAME") == blender_pose.ANIMATION_FORMAT_NAME
    assert _extract_top_level_constant("MAX_SUPPORTED_ANIMATION_VERSION") == blender_pose.ANIMATION_FORMAT_VERSION
    assert _extract_top_level_constant("bl_info")["version"] >= (0, 7, 0)


def test_build_animation_document_is_one_pose_document_per_solved_frame():
    document = _animation([0, 1, 5])
    assert document["format"] == "pose_craft_animation" and document["fps"] == 30.0
    assert [f["frame_index"] for f in document["frames"]] == [0, 1, 5]
    assert all(f["pose"]["format"] == "pose_craft_pose" and set(f["pose"]["bones"]) == set(sk.BONE_NAMES)
               for f in document["frames"])
    with pytest.raises(ValueError):
        blender_pose.build_animation_document([(3, {}), (2, {})], 30.0)


def test_parse_animation_document_validates_everything_before_keying():
    parse = _load_pure_functions(*_PARSE)["parse_animation_document"]
    parsed = parse(_animation([4, 5, 9]))
    assert [index for index, _pose in parsed["frames"]] == [4, 5, 9]
    assert parsed["first_frame_index"] == 4 and parsed["start_frame"] is None
    assert (parsed["chunk_index"], parsed["chunk_count"]) == (0, 1)

    def broken(change):
        document = json.loads(json.dumps(_animation([0, 1])))
        change(document)
        return document

    bad = [
        broken(lambda d: d.update(version=99)),
        broken(lambda d: d.update(fps=0)),
        broken(lambda d: d.update(frames=[])),
        broken(lambda d: d["frames"].reverse()),  # indices must increase
        broken(lambda d: d["frames"][1]["pose"]["bones"]["spine"].update(quaternion_wxyz=[1, 0, 0])),
        broken(lambda d: d["frames"][0].update(frame_index=-1)),
        broken(lambda d: d.update(chunk={"index": 2, "count": 2})),
        broken(lambda d: d.update(chunk={"index": 1, "count": 2})),  # a later chunk without start_frame
        {"format": "pose_craft_pose"},
    ]
    for document in bad:
        with pytest.raises(ValueError):
            parse(document)
    later = broken(lambda d: d.update(chunk={"index": 1, "count": 2}, start_frame=12, first_frame_index=0))
    assert parse(later)["start_frame"] == 12


def test_frames_keep_the_clips_timing_in_the_scenes_fps():
    numbers = _load_pure_functions(*_PARSE)["animation_frame_numbers"]
    assert numbers([0, 1, 2, 5], 0, 10, 30.0, 30.0) == [10, 11, 12, 15]  # a lost frame leaves a gap
    assert numbers([0, 30, 60], 0, 1, 30.0, 24.0) == [1, 25, 49]  # one second = 24 scene frames
    assert numbers([30, 31], 0, 100, 30.0, 30.0) == [130, 131]  # a later chunk keeps the first one's anchor


def test_quaternion_keys_stay_in_one_hemisphere():
    keep = _load_pure_functions("keep_quaternion_hemisphere")["keep_quaternion_hemisphere"]
    assert keep((1, 0, 0, 0), (-0.9, 0.1, 0, 0)) == (0.9, -0.1, 0.0, 0.0)  # same rotation, no long-way spin
    assert keep((1, 0, 0, 0), (0.9, 0.1, 0, 0)) == (0.9, 0.1, 0.0, 0.0)
    assert keep(None, (-1, 0, 0, 0)) == (-1.0, 0.0, 0.0, 0.0)


# --- the add-on side, on a simulated scene ----------------------------------------------------


class _Scene:
    def __init__(self, frame=1, fps=24, fps_base=1.0, floor=False):
        self.frame_current = frame
        self.render = types.SimpleNamespace(fps=fps, fps_base=fps_base)
        self.posecraft_match_rig_rest = True
        self.posecraft_floor_contact = floor
        self.posecraft_pose_controls = True
        self.posecraft_insert_keyframes = False
        self.posecraft_frame_step = 3
        self.posecraft_target = None
        self.frames_set = []

    def frame_set(self, frame):
        self.frame_current = frame
        self.frames_set.append(frame)


def _addon(scene, armature=None):
    """The add-on's Live Link apply path with bpy and the rig solve stood in:
    apply_pose_document records the frame each pose was applied at."""
    loaded = _load_pure_functions(
        *_PARSE, "_scene_fps", "_apply_animation_document", "_apply_live_link_document",
        "live_link_request_error", "pick_live_link_target", "build_live_link_reply", "_summarize",
    )
    log = {"applied_at": [], "keyed": [], "undo": []}

    def apply_pose_document(obj, document, match_rig_rest, floor_contact, pose_controls):
        log["applied_at"].append(scene.frame_current)
        names = [n for n in document["bones"] if n != "foot.R"]
        return names, ["foot.R"] if "foot.R" in document["bones"] else [], []

    def keyframe_applied(obj, names, sc, frame=None, advance=True, root_location=False):
        log["keyed"].append((frame, advance, len(names)))

    armature = armature or types.SimpleNamespace(type="ARMATURE", name="Rig")
    scene.posecraft_target = armature
    loaded.update(
        apply_pose_document=apply_pose_document,
        _keyframe_applied=keyframe_applied,
        _should_keyframe=lambda sc: False,
        _live_link_status="",
        bpy=types.SimpleNamespace(
            context=types.SimpleNamespace(scene=scene, view_layer=types.SimpleNamespace(objects=types.SimpleNamespace(active=None))),
            ops=types.SimpleNamespace(ed=types.SimpleNamespace(undo_push=lambda message: log["undo"].append(message))),
        ),
    )
    return loaded, log


def test_an_animation_is_keyed_frame_by_frame_and_the_current_frame_restored():
    scene = _Scene(frame=10, fps=30)
    addon, log = _addon(scene)
    document = _animation([0, 1, 3])
    document["live_link_protocol"] = 3
    reply = addon["_apply_live_link_document"](document)

    assert reply["ok"] and reply["animation_frames"] == 3
    assert (reply["start_frame"], reply["end_frame"]) == (10, 13)
    assert log["applied_at"] == [10, 11, 13]  # each pose solved at its own frame (frame 2 was lost: no key)
    assert log["keyed"] == [(10, False, 14), (11, False, 14), (13, False, 14)]  # never "Then advance"
    assert scene.frame_current == 10  # put back where the user was
    assert reply["unmatched"] == ["foot.R"] and reply["requested"] == 15 and reply["armature"] == "Rig"
    assert log["undo"] == ["PoseCraft animation"]  # one Ctrl+Z step
    assert "Keyed 3 frame(s)" in addon["_live_link_status"]


def test_chunks_share_one_anchor_and_one_undo_step():
    scene = _Scene(frame=1, fps=24)
    addon, log = _addon(scene)
    first = dict(_animation([0, 1]), chunk={"index": 0, "count": 2}, first_frame_index=0, live_link_protocol=3)
    reply = addon["_apply_live_link_document"](first)
    assert reply["start_frame"] == 1 and log["undo"] == []  # not yet: the animation isn't complete
    scene.frame_set(50)  # the user scrubbed in between
    second = dict(_animation([30, 31]), chunk={"index": 1, "count": 2}, first_frame_index=0,
                  start_frame=reply["start_frame"], live_link_protocol=3)
    second["fps"] = 30.0
    reply = addon["_apply_live_link_document"](second)
    # 30 fps into 24: clip frames 0-1 cover scene frames 1-1.8 (one key), 30-31 cover 25-25.8; the
    # stretch between them is a tracking gap (frames 2-29 were never sent), so it gets no keys.
    assert log["applied_at"] == [1, 25]
    assert reply["chunk_index"] == 1 and log["undo"] == ["PoseCraft animation"]
    assert scene.frame_current == 50


def test_a_bad_frame_aborts_the_chunk_before_anything_is_keyed():
    scene = _Scene()
    addon, log = _addon(scene)
    document = json.loads(json.dumps(_animation([0, 1, 2])))
    document["frames"][2]["pose"]["bones"]["head"]["quaternion_wxyz"] = [1, 0, "x", 0]
    with pytest.raises(ValueError, match="frame 2"):
        addon["_apply_live_link_document"](document)
    assert log["applied_at"] == [] and log["keyed"] == []


def test_an_older_addon_refuses_an_animation_with_update_the_addon():
    loaded = _load_pure_functions("live_link_request_error")
    loaded["LIVE_LINK_PROTOCOL_VERSION"] = 2  # what add-on 0.6.0 declared
    message = loaded["live_link_request_error"]({"format": "pose_craft_animation", "live_link_protocol": 3})
    assert "update" in message.lower()
    assert loaded["live_link_request_error"]({"format": "pose_craft_pose", "live_link_protocol": 2}) is None


# --- the app side: chunking under the add-on's cap ---------------------------------------------


def test_long_clips_are_chunked_under_the_addons_payload_cap():
    cap = _extract_top_level_constant("MAX_LIVE_LINK_PAYLOAD_BYTES")
    assert live_link.ANIMATION_CHUNK_MAX_BYTES < cap
    document = _animation(range(300), hands=True)  # 10 s at 30 fps, fingers included
    assert len(json.dumps(document)) > cap  # far too big for one request
    chunks = live_link.split_animation_frames(document["frames"])
    assert [f for chunk in chunks for f in chunk] == document["frames"]
    assert all(len(chunk) <= live_link.ANIMATION_CHUNK_MAX_FRAMES for chunk in chunks)
    for chunk in chunks:
        request = {k: v for k, v in document.items() if k != "frames"}
        request.update(frames=chunk, chunk={"index": 99, "count": 99}, first_frame_index=0, start_frame=123456,
                       live_link_protocol=3)
        assert len(json.dumps(request).encode("utf-8")) < cap
    with pytest.raises(ValueError):
        live_link.split_animation_frames([{"blob": "x" * 500}], max_bytes=100)


def _fake_addon_server(scene, addon, requests, reply_override=None):
    """Serves Live Link connections like the add-on: LiveLinkReader framing with
    the real payload cap, then _apply_live_link_document, then the reply."""
    reader_cls = _load_pure_functions("LiveLinkReader")["LiveLinkReader"]
    encode = _load_pure_functions("encode_live_link_message")["encode_live_link_message"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    server.settimeout(5)

    def serve():
        try:
            while True:
                conn, _addr = server.accept()
                with conn:
                    reader = reader_cls(deadline=1e18)
                    document = None
                    while document is None:
                        data = conn.recv(65536)
                        if not data:
                            break
                        document = reader.feed(data)
                    requests.append(document)
                    reply = reply_override(document) if reply_override else addon["_apply_live_link_document"](document)
                    conn.sendall(encode(reply))
        except OSError:
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return server


def test_send_animation_document_end_to_end_with_the_addons_framing():
    scene = _Scene(frame=5, fps=30)
    addon, log = _addon(scene)
    requests = []
    server = _fake_addon_server(scene, addon, requests)
    try:
        document = _animation([2, 3, 4, 6, 7], fps=30.0)
        progress = []
        reply = live_link.send_animation_document(
            document, port=server.getsockname()[1], timeout=5.0, max_frames=2,
            on_chunk=lambda done, total: progress.append((done, total)),
        )
    finally:
        server.close()
    assert reply["ok"] and reply["animation_frames"] == 5 and reply["chunks"] == 3
    assert (reply["start_frame"], reply["end_frame"]) == (5, 10)
    assert log["applied_at"] == [5, 6, 7, 9, 10]  # clip frame 2 on Blender's current frame, gap at 5 kept
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [r["live_link_protocol"] for r in requests] == [3, 3, 3]
    assert "start_frame" not in requests[0] and requests[1]["start_frame"] == requests[2]["start_frame"] == 5
    assert {r["first_frame_index"] for r in requests} == {2}
    assert log["undo"] == ["PoseCraft animation"]  # once, after the last chunk


def test_a_refused_chunk_stops_the_send_and_says_how_far_it_got():
    requests = []

    def refuse_second(document):
        if document["chunk"]["index"] == 0:
            return {"ok": True, "applied": ["spine"], "requested": 15, "armature": "Rig",
                    "animation_frames": 2, "start_frame": 1, "end_frame": 2}
        return {"ok": False, "error": "No target armature in Blender"}

    server = _fake_addon_server(None, None, requests, refuse_second)
    try:
        reply = live_link.send_animation_document(_animation(range(6)), port=server.getsockname()[1],
                                                  timeout=5.0, max_frames=2)
    finally:
        server.close()
    assert reply["ok"] is False and len(requests) == 2
    assert "No target armature" in reply["error"] and "after 2 of 6 frames" in reply["error"]


def test_an_addon_that_answers_like_a_single_pose_is_reported_as_too_old():
    server = _fake_addon_server(None, None, [], lambda d: {"ok": True, "applied": [], "requested": 15})
    try:
        with pytest.raises(ValueError, match="0.7.0"):
            live_link.send_animation_document(_animation([0]), port=server.getsockname()[1], timeout=5.0)
    finally:
        server.close()


# --- add-on 0.8.0: resampling to the scene's frame rate, root motion --------------------------------


def test_a_30_fps_clip_in_a_24_fps_scene_keys_every_frame_once():
    resample = _load_pure_functions(*_PARSE)["resample_animation"]
    poses = {i: blender_pose.build_pose_document(_rotations(i)) for i in range(45)}
    keys = resample([(i, poses[i], None) for i in range(45)], 0, 1, 30.0, 24.0)
    frames = [frame for frame, _pose, _offset in keys]
    assert frames == list(range(1, 37))  # 45 clip frames = 1.5 s = 36 scene frames, none twice
    assert keys[0][1] is poses[0] and keys[4][1] is poses[5]  # frame 5 = clip time 5/24 s = clip frame 5 exactly
    # Frame 2 is 1.25 clip frames in: a blend, between clip frames 1 and 2.
    bone = keys[1][1]["bones"]["spine"]["quaternion_wxyz"]
    a, b = poses[1]["bones"]["spine"]["quaternion_wxyz"], poses[2]["bones"]["spine"]["quaternion_wxyz"]
    expected = (Rotation.from_quat([*a[1:], a[0]]), Rotation.from_quat([*b[1:], b[0]]))
    from scipy.spatial.transform import Slerp

    slerped = Slerp([0, 1], Rotation.concatenate(expected))([0.25]).as_quat()[0]
    assert np.allclose(np.abs(np.dot([bone[1], bone[2], bone[3], bone[0]], slerped)), 1.0, atol=1e-6)


def test_a_slower_clip_gets_a_key_on_every_scene_frame_and_gaps_stay_unkeyed():
    resample = _load_pure_functions(*_PARSE)["resample_animation"]
    pose = blender_pose.build_pose_document(_rotations(1))
    keys = resample([(0, pose, [0, 0, 0]), (1, pose, [10, 0, 0])], 0, 0, 24.0, 60.0)
    assert [f for f, _p, _o in keys] == [0, 1, 2]  # 1/24 s = 2.5 scene frames
    assert [o[0] for _f, _p, o in keys] == pytest.approx([0.0, 4.0, 8.0])  # root travel blended too
    gap = resample([(0, pose, None), (3, pose, None)], 0, 0, 30.0, 30.0)
    assert [f for f, _p, _o in gap] == [0, 3]  # frames 1-2 were lost: no keys there
    later = resample([(0, pose, None), (1, pose, None)], 0, 10, 30.0, 30.0, include_first=False)
    assert [f for f, _p, _o in later] == [11]  # the previous chunk already keyed frame 10


def test_slerp_and_pose_blending():
    loaded = _load_pure_functions(*_PARSE)
    half = loaded["slerp_wxyz"]((1, 0, 0, 0), (0, 0, 0, 1), 0.5)
    assert half == pytest.approx([2 ** -0.5, 0, 0, 2 ** -0.5])
    assert loaded["slerp_wxyz"]((1, 0, 0, 0), (-1, 0, 0, 0), 0.5) == pytest.approx([1, 0, 0, 0])  # same rotation
    a = blender_pose.build_pose_document(_rotations(1))
    b = json.loads(json.dumps(blender_pose.build_pose_document(_rotations(2))))
    b["bones"]["head"]["low_confidence"] = True
    blended = loaded["interpolate_pose_documents"](a, b, 0.3)
    assert blended["bones"]["head"].get("low_confidence") is True
    assert loaded["interpolate_pose_documents"](a, b, 0.0) is a


def test_root_travel_rides_in_the_animation_document():
    frames = [(i, _rotations(i)) for i in range(3)]
    roots = [(0.0, 90.0, 0.0), (5.0, 80.0, -2.0), (10.0, 95.0, -4.0)]
    document = blender_pose.build_animation_document(frames, 30.0, root_positions_cm=roots)
    offsets = [f["root_offset"] for f in document["frames"]]
    assert offsets[0] == pytest.approx([0, 0, 0])
    assert offsets[2] == pytest.approx([10.0, 4.0, 0.0])  # engine +x right, -z forward -> Blender +x, +y; height dropped
    assert document["root_leg_length"] == pytest.approx(sk.BONE_LENGTHS_CM["thigh.L"] + sk.BONE_LENGTHS_CM["shin.L"])
    parsed = _load_pure_functions(*_PARSE)["parse_animation_document"](document)
    assert parsed["root_offsets"][2] == pytest.approx([10.0, 4.0, 0.0])
    assert parsed["root_leg_length"] == document["root_leg_length"]
    bad = json.loads(json.dumps(document))
    bad["frames"][1]["root_offset"] = [1, "x", 0]
    with pytest.raises(ValueError, match="root_offset"):
        _load_pure_functions(*_PARSE)["parse_animation_document"](bad)
    with pytest.raises(ValueError):
        blender_pose.build_animation_document(frames, 30.0, root_positions_cm=roots[:2])
    assert "root_leg_length" not in blender_pose.build_animation_document(frames, 30.0)


def test_root_motion_moves_the_root_on_every_key_when_on():
    scene = _Scene(frame=1, fps=30)
    scene.posecraft_root_motion = True
    addon, log = _addon(scene)
    moves = []

    def apply_root_motion(obj, names, offset, leg_length, state, floor_contact):
        moves.append((tuple(offset), leg_length))
        return True

    addon["_apply_root_motion"] = apply_root_motion
    roots = [(0.0, 90.0, 0.0), (3.0, 90.0, 0.0)]
    document = blender_pose.build_animation_document([(i, _rotations(i)) for i in range(2)], 30.0, root_positions_cm=roots)
    document["live_link_protocol"] = 3
    reply = addon["_apply_live_link_document"](document)
    assert reply["ok"] and [m[0][0] for m in moves] == pytest.approx([0.0, 3.0])
    scene.posecraft_root_motion = False
    moves.clear()
    addon["_apply_live_link_document"](document)
    assert moves == []

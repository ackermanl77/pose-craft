"""Opt-in: runs the add-on in a REAL headless Blender (tests/real_blender_checks.py).

Skipped unless POSECRAFT_BLENDER points at blender.exe (or blender), e.g.

    $env:POSECRAFT_BLENDER = "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe"
    python -m pytest -q tests/test_real_blender.py

Blender runs with --background --factory-startup and its user folders pointed
at the test's temporary folder, so the real Blender preferences and add-ons
are never touched. Takes ~15 s (most of it generating a Rigify rig).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.export import blender_pose, bvh_export
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_hands import _curl, _rest_hand  # noqa: E402
from test_retarget_frames import _articulated_pose, _standing_t_pose  # noqa: E402

BLENDER = os.environ.get("POSECRAFT_BLENDER", "")
pytestmark = pytest.mark.skipif(
    not (BLENDER and Path(BLENDER).is_file()), reason="set POSECRAFT_BLENDER to a blender executable"
)
CHECKS = Path(__file__).with_name("real_blender_checks.py")
# Engine (Y-up) -> Blender (Z-up), as blender_pose._ENGINE_TO_BLENDER_AXES.
E2B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
TOLERANCE_DEG = 0.01


def _write_inputs(work: Path) -> dict:
    landmarks = _articulated_pose()
    body = sk.solve_bone_rotations(landmarks)
    confidence = {name: 0.9 for name in body}
    confidence["shin.R"] = 0.2
    (work / "pose_body.json").write_text(json.dumps(blender_pose.build_pose_document(body, bone_confidence=confidence)))
    face = {"jawOpen": 0.7, "mouthSmileLeft": 0.4, "eyeBlinkRight": 1.0, "cheekPuff": 0.2}
    (work / "pose_face.json").write_text(json.dumps(blender_pose.build_pose_document(body, face_blendshapes=face)))

    hands = {}
    for side in sk.HAND_SIDES:
        hand = _rest_hand(np.zeros(3), side)
        for i, finger in enumerate(sk.FINGERS[1:]):
            hand = _curl(hand, finger, side, 25 + 10 * i)
        hands[side] = hand
    with_hands = sk.solve_bone_rotations(_standing_t_pose(), hands=hands)
    (work / "pose_hands.json").write_text(json.dumps(blender_pose.build_pose_document(with_hands)))

    crouch = _standing_t_pose()
    for side in ("LEFT", "RIGHT"):
        knee = crouch[getattr(sk, f"{side}_HIP")] + [0.0, -0.2, 0.3]
        shift = knee + [0.0, -0.4, -0.05] - crouch[getattr(sk, f"{side}_ANKLE")]
        for joint in ("ANKLE", "HEEL", "FOOT_INDEX"):
            crouch[getattr(sk, f"{side}_{joint}")] += shift
        crouch[getattr(sk, f"{side}_KNEE")] = knee
    (work / "pose_crouch.json").write_text(json.dumps(blender_pose.build_pose_document(sk.solve_bone_rotations(crouch))))

    heads = sk.rest_head_positions()
    rig = {}
    for name in sk.ALL_BONE_NAMES:
        head = E2B @ heads[name] / 100.0
        tail = head + E2B @ sk.bone_rest_direction(name) * sk.bone_length(sk.BONE_LENGTHS_CM, name) / 100.0
        rig[name] = {"head": head.tolist(), "tail": tail.tolist(), "parent": sk.bone_parent(name)}
    (work / "rig_posecraft.json").write_text(json.dumps(rig))

    root = bvh_export.floor_root_position(body)
    bvh_export.save_bvh_file(body, work / "pose.bvh", root_position=root)

    # 45 solved frames with a two-frame tracking gap: more than one Live Link chunk (30 frames).
    indices = list(range(20)) + list(range(22, 47))
    frames = [(i + 5, {n: Rotation.from_rotvec(r.as_rotvec() * i / 46) for n, r in body.items()}) for i in indices]
    (work / "anim.json").write_text(json.dumps(blender_pose.build_animation_document(frames, 30.0)))
    # 45 consecutive 30 fps frames walking 2 cm per frame along the subject's facing direction
    # (engine -z), for a 24 fps scene with root motion on (add-on 0.8.0).
    walk = [(i, body) for i in range(45)]
    roots = [(0.0, 90.0, -2.0 * i) for i in range(45)]
    (work / "anim24.json").write_text(json.dumps(blender_pose.build_animation_document(walk, 30.0, root_positions_cm=roots)))
    return {"root_height_m": root[1] / 100.0, "anim_frames": len(frames)}


def test_the_addon_in_a_real_blender(tmp_path):
    expected = _write_inputs(tmp_path)
    user = tmp_path / "blender-user"
    env = dict(
        os.environ,
        BLENDER_USER_RESOURCES=str(user),
        BLENDER_USER_CONFIG=str(user / "config"),
        BLENDER_USER_SCRIPTS=str(user / "scripts"),
        BLENDER_USER_EXTENSIONS=str(user / "extensions"),
        BLENDER_USER_DATAFILES=str(user / "datafiles"),
    )
    run = subprocess.run(
        [BLENDER, "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(CHECKS), "--", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    assert results["registered"]
    for match in (True, False):
        simple = results[f"simple_match_{match}"]
        assert simple["applied"] == 45 and simple["unmatched"] == []
        assert simple["max_err"] < TOLERANCE_DEG and simple["before"] > 30.0

    euler = results["euler_rig"]
    assert euler["operator"] == [["FINISHED"], ["FINISHED"]] and euler["modes_kept"]
    assert euler["keys"] == 2 * 15 * 3 and euler["key_frames"] == [1.0, 6.0] and euler["frame_after"] == 11
    assert euler["max_err"] < TOLERANCE_DEG
    assert results["floor"]["posed_min"] == pytest.approx(results["floor"]["rest_min"], abs=1e-4)

    # F12: the face sets (and keys) the shape keys named like ARKit's blendshapes, on a mesh parented
    # to the rig and on one bound by an Armature modifier; others are left alone.
    face = results["face"]
    assert face["operator"] == ["FINISHED"]
    assert face["parented"] == {"jawOpen": 0.7, "blendShape1.mouthSmileLeft": 0.4, "eye_blink_right": 1.0, "browInnerUp": 0.25}
    assert face["modifier"] == {"CheekPuff": 0.2}
    assert face["keyed"] == {"parented": 3, "modifier": 1}

    mixamo = results["mixamo"]
    assert mixamo["applied"] == 47 and mixamo["unmatched"] == []  # 3 spine bones for one
    assert mixamo["max_err"] < TOLERANCE_DEG

    if results["rigify_available"]:
        rigify = results["rigify"]
        assert rigify["unmatched"] == [] and rigify["spine_chain"] == ["spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003"]
        assert rigify["ik_fk"] == [1.0, 1.0, 1.0, 1.0]
        assert rigify["max_err"] < TOLERANCE_DEG

    if results["bvh_available"]:
        bvh = results["bvh"]
        assert bvh["bones"] == 15 and bvh["keys"] == 3 + 15 * 3
        assert bvh["root_z"] == pytest.approx(expected["root_height_m"], abs=1e-4)
        assert bvh["lowest_z"] == pytest.approx(0.0, abs=1e-4)
        assert bvh["max_err"] < TOLERANCE_DEG

    live = results["live_link"]
    pose_reply = live["pose_reply"]["reply"]
    assert pose_reply["ok"] and len(pose_reply["applied"]) == 15 and pose_reply["low_confidence"] == ["shin.R"]
    assert live["pose_err"] < TOLERANCE_DEG
    anim_reply = live["anim_reply"]["reply"]
    assert anim_reply["ok"] and anim_reply["chunks"] == 2 and anim_reply["animation_frames"] == expected["anim_frames"]
    assert (anim_reply["start_frame"], anim_reply["end_frame"]) == (1, 47)
    assert live["key_frames"] == [1.0, 47.0, expected["anim_frames"]]  # the gap stays unkeyed
    assert live["keys"] == expected["anim_frames"] * 15 * 4
    assert live["anim_err"] < TOLERANCE_DEG
    assert live["keys_after_one_undo"] == 0  # both chunks are one undo step
    assert live["timer_while_running"]
    resampled = results["resampled"]
    assert resampled["reply"]["ok"] and resampled["reply"]["animation_frames"] == 36
    assert resampled["key_frames"] == [1.0, 36.0, 36]  # 45 frames at 30 fps = 36 scene frames, each keyed once
    # Root motion: clip time at frame 36 = 35/24 s = 43.75 clip frames = 87.5 cm, on a rig of the same
    # proportions (scale 1:100) -> 0.875 m along Blender +Y.
    assert resampled["travel"] == pytest.approx([0.0, 0.875, 0.0], abs=2e-3)
    assert results["unregistered"] == {"classes_gone": True, "port_freed": True, "timer_gone": True}


def test_the_zip_installs_as_an_extension_in_a_real_blender(tmp_path):
    """K6: the release zip validates, installs and enables as a Blender 4.2+ extension."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_addon_zip", Path(__file__).parents[1] / "scripts" / "build_addon_zip.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    archive = builder.build(tmp_path)
    user = tmp_path / "blender-user"
    env = dict(os.environ, BLENDER_USER_RESOURCES=str(user), BLENDER_USER_CONFIG=str(user / "config"),
               BLENDER_USER_SCRIPTS=str(user / "scripts"), BLENDER_USER_EXTENSIONS=str(user / "extensions"),
               BLENDER_USER_DATAFILES=str(user / "datafiles"))
    version = subprocess.run([BLENDER, "--version"], capture_output=True, text=True, timeout=120).stdout
    if not version.startswith("Blender ") or tuple(int(p) for p in version.split()[1].split(".")[:2]) < (4, 2):
        pytest.skip("extensions need Blender 4.2+")

    def blender(*args):
        return subprocess.run([BLENDER, "--factory-startup", *args], env=env, capture_output=True, text=True, timeout=300)

    validate = blender("--command", "extension", "validate", str(archive))
    assert validate.returncode == 0, validate.stdout + validate.stderr
    install = blender("--command", "extension", "install-file", "-r", "user_default", "-e", str(archive))
    assert install.returncode == 0 and 'Installed "posecraft"' in install.stdout, install.stdout + install.stderr
    check = tmp_path / "check.py"
    check.write_text(
        "import bpy, sys\n"
        "ok = 'bl_ext.user_default.posecraft' in bpy.context.preferences.addons and hasattr(bpy.ops.posecraft, 'load_pose')\n"
        "sys.exit(0 if ok else 3)\n",
        encoding="utf-8",
    )
    enabled = subprocess.run([BLENDER, "--background", "--python-exit-code", "1", "--python", str(check)], env=env, capture_output=True, text=True, timeout=300)
    assert enabled.returncode == 0, enabled.stdout[-3000:] + enabled.stderr[-3000:]

"""B12: rotation-space smoothing of the solved bone rotations for animated
export (video.smooth_rotation_frames, the "Rotation smoothing" setting)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.estimation import video
from pose_craft.estimation.landmarker import DetectedPerson

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _standing_t_pose  # noqa: E402


def _angle(a: Rotation, b: Rotation) -> float:
    return float(np.degrees((a * b.inv()).magnitude()))


def _jittery(count=60, seed=0, noise_deg=4.0):
    rng = np.random.default_rng(seed)
    truth = [Rotation.from_euler("y", 30.0, degrees=True) for _ in range(count)]
    noisy = [{"spine": t * Rotation.from_rotvec(np.radians(rng.normal(0, noise_deg, 3)))} for t in truth]
    return truth, noisy


def test_zero_leaves_every_frame_untouched():
    _truth, frames = _jittery()
    out = video.smooth_rotation_frames(frames, 0.0)
    assert all(o["spine"] is f["spine"] for o, f in zip(out, frames))  # the very same rotations
    assert out is not frames and out[0] is not frames[0]  # but fresh containers
    assert video.smooth_rotation_frames([], 0.5) == []


def test_smoothing_reduces_jitter_and_keeps_a_still_pose():
    truth, frames = _jittery()
    raw = np.mean([_angle(f["spine"], t) for f, t in zip(frames, truth)])
    smooth = video.smooth_rotation_frames(frames, 0.6)
    smoothed = np.mean([_angle(f["spine"], t) for f, t in zip(smooth, truth)])
    assert smoothed < raw * 0.6
    still = [{"spine": truth[0]} for _ in range(10)]
    assert all(_angle(f["spine"], truth[0]) < 1e-6 for f in video.smooth_rotation_frames(still, 1.0))


def test_quaternion_sign_flips_dont_change_the_result():
    """q and -q are the same rotation: a solver that flips the sign between
    frames must not make the filter swing through 360 degrees."""
    _truth, frames = _jittery(30)
    flipped = [{"spine": Rotation.from_quat(-f["spine"].as_quat())} if k % 2 else f for k, f in enumerate(frames)]
    a = video.smooth_rotation_frames(frames, 0.7)
    b = video.smooth_rotation_frames(flipped, 0.7)
    assert max(_angle(x["spine"], y["spine"]) for x, y in zip(a, b)) < 1e-6


def test_zero_phase_no_lag_on_a_steady_turn():
    """A turn at constant speed comes out where it is, not behind (a one-way
    exponential filter lags by several frames at this strength)."""
    turn = [{"spine": Rotation.from_euler("y", 2.0 * k, degrees=True)} for k in range(80)]
    out = video.smooth_rotation_frames(turn, 0.9)
    assert max(_angle(o["spine"], t["spine"]) for o, t in zip(out[20:60], turn[20:60])) < 0.5


def test_a_lost_stretch_is_caught_up_on_not_bridged():
    """Frames 0-9 face one way, the tracker loses the person, frames 60-69
    face another: the first frame back is filtered across a 50-frame gap, so
    it stays (nearly) where it was solved -- the frames either side of a gap
    aren't treated as neighbours."""
    a, b = Rotation.from_euler("y", 0, degrees=True), Rotation.from_euler("y", 90, degrees=True)
    frames = [{"spine": a} for _ in range(10)] + [{"spine": b} for _ in range(10)]
    indices = list(range(10)) + list(range(60, 70))
    gap_aware = video.smooth_rotation_frames(frames, 0.8, indices)
    assert _angle(gap_aware[10]["spine"], b) < 1.0 and _angle(gap_aware[9]["spine"], a) < 1.0
    as_neighbours = video.smooth_rotation_frames(frames, 0.8)  # without the indices: smeared
    assert _angle(as_neighbours[10]["spine"], b) > 10.0
    with pytest.raises(ValueError):
        video.smooth_rotation_frames(frames, 0.5, [0, 1])


def test_each_bone_is_filtered_on_its_own_and_missing_bones_are_not_invented():
    a = Rotation.from_euler("x", 40, degrees=True)
    frames = [{"spine": a, "hand.L": a}, {"spine": a}, {"spine": a, "hand.L": Rotation.identity()}]
    out = video.smooth_rotation_frames(frames, 0.5, [0, 1, 2])
    assert [set(f) for f in out] == [{"spine", "hand.L"}, {"spine"}, {"spine", "hand.L"}]
    assert _angle(out[1]["spine"], a) < 1e-6  # untouched by the other bone


def test_slerp_quaternions_matches_scipy():
    rng = np.random.default_rng(5)
    q0, q1 = Rotation.random(20, random_state=rng), Rotation.random(20, random_state=rng)
    out = Rotation.from_quat(video.slerp_quaternions(q0.as_quat(), q1.as_quat(), 0.3))
    for k in range(20):
        expected = q0[k] * Rotation.from_rotvec(0.3 * (q0[k].inv() * q1[k]).as_rotvec())
        assert _angle(out[k], expected) < 1e-6


def test_animation_rotations_is_unchanged_with_smoothing_off():
    people = [DetectedPerson(_standing_t_pose() + 0.01 * k, np.full((33, 2), 0.5), np.ones(33)) for k in range(5)]
    plain, roots = video.animation_rotations(people, fps=30)
    off, roots_off = video.animation_rotations(people, fps=30, rotation_smoothing=0.0)
    assert roots == roots_off
    assert all(_angle(p[n], o[n]) == 0.0 for p, o in zip(plain, off) for n in p)

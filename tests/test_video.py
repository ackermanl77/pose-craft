"""Video input (B12): VIDEO-mode tracking, One Euro smoothing, foot locking and
the multi-frame BVH they feed."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose_craft.estimation import onnx_pose, video
from pose_craft.estimation.landmarker import DetectedPerson
from pose_craft.estimation.onnx_pose import BlazePose, NormRect, PoseResult
from pose_craft.export import bvh_export as bvh
from pose_craft.retarget import skeleton as sk

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _standing_t_pose  # noqa: E402

# --- One Euro filter -------------------------------------------------------------


def test_one_euro_filter_passes_a_constant_and_reduces_jitter():
    rng = np.random.default_rng(0)
    one_euro = video.OneEuroFilter(min_cutoff=1.0, beta=0.0)
    noisy = 1.0 + rng.normal(0, 0.01, 300)
    filtered = np.array([one_euro(np.array([v]), i / 30)[0] for i, v in enumerate(noisy)])
    assert abs(filtered[-100:].mean() - 1.0) < 0.005
    assert filtered[50:].std() < noisy[50:].std() / 3


def test_one_euro_beta_cuts_the_lag_on_fast_motion():
    """A fast ramp lags far behind with beta 0 and much less with beta > 0 --
    the filter's point: smooth when still, responsive when moving."""
    def lag(beta):
        f = video.OneEuroFilter(min_cutoff=1.0, beta=beta)
        out = [f(np.array([2.0 * t / 30]), t / 30)[0] for t in range(60)]
        return 2.0 * 59 / 30 - out[-1]

    assert lag(5.0) < lag(0.0) / 3


def test_one_euro_restarts_on_reset():
    f = video.OneEuroFilter()
    f(np.array([0.0]), 0.0)
    f.reset()
    assert f(np.array([5.0]), 0.1)[0] == 5.0


# --- foot locking ----------------------------------------------------------------


def _feet(left_xz, right_xz, left_y=0.0, right_y=0.0):
    points = {}
    for side, (x, z), y in (("L", left_xz, left_y), ("R", right_xz, right_y)):
        points[f"foot.{side}.head"] = np.array([x, y + 8.0, z])
        points[f"foot.{side}.tail"] = np.array([x, y, z + 18.0])
    return points


def test_a_planted_foot_stays_put_and_moves_the_root():
    """Walking: the planted left foot slides backward relative to the hips by
    1 cm a frame, so the root must move forward 1 cm a frame; the right foot is
    in the air (swinging) and is ignored."""
    frames = [_feet((10.0, -k * 1.0), (-10.0, 20.0 + 3 * k), right_y=10.0) for k in range(5)]
    roots = video.foot_locked_root_xz(frames, fps=30)
    np.testing.assert_allclose([r[1] for r in roots], [0, 1, 2, 3, 4])
    world_foot = [r + (f["foot.L.head"] + f["foot.L.tail"])[[0, 2]] / 2 for r, f in zip(roots, frames)]
    np.testing.assert_allclose(world_foot, [world_foot[0]] * 5, atol=1e-9)


def test_no_foot_locking_while_airborne_or_still():
    jump = [_feet((10.0, -k * 2.0), (-10.0, -k * 2.0), 20.0, 20.0) for k in range(4)]
    assert all(np.allclose(r, 0) for r in video.foot_locked_root_xz(jump, fps=30))
    still = [_feet((10.0, 0.0), (-10.0, 0.0)) for _ in range(4)]
    assert all(np.allclose(r, 0) for r in video.foot_locked_root_xz(still, fps=30))


def test_animation_rotations_stand_every_frame_on_the_floor():
    people = [DetectedPerson(_standing_t_pose(), np.full((33, 2), 0.5), np.ones(33)) for _ in range(3)]
    frames, roots = video.animation_rotations(people, fps=30)
    assert len(frames) == len(roots) == 3
    document = bvh.build_bvh_animation(frames, root_positions=roots)
    assert "Frames: 3" in document
    expected = sk.hip_height_above_floor(frames[0])
    assert all(r[1] == pytest.approx(expected) for r in roots)
    assert all(abs(r[0]) < 1e-9 and abs(r[2]) < 1e-9 for r in roots)


# --- tracking --------------------------------------------------------------------


class _CountingDetector:
    def __init__(self):
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        raw_boxes = np.zeros((1, 2254, 12), dtype=np.float32)
        raw_scores = np.full((1, 2254, 1), -50.0, dtype=np.float32)
        anchors = onnx_pose.ssd_anchor_centers() * 224
        raw_boxes[0, 0, :4] = [112 - anchors[0, 0], 112 - anchors[0, 1], 20, 20]
        raw_boxes[0, 0, 4:8] = [112 - anchors[0, 0], 150 - anchors[0, 1], 112 - anchors[0, 0], 112 - anchors[0, 1]]
        raw_scores[0, 0, 0] = 5.0
        return [raw_boxes, raw_scores]


class _LandmarkModel:
    def __init__(self, presence):
        self.presence = presence

    def run(self, names, feed):
        raw = np.zeros((39, 5), dtype=np.float32)
        raw[:, :2] = 128.0
        raw[33] = [128, 160, 0, 0, 0]  # hip center
        raw[34] = [128, 96, 0, 0, 0]  # body-scale point, above it
        outputs = {
            "Identity": raw.reshape(1, 195),
            "Identity_1": np.array([[self.presence]], dtype=np.float32),
            "Identity_2": np.zeros((1, 256, 256, 1), dtype=np.float32),
            "Identity_3": np.full((1, 64, 64, 39), -20.0, dtype=np.float32),
            "Identity_4": np.zeros((1, 117), dtype=np.float32),
        }
        return [outputs[n] for n in (names or outputs)]


def _tracking_pose(presence=0.9):
    pose = BlazePose.__new__(BlazePose)
    pose._detector = _CountingDetector()
    pose._detector_input = "input"
    pose._landmarks = _LandmarkModel(presence)
    pose._landmarks_input = "input_1"
    pose._landmarks_outputs = ["Identity", "Identity_1", "Identity_2", "Identity_3", "Identity_4"]
    return pose


def test_tracking_runs_the_detector_only_when_there_is_no_previous_frame():
    pose = _tracking_pose()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    first = pose.track(image, None)
    assert first is not None and pose._detector.calls == 1
    second = pose.track(image, first)
    third = pose.track(image, second)
    assert second is not None and third is not None
    assert pose._detector.calls == 1  # frames 2 and 3 reused the previous landmarks' ROI


def test_tracking_falls_back_to_the_detector_when_the_person_is_lost():
    pose = _tracking_pose(presence=0.1)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    lost = PoseResult(np.zeros((33, 2)), np.ones(33), np.zeros((33, 3)), 0.9, aux_xy=np.array([[0.5, 0.6], [0.5, 0.4]]))
    assert pose.track(image, lost) is None
    assert pose._detector.calls >= 1


def test_next_roi_follows_the_auxiliary_landmarks():
    result = PoseResult(np.zeros((33, 2)), np.ones(33), np.zeros((33, 3)), 0.9, aux_xy=np.array([[0.5, 0.6], [0.5, 0.4]]))
    roi = result.next_roi(1000, 1000)
    assert isinstance(roi, NormRect)
    assert (roi.x_center, roi.y_center) == pytest.approx((0.5, 0.6))
    assert roi.rotation == pytest.approx(0.0, abs=1e-9)  # upright: no crop rotation
    assert roi.width == pytest.approx(2 * 0.2 * 1.25)


# --- the whole video path ---------------------------------------------------------


class _FakeTrackingEstimator:
    """Returns a person whose hip landmark jitters around a fixed point."""

    def __init__(self):
        self.rng = np.random.default_rng(1)
        self.previous_seen = []

    def track(self, image_rgb, previous):
        self.previous_seen.append(previous)
        world = np.zeros((33, 3))
        world[:, 0] = 0.3 + self.rng.normal(0, 0.01)
        return PoseResult(np.full((33, 2), 0.5), np.ones(33), world, 0.9, aux_xy=np.array([[0.5, 0.6], [0.5, 0.4]]))


def _write_video(path: Path, frames: int = 12, fps: float = 25.0) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for k in range(frames):
        writer.write(np.full((48, 64, 3), k * 10, dtype=np.uint8))
    writer.release()


def test_solve_video_tracks_every_frame_and_smooths_the_landmarks(tmp_path):
    clip = tmp_path / "clip.avi"
    _write_video(clip)
    estimator = _FakeTrackingEstimator()
    progress = []

    result = video.solve_video(clip, estimator, on_progress=lambda done, total: progress.append(done))

    assert len(result.frames) == 12 and result.fps == pytest.approx(25.0)
    assert result.frame_size == (64, 48)
    assert progress[-1] == 12
    assert estimator.previous_seen[0] is None and all(p is not None for p in estimator.previous_seen[1:])
    xs = np.array([frame.person.landmarks[0, 0] for frame in result.frames])
    assert xs[3:].std() < 0.01  # the raw jitter is sigma 0.01


def test_solve_video_can_be_cancelled_and_rejects_a_missing_file(tmp_path):
    clip = tmp_path / "clip.avi"
    _write_video(clip)
    with pytest.raises(InterruptedError):
        video.solve_video(clip, _FakeTrackingEstimator(), should_cancel=lambda: True)
    with pytest.raises(OSError):
        video.solve_video(tmp_path / "missing.mp4", _FakeTrackingEstimator())


def test_a_walk_moves_the_bvh_root_forward():
    """A stride sequence: the planted foot drives the root forward, the way the
    report's foot locking describes."""
    people = []
    for k in range(8):
        lm = _standing_t_pose()
        # Planted left leg swings back relative to the hips (the body moves forward).
        angle = math.radians(2.0 * k)  # +X rotation swings the foot toward -Z (behind)
        hip = lm[sk.LEFT_HIP].copy()
        turn = Rotation.from_rotvec([angle, 0.0, 0.0])
        for name in ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX"):
            index = getattr(sk, f"LEFT_{name}")
            lm[index] = hip + turn.apply(lm[index] - hip)
        # The right leg is lifted (swinging), so it isn't planted.
        for name in ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX"):
            lm[getattr(sk, f"RIGHT_{name}")] += [0.0, 0.1, 0.1]
        people.append(DetectedPerson(lm, np.full((33, 2), 0.5), np.ones(33)))

    _frames, roots = video.animation_rotations(people, fps=30)
    forward = [r[2] for r in roots]
    assert forward[-1] > forward[0] + 5.0  # cm
    assert all(b >= a - 1e-9 for a, b in zip(forward, forward[1:]))

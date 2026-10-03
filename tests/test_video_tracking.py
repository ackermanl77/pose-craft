"""Multi-person identity tracking in video (B12): stable IDs across frames,
re-identification after a short loss, merged duplicates, per-person
smoothing and per-person animation export."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pose_craft.estimation import video
from pose_craft.estimation.landmarker import DetectedPerson
from pose_craft.estimation.onnx_pose import NormRect, PoseResult

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _standing_t_pose  # noqa: E402

_TEMPLATE = np.column_stack([np.linspace(-0.04, 0.04, 33), np.linspace(-0.15, 0.15, 33)])


def _result(center, world_x=0.0) -> PoseResult:
    center = np.asarray(center, dtype=float)
    world = np.zeros((33, 3))
    world[:, 0] = world_x
    aux = np.array([center + [0.0, 0.05], center + [0.0, -0.05]])
    return PoseResult(center + _TEMPLATE, np.ones(33), world, 0.9, aux_xy=aux)


class _Scene:
    """A fake BlazePose over a scripted scene: frame -> {name: center}. The
    landmark model finds whoever's centre lies inside the crop (nearest its
    middle); the detector finds everyone."""

    def __init__(self, frames: list[dict]):
        self.frames = frames
        self.frame = 0
        self.detector_calls = 0
        self.duplicate = False  # every crop finds the first person (tracks collapse)

    def _people(self):
        return self.frames[self.frame]

    def landmarks_for_roi(self, image, roi: NormRect, min_presence):
        people = self._people()
        if self.duplicate and people:
            name = sorted(people)[0]
            return _result(people[name], world_x=float(ord(name[0])))
        best, best_distance = None, None
        for name, center in people.items():
            distance = np.hypot(center[0] - roi.x_center, center[1] - roi.y_center)
            if distance < roi.width / 2 and (best_distance is None or distance < best_distance):
                best, best_distance = name, distance
        return None if best is None else _result(people[best], world_x=float(ord(best[0])))

    def detect_rois(self, image, max_people, min_detection, tiled=None):
        self.detector_calls += 1
        return [NormRect(c[0], c[1], 0.3, 0.3, 0.0) for c in list(self._people().values())[:max_people]]


def _run(scene: _Scene, max_people=4, **kwargs) -> list[dict]:
    tracker = video.MultiPersonTracker(scene, max_people, **kwargs)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    out = []
    for index in range(len(scene.frames)):
        scene.frame = index
        out.append({tid: chr(int(r.world[0, 0])) for tid, r in tracker.step(image).items()})
    return out, tracker


def test_two_people_keep_their_ids_left_to_right_as_they_move():
    frames = [{"a": (0.2 + 0.01 * k, 0.5), "b": (0.8 - 0.01 * k, 0.5)} for k in range(20)]
    ids, tracker = _run(_Scene(frames))
    assert ids[0] == {0: "a", 1: "b"}
    assert all(frame == {0: "a", 1: "b"} for frame in ids)
    assert tracker.detector_runs == 2  # frame 0, then one periodic look for newcomers (frame 10)


def test_a_person_lost_for_a_few_frames_gets_the_same_id_back():
    frames = []
    for k in range(20):
        people = {"a": (0.25, 0.5)}
        if not 6 <= k < 10:
            people["b"] = (0.75 + 0.002 * k, 0.5)
        frames.append(people)
    ids, _tracker = _run(_Scene(frames))
    assert all(1 not in frame for frame in ids[6:10])
    assert all(frame == {0: "a", 1: "b"} for frame in ids[10:])


def test_a_newcomer_gets_a_new_id_and_a_long_lost_person_too():
    frames = [{"a": (0.2, 0.5)} for _ in range(5)] + [{"a": (0.2, 0.5), "c": (0.7, 0.5)} for _ in range(25)]
    ids, _tracker = _run(_Scene(frames), redetect_every=5)
    assert ids[0] == {0: "a"}
    assert ids[-1] == {0: "a", 1: "c"}
    first_seen = next(i for i, frame in enumerate(ids) if 1 in frame)
    assert 5 <= first_seen <= 10  # found at the next periodic re-detection

    long_gap = [{"b": (0.5, 0.5)}] + [{} for _ in range(40)] + [{"b": (0.5, 0.5)}]
    ids, _tracker = _run(_Scene(long_gap), max_missed=10)
    assert ids[0] == {0: "b"} and ids[-1] == {1: "b"}  # too long gone: a new ID


def test_max_people_limits_the_tracks():
    frames = [{"a": (0.15, 0.5), "b": (0.5, 0.5), "c": (0.85, 0.5)} for _ in range(5)]
    ids, _tracker = _run(_Scene(frames), max_people=2)
    assert all(len(frame) == 2 for frame in ids)


def test_two_tracks_collapsing_onto_one_person_are_merged_into_the_older_id():
    frames = [{"a": (0.3, 0.5), "b": (0.7, 0.5)} for _ in range(6)]
    scene = _Scene(frames)
    tracker = video.MultiPersonTracker(scene, 4)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert set(tracker.step(image)) == {0, 1}
    scene.duplicate = True
    scene.frame = 1
    assert set(tracker.step(image)) == {0}


def test_greedy_assignment_and_match_cost():
    costs = np.array([[0.1, 0.9], [0.2, 0.3]])
    assert video.greedy_assignment(costs, 0.75) == [(0, 0), (1, 1)]
    assert video.greedy_assignment(costs, 0.15) == [(0, 0)]
    assert video.greedy_assignment(np.zeros((0, 3)), 1.0) == []
    near, far = _result((0.5, 0.5)), _result((0.52, 0.5))
    assert video.match_cost(near, near) == 0.0
    assert 0.0 < video.match_cost(near, far) < video.match_cost(near, _result((0.9, 0.5)))


def test_each_person_is_smoothed_separately():
    rng = np.random.default_rng(3)
    frames = []
    for k in range(60):
        people = {}
        for tid, base in ((0, 0.0), (1, 5.0)):
            lm = _standing_t_pose() + base + rng.normal(0, 0.01, (33, 3))
            people[tid] = DetectedPerson(lm, np.full((33, 2), 0.5), np.ones(33))
        if 20 <= k < 25:
            del people[1]  # a gap restarts only that person's filter
        frames.append(people)
    smoothed = video.smooth_tracks(frames, fps=30)
    assert all(1 not in frame for frame in smoothed[20:25])
    for tid, base in ((0, 0.0), (1, 5.0)):
        xs = np.array([f[tid].landmarks[0, 0] for f in smoothed[40:]])
        assert abs(xs.mean() - (_standing_t_pose()[0, 0] + base)) < 0.01  # no bleed between people
        assert xs.std() < 0.01


class _FakeTrackerEstimator:
    def __init__(self, scene):
        self.scene = scene
        self.trackers = []

    def new_tracker(self, max_people):
        tracker = video.MultiPersonTracker(self.scene, max_people)
        self.trackers.append(tracker)
        return tracker

    def track(self, image_rgb, previous):  # pragma: no cover - the single-person path isn't used here
        raise AssertionError("max_people > 1 must use the tracker")


def _write_video(path: Path, frames: int) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    for k in range(frames):
        writer.write(np.full((48, 64, 3), k * 10 % 255, dtype=np.uint8))
    writer.release()


def test_solve_video_tracks_several_people_with_stable_ids(tmp_path):
    clip = tmp_path / "two.avi"
    _write_video(clip, 10)
    scene = _Scene([{"a": (0.3, 0.5), "b": (0.7, 0.5)} for _ in range(10)])

    class _SceneStepper(_Scene):
        pass

    estimator = _FakeTrackerEstimator(scene)
    original_step = video.MultiPersonTracker.step

    def step(self, image):
        result = original_step(self, image)
        scene.frame = min(scene.frame + 1, len(scene.frames) - 1)
        return result

    video.MultiPersonTracker.step = step
    try:
        result = video.solve_video(clip, estimator, max_people=4)
    finally:
        video.MultiPersonTracker.step = original_step
    assert result.track_ids() == [0, 1]
    assert len(result.solved_frames(0)) == len(result.solved_frames(1)) == 10
    assert len(result.people_of(1)) == 10
    assert result.frames[0].person is result.frames[0].people[0]  # the single-person view still works


def test_video_frame_keeps_its_single_person_constructor():
    person = DetectedPerson(np.zeros((33, 3)), np.zeros((33, 2)), np.ones(33))
    assert video.VideoFrame(0, person).people == {0: person}
    assert video.VideoFrame(1, None).people == {}
    frame = video.VideoFrame(2, people={3: person})
    assert frame.person is person
    solve = video.VideoSolve("x", 25.0, (10, 10), [video.VideoFrame(0, person), frame, video.VideoFrame(3, None)])
    assert solve.track_ids() == [0, 3] and len(solve.solved_frames()) == 2
    with pytest.raises(KeyError):
        solve.frames[2].people[0]


# --- appearance re-identification (B12) ---------------------------------------------------------

_COLORS = {"a": (220, 40, 40), "b": (40, 60, 220)}


def _painted(people: dict, size=200) -> np.ndarray:
    """A frame with each person's landmark box filled with their own colour."""
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    for name, (cx, cy) in people.items():
        x0, x1 = int((cx - 0.04) * size), int(np.ceil((cx + 0.04) * size))
        y0, y1 = int((cy - 0.15) * size), int(np.ceil((cy + 0.15) * size))
        image[y0:y1, x0:x1] = _COLORS[name]
    return image


def _run_painted(frames: list[dict]) -> list[dict]:
    scene = _Scene(frames)
    tracker = video.MultiPersonTracker(scene, 4)
    out = []
    for index, people in enumerate(frames):
        scene.frame = index
        out.append({tid: chr(int(r.world[0, 0])) for tid, r in tracker.step(_painted(people)).items()})
    return out


def _crossing_behind_an_occluder() -> list[dict]:
    """Two people walk toward each other behind something (both lost for three
    frames) and come out having swapped sides -- each now nearer the OTHER
    one's last position, which is all keypoint distance can go on."""
    frames = [{"a": (0.30, 0.5), "b": (0.62, 0.5)} for _ in range(5)]
    frames += [{} for _ in range(3)]
    frames += [{"a": (0.50, 0.5), "b": (0.42, 0.5)} for _ in range(3)]
    return frames


def test_appearance_keeps_identities_through_a_crossing_occlusion():
    ids = _run_painted(_crossing_behind_an_occluder())
    assert ids[0] == {0: "a", 1: "b"}
    assert all(frame == {} for frame in ids[5:8])
    assert all(frame == {0: "a", 1: "b"} for frame in ids[8:])


def test_without_appearance_the_same_crossing_swaps_ids(monkeypatch):
    """The regression the appearance term exists for: geometry alone swaps."""
    monkeypatch.setattr(video, "APPEARANCE_WEIGHT", 0.0)
    ids = _run_painted(_crossing_behind_an_occluder())
    assert ids[-1] == {0: "b", 1: "a"}


def test_appearance_signature_and_cost():
    image = _painted({"a": (0.3, 0.5), "b": (0.7, 0.5)})
    xy_a = _result((0.3, 0.5)).image_xy
    xy_b = _result((0.7, 0.5)).image_xy
    a, b = video.appearance_signature(image, xy_a), video.appearance_signature(image, xy_b)
    assert a.shape == (video.APPEARANCE_HISTOGRAM_BINS**3,) and a.sum() == pytest.approx(1.0)
    assert video.appearance_cost(a, a) == 0.0
    assert video.appearance_cost(a, b) > 0.8  # different clothes: nearly maximal
    assert video.appearance_cost(None, b) == 0.0  # no signature is neutral, not evidence
    assert video.appearance_signature(image, np.full((33, 2), 1.5)) is None  # box outside the frame
    assert video.appearance_signature(np.zeros((10, 10)), xy_a) is None  # not an RGB image
    # A big box is sampled, not histogrammed pixel by pixel -- same answer.
    big = np.random.default_rng(1).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    box = np.array([[0.05, 0.05], [0.95, 0.95]])
    sampled = video.appearance_signature(big, box)
    full, _ = np.histogramdd(big[54:1026, 96:1824].reshape(-1, 3), bins=(4, 4, 4), range=[(0, 256)] * 3)
    assert video.appearance_cost(sampled, full.ravel() / full.sum()) < 0.1  # pure noise: the hardest case


def test_a_tracks_appearance_is_a_running_blend():
    old, new = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    blended = video.blend_appearance(old, new)
    assert blended == pytest.approx([video.APPEARANCE_MEMORY, 1 - video.APPEARANCE_MEMORY])
    assert video.blend_appearance(None, new) is new and video.blend_appearance(old, None) is old

"""Video -> an animated pose (B12): tracked people, solved frame by frame.

- Tracking (BlazePose.track): the detector runs on the first frame and
  whenever the person is lost; every other frame crops where the previous
  frame's landmarks say the person is -- one landmark inference per frame,
  as in MediaPipe's VIDEO mode.
- Several people (MultiPersonTracker, when max_people > 1): each person keeps
  a stable track ID. Every tracked person is followed the same way; the
  detector runs on the first frame, whenever someone is lost or nobody is
  tracked, and every
  REDETECT_EVERY_FRAMES frames while fewer than max_people are tracked. A new
  detection that overlaps a tracked person is ignored; the rest are matched
  to recently lost tracks by keypoint distance plus an appearance signature
  (a compact RGB histogram over the person's landmark box, reducing ID swaps
  after crossings/occlusion), and only then become new IDs. Two tracks that
  converge on one person are merged.
- Smoothing: a One Euro filter (Casiez et al., CHI 2012) on the world
  landmarks, per coordinate and per person -- strong smoothing when the body
  is still, little lag when it moves fast. `min_cutoff` (Hz) lowers jitter,
  `beta` lowers lag. Animated export can additionally smooth the solved bone
  rotations themselves (smooth_rotation_frames: a zero-phase exponential
  slerp per bone, gap-aware, the "Rotation smoothing" setting).
- Ground contact and foot locking: each frame's hips are lifted to stand the
  pose on the floor (skeleton.hip_height_above_floor), and while a foot rests
  on the floor the root moves so that foot stays put (foot_locked_root_xz) --
  which is also what gives a walk its forward motion, since the world
  landmarks themselves are hip-centred.

Pure numpy/OpenCV apart from the estimator itself; the GUI runs solve_video on
a worker thread and exports the result with bvh_export.build_bvh_animation.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.estimation.landmarker import (
    DetectedPerson,
    PoseEstimator,
    _world_to_engine,
    prepare_input_rgb,
)
from pose_craft.retarget.skeleton import (
    BONE_LENGTHS_CM,
    floor_height,
    forward_kinematics,
    solve_bone_rotations,
)

# One Euro defaults for world landmarks in meters at video frame rates: a
# 1.5 Hz cutoff hides the model's frame-to-frame jitter (a few mm) on a still
# body; beta 0.5 opens the filter up quickly for fast limbs.
DEFAULT_MIN_CUTOFF = 1.5
DEFAULT_BETA = 0.5
DEFAULT_D_CUTOFF = 1.0

# A foot counts as planted when its lowest point is within FOOT_LOCK_HEIGHT_CM
# of the frame's floor in both frames and it rose or fell slower than
# FOOT_LOCK_MAX_VERTICAL_CM_S. (Relative to the hips, a planted foot moves
# horizontally at the walking speed, so horizontal speed can't tell; a jump of
# more than FOOT_LOCK_MAX_SLIDE_CM_S between two frames is treated as an
# estimation glitch, not a step.) BVH units (cm).
FOOT_LOCK_HEIGHT_CM = 3.0
FOOT_LOCK_MAX_VERTICAL_CM_S = 30.0
FOOT_LOCK_MAX_SLIDE_CM_S = 400.0
APPEARANCE_HISTOGRAM_BINS = 4
APPEARANCE_WEIGHT = 0.35
# A track's signature keeps this much of its history each frame (blend_appearance).
APPEARANCE_MEMORY = 0.8
APPEARANCE_MAX_SAMPLES = 4096


def _smoothing_factor(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """The One Euro filter over an array of any shape (every element filtered
    independently). Call with each new sample and its timestamp (seconds)."""

    def __init__(self, min_cutoff: float = DEFAULT_MIN_CUTOFF, beta: float = DEFAULT_BETA, d_cutoff: float = DEFAULT_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _smoothing_factor(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()


_FOOT_POINTS = {"L": ("foot.L.head", "foot.L.tail"), "R": ("foot.R.head", "foot.R.tail")}


def foot_locked_root_xz(
    frames_points: list[dict[str, np.ndarray]],
    fps: float,
    lock_height: float = FOOT_LOCK_HEIGHT_CM,
    max_vertical_speed: float = FOOT_LOCK_MAX_VERTICAL_CM_S,
    max_slide_speed: float = FOOT_LOCK_MAX_SLIDE_CM_S,
) -> list[np.ndarray]:
    """Horizontal (X, Z) root positions for a sequence of hip-rooted,
    floor-relative skeletons (forward_kinematics points, Y = 0 at the floor):
    while a foot stays planted -- on the floor, not rising or falling --
    across two frames, the root moves by that foot's opposite displacement,
    so the foot stays fixed in the world (the lower foot wins if both are).
    Otherwise the root keeps its position. The first frame's root is at the
    origin."""
    roots = [np.zeros(2)]
    for previous, current in zip(frames_points, frames_points[1:]):
        step = None
        best_height = None
        for head, tail in _FOOT_POINTS.values():
            lowest_prev = min(previous[head][1], previous[tail][1])
            lowest_cur = min(current[head][1], current[tail][1])
            if max(lowest_prev, lowest_cur) > lock_height:
                continue
            if abs(lowest_cur - lowest_prev) * fps > max_vertical_speed:
                continue
            prev_xz = (previous[head] + previous[tail])[[0, 2]] / 2
            cur_xz = (current[head] + current[tail])[[0, 2]] / 2
            if np.linalg.norm(cur_xz - prev_xz) * fps > max_slide_speed:
                continue
            if best_height is None or lowest_cur < best_height:
                best_height, step = lowest_cur, prev_xz - cur_xz
        roots.append(roots[-1] + (step if step is not None else 0.0))
    return roots


# -- multi-person identity tracking (B12) ----------------------------------------

# While fewer than max_people are tracked, the detector looks for newcomers
# this often (and always when someone is lost).
REDETECT_EVERY_FRAMES = 10
# A lost person can be re-identified this many frames later; after that a
# detection of them gets a new ID.
MAX_MISSED_FRAMES = 30
# Matching a detection to a lost track: the mean distance between their
# corresponding landmarks, over the lost person's box diagonal, must be under this.
MATCH_MAX_COST = 0.75
# Two people whose landmark boxes overlap this much are one person.
DUPLICATE_IOU = 0.6
_BOX_MIN_VISIBILITY = 0.3


def landmark_box(image_xy: np.ndarray, visibility: np.ndarray | None = None) -> np.ndarray:
    """(xmin, ymin, xmax, ymax) of a person's visible landmarks (all of them
    when fewer than four are visible)."""
    points = np.asarray(image_xy, dtype=float)
    if visibility is not None:
        visible = points[np.asarray(visibility) >= _BOX_MIN_VISIBILITY]
        if len(visible) >= 4:
            points = visible
    return np.concatenate([points.min(axis=0), points.max(axis=0)])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match_cost(previous, current) -> float:
    """How far `current` (a PoseResult) is from a track's last `previous`
    one: the mean image distance between landmarks visible in both, over
    the previous landmark box's diagonal."""
    both = (np.asarray(previous.visibility) >= _BOX_MIN_VISIBILITY) & (
        np.asarray(current.visibility) >= _BOX_MIN_VISIBILITY
    )
    if both.sum() < 4:
        both = np.ones(len(previous.image_xy), dtype=bool)
    box = landmark_box(previous.image_xy, previous.visibility)
    diagonal = max(float(np.hypot(box[2] - box[0], box[3] - box[1])), 1e-6)
    distances = np.linalg.norm(np.asarray(previous.image_xy)[both] - np.asarray(current.image_xy)[both], axis=1)
    return float(distances.mean()) / diagonal


def appearance_signature(
    image_rgb: np.ndarray,
    image_xy: np.ndarray,
    visibility: np.ndarray | None = None,
    bins: int = APPEARANCE_HISTOGRAM_BINS,
) -> np.ndarray | None:
    """A compact RGB histogram over the landmark box, normalized to sum to 1.

    Small and cheap on purpose: the tracker only needs enough appearance
    signal to tell two nearby people apart after crossings/brief occlusions.
    Returns None when the landmark box is empty or outside the frame.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    box = landmark_box(image_xy, visibility)
    h, w = image.shape[:2]
    x0 = int(np.floor(np.clip(box[0], 0.0, 1.0) * w))
    x1 = int(np.ceil(np.clip(box[2], 0.0, 1.0) * w))
    y0 = int(np.floor(np.clip(box[1], 0.0, 1.0) * h))
    y1 = int(np.ceil(np.clip(box[3], 0.0, 1.0) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    # Every pixel isn't needed for a 64-bin histogram: a strided sample of at
    # most ~APPEARANCE_MAX_SAMPLES keeps this cheap on a full-HD person box.
    stride = max(1, int(np.sqrt((x1 - x0) * (y1 - y0) / APPEARANCE_MAX_SAMPLES)))
    crop = image[y0:y1:stride, x0:x1:stride, :3]
    if crop.size == 0:
        return None
    levels = crop.reshape(-1, 3).astype(np.int64) * bins // 256  # 0..bins-1 per channel
    flat = (levels[:, 0] * bins + levels[:, 1]) * bins + levels[:, 2]
    signature = np.bincount(flat, minlength=bins**3).astype(np.float64)
    total = float(signature.sum())
    return signature / total if total > 0 else None


def blend_appearance(previous: np.ndarray | None, current: np.ndarray | None) -> np.ndarray | None:
    """A track's running appearance: mostly what it looked like so far, a
    little of this frame (APPEARANCE_MEMORY), so a frame where someone walks
    in front of the person doesn't overwrite their signature."""
    if previous is None or current is None:
        return current if previous is None else previous
    return APPEARANCE_MEMORY * previous + (1.0 - APPEARANCE_MEMORY) * current


def appearance_cost(previous: np.ndarray | None, current: np.ndarray | None) -> float:
    """0 for identical appearance, 1 for maximally different.

    Missing signatures are treated as neutral rather than as evidence against a
    match, so geometry-only tracking keeps working on tiny or out-of-frame
    crops.
    """
    if previous is None or current is None:
        return 0.0
    return float(np.abs(previous - current).sum() / 2.0)


def track_match_cost(previous, current, previous_appearance: np.ndarray | None, current_appearance: np.ndarray | None) -> float:
    """Geometry-first identity cost, nudged by appearance to break crossings."""
    return match_cost(previous, current) + APPEARANCE_WEIGHT * appearance_cost(previous_appearance, current_appearance)


def greedy_assignment(costs: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """(row, column) pairs, cheapest first, each row and column used once,
    every pair under max_cost."""
    pairs: list[tuple[int, int]] = []
    if costs.size == 0:
        return pairs
    used_rows, used_columns = set(), set()
    for flat in np.argsort(costs, axis=None, kind="stable"):
        row, column = divmod(int(flat), costs.shape[1])
        if costs[row, column] >= max_cost:
            break
        if row in used_rows or column in used_columns:
            continue
        pairs.append((row, column))
        used_rows.add(row)
        used_columns.add(column)
    return pairs


@dataclasses.dataclass
class _Track:
    id: int
    last: object  # PoseResult
    appearance: np.ndarray | None = None
    missed: int = 0  # frames since last seen


class MultiPersonTracker:
    """Stable person IDs across video frames (see the module docstring).
    `pose` is an onnx_pose.BlazePose (anything with detect_rois and
    landmarks_for_roi). step() takes one prepared RGB frame and returns
    {track id: PoseResult} for the people found in it. IDs start at 0, given
    left to right among the people first seen together."""

    def __init__(
        self,
        pose,
        max_people: int,
        redetect_every: int = REDETECT_EVERY_FRAMES,
        max_missed: int = MAX_MISSED_FRAMES,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        self._pose = pose
        self.max_people = max(1, int(max_people))
        self.redetect_every = max(1, int(redetect_every))
        self.max_missed = max_missed
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self._tracks: list[_Track] = []
        self._next_id = 0
        self._frame = 0
        self.detector_runs = 0

    @staticmethod
    def _box(result) -> np.ndarray:
        return landmark_box(result.image_xy, result.visibility)

    def _overlaps(self, result, others) -> bool:
        box = self._box(result)
        return any(box_iou(box, self._box(other)) > DUPLICATE_IOU for other in others)

    def step(self, image_rgb: np.ndarray) -> dict:
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        current: dict = {}
        current_appearance: dict[int, np.ndarray | None] = {}
        lost_now = False
        # 1. Follow everyone seen in the previous frame from their own landmarks.
        for track in self._tracks:
            if track.missed:
                continue
            roi = track.last.next_roi(w, h)
            result = None if roi is None else self._pose.landmarks_for_roi(image_rgb, roi, self.min_presence_confidence)
            if result is None:
                lost_now = True
            else:
                current[track.id] = result
                current_appearance[track.id] = appearance_signature(image_rgb, result.image_xy, result.visibility)
        # 2. Two tracks that drifted onto one person: the older ID keeps them.
        for track_id in sorted(current):
            if self._overlaps(current[track_id], [current[o] for o in current if o < track_id]):
                del current[track_id]
                current_appearance.pop(track_id, None)
                lost_now = True
        # 3. Look for newcomers and lost people.
        if self._frame == 0 or not current or (
            len(current) < self.max_people and (lost_now or self._frame % self.redetect_every == 0)
        ):
            self.detector_runs += 1
            candidates = []
            candidate_appearance = []
            for roi in self._pose.detect_rois(image_rgb, self.max_people, self.min_detection_confidence):
                result = self._pose.landmarks_for_roi(image_rgb, roi, self.min_presence_confidence)
                if result is None or self._overlaps(result, list(current.values()) + candidates):
                    continue
                candidates.append(result)
                candidate_appearance.append(appearance_signature(image_rgb, result.image_xy, result.visibility))
            lost = [t for t in self._tracks if t.id not in current and t.missed < self.max_missed]
            costs = np.array(
                [
                    [track_match_cost(t.last, c, t.appearance, candidate_appearance[i]) for i, c in enumerate(candidates)]
                    for t in lost
                ],
                dtype=float,
            )
            costs = costs.reshape(len(lost), len(candidates))
            used = set()
            for row, column in greedy_assignment(costs, MATCH_MAX_COST):
                if len(current) >= self.max_people:
                    break
                current[lost[row].id] = candidates[column]
                current_appearance[lost[row].id] = candidate_appearance[column]
                used.add(column)
            newcomers = sorted(
                ((i, c) for i, c in enumerate(candidates) if i not in used),
                key=lambda item: float(np.mean(item[1].image_xy[:, 0])),
            )
            for index, result in newcomers:
                if len(current) >= self.max_people:
                    break
                self._tracks.append(_Track(self._next_id, result, candidate_appearance[index]))
                current[self._next_id] = result
                current_appearance[self._next_id] = candidate_appearance[index]
                self._next_id += 1
        # 4. Bookkeeping: seen tracks reset, the rest age and eventually expire.
        for track in self._tracks:
            if track.id in current:
                track.last, track.missed = current[track.id], 0
                track.appearance = blend_appearance(track.appearance, current_appearance.get(track.id))
            else:
                track.missed += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        self._frame += 1
        return current


@dataclasses.dataclass
class VideoFrame:
    index: int
    # The lowest-numbered person tracked in this frame (smoothed landmarks);
    # None when nobody was found. Kept for single-person callers.
    person: DetectedPerson | None = None
    # Everyone tracked in this frame: track ID -> smoothed landmarks.
    people: dict | None = None

    def __post_init__(self):
        if self.people is None:
            self.people = {} if self.person is None else {0: self.person}
        elif self.person is None and self.people:
            self.person = self.people[min(self.people)]


@dataclasses.dataclass
class VideoSolve:
    path: str
    fps: float
    frame_size: tuple[int, int]  # (width, height)
    frames: list[VideoFrame]

    def solved_frames(self, track_id: int | None = None) -> list[VideoFrame]:
        """Frames with anybody in them, or with person `track_id`."""
        if track_id is None:
            return [frame for frame in self.frames if frame.people]
        return [frame for frame in self.frames if track_id in frame.people]

    def track_ids(self) -> list[int]:
        """Every person tracked anywhere in the video, by ID."""
        return sorted({track_id for frame in self.frames for track_id in frame.people})

    def people_of(self, track_id: int) -> list[DetectedPerson]:
        """One person's landmarks in every frame they were found in."""
        return [frame.people[track_id] for frame in self.frames if track_id in frame.people]


# A hand the model missed for a moment (motion blur, a turned palm) keeps its last shape this long,
# instead of the fingers snapping straight for a frame or two (B21).
HAND_HOLD_SECONDS = 0.25


class _HandSmoother:
    """One person's hands through a video (B21): each hand's shape relative to its own wrist is
    One-Euro smoothed and put back on the smoothed body's wrist, so the hand can't drift away from
    the arm; a hand missing for up to HAND_HOLD_SECONDS keeps its last shape."""

    def __init__(self, min_cutoff: float, beta: float, fps: float):
        from pose_craft.estimation.landmarker import HandLandmarks

        self._make = HandLandmarks
        self._filters = {side: (OneEuroFilter(min_cutoff, beta), OneEuroFilter(min_cutoff, beta)) for side in ("L", "R")}
        self._last: dict = {}
        self._missing = {"L": 0, "R": 0}
        self._hold_frames = max(1, int(round(HAND_HOLD_SECONDS * fps)))

    def reset(self) -> None:
        for shape, image in self._filters.values():
            shape.reset()
            image.reset()
        self._last.clear()

    def __call__(self, person: DetectedPerson, smoothed_landmarks: np.ndarray, t: float) -> dict:
        from pose_craft.estimation.landmarker import _HAND_POSE_LANDMARKS

        hands = {}
        for side, (wrist, _index, _pinky) in _HAND_POSE_LANDMARKS.items():
            hand = person.hands.get(side)
            shape_filter, image_filter = self._filters[side]
            if hand is None:
                self._missing[side] += 1
                last = self._last.get(side)
                if last is not None and self._missing[side] <= self._hold_frames:
                    shape, image_xy, presence = last
                    hands[side] = self._make(world=shape + smoothed_landmarks[wrist], image_xy=image_xy, presence=presence)
                else:
                    shape_filter.reset()
                    image_filter.reset()
                    self._last.pop(side, None)
                continue
            self._missing[side] = 0
            shape = shape_filter(np.asarray(hand.world) - np.asarray(hand.world)[0], t)
            image_xy = image_filter(np.asarray(hand.image_xy), t)
            self._last[side] = (shape, image_xy, hand.presence)
            hands[side] = self._make(world=shape + smoothed_landmarks[wrist], image_xy=image_xy, presence=hand.presence)
        return hands


def smooth_people(
    people: list[DetectedPerson | None], fps: float, min_cutoff: float = DEFAULT_MIN_CUTOFF, beta: float = DEFAULT_BETA
) -> list[DetectedPerson | None]:
    """One Euro smoothing of each frame's engine-frame landmarks (the 2D image
    landmarks too, so the overlay matches), and of the hands when they were
    solved (_HandSmoother). A gap (None) restarts the filters."""
    world_filter = OneEuroFilter(min_cutoff, beta)
    image_filter = OneEuroFilter(min_cutoff, beta)
    hand_smoother = _HandSmoother(min_cutoff, beta, fps)
    smoothed: list[DetectedPerson | None] = []
    for index, person in enumerate(people):
        if person is None:
            world_filter.reset()
            image_filter.reset()
            hand_smoother.reset()
            smoothed.append(None)
            continue
        t = index / fps
        landmarks = world_filter(person.landmarks, t)
        smoothed.append(
            DetectedPerson(
                landmarks=landmarks,
                image_landmarks_xy=image_filter(person.image_landmarks_xy, t),
                visibility=person.visibility,
                landmark_presence=person.landmark_presence,
                hands=hand_smoother(person, landmarks, t),
            )
        )
    return smoothed


def smooth_tracks(
    frames_people: list[dict], fps: float, min_cutoff: float = DEFAULT_MIN_CUTOFF, beta: float = DEFAULT_BETA
) -> list[dict]:
    """smooth_people for every person separately: a list of {track id:
    DetectedPerson} per frame in, the same smoothed out (each person has their
    own filter, restarted wherever that person is missing)."""
    smoothed: list[dict] = [{} for _ in frames_people]
    for track_id in sorted({tid for people in frames_people for tid in people}):
        series = smooth_people([people.get(track_id) for people in frames_people], fps, min_cutoff, beta)
        for index, person in enumerate(series):
            if person is not None:
                smoothed[index][track_id] = person
    return smoothed


def slerp_quaternions(q0: np.ndarray, q1: np.ndarray, alpha) -> np.ndarray:
    """Row-wise spherical interpolation between unit quaternions (scipy's
    x, y, z, w order) by `alpha` (a scalar or one value per row), always
    along the shorter arc: q and -q are the same rotation, so a sign flip
    between frames is not a 360-degree turn."""
    q0 = np.asarray(q0, dtype=float).reshape(-1, 4)
    q1 = np.asarray(q1, dtype=float).reshape(-1, 4)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (len(q0),))
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0.0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6  # (nearly) the same rotation: plain lerp is exact enough
    safe = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
    w1 = np.where(near, alpha, np.sin(alpha * theta) / safe)
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _exponential_rotation_pass(
    frames_rotations: list[dict[str, Rotation]], frame_indices: list[int], keep: float
) -> list[dict[str, Rotation]]:
    """One causal pass: each bone moves `1 - keep` of the way from its
    filtered value toward each new frame (more after a gap: `1 - keep**gap`,
    as if the skipped frames had been filtered too). A bone seen for the
    first time starts the filter at its own value."""
    previous_quat: dict[str, np.ndarray] = {}
    previous_index: dict[str, int] = {}
    out: list[dict[str, Rotation]] = []
    for frame, index in zip(frames_rotations, frame_indices):
        result: dict[str, Rotation] = {}
        names = [name for name in frame if name in previous_quat]
        if names:
            current = np.array([frame[name].as_quat() for name in names])
            before = np.array([previous_quat[name] for name in names])
            gaps = np.array([max(1, index - previous_index[name]) for name in names], dtype=float)
            filtered = Rotation.from_quat(slerp_quaternions(before, current, 1.0 - keep**gaps))
            for k, name in enumerate(names):
                result[name] = filtered[k]
        for name, rotation in frame.items():
            if name not in result:
                result[name] = rotation
            previous_quat[name] = result[name].as_quat()
            previous_index[name] = index
        out.append({name: result[name] for name in frame})
    return out


def smooth_rotation_frames(
    frames_rotations: list[dict[str, Rotation]],
    strength: float = 0.0,
    frame_indices: list[int] | None = None,
) -> list[dict[str, Rotation]]:
    """Quaternion-aware smoothing over one person's solved bone rotations.

    `strength` is 0..1: 0 leaves the animation untouched (the same Rotation
    objects come back), 1 is the strongest practical smoothing. Each bone is
    filtered on its own with an exponential slerp, run forward and then
    backward over the clip (zero-phase), so the smoothed motion doesn't lag
    behind the video. frame_indices: each entry's frame number in the clip
    (default 0, 1, 2, ...): across frames the tracker lost, the filter
    catches up by the size of the gap instead of treating the frames either
    side as neighbours (a bone missing from some frames is treated the same
    way, and never invented there). Call it per person (track): tracks never
    share a filter.
    """
    if not frames_rotations or strength <= 0:
        return [dict(frame) for frame in frames_rotations]
    indices = list(range(len(frames_rotations))) if frame_indices is None else [int(i) for i in frame_indices]
    if len(indices) != len(frames_rotations):
        raise ValueError("frame_indices must have one entry per frame")
    strength = float(np.clip(strength, 0.0, 1.0))
    keep = 0.9 * strength  # per frame: how much of the filtered value stays
    forward = _exponential_rotation_pass(frames_rotations, indices, keep)
    backward = _exponential_rotation_pass(forward[::-1], [-i for i in indices[::-1]], keep)
    return backward[::-1]


def solve_video(
    path: str | Path,
    estimator: PoseEstimator,
    *,
    max_people: int = 1,
    min_cutoff: float = DEFAULT_MIN_CUTOFF,
    beta: float = DEFAULT_BETA,
    max_frames: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    hands: bool = False,
) -> VideoSolve:
    """Decodes `path` with OpenCV and tracks up to `max_people` people through
    it -- one (the best detection) with the estimator's own track(), several
    with a MultiPersonTracker (estimator.new_tracker), each with a stable ID
    -- smoothing each person's landmarks. `hands` (B21): also each tracked
    person's fingers, frame by frame (the estimator's hand model must be
    loaded: PoseEstimator.enable_hands). Raises OSError if the file can't be
    opened, InterruptedError when should_cancel() turns true."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw: list[dict] = []
        previous = None
        tracker = estimator.new_tracker(max_people) if max_people > 1 else None
        while max_frames is None or len(raw) < max_frames:
            if should_cancel is not None and should_cancel():
                raise InterruptedError("video solve cancelled")
            ok, frame = capture.read()
            if not ok:
                break
            image_rgb = prepare_input_rgb(frame)
            if tracker is None:
                previous = estimator.track(image_rgb, previous)
                results = {} if previous is None else {0: previous}
            else:
                results = tracker.step(image_rgb)
            frame_people = {
                track_id: DetectedPerson(
                    landmarks=_world_to_engine(result.world),
                    image_landmarks_xy=result.image_xy,
                    visibility=result.visibility,
                    landmark_presence=result.landmark_presence,
                )
                for track_id, result in results.items()
            }
            if hands:
                for track_id, result in results.items():
                    estimator.find_hands(image_rgb, frame_people[track_id], result)
            raw.append(frame_people)
            if on_progress is not None:
                on_progress(len(raw), total)
    finally:
        capture.release()
    people = smooth_tracks(raw, fps, min_cutoff, beta)
    return VideoSolve(str(path), fps, (width, height), [VideoFrame(i, people=p) for i, p in enumerate(people)])


def animation_root_positions(
    frames_rotations: list[dict[str, Rotation]], fps: float
) -> list[tuple[float, float, float]]:
    """BVH root positions (cm, this exporter's rest skeleton) for a sequence of
    solved poses: each frame stands on the floor (its hips at their height
    above the lowest joint), and planted feet drive the root's horizontal
    motion (foot_locked_root_xz)."""
    points = []
    for rotations in frames_rotations:
        posed = forward_kinematics(rotations, BONE_LENGTHS_CM)
        floor = np.array([0.0, floor_height(posed), 0.0])
        points.append({name: point - floor for name, point in posed.items()})
    roots_xz = foot_locked_root_xz(points, fps)
    # The points are floor-relative, so the hip center's height is the root's Y.
    return [(float(xz[0]), float(p["spine.head"][1]), float(xz[1])) for xz, p in zip(roots_xz, points)]


def animation_rotations(
    people: list[DetectedPerson],
    fps: float,
    mirror: Callable[[np.ndarray], np.ndarray] | None = None,
    orientation_correction: Rotation | None = None,
    rotation_smoothing: float = 0.0,
) -> tuple[list[dict[str, Rotation]], list[tuple[float, float, float]]]:
    """(local rotations per frame, BVH root position per frame in cm) for
    solved frames -- see animation_root_positions."""
    from pose_craft.retarget.skeleton import apply_orientation_correction

    frames = []
    for person in people:
        landmarks, visibility = person.landmarks, person.visibility
        if mirror is not None:
            landmarks, visibility = mirror(landmarks), mirror(visibility)
        rotations = solve_bone_rotations(landmarks, visibility=visibility)
        if orientation_correction is not None:
            rotations = apply_orientation_correction(rotations, orientation_correction)
        frames.append(rotations)
    frames = smooth_rotation_frames(frames, rotation_smoothing)
    return frames, animation_root_positions(frames, fps)

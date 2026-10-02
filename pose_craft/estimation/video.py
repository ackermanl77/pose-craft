"""Video -> an animated pose (B12): one tracked person, solved frame by frame.

- Tracking (BlazePose.track): the detector runs on the first frame and
  whenever the person is lost; every other frame crops where the previous
  frame's landmarks say the person is -- one landmark inference per frame,
  as in MediaPipe's VIDEO mode.
- Smoothing: a One Euro filter (Casiez et al., CHI 2012) on the world
  landmarks, per coordinate -- strong smoothing when the body is still, little
  lag when it moves fast. `min_cutoff` (Hz) lowers jitter, `beta` lowers lag.
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
from pathlib import Path
from typing import Callable

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
        for side, (head, tail) in _FOOT_POINTS.items():
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


@dataclasses.dataclass
class VideoFrame:
    index: int
    person: DetectedPerson | None  # smoothed landmarks; None when nobody was found


@dataclasses.dataclass
class VideoSolve:
    path: str
    fps: float
    frame_size: tuple[int, int]  # (width, height)
    frames: list[VideoFrame]

    def solved_frames(self) -> list[VideoFrame]:
        return [frame for frame in self.frames if frame.person is not None]


def smooth_people(
    people: list[DetectedPerson | None], fps: float, min_cutoff: float = DEFAULT_MIN_CUTOFF, beta: float = DEFAULT_BETA
) -> list[DetectedPerson | None]:
    """One Euro smoothing of each frame's engine-frame landmarks (the 2D image
    landmarks too, so the overlay matches). A gap (None) restarts the filter."""
    world_filter = OneEuroFilter(min_cutoff, beta)
    image_filter = OneEuroFilter(min_cutoff, beta)
    smoothed: list[DetectedPerson | None] = []
    for index, person in enumerate(people):
        if person is None:
            world_filter.reset()
            image_filter.reset()
            smoothed.append(None)
            continue
        t = index / fps
        smoothed.append(
            DetectedPerson(
                landmarks=world_filter(person.landmarks, t),
                image_landmarks_xy=image_filter(person.image_landmarks_xy, t),
                visibility=person.visibility,
            )
        )
    return smoothed


def solve_video(
    path: str | Path,
    estimator: PoseEstimator,
    *,
    min_cutoff: float = DEFAULT_MIN_CUTOFF,
    beta: float = DEFAULT_BETA,
    max_frames: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoSolve:
    """Decodes `path` with OpenCV and tracks one person through it (the
    leftmost one found first), smoothing their landmarks. Raises OSError if the
    file can't be opened, InterruptedError when should_cancel() turns true."""
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
        raw: list[DetectedPerson | None] = []
        previous = None
        while max_frames is None or len(raw) < max_frames:
            if should_cancel is not None and should_cancel():
                raise InterruptedError("video solve cancelled")
            ok, frame = capture.read()
            if not ok:
                break
            result = estimator.track(prepare_input_rgb(frame), previous)
            previous = result
            raw.append(
                None
                if result is None
                else DetectedPerson(
                    landmarks=_world_to_engine(result.world),
                    image_landmarks_xy=result.image_xy,
                    visibility=result.visibility,
                )
            )
            if on_progress is not None:
                on_progress(len(raw), total)
    finally:
        capture.release()
    people = smooth_people(raw, fps, min_cutoff, beta)
    return VideoSolve(str(path), fps, (width, height), [VideoFrame(i, p) for i, p in enumerate(people)])


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
    return frames, animation_root_positions(frames, fps)

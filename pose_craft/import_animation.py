"""Import an animation made by another tool (F18): a BVH from ComfyUI's SAM 3D Body,
a mocap library, Mixamo, Rokoko or PoseCraft itself becomes a tracked person -- so it can
be corrected, smoothed, retargeted onto a skeleton preset and sent through Live Link with
the same tools as a video PoseCraft tracked itself.

Any skeleton works; nothing depends on its bone names beyond finding five joints:

- the two **hands** (a joint named hand/wrist with a side) -- its parent is the elbow, its
  grandparent the shoulder (Mixamo LeftHand -> LeftForeArm -> LeftArm, UE5 hand_l ->
  lowerarm_l -> upperarm_l, SMPL L_Wrist -> L_Elbow -> L_Shoulder, Rigify hand.L ...);
- the two **feet** (foot/ankle) -- parent knee, grandparent hip;
- the **head** (head, no side).

Each frame's joint positions come from BVH forward kinematics. The points a BlazePose
skeleton has but a mocap skeleton doesn't -- ears, eyes, nose, mouth, index/pinky/thumb
tips, heels -- are placed around the head, hands and feet from offsets measured in the
file's own rest pose and turned by each joint's world rotation, so the head's turn and
the hands' and feet's roll come through. The rest pose sets the axes (up = hips to head,
left = right hip to left hip), so a Z-up file, an A-pose rest or a skeleton facing -Z all
import the same way, and lengths are scaled to a ~0.84 m leg.

The landmarks are hip-centred like BlazePose's world landmarks, so root motion is
re-estimated from the feet (foot locking), as for a video.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_craft.estimation.landmarker import DetectedPerson
from pose_craft.estimation.video import VideoFrame, VideoSolve
from pose_craft.retarget import skeleton as sk

LEG_LENGTH_M = 0.84
_FINGER_WORDS = ("thumb", "index", "middle", "ring", "pinky", "pinkie", "finger")


class ImportError_(ValueError):
    """A file that can't be imported (not a BVH, or no recognisable limbs)."""


@dataclass
class BvhJoint:
    name: str
    parent: int  # -1 for the root
    offset: np.ndarray
    channels: list[str]
    end_site: np.ndarray | None = None


@dataclass
class BvhClip:
    joints: list[BvhJoint]
    frames: np.ndarray  # (frame count, channel count)
    frame_time: float

    @property
    def fps(self) -> float:
        """Frames per second; a BVH's rounded frame time (0.033333) snaps back to 30."""
        if self.frame_time <= 0:
            return 30.0
        fps = 1.0 / self.frame_time
        return float(round(fps)) if abs(fps - round(fps)) < 0.01 else fps


def parse_bvh(text: str) -> BvhClip:
    tokens = text.split()
    joints: list[BvhJoint] = []
    stack: list[int] = []
    pos = 0

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ImportError_("the BVH ends in the middle of its HIERARCHY")
        pos += 1
        return tokens[pos - 1]

    if not tokens or tokens[0].upper() != "HIERARCHY":
        raise ImportError_("not a BVH file (no HIERARCHY)")
    pos = 1
    pending_end = False
    while pos < len(tokens):
        token = take()
        upper = token.upper()
        if upper in ("ROOT", "JOINT"):
            name = take()
            joints.append(BvhJoint(name, stack[-1] if stack else -1, np.zeros(3), []))
            pending_end = False
        elif upper == "END":
            take()  # "Site"
            pending_end = True
        elif token == "{":
            stack.append(-2 if pending_end else len(joints) - 1)
        elif token == "}":
            stack.pop()
            pending_end = False
        elif upper == "OFFSET":
            vector = np.array([float(take()), float(take()), float(take())])
            if stack and stack[-1] == -2:
                owner = next(i for i in reversed(stack[:-1]) if i >= 0)
                joints[owner].end_site = vector
            else:
                joints[-1].offset = vector
        elif upper == "CHANNELS":
            joints[-1].channels = [take() for _ in range(int(take()))]
        elif upper == "MOTION":
            break
    if not joints:
        raise ImportError_("the BVH has no joints")
    try:
        if take().rstrip(":").upper() != "FRAMES":
            raise ImportError_("the BVH's MOTION has no frame count")
        count = int(take())
        if take().upper().startswith("FRAME") and take().upper().startswith("TIME"):
            frame_time = float(take())
        else:
            raise ImportError_("the BVH's MOTION has no frame time")
        channel_count = sum(len(joint.channels) for joint in joints)
        values = np.array([float(v) for v in tokens[pos:pos + count * channel_count]], dtype=float)
    except (ValueError, StopIteration) as error:
        raise ImportError_(f"the BVH's MOTION can't be read: {error}") from None
    if values.size != count * channel_count:
        raise ImportError_(f"the BVH declares {count} frames but holds {values.size // max(1, channel_count)}")
    return BvhClip(joints, values.reshape(count, channel_count), frame_time)


def _local_transforms(clip: BvhClip, frame: np.ndarray):
    """(local rotations, root translation) of one MOTION line."""
    rotations, cursor, root_translation = [], 0, np.zeros(3)
    for index, joint in enumerate(clip.joints):
        rotation = Rotation.identity()
        for channel in joint.channels:
            value = frame[cursor]
            cursor += 1
            kind = channel.lower()
            if kind.endswith("position"):
                if index == 0:
                    root_translation["xyz".index(kind[0])] = value
            else:
                rotation = rotation * Rotation.from_euler(kind[0], value, degrees=True)
        rotations.append(rotation)
    return rotations, root_translation


def forward_kinematics(clip: BvhClip, frame: np.ndarray | None = None):
    """(positions (J, 3), world rotations [J], end-site positions {joint: (3,)}) of a frame,
    or of the rest pose when `frame` is None."""
    if frame is None:
        rotations, root = [Rotation.identity()] * len(clip.joints), np.zeros(3)
    else:
        rotations, root = _local_transforms(clip, frame)
    positions = np.zeros((len(clip.joints), 3))
    world: list[Rotation] = []
    ends = {}
    for index, joint in enumerate(clip.joints):
        if joint.parent < 0:
            positions[index] = root + joint.offset
            world.append(rotations[index])
        else:
            parent_world = world[joint.parent]
            positions[index] = positions[joint.parent] + parent_world.apply(joint.offset)
            world.append(parent_world * rotations[index])
        if joint.end_site is not None:
            ends[index] = positions[index] + world[index].apply(joint.end_site)
    return positions, world, ends


def _side(name: str) -> str | None:
    raw = name.split(":")[-1].split("|")[-1]
    lower = raw.lower()
    if "left" in lower or re.search(r"(^|[._\-\s])l([._\-\s]|$)", lower) or re.match(r"^l[A-Z]", raw):
        return "L"
    if "right" in lower or re.search(r"(^|[._\-\s])r([._\-\s]|$)", lower) or re.match(r"^r[A-Z]", raw):
        return "R"
    return None


def _depth(clip: BvhClip, index: int) -> int:
    depth = 0
    while clip.joints[index].parent >= 0:
        index = clip.joints[index].parent
        depth += 1
    return depth


def _find(clip: BvhClip, words, side: str | None) -> int | None:
    """The joint closest to the root whose name has one of `words` (and not a finger word) on `side`."""
    candidates = []
    for index, joint in enumerate(clip.joints):
        lower = joint.name.split(":")[-1].lower()
        if not any(word in lower for word in words) or any(word in lower for word in _FINGER_WORDS):
            continue
        if _side(joint.name) != side:
            continue
        candidates.append((_depth(clip, index), index))
    return min(candidates)[1] if candidates else None


def _children(clip: BvhClip, index: int) -> list[int]:
    return [i for i, joint in enumerate(clip.joints) if joint.parent == index]


@dataclass
class _Mapping:
    hand: dict  # "L"/"R" -> joint index
    foot: dict
    head: int


def find_limbs(clip: BvhClip) -> _Mapping:
    hands = {side: _find(clip, ("hand", "wrist"), side) for side in ("L", "R")}
    feet = {side: _find(clip, ("foot", "ankle"), side) for side in ("L", "R")}
    head = _find(clip, ("head",), None)
    missing = [f"{what} {side}" for what, found in (("hand", hands), ("foot", feet)) for side, index in found.items() if index is None]
    if head is None:
        missing.append("head")
    if missing:
        raise ImportError_(f"couldn't find these joints by name: {', '.join(missing)}")
    for index in [*hands.values(), *feet.values()]:
        parent = clip.joints[index].parent
        if parent < 0 or clip.joints[parent].parent < 0:
            raise ImportError_(f"'{clip.joints[index].name}' needs two parent joints (elbow/knee and shoulder/hip)")
    return _Mapping(hands, feet, head)


def _tip(clip: BvhClip, index: int, positions, ends) -> np.ndarray:
    """Where a joint's bone ends: its first child, else its end site, else its own position."""
    children = _children(clip, index)
    if children:
        return positions[children[0]]
    return ends.get(index, positions[index])


def landmarks_from_clip(clip: BvhClip) -> list[np.ndarray]:
    """One (33, 3) BlazePose-style landmark array per frame (metres, engine frame, hip-centred)."""
    limbs = find_limbs(clip)
    parent = lambda i: clip.joints[i].parent  # noqa: E731
    rest, _rest_world, rest_ends = forward_kinematics(clip)
    hips = {side: parent(parent(limbs.foot[side])) for side in ("L", "R")}
    hip_center = (rest[hips["L"]] + rest[hips["R"]]) / 2
    up = rest[limbs.head] - hip_center
    lateral = rest[hips["L"]] - rest[hips["R"]]
    if np.linalg.norm(up) < 1e-9 or np.linalg.norm(lateral) < 1e-9:
        raise ImportError_("the rest pose has no height or no hip width")
    up /= np.linalg.norm(up)
    forward = np.cross(lateral, up)
    forward /= np.linalg.norm(forward)
    lateral = np.cross(up, forward)
    to_engine = np.vstack([lateral, up, forward])  # rows: engine x (left), y (up), z (forward) in file axes
    knee = parent(limbs.foot["L"])
    leg = np.linalg.norm(rest[hips["L"]] - rest[knee]) + np.linalg.norm(rest[knee] - rest[limbs.foot["L"]])
    scale = LEG_LENGTH_M / leg if leg > 1e-9 else 1.0
    unit = LEG_LENGTH_M / scale  # one metre in file units, for the offsets below

    def offset(vector_engine) -> np.ndarray:
        """An engine-axes offset (metres) as a file-axes vector in the rest pose."""
        return to_engine.T @ (np.asarray(vector_engine, float) * unit)

    hand_offsets = {}
    for side in ("L", "R"):
        hand = limbs.hand[side]
        span = 0.25 * np.linalg.norm(_tip(clip, hand, rest, rest_ends) - rest[hand])
        hand_offsets[side] = {"index": to_engine.T @ np.array([0.0, 0.0, 1.0]) * span,
                              "pinky": to_engine.T @ np.array([0.0, 0.0, -1.0]) * span}
    frames = []
    for values in clip.frames:
        positions, world, ends = forward_kinematics(clip, values)
        landmarks = np.zeros((sk.NUM_LANDMARKS, 3))
        for side in ("L", "R"):
            hand, foot = limbs.hand[side], limbs.foot[side]
            elbow, knee_joint = parent(hand), parent(foot)
            shoulder, hip = parent(elbow), parent(knee_joint)
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_SHOULDER")] = positions[shoulder]
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_ELBOW")] = positions[elbow]
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_WRIST")] = positions[hand]
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_HIP")] = positions[hip]
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_KNEE")] = positions[knee_joint]
            landmarks[getattr(sk, f"{'LEFT' if side == 'L' else 'RIGHT'}_ANKLE")] = positions[foot]
            name = "LEFT" if side == "L" else "RIGHT"
            hand_end = _tip(clip, hand, positions, ends)
            landmarks[getattr(sk, f"{name}_INDEX")] = hand_end + world[hand].apply(hand_offsets[side]["index"])
            landmarks[getattr(sk, f"{name}_PINKY")] = hand_end + world[hand].apply(hand_offsets[side]["pinky"])
            landmarks[getattr(sk, f"{name}_THUMB")] = (positions[hand] + hand_end) / 2 + world[hand].apply(hand_offsets[side]["index"])
            toe = _tip(clip, foot, positions, ends)
            landmarks[getattr(sk, f"{name}_FOOT_INDEX")] = toe
            landmarks[getattr(sk, f"{name}_HEEL")] = positions[foot] + world[foot].apply(offset([0.0, -0.06, -0.06]))
        head, head_world = positions[limbs.head], world[limbs.head]
        center = head + head_world.apply(offset([0.0, 0.08, 0.0]))
        for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
            landmarks[getattr(sk, f"{side}_EAR")] = center + head_world.apply(offset([sign * 0.075, 0.0, 0.0]))
            landmarks[getattr(sk, f"{side}_EYE")] = center + head_world.apply(offset([sign * 0.032, 0.03, 0.08]))
            landmarks[getattr(sk, f"{side}_EYE_INNER")] = center + head_world.apply(offset([sign * 0.015, 0.03, 0.085]))
            landmarks[getattr(sk, f"{side}_EYE_OUTER")] = center + head_world.apply(offset([sign * 0.045, 0.03, 0.075]))
        landmarks[sk.NOSE] = center + head_world.apply(offset([0.0, 0.0, 0.1]))
        landmarks[sk.MOUTH_LEFT] = center + head_world.apply(offset([0.025, -0.04, 0.085]))
        landmarks[sk.MOUTH_RIGHT] = center + head_world.apply(offset([-0.025, -0.04, 0.085]))
        hip_mid = (landmarks[sk.LEFT_HIP] + landmarks[sk.RIGHT_HIP]) / 2
        frames.append(((landmarks - hip_mid) @ to_engine.T) * scale)
    return frames


def _image_xy(landmarks: np.ndarray) -> np.ndarray:
    """A front view, for selecting the person in the (photo-less) preview."""
    xy = np.stack([0.5 + landmarks[:, 0] * 0.35, 0.5 - landmarks[:, 1] * 0.35], axis=1)
    return np.clip(xy, 0.0, 1.0)


def import_bvh(path: str | Path) -> VideoSolve:
    """The BVH at `path` as a one-person VideoSolve (track 0), at the file's frame rate."""
    path = Path(path)
    try:
        clip = parse_bvh(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as error:
        raise ImportError_(f"couldn't read {path.name}: {error}") from None
    frames = []
    for index, landmarks in enumerate(landmarks_from_clip(clip)):
        person = DetectedPerson(landmarks=landmarks, image_landmarks_xy=_image_xy(landmarks),
                                visibility=np.ones(sk.NUM_LANDMARKS))
        frames.append(VideoFrame(index=index, people={0: person}))
    return VideoSolve(path=str(path), fps=clip.fps, frame_size=(1000, 1000), frames=frames)

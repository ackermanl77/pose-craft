"""Recording from a webcam (F14): a short clip saved as an .mp4, then tracked like any video
(estimation.video.solve_video), so the hands, face, edits, exports and Send to Blender all apply.

A webcam's reported frame rate is often not what it delivers (low light, USB bandwidth), so the
clip is written at the reported rate and, if the frames actually came at a different rate,
rewritten at the measured one -- the animation then plays at the speed it was performed.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

# The probe stops at the first index that doesn't open (cameras are numbered 0, 1, ...).
MAX_CAMERAS = 6
DEFAULT_FPS = 30.0
# Rewrite the clip when the measured rate is this far from the one it was written at.
FPS_TOLERANCE = 0.1


def recordings_dir() -> Path:
    """Where recordings are kept: %LOCALAPPDATA%\\PoseCraft\\recordings (~/.local/share/PoseCraft/recordings
    elsewhere), or POSECRAFT_RECORDINGS_DIR."""
    override = os.environ.get("POSECRAFT_RECORDINGS_DIR")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PoseCraft" / "recordings"
    return Path.home() / ".local" / "share" / "PoseCraft" / "recordings"


def open_capture(source):
    """cv2.VideoCapture for a camera index (DirectShow on Windows: it opens fast and names
    devices consistently) or a file path (tests, or a capture card's stream URL)."""
    import cv2

    if isinstance(source, int) and sys.platform == "win32":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def list_cameras(max_cameras: int = MAX_CAMERAS) -> list[int]:
    """The camera indices that open and deliver a frame."""
    found = []
    for index in range(max_cameras):
        capture = open_capture(index)
        try:
            if not capture.isOpened():
                break
            ok, _frame = capture.read()
            if ok:
                found.append(index)
        finally:
            capture.release()
    return found


@dataclasses.dataclass
class RecordedClip:
    path: Path
    frames: int
    fps: float  # the rate the clip is written at (measured for a camera)


def record_clip(
    source,
    path: Path,
    seconds: float,
    *,
    on_frame: Callable[[np.ndarray], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RecordedClip:
    """Records up to `seconds` from `source` (a camera index or a file) into `path` (.mp4).
    on_frame(bgr) sees every frame (a live preview); should_cancel() stops early, keeping what was
    recorded. Raises OSError when the source doesn't open or delivers nothing."""
    import cv2

    capture = open_capture(source)
    if not capture.isOpened():
        raise OSError(f"couldn't open the camera {source!r}")
    reported = capture.get(cv2.CAP_PROP_FPS) or 0.0
    fps = reported if 1.0 <= reported <= 240.0 else DEFAULT_FPS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    started = None
    try:
        while True:
            if should_cancel is not None and should_cancel():
                break
            ok, frame = capture.read()
            if not ok:
                break  # a file ended, or the camera went away
            now = clock()
            if started is None:
                started = now
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise OSError(f"couldn't write {path}")
            writer.write(frame)
            frames += 1
            if on_frame is not None:
                on_frame(frame)
            if isinstance(source, int) and now - started >= seconds:
                break
            if not isinstance(source, int) and frames >= round(seconds * fps):
                break
        elapsed = (clock() - started) if started is not None else 0.0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        path.unlink(missing_ok=True)
        raise OSError(f"the camera {source!r} delivered no frames")
    if isinstance(source, int) and elapsed > 0:
        measured = frames / elapsed
        if abs(measured - fps) > FPS_TOLERANCE * fps:
            _rewrite_at(path, measured)
            fps = measured
    return RecordedClip(path, frames, float(fps))


def _rewrite_at(path: Path, fps: float) -> None:
    """Re-encodes `path` at `fps` (same frames, new timing)."""
    import cv2

    tmp = path.with_name(path.stem + ".retimed" + path.suffix)
    capture = cv2.VideoCapture(str(path))
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    tmp.replace(path)

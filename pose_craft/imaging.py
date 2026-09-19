"""Small OpenCV helpers that aren't specific to pose estimation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Like `cv2.imread`, but works with non-ASCII paths on Windows.

    `cv2.imread` opens the path with a narrow (ANSI) codepage internally on
    Windows, so any path containing characters outside it silently fails and
    returns None. Reading the bytes ourselves and decoding in-memory sidesteps
    that entirely. Returns None on a missing/corrupt/unreadable file, same as
    `cv2.imread`, rather than raising.
    """
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


# DAZ Studio's own Face Transfer feature (a photo-to-3D feature for faces, the
# same category of tool as PoseCraft but for face shape instead of body pose)
# documents a minimum source resolution and framing/lighting guidance for best
# results. There's no exact equivalent number for MediaPipe's full-body pose
# model, but the same category of advice applies -- so this project borrows the
# structure (a resolution floor plus non-checkable framing tips) rather than a
# specific number from DAZ's own docs.
MIN_RECOMMENDED_DIMENSION = 480

PHOTO_GUIDANCE = (
    "Full body in frame, hands and feet not cut off",
    "Facing the camera or at a slight angle, not straight from the side",
    "Even, bright lighting -- avoid strong shadows across the body",
    "Limbs not hidden behind the body, clothing, or other people",
)


def assess_photo_quality(image_bgr: np.ndarray) -> list[str]:
    """Returns human-readable warnings about the photo before solving. Doesn't
    block solving (a low-resolution or badly-framed photo can still produce a
    usable pose, just a less reliable one) -- only resolution is actually
    checkable in code; PHOTO_GUIDANCE's framing/lighting tips are shown
    unconditionally by the caller instead, since there's no reliable way to
    detect "is this person facing sideways" from the raw image alone.
    """
    warnings = []
    height, width = image_bgr.shape[:2]
    if min(width, height) < MIN_RECOMMENDED_DIMENSION:
        warnings.append(
            f"Photo is {width}x{height}, below the recommended "
            f"{MIN_RECOMMENDED_DIMENSION}px minimum on the shorter side -- "
            "pose estimation is likely to be less accurate on low-resolution photos."
        )
    return warnings

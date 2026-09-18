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

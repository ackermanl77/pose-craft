from __future__ import annotations

import cv2
import numpy as np

from pose_craft.imaging import MIN_RECOMMENDED_DIMENSION, assess_photo_quality, imread_unicode


def _write_png(path, image: np.ndarray) -> None:
    # cv2.imwrite has the same narrow-codepage limitation on Windows that this
    # module works around for reading -- encode in-memory and write the bytes
    # through plain Python file I/O instead, so the fixture itself doesn't
    # depend on the behavior being tested.
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())


def test_imread_unicode_reads_ascii_path(tmp_path):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # pure red in BGR
    path = tmp_path / "plain.png"
    _write_png(path, image)

    loaded = imread_unicode(path)
    assert loaded is not None
    assert loaded.shape == (4, 4, 3)
    assert np.array_equal(loaded[0, 0], [0, 0, 255])


def test_imread_unicode_reads_non_ascii_path(tmp_path):
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    path = tmp_path / "fotografía_日本語.png"
    _write_png(path, image)

    loaded = imread_unicode(path)
    assert loaded is not None
    assert loaded.shape == (4, 4, 3)


def test_imread_unicode_missing_file_returns_none(tmp_path):
    assert imread_unicode(tmp_path / "does_not_exist.png") is None


def test_assess_photo_quality_flags_low_resolution():
    small = np.zeros((200, 300, 3), dtype=np.uint8)
    warnings = assess_photo_quality(small)
    assert len(warnings) == 1
    assert "200" in warnings[0] and "300" in warnings[0]


def test_assess_photo_quality_accepts_recommended_resolution():
    big = np.zeros((MIN_RECOMMENDED_DIMENSION, MIN_RECOMMENDED_DIMENSION * 2, 3), dtype=np.uint8)
    assert assess_photo_quality(big) == []


def test_assess_photo_quality_checks_the_shorter_side():
    # Wide but short -- the short side (300) is what should trigger the warning,
    # not the long side (2000) alone.
    wide_but_short = np.zeros((300, 2000, 3), dtype=np.uint8)
    assert len(assess_photo_quality(wide_but_short)) == 1

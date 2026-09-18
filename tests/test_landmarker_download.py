from __future__ import annotations

import pytest

from pose_craft.estimation import landmarker


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(landmarker, "CACHE_DIR", tmp_path)
    return tmp_path


def _fake_urlretrieve(payload: bytes):
    def _retrieve(url, filename):
        with open(filename, "wb") as f:
            f.write(payload)

    return _retrieve


def test_downloads_and_caches_model(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES + 10)
    monkeypatch.setattr(landmarker.urllib.request, "urlretrieve", _fake_urlretrieve(payload))

    dest = landmarker.ensure_model_downloaded("lite")
    assert dest.exists()
    assert dest.read_bytes() == payload

    # No leftover .part temp files.
    assert list(tmp_path.glob("*.part")) == []


def test_second_call_does_not_redownload(monkeypatch, tmp_path):
    calls = {"count": 0}

    def _retrieve(url, filename):
        calls["count"] += 1
        with open(filename, "wb") as f:
            f.write(b"x" * (landmarker.MIN_MODEL_BYTES + 1))

    monkeypatch.setattr(landmarker.urllib.request, "urlretrieve", _retrieve)

    landmarker.ensure_model_downloaded("lite")
    landmarker.ensure_model_downloaded("lite")
    assert calls["count"] == 1


def test_truncated_download_is_not_cached_and_leaves_no_partial_file(monkeypatch, tmp_path):
    truncated = b"x" * 10  # well under MIN_MODEL_BYTES
    monkeypatch.setattr(landmarker.urllib.request, "urlretrieve", _fake_urlretrieve(truncated))

    with pytest.raises(OSError):
        landmarker.ensure_model_downloaded("lite")

    dest = tmp_path / "pose_landmarker_lite.task"
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_preexisting_truncated_file_is_redownloaded(monkeypatch, tmp_path):
    dest = tmp_path / "pose_landmarker_lite.task"
    dest.write_bytes(b"corrupt")  # simulates a previous interrupted download

    good_payload = b"x" * (landmarker.MIN_MODEL_BYTES + 1)
    monkeypatch.setattr(landmarker.urllib.request, "urlretrieve", _fake_urlretrieve(good_payload))

    result = landmarker.ensure_model_downloaded("lite")
    assert result.read_bytes() == good_payload

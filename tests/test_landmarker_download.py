from __future__ import annotations

import hashlib
import io

import pytest

from pose_craft.estimation import landmarker

_PINNED_SHA256 = landmarker.MODEL_SHA256  # captured before the fixture below blanks it


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(landmarker, "CACHE_DIR", tmp_path)
    # The fake payloads below aren't real models; the R3 tests at the end pin
    # their own hashes.
    monkeypatch.setattr(landmarker, "MODEL_SHA256", {})
    return tmp_path


class _FakeResponse(io.BytesIO):
    """Stands in for urllib's HTTP response: a readable, closable body plus headers."""

    def __init__(self, payload: bytes, send_length: bool = True):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))} if send_length else {}


def _fake_urlopen(payload: bytes, calls: list | None = None, send_length: bool = True):
    def _open(url, timeout=None):
        if calls is not None:
            calls.append({"url": url, "timeout": timeout})
        return _FakeResponse(payload, send_length)

    return _open


def test_downloads_and_caches_model(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES + 10)
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload))

    dest = landmarker.ensure_model_downloaded("lite")
    assert dest.exists()
    assert dest.read_bytes() == payload

    # No leftover .part temp files.
    assert list(tmp_path.glob("*.part")) == []


def test_download_uses_a_timeout(monkeypatch):
    """Regression test: urlretrieve had no timeout, so a stalled connection hung
    the solve forever (and, through closeEvent's wait, the whole app)."""
    calls = []
    monkeypatch.setattr(
        landmarker.urllib.request, "urlopen", _fake_urlopen(b"x" * (landmarker.MIN_MODEL_BYTES + 1), calls)
    )

    landmarker.ensure_model_downloaded("lite")

    assert calls[0]["timeout"] == landmarker.DOWNLOAD_TIMEOUT_SECONDS
    assert 0 < landmarker.DOWNLOAD_TIMEOUT_SECONDS <= 60


def test_cancelled_download_raises_and_leaves_nothing_behind(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES * 3)
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload))
    progress = []

    with pytest.raises(landmarker.DownloadCancelled):
        landmarker.ensure_model_downloaded(
            "lite",
            on_progress=lambda downloaded, total: progress.append(downloaded),
            should_cancel=lambda: len(progress) >= 2,
        )

    assert 0 < progress[-1] < len(payload)  # stopped part-way, not after the whole file
    assert not (tmp_path / "pose_landmarker_lite.task").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_second_call_does_not_redownload(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        landmarker.urllib.request, "urlopen", _fake_urlopen(b"x" * (landmarker.MIN_MODEL_BYTES + 1), calls)
    )

    landmarker.ensure_model_downloaded("lite")
    landmarker.ensure_model_downloaded("lite")
    assert len(calls) == 1


def test_truncated_download_is_not_cached_and_leaves_no_partial_file(monkeypatch, tmp_path):
    truncated = b"x" * 10  # well under MIN_MODEL_BYTES
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(truncated))

    with pytest.raises(OSError):
        landmarker.ensure_model_downloaded("lite")

    dest = tmp_path / "pose_landmarker_lite.task"
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_preexisting_truncated_file_is_redownloaded(monkeypatch, tmp_path):
    dest = tmp_path / "pose_landmarker_lite.task"
    dest.write_bytes(b"corrupt")  # simulates a previous interrupted download

    good_payload = b"x" * (landmarker.MIN_MODEL_BYTES + 1)
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(good_payload))

    result = landmarker.ensure_model_downloaded("lite")
    assert result.read_bytes() == good_payload


def test_progress_callback_receives_download_updates(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES * 2 + 100)
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload))

    updates = []
    landmarker.ensure_model_downloaded("lite", on_progress=lambda downloaded, total: updates.append((downloaded, total)))

    assert len(updates) >= 2
    assert all(total == len(payload) for _downloaded, total in updates)
    assert [downloaded for downloaded, _total in updates] == sorted(d for d, _ in updates)
    assert updates[-1][0] == len(payload)


def test_progress_without_content_length_reports_zero_total(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES + 1)
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload, send_length=False))

    updates = []
    landmarker.ensure_model_downloaded("lite", on_progress=lambda downloaded, total: updates.append((downloaded, total)))

    assert updates[-1] == (len(payload), 0)


# --- R3: pinned model versions and checksums --------------------------------------


def test_every_model_url_is_pinned_to_a_version_with_a_checksum():
    for variant, url in landmarker.MODEL_URLS.items():
        assert "/latest/" not in url and "/float16/1/" in url, url
        assert _PINNED_SHA256[variant], variant
        assert all(len(h) == 64 and int(h, 16) >= 0 for h in _PINNED_SHA256[variant])


def test_download_with_the_wrong_checksum_is_rejected_and_not_saved(monkeypatch, tmp_path):
    payload = b"x" * (landmarker.MIN_MODEL_BYTES + 1)
    monkeypatch.setattr(landmarker, "MODEL_SHA256", {"lite": ("0" * 64,)})
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload))

    with pytest.raises(landmarker.ModelChecksumError):
        landmarker.ensure_model_downloaded("lite")
    assert not (tmp_path / "pose_landmarker_lite.task").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_with_the_pinned_checksum_is_saved(monkeypatch, tmp_path):
    payload = b"y" * (landmarker.MIN_MODEL_BYTES + 1)
    monkeypatch.setattr(landmarker, "MODEL_SHA256", {"lite": (hashlib.sha256(payload).hexdigest(),)})
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(payload))

    assert landmarker.ensure_model_downloaded("lite").read_bytes() == payload


def test_cached_file_with_the_wrong_checksum_is_downloaded_again(monkeypatch, tmp_path):
    good = b"g" * (landmarker.MIN_MODEL_BYTES + 1)
    (tmp_path / "pose_landmarker_lite.task").write_bytes(b"t" * (landmarker.MIN_MODEL_BYTES + 1))  # tampered
    monkeypatch.setattr(landmarker, "MODEL_SHA256", {"lite": (hashlib.sha256(good).hexdigest(),)})
    calls = []
    monkeypatch.setattr(landmarker.urllib.request, "urlopen", _fake_urlopen(good, calls))

    assert landmarker.ensure_model_downloaded("lite").read_bytes() == good
    assert len(calls) == 1

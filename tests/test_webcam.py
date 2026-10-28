"""Recording from a webcam (F14), with a video file standing in for the camera: the clip, its
frame rate, stopping early, and the dialog handing the clip to the video tracking."""

from __future__ import annotations

import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose_craft.estimation import webcam  # noqa: E402


def _video(path, frames=30, fps=30.0, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for k in range(frames):
        writer.write(np.full((size[1], size[0], 3), (8 * k) % 256, dtype=np.uint8))
    writer.release()
    return path


def _frames_and_fps(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return count, fps


def test_a_file_source_records_the_requested_length(tmp_path):
    source = _video(tmp_path / "source.avi", frames=60)
    seen = []
    clip = webcam.record_clip(str(source), tmp_path / "out" / "clip.mp4", 1.0, on_frame=lambda f: seen.append(f.shape))
    assert clip.frames == 30 and clip.fps == pytest.approx(30.0) and len(seen) == 30
    assert seen[0] == (48, 64, 3)
    assert _frames_and_fps(clip.path) == (30, pytest.approx(30.0))


def test_stopping_early_keeps_what_was_recorded(tmp_path):
    source = _video(tmp_path / "source.avi", frames=60)
    seen = []
    clip = webcam.record_clip(str(source), tmp_path / "clip.mp4", 10.0, on_frame=seen.append,
                              should_cancel=lambda: len(seen) >= 12)
    assert clip.frames == 12 and _frames_and_fps(clip.path)[0] == 12


def test_a_camera_slower_than_it_reports_is_retimed_to_the_measured_rate(tmp_path, monkeypatch):
    """A camera that claims 30 fps but delivers 15: the clip plays at 15 so the motion keeps its speed."""
    source = _video(tmp_path / "source.avi", frames=60, fps=30.0)
    monkeypatch.setattr(webcam, "open_capture", lambda s: cv2.VideoCapture(str(source)))
    ticks = iter(k / 15.0 for k in range(1000))  # a frame every 1/15 s
    clip = webcam.record_clip(0, tmp_path / "clip.mp4", 2.0, clock=lambda: next(ticks))
    assert clip.frames == 31  # 0 .. 2.0 s inclusive
    assert clip.fps == pytest.approx(15.0, rel=0.05)
    assert _frames_and_fps(clip.path) == (31, pytest.approx(clip.fps, rel=0.01))
    assert not list(tmp_path.glob("*.retimed.*"))


def test_a_source_that_does_not_open_or_delivers_nothing_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(OSError, match="couldn't open"):
        webcam.record_clip(str(tmp_path / "missing.avi"), tmp_path / "clip.mp4", 1.0)

    class _Empty:
        def isOpened(self):
            return True

        def get(self, prop):
            return 30.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(webcam, "open_capture", lambda s: _Empty())
    with pytest.raises(OSError, match="no frames"):
        webcam.record_clip(0, tmp_path / "clip.mp4", 1.0)
    assert not (tmp_path / "clip.mp4").exists()


def test_recordings_go_to_the_per_user_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("POSECRAFT_RECORDINGS_DIR", str(tmp_path / "rec"))
    assert webcam.recordings_dir() == tmp_path / "rec"
    monkeypatch.delenv("POSECRAFT_RECORDINGS_DIR")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert webcam.recordings_dir() == tmp_path / "PoseCraft" / "recordings"


# --- the dialog and the main window ------------------------------------------------------

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait_until(qapp, predicate, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_the_dialog_records_with_a_preview_and_remembers_its_settings(qapp, tmp_path):
    from pose_craft.gui.webcam_dialog import WebcamDialog

    source = _video(tmp_path / "source.avi", frames=60)
    settings = QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat)
    dialog = WebcamDialog(settings=settings, sources=[str(source)], out_dir=tmp_path / "rec")
    dialog.seconds_spin.setValue(1)
    dialog.countdown_spin.setValue(0)
    dialog.record_btn.click()
    assert dialog.record_btn.text() == "&Stop"
    assert _wait_until(qapp, lambda: dialog.result() == QDialog.Accepted)
    assert dialog.clip.frames == 30 and dialog.clip.path.parent == tmp_path / "rec"
    assert dialog.preview.pixmap() is not None and not dialog.preview.pixmap().isNull()
    assert int(settings.value("webcam/seconds")) == 1 and int(settings.value("webcam/countdown")) == 0


def test_the_dialog_counts_down_and_cancel_discards_the_recording(qapp, tmp_path, monkeypatch):
    from pose_craft.gui import webcam_dialog

    source = _video(tmp_path / "source.avi", frames=600)
    real_record = webcam.record_clip

    def slow_record(*args, on_frame=None, **kwargs):  # a camera delivers in real time
        return real_record(*args, on_frame=lambda f: (time.sleep(0.01), on_frame(f)), **kwargs)

    monkeypatch.setattr(webcam_dialog.webcam, "record_clip", slow_record)
    dialog = webcam_dialog.WebcamDialog(sources=[str(source)], out_dir=tmp_path / "rec")
    dialog._timer.setInterval(20)
    dialog.seconds_spin.setValue(20)
    dialog.countdown_spin.setValue(2)
    dialog.record_btn.click()
    assert dialog.status.text() == "Recording in 2..." and not dialog.record_btn.isEnabled()
    assert _wait_until(qapp, lambda: dialog._worker is not None)
    assert _wait_until(qapp, lambda: dialog._recording_path.exists())
    dialog.reject()
    assert dialog.clip is None and not list((tmp_path / "rec").glob("*.mp4"))


def test_no_camera_means_no_record_button(qapp, tmp_path):
    from pose_craft.gui.webcam_dialog import WebcamDialog

    dialog = WebcamDialog(sources=[], out_dir=tmp_path)
    assert not dialog.record_btn.isEnabled() and "No camera" in dialog.status.text()


def test_record_from_webcam_tracks_the_clip_like_an_opened_video(qapp, tmp_path, monkeypatch):
    from pose_craft.gui import webcam_dialog
    from pose_craft.gui.main_window import MainWindow

    clip = webcam.RecordedClip(tmp_path / "webcam.mp4", 30, 30.0)

    class _Accepting:
        Accepted = QDialog.Accepted

        def __init__(self, parent, settings=None, sources=None):
            self.clip = clip

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(webcam_dialog, "WebcamDialog", _Accepting)
    window = MainWindow(settings=QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat))
    loaded = []
    monkeypatch.setattr(window, "load_video", loaded.append)
    try:
        assert window.record_webcam_action.text() == "Record from &Webcam..."
        window._on_record_webcam()
        assert loaded == [str(clip.path)]
        window._set_busy(True, "solving")
        assert not window.record_webcam_action.isEnabled()
        window._set_busy(False)
        assert window.record_webcam_action.isEnabled()
    finally:
        window.close()

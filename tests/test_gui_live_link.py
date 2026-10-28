"""Offscreen Qt tests for the video/Live Link work: rotation smoothing in the
window and Send Animation to Blender (B12), Export All People as BVH and
Auto-send on change (F10), the photo view's hover highlighting (F5) and the
keyboard-focus/accessible-name pass (F8). Widgets are driven programmatically -- no OS input is simulated."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QMouseEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QSlider,
)

from pose_craft.estimation.video import VideoFrame, VideoSolve  # noqa: E402
from pose_craft.export.blender_pose import build_pose_document  # noqa: E402
from pose_craft.gui import main_window as mw  # noqa: E402
from pose_craft.gui import photo_view  # noqa: E402
from pose_craft.gui.main_window import MainWindow  # noqa: E402
from pose_craft.retarget import skeleton as sk  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_gui_flows import _FakeEstimator, _person, _wait_until  # noqa: E402
from test_gui_important import _window_with_photo  # noqa: E402
from test_retarget_frames import _articulated_pose, _standing_t_pose  # noqa: E402

_OK_REPLY = {"ok": True, "applied": ["spine"], "requested": 15, "armature": "Rig"}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "posecraft.ini"), QSettings.IniFormat)


@pytest.fixture
def no_dialogs(monkeypatch):
    shown = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(mw.QMessageBox, name, lambda *args, _n=name, **kwargs: shown.append((_n, args)))
    return shown


def _pump(qapp, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def _angle(a, b) -> float:
    return float(np.degrees((a * b.inv()).magnitude()))


def _wobbling_video(tmp_path) -> VideoSolve:
    """Two tracks whose pose changes every frame (so smoothing has something
    to do); track 0 is lost in frame 1, track 1 is in every frame."""
    import cv2

    clip = tmp_path / "wobble.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
    for k in range(6):
        writer.write(np.full((48, 64, 3), 30 * k, dtype=np.uint8))
    writer.release()
    still, bent = _standing_t_pose(), _articulated_pose()
    frames = []
    for k in range(6):
        pose = still if k % 2 else bent
        other = bent if k % 2 else still
        people = {1: _person(other, x_offset=0.3)}
        if k != 1:
            people[0] = _person(pose)
        frames.append(VideoFrame(k, people=people))
    return VideoSolve(str(clip), 24.0, (64, 48), frames)


def _video_window(settings, tmp_path, smoothing: int) -> MainWindow:
    window = MainWindow(settings=settings)
    window.video_rotation_smoothing_spin.setValue(smoothing)
    window._on_video_finished(_wobbling_video(tmp_path), _FakeEstimator())
    return window


# --- B12: rotation smoothing in the window ---------------------------------------------------------


def test_zero_rotation_smoothing_exports_the_frames_as_solved(qapp, settings, tmp_path, no_dialogs):
    window = _video_window(settings, tmp_path, 0)
    try:
        indices, frames, _roots = window._video_track_frames(0)
        assert indices == [0, 2, 3, 4, 5]  # the lost frame is skipped, not invented
        raw = [window._document_inputs_for(p, False, window._orientation_correction(), None, window._photo_aspect())[0]
               for p in window._video.people_of(0)]
        assert all(_angle(f[n], r[n]) == 0.0 for f, r in zip(frames, raw) for n in r)
    finally:
        window.close()


def test_rotation_smoothing_reaches_the_preview_and_is_per_track(qapp, settings, tmp_path, monkeypatch, no_dialogs):
    window = _video_window(settings, tmp_path, 80)
    try:
        shown = []
        monkeypatch.setattr(window.viewport, "show_skeleton", lambda positions, **kw: shown.append(kw["rotations"]))
        window.frame_slider.setValue(2)
        indices, frames, _roots = window._video_track_frames(window._edit_key(window.person_combo.currentIndex()))
        expected = frames[indices.index(2)]
        assert all(_angle(shown[-1][n], expected[n]) < 1e-9 for n in expected)
        raw, _confidence = window._current_person_document_inputs()
        assert max(_angle(shown[-1][n], raw[n]) for n in raw) > 1.0  # smoothed, not the raw solve
        # Each track has its own filter: track 1's result doesn't depend on track 0's frames.
        _i0, smooth0, _r = window._video_track_frames(0)
        _i1, smooth1, _r = window._video_track_frames(1)
        assert smooth0 is not smooth1 and len(smooth1) == 6
    finally:
        window.close()


def test_track_frames_are_cached_until_an_edit_or_option_changes(qapp, settings, tmp_path, no_dialogs):
    window = _video_window(settings, tmp_path, 35)
    try:
        first = window._video_track_frames(0)
        assert window._video_track_frames(0) is first  # scrubbing doesn't re-solve the whole track
        window.person_combo.setCurrentIndex(window._person_keys.index(0))
        window.flip_bone_depth(window.person_combo.currentIndex(), "forearm.L")
        edited = window._video_track_frames(0)
        assert edited is not first
        window.video_rotation_smoothing_spin.setValue(60)
        assert window._video_track_frames(0) is not edited
        window.undo_stack.undo()
        window.video_rotation_smoothing_spin.setValue(35)
        again = window._video_track_frames(0)
        assert all(_angle(a[n], b[n]) < 1e-12 for a, b in zip(again[1], first[1]) for n in a)
    finally:
        window.close()


def test_live_link_uses_the_displayed_persons_track(qapp, settings, tmp_path, no_dialogs):
    """Regression: with the selected person out of frame, the person shown
    instead got the raw (unsmoothed) pose, because the lookup used the
    selected track rather than the displayed one."""
    window = _video_window(settings, tmp_path, 80)
    try:
        window.person_combo.setCurrentIndex(window._person_keys.index(0))
        window.frame_slider.setValue(1)  # track 0 is lost here; track 1 is shown
        assert window._person_keys == [1]
        indices, frames, _roots = window._video_track_frames(1)
        document = window._current_live_link_document()
        expected = build_pose_document(frames[indices.index(1)])
        for name in sk.BONE_NAMES:
            assert document["bones"][name]["quaternion_wxyz"] == pytest.approx(
                expected["bones"][name]["quaternion_wxyz"]
            )
    finally:
        window.close()


def test_rotation_smoothing_is_remembered(qapp, settings, tmp_path):
    window = MainWindow(settings=settings)
    window.video_rotation_smoothing_spin.setValue(0)
    window.auto_send_checkbox.setChecked(True)
    window.auto_send_debounce_spin.setValue(800)
    window.close()
    again = MainWindow(settings=settings)
    try:
        assert again.video_rotation_smoothing_spin.value() == 0
        assert again.auto_send_checkbox.isChecked() and again.auto_send_debounce_spin.value() == 800
    finally:
        again.close()
    fresh = MainWindow(settings=QSettings(str(tmp_path / "fresh.ini"), QSettings.IniFormat))
    try:
        assert fresh.video_rotation_smoothing_spin.value() == mw.DEFAULT_VIDEO_ROTATION_SMOOTHING
        assert not fresh.auto_send_checkbox.isChecked()  # off unless the user turns it on
    finally:
        fresh.close()


# --- B12: Send Animation to Blender ------------------------------------------------------------------


def test_send_animation_sends_the_tracks_smoothed_frames_off_the_ui_thread(
    qapp, settings, tmp_path, monkeypatch, no_dialogs
):
    sent = []
    gate = threading.Event()

    def fake_send_animation(document, host, port):
        sent.append(document)
        gate.wait(5)
        return {"ok": True, "applied": list(sk.BONE_NAMES), "requested": 15, "armature": "Rig",
                "animation_frames": len(document["frames"]), "start_frame": 1, "end_frame": 6, "chunks": 1}

    monkeypatch.setattr(mw, "send_animation_document", fake_send_animation)
    monkeypatch.setattr(mw, "send_pose_document", lambda *a, **k: pytest.fail("a single pose was sent"))
    window = _video_window(settings, tmp_path, 50)
    try:
        window.person_combo.setCurrentIndex(window._person_keys.index(0))
        assert window.send_animation_btn.isEnabled()
        window.send_animation_btn.click()
        assert _wait_until(qapp, lambda: bool(sent))
        assert not window.send_animation_btn.isEnabled() and not window.send_to_blender_btn.isEnabled()
        assert "5-frame animation" in window.statusBar().currentMessage()
        gate.set()
        assert _wait_until(qapp, lambda: "Keyed 5 frame(s)" in window.statusBar().currentMessage())
        assert "Blender frames 1-6" in window.statusBar().currentMessage()
        assert window.send_animation_btn.isEnabled()

        document = sent[0]
        assert document["format"] == "pose_craft_animation" and document["fps"] == 24.0
        indices, frames, _roots = window._video_track_frames(0)
        assert [f["frame_index"] for f in document["frames"]] == indices == [0, 2, 3, 4, 5]
        expected = build_pose_document(frames[1])
        assert document["frames"][1]["pose"]["bones"]["forearm.L"]["quaternion_wxyz"] == pytest.approx(
            expected["bones"]["forearm.L"]["quaternion_wxyz"]
        )
        assert no_dialogs == []
    finally:
        gate.set()
        window.close()


def test_a_refused_animation_is_reported(qapp, settings, tmp_path, monkeypatch, no_dialogs):
    monkeypatch.setattr(mw, "send_animation_document",
                        lambda document, host, port: {"ok": False, "error": "update the PoseCraft add-on"})
    window = _video_window(settings, tmp_path, 0)
    try:
        window.send_animation_btn.click()
        assert _wait_until(qapp, lambda: bool(no_dialogs))
        assert "update the PoseCraft add-on" in no_dialogs[0][1][2]
    finally:
        window.close()


def test_the_worker_routes_animations_to_the_chunked_sender(monkeypatch):
    calls = []
    monkeypatch.setattr(mw, "send_animation_document", lambda d, host, port: calls.append("animation") or {})
    monkeypatch.setattr(mw, "send_pose_document", lambda d, host, port: calls.append("pose") or {})
    mw.SendToBlenderWorker({"format": "pose_craft_animation", "frames": []}, "127.0.0.1", 1).run()
    mw.SendToBlenderWorker({"format": "pose_craft_pose"}, "127.0.0.1", 1).run()
    assert calls == ["animation", "pose"]


# --- F10: Auto-send on change -------------------------------------------------------------------------


def _auto_window(settings, monkeypatch, send):
    monkeypatch.setattr(mw, "send_pose_document", send)
    window = _window_with_photo(settings, [_person(), _person(_articulated_pose(), x_offset=0.3)])
    window.auto_send_debounce_spin.setValue(60)
    return window


def test_auto_send_debounces_a_burst_of_changes_into_one_send(qapp, settings, monkeypatch, no_dialogs):
    sent = []
    window = _auto_window(settings, monkeypatch, lambda document, host, port: sent.append(document) or _OK_REPLY)
    try:
        _pump(qapp, 0.15)
        assert sent == []  # nothing goes out until the user turns it on
        window.auto_send_checkbox.setChecked(True)
        for k in range(6):  # a burst: selection, mirror, orientation...
            window.person_combo.setCurrentIndex(k % 2)
            window.mirror_checkbox.setChecked(k % 3 == 0)
        _pump(qapp, 0.4)
        assert len(sent) == 1
        assert "Applied 1/15" in window.statusBar().currentMessage()
        window.person_combo.setCurrentIndex(1 - window.person_combo.currentIndex())
        assert _wait_until(qapp, lambda: len(sent) == 2)
        _pump(qapp, 0.2)
        assert len(sent) == 2  # quiet again: no repeats
        assert no_dialogs == []
    finally:
        window.close()


def test_auto_send_waits_for_a_running_send_instead_of_piling_up(qapp, settings, monkeypatch, no_dialogs):
    gate, sent = threading.Event(), []

    def slow(document, host, port):
        sent.append(document)
        gate.wait(5)
        return _OK_REPLY

    window = _auto_window(settings, monkeypatch, slow)
    try:
        window.auto_send_checkbox.setChecked(True)
        assert _wait_until(qapp, lambda: len(sent) == 1)
        for k in range(4):
            window.person_combo.setCurrentIndex(k % 2)
            _pump(qapp, 0.1)
        assert len(sent) == 1  # only one connection at a time
        gate.set()
        assert _wait_until(qapp, lambda: len(sent) == 2)  # the latest state follows once it's free
        _pump(qapp, 0.3)
        assert len(sent) == 2
    finally:
        gate.set()
        window.close()


def test_auto_send_needs_a_person(qapp, settings, monkeypatch, no_dialogs):
    sent = []
    monkeypatch.setattr(mw, "send_pose_document", lambda *a, **k: sent.append(a) or _OK_REPLY)
    window = MainWindow(settings=settings)
    try:
        window.auto_send_debounce_spin.setValue(50)
        window.auto_send_checkbox.setChecked(True)
        window._queue_auto_send()
        assert not window._auto_send_timer.isActive()
        _pump(qapp, 0.2)
        assert sent == []
    finally:
        window.close()


def test_auto_send_pauses_when_blender_is_unreachable_and_a_manual_send_resumes(
    qapp, settings, monkeypatch, no_dialogs
):
    attempts = []

    def refused(document, host, port):
        attempts.append(document)
        raise ConnectionRefusedError("nothing listening")

    window = _auto_window(settings, monkeypatch, refused)
    try:
        window.auto_send_checkbox.setChecked(True)
        assert _wait_until(qapp, lambda: "Auto-send paused" in window.statusBar().currentMessage())
        assert no_dialogs == []  # automatic sends never pop up a dialog
        for k in range(3):
            window.person_combo.setCurrentIndex(k % 2)
        _pump(qapp, 0.3)
        assert len(attempts) == 1 and not window._auto_send_timer.isActive()

        monkeypatch.setattr(mw, "send_pose_document", lambda d, host, port: attempts.append(d) or _OK_REPLY)
        window._on_send_to_blender()  # a manual send resumes it
        assert _wait_until(qapp, lambda: "Applied" in window.statusBar().currentMessage())
        window.person_combo.setCurrentIndex(1 - window.person_combo.currentIndex())
        assert _wait_until(qapp, lambda: len(attempts) == 3)
    finally:
        window.close()


def test_closing_stops_a_pending_auto_send(qapp, settings, monkeypatch, no_dialogs):
    sent = []
    window = _auto_window(settings, monkeypatch, lambda d, host, port: sent.append(d) or _OK_REPLY)
    window.auto_send_debounce_spin.setValue(300)
    window.auto_send_checkbox.setChecked(True)
    assert window._auto_send_timer.isActive()
    window.close()
    assert not window._auto_send_timer.isActive()
    _pump(qapp, 0.5)
    assert sent == []


# --- F10: photo export, all people as BVH ----------------------------------------------------------------


def test_export_all_people_as_bvh_writes_one_file_each(qapp, settings, tmp_path, monkeypatch, no_dialogs):
    window = _window_with_photo(settings, [_person(), _person(_articulated_pose(), x_offset=0.3)])
    try:
        window._source_path = str(tmp_path / "group.jpg")
        folder = tmp_path / "out"
        folder.mkdir()
        monkeypatch.setattr(mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
        assert window.export_all_bvh_btn.isEnabled()
        window.export_all_bvh_btn.click()
        assert sorted(p.name for p in folder.iterdir()) == ["group_person1.bvh", "group_person2.bvh"]
        assert all((folder / n).read_text(encoding="utf-8").startswith("HIERARCHY") for n in
                   ("group_person1.bvh", "group_person2.bvh"))
        asked = []
        monkeypatch.setattr(mw.QMessageBox, "question", lambda *a, **k: asked.append(a) or mw.QMessageBox.No)
        before = (folder / "group_person1.bvh").stat().st_mtime_ns
        window.export_all_bvh_btn.click()
        assert asked and (folder / "group_person1.bvh").stat().st_mtime_ns == before
    finally:
        window.close()


# --- F5: hover highlighting on the photo -----------------------------------------------------------


def _hover_view(qapp):
    view = photo_view.PhotoView()
    view.resize(400, 400)
    pixmap = QPixmap(400, 400)
    pixmap.fill(QColor(20, 20, 20))
    view.set_source_pixmap(pixmap)
    people = []
    for k, (landmarks, offset) in enumerate(((_standing_t_pose(), -0.2), (_articulated_pose(), 0.2))):
        person = _person(landmarks, x_offset=offset)
        people.append(photo_view.PhotoOverlayPerson(
            joints_2d=sk.image_joint_positions_2d(person.image_landmarks_xy),
            confidence={n: 1.0 for n in sk.BONE_NAMES},
            selected=(k == 0),
            label=f"Person {k + 7}",
        ))
    view.set_skeleton_overlay(people)
    return view, people


def _widget_point(view, xy) -> QPointF:
    pixmap, ox, oy = view._displayed_pixmap_geometry()
    return QPointF(ox + xy[0] * pixmap.width(), oy + xy[1] * pixmap.height())


def _move(view, pos: QPointF) -> None:
    view.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, pos, view.mapToGlobal(pos), Qt.NoButton, Qt.NoButton,
                                    Qt.NoModifier))


def test_hovering_a_bone_highlights_and_names_it(qapp):
    view, people = _hover_view(qapp)
    joints = people[1].joints_2d
    middle = (np.asarray(joints["shin.L.head"]) + np.asarray(joints["shin.L.tail"])) / 2
    pos = _widget_point(view, middle)
    _move(view, pos)
    assert view.hover_info() == (1, "shin.L", None)
    assert view.toolTip() == "Person 8: shin.L"  # the label the window passes (a video's track ID)
    qapp.processEvents()
    grabbed = view.grab()
    scale = grabbed.devicePixelRatio()
    color = grabbed.toImage().pixelColor(round(pos.x() * scale), round(pos.y() * scale))
    assert (color.red(), color.green(), color.blue()) == (
        photo_view.HOVER_COLOR.red(), photo_view.HOVER_COLOR.green(), photo_view.HOVER_COLOR.blue()
    ), f"hover pixel differs at device scale {scale} (grab size {grabbed.size()}, widget size {view.size()})"

    # A redraw while the mouse rests keeps the highlight on what's under it.
    view.set_skeleton_overlay(list(people))
    assert view.hover_info() == (1, "shin.L", None)

    joint = _widget_point(view, joints["forearm.R.tail"])
    _move(view, joint)
    person, _bone, joint_key = view.hover_info()
    assert person == 1 and joint_key in {"forearm.R.tail", "hand.R.head"}  # the wrist, by either name
    assert view.toolTip() == f"Person 8: {joint_key}"

    _move(view, QPointF(2, 2))  # empty corner
    assert view.hover_info() == (None, None, None) and view.toolTip() == ""
    _move(view, pos)
    view.leaveEvent(QEvent(QEvent.Leave))
    assert view.hover_info() == (None, None, None) and view.toolTip() == ""
    view.grab()  # paints without a hover too


def test_hover_tolerates_a_bone_without_its_tail(qapp):
    view, people = _hover_view(qapp)
    joints = dict(people[0].joints_2d)
    del joints["hand.L.tail"]
    view.set_skeleton_overlay([photo_view.PhotoOverlayPerson(joints, {}, True)])
    _move(view, _widget_point(view, joints["hand.L.head"]))
    assert view.hover_info()[0] == 0
    assert view.toolTip().startswith("Person 1:")  # no label: named by position


def test_the_window_labels_people_for_the_hover_tooltip(qapp, settings, tmp_path, no_dialogs):
    window = _video_window(settings, tmp_path, 0)
    try:
        window.frame_slider.setValue(1)  # only track 1, shown first
        assert [p.label for p in window.photo_view._overlay_people] == ["Person 2"]
    finally:
        window.close()


# --- F8: focus cue, accessible names, unique mnemonics ------------------------------------


def _named(widget, buddies) -> bool:
    if widget.accessibleName():
        return True
    if isinstance(widget, QAbstractButton) and widget.text().replace("&", "").strip():
        return True
    return widget in buddies


def test_every_control_has_an_accessible_name_and_a_focus_cue(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        assert ":focus" in window.styleSheet()
        buddies = {label.buddy() for label in window.findChildren(QLabel) if label.buddy() is not None}
        controls = [
            w for kind in (QAbstractButton, QComboBox, QAbstractSpinBox, QSlider, QLineEdit)
            for w in window.findChildren(kind)
            if not isinstance(w.parent(), (QAbstractSpinBox, QComboBox))  # their internal line edits
            and not w.objectName().startswith("qt_")  # Qt's own (the menu bar's overflow button)
        ]
        unnamed = [w.objectName() or type(w).__name__ + ":" + getattr(w, "text", lambda: "")()
                   for w in controls if not _named(w, buddies)]
        assert unnamed == []
        for widget in (window.send_animation_btn, window.auto_send_checkbox, window.auto_send_debounce_spin,
                       window.video_rotation_smoothing_spin, window.export_all_bvh_btn):
            assert widget.accessibleName()
    finally:
        window.close()


def test_new_mnemonics_dont_collide_in_the_window(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        keys = {}
        for widget in window.findChildren(QLabel) + window.findChildren(QAbstractButton):
            text = widget.text()
            index = text.find("&")
            if index >= 0 and index + 1 < len(text) and text[index + 1] != "&":
                keys.setdefault(text[index + 1].lower(), []).append(text)
        for text in ("Rotation smoo&thing:", "A&uto-send on change", "&Debounce:"):
            assert keys[text[text.index("&") + 1].lower()] == [text], keys
    finally:
        window.close()

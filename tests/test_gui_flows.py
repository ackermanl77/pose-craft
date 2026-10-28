"""GUI flows on a real Qt event loop (T3): the worker threads, busy states,
cancel, warm-up, settings, drag-and-drop, manual correction and the export
paths -- the concurrency guarantees that used to be checked only by one-off
scripts. Runs offscreen (QT_QPA_PLATFORM=offscreen) with fake estimators; no
model or GPU needed. Uses plain QApplication.processEvents polling rather
than pytest-qt, so it needs nothing beyond PySide6."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QCloseEvent, QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pose_craft.estimation.landmarker import DetectedPerson, DownloadCancelled  # noqa: E402
from pose_craft.export.blender_pose import build_pose_document  # noqa: E402
from pose_craft.gui import main_window as mw  # noqa: E402
from pose_craft.gui import photo_view, viewport_3d  # noqa: E402
from pose_craft.gui.main_window import MainWindow  # noqa: E402
from pose_craft.retarget import skeleton as sk  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_retarget_frames import _articulated_pose, _standing_t_pose  # noqa: E402


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


def _wait_until(qapp, predicate, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return predicate()


def _person(landmarks=None, x_offset=0.0) -> DetectedPerson:
    landmarks = _standing_t_pose() if landmarks is None else landmarks
    xy = np.column_stack([0.5 + landmarks[:, 0] * 0.5 + x_offset, 0.5 - landmarks[:, 1] * 0.5])
    return DetectedPerson(landmarks=landmarks, image_landmarks_xy=xy, visibility=np.ones(sk.NUM_LANDMARKS))


class _FakeEstimator:
    device = "CPU"
    use_gpu = True
    max_people = 4
    group_photo = False

    def __init__(self, people=None, gate: threading.Event | None = None, variant="full"):
        self.people = people if people is not None else [_person()]
        self.gate = gate
        self.variant = variant
        self.calls: list[str] = []
        self.entered = threading.Event()

    def detect(self, image_bgr):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        self.calls.append("detect")
        return self.people

    def warm_up(self):
        if self.gate is not None:
            self.gate.wait(10)
        self.calls.append("warm_up")

    def close(self):
        self.calls.append("close")


def _use_estimator(monkeypatch, estimator) -> list:
    created = []

    def for_variant(cls, variant, max_people=4, use_gpu=True, on_progress=None, should_cancel=None):
        created.append(variant)
        return estimator

    monkeypatch.setattr(mw.PoseEstimator, "for_variant", classmethod(for_variant))
    return created


def _window_with_photo(settings, **kwargs) -> MainWindow:
    window = MainWindow(settings=settings, **kwargs)
    window._image_bgr = np.zeros((600, 600, 3), dtype=np.uint8)
    window.solve_btn.setEnabled(True)
    return window


# --- the solve worker -----------------------------------------------------------


def test_a_running_solve_disables_loading_and_shows_progress_then_restores(qapp, settings, monkeypatch, no_dialogs):
    gate = threading.Event()
    _use_estimator(monkeypatch, _FakeEstimator(gate=gate))
    window = _window_with_photo(settings)
    try:
        window._on_solve_pose()
        assert window._worker.isRunning()
        for widget in (window.load_btn, window.load_video_btn, window.solve_btn, window.model_variant_combo):
            assert not widget.isEnabled()
        assert not window.open_action.isEnabled() and not window.solve_action.isEnabled()
        assert not window.progress_bar.isHidden() and not window.cancel_btn.isHidden()

        window._on_solve_pose()  # a second click does nothing while one runs
        gate.set()
        assert _wait_until(qapp, lambda: window.person_combo.count() == 1)
        assert window.load_btn.isEnabled() and window.solve_btn.isEnabled()
        assert window.progress_bar.isHidden() and window.cancel_btn.isHidden()
        assert window.person_combo.itemText(0) == f"Person 1: {len(sk.BONE_NAMES)}/{len(sk.BONE_NAMES)} bones confident"
        assert window.export_btn.isEnabled() and window.send_to_blender_btn.isEnabled()
        assert no_dialogs == []
    finally:
        gate.set()
        window.close()


def test_cancel_stops_a_download_without_an_error_dialog(qapp, settings, monkeypatch, no_dialogs):
    started = threading.Event()

    def slow_download(cls, variant, max_people=4, use_gpu=True, on_progress=None, should_cancel=None):
        started.set()
        for _ in range(500):
            if should_cancel():
                raise DownloadCancelled("cancelled")
            on_progress(1, 100)
            time.sleep(0.01)
        raise AssertionError("cancel was never seen")

    monkeypatch.setattr(mw.PoseEstimator, "for_variant", classmethod(slow_download))
    window = _window_with_photo(settings)
    try:
        window._on_solve_pose()
        assert started.wait(5)
        assert _wait_until(qapp, lambda: window.progress_bar.maximum() == 100)  # determinate during a download
        window.cancel_btn.click()
        assert _wait_until(qapp, lambda: not window._worker.isRunning() and window.load_btn.isEnabled())
        assert window.statusBar().currentMessage() == "Cancelled"
        assert no_dialogs == []
    finally:
        window.close()


def test_close_waits_for_a_running_solve_before_closing_the_estimator(qapp, settings, monkeypatch):
    gate = threading.Event()
    estimator = _FakeEstimator(gate=gate)
    window = _window_with_photo(settings)
    window._estimator = estimator  # the cached estimator, as after a first solve
    window._on_solve_pose()
    assert estimator.entered.wait(5)  # the worker is inside detect()
    threading.Timer(0.3, gate.set).start()

    start = time.monotonic()
    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted()
    assert 0.2 < time.monotonic() - start < 5
    assert estimator.calls == ["detect", "close"]  # never closed while detect() was running


def test_send_to_blender_runs_off_the_ui_thread(qapp, settings, monkeypatch, no_dialogs):
    def slow_send(document, host, port):
        time.sleep(0.5)
        return {"ok": True, "applied": ["spine"], "requested": 15, "armature": "Rig"}

    monkeypatch.setattr(mw, "send_pose_document", slow_send)
    window = MainWindow(settings=settings)
    try:
        window._people = [_person()]
        window._fill_person_combo()
        start = time.monotonic()
        window._on_send_to_blender()
        assert time.monotonic() - start < 0.3
        assert not window.send_to_blender_btn.isEnabled()
        assert _wait_until(qapp, lambda: "Applied 1/15" in window.statusBar().currentMessage())
        assert window.send_to_blender_btn.isEnabled()
    finally:
        window.close()


# --- warm-up (P3) -------------------------------------------------------------------


def test_startup_warm_up_hands_its_estimator_to_the_first_solve(qapp, settings, monkeypatch, no_dialogs):
    estimator = _FakeEstimator()
    created = _use_estimator(monkeypatch, estimator)
    monkeypatch.setattr(mw, "is_model_available", lambda variant: True)
    window = _window_with_photo(settings, warm_up=True)
    try:
        assert _wait_until(qapp, lambda: window._estimator is estimator)
        assert estimator.calls == ["warm_up"]
        window._on_solve_pose()
        assert _wait_until(qapp, lambda: window.person_combo.count() == 1)
        assert created == ["full"]  # the solve reused the warmed-up estimator
    finally:
        window.close()


def test_solve_clicked_during_warm_up_runs_once_it_is_ready(qapp, settings, monkeypatch, no_dialogs):
    gate = threading.Event()
    estimator = _FakeEstimator(gate=gate)
    _use_estimator(monkeypatch, estimator)
    monkeypatch.setattr(mw, "is_model_available", lambda variant: True)
    window = _window_with_photo(settings, warm_up=True)
    try:
        window._on_solve_pose()
        assert window._pending_solve and not window.solve_btn.isEnabled()
        gate.set()
        assert _wait_until(qapp, lambda: window.person_combo.count() == 1)
        assert estimator.calls[:2] == ["warm_up", "detect"]
    finally:
        gate.set()
        window.close()


def test_no_warm_up_without_a_cached_model(qapp, settings, monkeypatch):
    monkeypatch.setattr(mw, "is_model_available", lambda variant: False)
    created = _use_estimator(monkeypatch, _FakeEstimator())
    window = MainWindow(settings=settings, warm_up=True)
    try:
        assert window._warmup_worker is None and created == []
    finally:
        window.close()


# --- settings, recent files, drag-and-drop (F7) -------------------------------------


def _write_photo(path: Path, size=(600, 800)) -> Path:
    import cv2

    cv2.imwrite(str(path), np.full((*size, 3), 90, dtype=np.uint8))
    return path


def test_settings_survive_a_restart(qapp, settings, tmp_path, no_dialogs):
    window = MainWindow(settings=settings)
    window.live_link_host_edit.setText("127.0.0.2")
    window.live_link_port_spin.setValue(9000)
    window.model_variant_combo.setCurrentText("heavy")
    window.use_gpu_checkbox.setChecked(False)
    window.group_photo_checkbox.setChecked(True)
    window.max_people_spin.setValue(6)
    window.load_photo(str(_write_photo(tmp_path / "first.png")))
    window.close()

    again = MainWindow(settings=settings)
    try:
        assert again.live_link_host_edit.text() == "127.0.0.2"
        assert again.live_link_port_spin.value() == 9000
        assert again.model_variant_combo.currentText() == "heavy"
        assert not again.use_gpu_checkbox.isChecked()
        assert again.group_photo_checkbox.isChecked() and again.max_people_spin.value() == 6
        assert again._last_dir() == str(tmp_path)
        assert [a.text() for a in again.recent_menu.actions()] == ["first.png"]
    finally:
        again.close()


def test_recent_files_are_unique_and_bounded():
    recent = []
    for i in range(12):
        recent = mw._update_recent_files(recent, f"C:/photos/p{i}.jpg")
    recent = mw._update_recent_files(recent, "C:/PHOTOS/p5.jpg")
    assert len(recent) == mw.MAX_RECENT_FILES
    assert Path(recent[0]).name == "p5.jpg"
    assert sum(1 for p in recent if Path(p).name.lower() == "p5.jpg") == 1


def test_dropping_a_photo_loads_it(qapp, settings, tmp_path, no_dialogs):
    photo = _write_photo(tmp_path / "dropped.jpg")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "notes.txt")), QUrl.fromLocalFile(str(photo))])
    window = MainWindow(settings=settings)
    try:
        path = window._dropped_path(mime)
        assert Path(path) == photo
        window.open_path(path)
        assert window._image_bgr is not None and window.solve_btn.isEnabled()
        assert window.acceptDrops()
    finally:
        window.close()


def test_menu_shortcuts(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        def keys(action):
            return {s.toString() for s in action.shortcuts()}

        assert keys(window.open_action) == {QKeySequence(QKeySequence.Open).toString()}
        assert {"F5", "Ctrl+Return"} <= keys(window.solve_action)
        assert keys(window.export_action) == {"Ctrl+S"}
        assert keys(window.export_bvh_action) == {"Ctrl+Shift+S"}
        assert keys(window.send_action) == {"Ctrl+B"}
        assert keys(window.mirror_action) == {"M"}
        assert keys(window.next_person_action) == {"PgDown"}
        window.mirror_action.trigger()
        assert window.mirror_checkbox.isChecked()
    finally:
        window.close()


def test_a_low_resolution_photo_shows_a_banner_not_a_modal_dialog(qapp, settings, tmp_path, no_dialogs):
    window = MainWindow(settings=settings)
    try:
        window.load_photo(str(_write_photo(tmp_path / "tiny.png", size=(120, 160))))
        assert not window.banner.isHidden() and window.banner_label.text()
        assert no_dialogs == []
        window.load_photo(str(_write_photo(tmp_path / "big.png")))
        assert window.banner.isHidden()
    finally:
        window.close()


# --- person picking (F5) -------------------------------------------------------------


def test_clicking_empty_background_keeps_the_selection_and_a_skeleton_click_picks(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        window._people = [_person(x_offset=-0.3), _person(x_offset=0.3)]
        window._fill_person_combo()
        window.person_combo.setCurrentIndex(0)
        right = sk.image_joint_positions_2d(window._people[1].image_landmarks_xy)
        hand = right["forearm.L.tail"]

        window._on_photo_clicked(0.5, 0.98)  # empty floor below everyone
        assert window.person_combo.currentIndex() == 0
        window._on_photo_clicked(float(hand[0]), float(hand[1]) + 0.01)
        assert window.person_combo.currentIndex() == 1
    finally:
        window.close()


def test_nearest_bone_measures_distance_to_the_segment():
    joints = {f"{name}.{end}": np.zeros(2) for name in sk.BONE_NAMES for end in ("head", "tail")}
    joints["forearm.L.head"], joints["forearm.L.tail"] = np.array([0.2, 0.5]), np.array([0.6, 0.5])
    bone, distance = mw._nearest_bone(joints, 0.4, 0.52, bones=("forearm.L", "spine"))
    assert bone == "forearm.L" and distance == pytest.approx(0.02)


# --- manual correction (F4) ------------------------------------------------------------


def _bent_person() -> DetectedPerson:
    return _person(_articulated_pose())


def test_flipping_a_limb_changes_only_that_limb_and_is_undoable(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        window._people = [_bent_person()]
        window._fill_person_combo()
        before, _ = window._current_person_document_inputs()

        window.flip_bone_depth(0, "forearm.L")
        after, _ = window._current_person_document_inputs()
        changed = {n for n in sk.BONE_NAMES if (after[n] * before[n].inv()).magnitude() > 1e-6}
        assert "forearm.L" in changed
        assert changed <= {"upper_arm.L", "forearm.L", "hand.L"}  # the limb, nothing else

        # The flip mirrors the forearm's depth about the elbow.
        lm = window._selected_landmarks()
        original = window._people[0].landmarks
        assert lm[sk.LEFT_WRIST][2] - lm[sk.LEFT_ELBOW][2] == pytest.approx(
            -(original[sk.LEFT_WRIST][2] - original[sk.LEFT_ELBOW][2])
        )

        window.undo_stack.undo()
        undone, _ = window._current_person_document_inputs()
        assert all((undone[n] * before[n].inv()).magnitude() < 1e-9 for n in sk.BONE_NAMES)
        window.undo_stack.redo()
        assert window._person_edits(0).flipped == {"forearm.L"}
    finally:
        window.close()


def test_a_bone_adjustment_rotates_it_about_its_own_axis_and_reaches_every_export(qapp, settings, tmp_path, monkeypatch):
    import json

    window = MainWindow(settings=settings)
    try:
        window._people = [_bent_person()]
        window._fill_person_combo()
        before, _ = window._current_person_document_inputs()
        window.adjust_bone_combo.setCurrentText("forearm.R")
        window.adjust_spins[1].setValue(30.0)  # twist 30 degrees about the bone's own Y (along it)

        after, _ = window._current_person_document_inputs()
        relative = before["forearm.R"].inv() * after["forearm.R"]
        expected = sk.bone_axes_rotation("forearm.R", (0.0, 30.0, 0.0))
        np.testing.assert_allclose(relative.as_rotvec(), expected.as_rotvec(), atol=1e-9)
        assert expected.magnitude() == pytest.approx(np.radians(30))

        out = tmp_path / "adjusted.json"
        monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._on_export_pose()
        document = json.loads(out.read_text(encoding="utf-8"))
        exported = document["bones"]["forearm.R"]["quaternion_wxyz"]
        assert exported == pytest.approx(build_pose_document(after)["bones"]["forearm.R"]["quaternion_wxyz"])
        assert exported != pytest.approx(build_pose_document(before)["bones"]["forearm.R"]["quaternion_wxyz"])
        assert "root" in document and document["root"]["hip_height"] > 0

        window.reset_bone_btn.click()
        assert window._person_edits(0).is_empty()
        assert window.adjust_spins[1].value() == 0.0
        window.undo_stack.undo()  # back to 30 degrees
        assert window.adjust_spins[1].value() == pytest.approx(30.0)
    finally:
        window.close()


def test_edits_are_cleared_by_a_new_solve(qapp, settings, no_dialogs):
    window = MainWindow(settings=settings)
    try:
        window._people = [_bent_person()]
        window._fill_person_combo()
        window.flip_bone_depth(0, "shin.L")
        window._on_solve_finished([_bent_person()], _FakeEstimator())
        assert window._edits == {} and window.undo_stack.count() == 0
    finally:
        window.close()


# --- exports ---------------------------------------------------------------------------


def test_default_export_names_use_the_photo_stem():
    assert mw.default_export_name("C:/shoot/IMG_0042.jpg", 1, ".json") == "IMG_0042_person2.json"
    assert mw.default_export_name(None, 0, ".bvh") == "pose_person1.bvh"


def test_export_all_asks_before_overwriting(qapp, settings, tmp_path, monkeypatch):
    window = MainWindow(settings=settings)
    try:
        window._people = [_person(), _person()]
        window._fill_person_combo()
        existing = tmp_path / "pose_person1.json"
        existing.write_text("keep me", encoding="utf-8")
        monkeypatch.setattr(mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
        asked = []
        monkeypatch.setattr(mw.QMessageBox, "question", lambda *a, **k: asked.append(a) or mw.QMessageBox.No)
        window._on_export_all()
        assert asked and existing.read_text(encoding="utf-8") == "keep me"
        assert not (tmp_path / "pose_person2.json").exists()
    finally:
        window.close()


def test_exported_bvh_stands_on_the_floor(qapp, settings, tmp_path, monkeypatch):
    window = MainWindow(settings=settings)
    try:
        window._people = [_bent_person()]
        window._fill_person_combo()
        out = tmp_path / "pose.bvh"
        monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._on_export_bvh()
        rotations, _ = window._current_person_document_inputs()
        root_y = float(out.read_text(encoding="utf-8").strip().splitlines()[-1].split()[1])
        assert root_y == pytest.approx(sk.hip_height_above_floor(rotations), abs=1e-5)
    finally:
        window.close()


# --- video (B12) ------------------------------------------------------------------------


def test_a_tracked_video_can_be_scrubbed_and_exported_as_an_animation(qapp, settings, tmp_path, monkeypatch, no_dialogs):
    import cv2

    from pose_craft.estimation.video import VideoFrame, VideoSolve

    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
    for k in range(4):
        writer.write(np.full((48, 64, 3), 40 * k, dtype=np.uint8))
    writer.release()
    frames = [VideoFrame(0, _person()), VideoFrame(1, None), VideoFrame(2, _bent_person()), VideoFrame(3, _person())]
    result = VideoSolve(str(clip), 24.0, (64, 48), frames)

    window = MainWindow(settings=settings)
    try:
        window._on_video_finished(result, _FakeEstimator())
        assert not window.video_group.isHidden()
        assert window.frame_slider.maximum() == 3
        assert window.person_combo.count() == 1 and window.export_btn.isEnabled()
        window.frame_slider.setValue(1)  # nobody tracked in this frame
        assert window.person_combo.count() == 0 and not window.export_btn.isEnabled()
        window.frame_slider.setValue(2)
        assert window.person_combo.count() == 1

        out = tmp_path / "anim.bvh"
        monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._on_export_animation()
        text = out.read_text(encoding="utf-8")
        assert "Frames: 3" in text and "Frame Time: 0.041667" in text
    finally:
        window.close()


# --- legend and viewport (F3, F9) -------------------------------------------------------


def test_low_confidence_bones_are_dashed_not_just_recoloured(qapp):
    color, width, style = photo_view.bone_pen_style(True, 0.1)
    assert style == Qt.DashLine and color == photo_view.SELECTED_LOW_CONFIDENCE_COLOR
    assert photo_view.bone_pen_style(True, 0.9)[2] == Qt.SolidLine
    assert photo_view.bone_pen_style(False, 0.1)[0] == photo_view.UNSELECTED_COLOR
    entries = photo_view.legend_entries()
    assert (photo_view.SELECTED_LOW_CONFIDENCE_COLOR, Qt.DashLine) in [(c, s) for c, s, _t in entries]
    # Joints are neutral, so nothing drawn on the photo matches the warning colour except low-confidence bones.
    assert photo_view.SELECTED_JOINT_COLOR != photo_view.SELECTED_LOW_CONFIDENCE_COLOR


def test_viewport_dashes_low_confidence_bones():
    points = sk.joint_positions(_standing_t_pose())
    confidence = {name: 1.0 for name in sk.BONE_NAMES}
    confidence["shin.L"] = 0.1
    vertices, colors = viewport_3d._bone_segments(points, confidence)
    assert len(vertices) == 2 * (len(sk.BONE_NAMES) - 1) + 2 * viewport_3d.DASHES_PER_BONE
    low = np.all(np.isclose(colors, viewport_3d.LOW_CONFIDENCE_BONE_COLOR), axis=1)
    assert low.sum() == 2 * viewport_3d.DASHES_PER_BONE


def test_octahedral_bone_turns_with_the_bones_roll():
    vertices, faces = viewport_3d.octahedron_mesh(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert vertices.shape == (6, 3) and faces.shape == (8, 3)
    np.testing.assert_allclose(vertices[1], [0.1, 0.1, 0.0])  # the waist vertex sits along the roll axis
    rolled, _ = viewport_3d.octahedron_mesh(np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(rolled[1], [0.0, 0.1, 0.1])


def test_camera_presets_and_labels(qapp, settings):
    window = MainWindow(settings=settings)
    try:
        window._people = [_person()]
        window._fill_person_combo()
        window.viewport.set_camera_preset("top")
        assert window.viewport.opts["elevation"] == 90
        window.viewport.set_camera_preset("front")
        assert (window.viewport.opts["elevation"], window.viewport.opts["azimuth"]) == (0, -90)
        window.labels_action.trigger()
        assert len(window.viewport._labels) == len(sk.BONE_NAMES)
        window.labels_action.trigger()
        assert window.viewport._labels == []
    finally:
        window.close()


def test_the_controls_panel_never_cuts_its_content_off(qapp, settings):
    """UI1: the left panel's width was taken once while building the window (without the scrollbar),
    and a combo that grew for a long person label pushed the content past it."""
    from PySide6.QtWidgets import QScrollArea

    window = MainWindow(settings=settings)
    try:
        window.resize(1600, 950)
        window.show()
        window.person_combo.addItem("Person 1: 11/15 bones confident, 4 low-confidence, 2 out of frame")
        _wait_until(qapp, lambda: False, timeout=0.3)
        scroll = window.findChildren(QScrollArea)[0]
        assert scroll.viewport().width() >= scroll.widget().minimumSizeHint().width()
    finally:
        window.close()


def test_the_main_actions_stay_in_view_and_solve_stands_out(qapp, settings):
    """UI2: a toolbar keeps load, solve, export and send in view whatever the panel's scroll
    position; it shares the menus' actions (so their enabled state), and Solve is styled as the
    primary action there and in the panel."""
    window = MainWindow(settings=settings)
    try:
        window.resize(1100, 700)
        window.show()
        toolbar = window.main_toolbar
        assert toolbar.isVisible() and not toolbar.isMovable()
        assert [a for a in toolbar.actions() if not a.isSeparator()] == [
            window.open_action, window.open_video_action, window.solve_action,
            window.export_action, window.export_bvh_action, window.send_action]
        assert toolbar.widgetForAction(window.solve_action).objectName() == "primaryAction"
        assert window.solve_btn.objectName() == "primaryAction"
        # Nothing solved yet: solve, export and send are off in the toolbar too.
        assert not window.solve_action.isEnabled() and not window.export_action.isEnabled()
        assert not toolbar.widgetForAction(window.send_action).isEnabled()
        window._set_result_actions_enabled(True)
        assert toolbar.widgetForAction(window.export_bvh_action).isEnabled()
    finally:
        window.close()


def test_the_3d_preview_has_navigation_and_an_axis_gizmo_that_matches_the_camera(qapp, settings):
    """UI3: view presets, fit and zoom buttons over the preview, and a corner gizmo whose axes point
    where the camera's own projection puts them."""
    import numpy as np

    from pose_craft.gui import viewport_3d

    window = MainWindow(settings=settings)
    try:
        window.resize(1400, 800)
        window.show()
        qapp.processEvents()
        viewport = window.viewport
        nav = viewport.navigation
        assert nav.isVisible() and viewport.axis_gizmo.isVisible()
        assert nav.geometry().right() <= viewport.width() and viewport.axis_gizmo.geometry().bottom() <= viewport.height()
        display_to_engine = viewport_3d._ENGINE_TO_DISPLAY  # a rotation: its transpose is its inverse
        for preset in ("front", "side", "top", "default"):
            nav.buttons[preset].click()
            elevation, azimuth = viewport_3d.CAMERA_PRESETS[preset]
            assert (viewport.opts["elevation"], viewport.opts["azimuth"]) == (elevation, azimuth)
            centre = np.array([viewport.opts["center"].x(), viewport.opts["center"].y(), viewport.opts["center"].z()])
            points_display = np.vstack([centre, centre + 0.3 * np.eye(3)])
            screen = viewport.project_to_screen(points_display @ display_to_engine)
            for index, (name, (dx, dy, _depth)) in enumerate(viewport_3d.axis_gizmo_directions(elevation, azimuth).items()):
                drawn = screen[index + 1] - screen[0]
                if np.hypot(dx, dy) < 0.2:  # pointing (almost) at the camera: no direction to compare
                    continue
                cosine = (drawn @ np.array([dx, dy])) / (np.linalg.norm(drawn) * np.hypot(dx, dy))
                assert cosine > 0.95, (preset, name, drawn, (dx, dy))
        before = viewport.opts["distance"]
        nav.buttons["zoom_in"].click()
        assert viewport.opts["distance"] == pytest.approx(before / viewport_3d.ZOOM_STEP)
        nav.buttons["zoom_out"].click()
        nav.buttons["zoom_out"].click()
        assert viewport.opts["distance"] == pytest.approx(before * viewport_3d.ZOOM_STEP)
    finally:
        window.close()

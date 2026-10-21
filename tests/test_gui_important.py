"""GUI flows for this round's items, on an offscreen Qt event loop with fake
estimators: the Hands and fingers option (B11), video tracks (B12), dragging
2D joints (F4), and the 3D view's hover label and photo backdrop (F9).
Widgets are driven through their own methods -- no synthetic input events."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pose_craft.estimation.landmarker import DetectedPerson, HandLandmarks  # noqa: E402
from pose_craft.gui import main_window as mw  # noqa: E402
from pose_craft.gui import viewport_3d  # noqa: E402
from pose_craft.gui.main_window import MainWindow  # noqa: E402
from pose_craft.retarget import skeleton as sk  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_gui_flows import _FakeEstimator, _person, _use_estimator, _wait_until  # noqa: E402
from test_hands import _curl, _rest_hand  # noqa: E402
from test_retarget_frames import _articulated_pose  # noqa: E402


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


def _person_with_hands() -> DetectedPerson:
    person = _person(_articulated_pose())
    hands = {}
    for side, wrist in (("L", sk.LEFT_WRIST), ("R", sk.RIGHT_WRIST)):
        world = _curl(_rest_hand(person.landmarks[wrist], side), "f_index", side, 40.0)
        xy = np.column_stack([0.5 + world[:, 0] * 0.5, 0.5 - world[:, 1] * 0.5])
        hands[side] = HandLandmarks(world, xy, 0.95)
    person.hands = hands
    return person


def _window_with_photo(settings, people) -> MainWindow:
    window = MainWindow(settings=settings)
    image = np.full((600, 600, 3), 90, dtype=np.uint8)
    window._image_bgr = image
    window.photo_view.resize(400, 400)
    window._set_photo_preview(image)
    window._people = people
    window._person_keys = list(range(len(people)))
    window._fill_person_combo()
    window._update_preview_for_selected_person()
    return window


# --- B11: the Hands and fingers option -------------------------------------------------


class _HandsEstimator(_FakeEstimator):
    def __init__(self, **kwargs):
        super().__init__(people=[_person_with_hands()], **kwargs)
        self.enabled = 0

    def enable_hands(self, on_progress=None, should_cancel=None):
        self.enabled += 1


def test_hands_are_off_by_default_and_the_option_reaches_the_solve(qapp, settings, monkeypatch, no_dialogs):
    estimator = _HandsEstimator()
    _use_estimator(monkeypatch, estimator)
    window = MainWindow(settings=settings)
    try:
        assert not window.hands_checkbox.isChecked()
        window._image_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        window._on_solve_pose()
        assert _wait_until(qapp, lambda: window.person_combo.count() == 1)
        assert estimator.enabled == 0 and estimator.hands is False

        window.hands_checkbox.setChecked(True)
        window._on_solve_pose()
        assert _wait_until(qapp, lambda: not window._worker.isRunning() and window.person_combo.count() == 1)
        assert estimator.enabled == 1 and estimator.hands is True
        assert "2 hand(s) found" in window.statusBar().currentMessage()
        window._save_settings()
        assert MainWindow(settings=settings).hands_checkbox.isChecked()  # remembered
    finally:
        window.close()


def test_finger_bones_reach_the_exports_only_when_hands_were_found(qapp, settings, tmp_path, monkeypatch):
    window = _window_with_photo(settings, [_person_with_hands(), _person(_articulated_pose(), x_offset=0.3)])
    try:
        out = tmp_path / "hands.json"
        bvh_out = tmp_path / "hands.bvh"
        monkeypatch.setattr(
            mw.QFileDialog, "getSaveFileName", lambda _p, _t, name, *a, **k: (str(bvh_out if name.endswith(".bvh") else out), "")
        )
        window._on_export_pose()
        bones = json.loads(out.read_text(encoding="utf-8"))["bones"]
        assert set(bones) == set(sk.ALL_BONE_NAMES)
        assert bones["f_index.02.L"]["confidence"] == pytest.approx(0.95)

        window._on_export_bvh()
        assert "JOINT f_pinky.03.R" in bvh_out.read_text(encoding="utf-8")

        window.person_combo.setCurrentIndex(1)  # the person without hands: body only, as before
        window._on_export_pose()
        assert set(json.loads(out.read_text(encoding="utf-8"))["bones"]) == set(sk.BONE_NAMES)
        window._on_export_bvh()
        assert "f_" not in bvh_out.read_text(encoding="utf-8")

        # The preview draws the fingers, and mirroring swaps the hands with the body.
        window.person_combo.setCurrentIndex(0)
        assert "f_index.03.L.tail" in window.viewport._joint_positions
        window.mirror_checkbox.setChecked(True)
        rotations, _confidence = window._current_person_document_inputs()
        unmirrored = sk.solve_bone_rotations(
            sk.mirror_landmarks(window._people[0].landmarks),
            hands=sk.mirror_hands(window._people[0].hand_world()),
        )
        assert (rotations["f_index.01.R"] * unmirrored["f_index.01.R"].inv()).magnitude() < 1e-9
    finally:
        window.close()


# --- F4: dragging a joint on the photo ---------------------------------------------------------


def test_move_joint_keeps_the_bone_length_and_its_depth_side():
    lm = _articulated_pose()
    xy = np.column_stack([0.5 + lm[:, 0] * 0.5, 0.5 - lm[:, 1] * 0.5])
    target = xy[sk.LEFT_ELBOW] - 0.4 * (xy[sk.LEFT_ELBOW] - xy[sk.LEFT_SHOULDER]) + [0.0, 0.01]
    moved, moved_xy, _hands, _hands_xy = sk.move_joint(lm, xy, sk.LEFT_ELBOW, target)
    np.testing.assert_allclose(moved_xy[sk.LEFT_ELBOW], target)
    before = np.linalg.norm(lm[sk.LEFT_ELBOW] - lm[sk.LEFT_SHOULDER])
    after = np.linalg.norm(moved[sk.LEFT_ELBOW] - moved[sk.LEFT_SHOULDER])
    assert after == pytest.approx(before)
    assert np.sign(moved[sk.LEFT_ELBOW][2] - lm[sk.LEFT_SHOULDER][2]) == np.sign(lm[sk.LEFT_ELBOW][2] - lm[sk.LEFT_SHOULDER][2]) or abs(lm[sk.LEFT_ELBOW][2] - lm[sk.LEFT_SHOULDER][2]) < 1e-9
    # The image direction of the upper arm now points at the dragged position.
    image_dir = target - xy[sk.LEFT_SHOULDER]
    world_dir = (moved[sk.LEFT_ELBOW] - moved[sk.LEFT_SHOULDER])[:2] * [1, -1]
    assert np.dot(image_dir, world_dir) / np.linalg.norm(image_dir) / np.linalg.norm(world_dir) > 0.999
    # Everything else stays; a wrist drag carries the hand landmarks along.
    untouched = [i for i in range(33) if i != sk.LEFT_ELBOW]
    np.testing.assert_allclose(moved[untouched], lm[untouched])
    hand = {"L": _rest_hand(lm[sk.LEFT_WRIST], "L")}
    moved, _xy, hands, _hxy = sk.move_joint(lm, xy, sk.LEFT_WRIST, xy[sk.LEFT_WRIST] + [0.0, 0.02], hands=hand)
    shift = moved[sk.LEFT_WRIST] - lm[sk.LEFT_WRIST]
    np.testing.assert_allclose(moved[sk.LEFT_INDEX] - lm[sk.LEFT_INDEX], shift)
    np.testing.assert_allclose(hands["L"] - hand["L"], np.broadcast_to(shift, (21, 3)))
    # Dragged farther than the bone reaches: it lies in the image plane, pointing at the drag.
    far = xy[sk.LEFT_KNEE] + [0.3, 0.0]
    moved, _xy, _h, _hxy = sk.move_joint(lm, xy, sk.LEFT_KNEE, far)
    assert moved[sk.LEFT_KNEE][2] == pytest.approx(lm[sk.LEFT_HIP][2])
    with pytest.raises(ValueError):
        sk.move_joint(lm, xy, sk.NOSE, (0.5, 0.5))


def _widget_pos(window, xy) -> QPointF:
    pixmap, ox, oy = window.photo_view._displayed_pixmap_geometry()
    return QPointF(ox + xy[0] * pixmap.width(), oy + xy[1] * pixmap.height())


def test_dragging_a_joint_resolves_the_limb_and_is_undoable(qapp, settings):
    window = _window_with_photo(settings, [_person(_articulated_pose())])
    try:
        handles = window.photo_view.drag_handles()
        assert set(handles) == set(sk.DRAGGABLE_JOINTS)
        before, _ = window._current_person_document_inputs()
        dragged = []
        window.photo_view.joint_dragged.connect(lambda *args: dragged.append(args))

        start = _widget_pos(window, handles[sk.RIGHT_ELBOW])
        assert window.photo_view.begin_drag(start)
        window.photo_view.drag_to(start + QPointF(25, 30))
        assert window.photo_view.end_drag(start + QPointF(25, 30))
        assert dragged and dragged[0][0] == sk.RIGHT_ELBOW
        edits = window._person_edits(0)
        assert dict(edits.moved)[sk.RIGHT_ELBOW] == pytest.approx(dragged[0][1:])

        after, _ = window._current_person_document_inputs()
        changed = {n for n in sk.BONE_NAMES if (after[n] * before[n].inv()).magnitude() > 1e-6}
        assert "upper_arm.R" in changed and changed <= {"upper_arm.R", "forearm.R", "hand.R"}
        np.testing.assert_allclose(window.photo_view.drag_handles()[sk.RIGHT_ELBOW], dragged[0][1:])
        assert window.undo_stack.undoText() == "Move right elbow"

        window.undo_stack.undo()
        undone, _ = window._current_person_document_inputs()
        assert all((undone[n] * before[n].inv()).magnitude() < 1e-9 for n in sk.BONE_NAMES)
        window.undo_stack.redo()
        window.reset_edits_btn.click()
        assert window._person_edits(0).is_empty()

        # A press that barely moves is a click, not a drag.
        assert window.photo_view.begin_drag(start)
        assert not window.photo_view.end_drag(start + QPointF(1, 1))
        assert len(dragged) == 1
        assert window.photo_view.handle_at(QPointF(0, 0)) is None
    finally:
        window.close()


# --- B12: video tracks in the window ------------------------------------------------------------


def _two_track_video(tmp_path):
    import cv2

    from pose_craft.estimation.video import VideoFrame, VideoSolve

    clip = tmp_path / "duo.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
    for k in range(4):
        writer.write(np.full((48, 64, 3), 40 * k, dtype=np.uint8))
    writer.release()
    a, b = _person(), _person(_articulated_pose(), x_offset=0.3)
    frames = [
        VideoFrame(0, people={0: a, 1: b}),
        VideoFrame(1, people={1: b}),
        VideoFrame(2, people={0: a, 1: b}),
        VideoFrame(3, people={0: a, 1: b}),
    ]
    return VideoSolve(str(clip), 24.0, (64, 48), frames)


def test_video_tracks_keep_the_selected_person_and_their_edits(qapp, settings, tmp_path, monkeypatch, no_dialogs):
    window = MainWindow(settings=settings)
    try:
        window._on_video_finished(_two_track_video(tmp_path), _FakeEstimator())
        assert window.person_combo.count() == 2
        assert window.person_combo.itemText(1).startswith("Person 2:")
        assert window.photo_view.drag_handles() == {}  # no joint dragging on video frames
        window.person_combo.setCurrentIndex(1)  # track 1
        window.flip_bone_depth(1, "forearm.L")
        assert set(window._edits) == {1}

        window.frame_slider.setValue(1)  # only track 1 here: still selected, at combo index 0
        assert window.person_combo.count() == 1 and window._person_keys == [1]
        assert window._person_edits(0).flipped == {"forearm.L"}
        window.frame_slider.setValue(2)
        assert window.person_combo.currentIndex() == 1

        out = tmp_path / "one.bvh"
        monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._on_export_animation()
        assert "Frames: 4" in out.read_text(encoding="utf-8")  # track 1 is in every frame
        assert "Person 2" in window.statusBar().currentMessage()

        folder = tmp_path / "all"
        folder.mkdir()
        monkeypatch.setattr(mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
        window._on_export_all_tracks()
        files = sorted(p.name for p in folder.iterdir())
        assert files == ["duo_person1.bvh", "duo_person2.bvh"]
        assert "Frames: 3" in (folder / "duo_person1.bvh").read_text(encoding="utf-8")
    finally:
        window.close()


def test_video_worker_tracks_up_to_max_people(qapp, settings, monkeypatch):
    seen = {}

    def fake_solve_video(path, estimator, **kwargs):
        seen.update(kwargs)
        raise InterruptedError

    import pose_craft.estimation.video as video_module

    monkeypatch.setattr(video_module, "solve_video", fake_solve_video)
    worker = mw.VideoSolveWorker("x.mp4", _FakeEstimator(), "full", False, max_people=3)
    done = threading.Event()
    worker.cancelled.connect(done.set)
    worker.run()
    assert seen["max_people"] == 3


# --- F9: hover label and photo backdrop -----------------------------------------------------------


def test_hovering_a_bone_shows_its_name_and_local_euler_angles(qapp):
    view = viewport_3d.SkeletonViewport()
    view.resize(400, 400)
    try:
        landmarks = _articulated_pose()
        rotations = sk.solve_bone_rotations(landmarks)
        points = sk.forward_kinematics(rotations, *sk.measured_proportions(landmarks))
        confidence = {name: 0.9 for name in sk.BONE_NAMES}
        view.show_skeleton(points, confidence=confidence, rotations=rotations)
        middle = (points["thigh.L.head"] + points["thigh.L.tail"]) / 2
        x, y = view.project_to_screen(middle[None])[0]
        assert 0 <= x <= 400 and 0 <= y <= 400
        assert view.hover_at(x, y) == "thigh.L"
        text = view.hover_label_text()
        ex, ey, ez = sk.bone_local_euler_deg("thigh.L", rotations["thigh.L"])
        assert text.startswith("thigh.L") and f"X {ex:+.1f}°" in text and f"Z {ez:+.1f}°" in text
        assert "confidence 90%" in text
        assert view.hover_at(1, 1) is None and view.hover_label_text() == ""
        view.hover_enabled = False
        assert view.hover_at(x, y) is None
    finally:
        view.close()


def test_euler_readout_inverts_the_manual_bone_adjustment():
    angles = (12.0, -30.0, 45.0)
    for bone in ("forearm.R", "thigh.L", "f_ring.02.L", "thumb.01.R"):
        x, y, z = sk.bone_local_euler_deg(bone, sk.bone_axes_rotation(bone, angles))
        assert (x, y, z) == pytest.approx(angles, abs=1e-9)


def test_the_backdrop_lines_the_photo_up_with_the_landmarks():
    landmarks = _articulated_pose()
    aspect = 4 / 3
    scale = 0.6
    xy = np.column_stack([0.5 + landmarks[:, 0] * scale / aspect, 0.4 - landmarks[:, 1] * scale])
    corners = viewport_3d.photo_backdrop_corners(landmarks, xy, aspect)
    tl, tr, _br, bl = corners
    for index in (sk.LEFT_WRIST, sk.RIGHT_ANKLE, sk.NOSE):
        point = tl + xy[index, 0] * (tr - tl) + xy[index, 1] * (bl - tl)
        np.testing.assert_allclose(point[:2], landmarks[index, :2], atol=1e-9)
    assert np.all(corners[:, 2] < landmarks[:, 2].min())  # behind the person
    transform = viewport_3d.backdrop_transform(corners, 320, 240)
    mapped = transform.map(QPointF(320, 240))
    np.testing.assert_allclose([mapped.x(), mapped.y()], viewport_3d._engine_to_display(corners[2:3])[0, :2], atol=1e-4)


def test_the_backdrop_toggle_adds_and_removes_the_photo_plane(qapp, settings):
    window = _window_with_photo(settings, [_person(_articulated_pose())])
    try:
        assert window.viewport._backdrop is None
        window.backdrop_action.setChecked(True)
        window._on_toggle_backdrop(True)
        assert window.viewport._backdrop is not None
        window.mirror_checkbox.setChecked(True)  # re-placed for the mirrored pose
        assert window.viewport._backdrop is not None
        window.backdrop_action.setChecked(False)
        window._on_toggle_backdrop(False)
        assert window.viewport._backdrop is None
    finally:
        window.close()


# --- F16: the photo's own camera and the backdrop's opacity --------------------------------------


def _screen_offsets(view, landmarks, xy, corners):
    """How far (px) each landmark lands from where its pixel of the photo lands on screen."""
    tl, tr, _br, bl = corners
    on_photo = np.array([tl + x * (tr - tl) + y * (bl - tl) for x, y in xy])
    return np.linalg.norm(view.project_to_screen(landmarks) - view.project_to_screen(on_photo), axis=1)


def test_viewing_from_the_photo_camera_overlays_photo_and_skeleton(qapp):
    landmarks = _articulated_pose()
    aspect = 4 / 3
    xy = np.column_stack([0.5 + landmarks[:, 0] * 0.6 / aspect, 0.4 - landmarks[:, 1] * 0.6])
    corners = viewport_3d.photo_backdrop_corners(landmarks, xy, aspect)
    view = viewport_3d.SkeletonViewport()
    view.resize(400, 300)
    try:
        assert view.view_from_photo_camera() is False  # nothing to align to yet
        view.set_backdrop(np.zeros((240, 320, 3), np.uint8), corners)
        rotations = sk.solve_bone_rotations(landmarks)
        view.show_skeleton(sk.forward_kinematics(rotations, *sk.measured_proportions(landmarks)))
        view.set_camera_preset("front")
        front = _screen_offsets(view, landmarks, xy, corners).max()
        assert view.view_from_photo_camera() is True
        assert view.opts["fov"] == viewport_3d.PHOTO_CAMERA_FOV
        aligned = _screen_offsets(view, landmarks, xy, corners).max()
        assert aligned < 6.0, "every landmark within a few pixels of its spot on the photo"
        assert aligned < front / 3, "much closer than the perspective front view"
        photo = view.project_to_screen(corners)
        assert np.all((photo >= -1) & (photo <= np.array([401, 301]))), "the whole photo is in view"
        view.set_camera_preset("side")
        assert view.opts["fov"] == viewport_3d.DEFAULT_FOV
    finally:
        view.close()


def test_backdrop_opacity_changes_the_plane_and_is_remembered(qapp, settings):
    window = _window_with_photo(settings, [_person(_articulated_pose())])
    try:
        window.backdrop_opacity_slider.setValue(40)
        window._on_view_from_photo_camera()  # turns the backdrop on and aligns the view
        assert window.backdrop_action.isChecked() and window.viewport.has_backdrop()
        assert window.viewport.opts["fov"] == viewport_3d.PHOTO_CAMERA_FOV
        assert int(window.viewport._backdrop_rgba[0, 0, 3]) == round(255 * 0.4)
        window.backdrop_opacity_slider.setValue(90)
        assert int(window.viewport._backdrop_rgba[0, 0, 3]) == round(255 * 0.9)
        window._save_settings()
    finally:
        window.close()
    again = MainWindow(settings=settings)
    try:
        assert again.backdrop_opacity_slider.value() == 90
    finally:
        again.close()

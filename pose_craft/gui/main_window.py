from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack
from PySide6.QtWidgets import (  # noqa: F401 -- QFileDialog: the tests patch it as mw.QFileDialog
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from pose_craft.diagnostics import diagnostics_report, format_report, log_file_hint
from pose_craft.estimation.landmarker import (
    MODEL_URLS,
    DetectedPerson,
    DownloadCancelled,
    PoseEstimator,
    is_model_available,
)
from pose_craft.export import live_link
from pose_craft.export.blender_pose import (
    ANIMATION_FORMAT_NAME,
    LOW_CONFIDENCE_THRESHOLD,
    build_animation_document,
    build_pose_document,
    low_confidence_threshold,
)
from pose_craft.export.live_link import send_animation_document, send_pose_document
from pose_craft.gui.batch_dialog import BatchDialog, BatchSettings
from pose_craft.gui.photo_view import PhotoOverlayPerson
from pose_craft.gui.viewport_3d import DEFAULT_BACKDROP_OPACITY, photo_backdrop_corners
from pose_craft.gui.window_export import SKELETON_FILTERS, ExportMixin, save_skeleton_animation  # noqa: F401
from pose_craft.gui.window_inputs import (  # noqa: F401 -- also used as mw.* by the tests
    _PHOTO_FILTER,
    _VIDEO_FILTER,
    MAX_RECENT_FILES,
    PHOTO_SUFFIXES,
    VIDEO_SUFFIXES,
    InputsMixin,
    _update_recent_files,
)
from pose_craft.gui.window_layout import (  # noqa: F401
    DEFAULT_AUTO_SEND_DEBOUNCE_MS,
    DEFAULT_VIDEO_ROTATION_SMOOTHING,
    LayoutMixin,
)
from pose_craft.posing import (  # noqa: F401 -- EditedPerson, edited_person, default_export_name: also used as mw.*
    EditedPerson,
    PersonEdits,
    default_export_name,
    document_inputs,
    edited_person,
    out_of_frame_bones,
    root_for,
)
from pose_craft.retarget.skeleton import (
    BONE_NAMES,
    DRAGGABLE_JOINTS,
    FLIPPABLE_BONES,
    JOINT_LABELS,
    ORIENTATION_CORRECTIONS,
    bone_confidence,
    bone_lateral_axes,
    floor_height,
    forward_kinematics,
    image_joint_positions_2d,
    joint_positions,
    measured_proportions,
    mirror_hands,
    mirror_landmarks,
    rotate_joint_positions,
)

log = logging.getLogger(__name__)

# How long closeEvent waits for a background worker before hiding the window
# and finishing the close once the worker returns (a download blocked in a
# network read can take up to landmarker.DOWNLOAD_TIMEOUT_SECONDS to notice a
# cancel).
_CLOSE_WAIT_MS = 5000

# A click farther than this (as a fraction of the photo's diagonal) from every
# person's drawn skeleton doesn't change the selection.
PICK_MAX_DISTANCE = 0.05


def default_settings() -> QSettings:
    """PoseCraft's settings file (an INI under the user's config folder)."""
    return QSettings(QSettings.IniFormat, QSettings.UserScope, "PoseCraft", "PoseCraft")


class SolveWorker(QThread):
    """Runs model download + pose detection off the UI thread -- the first run in
    particular can take a while while the landmarker model downloads. Reuses an
    already-created PoseEstimator when the caller has one for the same model and
    device (see MainWindow._estimator) so the model only has to load once per app
    session, not once per click.
    """

    finished_ok = Signal(list, object)  # list[DetectedPerson], PoseEstimator
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, int)  # downloaded_bytes, total_bytes -- only during a real download

    def __init__(
        self,
        image_bgr: np.ndarray,
        estimator: PoseEstimator | None,
        max_people: int,
        model_variant: str,
        use_gpu: bool = True,
        group_photo: bool = False,
        hands: bool = False,
        min_detection_confidence: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)
        self._min_detection_confidence = min_detection_confidence
        self._image_bgr = image_bgr
        self._estimator = estimator
        self._max_people = max_people
        self._model_variant = model_variant
        self._use_gpu = use_gpu
        self._group_photo = group_photo
        self._hands = hands
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Asks a running model download to stop at its next chunk (see
        landmarker.ensure_model_downloaded); a solve that's already loading the
        model is dropped as soon as that finishes. An inference already in
        progress can't be interrupted, but it takes milliseconds."""
        self._cancel.set()

    def run(self) -> None:
        estimator = self._estimator
        newly_created = estimator is None
        try:
            if estimator is None:
                estimator = PoseEstimator.for_variant(
                    self._model_variant,
                    max_people=self._max_people,
                    use_gpu=self._use_gpu,
                    on_progress=lambda downloaded, total: self.progress.emit(downloaded, total),
                    should_cancel=self._cancel.is_set,
                )
            if self._cancel.is_set():
                raise DownloadCancelled("solve cancelled")
            estimator.max_people = self._max_people
            estimator.group_photo = self._group_photo
            estimator.hands = self._hands
            estimator.min_detection_confidence = self._min_detection_confidence
            if self._hands:
                # B11: loads (on first use, downloads) the hand landmark model.
                estimator.enable_hands(
                    on_progress=lambda downloaded, total: self.progress.emit(downloaded, total),
                    should_cancel=self._cancel.is_set,
                )
                if self._cancel.is_set():
                    raise DownloadCancelled("solve cancelled")
            people = estimator.detect(self._image_bgr)
            if self._cancel.is_set():
                raise DownloadCancelled("solve cancelled")
            self.finished_ok.emit(people, estimator)
        except DownloadCancelled:
            if newly_created and estimator is not None:
                estimator.close()
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            log.exception("pose solve failed")
            if newly_created and estimator is not None:
                estimator.close()
            self.failed.emit(str(exc))


class WarmUpWorker(QThread):
    """Creates the estimator for the remembered model and device at startup and
    runs one blank inference through it (P3), so the first Solve doesn't pay
    the GPU's cold start -- 15.7 s for a fresh CUDA process was measured.
    Only runs when the model is already available offline (never downloads)."""

    finished_ok = Signal(object)  # PoseEstimator
    failed = Signal(str)

    def __init__(self, model_variant: str, use_gpu: bool, max_people: int, parent=None):
        super().__init__(parent)
        self.model_variant = model_variant
        self.use_gpu = use_gpu
        self._max_people = max_people

    def run(self) -> None:
        try:
            estimator = PoseEstimator.for_variant(self.model_variant, max_people=self._max_people, use_gpu=self.use_gpu)
            estimator.warm_up()
            self.finished_ok.emit(estimator)
        except Exception as exc:  # noqa: BLE001 - a failed warm-up just means the first solve loads it
            log.warning("model warm-up failed: %s", exc)
            self.failed.emit(str(exc))


class VideoSolveWorker(QThread):
    """Tracks up to max_people people through a video (estimation/video.py,
    stable IDs per person) off the UI thread."""

    finished_ok = Signal(object, object)  # VideoSolve, PoseEstimator
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, int)  # frames done, total frames (0 if unknown)

    def __init__(
        self, path: str, estimator: PoseEstimator | None, model_variant: str, use_gpu: bool, max_people: int = 1, hands: bool = False,
        parent=None, min_detection_confidence: float = 0.5,
    ):
        super().__init__(parent)
        self._min_detection_confidence = min_detection_confidence
        self._max_people = max_people
        self._path = path
        self._estimator = estimator
        self._model_variant = model_variant
        self._use_gpu = use_gpu
        self._hands = hands
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from pose_craft.estimation.video import solve_video

        estimator = self._estimator
        newly_created = estimator is None
        try:
            if estimator is None:
                estimator = PoseEstimator.for_variant(
                    self._model_variant, use_gpu=self._use_gpu, should_cancel=self._cancel.is_set
                )
            estimator.min_detection_confidence = self._min_detection_confidence
            if self._hands:  # fingers in video (B21): the hand model, downloaded once if needed
                estimator.enable_hands(should_cancel=self._cancel.is_set)
            result = solve_video(
                self._path,
                estimator,
                max_people=self._max_people,
                on_progress=lambda done, total: self.progress.emit(done, total),
                should_cancel=self._cancel.is_set,
                hands=self._hands,
            )
            self.finished_ok.emit(result, estimator)
        except (DownloadCancelled, InterruptedError):
            if newly_created and estimator is not None:
                estimator.close()
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            log.exception("video solve failed")
            if newly_created and estimator is not None:
                estimator.close()
            self.failed.emit(str(exc))


class SendToBlenderWorker(QThread):
    """Runs the Live Link socket connection off the UI thread. A synchronous call
    here would freeze the whole window for up to send_pose_document's timeout
    (3s by default) whenever Blender's Live Link isn't running yet -- a likely
    first-time scenario, not an edge case.
    """

    finished_ok = Signal(object)  # the add-on's reply dict, or None from an add-on too old to reply
    failed = Signal(str)

    def __init__(self, document: dict, host: str, port: int, parent=None):
        super().__init__(parent)
        self._document = document
        self._host = host
        self._port = port

    def run(self) -> None:
        try:
            if self._document.get("format") == ANIMATION_FORMAT_NAME:
                # Chunked under the add-on's payload cap (live_link.send_animation_document).
                reply = send_animation_document(self._document, host=self._host, port=self._port)
            else:
                reply = send_pose_document(self._document, host=self._host, port=self._port)
            self.finished_ok.emit(reply)
        except (OSError, ValueError) as exc:  # ValueError: a malformed reply
            log.warning("Live Link send failed: %s", exc)
            self.failed.emit(str(exc))


def _nearest_person_index(people: list[DetectedPerson], x: float, y: float) -> int:
    distances = [((p.centroid_xy()[0] - x) ** 2 + (p.centroid_xy()[1] - y) ** 2) for p in people]
    return int(np.argmin(distances))


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denominator = float(np.dot(ab, ab))
    t = 0.0 if denominator < 1e-18 else float(np.clip(np.dot(point - a, ab) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _nearest_bone(
    joints_2d: dict[str, np.ndarray], x: float, y: float, aspect: float = 1.0, bones=BONE_NAMES
) -> tuple[str, float]:
    """(bone name, distance) of the drawn bone segment nearest the normalized
    point (x, y). `aspect` is the photo's width / height, so distances are
    measured in proportion to the photo's real shape (as a fraction of its
    height)."""
    scale = np.array([aspect, 1.0])
    point = np.array([x, y]) * scale
    best = ("", float("inf"))
    for name in bones:
        distance = _point_segment_distance(
            point, joints_2d[f"{name}.head"] * scale, joints_2d[f"{name}.tail"] * scale
        )
        if distance < best[1]:
            best = (name, distance)
    return best


def _person_at(
    people_joints_2d: list[dict[str, np.ndarray]], x: float, y: float, aspect: float = 1.0,
    max_distance: float = PICK_MAX_DISTANCE,
) -> int | None:
    """The person whose drawn skeleton passes nearest the click (F5), or None
    when the click is farther than `max_distance` of the photo's diagonal from
    everyone -- a click on empty background no longer changes the selection,
    and in a crowded shot the skeleton under the cursor wins rather than the
    one whose centroid happens to be closest."""
    diagonal = float(np.hypot(aspect, 1.0))
    best_index, best_distance = None, float("inf")
    for index, joints in enumerate(people_joints_2d):
        _bone, distance = _nearest_bone(joints, x, y, aspect)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index if best_distance <= max_distance * diagonal else None


def _low_confidence_bone_count(confidence: dict[str, float]) -> int:
    """How many bones in a bone_confidence() dict fall below
    blender_pose.LOW_CONFIDENCE_THRESHOLD -- the same threshold the photo overlay and the
    3D viewport use to color a bone as low-confidence. A plain function (rather than a
    method) so the post-solve status message logic can be exercised directly without a
    real solve or Qt widgets.
    """
    return sum(1 for score in confidence.values() if score < low_confidence_threshold())


def _low_confidence_status_suffix(confidence: dict[str, float]) -> str:
    """The post-solve status message's low-confidence note (appended to "Solved N
    pose(s)" the same way the max-people-limit warning already is), pulled out as its
    own pure function -- like _low_confidence_bone_count above -- so the exact wording
    is testable without a real solve or Qt widgets. "" (nothing appended) when no bone
    is low-confidence.
    """
    low_count = _low_confidence_bone_count(confidence)
    if low_count == 0:
        return ""
    return f" -- {low_count} of {len(confidence)} bones low-confidence"


def _person_label(index: int, confidence: dict[str, float]) -> str:
    """The person picker's entry: "Person 2: 13/15 bones confident"."""
    confident = len(confidence) - _low_confidence_bone_count(confidence)
    return f"Person {index + 1}: {confident}/{len(confidence)} bones confident"


def _live_link_reply_message(reply: dict | None) -> tuple[str, bool]:
    """Status text for the Blender add-on's Live Link reply, and whether the pose
    was applied. `reply` is None when the add-on predates the reply protocol
    (it applied or rejected the pose without saying which)."""
    if reply is None:
        return (
            "Pose sent to Blender, but the add-on didn't confirm it -- update the PoseCraft add-on "
            "to see what was applied",
            True,
        )
    if not reply.get("ok"):
        return f"Blender didn't apply the pose: {reply.get('error') or 'unknown error'}", False
    applied = reply.get("applied") or []
    requested = reply.get("requested", len(applied))
    frame_count = int(reply.get("animation_frames") or 0)
    if frame_count > 0:
        message = (
            f"Keyed {frame_count} frame(s), {len(applied)}/{requested} bones per frame, on "
            f"'{reply.get('armature', '?')}'"
        )
        if reply.get("start_frame") is not None and reply.get("end_frame") is not None:
            message += f" (Blender frames {reply['start_frame']}-{reply['end_frame']})"
    else:
        message = f"Applied {len(applied)}/{requested} bones to '{reply.get('armature', '?')}'"
    unmatched = reply.get("unmatched") or []
    if unmatched:
        message += f" (unmatched: {', '.join(unmatched)})"
    low_confidence = reply.get("low_confidence") or []
    if low_confidence:
        message += f"; {len(low_confidence)} low-confidence"
    return message, True


def _preview_positions(
    landmarks: np.ndarray, rotations: dict, correction, hands: dict | None = None, extra_points: dict | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """What the 3D preview draws: (the skeleton posed by forward kinematics from
    the exported rotations, the raw landmark joints as a ghost). The FK skeleton
    uses the subject's own segment lengths and is rooted at the hip center, so
    it sits on the ghost wherever the rotations reproduce the landmarks and
    visibly departs from it where they don't. The ghost gets the same body
    orientation fix the rotations already carry. Both are then lifted so the
    skeleton's lowest joint stands on the grid at Y = 0 (B6) -- the world
    landmarks are hip-centred, so without this every pose stood half through
    the floor.

    hands: the hand landmarks the rotations were solved with (B11), adding
    finger joints to the ghost. extra_points: more ghost-space points (the
    photo backdrop's corners, F9) to move the same way as the ghost; they're
    returned in the ghost dict under their own keys."""
    raw = joint_positions(landmarks, hands)
    lengths, shoulder_half, hip_half = measured_proportions(landmarks, hands)
    solved = forward_kinematics(rotations, lengths, shoulder_half, hip_half, root_position=raw["spine.head"])
    ghost = rotate_joint_positions({**raw, **(extra_points or {})}, correction, pivot=raw["spine.head"])
    lift = np.array([0.0, -floor_height(solved), 0.0])
    return {k: v + lift for k, v in solved.items()}, {k: v + lift for k, v in ghost.items()}


ABOUT_TEXT = (
    "PoseCraft {version}\n\nTurns a photo of a person into a Blender pose: MediaPipe BlazePose landmarks "
    "on ONNX Runtime, retargeted to an armature.\n\nHelp > Diagnostics shows the runtime and GPU details."
)


class _EditCommand(QUndoCommand):
    """One manual correction as an undo step: swaps a person's PersonEdits."""

    def __init__(self, window: MainWindow, person_key: int, before: PersonEdits, after: PersonEdits, text: str):
        super().__init__(text)
        # person_key: the person's index in a photo, their track ID in a video.
        self._window, self._index, self._before, self._after = window, person_key, before, after

    def redo(self) -> None:
        self._window._set_person_edits(self._index, self._after)

    def undo(self) -> None:
        self._window._set_person_edits(self._index, self._before)


class MainWindow(LayoutMixin, InputsMixin, ExportMixin, QMainWindow):
    def __init__(self, settings: QSettings | None = None, warm_up: bool = False):
        super().__init__()
        self.setWindowTitle("PoseCraft - Photo to Pose")
        self.resize(1200, 720)
        self.setAcceptDrops(True)

        self._settings = settings if settings is not None else default_settings()
        self._image_bgr: np.ndarray | None = None
        self._source_path: str | None = None
        self._people: list[DetectedPerson] = []
        # Edit key per entry of _people: its index for a photo, its track ID
        # for a video (so a person's edits follow them from frame to frame).
        self._person_keys: list[int] = []
        self._selected_track: int | None = None
        self._edits: dict[int, PersonEdits] = {}
        self._preview_bgr: np.ndarray | None = None  # what the photo pane shows (backdrop, F9)
        self._video = None  # estimation.video.VideoSolve while a video is loaded
        self._worker: SolveWorker | None = None
        self._video_worker: VideoSolveWorker | None = None
        self._warmup_worker: WarmUpWorker | None = None
        self._send_worker: SendToBlenderWorker | None = None
        self._estimator: PoseEstimator | None = None
        self._pending_solve = False  # Solve clicked while the warm-up was still running
        self._closing = False  # set by closeEvent; suppresses the dialogs of work it cancelled
        self._syncing_adjust = False
        self._send_origin = "manual"
        self._auto_send_paused = False  # set when an automatic send couldn't reach Blender
        self._track_frames_cache: dict = {}  # (video, track, options) -> (edits, result)
        self.undo_stack = QUndoStack(self)
        self.undo_stack.indexChanged.connect(lambda _index: self._sync_adjust_controls())
        self._auto_send_timer = QTimer(self)
        self._auto_send_timer.setSingleShot(True)
        self._auto_send_timer.timeout.connect(self._send_auto_pose)

        self._build_ui()
        self._build_menus()
        self._restore_settings()
        self._apply_accessibility_style()
        self._set_result_actions_enabled(False)
        if warm_up:
            self._start_warm_up()

    # -- layout ------------------------------------------------------------------


    # -- batch (B13) ---------------------------------------------------------------------

    def _on_batch_folder(self) -> BatchDialog:
        """Solve a folder of photos with the current detection settings."""
        settings = BatchSettings(
            variant=self.model_variant_combo.currentText(),
            use_gpu=self.use_gpu_checkbox.isChecked(),
            max_people=self.max_people_spin.value(),
            group_photo=self.group_photo_checkbox.isChecked(),
            hands=self.hands_checkbox.isChecked(),
            detection_threshold=self.detection_threshold_spin.value(),
        )
        dialog = BatchDialog(settings, self._last_dir(), self.mirror_checkbox.isChecked(), self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()
        return dialog

    # -- help (R4) -----------------------------------------------------------------------

    def _on_show_diagnostics(self) -> QDialog:
        """Versions, providers, GPUs and why the GPU isn't used, with Copy (R4)."""
        text = format_report(diagnostics_report(self._estimator))
        dialog = QDialog(self)
        dialog.setWindowTitle("PoseCraft Diagnostics")
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setObjectName("diagnosticsText")
        font = view.font()
        font.setFamily("Consolas")
        view.setFont(font)
        view.setMinimumSize(640, 220)
        layout.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = buttons.addButton("&Copy", QDialogButtonBox.ActionRole)
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(text))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()
        return dialog

    def _on_about(self) -> None:
        from pose_craft import __version__

        QMessageBox.about(self, "About PoseCraft", ABOUT_TEXT.format(version=__version__))

    # -- settings (F7) --------------------------------------------------------------

    def _restore_settings(self) -> None:
        s = self._settings
        self.live_link_host_edit.setText(str(s.value("live_link/host", live_link.DEFAULT_HOST)))
        self.live_link_port_spin.setValue(int(s.value("live_link/port", live_link.DEFAULT_PORT)))
        variant = str(s.value("detect/model", "full"))
        if variant in MODEL_URLS:
            self.model_variant_combo.setCurrentText(variant)
        self.use_gpu_checkbox.setChecked(str(s.value("detect/use_gpu", "true")).lower() == "true")
        self.max_people_spin.setValue(int(s.value("detect/max_people", 4)))
        self.group_photo_checkbox.setChecked(str(s.value("detect/group_photo", "false")).lower() == "true")
        self.hands_checkbox.setChecked(str(s.value("detect/hands", "false")).lower() == "true")
        self.detection_threshold_spin.setValue(float(s.value("detect/detection_threshold", 0.5)))
        self.low_confidence_spin.setValue(float(s.value("detect/low_confidence_threshold", LOW_CONFIDENCE_THRESHOLD)))
        self.video_rotation_smoothing_spin.setValue(int(s.value("video/rotation_smoothing", DEFAULT_VIDEO_ROTATION_SMOOTHING)))
        self.backdrop_action.setChecked(str(s.value("view/backdrop", "false")).lower() == "true")
        self.backdrop_opacity_slider.setValue(int(s.value("view/backdrop_opacity", round(100 * DEFAULT_BACKDROP_OPACITY))))
        self.auto_send_checkbox.setChecked(str(s.value("live_link/auto_send", "false")).lower() == "true")
        self.auto_send_debounce_spin.setValue(int(s.value("live_link/auto_send_debounce_ms", DEFAULT_AUTO_SEND_DEBOUNCE_MS)))
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter = s.value("window/splitter")
        if splitter is not None:
            self.splitter.restoreState(splitter)
        self._rebuild_recent_menu()

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("live_link/host", self.live_link_host_edit.text().strip() or live_link.DEFAULT_HOST)
        s.setValue("live_link/port", self.live_link_port_spin.value())
        s.setValue("detect/model", self.model_variant_combo.currentText())
        s.setValue("detect/use_gpu", "true" if self.use_gpu_checkbox.isChecked() else "false")
        s.setValue("detect/max_people", self.max_people_spin.value())
        s.setValue("detect/group_photo", "true" if self.group_photo_checkbox.isChecked() else "false")
        s.setValue("detect/hands", "true" if self.hands_checkbox.isChecked() else "false")
        s.setValue("detect/detection_threshold", self.detection_threshold_spin.value())
        s.setValue("detect/low_confidence_threshold", self.low_confidence_spin.value())
        s.setValue("video/rotation_smoothing", self.video_rotation_smoothing_spin.value())
        s.setValue("view/backdrop", "true" if self.backdrop_action.isChecked() else "false")
        s.setValue("view/backdrop_opacity", self.backdrop_opacity_slider.value())
        s.setValue("live_link/auto_send", "true" if self.auto_send_checkbox.isChecked() else "false")
        s.setValue("live_link/auto_send_debounce_ms", self.auto_send_debounce_spin.value())
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.splitter.saveState())
        s.sync()


    # -- busy / enabled state (C1) --------------------------------------------------

    def _set_busy(self, busy: bool, message: str = "", cancellable: bool = True) -> None:
        """One place for "a background solve is running": what it disables (a new
        photo mid-solve would let this solve's results land on it; the model,
        device and people count are baked into the running worker), the
        progress bar (indeterminate until a download reports progress), the
        Cancel button and the busy cursor."""
        for widget in (
            self.load_btn, self.load_video_btn, self.solve_btn, self.max_people_spin,
            self.model_variant_combo, self.use_gpu_checkbox, self.group_photo_checkbox, self.hands_checkbox,
        ):
            widget.setEnabled(not busy)
        for action in (self.open_action, self.open_video_action, self.solve_action, self.recent_menu.menuAction()):
            action.setEnabled(not busy)
        if not busy:
            self.solve_btn.setEnabled(self._image_bgr is not None)
            self.solve_action.setEnabled(self._image_bgr is not None)
            self.recent_menu.setEnabled(bool(self._recent_files()))
        self.progress_bar.setVisible(busy)
        self.progress_bar.setRange(0, 0)  # indeterminate until progress arrives
        self.cancel_btn.setVisible(busy and cancellable)
        self.cancel_action.setEnabled(busy and cancellable)
        if busy:
            self.setCursor(Qt.BusyCursor)
            if message:
                self.statusBar().showMessage(message)
        else:
            self.unsetCursor()

    def _set_result_actions_enabled(self, has_person: bool) -> None:
        """Everything that needs a solved, selected person (and Export All, which
        needs any people)."""
        for widget in (self.export_btn, self.export_bvh_btn, self.export_daz_btn, self.export_all_bvh_btn,
                       self.send_to_blender_btn, self.reset_bone_btn,
                       self.reset_edits_btn, self.adjust_bone_combo, *self.adjust_spins):
            widget.setEnabled(has_person)
        for action in (self.export_action, self.export_bvh_action, self.export_daz_action, self.send_action,
                       self.add_to_library_action, self.next_person_action, self.previous_person_action):
            action.setEnabled(has_person)
        has_people = bool(self._people) and self._video is None
        self.export_all_btn.setEnabled(has_people)
        self.export_all_bvh_btn.setEnabled(has_people)
        self.export_all_action.setEnabled(has_people)
        self.export_animation_btn.setEnabled(self._video is not None and bool(self._video.solved_frames()))
        self.export_all_tracks_btn.setEnabled(self._video is not None and len(self._video.track_ids()) > 0)
        self.send_animation_btn.setEnabled(self._video is not None and bool(self._video.solved_frames()))

    def _on_auto_send_toggled(self, checked: bool) -> None:
        self._auto_send_paused = False  # ticking it again is how a paused auto-send resumes
        if checked:
            self._queue_auto_send()
        else:
            self._auto_send_timer.stop()

    def _has_selected_person(self) -> bool:
        return 0 <= self.person_combo.currentIndex() < len(self._people)

    def _queue_auto_send(self) -> None:
        """(Re)starts the debounce: a burst of changes (scrubbing, dragging a
        spin box) sends once, after it settles. Nothing is queued without a
        selected person, while paused or while closing."""
        if not self.auto_send_checkbox.isChecked() or self._closing or self._auto_send_paused:
            return
        if not self._has_selected_person():
            self._auto_send_timer.stop()
            return
        self._auto_send_timer.start(self.auto_send_debounce_spin.value())

    def _send_auto_pose(self) -> None:
        if self._closing or self._auto_send_paused or not self.auto_send_checkbox.isChecked():
            return
        if self._send_worker is not None and self._send_worker.isRunning():
            self._queue_auto_send()  # one send at a time; try again after another debounce
            return
        document = self._current_live_link_document()
        if document is None:
            return
        self._start_live_link_send(document, automatic=True)

    def _worker_running(self) -> bool:
        return any(w is not None and w.isRunning() for w in (self._worker, self._video_worker))

    # -- closing -------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Closing the shared PoseEstimator while a background thread is still
        # calling into it (or destroying a QThread object while it's still
        # running) is unsafe, so in-flight work gets to finish first -- but
        # never with an unbounded wait: a model download stalled in a network
        # read used to make the window unclosable. Ask the solve to cancel and
        # wait up to _CLOSE_WAIT_MS; if a worker is still running after that,
        # hide the window now and finish closing when the worker returns (the
        # download's socket timeout bounds how long that can take).
        self._closing = True
        self._auto_send_timer.stop()
        workers = (self._worker, self._video_worker, self._warmup_worker, self._send_worker)
        running = [w for w in workers if w is not None and w.isRunning()]
        for worker in running:
            if hasattr(worker, "cancel"):
                worker.cancel()
        still_running = [w for w in running if not w.wait(_CLOSE_WAIT_MS)]
        if still_running:
            self.hide()
            for worker in still_running:
                worker.finished.connect(self.close)
            event.ignore()
            return
        try:
            self._save_settings()
        except Exception:  # noqa: BLE001 -- settings must never block closing
            log.exception("could not save settings")
        if self._estimator is not None:
            self._estimator.close()
        super().closeEvent(event)

    # -- loading ---------------------------------------------------------------------


    def _clear_results(self) -> None:
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        self.person_combo.blockSignals(False)
        self._people = []
        self._person_keys = []
        self._selected_track = None
        self._edits = {}
        self._track_frames_cache = {}
        self.undo_stack.clear()
        self.viewport.clear_skeleton()
        self.photo_view.set_skeleton_overlay([])
        self._set_result_actions_enabled(False)


    # -- solving -----------------------------------------------------------------------

    def _estimator_matches(self, estimator: PoseEstimator | None) -> bool:
        return (
            estimator is not None
            and getattr(estimator, "variant", None) == self.model_variant_combo.currentText()
            and estimator.use_gpu == self.use_gpu_checkbox.isChecked()
        )

    def _take_estimator(self) -> PoseEstimator | None:
        """The cached estimator if it still matches the chosen model and device
        (both are fixed when the ONNX Runtime sessions are created); otherwise
        it's closed and None is returned, so the worker builds a new one."""
        if self._estimator is not None and not self._estimator_matches(self._estimator):
            self._estimator.close()
            self._estimator = None
        return self._estimator

    def _on_solve_pose(self) -> None:
        if self._image_bgr is None or self._worker_running():
            return  # defense in depth -- solve_btn being disabled should already prevent this
        if self._warmup_worker is not None and self._warmup_worker.isRunning():
            # The model is being prepared already; solve as soon as it's ready.
            self._pending_solve = True
            self._set_busy(True, "Preparing the pose model...", cancellable=False)
            return

        estimator = self._take_estimator()
        first_run_note = "" if estimator is not None else " (loading the model -- the first run also downloads it)"
        self._set_busy(True, f"Solving pose...{first_run_note}")

        self._worker = SolveWorker(
            self._image_bgr,
            estimator,
            self.max_people_spin.value(),
            self.model_variant_combo.currentText(),
            self.use_gpu_checkbox.isChecked(),
            self.group_photo_checkbox.isChecked(),
            self.hands_checkbox.isChecked(),
            self.detection_threshold_spin.value(),
        )
        self._worker.finished_ok.connect(self._on_solve_finished)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.cancelled.connect(self._on_solve_cancelled)
        self._worker.progress.connect(self._on_solve_progress)
        self._worker.start()

    def _on_cancel(self) -> None:
        for worker in (self._worker, self._video_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                self.statusBar().showMessage("Cancelling...")

    def _on_solve_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(downloaded * 100 // total)
            self.statusBar().showMessage(f"Downloading model... {downloaded * 100 // total}%")
        else:
            self.statusBar().showMessage(f"Downloading model... {downloaded // 1024} KB")

    def _on_solve_finished(self, people: list[DetectedPerson], estimator: PoseEstimator) -> None:
        if self._closing:
            if estimator is not self._estimator:
                estimator.close()
            return
        self._set_busy(False)
        self._clear_video()
        self._estimator = estimator
        self._clear_results()
        self._people = people
        self._person_keys = list(range(len(people)))

        if not people:
            # Clear everything the previous solve left behind -- the person list,
            # photo overlay, 3D skeleton and export buttons all still referred to
            # people from the last solve, which no longer exist.
            self.statusBar().showMessage(f"No person detected in this photo (ran on {estimator.device})")
            if not self._closing:
                QMessageBox.information(self, "Solve Pose", "No person was detected in this photo.")
            return

        self._fill_person_combo()
        status = f"Solved {len(people)} pose(s) on {estimator.device}"
        if len(people) >= estimator.max_people:
            status += (
                f" -- reached the max-people limit ({estimator.max_people}); "
                "there may be more people in the photo than were detected"
            )
        # person_combo.setCurrentIndex(0) above selects the previewed person; summarize
        # that person's bone confidence here too, next to it, following the same
        # append-onto-`status` pattern the max-people warning above already uses.
        previewed_confidence = bone_confidence(people[self.person_combo.currentIndex()].visibility)
        status += _low_confidence_status_suffix(previewed_confidence)
        if self.hands_checkbox.isChecked():
            hands = sum(len(getattr(person, "hands", {}) or {}) for person in people)
            status += f" -- {hands} hand(s) found"
        self.statusBar().showMessage(status)
        self._update_preview_for_selected_person()

    def _fill_person_combo(self) -> None:
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        for i, person in enumerate(self._people):
            self.person_combo.addItem(_person_label(i, bone_confidence(person.visibility)))
        self.person_combo.blockSignals(False)
        self.person_combo.setCurrentIndex(0)

    def _on_solve_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Solve failed")
        if self._closing:
            return  # cancelled by closeEvent -- not an error worth a dialog
        QMessageBox.critical(
            self, "Solve Pose", f"Pose solving failed:\n{message}\n\nDetails are in the log: {log_file_hint()}"
        )

    def _on_solve_cancelled(self) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Cancelled")

    # -- warm-up (P3) ------------------------------------------------------------------

    def _start_warm_up(self) -> None:
        variant = self.model_variant_combo.currentText()
        if not is_model_available(variant):
            return  # never download in the background; the first Solve does it, with progress
        self._warmup_worker = WarmUpWorker(variant, self.use_gpu_checkbox.isChecked(), self.max_people_spin.value())
        self._warmup_worker.finished_ok.connect(self._on_warm_up_finished)
        self._warmup_worker.failed.connect(self._on_warm_up_failed)
        self.statusBar().showMessage("Preparing the pose model in the background...")
        self._warmup_worker.start()

    def _on_warm_up_finished(self, estimator: PoseEstimator) -> None:
        if self._closing:
            estimator.close()
            return
        if self._estimator is None and self._estimator_matches(estimator):
            self._estimator = estimator
            reason = getattr(estimator, "gpu_fallback_reason", None)
            note = f" (GPU not used: {reason})" if reason else ""
            self.statusBar().showMessage(f"Ready -- pose model loaded on {estimator.device}{note}")
            # The message is replaced by the next status; the tooltip keeps the reason (R4).
            self.statusBar().setToolTip(
                f"Pose model on {estimator.device}." + (f" GPU not used: {reason}." if reason else "")
                + " Help > Diagnostics has the details."
            )
        else:
            estimator.close()  # the model or device was changed meanwhile
        self._resume_pending_solve()

    def _on_warm_up_failed(self, message: str) -> None:
        if not self._closing:
            self.statusBar().showMessage("Ready")
        self._resume_pending_solve()

    def _resume_pending_solve(self) -> None:
        if self._pending_solve and not self._closing:
            self._pending_solve = False
            self._set_busy(False)
            self._on_solve_pose()

    # -- video (B12) --------------------------------------------------------------------


    def load_video(self, path: str) -> None:
        if self._worker_running():
            return
        self._remember_file(path)
        estimator = self._take_estimator()
        self._set_busy(True, f"Tracking {Path(path).name}...")
        self._video_worker = VideoSolveWorker(
            path, estimator, self.model_variant_combo.currentText(), self.use_gpu_checkbox.isChecked(),
            self.max_people_spin.value(), min_detection_confidence=self.detection_threshold_spin.value(),
            hands=self.hands_checkbox.isChecked(),
        )
        self._video_worker.finished_ok.connect(self._on_video_finished)
        self._video_worker.failed.connect(self._on_video_failed)
        self._video_worker.cancelled.connect(self._on_solve_cancelled)
        self._video_worker.progress.connect(self._on_video_progress)
        self._video_worker.start()

    def _on_video_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"Tracking frame {done}" + (f" of {total}" if total else "") + "...")

    def _on_video_finished(self, result, estimator: PoseEstimator) -> None:
        if self._closing:
            if estimator is not self._estimator:
                estimator.close()
            return
        self._set_busy(False)
        self._estimator = estimator
        self._image_bgr = None
        self._source_path = result.path
        self._clear_results()
        self._video = result
        solved = len(result.solved_frames())
        self.video_group.setVisible(True)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(result.frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._on_frame_changed(0)
        tracks = len(result.track_ids())
        self.statusBar().showMessage(
            f"Tracked {tracks} {'person' if tracks == 1 else 'people'} in {solved} of {len(result.frames)} frames "
            f"at {result.fps:.0f} fps on {estimator.device}"
        )

    def _on_video_failed(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage("Video failed")
        if not self._closing:
            QMessageBox.critical(self, "Load Video", f"Could not track the video:\n{message}")

    def _clear_video(self) -> None:
        self._video = None
        self._track_frames_cache = {}
        self.video_group.setVisible(False)
        self.frame_label.setText("no video")

    def _on_frame_changed(self, index: int) -> None:
        if self._video is None or not (0 <= index < len(self._video.frames)):
            return
        frame = self._video.frames[index]
        self.frame_label.setText(f"{index + 1}/{len(self._video.frames)}")
        image = self._read_video_frame(index)
        if image is not None:
            self._set_photo_preview(image)
        # Everyone tracked in this frame, by track ID (B12); the same person
        # stays selected from frame to frame while they're in view.
        self._person_keys = sorted(frame.people)
        self._people = [frame.people[key] for key in self._person_keys]
        if self._selected_track is None and self._person_keys:
            self._selected_track = self._person_keys[0]
        self.person_combo.blockSignals(True)
        self.person_combo.clear()
        for key, person in zip(self._person_keys, self._people):
            self.person_combo.addItem(_person_label(key, bone_confidence(person.visibility)))
        self.person_combo.blockSignals(False)
        if self._people:
            selected = self._person_keys.index(self._selected_track) if self._selected_track in self._person_keys else 0
            self.person_combo.setCurrentIndex(selected)
        self._update_preview_for_selected_person()

    def _read_video_frame(self, index: int) -> np.ndarray | None:
        import cv2

        capture = cv2.VideoCapture(self._video.path)
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            return image if ok else None
        finally:
            capture.release()

    def _video_rotation_smoothing_strength(self) -> float:
        return self.video_rotation_smoothing_spin.value() / 100.0

    def _video_track_frames(self, track_id: int) -> tuple[list[int], list[dict], list]:
        """(frame indices, local rotations per solved frame, BVH root positions)
        for one tracked person, with their edits, mirror/orientation fix, and
        optional quaternion-aware rotation smoothing."""
        from pose_craft.estimation.video import animation_root_positions, smooth_rotation_frames

        edits = self._edits.get(track_id)
        # Solving every frame of a track isn't free, and the preview, auto-send
        # and every export ask for it on each change: keep the last result per
        # track until the video, the person's edits or an option changes.
        key = (
            id(self._video), track_id, self.mirror_checkbox.isChecked(), self.orientation_combo.currentText(),
            self.video_rotation_smoothing_spin.value(), self._photo_aspect(),
        )
        cached = self._track_frames_cache.get(key)
        if cached is not None and cached[0] == edits:
            return cached[1]
        frame_indices = [frame.index for frame in self._video.solved_frames(track_id)]
        frames = [
            self._document_inputs_for(
                p, self.mirror_checkbox.isChecked(), self._orientation_correction(), edits, self._photo_aspect()
            )[0]
            for p in self._video.people_of(track_id)
        ]
        frames = smooth_rotation_frames(frames, self._video_rotation_smoothing_strength(), frame_indices)
        result = (frame_indices, frames, animation_root_positions(frames, self._video.fps) if frames else [])
        # One entry per track: a changed option replaces it.
        self._track_frames_cache = {k: v for k, v in self._track_frames_cache.items() if k[:2] != key[:2]}
        self._track_frames_cache[key] = (edits, result)
        return result

    def _video_frame_rotations(self, index: int, rotations: dict) -> dict:
        """Person `index` of the current video frame: their rotations as their
        track's animation has them (rotation smoothing included); `rotations`
        unchanged for a photo -- so the preview and Live Link show exactly
        what the animation export writes."""
        if self._video is None or not (0 <= index < len(self._person_keys)):
            return rotations
        indices, rotation_frames, _roots = self._video_track_frames(self._person_keys[index])
        frame_index = self.frame_slider.value()
        for index_value, frame_rotations in zip(indices, rotation_frames):
            if index_value == frame_index:
                return frame_rotations
        return rotations

    def _current_track(self) -> int | None:
        if self._video is None:
            return None
        if self._selected_track is not None and self._selected_track in self._video.track_ids():
            return self._selected_track
        tracks = self._video.track_ids()
        return tracks[0] if tracks else None


    # -- selection -----------------------------------------------------------------------

    def _on_person_changed(self) -> None:
        index = self.person_combo.currentIndex()
        if self._video is not None and 0 <= index < len(self._person_keys):
            self._selected_track = self._person_keys[index]
        self._update_preview_for_selected_person()

    def _cycle_person(self, step: int) -> None:
        count = self.person_combo.count()
        if count:
            self.person_combo.setCurrentIndex((self.person_combo.currentIndex() + step) % count)

    def _photo_aspect(self) -> float:
        if self._image_bgr is not None:
            h, w = self._image_bgr.shape[:2]
            return w / h
        if self._video is not None:
            w, h = self._video.frame_size
            return w / h if h else 1.0
        return 1.0

    def _people_joints_2d(self) -> list[dict[str, np.ndarray]]:
        """Every person's drawn 2D skeleton, edits included (what's on screen)."""
        return [
            image_joint_positions_2d(self._edited(i).image_xy) for i in range(len(self._people))
        ]

    def _on_photo_clicked(self, x: float, y: float) -> None:
        if not self._people:
            return
        joints = self._people_joints_2d()
        index = _person_at(joints, x, y, self._photo_aspect())
        if index is not None:
            self.person_combo.setCurrentIndex(index)  # triggers _on_person_changed via the signal

    def _on_photo_context(self, x: float, y: float, global_pos) -> None:
        """Right-click a limb (F4): offer to flip that limb's depth."""
        if not self._people:
            return
        joints = self._people_joints_2d()
        index = _person_at(joints, x, y, self._photo_aspect())
        if index is None:
            return
        self.person_combo.setCurrentIndex(index)
        bone, _distance = _nearest_bone(joints[index], x, y, self._photo_aspect(), FLIPPABLE_BONES)
        menu = QMenu(self)
        flipped = bone in self._person_edits(index).flipped
        action = menu.addAction(f"{'Unflip' if flipped else 'Flip'} depth: {bone}")
        action.triggered.connect(lambda: self.flip_bone_depth(index, bone))
        reset = menu.addAction("Reset all edits for this person")
        reset.setEnabled(not self._person_edits(index).is_empty())
        reset.triggered.connect(self._on_reset_edits)
        menu.exec(global_pos)

    def _edited(self, index: int) -> EditedPerson:
        return edited_person(self._people[index], self._person_edits(index), self._photo_aspect())

    def _selected_landmarks(self) -> np.ndarray | None:
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return None
        landmarks = self._edited(index).landmarks
        if self.mirror_checkbox.isChecked():
            landmarks = mirror_landmarks(landmarks)
        return landmarks

    def _orientation_correction(self):
        return ORIENTATION_CORRECTIONS[self.orientation_combo.currentText()]

    def _update_preview_for_selected_person(self) -> None:
        landmarks = self._selected_landmarks()
        if landmarks is None:
            self._set_result_actions_enabled(False)
            self.photo_view.set_skeleton_overlay([])
            self.photo_view.set_drag_handles({})
            self.viewport.clear_skeleton()
            self.viewport.set_backdrop(None)
            return

        # The preview draws the exported rotations themselves (forward
        # kinematics), with the raw landmarks as a faint ghost, so whatever the
        # export gets wrong shows up here before the user exports it. The
        # rotations already carry the mirror, orientation and manual fixes.
        selected_index = self.person_combo.currentIndex()
        edited = self._edited(selected_index)
        rotations, selected_confidence = self._current_person_document_inputs()
        # The displayed person's own track (not _selected_track, which may be
        # out of this frame while someone else is shown).
        rotations = self._video_frame_rotations(selected_index, rotations)
        hands = mirror_hands(edited.hands) if self.mirror_checkbox.isChecked() else edited.hands
        corners = self._backdrop_corners(edited) if self.backdrop_action.isChecked() else None
        extra = {f"_backdrop.{i}": c for i, c in enumerate(corners)} if corners is not None else None
        positions, ghost = _preview_positions(landmarks, rotations, self._orientation_correction(), hands, extra)
        if corners is not None:
            corners = np.array([ghost.pop(f"_backdrop.{i}") for i in range(4)])
        self.viewport.show_skeleton(
            positions, confidence=selected_confidence, ghost_points=ghost, lateral_axes=bone_lateral_axes(rotations),
            rotations=rotations,
        )
        self._update_backdrop(corners)
        self._set_result_actions_enabled(True)
        self._sync_adjust_controls()

        overlay = []
        for i in range(len(self._people)):
            person = self._edited(i)
            overlay.append(
                PhotoOverlayPerson(
                    joints_2d=image_joint_positions_2d(person.image_xy, person.hands_xy),
                    confidence=bone_confidence(person.visibility, person.hand_presence),
                    selected=(i == selected_index),
                    out_of_frame=frozenset(person.out_of_frame()),
                    label=f"Person {self._edit_key(i) + 1}",  # a video's track ID, as in the Person list
                )
            )
        self.photo_view.set_skeleton_overlay(overlay)
        # F4: the selected person's joints can be dragged (photos only -- a
        # video frame's joints differ from frame to frame).
        draggable = self._video is None and self._image_bgr is not None
        self.photo_view.set_drag_handles(
            {joint: edited.image_xy[joint] for joint in DRAGGABLE_JOINTS} if draggable else {}
        )
        self.photo_view.setAccessibleDescription(self._overlay_description(selected_index, selected_confidence))
        self._queue_auto_send()

    def _overlay_description(self, selected_index: int, confidence: dict[str, float]) -> str:
        low = [name for name, score in confidence.items() if score < low_confidence_threshold()]
        text = f"{len(self._people)} people detected; Person {selected_index + 1} selected"
        if low:
            text += f"; {len(low)} low-confidence bones: {', '.join(low)}"
        return text

    def _on_toggle_labels(self, checked: bool) -> None:
        self.viewport.show_labels = checked
        self._update_preview_for_selected_person()

    def _on_toggle_hover(self, checked: bool) -> None:
        self.viewport.hover_enabled = checked

    def _on_toggle_backdrop(self, _checked: bool) -> None:
        self._update_preview_for_selected_person()

    def _on_view_from_photo_camera(self) -> None:
        """F16: turns the backdrop on if needed, then views it straight on."""
        if not self.backdrop_action.isChecked():
            self.backdrop_action.setChecked(True)
            self._update_preview_for_selected_person()
        if not self.viewport.view_from_photo_camera():
            self.statusBar().showMessage("Solve a pose first: the photo is placed from the person's fit", 5000)

    def _backdrop_corners(self, edited: EditedPerson) -> np.ndarray | None:
        """F9: where the photo goes behind the 3D preview, in the ghost's
        space (mirrored with it when the mirror is on)."""
        if self._preview_bgr is None:
            return None
        corners = photo_backdrop_corners(edited.landmarks, edited.image_xy, self._photo_aspect(), edited.visibility)
        if corners is not None and self.mirror_checkbox.isChecked():
            corners = corners * np.array([-1.0, 1.0, 1.0])  # the photo is mirrored with the pose
        return corners

    def _update_backdrop(self, corners: np.ndarray | None) -> None:
        if corners is None or self._preview_bgr is None:
            self.viewport.set_backdrop(None)
            return
        self.viewport.set_backdrop(np.ascontiguousarray(self._preview_bgr[:, :, ::-1]), corners)

    # -- manual correction (F4) ------------------------------------------------------------

    def _edit_key(self, index: int) -> int:
        """The key a person's edits are stored under: their index in a photo,
        their track ID in a video."""
        return self._person_keys[index] if 0 <= index < len(self._person_keys) else index

    def _person_edits(self, index: int) -> PersonEdits:
        return self._edits.get(self._edit_key(index), PersonEdits())

    def _set_person_edits(self, key: int, edits: PersonEdits) -> None:
        """Used by the undo commands: stores the edits and refreshes everything."""
        if edits.is_empty():
            self._edits.pop(key, None)
        else:
            self._edits[key] = edits
        if key == self._edit_key(self.person_combo.currentIndex()):
            self._update_preview_for_selected_person()

    def _push_edit(self, index: int, after: PersonEdits, text: str) -> None:
        before = self._person_edits(index)
        if after != before:
            self.undo_stack.push(_EditCommand(self, self._edit_key(index), before, after, text))

    def move_joint_to(self, index: int, joint: int, xy: tuple[float, float] | None) -> None:
        """F4: moves one of person `index`'s joints to a normalized photo
        position (None: back to where it was detected), as an undo step; the
        limb is re-solved from the moved joint."""
        label = JOINT_LABELS.get(joint, f"joint {joint}")
        text = f"Reset {label}" if xy is None else f"Move {label}"
        self._push_edit(index, self._person_edits(index).with_joint_moved(joint, xy), text)

    def _on_joint_dragged(self, joint: int, x: float, y: float) -> None:
        index = self.person_combo.currentIndex()
        if 0 <= index < len(self._people) and self._video is None:
            self.move_joint_to(index, joint, (x, y))

    def flip_bone_depth(self, index: int, bone: str) -> None:
        """Toggles a limb's depth flip for one person, as an undo step."""
        self._push_edit(index, self._person_edits(index).with_flip_toggled(bone), f"Flip depth of {bone}")

    def _set_override(self, bone: str, angles: tuple[float, float, float] | None) -> None:
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return
        text = f"Reset {bone}" if angles is None else f"Rotate {bone}"
        self._push_edit(index, self._person_edits(index).with_override(bone, angles), text)

    def _on_adjust_changed(self, _value: float) -> None:
        if self._syncing_adjust:
            return
        self._set_override(self.adjust_bone_combo.currentText(), tuple(spin.value() for spin in self.adjust_spins))

    def _sync_adjust_controls(self) -> None:
        """Shows the selected bone's stored adjustment in the spin boxes."""
        index = self.person_combo.currentIndex()
        angles = self._person_edits(index).override_for(self.adjust_bone_combo.currentText())
        self._syncing_adjust = True
        try:
            for spin, value in zip(self.adjust_spins, angles):
                spin.setValue(value)
        finally:
            self._syncing_adjust = False

    def _on_reset_edits(self) -> None:
        index = self.person_combo.currentIndex()
        if 0 <= index < len(self._people):
            self._push_edit(index, PersonEdits(), "Reset all edits")

    # -- export -------------------------------------------------------------------------


    def _current_live_link_document(self) -> dict | None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return None
        rotations, confidence = inputs
        rotations = self._video_frame_rotations(self.person_combo.currentIndex(), rotations)
        hip_height, leg_length = self._root_for(self.person_combo.currentIndex(), rotations)
        return build_pose_document(
            rotations, source_note="Sent live from PoseCraft", bone_confidence=confidence,
            hip_height=hip_height, leg_length=leg_length,
            out_of_frame=self._out_of_frame_for(self.person_combo.currentIndex()),
        )

    def _current_animation_live_link_document(self, track: int) -> dict | None:
        if self._video is None:
            return None
        frame_indices, rotations_by_frame, roots = self._video_track_frames(track)
        if not rotations_by_frame:
            return None
        return build_animation_document(
            list(zip(frame_indices, rotations_by_frame)), self._video.fps,
            source=f"{Path(self._video.path).name}, Person {track + 1}",
            root_positions_cm=roots if len(roots) == len(rotations_by_frame) else None,
        )

    def _start_live_link_send(self, document: dict, automatic: bool = False) -> None:
        if self._send_worker is not None and self._send_worker.isRunning():
            if not automatic:
                self.statusBar().showMessage("Still sending to Blender -- try again in a moment")
            return
        if not automatic:
            self._auto_send_paused = False  # a manual send is how a paused auto-send resumes
        animation = document.get("format") == ANIMATION_FORMAT_NAME
        host = self.live_link_host_edit.text().strip() or live_link.DEFAULT_HOST
        port = self.live_link_port_spin.value()
        self._send_origin = "auto" if automatic else "manual"
        self.send_to_blender_btn.setEnabled(False)
        self.send_animation_btn.setEnabled(False)
        what = f"{len(document.get('frames') or [])}-frame animation" if animation else "pose"
        self.statusBar().showMessage(
            f"{'Auto-sending' if automatic else 'Sending'} {what} to Blender at {host}:{port}..."
        )
        self._send_worker = SendToBlenderWorker(document, host, port)
        self._send_worker.finished_ok.connect(self._on_send_to_blender_finished)
        self._send_worker.failed.connect(self._on_send_to_blender_failed)
        self._send_worker.start()


    def _on_send_to_blender(self) -> None:
        document = self._current_live_link_document()
        if document is None:
            return
        self._start_live_link_send(document, automatic=False)

    def _on_send_animation_to_blender(self) -> None:
        track = self._current_track()
        if track is None:
            return
        document = self._current_animation_live_link_document(track)
        if document is None:
            return
        self._start_live_link_send(document, automatic=False)

    def _on_send_to_blender_finished(self, reply: dict | None) -> None:
        self.send_to_blender_btn.setEnabled(bool(self._people))
        self.send_animation_btn.setEnabled(self._video is not None and bool(self._video.solved_frames()))
        message, ok = _live_link_reply_message(reply)
        self.statusBar().showMessage(message)
        if not ok and not self._closing and self._send_origin != "auto":
            QMessageBox.warning(self, "Send to Blender", message)
        self._send_origin = "manual"

    def _on_send_to_blender_failed(self, message: str) -> None:
        self.send_to_blender_btn.setEnabled(bool(self._people))
        self.send_animation_btn.setEnabled(self._video is not None and bool(self._video.solved_frames()))
        if self._send_origin == "auto" and not self._closing:
            # Don't retry on every later change: an unreachable Live Link would
            # mean a failed connection (and a status flash) per edit or frame.
            self._auto_send_paused = True
            self._auto_send_timer.stop()
            self.statusBar().showMessage(
                "Auto-send paused: Blender's Live Link isn't reachable -- start it in Blender, then Send "
                "manually or re-tick Auto-send to resume"
            )
            self._send_origin = "manual"
            return
        self.statusBar().showMessage("Send to Blender failed")
        if self._closing:
            self._send_origin = "manual"
            return
        QMessageBox.warning(
            self,
            "Send to Blender",
            "Could not reach Blender's Live Link.\n\n"
            "In Blender, open the PoseCraft sidebar tab and click 'Start Live Link', "
            "then try again.\n\n"
            f"Details: {message}",
        )
        self._send_origin = "manual"

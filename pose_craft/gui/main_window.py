from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
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
    save_pose_file,
    set_low_confidence_threshold,
)
from pose_craft.export.bvh_export import floor_root_position, save_bvh_animation, save_bvh_file
from pose_craft.export.live_link import send_animation_document, send_pose_document
from pose_craft.gui.batch_dialog import BatchDialog, BatchSettings
from pose_craft.gui.photo_view import PhotoOverlayPerson, PhotoView, legend_entries, legend_swatch
from pose_craft.gui.viewport_3d import SkeletonViewport, photo_backdrop_corners
from pose_craft.imaging import PHOTO_GUIDANCE, assess_photo_quality, imread_unicode
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

# BVH or binary glTF (F11), chosen by the file name's extension in the save dialog.
SKELETON_FILTERS = "BVH (*.bvh);;glTF binary (*.glb)"


def save_skeleton_animation(frames, path, *, frame_time, root_positions):
    if str(path).lower().endswith(".glb"):
        from pose_craft.export.gltf_export import save_gltf_animation

        save_gltf_animation(frames, path, frame_time=frame_time, root_positions=root_positions)
    else:
        save_bvh_animation(frames, path, frame_time=frame_time, root_positions=root_positions)


log = logging.getLogger(__name__)

# How long closeEvent waits for a background worker before hiding the window
# and finishing the close once the worker returns (a download blocked in a
# network read can take up to landmarker.DOWNLOAD_TIMEOUT_SECONDS to notice a
# cancel).
_CLOSE_WAIT_MS = 5000

PHOTO_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
_PHOTO_FILTER = "Images (" + " ".join(f"*{s}" for s in PHOTO_SUFFIXES) + ")"
_VIDEO_FILTER = "Videos (" + " ".join(f"*{s}" for s in VIDEO_SUFFIXES) + ")"
MAX_RECENT_FILES = 8
# A click farther than this (as a fraction of the photo's diagonal) from every
# person's drawn skeleton doesn't change the selection.
PICK_MAX_DISTANCE = 0.05
DEFAULT_VIDEO_ROTATION_SMOOTHING = 35
DEFAULT_AUTO_SEND_DEBOUNCE_MS = 350


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


def _update_recent_files(recent: list[str], path: str, limit: int = MAX_RECENT_FILES) -> list[str]:
    """`path` moved to the front of the recent-files list (case-insensitively
    unique, as Windows paths are), trimmed to `limit`."""
    key = str(Path(path)).lower()
    kept = [p for p in recent if str(Path(p)).lower() != key]
    return [str(Path(path))] + kept[: limit - 1]


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


class MainWindow(QMainWindow):
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

    def _build_ui(self) -> None:
        # Three panes side by side in a splitter: the controls (scrollable, so
        # they never set the window's minimum height), the photo, and the 3D
        # preview. Stacking the photo on top of the controls used to make the
        # window's minimum height 1124 px, taller than a 1080p screen.
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self._build_detect_group())
        controls_layout.addWidget(self._build_person_group())
        controls_layout.addWidget(self._build_correction_group())
        controls_layout.addWidget(self._build_video_group())
        controls_layout.addWidget(self._build_export_group())
        controls_layout.addWidget(self._build_live_link_group())
        controls_layout.addStretch(1)
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(controls.minimumSizeHint().width() + 24)
        controls_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        photo_pane = QWidget()
        photo_layout = QVBoxLayout(photo_pane)
        photo_layout.setContentsMargins(0, 0, 0, 0)
        photo_layout.addWidget(self._build_banner())
        self.photo_view = PhotoView()
        self.photo_view.setAccessibleName("Photo")
        self.photo_view.clicked_normalized.connect(self._on_photo_clicked)
        self.photo_view.context_requested.connect(self._on_photo_context)
        self.photo_view.joint_dragged.connect(self._on_joint_dragged)
        photo_layout.addWidget(self.photo_view, stretch=1)
        photo_layout.addWidget(self._build_legend())

        preview_pane = QWidget()
        preview_layout = QVBoxLayout(preview_pane)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_label = QLabel("3D preview (the exported rotations; faint dots are the raw landmarks)")
        preview_label.setWordWrap(True)
        preview_layout.addWidget(preview_label)
        self.viewport = SkeletonViewport()
        self.viewport.setMinimumSize(240, 240)
        self.viewport.setToolTip("Hover a bone for its name, local rotation and confidence.")
        self.viewport.setAccessibleName("3D preview")
        preview_layout.addWidget(self.viewport, stretch=1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(controls_scroll)
        self.splitter.addWidget(photo_pane)
        self.splitter.addWidget(preview_pane)
        self.splitter.setCollapsible(0, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 1)
        self.setCentralWidget(self.splitter)

        self.setStatusBar(QStatusBar())
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(180)
        self.progress_bar.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().showMessage("Ready")

    def _apply_accessibility_style(self) -> None:
        """High-visibility keyboard focus cues on top of the platform theme."""
        self.setStyleSheet(
            """
            QPushButton:focus, QToolButton:focus, QCheckBox:focus, QComboBox:focus,
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QSlider:focus {
                outline: 2px solid #f6c453;
                outline-offset: 1px;
                border: 1px solid #f6c453;
            }
            """
        )

    def _build_banner(self) -> QWidget:
        """A non-modal warning strip above the photo (F6): photo-quality notes
        used to be a modal popup on every load."""
        self.banner = QWidget()
        self.banner.setStyleSheet("background:#5a4a1a; color:#fff3c4;")
        layout = QHBoxLayout(self.banner)
        layout.setContentsMargins(8, 4, 4, 4)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        layout.addWidget(self.banner_label, stretch=1)
        close = QToolButton()
        close.setText("✕")
        close.setAccessibleName("Dismiss warning")
        close.clicked.connect(lambda: self.banner.setVisible(False))
        layout.addWidget(close)
        self.banner.setVisible(False)
        return self.banner

    def _build_legend(self) -> QWidget:
        """Swatches drawn with the overlay's own colours and line styles (F3)."""
        legend = QWidget()
        layout = QGridLayout(legend)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(0)
        for i, (color, style, text) in enumerate(legend_entries()):
            swatch = QLabel()
            swatch.setPixmap(legend_swatch(color, style))
            row, column = divmod(i, 2)
            layout.addWidget(swatch, row, 2 * column)
            layout.addWidget(QLabel(text), row, 2 * column + 1)
        layout.setColumnStretch(4, 1)
        legend.setToolTip(self._legend_tooltip())
        self.legend = legend
        return legend

    @staticmethod
    def _legend_tooltip() -> str:
        return (
            f"Low-confidence bones (visibility under {low_confidence_threshold():.0%}) are dashed: the landmarks "
            "behind them were likely hidden, so check them before exporting. Dotted bones reach outside the "
            "photo (a cut-off limb): their pose is a guess. Right-click a limb to flip its depth."
        )

    def _build_detect_group(self) -> QGroupBox:
        group = QGroupBox("1. Load && Detect")
        layout = QVBoxLayout(group)

        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Photo...")
        self.load_btn.setAccessibleName("Load photo")
        self.load_btn.clicked.connect(self._on_load_photo)
        load_row.addWidget(self.load_btn)
        self.load_video_btn = QPushButton("Load Video...")
        self.load_video_btn.setAccessibleName("Load video")
        self.load_video_btn.setToolTip("Track one person through a video clip and export it as an animated BVH.")
        self.load_video_btn.clicked.connect(self._on_load_video)
        load_row.addWidget(self.load_video_btn)
        layout.addLayout(load_row)

        # The tips used to be a permanent multi-line label; they're a toggle now
        # so they don't cost the controls column its height.
        tips_text = "For best results:\n" + "\n".join(f"- {tip}" for tip in PHOTO_GUIDANCE)
        self.load_btn.setToolTip(tips_text)
        self.tips_toggle = QPushButton("Photo tips")
        self.tips_toggle.setCheckable(True)
        self.tips_toggle.setFlat(True)
        tips_label = QLabel(tips_text)
        tips_label.setWordWrap(True)
        tips_label.setVisible(False)
        self.tips_toggle.toggled.connect(tips_label.setVisible)
        layout.addWidget(self.tips_toggle)
        layout.addWidget(tips_label)

        max_people_row = QHBoxLayout()
        max_people_label = QLabel("Max &people to detect:")
        max_people_row.addWidget(max_people_label)
        self.max_people_spin = QSpinBox()
        self.max_people_spin.setRange(1, 8)
        self.max_people_spin.setValue(4)
        self.max_people_spin.setAccessibleName("Maximum people to detect")
        self.max_people_spin.setToolTip(
            "A person beyond this count in a group photo won't be detected at all."
        )
        max_people_label.setBuddy(self.max_people_spin)
        max_people_row.addWidget(self.max_people_spin)
        layout.addLayout(max_people_row)

        model_row = QHBoxLayout()
        model_label = QLabel("Model &quality:")
        model_row.addWidget(model_label)
        self.model_variant_combo = QComboBox()
        self.model_variant_combo.addItems(list(MODEL_URLS))
        self.model_variant_combo.setCurrentText("full")
        self.model_variant_combo.setAccessibleName("Model quality")
        self.model_variant_combo.setToolTip(
            "lite is fastest and best for low-end machines; heavy is the most accurate "
            "but slowest. Changing this after a solve rebuilds the pose model (and downloads "
            "it on first use) on the next Solve Pose."
        )
        model_label.setBuddy(self.model_variant_combo)
        model_row.addWidget(self.model_variant_combo)
        layout.addLayout(model_row)

        self.use_gpu_checkbox = QCheckBox("Use &GPU when available")
        self.use_gpu_checkbox.setAccessibleName("Use GPU when available")
        self.use_gpu_checkbox.setChecked(True)
        self.use_gpu_checkbox.setToolTip(
            "Runs pose estimation on the GPU through ONNX Runtime: DirectML (any DirectX 12 GPU) "
            "in a DirectML build, CUDA (NVIDIA) in a CUDA build. The GPU is used only when a hardware "
            "GPU is found (DirectML ignores software renderers) and a check run at load time gives the "
            "same results as the CPU; "
            "otherwise it falls back to the CPU -- the status bar says which one is used, and why."
        )
        layout.addWidget(self.use_gpu_checkbox)

        self.group_photo_checkbox = QCheckBox("Group photo (slower)")
        self.group_photo_checkbox.setAccessibleName("Group photo")
        self.group_photo_checkbox.setToolTip(
            "Also searches overlapping tiles of the photo, so people whose faces are only a few pixels "
            "wide in a wide group shot are found. Without it, tiles are only tried when nobody is found."
        )
        layout.addWidget(self.group_photo_checkbox)

        self.hands_checkbox = QCheckBox("&Hands and fingers")
        self.hands_checkbox.setAccessibleName("Hands and fingers")
        self.hands_checkbox.setToolTip(
            "Also runs MediaPipe's hand landmark model on each person's hands (cropped from the pose's wrist "
            "and knuckles) and exports 15 finger bones per hand found (Rigify names: thumb.01.L, f_index.01.L, "
            "...) in the JSON, BVH and Live Link. A few ms per hand; the first use downloads the hand model. "
            "Off: the 15-bone body pose exactly as before."
        )
        layout.addWidget(self.hands_checkbox)
        layout.addWidget(self._build_advanced_detect())

        self.solve_btn = QPushButton("Solve Pose")
        self.solve_btn.setAccessibleName("Solve pose")
        self.solve_btn.setToolTip("Detect and solve every person in the photo (F5)")
        self.solve_btn.clicked.connect(self._on_solve_pose)
        self.solve_btn.setEnabled(False)
        layout.addWidget(self.solve_btn)

        return group

    def _build_advanced_detect(self) -> QWidget:
        """B10: the detection and low-confidence thresholds, behind an "Advanced"
        toggle (collapsed by default; both remembered)."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        outer.addWidget(self.advanced_toggle)
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 0, 0, 0)
        self.detection_threshold_spin = QDoubleSpinBox()
        self.detection_threshold_spin.setRange(0.1, 0.9)
        self.detection_threshold_spin.setSingleStep(0.05)
        self.detection_threshold_spin.setValue(0.5)
        self.detection_threshold_spin.setAccessibleName("Detection threshold")
        self.detection_threshold_spin.setToolTip(
            "How sure the person detector must be. Lower it (e.g. 0.3) when a person in a hard photo -- "
            "small, dark, an unusual pose -- isn't found; raise it to drop doubtful detections. "
            "Applies to the next solve."
        )
        grid.addWidget(QLabel("Detection threshold"), 0, 0)
        grid.addWidget(self.detection_threshold_spin, 0, 1)
        self.low_confidence_spin = QDoubleSpinBox()
        self.low_confidence_spin.setRange(0.1, 0.9)
        self.low_confidence_spin.setSingleStep(0.05)
        self.low_confidence_spin.setValue(low_confidence_threshold())
        self.low_confidence_spin.setAccessibleName("Low-confidence threshold")
        self.low_confidence_spin.setToolTip(
            "Bones whose landmarks' visibility is below this are flagged low-confidence (dashed here, "
            "low_confidence in exports). Raise it to be warned about more bones."
        )
        self.low_confidence_spin.valueChanged.connect(self._on_low_confidence_changed)
        grid.addWidget(QLabel("Low confidence below"), 1, 0)
        grid.addWidget(self.low_confidence_spin, 1, 1)
        panel.setVisible(False)
        outer.addWidget(panel)

        def toggle(checked: bool) -> None:
            panel.setVisible(checked)
            self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        self.advanced_toggle.toggled.connect(toggle)
        self.advanced_panel = panel
        return container

    def _on_low_confidence_changed(self, value: float) -> None:
        set_low_confidence_threshold(value)
        if getattr(self, "legend", None) is not None:
            self.legend.setToolTip(self._legend_tooltip())
        if self._people:
            self._update_preview_for_selected_person()

    def _build_person_group(self) -> QGroupBox:
        group = QGroupBox("2. Select Person")
        layout = QVBoxLayout(group)

        person_row = QHBoxLayout()
        person_label = QLabel("P&erson:")
        person_row.addWidget(person_label)
        self.person_combo = QComboBox()
        self.person_combo.setAccessibleName("Selected person")
        self.person_combo.currentIndexChanged.connect(self._on_person_changed)
        person_label.setBuddy(self.person_combo)
        person_row.addWidget(self.person_combo)
        layout.addLayout(person_row)
        click_hint = QLabel("(or click the person in the photo; Page Up/Down cycles)")
        click_hint.setWordWrap(True)
        layout.addWidget(click_hint)

        self.mirror_checkbox = QCheckBox("&Mirror left/right")
        self.mirror_checkbox.setAccessibleName("Mirror left and right")
        self.mirror_checkbox.setToolTip(
            "Body orientation fix: swaps the left and right sides if the estimate labeled them the wrong way round."
        )
        self.mirror_checkbox.stateChanged.connect(self._on_person_changed)
        layout.addWidget(self.mirror_checkbox)

        orientation_row = QHBoxLayout()
        orientation_label = QLabel("Body &orientation fix:")
        orientation_row.addWidget(orientation_label)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(list(ORIENTATION_CORRECTIONS.keys()))
        self.orientation_combo.setAccessibleName("Body orientation fix")
        self.orientation_combo.setToolTip(
            "If the solved pose comes out upside-down or facing the wrong way -- see "
            "_world_to_engine()'s axis-convention caveat -- pick the correction "
            "that fixes it. Applied as an extra rotation on top of the solved pose, live in "
            "the 3D preview and in every export."
        )
        self.orientation_combo.currentIndexChanged.connect(self._on_person_changed)
        orientation_label.setBuddy(self.orientation_combo)
        orientation_row.addWidget(self.orientation_combo)
        layout.addLayout(orientation_row)

        return group

    def _build_correction_group(self) -> QGroupBox:
        """Manual correction (F4): rotate one bone about its own axes, and
        (from the photo's right-click menu) flip a limb's depth."""
        group = QGroupBox("3. Correct")
        layout = QVBoxLayout(group)
        hint = QLabel(
            "Drag a ringed joint on the photo to move it (its limb is re-solved); right-click a limb to flip "
            "its depth (toward / away from the camera)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        bone_row = QHBoxLayout()
        bone_label = QLabel("Adjust &bone:")
        bone_row.addWidget(bone_label)
        self.adjust_bone_combo = QComboBox()
        self.adjust_bone_combo.addItems(BONE_NAMES)
        self.adjust_bone_combo.currentIndexChanged.connect(lambda _i: self._sync_adjust_controls())
        bone_label.setBuddy(self.adjust_bone_combo)
        bone_row.addWidget(self.adjust_bone_combo)
        layout.addLayout(bone_row)

        angles_row = QHBoxLayout()
        self.adjust_spins: list[QDoubleSpinBox] = []
        for axis in "XYZ":
            angles_row.addWidget(QLabel(axis))
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setSingleStep(5.0)
            spin.setDecimals(1)
            spin.setSuffix("°")
            spin.setKeyboardTracking(False)
            spin.setAccessibleName(f"Rotate the bone about its own {axis} axis")
            spin.setToolTip(
                "Degrees about the bone's own axes, as in Blender: Y runs along the bone (twist), "
                "X and Z bend it. Ctrl+Z undoes."
            )
            spin.valueChanged.connect(self._on_adjust_changed)
            self.adjust_spins.append(spin)
            angles_row.addWidget(spin)
        layout.addLayout(angles_row)

        reset_row = QHBoxLayout()
        self.reset_bone_btn = QPushButton("Reset bone")
        self.reset_bone_btn.clicked.connect(lambda: self._set_override(self.adjust_bone_combo.currentText(), None))
        reset_row.addWidget(self.reset_bone_btn)
        self.reset_edits_btn = QPushButton("Reset all edits")
        self.reset_edits_btn.clicked.connect(self._on_reset_edits)
        reset_row.addWidget(self.reset_edits_btn)
        layout.addLayout(reset_row)
        return group

    def _build_video_group(self) -> QGroupBox:
        group = QGroupBox("Video")
        layout = QVBoxLayout(group)
        frame_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setAccessibleName("Video frame")
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        frame_row.addWidget(self.frame_slider, stretch=1)
        self.frame_label = QLabel("no video")
        frame_row.addWidget(self.frame_label)
        layout.addLayout(frame_row)
        smoothing_row = QHBoxLayout()
        smoothing_label = QLabel("Rotation smoo&thing:")
        smoothing_row.addWidget(smoothing_label)
        self.video_rotation_smoothing_spin = QSpinBox()
        self.video_rotation_smoothing_spin.setRange(0, 100)
        self.video_rotation_smoothing_spin.setValue(DEFAULT_VIDEO_ROTATION_SMOOTHING)
        self.video_rotation_smoothing_spin.setSuffix("%")
        self.video_rotation_smoothing_spin.setAccessibleName("Rotation smoothing")
        self.video_rotation_smoothing_spin.setToolTip(
            "Quaternion-aware smoothing of the animation's joint rotations (the preview, BVH export and "
            "Send Animation all use it). Zero-phase, so it doesn't lag behind the video. "
            "0% keeps every frame as solved; higher values reduce jitter after tracking."
        )
        self.video_rotation_smoothing_spin.valueChanged.connect(lambda _value: self._on_person_changed())
        smoothing_label.setBuddy(self.video_rotation_smoothing_spin)
        smoothing_row.addWidget(self.video_rotation_smoothing_spin)
        smoothing_row.addStretch(1)
        layout.addLayout(smoothing_row)
        self.export_animation_btn = QPushButton("Export Animation BVH...")
        self.export_animation_btn.setAccessibleName("Export animation BVH")
        self.export_animation_btn.setToolTip(
            "The selected person's solved frames as one BVH animation, smoothed, standing on the floor, with "
            "planted feet driving the root's motion."
        )
        self.export_animation_btn.clicked.connect(self._on_export_animation)
        layout.addWidget(self.export_animation_btn)
        self.export_all_tracks_btn = QPushButton("Export All People (BVH)...")
        self.export_all_tracks_btn.setAccessibleName("Export all tracked people as BVH")
        self.export_all_tracks_btn.setToolTip(
            "One animated BVH per tracked person (<video>_person<N>.bvh), to a folder you choose."
        )
        self.export_all_tracks_btn.clicked.connect(self._on_export_all_tracks)
        layout.addWidget(self.export_all_tracks_btn)
        self.send_animation_btn = QPushButton("Send Animation to Blender")
        self.send_animation_btn.setAccessibleName("Send animation to Blender")
        self.send_animation_btn.setToolTip(
            "Live Link the selected tracked person as keyed animation, spaced at the clip's fps from Blender's "
            "current frame."
        )
        self.send_animation_btn.clicked.connect(self._on_send_animation_to_blender)
        layout.addWidget(self.send_animation_btn)
        self.video_group = group
        group.setVisible(False)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("4. Export")
        layout = QVBoxLayout(group)

        self.export_btn = QPushButton("Export Pose JSON...")
        self.export_btn.setAccessibleName("Export pose JSON")
        self.export_btn.clicked.connect(self._on_export_pose)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)

        self.export_all_btn = QPushButton("Export All People...")
        self.export_all_btn.setAccessibleName("Export all people as pose JSON")
        self.export_all_btn.clicked.connect(self._on_export_all)
        self.export_all_btn.setEnabled(False)
        layout.addWidget(self.export_all_btn)

        self.export_bvh_btn = QPushButton("Export BVH...")
        self.export_bvh_btn.setAccessibleName("Export BVH")
        self.export_bvh_btn.clicked.connect(self._on_export_bvh)
        self.export_bvh_btn.setEnabled(False)
        layout.addWidget(self.export_bvh_btn)

        self.export_daz_btn = QPushButton("Export DAZ Pose (Genesis 9)...")
        self.export_daz_btn.setAccessibleName("Export DAZ Studio pose preset for Genesis 9")
        self.export_daz_btn.setToolTip("A .duf pose preset: in DAZ Studio, select a Genesis 9 figure and load it")
        self.export_daz_btn.clicked.connect(self._on_export_daz)
        self.export_daz_btn.setEnabled(False)
        layout.addWidget(self.export_daz_btn)

        self.export_all_bvh_btn = QPushButton("Export All People (BVH)...")
        self.export_all_bvh_btn.setAccessibleName("Export all people as BVH")
        self.export_all_bvh_btn.setToolTip("One BVH per detected person (<photo>_person<N>.bvh), to a folder you choose.")
        self.export_all_bvh_btn.clicked.connect(self._on_export_all_bvh)
        self.export_all_bvh_btn.setEnabled(False)
        layout.addWidget(self.export_all_bvh_btn)

        return group

    def _build_live_link_group(self) -> QGroupBox:
        group = QGroupBox("5. Blender Live Link")
        layout = QVBoxLayout(group)

        live_link_row = QHBoxLayout()
        host_label = QLabel("Blender &host:")
        live_link_row.addWidget(host_label)
        self.live_link_host_edit = QLineEdit(live_link.DEFAULT_HOST)
        self.live_link_host_edit.setAccessibleName("Blender host")
        self.live_link_host_edit.setMaximumWidth(100)
        self.live_link_host_edit.setToolTip(
            "The Blender add-on only ever listens on 127.0.0.1 (localhost) -- change this "
            "only if PoseCraft and Blender already talk over some other configured "
            "loopback/tunnel, not to reach Blender on a different machine."
        )
        host_label.setBuddy(self.live_link_host_edit)
        live_link_row.addWidget(self.live_link_host_edit)
        port_label = QLabel("po&rt:")
        live_link_row.addWidget(port_label)
        self.live_link_port_spin = QSpinBox()
        self.live_link_port_spin.setRange(1, 65535)
        self.live_link_port_spin.setValue(live_link.DEFAULT_PORT)
        self.live_link_port_spin.setAccessibleName("Blender port")
        self.live_link_port_spin.setToolTip(
            "Must match the port set in Blender's PoseCraft panel before Live Link is started there."
        )
        port_label.setBuddy(self.live_link_port_spin)
        live_link_row.addWidget(self.live_link_port_spin)
        layout.addLayout(live_link_row)

        self.send_to_blender_btn = QPushButton("Send to Blender (Live Link)")
        self.send_to_blender_btn.setAccessibleName("Send to Blender")
        self.send_to_blender_btn.setToolTip("Ctrl+B")
        self.send_to_blender_btn.clicked.connect(self._on_send_to_blender)
        self.send_to_blender_btn.setEnabled(False)
        layout.addWidget(self.send_to_blender_btn)

        auto_row = QHBoxLayout()
        self.auto_send_checkbox = QCheckBox("A&uto-send on change")
        self.auto_send_checkbox.setAccessibleName("Auto-send on change")
        self.auto_send_checkbox.setToolTip(
            "After a short debounce, re-send the currently shown pose whenever the selection, video frame, "
            "manual edits or pose options change. Pauses itself if Blender's Live Link can't be reached "
            "(a manual Send, or ticking this again, resumes it)."
        )
        self.auto_send_checkbox.toggled.connect(self._on_auto_send_toggled)
        auto_row.addWidget(self.auto_send_checkbox)
        auto_row.addStretch(1)
        layout.addLayout(auto_row)

        debounce_row = QHBoxLayout()
        debounce_label = QLabel("&Debounce:")
        debounce_row.addWidget(debounce_label)
        self.auto_send_debounce_spin = QSpinBox()
        self.auto_send_debounce_spin.setRange(50, 5000)
        self.auto_send_debounce_spin.setSingleStep(50)
        self.auto_send_debounce_spin.setValue(DEFAULT_AUTO_SEND_DEBOUNCE_MS)
        self.auto_send_debounce_spin.setAccessibleName("Auto-send debounce")
        self.auto_send_debounce_spin.setSuffix(" ms")
        self.auto_send_debounce_spin.setToolTip("How long PoseCraft waits after a change before auto-sending.")
        debounce_label.setBuddy(self.auto_send_debounce_spin)
        debounce_row.addWidget(self.auto_send_debounce_spin)
        debounce_row.addStretch(1)
        layout.addLayout(debounce_row)

        return group

    def _action(self, menu: QMenu, text: str, slot, shortcut=None, checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            shortcuts = shortcut if isinstance(shortcut, list) else [shortcut]
            action.setShortcuts([QKeySequence(s) for s in shortcuts])
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        """Menus with keyboard shortcuts (F7) for the load -> solve -> send loop."""
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self.open_action = self._action(file_menu, "&Open Photo...", self._on_load_photo, QKeySequence.Open)
        self.open_video_action = self._action(file_menu, "Open &Video...", self._on_load_video, "Ctrl+Shift+O")
        self.recent_menu = file_menu.addMenu("Open &Recent")
        file_menu.addSeparator()
        self.export_action = self._action(file_menu, "&Export Pose JSON...", self._on_export_pose, "Ctrl+S")
        self.export_bvh_action = self._action(file_menu, "Export &BVH...", self._on_export_bvh, "Ctrl+Shift+S")
        self.export_daz_action = self._action(file_menu, "Export &DAZ Pose (Genesis 9)...", self._on_export_daz)
        self.export_all_action = self._action(file_menu, "Export &All People...", self._on_export_all)
        self.batch_action = self._action(file_menu, "Batch &Process Folder...", self._on_batch_folder)
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, QKeySequence.Quit)

        edit_menu = bar.addMenu("&Edit")
        undo = self.undo_stack.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.Undo)
        redo = self.undo_stack.createRedoAction(self, "&Redo")
        redo.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")])
        edit_menu.addAction(undo)
        edit_menu.addAction(redo)

        pose_menu = bar.addMenu("&Pose")
        self.solve_action = self._action(pose_menu, "&Solve", self._on_solve_pose, ["F5", "Ctrl+Return"])
        self.send_action = self._action(pose_menu, "Send to &Blender", self._on_send_to_blender, "Ctrl+B")
        self.mirror_action = self._action(pose_menu, "&Mirror Left/Right", self.mirror_checkbox.toggle, "M")
        self.next_person_action = self._action(pose_menu, "&Next Person", lambda: self._cycle_person(1), "PgDown")
        self.previous_person_action = self._action(
            pose_menu, "&Previous Person", lambda: self._cycle_person(-1), "PgUp"
        )
        self.cancel_action = self._action(pose_menu, "&Cancel", self._on_cancel, "Esc")

        view_menu = bar.addMenu("&View")
        self._action(view_menu, "&Frame Skeleton", self.viewport.frame_skeleton, "F")
        self._action(view_menu, "Fro&nt", lambda: self.viewport.set_camera_preset("front"), "1")
        self._action(view_menu, "&Side", lambda: self.viewport.set_camera_preset("side"), "3")
        self._action(view_menu, "&Top", lambda: self.viewport.set_camera_preset("top"), "7")
        self._action(view_menu, "&Reset Camera", lambda: self.viewport.set_camera_preset("default"), "0")
        self.labels_action = self._action(view_menu, "Bone &Labels", self._on_toggle_labels, "L", checkable=True)
        self.hover_action = self._action(view_menu, "&Hover Info", self._on_toggle_hover, "H", checkable=True)
        self.hover_action.setChecked(True)
        self.backdrop_action = self._action(
            view_menu, "Photo &Backdrop", self._on_toggle_backdrop, "Shift+B", checkable=True
        )

        help_menu = bar.addMenu("&Help")
        self.diagnostics_action = self._action(help_menu, "&Diagnostics...", self._on_show_diagnostics)
        self._action(help_menu, "&About PoseCraft", self._on_about)

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
        s.setValue("live_link/auto_send", "true" if self.auto_send_checkbox.isChecked() else "false")
        s.setValue("live_link/auto_send_debounce_ms", self.auto_send_debounce_spin.value())
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.splitter.saveState())
        s.sync()

    def _recent_files(self) -> list[str]:
        value = self._settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [str(v) for v in (value or [])]

    def _remember_file(self, path: str) -> None:
        self._settings.setValue("recent_files", _update_recent_files(self._recent_files(), path))
        self._settings.setValue("last_dir", str(Path(path).parent))
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent_files()
        for path in recent:
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.open_path(p))
        self.recent_menu.setEnabled(bool(recent))

    def _last_dir(self) -> str:
        return str(self._settings.value("last_dir", ""))

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
                       self.next_person_action, self.previous_person_action):
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

    def dragEnterEvent(self, event) -> None:
        if self._dropped_path(event.mimeData()) is not None and not self._worker_running():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = self._dropped_path(event.mimeData())
        if path is not None:
            event.acceptProposedAction()
            self.open_path(path)

    @staticmethod
    def _dropped_path(mime) -> str | None:
        """The first dropped local file that's a photo or a video, if any."""
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in PHOTO_SUFFIXES + VIDEO_SUFFIXES:
                return url.toLocalFile()
        return None

    def open_path(self, path: str) -> None:
        """Opens a photo or a video by extension (drag-and-drop, recent files)."""
        if Path(path).suffix.lower() in VIDEO_SUFFIXES:
            self.load_video(path)
        else:
            self.load_photo(path)

    def _on_load_photo(self) -> None:
        # Guards against the same race closeEvent guards against: loading a new
        # photo while a solve for the OLD photo is still running would let that
        # solve's results land on the new photo once it finishes, and risks two
        # threads calling the shared PoseEstimator concurrently. load_btn is
        # disabled for the duration of a solve (see _set_busy) so this
        # shouldn't be reachable from the UI, but checking here too costs nothing.
        if self._worker_running():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Photo", self._last_dir(), _PHOTO_FILTER)
        if path:
            self.load_photo(path)

    def load_photo(self, path: str) -> bool:
        if self._worker_running():
            return False
        image_bgr = imread_unicode(path)
        if image_bgr is None:
            QMessageBox.warning(self, "Load Photo", f"Could not read image:\n{path}")
            return False

        self._clear_video()
        self._image_bgr = image_bgr
        self._source_path = path
        self._set_photo_preview(image_bgr)
        self._clear_results()
        self.solve_btn.setEnabled(True)
        self.solve_action.setEnabled(True)
        self._remember_file(path)
        self.statusBar().showMessage(f"Loaded {Path(path).name}")

        quality_warnings = assess_photo_quality(image_bgr)
        self.banner_label.setText("\n".join(quality_warnings))
        self.banner.setVisible(bool(quality_warnings))
        return True

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

    def _set_photo_preview(self, image_bgr: np.ndarray) -> None:
        self._preview_bgr = image_bgr
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.photo_view.set_source_pixmap(QPixmap.fromImage(qimage))

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

    def _on_load_video(self) -> None:
        if self._worker_running():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Video", self._last_dir(), _VIDEO_FILTER)
        if path:
            self.load_video(path)

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

    def _on_export_animation(self) -> None:
        track = self._current_track()
        if track is None:
            return
        _indices, frames, roots = self._video_track_frames(track)
        if not frames:
            return
        stem = Path(self._video.path).stem
        name = f"{stem}.bvh" if len(self._video.track_ids()) == 1 else default_export_name(self._video.path, track, ".bvh")
        path, _ = QFileDialog.getSaveFileName(self, "Export Animation BVH", str(Path(self._last_dir()) / name), SKELETON_FILTERS)
        if not path:
            return
        try:
            save_skeleton_animation(frames, path, frame_time=1.0 / self._video.fps, root_positions=roots)
        except OSError as exc:
            self._report_export_error("Export Animation BVH", path, exc)
            return
        self.statusBar().showMessage(f"Exported {len(frames)} frames of Person {track + 1} to {path}")

    def _on_export_all_tracks(self) -> None:
        """B12: one animated BVH per tracked person."""
        if self._video is None or not self._video.track_ids():
            return
        directory = QFileDialog.getExistingDirectory(self, "Export All People To Folder", self._last_dir())
        if not directory:
            return
        tracks = self._video.track_ids()
        paths = {track: Path(directory) / default_export_name(self._video.path, track, ".bvh") for track in tracks}
        existing = [p for p in paths.values() if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return
        for track, path in paths.items():
            _indices, frames, roots = self._video_track_frames(track)
            try:
                save_bvh_animation(frames, path, frame_time=1.0 / self._video.fps, root_positions=roots)
            except OSError as exc:
                self._report_export_error("Export All People", str(path), exc)
                return
        self.statusBar().showMessage(f"Exported {len(tracks)} animation(s) to {directory}")

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

    @staticmethod
    def _document_inputs_for(
        person: DetectedPerson,
        mirror: bool,
        orientation_correction=None,
        edits: PersonEdits | None = None,
        aspect: float = 1.0,
    ) -> tuple[dict, dict]:
        """Shared by _current_person_document_inputs (the selected person) and
        _on_export_all (every detected person): mirror_landmarks swaps the
        left/right rows (and, for 3D positions only, reflects X) -- a 1-D
        `visibility` array gets just the row swap, and it needs to be swapped
        the same way `landmarks` is, or the confidence for e.g. a mirrored thigh.L would
        report the ORIGINAL left leg's visibility instead of the real
        (now swapped-in) right leg's.

        orientation_correction (a scipy Rotation, or None for no change) is the
        same body-orientation fix applied live in the 3D preview -- see
        skeleton.apply_orientation_correction -- composed onto the root bone's
        rotation so every export (JSON, BVH, Live Link) agrees with what's shown
        on screen.

        edits: the person's manual corrections (F4) -- moved joints and depth
        flips on the raw landmarks before mirroring (they name what was
        clicked in the photo), bone adjustments on the solved rotations.
        aspect: the photo's width / height, for turning a joint drag into 3D.

        The person's hand landmarks (B11), when the hands were solved, add
        finger bones; they're mirrored and flipped along with the body.
        """
        return document_inputs(person, mirror, orientation_correction, edits, aspect)

    def _current_person_document_inputs(self):
        index = self.person_combo.currentIndex()
        if index < 0 or index >= len(self._people):
            return None
        return self._person_document_inputs(index)

    def _person_document_inputs(self, index: int):
        return self._document_inputs_for(
            self._people[index],
            self.mirror_checkbox.isChecked(),
            self._orientation_correction(),
            self._person_edits(index),
            self._photo_aspect(),
        )

    def _out_of_frame_for(self, index: int) -> set[str]:
        """Person `index`'s bones reaching outside the photo (B10), named as exported."""
        return out_of_frame_bones(self._edited(index), self.mirror_checkbox.isChecked())

    def _root_for(self, index: int, rotations: dict) -> tuple[float, float]:
        """(hip_height, leg_length) in meters for person `index`'s export (B6)."""
        return root_for(self._edited(index), self.mirror_checkbox.isChecked(), rotations)

    def _export_path(self, title: str, name: str, file_filter: str) -> str:
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(self._last_dir()) / name), file_filter)
        if path:
            self._settings.setValue("last_dir", str(Path(path).parent))
        return path

    def _on_export_pose(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, confidence = inputs
        index = self.person_combo.currentIndex()

        path = self._export_path(
            "Export Pose JSON", default_export_name(self._source_path, index, ".json"), "Pose JSON (*.json)"
        )
        if not path:
            return

        hip_height, leg_length = self._root_for(index, rotations)
        try:
            save_pose_file(
                rotations, path, source_note="Exported from PoseCraft", bone_confidence=confidence,
                hip_height=hip_height, leg_length=leg_length, out_of_frame=self._out_of_frame_for(index),
            )
        except OSError as exc:
            self._report_export_error("Export Pose JSON", path, exc)
            return
        self.statusBar().showMessage(f"Exported pose to {path}")

    def _on_export_all(self) -> None:
        if not self._people:
            return

        directory = QFileDialog.getExistingDirectory(self, "Export All Poses To Folder", self._last_dir())
        if not directory:
            return

        # Mirror and the orientation fix are both properties of the camera/photo,
        # not of one person, so they apply to every person in a group photo the
        # same way single-person export already applies them via
        # _current_person_document_inputs().
        mirror = self.mirror_checkbox.isChecked()
        orientation_correction = self._orientation_correction()
        paths = [Path(directory) / default_export_name(self._source_path, i, ".json") for i in range(len(self._people))]
        existing = [p for p in paths if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return

        for index, (person, path) in enumerate(zip(self._people, paths)):
            rotations, confidence = self._document_inputs_for(
                person, mirror, orientation_correction, self._person_edits(index), self._photo_aspect()
            )
            hip_height, leg_length = self._root_for(index, rotations)
            try:
                save_pose_file(
                    rotations, path, source_note="Exported from PoseCraft (batch)", bone_confidence=confidence,
                    hip_height=hip_height, leg_length=leg_length, out_of_frame=self._out_of_frame_for(index),
                )
            except OSError as exc:
                self._report_export_error("Export All People", str(path), exc)
                return

        self.statusBar().showMessage(f"Exported {len(self._people)} pose(s) to {directory}")

    def _on_export_bvh(self) -> None:
        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, _confidence = inputs

        path = self._export_path(
            "Export BVH", default_export_name(self._source_path, self.person_combo.currentIndex(), ".bvh"), SKELETON_FILTERS
        )
        if not path:
            return

        try:
            # Standing on the floor (B6): a crouch is exported lowered. .glb: glTF (F11).
            if path.lower().endswith(".glb"):
                from pose_craft.export.gltf_export import save_gltf_pose

                save_gltf_pose(rotations, path, root_position=floor_root_position(rotations))
            else:
                save_bvh_file(rotations, path, root_position=floor_root_position(rotations))
        except OSError as exc:
            self._report_export_error("Export BVH", path, exc)
            return
        self.statusBar().showMessage(f"Exported BVH to {path}")

    def _on_export_daz(self) -> None:
        from pose_craft.export.daz_pose import (
            DazFigureNotFound,
            channels_past_limits,
            load_daz_figure,
            save_daz_pose_file,
        )

        inputs = self._current_person_document_inputs()
        if inputs is None:
            return
        rotations, _confidence = inputs
        index = self.person_combo.currentIndex()
        try:
            figure = load_daz_figure()
        except (DazFigureNotFound, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export DAZ Pose", str(exc))
            return
        path = self._export_path(
            "Export DAZ Pose (Genesis 9)", default_export_name(self._source_path, index, ".duf"),
            "DAZ Studio pose preset (*.duf)",
        )
        if not path:
            return
        hip_height, leg_length = self._root_for(index, rotations)
        try:
            save_daz_pose_file(rotations, path, figure, hip_height=hip_height, leg_length=leg_length)
        except OSError as exc:
            self._report_export_error("Export DAZ Pose", path, exc)
            return
        past = channels_past_limits(rotations, figure)
        note = ""
        if past:
            bones = sorted({bone for bone, _axis, _value, _limit in past})
            note = f" -- beyond Genesis 9's joint limits (DAZ will clamp): {', '.join(bones)}"
        self.statusBar().showMessage(f"Exported DAZ pose to {path}{note}")

    def _on_export_all_bvh(self) -> None:
        if not self._people or self._video is not None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export All BVHs To Folder", self._last_dir())
        if not directory:
            return
        paths = [Path(directory) / default_export_name(self._source_path, i, ".bvh") for i in range(len(self._people))]
        existing = [p for p in paths if p.exists()]
        if existing and QMessageBox.question(
            self, "Export All People (BVH)", f"Overwrite {len(existing)} existing file(s) in\n{directory}?"
        ) != QMessageBox.Yes:
            return
        for index, (person, path) in enumerate(zip(self._people, paths)):
            rotations, _confidence = self._document_inputs_for(
                person, self.mirror_checkbox.isChecked(), self._orientation_correction(), self._person_edits(index),
                self._photo_aspect(),
            )
            try:
                save_bvh_file(rotations, path, root_position=floor_root_position(rotations))
            except OSError as exc:
                self._report_export_error("Export All People (BVH)", str(path), exc)
                return
        self.statusBar().showMessage(f"Exported {len(paths)} BVH file(s) to {directory}")

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

    def _report_export_error(self, title: str, path: str, exc: OSError) -> None:
        """A failed write (read-only folder, locked file, OneDrive placeholder)
        used to raise inside the Qt slot, which the windowed build has no
        console to show -- the export silently did nothing."""
        log.warning("%s failed for %s: %s", title, path, exc)
        self.statusBar().showMessage(f"{title} failed")
        QMessageBox.warning(self, title, f"Could not write:\n{path}\n\n{exc}")

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

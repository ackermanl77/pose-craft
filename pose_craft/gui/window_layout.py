"""MainWindow's widgets, group boxes and menus (C4: split out of main_window.py).

A mixin: the builders run from MainWindow.__init__ and set the widgets and actions as attributes of
the window, connecting them to slots the other parts of MainWindow define."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
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
    QWidgetAction,
)

from pose_craft.estimation.landmarker import (
    MODEL_URLS,
)
from pose_craft.export import live_link
from pose_craft.export.blender_pose import (
    low_confidence_threshold,
    set_low_confidence_threshold,
)
from pose_craft.export.skeleton_presets import SKELETON_CHOICES
from pose_craft.gui.photo_view import PhotoView, legend_entries, legend_swatch
from pose_craft.gui.viewport_3d import DEFAULT_BACKDROP_OPACITY, SkeletonViewport
from pose_craft.imaging import PHOTO_GUIDANCE
from pose_craft.retarget.skeleton import (
    BONE_NAMES,
    ORIENTATION_CORRECTIONS,
)

DEFAULT_VIDEO_ROTATION_SMOOTHING = 35

DEFAULT_AUTO_SEND_DEBOUNCE_MS = 350


class _FitScrollWidth(QObject):
    """Keep a vertical-only scroll area at least as wide as its content plus the scrollbar.

    The minimum used to be taken once while the window was built, before fonts and styles
    settled and without the scrollbar, so the controls panel cut its widest rows off on the
    right (UI1). This refits whenever the content's layout changes."""

    def __init__(self, scroll):
        super().__init__(scroll)
        self.scroll = scroll
        scroll.widget().installEventFilter(self)
        self.fit()

    def fit(self) -> None:
        scroll = self.scroll
        need = (scroll.widget().minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                + 2 * scroll.frameWidth())
        if scroll.minimumWidth() != need:
            scroll.setMinimumWidth(need)

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Show):
            self.fit()
        return False


class LayoutMixin:
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
        # The panel's width is fixed (below); a combo that grew to fit a long item later ("Person 1:
        # 11/15 bones confident...") pushed the content past it and the right side was cut off
        # (UI1). Combos elide instead; their drop-down still lists the whole text.
        for combo in controls.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._controls_width_fitter = _FitScrollWidth(controls_scroll)
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
            /* UI2: the main action (Solve Pose, in the panel and the toolbar) stands out. */
            QPushButton#primaryAction, QToolButton#primaryAction {
                background: #2f6fd6; color: #ffffff; font-weight: bold;
                border: 1px solid #2559ad; border-radius: 4px; padding: 5px 14px;
            }
            QPushButton#primaryAction:hover, QToolButton#primaryAction:hover { background: #3b7de6; }
            QPushButton#primaryAction:disabled, QToolButton#primaryAction:disabled {
                background: #5d6b80; color: #d0d4da; border-color: #5d6b80;
            }
            QPushButton#primaryAction:focus, QToolButton#primaryAction:focus { border: 2px solid #f6c453; }
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
            swatch.setAccessibleName(f"{text} legend swatch")
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

        # Shown instead of the Load and Solve buttons while the toolbar has them (UI5).
        self.toolbar_hint = QLabel("Load a photo or video and solve from the toolbar (Ctrl+O, F5).")
        self.toolbar_hint.setWordWrap(True)
        layout.addWidget(self.toolbar_hint)

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
        self.face_checkbox = QCheckBox("Face e&xpressions")
        self.face_checkbox.setAccessibleName("Face expressions")
        self.face_checkbox.setToolTip(
            "Also reads each person's facial expression (MediaPipe's face landmark and blendshape models, "
            "about 4 MB, downloaded on first use) as the 52 ARKit blendshapes -- jawOpen, mouthSmileLeft, "
            "eyeBlinkRight... -- sent in the JSON and over Live Link, where the Blender add-on sets the "
            "rig's shape keys with those names. Off: no face pass."
        )
        layout.addWidget(self.face_checkbox)
        layout.addWidget(self._build_advanced_detect())

        self.solve_btn = QPushButton("Solve Pose")
        self.solve_btn.setObjectName("primaryAction")
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
            "its depth (toward / away from the camera). In the 3D preview, click a bone and press R to rotate it."
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

        # B22: the skeleton BVH/glTF/FBX files are written on.
        skeleton_row = QHBoxLayout()
        skeleton_row.addWidget(QLabel("Skeleton:"))
        self.skeleton_combo = QComboBox()
        for key, label in SKELETON_CHOICES:
            self.skeleton_combo.addItem(label, key)
        self.skeleton_combo.setAccessibleName("Skeleton for BVH, glTF and FBX files")
        self.skeleton_combo.setToolTip(
            "Bone names, hierarchy and rest pose of BVH, glTF and FBX exports: PoseCraft's own (Rigify names), "
            "Mixamo, the Unreal Engine 5 Mannequin (A-pose rest) or HumanIK (MotionBuilder, Maya)."
        )
        skeleton_row.addWidget(self.skeleton_combo, 1)
        layout.addLayout(skeleton_row)

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
        self.record_webcam_action = self._action(file_menu, "Record from &Webcam...", self._on_record_webcam)
        self.import_animation_action = self._action(file_menu, "&Import Animation (BVH)...", self._on_import_animation)
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
        # UI7: a bone's adjustment onto another bone or person, or mirrored onto the other side.
        # A focused text field still gets Ctrl+C / Ctrl+V (Qt gives it standard keys first).
        edit_menu.addSeparator()
        self.copy_bone_action = self._action(edit_menu, "&Copy Bone Rotation", self._on_copy_bone, QKeySequence.Copy)
        self.paste_bone_action = self._action(
            edit_menu, "&Paste Bone Rotation", lambda: self._on_paste_bone(mirrored=False), QKeySequence.Paste
        )
        self.paste_mirrored_action = self._action(
            edit_menu, "Paste &Mirrored to Other Side", lambda: self._on_paste_bone(mirrored=True), "Ctrl+Shift+V"
        )

        pose_menu = bar.addMenu("&Pose")
        self.solve_action = self._action(pose_menu, "&Solve", self._on_solve_pose, ["F5", "Ctrl+Return"])
        self.solve_action.setEnabled(False)  # like the panel's Solve Pose, until a photo is loaded
        self.send_action = self._action(pose_menu, "Send to &Blender", self._on_send_to_blender, "Ctrl+B")
        # F15: keep poses for later -- re-export, mirror and blend them without the photo.
        self.add_to_library_action = self._action(
            pose_menu, "&Add to Pose Library...", self._on_add_to_library, "Ctrl+Shift+L"
        )
        self.library_action = self._action(pose_menu, "Pose &Library...", self._on_open_library, "Ctrl+Shift+P")
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
        # F16: line the view up with the camera the pose was fitted with, and fade the photo.
        self.photo_camera_action = self._action(
            view_menu, "View from &Photo Camera", self._on_view_from_photo_camera, "5"
        )
        self.backdrop_opacity_slider = QSlider(Qt.Horizontal)
        self.backdrop_opacity_slider.setRange(10, 100)
        self.backdrop_opacity_slider.setValue(round(100 * DEFAULT_BACKDROP_OPACITY))
        self.backdrop_opacity_slider.setAccessibleName("Photo backdrop opacity")
        self.backdrop_opacity_slider.setToolTip("How opaque the photo behind the 3D preview is")
        self.backdrop_opacity_slider.valueChanged.connect(lambda v: self.viewport.set_backdrop_opacity(v / 100))
        opacity_row = QWidget()
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(24, 2, 8, 2)
        opacity_label = QLabel("Backdrop opacity")
        opacity_label.setBuddy(self.backdrop_opacity_slider)
        opacity_layout.addWidget(opacity_label)
        opacity_layout.addWidget(self.backdrop_opacity_slider)
        opacity_action = QWidgetAction(self)
        opacity_action.setDefaultWidget(opacity_row)
        view_menu.addAction(opacity_action)

        help_menu = bar.addMenu("&Help")
        self.diagnostics_action = self._action(help_menu, "&Diagnostics...", self._on_show_diagnostics)
        self._action(help_menu, "&About PoseCraft", self._on_about)

        # UI2: the load -> solve -> export/send loop always in view -- the panel's own export and
        # send buttons can be below the fold. The same actions as the menus, so they share their
        # shortcuts and enabled state.
        self.main_toolbar = self.addToolBar("Main toolbar")
        self.main_toolbar.setObjectName("mainToolbar")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for action in (self.open_action, self.open_video_action, None, self.solve_action, None,
                       self.export_action, self.export_bvh_action, self.send_action):
            if action is None:
                self.main_toolbar.addSeparator()
            else:
                self.main_toolbar.addAction(action)
        self.main_toolbar.widgetForAction(self.solve_action).setObjectName("primaryAction")
        # UI5: the panel's Load/Solve/Export/Send buttons repeat the toolbar, so they show only
        # while the toolbar is hidden (its right-click menu can hide it).
        toolbar_toggle = self.main_toolbar.toggleViewAction()
        toolbar_toggle.toggled.connect(lambda toolbar_shown: self._show_panel_duplicates(not toolbar_shown))
        self._show_panel_duplicates(not toolbar_toggle.isChecked())

    def _show_panel_duplicates(self, show: bool) -> None:
        for button in (self.load_btn, self.load_video_btn, self.solve_btn, self.export_btn, self.export_bvh_btn,
                       self.send_to_blender_btn):
            button.setVisible(show)
        self.toolbar_hint.setVisible(not show)

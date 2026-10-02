# Project Status

## Architecture

```text
Photo (loaded via imaging.imread_unicode, unicode-path-safe;
       imaging.assess_photo_quality flags low resolution before solving)
        |
        v
PoseEstimator (pose_craft/estimation/landmarker.py)
  -- MediaPipe's BlazePose models: bundled pre-converted ONNX in a release build
     (scripts/convert_models.py), else the pinned, SHA-256-verified .task bundle
     downloaded and converted once (estimation/tflite_to_onnx.py); run on ONNX
     Runtime -- DirectML, then CUDA (TF32 off) when available, CPU otherwise;
     photos over 2048 px are area-downscaled first (estimation/onnx_pose.py
     re-implements MediaPipe's detector -> ROI -> landmark calculator graph);
     multi-person, with overlapping-tile detection for group photos
  -- video (estimation/video.py): one person tracked frame to frame
     (MediaPipe VIDEO-mode ROI from the previous landmarks), One Euro
     smoothing, floor contact and foot-locked root motion
        |  landmarks converted to the engine frame (right-handed, Y-up, +X image right =
        |  subject's left, +Z toward camera = the way the subject faces);
        |  each DetectedPerson also carries 2D image-space landmarks + per-landmark visibility
        v
retarget/skeleton.py -- 15-bone humanoid hierarchy, T-pose rest directions
  -- a full frame per bone (direction + roll): spine from the hip/shoulder lines,
     head from the ear line (neck takes 40% of its twist), upper arm/thigh roll
     from the elbow/knee bend plane (forearm/shin are pure hinges), hand from
     index/pinky (forearm takes half its twist), foot from heel -> toe and the
     ankle-heel-toe plane; unmeasurable roll is carried from the parent
  -- world -> parent-relative; forward_kinematics() poses the rest skeleton
     from those rotations (the 3D preview and the BVH use the same rest skeleton)
  -- bone_confidence() derives a per-bone score from landmark visibility
  -- manual correction: flip_limb_depth() (per-limb depth flip) and
     apply_bone_overrides()/bone_axes_rotation() (per-bone adjustments about
     Blender-style bone axes); hip_height_above_floor() for ground contact
        v
export/blender_pose.py -- axis-converts Y-up -> Blender Z-up, writes JSON (format v2:
     local + world rotation + rest direction per bone, optional "root"
     hip height / leg length); export/bvh_export.py -- single-frame or
     animated BVH, the root standing on the floor
  -- low-confidence bones flagged "low_confidence": true
        |
        +---------------------------------------------+
        v  (file)                                      v  (live)
pose_craft_pose .json file            export/live_link.py: length-prefixed
        |                              JSON over TCP to host:port (UI-configurable,
        |                              default 127.0.0.1:8642); the add-on replies
        |                              with what it applied (protocol 2)
        v                                              |
export/blender_addon/import_pose.py (installed inside Blender)
  -- Load Pose JSON operator, or a Live Link server (bpy.app.timers-polled,
     fully non-blocking: accept and incremental reads with a 2 s per-message
     deadline, 256 KB cap) that applies a received pose to the captured
     target armature, as its own undo step, optionally keyframed
  -- bone name resolution: an explicit per-armature manual mapping row
     (POSECRAFT_BoneMapping, saved with the .blend file) wins if set, else
     falls back to name-guessing (exact, Rigify upper_arm_fk.L/_ik, DEF-/ORG-, Mixamo)
  -- "Match rig rest pose" (default): rig_pose_basis() aims each mapped bone at
     PoseCraft's world target against the rig's own rest direction and posed
     parent (A-pose, Mixamo Hips, clavicles, bones without Inherit Rotation),
     spreading a bone mapped to a chain (Mixamo Spine/Spine1/Spine2) over it;
     off / v1 files: R^-1 q R per bone; each bone keeps its rotation mode
```

`pose_craft/gui/main_window.py` is the only orchestrator: it owns the loaded
photo (or tracked video), runs `PoseEstimator.detect()` on a background `QThread` (`SolveWorker`)
so the UI doesn't freeze while the pose-landmarker model downloads/runs, and
re-solves `retarget.skeleton.solve_bone_rotations()` synchronously (cheap, no
model inference) whenever the person picker, a photo click, or the mirror
checkbox changes. The `PoseEstimator` created for the first solve is cached on
`MainWindow._estimator` and reused for every later solve in the same session,
rather than recreating the ONNX Runtime sessions every click. Because that estimator
is shared, `load_btn` and `solve_btn` are both disabled for a solve's whole
duration (not just `solve_btn`) so a new photo can't be loaded, and a second
solve can't be started, while a `SolveWorker` is still running against the
same instance; `closeEvent` additionally calls `QThread.wait()` on any running
`SolveWorker`/`SendToBlenderWorker` before closing the estimator or letting
the window actually close, so shutdown can't race a background thread that's
still using it. `SendToBlenderWorker` runs the Live Link socket send
(`export/live_link.py`) on its own background thread the same way, rather
than blocking the UI thread for its connection timeout, and Live Link's target
host/port are now editable in the UI (`live_link_host_edit`/`live_link_port_spin`,
defaulting to `export.live_link.DEFAULT_HOST`/`DEFAULT_PORT`) rather than fixed
to `127.0.0.1:8642`. `SolveWorker` also takes a `max_people` value from a
`max_people_spin` control (1-8, default 4), applied per solve
(`PoseEstimator.max_people`) -- the ONNX pipeline takes it per call, so unlike
the old MediaPipe graph (which baked `num_poses` in at creation) changing it no
longer rebuilds the estimator. Changing the model variant or the "Use GPU"
checkbox (`use_gpu_checkbox`) does close and rebuild it, since both are fixed
when the ONNX Runtime sessions are created; the post-solve status message says
which device ("on GPU"/"on CPU") the solve actually ran on. `gui/photo_view.py`'s
`PhotoView` keeps the full-resolution `QPixmap` so it can
rescale cleanly on window resize, and turns a click into a normalized
image-space point that `main_window.py` matches against each detected
person's 2D centroid (`_nearest_person_index`) to drive the person picker.
`PhotoView` also draws every detected person's 2D skeleton directly on the
photo in its `paintEvent` (`retarget.skeleton.image_joint_positions_2d()` --
a 2D-image-space analogue of `joint_positions()`, with a 2D-safe override for
the one bone, `head`, whose 3D version isn't dimension-agnostic), color-coded
by confidence and with the selected person drawn on top of the others,
dimmed -- deliberately independent of the mirror checkbox, since mirroring
relabels which exported bone name a side's data feeds, it doesn't move
anyone's arm in the photo (see `PhotoOverlayPerson`'s docstring).
`gui/viewport_3d.py` is a thin wrapper around `pyqtgraph.opengl.GLViewWidget`
that draws `retarget.skeleton.forward_kinematics()` of the exported rotations (raw landmarks as ghost points) as line segments + points and
auto-frames the camera to the skeleton's own bounding box on every update
(`_compute_camera_frame`) -- it has no other state of its own beyond the
current skeleton, `main_window.py` always calls `clear_skeleton()` before
`show_skeleton()` on every update rather than diffing. The window is a three-pane `QSplitter` (scrollable controls, photo, 3D view); the control
panel is split into four `QGroupBox` sections (`_build_detect_group`,
`_build_person_group`, `_build_export_group`, `_build_live_link_group`)
instead of one flat stack of ~15 controls, plus a "3. Correct" group (per-bone
adjustment spin boxes; depth flips come from the photo's right-click menu)
and a "Video" group (frame slider, animation export) that appears once a
video is tracked.

Added in the P2 round (2026-09-30): busy state lives in `_set_busy()` and
`_set_result_actions_enabled()` (one place for what a running solve disables,
the status-bar progress bar and the Cancel button); `WarmUpWorker` creates
and warms the remembered model at startup when it's available offline
(`MainWindow(warm_up=True)`, which `app.py` passes; a Solve clicked meanwhile
waits for it); `VideoSolveWorker` tracks a video; manual corrections are
`PersonEdits` per person, changed only through `QUndoStack` commands, and
applied in `_document_inputs_for()` so every export path gets them; settings
(last folder, recent files, host/port, model, GPU, group photo, max people,
window and splitter geometry) persist through `QSettings` (an INI under the
user's config folder; `tests/conftest.py` points it at a temp folder).

The retargeting math (`retarget/skeleton.py`), the axis-conversion math
(`export/blender_pose.py`), the atomic-download logic and unicode-path
handling (`estimation/landmarker.py`, `imaging.py`), and the Live Link wire
protocol (`export/live_link.py`) are the modules with real unit test coverage
because they're pure numpy/scipy/stdlib with no GUI/model/Blender dependency
-- see "Verification" below for why the GUI and Blender add-on are checked
differently.

## Primary Entry Points

| Purpose | Entry point |
|---|---|
| Desktop app (load photo, solve, preview, export) | `pose_craft/app.py`, run as `python -m pose_craft.app` |
| Pose estimation entry point, world-axis conversion, cached `PoseEstimator`, model download + ONNX conversion cache | `pose_craft/estimation/landmarker.py` |
| BlazePose pipeline on ONNX Runtime (detector, group-photo tiles, ROI, landmarks, video tracking, GPU/CPU session setup) | `pose_craft/estimation/onnx_pose.py` |
| Video: tracking a person through a clip, One Euro smoothing, foot locking | `pose_craft/estimation/video.py` |
| TFLite -> ONNX converter for the pose (and hand) models | `pose_craft/estimation/tflite_to_onnx.py` |
| Pre-convert the models for a release build | `scripts/convert_models.py` |
| Release build (DirectML + CPU zips, add-on zip, SHA256SUMS) | `.github/workflows/release.yml` |
| Bone retargeting math + per-bone confidence | `pose_craft/retarget/skeleton.py` |
| Pose file format (write from the app, read by the add-on) | `pose_craft/export/blender_pose.py` |
| Standard BVH export (own additive path, alongside the JSON/Live Link one) | `pose_craft/export/bvh_export.py` |
| Live Blender link client (app side) | `pose_craft/export/live_link.py` |
| Blender-side importer + Live Link server (install manually inside Blender, not via pip) | `pose_craft/export/blender_addon/import_pose.py` |
| Unicode-safe image loading + pre-solve photo quality checks | `pose_craft/imaging.py` |
| Clickable/resizable photo widget | `pose_craft/gui/photo_view.py` |
| PyInstaller build spec | `PoseCraft.spec` |
| App icon generation (re-run only if the design changes) | `scripts/generate_icon.py` |

## Release

```text
dist/PoseCraft/PoseCraft.exe        -- now built with assets/icon.ico as the exe icon
dist/PoseCraft/_internal/           -- PySide6, OpenCV, ONNX Runtime + (GPU build) the CUDA 13/cuDNN 9 DLLs
dist/PoseCraft/_internal/models/    -- pre-converted ONNX models + manifest.json (when build/models existed)
```

Built with `pyinstaller PoseCraft.spec` (onedir, `--windowed`, `collect_all`
for `onnxruntime`/`cv2`, pyqtgraph's modules and data without its examples,
and -- only when no pre-converted models are present -- `onnx`/`tflite`;
`scripts/convert_models.py` writes the three variants' ONNX models plus one
shared detector and a manifest to `build/models`, and the spec then bundles
them under `models/` and leaves the converter out, so the app needs no
download or network on first run. The spec also excludes matplotlib, PIL and
the unused Qt modules (Quick/Qml/Network/Test/Pdf/VirtualKeyboard) and filters
their DLLs. None of these spec changes has been through a real build yet.
A CUDA build (`POSECRAFT_FLAVOR=cuda`) also bundles
the NVIDIA CUDA/cuDNN DLLs from the pip `nvidia-*` packages kept in their
site-packages layout under `_internal/nvidia/` (where `onnxruntime.preload_dlls()` looks) when installed -- that's what makes the GPU build
so much bigger than the old ~480 MB one; a build from a venv without
`requirements-gpu.txt` leaves them out and runs CPU-only -- plus `assets/icon.png`+`assets/icon.ico`
bundled as `datas` so `app.py` can set the window icon at runtime via
`sys._MEIPASS`). `dist/` and `build/` are gitignored -- distribute the built
folder directly (zip it, or wrap it in an installer) rather than committing
it. Not code-signed: expect SmartScreen to flag it as unrecognized on a
machine other than the one it was built on.

`requirements.txt` is now pinned to the exact versions this project was built
and tested against (was unpinned before); `requirements-test.txt` is a
separate, lighter pin set (numpy/scipy/opencv-python-headless/pytest) for CI.
`.github/workflows/ci.yml` runs `compileall` + `pytest` on push/PR using only
`requirements-test.txt` -- verified locally by installing that file into a
throwaway venv and confirming the same pass/skip counts CI would see. There is
also `.github/workflows/release.yml` (a `v*` tag or a manual run): it builds
the DirectML and CPU flavours on `windows-latest` (tests, pre-converted
models, PyInstaller, a bundled-model check), zips each, zips the Blender
add-on, writes `SHA256SUMS` and publishes a GitHub Release for a tag. It has
not been run yet. `requirements.txt` now uses `opencv-python-headless` (the
app never used OpenCV's GUI or contrib modules; the headless wheel still has
the FFmpeg video reader the video path needs).

## Verification

```bash
pytest
```

covers `retarget/skeleton.py` (including `image_joint_positions_2d()`, the 2D
overlay math), `export/blender_pose.py`, `estimation/landmarker.py` (atomic
download logic and the progress-callback contract, with `urlretrieve`
mocked), `imaging.py` (real cv2 encode/decode round trips, including a
non-ASCII filename), `export/live_link.py` (a real loopback TCP server
standing in for the Blender add-on, checking the exact wire format both sides
need to agree on), and `export/blender_addon/import_pose.py`'s
`POSECRAFT_BONE_NAMES` constant (`tests/test_blender_addon_source.py` parses
the add-on's *source* with `ast` -- it can't be imported here, since it needs
`bpy` -- to catch that list drifting out of sync with `skeleton.BONE_NAMES`)
-- all pure numpy/scipy/stdlib/cv2/ast, no PySide6/mediapipe/Blender needed to
run them. `tests/test_main_window_helpers.py` additionally covers
`main_window._nearest_person_index()` (the click-to-pick math) and the
mirror/confidence consistency fix (see "Recently Fixed"), and
`tests/test_viewport_camera.py` covers the auto-framing math
(`_compute_camera_frame`); both guard their PySide6 import with
`pytest.importorskip`, so they run locally and skip cleanly in CI's lighter
venv. Last full local run (with PySide6 installed, 2026-09-30, after the P2
round): **298 passed, 6 skipped** (the LiteRT parity test, the real-photo
golden test, and the four opt-in GPU parity tests). `tests/test_gui_flows.py`
now drives the real window on an offscreen Qt event loop with fake
estimators: busy states, cancel, the close-waits-for-the-worker guarantee,
the non-blocking send, warm-up, settings, drag-and-drop, picking, manual
correction with undo, exports and video scrubbing -- the threading checks
below that used to be one-off scripts. `tests/test_gpu_parity.py` (opt-in:
`POSECRAFT_GPU_TESTS=1`) compares every available GPU provider with the CPU;
it passed on CUDA (RTX 5080) and fails, as it should, with TF32 switched back
on. CI's lighter venv also skips the Qt
tests; re-run `pytest` rather than trusting a number written here.

Everything else has been checked manually rather than by automated test,
since it needs a real display/GPU/Blender install that isn't available in
this dev environment:

- The GUI's imports and `MainWindow` construction were smoke-tested (`QApplication` +
  `MainWindow()` + `processEvents()`, no crash, `isVisible()` true) after every
  round of GUI changes in this project -- this proves the widget tree builds,
  not that solving/exporting/clicking a real photo works end to end.
- The threading fixes (button-disabling during a solve, `closeEvent` waiting
  for in-flight workers, `SendToBlenderWorker` not blocking the UI thread)
  are now in the suite (`tests/test_gui_flows.py`, see above). The 3D view's
  actual OpenGL drawing (octahedral bones, labels, gizmo) still isn't: an
  offscreen Qt has no GL context to render into, so those tests check the
  geometry and the items created, not pixels.
- `max_people` is applied per call (`PoseEstimator.max_people`), so changing
  it no longer rebuilds the estimator; only the model variant and the GPU
  checkbox do.
- The GPU paths were checked on an RTX 5080 with the project's own
  `BlazePose` on `two_people.jpg` (2026-09-30): CUDA with `use_tf32=0`
  9.3/11.1/39.1 ms (lite/full/heavy) and at most 0.017 px from the CPU;
  DirectML (onnxruntime-directml 1.24.4) 4.0/4.5/7.6 ms and at most 0.021 px;
  CPU 22-42 ms. The CUDA run used a venv with cuFFT, nvrtc, nvJitLink and
  cudnn_adv removed, confirming the K1 spec exclusions still work.
- The photo overlay (`PhotoView.paintEvent`, drawing each detected person's 2D
  skeleton on the photo) was checked by constructing a real `MainWindow` with
  fake `DetectedPerson` data, calling `.grab()` to force an actual Qt paint
  pass (not just confirm the code compiles), and visually inspecting the
  rendered output: the selected person's skeleton renders in the bright color
  on top, an unselected person renders dimmed underneath, and a bone whose
  landmarks were given deliberately low visibility renders in the low-confidence
  color. Confirms the drawing code runs and looks right for that one
  synthetic case; not a substitute for seeing it against a real photo.
- The built `.exe` was launched and confirmed to stay running (no immediate
  missing-DLL crash) rather than close under it and left running.
- The Blender add-on (`import_pose.py`), including its new Live Link server,
  has **not** been loaded into a real Blender instance or tested against an
  actual Rigify-generated armature -- no Blender install exists in this dev
  environment. Its bone-name matching and its non-blocking-socket `bpy.app.timers`
  polling loop are both unverified beyond code review; see "Known Issues".

Re-run `pytest` yourself before trusting a pass count, and treat "manually
checked" above as weaker evidence than an automated test -- it was true at the
time each check ran, not guaranteed to stay true as the code changes.

## Recently Fixed

- **Improvement-report P2 round (2026-09-30).** Each item has tests that
  failed before the change (298 passed, 6 skipped after it):
  - **Group photos (B8).** The detector only sees the whole photo at 224 px,
    so small faces vanished: on a 2x2 collage of a two-person photo (8 people)
    the whole image found 1; with tiles, 8 (a 3x3 collage: 0, then 4). Tiles
    are searched always with the new "Group photo" checkbox, and
    automatically when the whole photo yields nobody. A tile detection never
    replaces a whole-image one.
  - **Ground contact (B6).** The world landmarks are hip-centred, so every
    pose stood at one hip height. The 3D preview now stands the skeleton on
    the grid, the BVH root sits at the pose's hip height above its lowest
    joint (a crouch exports lowered), and the JSON carries a "root"
    hip_height/leg_length. The add-on's new "Keep feet on the floor" option
    moves the rig's root so the pose stands on the rig's own floor.
  - **Video (B12).** Load Video tracks one person (the detector runs once,
    then the previous frame's landmarks give the next crop: 1 detector run for
    40 frames, ~11 ms/frame on the CPU), smooths with a One Euro filter, and
    exports an animated BVH whose root follows planted feet. Scrub frames with
    the slider.
  - **Live Link (B16-B18, plus B19/B20).** The server no longer blocks
    Blender's UI: reads are non-blocking and incremental with a 2 s deadline
    per message and a 256 KB cap. The target armature is captured when Live
    Link starts. Each received pose is an undo step; optional keyframing
    (also when auto-keying is on) with a frame step. Bones keep their own
    rotation mode, and the panel redraws after each pose. Add-on 0.5.0.
  - **Rig-aware retargeting (B5, more).** A mapping row can name a chain
    ("Extra Bones"); the rotation is spread over it (Mixamo's three spine
    bones automatically), and bones with Inherit Rotation off are solved the
    way Blender evaluates them.
  - **GUI (F3, F4, F6, F7, F9, P3, plus F5/C1).** A legend drawn with the
    overlay's own colours; low-confidence bones dashed (2D and 3D), joints
    neutral. Manual correction: right-click a limb to flip its depth, or
    rotate any bone about its own X/Y/Z, with undo/redo. A progress bar and
    Cancel; the photo-quality warning is a dismissable banner; the model warms
    up in the background at startup. Menus and shortcuts, drag-and-drop,
    recent files, remembered settings, `<photo>_person<N>` export names
    (Export All asks before overwriting). The 3D view: floor grid, XYZ gizmo,
    front/side/top presets, bone labels, octahedral bones showing roll.
    Clicking empty background no longer changes the selected person.
  - **Models (R3, K3, P5 in part).** The model URLs are pinned to version 1
    and SHA-256-verified (the old ".../latest/" "full" file differs only in
    zip timestamps, so existing caches stay valid). A release build bundles
    pre-converted models. The converter also handles PRELU, MEAN and
    FULLY_CONNECTED (MediaPipe's hand models; 2e-6 / 4e-6 relative error
    against LiteRT) -- the first step toward finger poses (B11).
  - **Tests and release (T3, T5, K4, K5).** GUI flow tests, the opt-in GPU
    parity test, spec trimming, headless OpenCV and a release workflow (see
    "Release"; none of it built or run yet).
  - **Stale docs (C2).** Fixed the pass count, the Live Link timeout
    contradiction, the CI comment, and the user guide's missing orientation
    fix; two historical entries below now say what superseded them.

- **Improvement-report P1 round (2026-09-30).** Each item has tests that failed
  before the fix:
  - **Retargeting solved only each bone's direction.** A shortest-arc swing
    left every bone's roll arbitrary: a body turned 60 degrees solved to a
    root rotation of 0 (BVH shoulders ~18 cm off), a 45-degree head turn to 0,
    a 90-degree elbow to a 120-degree forearm rotation with its axis 55
    degrees off the hinge, and a flat foot to 25 degrees toe-down. Every bone
    now gets a full frame (see Architecture), and a bone whose roll can't be
    measured carries its parent's, so a rigid rotation of the whole body
    changes only the root (checked for 50 random rotations). The foot's rest
    direction now slopes 25 degrees down (`FOOT_REST_PITCH_DEG`), which also
    changes the BVH rest skeleton's foot End Site; `solve_bone_rotations`
    takes an optional `visibility` for the heel fallback. Tests:
    `tests/test_retarget_frames.py`, plus BVH forward kinematics for a 60-degree
    turned body in `test_axis_conventions.py`.
  - **The 3D preview drew raw landmarks, not the export.** It now draws
    `skeleton.forward_kinematics()` of the exported rotations (with the
    subject's measured proportions), with the raw landmarks as faint ghost
    points. The rest skeleton moved from `bvh_export.py` to `skeleton.py`
    (re-exported under the old names).
  - **GPU.** `onnx_pose.select_providers` tries DirectML, then CUDA, then the
    CPU; DirectML sessions get `enable_mem_pattern=False` and sequential
    execution; CUDA gets `use_tf32=0` (TF32 moved landmarks up to 12 px).
    `device` now says "GPU (DirectML)"/"GPU (CUDA)". New
    `requirements-dml.txt` and a `POSECRAFT_FLAVOR=dml|cuda|cpu` switch in
    `PoseCraft.spec`, which also drops cuFFT, nvrtc, nvJitLink and cudnn_adv
    (~520 MB) from CUDA builds. Built and smoke-tested on 2026-10-01 (see
    Known Issues for the sizes).
  - **Large photos were warped straight from full resolution.**
    `landmarker.prepare_input_rgb` shrinks anything over 2048 px with
    `INTER_AREA` first. On the sample photos upscaled 8x with noise: median
    2D error 2.15 -> 1.42 px, 3.8/14.0 -> 1.1/7.8 px and 10.2 -> 3.9 px; world
    error 37 -> 11 mm, 34/46 -> 10/20 mm and 28 -> 19 mm; 148-198 ms -> 27-42 ms.
  - **The window didn't fit a 1080p screen** (minimum height 1124 px). Now a
    three-pane splitter with a scrollable controls column and collapsible
    photo tips; minimum size (offscreen) 1034 x 280.
  - **Live Link was fire-and-forget.** It's request/response now (protocol 2):
    the add-on replies with applied/unmatched/low-confidence bones, the
    armature and any error, and the app shows "Applied 13/15 bones to
    'Armature'" or the error. An older add-on's silence is reported as
    "sent but not confirmed". The add-on rejects a newer protocol.
  - **The add-on assumed a PoseCraft-shaped T-pose rig.** Pose format v2 adds
    each bone's world rotation and rest direction; with "Match rig rest pose"
    (default on) the add-on aims each rig bone against its own rest direction
    and actual posed parent (`rig_pose_basis`), so A-pose arms, Mixamo's Hips
    and clavicles work. Tested against a simulated Mixamo-like A-pose rig
    (every mapped bone exact; the v1 path left the arm >20 degrees off) and
    shown identical to the v1 formula on a PoseCraft-shaped rig. Add-on 0.4.0.
  - **Errors had nowhere to go in the windowed build.** Export failures now
    show a dialog; `pose_craft/diagnostics.py` logs to
    `%LOCALAPPDATA%\PoseCraft\logs\posecraft.log` and turns uncaught
    exceptions (any thread) into a dialog; solve failures log their traceback.
  - **The model download had no timeout** and `closeEvent` waited forever.
    Downloads stream with a 30 s socket timeout and can be cancelled;
    closing cancels, waits up to 5 s, then hides the window and finishes
    closing when the worker returns.
  - **No real-photo test.** `scripts/generate_goldens.py` plus
    `tests/test_real_photo_golden.py` (2 px / 15 mm / 5 degree tolerances),
    which skips until licensed photos are committed to `tests/data/golden`.

- **Review fixes (axis convention, mirror, BVH, head, stale UI, add-on).**
  Each now has a test that describes what should happen, not just the math
  (`tests/test_axis_conventions.py`, plus additions to the existing test files):
  - **Depth was mirrored.** `_world_to_engine()` flipped only Y. MediaPipe's
    world frame is +y down with smaller z closer to the camera, so the result
    was left-handed: a knee raised toward the camera exported *behind* the rig
    (Blender +Y). It now negates Y and Z (a 180-degree rotation about X),
    giving a right-handed frame where +X is image right (the subject's left,
    which is why `LEFT_X = +X` was already right) and +Z points toward the
    camera. `FORWARD` (the foot rest direction) is now +Z, and the docstrings
    that said "+X = subject's right" are fixed. `blender_pose.py`'s axis
    conversion was already right for this frame (a character faces -Y, .L is
    on +X). Tests: a knee toward the camera lands in front of the rig in
    Blender and in the BVH, and a left arm raised out to the subject's left
    stays on the left. Side effect: the hand-twist zero reference (+Z) now
    means a palm-down T-pose hand. In the old left-handed frame, a palm-down
    hand solved to a 180-degree twist.
  - **The 3D viewport drew the skeleton lying on the grid.** pyqtgraph's view
    is Z-up, but it was given raw Y-up engine points. `viewport_3d.py` now
    converts to Z-up (`_engine_to_display`, the same axis change as the Blender
    export) and starts the camera in front of the subject. The orientation
    presets still rotate in the engine frame before that conversion, so they
    behave as before.
  - **The mirror checkbox rotated the limbs 180 degrees.** `mirror_landmarks()`
    swapped the left/right rows but didn't reflect them, so a mirrored T-pose
    had both arms pointing through the body. It now also negates X (for 3D)
    or maps x to 1 - x (for 2D normalized coordinates). 1-D visibility arrays
    still only get the row swap. The 3D preview's low-confidence coloring now
    uses mirrored visibility too.
  - **BVH offsets were shifted one level.** Each JOINT's OFFSET was the
    joint's own bone vector instead of its head relative to its parent's
    head. The rest skeleton (`_rest_heads`) now uses chain offsets plus
    `SHOULDER_HALF_WIDTH_CM`/`HIP_HALF_WIDTH_CM` lateral attachments, with the
    root at zero. Each End Site is the leaf bone's own vector. A
    forward-kinematics test evaluates the written BVH (rest offsets plus
    rotations) and checks that it reproduces every solved joint position.
    scipy's gimbal-lock warning at exactly +/-90 degrees about X is now
    suppressed, because the angles it returns are still exact.
  - **A level head exported pitched ~20 degrees back.** `_head_top`
    extrapolated along the nose-to-ear line, and the nose sits below and in
    front of the ears. The head's up axis is now the cross product of the
    face-forward vector (ear midpoint to eye midpoint) and the ear-to-ear line.
    A level head gives about 0 degrees, and a 30-degree nod gives 30 degrees
    forward.
  - **A re-solve that found nobody left stale UI.** The person list, photo
    overlay and 3D skeleton from the previous solve stayed up, and every
    export button stayed enabled. That path now clears all of them.
  - **Blender add-on.** Unreadable or invalid JSON now shows an error report
    instead of a traceback. The Rigify templates now put the tag before the
    side suffix (`upper_arm_fk.L`, not `upper_arm.L_fk`) and prefer control
    bones over constraint-driven DEF-/ORG- bones. Mixamo names
    (`mixamorig:LeftArm`, ...) were added. Rotations are now converted into
    each bone's rest frame before being applied (`armature_delta_to_basis`,
    `R^-1 @ q @ R` with `R = bone.matrix_local`). Previously the
    armature-space delta was written straight into `rotation_quaternion`,
    which rotated almost every bone about the wrong axes. The pure helpers
    (`bone_name_candidates`, `armature_delta_to_basis`) are tested by
    compiling them out of the add-on's source with `ast`. The second one is
    checked against a simulation of Blender's pose evaluation with
    non-aligned rest frames.
  - **Packaging.** `PoseCraft.spec` now leaves out `curand64_10.dll` (~60 MB).
    pefile shows none of the bundled onnxruntime, cuDNN or cuBLAS DLLs import
    it (statically or delay-loaded), and `onnxruntime.preload_dlls()` only
    preloads curand on Linux. The Load Photo dialog now accepts `.tif`/`.tiff`.

- **Pose estimation could only run on the CPU.** MediaPipe's Windows wheels
  are built without GPU support, so the `mediapipe` package couldn't use an
  NVIDIA GPU at all. **Fixed** by moving inference to ONNX Runtime:
  `estimation/tflite_to_onnx.py` converts the two TFLite graphs inside
  MediaPipe's own `.task` bundle to ONNX (a small purpose-built converter --
  `tf2onnx` needs TensorFlow, which has no Python 3.14 build, and `tflite2onnx`
  mishandles these models' float16/sparse weights and PAD ops), cached next to
  the download with the converter version in the filename; `estimation/onnx_pose.py`
  re-implements MediaPipe's calculator graph around them (SSD anchors, weighted
  NMS, letterbox removal, alignment-point ROI, rotated crop, heatmap refinement,
  landmark/world projection), following MediaPipe's C++ sources calculator by
  calculator. Verified: the converted graphs match LiteRT's output to ~1e-5
  relative error for all three variants (`tests/test_tflite_to_onnx.py`, run
  locally where LiteRT is available); end to end against the `mediapipe`
  package on real photos, visible landmarks agree to a median of 0.1-1.5 px
  with the same people detected -- the largest remaining differences are on
  ambiguous landmarks where MediaPipe's *own* output moves as much under +/-1
  intensity-level noise. On an RTX 5080, one solve takes ~12/13/18 ms
  (lite/full/heavy, two people) on the GPU vs ~25/28/40 ms on the CPU.
  `mediapipe` is no longer a dependency.

- **The 3D viewport (the larger, primary preview checked right before
  Export/Live Link) had no confidence indication at all**, unlike the 2D photo
  overlay (`photo_view.py`), which already color-codes low-confidence bones but
  has no on-screen legend explaining the colors, and the post-solve status
  message never summarized how many bones came out low-confidence even though
  `bone_confidence()` is computed right there. **Fixed**: `viewport_3d.py`'s
  `show_skeleton()` gained an optional `confidence: dict[str, float] | None`
  parameter (default `None` preserves the original single-`BONE_COLOR`
  behavior exactly); when given, each bone segment below
  `blender_pose.LOW_CONFIDENCE_THRESHOLD` is drawn in a new
  `LOW_CONFIDENCE_BONE_COLOR` matching `photo_view.py`'s low-confidence color,
  via a per-vertex color array on the single `GLLinePlotItem` (both endpoints
  of a segment share a color, since GLLinePlotItem has no per-segment
  primitive) -- the color math is pulled into a pure `_bone_line_colors()`
  helper, the same pattern `_compute_camera_frame()` already used, so it's
  testable without a real GL context. `main_window.py`'s
  `_update_preview_for_selected_person()` passes the already-computed
  `bone_confidence(person.visibility)` dict through. Added a muted-caption
  `QLabel` legend under `photo_view` explaining the blue/orange color coding.
  `_on_solve_finished()`'s status message now appends "-- N of M bones
  low-confidence" (via a new pure `_low_confidence_status_suffix()` helper,
  empty string when nothing is low) for the person selected right after a
  solve, following the same append-onto-`status` pattern the max-people-limit
  warning already used. Verified with new tests in `test_viewport_camera.py`
  (`_bone_line_colors()`: with vs. without confidence data, threshold
  boundary, missing-bone default) and `test_main_window_helpers.py`
  (`_low_confidence_bone_count()`/`_low_confidence_status_suffix()`) --
  `_update_preview_for_selected_person()`'s actual GL rendering call remains
  untested by `pytest`, consistent with this project's existing convention of
  keeping real OpenGL calls out of the automated suite (see "Verification").
- **No user-facing way to fix an upside-down or wrong-facing pose, only
  left/right mirroring.** "Known Issues" already flagged
  `_mediapipe_world_to_engine()`'s axis convention as an unverified guess that
  could come out upside-down or mirrored, but the mirror checkbox only ever
  covered the left/right case. **Fixed**: `retarget/skeleton.py` gained
  `ORIENTATION_CORRECTIONS` (Default / Flip Upside-Down / Rotate 90° Left /
  Rotate 90° Right, engine-frame rotations) plus `apply_orientation_correction()`
  (composes the chosen correction onto the solved root/"spine" bone's local
  rotation -- the one bone whose local rotation is also its world rotation, so
  this rigidly re-orients the whole skeleton without touching any other bone's
  parent-relative angle) and `rotate_joint_positions()` (the position-space
  equivalent, since the 3D viewport draws `joint_positions()` -- raw landmark
  positions -- rather than anything forward-kinematics-derived from the
  rotations dict). `gui/main_window.py` adds an "orientation_combo" next to the
  mirror checkbox, live in the 3D preview (`_update_preview_for_selected_person`)
  and composed into every export path (`_document_inputs_for`, now taking an
  `orientation_correction` argument used by `_current_person_document_inputs`
  and `_on_export_all` alike, so JSON/BVH/Live Link all agree with what's shown
  on screen). Defaults to "Default" (identity), so existing behavior/exports
  are unaffected until a user actually picks a correction. Verified with new
  tests in `test_retarget.py` (the root-only-change guarantee, the pivot-based
  position rotation) and `test_main_window_helpers.py` (the combo reaching the
  exported rotations dict).
- **No standard-interchange export, JSON/Live Link only.** Any tool other than
  the companion Blender add-on (motion-capture viewers, game engines, other DCC
  apps) had no way to consume a solved pose. **Fixed**: added
  `export/bvh_export.py` (`build_bvh_document()`/`save_bvh_file()`), an
  additive path alongside the existing JSON/Live Link one -- same 15-bone
  hierarchy and rest directions as `retarget/skeleton.py`'s `BONES` table, a
  new `BONE_LENGTHS_CM` table for BVH's required static joint OFFSETs (lengths
  skeleton.py itself has no notion of, since it only ever compares directions
  against real photo-derived positions), and a single-frame MOTION block with
  each bone's local rotation converted to intrinsic Z/X/Y Euler angles
  (`BVH_ROTATION_ORDER = "ZXY"`, matching the "Zrotation Xrotation Yrotation"
  channel order it always writes). Documents its own explicit (currently
  identity, since it targets the same Y-up/+X-left/+Z-forward convention as
  the engine frame) axis conversion the same way `blender_pose.py`'s
  `_ENGINE_TO_BLENDER_AXES` does, for the same reason: BVH has no single
  universal up-axis convention across tools, unlike this project's Blender
  target. Wired into `gui/main_window.py` as a new "Export BVH..." button next
  to "Export Pose JSON...", reusing `_current_person_document_inputs()` so
  mirroring and the new orientation-correction control above apply to it too.
  Verified with new tests in `test_bvh_export.py`: HIERARCHY nesting (1 ROOT +
  14 JOINT + 5 End Site, one per leaf bone, balanced braces), the MOTION
  block's exact channel count (6 root + 14×3), and a known rotation surviving
  a round trip through the written Euler values back into a `Rotation`.
- **Model download wasn't atomic.** `ensure_model_downloaded()` only checked
  `dest.exists()`, so an interrupted first-run download left a corrupt file
  that got silently reused as "already downloaded" forever after. **Fixed**:
  downloads now go to a `.part` temp file in the same cache directory, get a
  minimum-size sanity check (`MIN_MODEL_BYTES`), and are only published to the
  real destination via an atomic `Path.replace()`; a pre-existing file that
  fails the size check is redownloaded rather than trusted. Verified with
  `tests/test_landmarker_download.py` (mocked `urlretrieve`): a good download
  is cached and not repeated on a second call, a truncated download raises and
  leaves no `.task` or `.part` file behind, and a corrupt pre-existing file
  gets redownloaded.
- **A fresh MediaPipe `PoseLandmarker` was created on every solve.** Each
  click re-paid the model-load/graph-build cost. **Fixed**: `landmarker.py`
  now has a `PoseEstimator` class that owns one live landmarker instance;
  `MainWindow` creates it once on the first solve and reuses it (via
  `SolveWorker`) for every later solve in the session, closing it in
  `closeEvent`. (The old one-shot `detect_poses()` convenience wrapper this
  superseded was later deleted once nothing called it anymore -- see the dead
  code entry below.)
- **Landmark visibility/confidence was discarded.** MediaPipe scores how
  visible each landmark is, but nothing carried that signal past detection --
  a hidden/occluded limb (PoseForge's own "hidden limb" caveat) failed
  silently instead of being flagged. **Fixed**: `DetectedPerson` now carries a
  `visibility` array; `retarget/skeleton.py::bone_confidence()` reduces it to
  a per-bone score (the minimum visibility among that bone's contributing
  landmarks, via a new explicit `visibility_landmarks` field on `BoneSpec`
  rather than reusing the geometric `head_fn`/`tail_fn`, which do arithmetic
  -- like the head bone's extrapolated skull point -- that has no honest
  "visibility" of its own); `export/blender_pose.py::build_pose_document()`
  adds `"confidence"`/`"low_confidence"` per bone when a confidence dict is
  passed in, and the Blender add-on now reports which applied bones were
  low-confidence, separately from unmatched ones. Verified with new tests in
  `test_retarget.py` and `test_blender_pose.py`.
- **`cv2.imread` didn't handle non-ASCII Windows paths reliably.** OpenCV
  opens paths through a narrow/ANSI codepage on Windows, so a photo path with
  non-ASCII characters would silently fail to load. **Fixed**: added
  `imaging.imread_unicode()` (reads the bytes with `np.fromfile`, decodes
  in-memory with `cv2.imdecode`) and switched `main_window.py`'s photo loader
  to use it. Verified with `tests/test_imaging.py`, including a path with
  accented and Japanese characters.
- **No click-to-select-in-photo person picker.** Multi-person selection was
  dropdown-only; PoseForge lets you click the person directly in the image.
  **Fixed**: `gui/photo_view.py`'s new `PhotoView` widget emits a normalized
  image-space point on click, and `main_window._nearest_person_index()`
  matches it against each detected person's 2D landmark centroid to drive the
  same person picker the dropdown uses. Verified with
  `tests/test_main_window_helpers.py`.
- **Export was file-based only, no live "Open in Blender."** Workflow was:
  export JSON, switch to Blender, use the add-on's file picker, every time.
  **Fixed**: added `export/live_link.py` (a small length-prefixed JSON-over-TCP
  client) and a matching non-blocking server in the Blender add-on
  (`bpy.app.timers`-polled, since Blender's Python is single-threaded and a
  blocking `accept()` would freeze the UI), toggled from a "Start Live Link"
  button in the add-on's panel; the app gained a "Send to Blender (Live Link)"
  button that posts the currently selected person's pose to
  `127.0.0.1:8642`. Verified with `tests/test_live_link.py` against a real
  loopback socket server standing in for the add-on's side of the protocol --
  the add-on's actual Blender-side server code is still unverified in a real
  Blender instance, see "Known Issues".
- **No batch export across multiple detected people.** **Fixed**: added an
  "Export All People..." button that writes one `pose_personN.json` per
  detected person into a chosen folder.
- **Photo preview didn't rescale on window resize.** **Fixed**: `PhotoView`
  now keeps the original full-resolution `QPixmap` and re-derives the scaled
  display pixmap in `resizeEvent`, instead of scaling once at load time.
- **Dependencies were unpinned.** **Fixed**: `requirements.txt` now pins exact
  versions; a new `requirements-test.txt` pins a lighter subset for CI.
- **No CI.** **Fixed**: added `.github/workflows/ci.yml` (`compileall` +
  `pytest` on push/PR, `requirements-test.txt` only -- see "Release" for why).
- **No app icon.** **Fixed**: `scripts/generate_icon.py` renders a simple
  stick-figure-skeleton icon (in the same bone/joint colors as the 3D
  viewport) to `assets/icon.png`/`assets/icon.ico`; wired into both the live
  app's window icon (`app.py`, via `QApplication.setWindowIcon`) and the built
  `.exe` (`PoseCraft.spec`'s `icon=` and bundled `datas`).
- **No pre-solve photo-quality guidance.** DAZ Studio 6's own built-in Face
  Transfer feature (a photo-to-3D feature for faces -- same category of tool
  as PoseCraft, for faces instead of full-body pose) documents minimum
  resolution and framing/lighting requirements for its users; PoseCraft had no
  equivalent and just silently produced a worse result on a bad photo.
  **Fixed**: `imaging.assess_photo_quality()` warns when a loaded photo is
  below `MIN_RECOMMENDED_DIMENSION` (480px on the shorter side) --
  `main_window.py` shows this as a popup on load -- and a static
  `PHOTO_GUIDANCE` tips list (full body in frame, facing the camera, even
  lighting, limbs not hidden) is now always visible under the Load Photo
  button. Only resolution is actually code-checkable; the framing/lighting
  tips are shown unconditionally since there's no reliable way to detect "is
  this person facing sideways" from the raw image. Verified with
  `tests/test_imaging.py`.
- **Hand/wrist twist was left at identity along with everything else.**
  Researching DAZ's own Genesis rig (which splits each limb into a `*Bend` and
  a `*Twist` bone) clarified that DAZ's Bend-drives-Twist linkage is a
  mesh-deformation trick for an *artist-chosen* twist value, not a way to
  infer an *unknown* twist from a photo -- so it doesn't actually generalize
  here. What does generalize: MediaPipe's index/pinky knuckle landmarks
  straddle the wrist's own axis, an independent reference no other limb
  segment has in the 33-point set. (Superseded by the P1 round's full
  per-bone frames: the spine, head, bent limbs and feet do have a second
  reference after all, see above.) **Fixed**: `retarget/skeleton.py` now
  solves real twist for `hand.L`/`hand.R` from that lateral spread (comparing
  it against the already-solved swing's carried-forward rest reference);
  every other bone is unaffected and stays at identity, honestly, since no
  equivalent second landmark exists for forearm/upper-arm/thigh/shin rotation.
  Verified with new tests in `test_retarget.py`, including a synthetic 90-degree
  wrist twist recovered to within floating-point precision.
- **Loading a new photo or starting a second solve while one was already
  running could corrupt state or crash.** `_on_load_photo` unconditionally
  re-enabled `solve_btn`, so clicking Solve, then loading a different photo,
  then clicking Solve again could leave two `SolveWorker` threads calling the
  shared `PoseEstimator.detect()` concurrently -- not documented as safe by
  MediaPipe -- and if the first solve finished after the second photo loaded,
  its results (for the *old* photo) would land on the person picker while the
  *new* photo was on screen. Separately, closing the window mid-solve called
  `self._estimator.close()` immediately, while a background thread might still
  be calling into that same instance. **Fixed**: `load_btn` is now disabled
  for a solve's entire duration alongside `solve_btn` (and `_on_load_photo`/
  `_on_solve_pose` both also check `self._worker.isRunning()` directly as a
  second line of defense, not just the button state); `closeEvent` now calls
  `QThread.wait()` on any running `SolveWorker` or `SendToBlenderWorker`
  before closing the estimator or completing the close. Verified with two
  scripted integration checks (not part of the `pytest` suite, since they need
  a real Qt event loop and timed background threads): one confirms
  `load_btn` is actually `False` while a fake slow `detect()` is running and
  that a same-thread call to `_on_load_photo()` during that window is a no-op;
  the other confirms `closeEvent` blocks until a fake slow `detect()` finishes
  before `estimator.close()` runs, by asserting the exact order those two
  events happen in.
- **"Send to Blender" blocked the whole UI while connecting.**
  `_on_send_to_blender` called `export.live_link.send_pose_document()`
  directly on the UI thread -- its default 3-second connection timeout meant
  the entire window froze for up to 3 seconds any time Blender's Live Link
  wasn't running yet, a likely first-time scenario, not an edge case.
  **Fixed**: added `SendToBlenderWorker` (the same background-`QThread`
  pattern `SolveWorker` already used) so the send happens off the UI thread;
  `send_to_blender_btn` disables for the duration and re-enables on success or
  failure. Verified with a scripted integration check: a mocked
  `send_pose_document` that sleeps 0.6s and then raises, showing
  `_on_send_to_blender()` itself returns in ~1ms (not 0.6s) and the button is
  disabled for the duration.
- **Batch export ignored the mirror checkbox.** `_on_export_all` called
  `solve_bone_rotations(person.landmarks)` directly on unmirrored landmarks
  for every person, while single-person export respected `mirror_checkbox` via
  `_selected_landmarks()` -- inconsistent, and silently wrong for every person
  in a group photo whenever the photo's orientation needed the mirror fix.
  **Fixed**: `_on_export_all` now applies `mirror_landmarks()` to every
  person's landmarks (and visibility, see the next entry) when the checkbox is
  checked, the same as the single-person path.
- **Mirrored confidence didn't match the landmark data actually used.**
  Found while fixing the batch-export mirror issue above: even in the existing
  single-person path, `_current_person_document_inputs()` mirrored the
  landmarks (so rotations used the correct swapped-in data) but *not* the
  visibility array, so a mirrored `thigh.L`'s reported `low_confidence` came
  from the original left leg's visibility instead of the real right leg's data
  that was actually used to compute its rotation. **Fixed**: both
  `_current_person_document_inputs()` and `_on_export_all()` now mirror
  `visibility` in lockstep with `landmarks` (`mirror_landmarks()` just swaps
  left/right rows, so it works on any per-landmark array, not only 3D
  positions). Verified with a new regression test in
  `test_main_window_helpers.py` that gives the real right leg deliberately low
  visibility and confirms the mirrored `thigh.L` reports that low score.
- **No progress indicator during the first-run model download.** The status
  bar just said "may take a moment" with no percentage. **Fixed**:
  `ensure_model_downloaded()` takes an optional `on_progress(downloaded_bytes,
  total_bytes)` callback (via `urlretrieve`'s `reporthook`, only passed through
  when a callback is actually given, so existing callers/tests using the plain
  `(url, filename)` stub signature are unaffected); `SolveWorker` forwards it
  through a new `progress` signal, and the status bar now shows a live
  percentage (or a KB count if the server didn't send a `Content-Length`).
  Verified with new tests in `test_landmarker_download.py`, including that
  progress values are clamped to the real total even if the last reported
  block would overshoot it.
- **`PoseEstimator`'s 4-person cap was hardcoded and undocumented.** A 5th
  person in a group photo was silently never detected, with no UI indication
  why. **Fixed**: added a "Max people to detect" spin box (1-8, default 4);
  since MediaPipe bakes `num_poses` into the landmarker graph at creation
  time, changing the value closed and discarded the cached `PoseEstimator`.
  (No longer: the ONNX pipeline takes it per call, so nothing is rebuilt.)
- **Live Link's host/port were hardcoded.** No way to work around
  `127.0.0.1:8642` being unavailable. **Fixed**: added editable host/port
  fields in the UI, defaulting to `export.live_link.DEFAULT_HOST`/`DEFAULT_PORT`;
  `SendToBlenderWorker` now takes them as constructor arguments instead of
  always using the module defaults. (This was incomplete on its own -- see the
  next entry.)
- **The app-side Live Link port control above didn't work, because the
  Blender add-on's listening port was still hardcoded.** Changing the port in
  PoseCraft just pointed the client at a port Blender was never told to
  listen on -- the send would silently fail with "could not reach Blender's
  Live Link" for exactly the port-conflict scenario the app-side control was
  built to solve. **Fixed**: `import_pose.py`'s `LIVE_LINK_PORT` constant
  became `DEFAULT_LIVE_LINK_PORT`, and `_start_live_link_server()` now takes
  a `port` argument read from a new `bpy.types.Scene.posecraft_live_link_port`
  property, editable in the panel (locked while Live Link is already running,
  since a bound socket's port can't change without restarting it) and
  registered/unregistered alongside the add-on's operator classes. The host
  stays fixed at `127.0.0.1` deliberately -- exposing this socket (arbitrary
  bone rotations applied on request, no authentication) to other network
  interfaces needs to be an explicit choice this add-on doesn't make for you,
  not a side effect of a UI field. Not verified in a real Blender instance
  (same caveat as the rest of this add-on, see "Known Issues") -- the fix is
  a straightforward `bpy.props.IntProperty` plumbed through, but review it
  there before trusting a non-default port works.
- **`landmarker.detect_poses()` was dead code.** The one-shot convenience
  wrapper it provided was fully superseded once `PoseEstimator` + `SolveWorker`
  started handling caching directly in `main_window.py`; nothing in the app or
  test suite called it anymore. **Fixed**: deleted it rather than leave an
  untested, unused function as part of the module's API surface.
- **The 3D viewport's camera was fixed** (`distance=3.0`, a fixed center),
  so an unusually large/small person or an extreme pose (crouching, arms flung
  wide) could render partially out of view. **Fixed**: `viewport_3d.py` now
  computes the skeleton's actual bounding sphere on every `show_skeleton()`
  call and reframes distance + center to it (`_compute_camera_frame`), while
  leaving elevation/azimuth alone so a manual camera rotation isn't reset on
  every solve. Verified with new tests in `test_viewport_camera.py`.
- **Bone-name matching had no escape hatch when the auto-guess failed.**
  Researching comparable Blender mocap add-ons (BlendArMocap, Open Mocap)
  showed they all solve arbitrary custom rigs the same way: a saved manual
  mapping, not just better name-guessing heuristics. `import_pose.py`'s
  `NAME_CANDIDATES_TEMPLATES` guessing was the *only* option before this,
  with no way to fix a mismatch other than renaming bones on your own rig.
  **Fixed**: added a "Bone Mapping" section to the panel -- "Initialize/Refresh"
  creates one row per PoseCraft bone name (`POSECRAFT_BONE_NAMES`, a
  hand-maintained mirror of `retarget/skeleton.py::BONE_NAMES` since the
  add-on can't import that package), each with a searchable target-bone
  dropdown (`prop_search` against the armature's own bones); "Auto-Guess
  Unset" pre-fills blank rows with the existing heuristic so only mismatches
  need manual attention. Saved as a `CollectionProperty` on the armature
  Object, so it persists with the .blend file. An explicit non-blank mapping
  row always wins over the heuristic in `_resolve_bone_name()` -- even if the
  named bone no longer exists (renamed/deleted since), in which case it's
  reported unmatched rather than silently guessing something the user didn't
  ask for. Verified: `tests/test_blender_addon_source.py` parses the add-on's
  source with `ast` (rather than importing it, which needs `bpy`) to confirm
  `POSECRAFT_BONE_NAMES` stays in sync with `skeleton.BONE_NAMES`, and that
  the file is syntactically valid -- the actual Blender-side behavior (the
  UI list, `prop_search`, property registration/persistence) is unverified in
  a real Blender instance, same standing caveat as the rest of this add-on.
- **Nothing was drawn on the photo, and the control panel had grown into one
  flat, ungrouped stack of ~15 controls.** `DetectedPerson.image_landmarks_xy`
  and `bone_confidence` were already computed but only used internally
  (click-to-pick, JSON export flags) -- the photo panel showed the raw image
  with no way to see detection quality in context, or which of several people
  was currently selected, without looking at the separate 3D viewport.
  Meanwhile every control from Load Photo through Send to Blender lived in
  one undifferentiated `QVBoxLayout`. **Fixed**: added
  `retarget/skeleton.py::image_joint_positions_2d()` (a 2D-image-space
  analogue of `joint_positions()`, with a 2D-safe override for the `head`
  bone's extrapolated top-of-skull point, since the 3D version's offset is a
  3-component vector that doesn't apply to 2D coordinates) and wired it into
  `PhotoView.paintEvent`: every detected person's skeleton now draws on the
  photo, the selected person in a bright color on top (low-confidence bones
  in a distinct warning color, reusing `blender_pose.LOW_CONFIDENCE_THRESHOLD`
  so the photo view and the export agree on what counts as low-confidence),
  everyone else dimmed underneath -- deliberately using each person's raw,
  unmirrored landmarks and visibility regardless of the mirror checkbox,
  since mirroring only relabels which exported bone name a side's data feeds,
  it doesn't move anyone's arm in the photo (see `PhotoOverlayPerson`'s
  docstring for why mirroring the overlay would actually be wrong, not just
  unnecessary). Separately, the control panel is now four `QGroupBox`
  sections ("1. Load & Detect", "2. Select Person", "3. Export", "4. Blender
  Live Link") instead of one flat stack. Verified: `image_joint_positions_2d()`
  has new tests in `test_retarget.py`; the overlay's actual drawing was
  checked by grabbing a real rendered frame from a live `MainWindow` with
  fake multi-person data (one person given deliberately low visibility) and
  visually confirming the selected/unselected/low-confidence coloring all
  render correctly (see "Verification").

## Known Issues

- **The CUDA build is large.** All three flavours were built and smoke-tested
  on 2026-10-01 (Python 3.14.7, PyInstaller 6.22.3, models bundled). Each exe
  started and ran its warm-up inference on the expected provider:

  | Flavour | Size | Files | Provider | lite / full / heavy on an RTX 5080 |
  |---|---|---|---|---|
  | dml | 508 MB | 829 | DmlExecutionProvider | 1.3 / 1.6 / 2.6 ms |
  | cpu | 472 MB | 829 | CPUExecutionProvider | 9.2 / 10.1 / 16.9 ms |
  | cuda | 1.73 GB | 843 | CUDAExecutionProvider | 4.6 / 4.9 / 18.5 ms |

  Timings are the median of one detector run plus one landmark-model run on
  random input, not a full solve on a real photo. In the CUDA build the
  `nvidia` folder is 1.02 GB (cuBLASLt alone 493 MB). The DirectML parity test
  passes for all three variants. Which flavour to publish as the main download
  is still a release decision. The DirectML wheel lags the main onnxruntime
  release (1.24.4 vs 1.30.0) and DirectML is in maintenance mode.
- **The ONNX pipeline is a re-implementation, not MediaPipe itself.** It follows
  MediaPipe's sources and matches its output closely (see "Recently Fixed"),
  but it isn't bit-identical, and a future MediaPipe model update that changes
  the graph's op set or I/O would need `tflite_to_onnx.py`/`onnx_pose.py` updates
  (the converter raises `UnsupportedTFLiteOp` rather than guessing).

Ranked roughly by how much they affect whether an exported pose is usable:

- **A straight limb's roll is inferred, not measured.** Upper-arm and thigh
  roll come from the elbow/knee bend plane; below a 30-degree bend
  (`FULL_BEND_DEG`) it blends into the roll carried from the torso, so a
  straight arm's roll is whatever the shoulders imply. Forearm twist is half
  the hand's (`FOREARM_TWIST_SHARE`), which assumes the knuckle landmarks are
  good. Foot roll comes from the ankle-heel-toe plane, which is only a few cm
  across, so it's noisy; a hidden heel falls back to a fixed 25-degree pitch.
  The single spine bone splits a shoulder-vs-hip twist evenly. None of this
  has been compared against real photos with known poses yet (see the golden
  test, which has no photos committed).
- **The world-landmark axis convention follows MediaPipe's documentation
  (+y down, smaller z = closer to the camera; see "Recently Fixed"), but the
  whole export hasn't been checked against a real photo.** Isolated in
  `_world_to_engine()` in `estimation/landmarker.py` specifically so it's the
  one place to fix if an exported pose comes out upside-down or mirrored
  against the source photo. No live solve with a real photo has been compared
  against a Blender rig in this dev environment (no working display was used to run the live solve loop
  end-to-end against a real photo). The GUI now has a user-facing mitigation
  for this specific failure mode (see "Recently Fixed" -- the orientation
  combo next to the mirror checkbox), so an upside-down/wrong-facing result no
  longer requires a code fix to work around, only picking the right dropdown
  entry; it doesn't make the underlying axis-convention guess itself any more
  verified.
- **Rig-aware retargeting is unverified in Blender.** "Match rig rest pose"
  handles different rest directions, extra unmapped bones, chains (Extra
  Bones; Mixamo spines automatically) and bones with Inherit Rotation off --
  all tested only against a simulation of Blender's pose evaluation, as are
  floor contact, keyframing and rotation modes. Not done: Rigify's
  constraint-driven
  MCH chains (posing the controls works; a DEF/ORG-only rig is driven by
  constraints the solve doesn't model). A v1 add-on rejects v2 files
  ("update the add-on"), by design.
- **The Blender add-on -- bone-name matching, the manual Bone Mapping UI, the
  rig-aware solve, and the Live Link server and its reply -- is unverified
  against a real Blender install.** The Live Link server's reads are
  non-blocking now (B16; the framing is unit-tested on a real socket), but
  the timer loop itself has only been reviewed, not run in Blender.
  `bone_name_candidates()` in `import_pose.py` guesses at Rigify
  (`upper_arm_fk.L`, `_ik`, then `DEF-`/`ORG-`) and Mixamo (`mixamorig:*`)
  names, and its pure helpers are unit-tested, but it was never loaded into
  Blender or run against an actual Rigify or Mixamo armature; the manual
  bone-mapping panel (`prop_search`, the `CollectionProperty` on `Object`,
  the `UIList`) and the Live Link server's `bpy.app.timers`-based
  non-blocking accept loop are likewise only checked by code review and a
  source-level consistency test (`test_blender_addon_source.py`), not a real
  run. Check the add-on's unmatched-bone warning after your first real import,
  and use the Bone Mapping panel to fix any mismatch by hand -- that's the
  whole reason it exists now, since extending the auto-guess template list is
  a losing game against every possible rig's own naming convention.
- **No code signing** on the built `.exe` -- expect SmartScreen to flag it as
  unrecognized on a machine other than the one it was built on.
- **The release workflow has never run** (`.github/workflows/release.yml`),
  and the spec's K3/K4 changes have never been through a build: run it once
  (a manual `workflow_dispatch`) and launch each zip before trusting it. No
  headless-Blender job exists yet (T4).
- **No hands or fingers yet (B11).** The converter can now run MediaPipe's
  hand models, but the pose-driven hand crop, the hand landmark pass and
  finger bones (export, BVH, add-on mapping) aren't built.
- **Video tracks one person** (the leftmost found in the first frame) and
  re-detects only when it's lost; there's no identity tracking across
  people, no rotation-space smoothing, and the root's horizontal motion comes
  only from planted feet (a jump or a slide has none).
- **No DAZ Studio export target -- deliberately not started, not just missing.**
  PoseCraft only exports to Blender; the original PoseForge this project is
  inspired by targets DAZ Studio directly. Investigated adding a `.duf` export
  alongside the Blender one: DAZ's DSON format is JSON-based (not a proprietary
  binary format), so it's not structurally out of reach, but it's a
  meaningfully bigger scope than either export/blender_pose.py's flat
  quaternion-per-bone JSON: DSON models a full asset graph (channels, ERC
  links, morphs), and Genesis 8 vs Genesis 8.1 vs Genesis 9 use different bone
  names, which is exactly the problem a paid third-party DAZ asset ("Bone
  Minion") exists solely to solve within DAZ's own ecosystem, and the same
  reason PoseForge itself has a "pick your generation" control rather than one
  fixed bone mapping. Revisit this as a real feature addition (with its own
  bone-name table per Genesis generation and its own test coverage), not as an
  extension of the existing Blender exporter.

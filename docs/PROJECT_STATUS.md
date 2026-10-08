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
     Runtime -- DirectML on the best hardware DX12 adapter, then CUDA (TF32 off)
     when available, each kept only if it matches the CPU on a probe input at
     load time, CPU otherwise;
     photos over 2048 px are area-downscaled first (estimation/onnx_pose.py
     re-implements MediaPipe's detector -> ROI -> landmark calculator graph);
     multi-person, with overlapping-tile detection for group photos
  -- optional hands (B11, "Hands and fingers"): MediaPipe's 21-point hand
     landmark model (pinned + SHA-256-checked like the pose models, palm
     detector never converted) on crops derived from each person's pose
     wrist/index/pinky landmarks (Holistic's HandDetectionsFromPoseToRects),
     refined once from the hand's own landmarks (onnx_pose.HandLandmarker)
  -- video (estimation/video.py): up to max_people people with stable track
     IDs (MultiPersonTracker: each followed from its own previous landmarks,
     detector on frame 0, on a loss and every 10 frames while under the
     limit; lost tracks re-identified by keypoint distance plus an RGB
     appearance histogram for 30 frames; converged tracks merged), per-person
     One Euro smoothing, optional zero-phase rotation smoothing
     (smooth_rotation_frames), floor contact and foot-locked root motion
        |  landmarks converted to the engine frame (right-handed, Y-up, +X image right =
        |  subject's left, +Z toward camera = the way the subject faces);
        |  each DetectedPerson also carries 2D image-space landmarks + per-landmark visibility
        v
retarget/skeleton.py -- 15-bone humanoid hierarchy, T-pose rest directions,
     plus 15 finger bones per hand (Rigify names) when hand landmarks are
     given -- the hand then framed from the palm, finger knuckle segments by
     the palm's lateral axis (the thumb by the back-of-hand normal), the
     other segments as hinges
  -- a full frame per bone (direction + roll): spine from the hip/shoulder lines,
     head from the ear line (neck takes 40% of its twist), upper arm/thigh roll
     from the elbow/knee bend plane (forearm/shin are pure hinges), hand from
     index/pinky (forearm takes half its twist), foot from heel -> toe and the
     ankle-heel-toe plane; unmeasurable roll is carried from the parent
  -- world -> parent-relative; forward_kinematics() poses the rest skeleton
     from those rotations (the 3D preview and the BVH use the same rest skeleton)
  -- bone_confidence() derives a per-bone score from landmark visibility
  -- manual correction: flip_limb_depth()/flip_hand_depth() (per-limb depth
     flip), move_joint() (a 2D joint drag lifted to 3D through a fitted
     weak-perspective camera, image_to_world_scale(), keeping the bone's
     length and depth side) and apply_bone_overrides()/bone_axes_rotation()
     (per-bone adjustments about Blender-style bone axes; bone_local_euler_deg()
     is the inverse, for the 3D hover label); hip_height_above_floor() for
     ground contact
        v
export/blender_pose.py -- axis-converts Y-up -> Blender Z-up, writes JSON (format v2:
     local + world rotation + rest direction per bone, optional "root"
     hip height / leg length); export/bvh_export.py -- single-frame or
     animated BVH, the root standing on the floor
  -- low-confidence bones flagged "low_confidence": true, bones reaching
     outside the photo "out_of_frame": true (per-landmark presence, B10)
        |
        +---------------------------------------------+
        v  (file)                                      v  (live)
pose_craft_pose .json file            export/live_link.py: length-prefixed
        |                              JSON over TCP to host:port (UI-configurable,
        |                              default 127.0.0.1:8642); the add-on replies
        |                              with what it applied (protocol 2 for a
        |                              pose; protocol 3 for a video track sent as
        |                              a "pose_craft_animation" document, v1, in
        |                              chunks of <= 30 frames / 200 KB)
        v                                              |
export/blender_addon/import_pose.py (installed inside Blender; add-on 0.7.1,
  bl_info minimum Blender 3.6, verified in Blender 5.2.2)
  -- Load Pose JSON operator, or a Live Link server (bpy.app.timers-polled,
     fully non-blocking: accept and incremental reads with a 2 s per-message
     deadline, 256 KB cap) that applies a received pose to the captured
     target armature, as its own undo step, optionally keyframed -- or keys
     a whole animation (every frame through the same retarget path, at the
     clip's timing in the scene's frame rate, one undo step)
  -- bone name resolution: an explicit per-armature manual mapping row
     (POSECRAFT_BoneMapping, saved with the .blend file) wins if set, else
     falls back to name-guessing (exact, Rigify upper_arm_fk.L/_ik, DEF-/ORG-, Mixamo)
  -- "Pose rig controls" (default): a mapped bone whose rotation a constraint
     overrides (Rigify DEF-/ORG-) is redirected to the bone driving it
     (control_bone_for / constraint_driver, FK preferred), and the Rigify
     limbs posed are switched to FK (rigify_fk_switches: IK_FK = 1 on
     upper_arm_parent.L / thigh_parent.L)
  -- "Match rig rest pose" (default): rig_pose_basis_evaluated() aims each
     mapped bone at PoseCraft's world target against the rig's own rest
     direction and its parent's pose as Blender evaluated it (constraints and
     drivers included; level by level, two passes), which covers A-pose,
     Mixamo Hips, clavicles, bones without Inherit Rotation and Rigify's MCH
     parents; a bone mapped to a chain (Mixamo Spine/Spine1/Spine2, Rigify
     spine_fk/.001-.003) is spread over it; off / v1 files: R^-1 q R per bone;
     each bone keeps its rotation mode
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
`show_skeleton()` on every update rather than diffing; it also draws the
hover label and the optional photo backdrop. The window is a three-pane
`QSplitter` (scrollable controls, photo, 3D view); the control panel is split
into `QGroupBox` sections -- "1. Load & Detect" (`_build_detect_group`), "2.
Select Person" (`_build_person_group`), "3. Correct"
(`_build_correction_group`: per-bone adjustment spin boxes; joint drags and
depth flips happen on the photo), "4. Export" (`_build_export_group`), "5.
Blender Live Link" (`_build_live_link_group`, incl. auto-send) -- plus a
"Video" group (`_build_video_group`: frame slider, rotation smoothing,
animation export, Send Animation) that appears once a video is tracked.

Added in the P2 round (2026-09-30): busy state lives in `_set_busy()` and
`_set_result_actions_enabled()` (one place for what a running solve disables,
the status-bar progress bar and the Cancel button); `WarmUpWorker` creates
and warms the remembered model at startup when it's available offline
(`MainWindow(warm_up=True)`, which `app.py` passes; a Solve clicked meanwhile
waits for it); `VideoSolveWorker` tracks a video; manual corrections are
`PersonEdits` per person, changed only through `QUndoStack` commands, and
applied in `_document_inputs_for()` so every export path gets them; settings
(last folder, recent files, host/port, model, GPU, group photo, max people,
hands, rotation smoothing, photo backdrop, auto-send and its debounce, window
and splitter geometry) persist through `QSettings` (an INI under the user's
config folder; `tests/conftest.py` points it at a temp folder).

The pure modules -- retargeting (`retarget/skeleton.py`), axis conversion and
the file formats (`export/blender_pose.py`, `export/bvh_export.py`), model
download/conversion and the ONNX pipeline (`estimation/`), video tracking and
smoothing (`estimation/video.py`), `imaging.py` and the Live Link wire
protocol (`export/live_link.py`) -- are unit-tested directly; the GUI is
driven on an offscreen Qt event loop with fake estimators, and the add-on's
pure helpers are compiled out of its source with `ast` (plus an opt-in run in
a real Blender) -- see "Verification" below.

## Primary Entry Points

| Purpose | Entry point |
|---|---|
| Desktop app (load photo, solve, preview, export) | `pose_craft/app.py`, run as `python -m pose_craft.app` |
| Pose estimation entry point, world-axis conversion, cached `PoseEstimator`, model download + ONNX conversion cache | `pose_craft/estimation/landmarker.py` |
| BlazePose pipeline on ONNX Runtime (detector, group-photo tiles, ROI, landmarks, video tracking, GPU/CPU session setup) | `pose_craft/estimation/onnx_pose.py` |
| Video: tracking people through a clip with stable IDs, per-person One Euro smoothing, foot locking | `pose_craft/estimation/video.py` |
| TFLite -> ONNX converter for the pose (and hand) models | `pose_craft/estimation/tflite_to_onnx.py` |
| Pre-convert the models for a release build | `scripts/convert_models.py` |
| Release build (DirectML + CPU zips, add-on zip, SHA256SUMS) | `.github/workflows/release.yml` |
| Bone retargeting math + per-bone confidence | `pose_craft/retarget/skeleton.py` |
| Pose file format (write from the app, read by the add-on) | `pose_craft/export/blender_pose.py` |
| Standard BVH export (own additive path, alongside the JSON/Live Link one) | `pose_craft/export/bvh_export.py` |
| Live Blender link client (app side) | `pose_craft/export/live_link.py` |
| Blender-side importer + Live Link server (install manually inside Blender, not via pip) | `pose_craft/export/blender_addon/import_pose.py` |
| Unicode-safe image loading + pre-solve photo quality checks | `pose_craft/imaging.py` |
| Main window: controls, workers, undo, settings, menus | `pose_craft/gui/main_window.py` |
| Photo widget: overlay, click/hover picking, joint drag | `pose_craft/gui/photo_view.py` |
| 3D preview: FK skeleton, floor/gizmo, hover label, photo backdrop | `pose_craft/gui/viewport_3d.py` |
| Log file + uncaught-exception dialogs | `pose_craft/diagnostics.py` |
| PyInstaller build spec (`POSECRAFT_FLAVOR=dml` default, `cuda`, `cpu`) | `PoseCraft.spec` |
| CI (compileall + pytest, Ubuntu, Python 3.12, `requirements-test.txt`) | `.github/workflows/ci.yml` |
| Real-photo golden generator / real-Blender add-on check (opt-in) | `scripts/generate_goldens.py`, `tests/test_real_blender.py` |
| App icon generation (re-run only if the design changes) | `scripts/generate_icon.py` |

## Release

```text
dist/PoseCraft/PoseCraft.exe        -- built with assets/icon.ico as the exe icon
dist/PoseCraft/_internal/           -- PySide6, OpenCV, ONNX Runtime (DirectML, CPU or CUDA build) + (CUDA flavour only) the CUDA 13/cuDNN 9 DLLs
dist/PoseCraft/_internal/models/    -- pre-converted ONNX models (lite/full/heavy + shared detector + hand) + manifest.json (when build/models existed)
```

Built with `pyinstaller PoseCraft.spec` (onedir, `--windowed`, `collect_all`
for `onnxruntime`/`cv2`, pyqtgraph's modules and data without its examples,
and -- only when no pre-converted models are present -- `onnx`/`tflite`;
`scripts/convert_models.py` writes the three variants' ONNX models plus one
shared detector and a manifest to `build/models`, and the spec then bundles
them under `models/` and leaves the converter out, so the app needs no
download or network on first run. The spec also excludes matplotlib, PIL and
the unused Qt modules (Quick/Qml/Network/Test/Pdf/VirtualKeyboard) and filters
their DLLs. All three flavours were built with these spec changes on
2026-10-01 (sizes under "Known Issues"); the hand model was added to the
bundle afterwards (2026-10-02) and hasn't been through a full build since.
`POSECRAFT_FLAVOR` picks the flavour: `dml` (the default when unset; the
build venv must have `onnxruntime-directml`, or the spec stops), `cuda` or
`cpu`. A CUDA build (`POSECRAFT_FLAVOR=cuda`) also bundles
the NVIDIA CUDA/cuDNN DLLs from the pip `nvidia-*` packages kept in their
site-packages layout under `_internal/nvidia/` (where `onnxruntime.preload_dlls()` looks) when installed -- that's what makes the GPU build
so much bigger than the old ~480 MB one; a build from a venv without
`requirements-gpu.txt` leaves them out and runs CPU-only -- plus `assets/icon.png`+`assets/icon.ico`
bundled as `datas` so `app.py` can set the window icon at runtime via
`sys._MEIPASS`). `dist/` and `build/` are gitignored -- distribute the built
folder directly (zip it, or wrap it in an installer) rather than committing
it. Release builds aren't code-signed yet: expect SmartScreen to flag them as unrecognized on a
machine other than the one they were built on. `scripts/sign_build.py` signs `dist/PoseCraft/PoseCraft.exe`
with the certificate in `CODESIGN_THUMBPRINT` (checked with a self-signed test certificate,
2026-10-07); a trusted certificate is what removes the warning.

`requirements.txt` is now pinned to the exact versions this project was built
and tested against (was unpinned before); `requirements-test.txt` is a
separate, lighter pin set (numpy/scipy/opencv-python-headless/pytest plus the
CPU onnxruntime, onnx and tflite) for CI.
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

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest -q
```

**Current count (2026-10-07, after the lower-priority round): 423 passed, 8
skipped** in about 30 s; `ruff check` and `mypy` clean. The eight skips are the
LiteRT reference test, the real-photo golden test (no photos committed), the
four opt-in GPU parity tests (`POSECRAFT_GPU_TESTS=1`) and the two opt-in
real-Blender tests (`POSECRAFT_BLENDER`: the Live Link/rig check and the
extension install). Earlier counts: 402/7 after the real-Blender check
(2026-10-03), 298/6 after the P2 round
(2026-09-30), 357/6 after the "important items" round (2026-10-02), 401/6
after the video/Live Link round (2026-10-03).

The suite covers `retarget/skeleton.py` (including `image_joint_positions_2d()`, the 2D
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
venv. The hands, Rigify, video tracking, rotation smoothing, animation and
Live Link GUI tests (`test_hands.py`, `test_rigify.py`,
`test_video_tracking.py`, `test_rotation_smoothing.py`,
`test_blender_animation.py`, `test_gui_important.py`,
`test_gui_live_link.py`) cover the 2026-10-02/03 features; the real-hand-model
test runs only where the hand model is cached. `tests/test_gui_flows.py`
drives the real window on an offscreen Qt event loop with fake
estimators: busy states, cancel, the close-waits-for-the-worker guarantee,
the non-blocking send, warm-up, settings, drag-and-drop, picking, manual
correction with undo, exports and video scrubbing -- the threading checks
below that used to be one-off scripts. `tests/test_gpu_parity.py` (opt-in:
`POSECRAFT_GPU_TESTS=1`) compares every available GPU provider with the CPU;
it passed on CUDA (RTX 5080) and fails, as it should, with TF32 switched back
on. CI's lighter venv also skips the Qt
tests; re-run `pytest` rather than trusting a number written here.

Everything else has been checked manually (or by an opt-in test) rather than
by the default suite, since it needs a real display, GPU or Blender install:

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
- Each built flavour's `.exe` (2026-10-01) was launched, stayed running (no
  missing-DLL crash) and ran its warm-up inference on the expected provider
  (see "Known Issues" for sizes and timings).
- The Blender add-on (`import_pose.py`) was run in a **real Blender 5.2.2
  LTS, headless** (`--background --factory-startup`, user folders pointed at
  a temporary folder so the real preferences/add-ons were untouched), with
  documents generated by the app's own code from synthetic poses. Measured
  error = the angle between each posed rig bone and the document's target
  direction (world rotation x rest direction):
  - Install from the .py file, enable, disable: registers and unregisters
    cleanly, no deprecation warnings on 5.2 (the legacy `bl_info` add-on
    path still works; no extension manifest needed).
  - Load Pose JSON on a PoseCraft-shaped rig (45 bones with fingers), with
    "Match rig rest pose" on and off: all bones applied, max error < 0.001
    deg (the rest pose was 95-165 deg off). XYZ / ZXY Euler, axis-angle and
    quaternion bones: same error, each bone keeps its rotation mode.
  - A Mixamo-style A-pose rig (`mixamorig:` names, Hips, clavicles,
    Spine/Spine1/Spine2, 30 finger bones, random bone rolls, object moved
    off the origin): 47 bones applied (the spine spread over three), 0
    unmatched, max error < 0.001 deg; Auto-Guess fills all 45 mapping rows
    (spine with "Spine1, Spine2" as extra bones) and posing through those
    rows gives the same result. With "Match rig rest pose" off the A-pose
    arms are 45 deg off, as documented.
  - A **generated Rigify rig** (bundled Rigify enabled, Human metarig,
    `pose.rigify_generate`, 706 bones) with "Pose rig controls" on: 48
    controls posed (`spine_fk`..`spine_fk.003`, `neck`, `head`, the FK limb
    controls, the finger controls), all four `IK_FK` switches set to 1.0, and
    the ORG- (and DEF-) bones land on the targets with max error < 0.001 deg
    for the body and all 30 fingers (the neck measured as the ORG-spine.004
    -> .005 chord; each single neck bone differs from it by Rigify's own
    5.6 deg rest curvature). With "Pose rig controls" off the limbs stay on IK
    and miss by up to 37-120 deg depending on the pose -- the option is what
    makes Rigify work.
  - Keep feet on the floor (a crouch): the lowest posed point lands exactly
    on the rest pose's lowest point on both the PoseCraft-shaped rig and the
    Rigify rig (DEF- bones); without it the crouch floated 20-26 cm up.
  - Insert keyframes + Then advance 5: two Load Pose JSON calls keyed frames
    1 and 6 (60 F-curves, 120 keys), current frame 11; with floor contact the
    root's location is keyed too. An invalid JSON file is reported, nothing
    raised.
  - BVH: the app's single-pose and 40-frame BVH imported with Blender's own
    importer: 15 bones, 40 keyed frames at 30 fps, root height equal to
    `floor_root_position()` (< 1e-6 m), lowest point on the floor (z = 0),
    limb directions < 0.001 deg from the JSON's targets for the same pose.
  - **Live Link, for real**: the server started by the panel operator on a
    non-default port, the app's own `live_link` client sending from a
    separate process. Timers don't fire while a `--background` script runs,
    so the script called the timer callback itself (what `bpy.app.timers`
    does in the UI). A single pose: reply ok, 15 applied, `shin.R` flagged
    low-confidence, error < 0.001 deg, applied to the captured target with
    another object active. A 45-frame animation with a 2-frame gap (two
    chunks): keyed from the current frame, start/end frames anchored across
    chunks, the current frame restored, error < 0.001 deg on every keyed
    frame (also on the Rigify rig, ~2 s for 45 frames), exactly one
    "PoseCraft animation" undo push -- and one `ed.undo` removes the whole
    animation (both chunks) while the single pose before it stays. A sender
    that stalls is dropped after the 2 s deadline; an oversized length
    header gets an error reply. Stop, restart and disabling the add-on free
    the port.
  - Found and fixed: Stop Live Link / disabling the add-on left the
    persistent timer registered until its next tick (harmless in the UI,
    but a stale callback of an unloaded module); `_stop_live_link_server()`
    now unregisters it (add-on 0.7.1).
  - Not checkable headless: the panel's drawing (the sidebar, the Bone
    Mapping `UIList`/`prop_search`), the Load Pose JSON file browser, and
    the timer being fired by Blender's own event loop rather than the test.
  `tests/test_real_blender.py` repeats these checks in one Blender run (~5 s)
  when `POSECRAFT_BLENDER` points at `blender.exe`; it's skipped otherwise
  (CI has no Blender).

Re-run `pytest` yourself before trusting a pass count, and treat "manually
checked" above as weaker evidence than an automated test -- it was true at the
time each check ran, not guaranteed to stay true as the code changes.

## Recently Fixed

- **Lower-priority round (2026-10-07): B10, B13, P5, R4, C3, T6, K6, signing.**
  - **B10** the landmark tensor's presence column rides on
    `PoseResult`/`DetectedPerson`; bones reaching a landmark outside the photo
    are drawn dotted and flagged `out_of_frame` in JSON/Live Link, apart from
    `low_confidence` (checked on a real head-and-shoulders portrait). An
    Advanced toggle sets the detection and low-confidence thresholds, both
    remembered.
  - **B13** folder batch processing: `pose_craft/batch.py`, `python -m
    pose_craft.cli --input DIR --output DIR [--format json|bvh] [--mirror]
    [--model] [--cpu] [--hands] [--group-photo] [--detection-threshold]`, and
    File > Batch Process Folder with progress and Cancel. The export logic
    moved to the Qt-free `pose_craft/posing.py`, so both match Export All People.
  - **P5** the runtime cache stores the byte-identical detector once by content
    hash; BlazePose shares one detector session per file and device.
  - **R4** Help > Diagnostics (versions, ONNX Runtime providers, DirectX 12
    GPUs, the model's device and fallback reason, the log file; Copy) and
    Help > About; the GPU-fallback reason stays in the status bar's tooltip.
  - **C3/T6** `pyproject.toml` (pytest/ruff/mypy config), ruff and mypy in a CI
    lint job, and a `windows-latest` job running the whole suite offscreen with
    PySide6 and coverage.
  - **K6** `blender_manifest.toml` ships in the add-on zip built by
    `scripts/build_addon_zip.py`: one zip for Blender 4.2+ extensions and
    3.6-4.1 legacy installs (installed and enabled in a real Blender 5.2).
  - **Signing** `scripts/sign_build.py` Authenticode-signs
    `dist/PoseCraft/PoseCraft.exe` (see "Release").

- **Real-Blender check (2026-10-03): B5, B12, T4.** The add-on ran headless
  in Blender 5.2.2 against a PoseCraft-shaped rig, a Mixamo-style rig, a
  generated Rigify rig, Blender's BVH importer and a real Live Link server
  (results under "Verification"). Everything landed on its targets (< 0.001
  deg); one bug: stopping Live Link (or disabling the add-on) left its
  persistent timer registered until the next tick -- now unregistered
  (add-on 0.7.1, `test_stopping_live_link_unregisters_its_timer`). New
  opt-in `tests/test_real_blender.py` (needs `POSECRAFT_BLENDER`).

- **Video / Live Link round (2026-10-03): B12, F5, F8, F10.** 357 passed / 6
  skipped before, 401 / 6 after (new: `tests/test_blender_animation.py`,
  `test_gui_live_link.py`, `test_rotation_smoothing.py`; more in
  `test_video_tracking.py`). Finishes work that was half-wired in the tree:
  - **Appearance re-identification (B12).** A lost track is matched by
    keypoint distance + 0.35 x an RGB-histogram distance (4x4x4 bins over the
    landmark box). Fixed: the histogram is now a strided sample of at most
    ~4096 pixels via `bincount` (was `histogramdd` over every pixel of a
    full-HD box each frame), and a track's signature is a running blend (0.8
    memory) so one occluded frame doesn't overwrite it. A scripted crossing
    behind an occluder keeps its IDs with appearance and swaps without it.
  - **Rotation smoothing (B12).** "Rotation smoothing" (0-100 %, default 35,
    remembered) smooths each track's solved bone rotations. Fixed: it was a
    one-way exponential filter (up to ~9 frames of lag at 100 %) that bridged
    tracking gaps as if the frames were adjacent and built a scipy Slerp per
    bone per frame; it's now a vectorized shortest-arc slerp run forwards and
    backwards (zero-phase), gap-aware (catches up by `1 - keep**gap`), sign-flip
    safe; 0 returns the very same Rotation objects. The window caches each
    track's solved frames (they were re-solved twice on every preview update)
    and uses the displayed person's track -- with the selected person out of
    frame, the person shown used to get the unsmoothed pose.
  - **Send Animation to Blender (B12).** The app's half was there; the add-on
    rejected the document ("Not a PoseCraft pose file"). Add-on 0.7.0 adds
    `pose_craft_animation` (Live Link protocol 3): the whole chunk is
    validated first, each frame is posed through `apply_pose_document` (the
    same retarget path) at `start + round((index - first) * scene_fps /
    clip_fps)` and keyed, the current frame is restored, quaternion keys are
    kept in one hemisphere, and one undo step is pushed after the last chunk.
    The 256 KB payload cap is kept: `live_link.send_animation_document` splits
    the clip into chunks of at most 30 frames / 200 KB, the first reply's
    start frame anchors the rest. Single poses still say protocol 2, so a
    0.4-0.6 add-on keeps working; an older add-on refuses animations with
    "update the add-on". Root motion isn't sent. Tested with the add-on's
    functions on a simulated scene and through a real socket with the
    add-on's framing -- since verified in Blender 5.2.2 (see "Verification").
  - **Auto-send on change (F10)** (off by default; it and its debounce are
    remembered): debounced, one send in flight at a time, no send without a
    selected person, pauses after an unreachable Live Link instead of
    retrying on every change (a manual send resumes it), no dialogs, and the
    timer stops on close. Fixed: the queue check built a whole Live Link
    document (re-solving a video track) on every change. **Export All People
    (BVH)** for photos (F10), asking before overwriting.
  - **Photo hover (F5).** Hovering highlights the bone/joint/person and a
    tooltip names it (with the track's number in a video); a redraw while
    the mouse rests keeps the highlight; a bone without its tail no longer
    raises.
  - **Accessibility (F8).** A yellow keyboard-focus outline, accessible names
    on every new and existing control (a test checks every button, combo,
    spin box, slider and field has a name or buddy label), and mnemonics that
    don't collide (`&Rotation smoothing` clashed with `po&rt`). No real
    screen-reader check.

- **"Important items" round (2026-10-02): B5 Rigify, B11, B12, F4, F9.** 307
  passed / 6 skipped before, 357 / 6 after (new: `tests/test_hands.py`,
  `test_rigify.py`, `test_video_tracking.py`, `test_gui_important.py`):
  - **Hands and fingers (B11).** A "Hands and fingers" checkbox (off by
    default, remembered). The hand crop comes from the pose's wrist, index and
    pinky landmarks as in MediaPipe Holistic (scale 2.7, shift -0.1), then one
    re-crop from the hand's own landmarks (scale 2.0); the palm detector is
    never converted. The hand model is pinned
    (`.../hand_landmarker/float16/1/`, SHA-256 `fbc2a300...`, identical to
    `/latest/`) and bundled by `scripts/convert_models.py` (manifest key
    "hand", 11 MB as float32 ONNX). Measured on an RTX 5080: 1.2 ms per hand
    pass on the CPU, 0.47 ms on DirectML, 1.3 ms on CUDA; both GPU providers
    within 3.4e-6 of the CPU (landmarks 0.0001 px). 15 finger bones per hand
    (Rigify names) in the JSON (still format 2: a body-only file is
    unchanged), the BVH (finger joints only when solved) and the add-on
    (Rigify finger controls match by name, Mixamo `LeftHandIndex1`... via a
    table, Bone Mapping rows for all 45 bones). A rest hand solves to straight
    fingers, a curl to a pure hinge about the knuckle line, finger rotations
    are invariant to turning the whole body, and FK with measured hand
    proportions lands on the hand landmarks. Not checked on a real photo of
    hands; photos only (video stays body-only).
  - **Multi-person video (B12).** Load Video tracks up to "Max people" people
    with stable IDs (`video.MultiPersonTracker`), smooths each separately and
    exports one animated BVH per person ("Export All People (BVH)...", or the
    selected person with "Export Animation BVH..."). Manual corrections in a
    video belong to the track and apply to all its frames; the selected
    person stays selected while scrubbing. Tested on scripted scenes
    (moving people, a short loss re-identified, a newcomer, a long loss
    getting a new ID, converging tracks merged); not on real footage.
  - **Rigify (B5).** "Pose rig controls" (on by default): constraint-driven
    DEF-/ORG- bones redirected to their FK controls, posed limbs switched to
    FK, and every bone solved against its evaluated parent. Tested against a
    simulated Rigify-like rig (Copy Transforms chains, an IK/FK-driven
    influence, an MCH parent following the torso): each of the three fixes
    is needed for the deform bones to reach the targets, and with none of
    the constraints the evaluated solve equals the old one. Add-on 0.6.0.
    Since verified on a generated Rigify rig in Blender 5.2.2 (see "Verification").
  - **2D joint drag (F4).** The selected person's shoulders, elbows, wrists,
    hips, knees and ankles are draggable on the photo; a drag is lifted to 3D
    through a weak-perspective camera fitted to the person (keeping the bone
    length and depth side), re-solves the limb, and is one undo step.
  - **3D view (F9).** A hover label (bone name, local X/Y/Z Euler degrees
    about the bone's own axes, confidence) and an optional photo backdrop
    placed with the same camera fit, so the photo's body lines up with the
    raw landmarks. Neither is checked as rendered pixels (offscreen Qt has no
    GL context): the tests check the projection, the label text, the plane's
    corners and transform.

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
    the slider. (Superseded 2026-10-02 by multi-person tracking, see above.)
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
    "Release"; the spec changes were built on 2026-10-01, the workflow
    hasn't run).
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
  loopback socket server standing in for the add-on's side of the protocol
  (the add-on's Blender-side server has since been run in a real Blender
  5.2.2, see "Verification").
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
  not a side effect of a UI field. Since verified in Blender 5.2.2: the
  real-Blender check starts the server on a non-default port and the app's
  client reaches it (see "Verification").
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
  the file is syntactically valid. In Blender 5.2.2 (see "Verification")
  registration, Auto-Guess and posing through mapping rows have since been
  checked; the UI list and `prop_search` drawing still haven't.
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
  passes for all three variants. **DirectML is the default flavour and the
  main download** (decided 2026-10-02): fastest on the GPU measured, nearly the
  CPU build's size, and it runs on AMD/Intel GPUs too. It is used only on a
  hardware DirectX 12 adapter (`onnx_pose.best_directml_adapter()` via DXGI:
  software adapters such as Microsoft Basic Render Driver are excluded, the
  adapter with the most dedicated memory is passed as `device_id`) and only if
  each model's outputs on a probe image match the CPU's to 1% of the output
  range (`gpu_session_problem()`; measured worst case 7e-4, landmarks within
  0.004 px, on an RTX 5080 and an AMD Radeon iGPU). Otherwise the app runs on
  the CPU and reports why (`gpu_fallback_reason`, shown in the status bar).
  The DirectML wheel lags the main onnxruntime release (1.24.4 vs 1.30.0) and
  DirectML is in maintenance mode.
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
- **The Blender add-on is verified headless in Blender 5.2.2 only** (see
  "Verification": a PoseCraft-shaped rig, a Mixamo-style rig built in the
  test, a freshly generated Rigify Human rig, the BVH importer and a real
  Live Link server). Still unchecked: the panel's UI (sidebar drawing, the
  Bone Mapping list and its bone search field, the file browser), Live Link
  with the timer fired by Blender's own UI event loop (headless, the test
  calls the same callback itself), other Blender versions (`bl_info` says
  3.6+), a rig imported from an actual Mixamo FBX, and Rigify rigs other
  than the stock Human metarig (customised or older generations). A Rigify
  rig is left in FK; switch limbs back to IK yourself. Check the add-on's
  unmatched-bone warning after a first import on any other rig, and use
  Bone Mapping to fix a mismatch. A v1 add-on rejects v2 files ("update the
  add-on"), by design.
- **An animation into a scene with a lower frame rate keys some frames
  twice.** A 30 fps clip in a 24 fps scene maps several clip frames onto
  the same Blender frame (rounding), and the later one wins: 45 clip frames
  became 36 keyed frames in the real-Blender check, while the reply's
  `animation_frames` still counts 45. The timing is right; set the scene to
  the clip's frame rate to keep every frame.
- **No trusted code signing** on the built `.exe` -- `scripts/sign_build.py`
  works (checked with a self-signed test certificate), but until it runs with a
  real certificate SmartScreen flags the app as unrecognized on other machines.
- **The release workflow has never run** (`.github/workflows/release.yml`;
  no tag has been pushed), and no local build has bundled the hand model yet
  (the 2026-10-01 builds predate it): run the workflow once (a manual
  `workflow_dispatch`) and launch each zip before trusting it. No
  headless-Blender job runs in CI (no Blender on the runners); the opt-in
  `tests/test_real_blender.py` is the local equivalent (T4).
- **Fingers are unverified on real photos (B11).** The hand pass follows
  MediaPipe Holistic's crop rule without its re-crop model, finger twist
  isn't measured (the knuckle segment is framed by the palm, the rest are
  hinges, the thumb's roll follows the palm), hands whose pose landmarks are
  under 0.5 visibility aren't searched, and video is body-only.
- **Video identities are heuristic (B12).** Tracks are followed by crop
  overlap; a lost track is re-identified by keypoint distance plus a colour
  histogram, so two similarly dressed people crossing closely can still
  swap IDs (appearance isn't used while two tracks overlap), and someone gone
  for over 30 frames comes back as a new person. The root's horizontal motion
  comes only from planted feet (a jump or a slide has none), and Send
  Animation to Blender sends rotations only (no root motion). Tested on
  scripted scenes, not real footage.
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

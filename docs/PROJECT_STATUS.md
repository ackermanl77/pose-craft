# Project Status

## Architecture

```text
Photo (loaded via imaging.imread_unicode, unicode-path-safe)
        |
        v
PoseEstimator (pose_craft/estimation/landmarker.py)
  -- wraps one cached MediaPipe PoseLandmarker instance, multi-person via num_poses
        |  landmarks converted to the engine frame (Y-up, +X right, +Z toward camera);
        |  each DetectedPerson also carries 2D image-space landmarks + per-landmark visibility
        v
retarget/skeleton.py -- 15-bone humanoid hierarchy, T-pose rest directions
  -- per-bone shortest-arc (swing-only) quaternion, world -> parent-relative
  -- bone_confidence() derives a per-bone score from landmark visibility
        v
export/blender_pose.py -- axis-converts Y-up -> Blender Z-up, writes JSON
  -- low-confidence bones flagged "low_confidence": true
        |
        +---------------------------------------------+
        v  (file)                                      v  (live)
pose_craft_pose .json file            export/live_link.py: length-prefixed
        |                              JSON over TCP to 127.0.0.1:8642
        v                                              |
export/blender_addon/import_pose.py (installed inside Blender)
  -- Load Pose JSON operator, or a Live Link server (bpy.app.timers-polled,
     non-blocking accept) that applies a received pose automatically
  -- applies bone quaternions to the active armature's pose bones
```

`pose_craft/gui/main_window.py` is the only orchestrator: it owns the loaded
photo, runs `PoseEstimator.detect()` on a background `QThread` (`SolveWorker`)
so the UI doesn't freeze while the pose-landmarker model downloads/runs, and
re-solves `retarget.skeleton.solve_bone_rotations()` synchronously (cheap, no
model inference) whenever the person picker, a photo click, or the mirror
checkbox changes. The `PoseEstimator` created for the first solve is cached on
`MainWindow._estimator` and reused for every later solve in the same session
(closed in `closeEvent`), rather than rebuilding MediaPipe's graph every click.
`gui/photo_view.py`'s `PhotoView` keeps the full-resolution `QPixmap` so it can
rescale cleanly on window resize, and turns a click into a normalized
image-space point that `main_window.py` matches against each detected
person's 2D centroid (`_nearest_person_index`) to drive the person picker.
`gui/viewport_3d.py` is a thin wrapper around `pyqtgraph.opengl.GLViewWidget`
that draws `retarget.skeleton.joint_positions()` as line segments + points --
it has no state of its own beyond the current skeleton, `main_window.py`
always calls `clear_skeleton()` before `show_skeleton()` on every update
rather than diffing.

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
| Pose estimation, MediaPipe axis conversion, cached `PoseEstimator` | `pose_craft/estimation/landmarker.py` |
| Bone retargeting math + per-bone confidence | `pose_craft/retarget/skeleton.py` |
| Pose file format (write from the app, read by the add-on) | `pose_craft/export/blender_pose.py` |
| Live Blender link client (app side) | `pose_craft/export/live_link.py` |
| Blender-side importer + Live Link server (install manually inside Blender, not via pip) | `pose_craft/export/blender_addon/import_pose.py` |
| Unicode-safe image loading | `pose_craft/imaging.py` |
| Clickable/resizable photo widget | `pose_craft/gui/photo_view.py` |
| PyInstaller build spec | `PoseCraft.spec` |
| App icon generation (re-run only if the design changes) | `scripts/generate_icon.py` |

## Release

```text
dist/PoseCraft/PoseCraft.exe        -- now built with assets/icon.ico as the exe icon
dist/PoseCraft/_internal/           -- PySide6, OpenCV, MediaPipe native libs, ~480 MB total
```

Built with `pyinstaller PoseCraft.spec` (onedir, `--windowed`, `collect_all`
for `mediapipe`/`cv2`/`pyqtgraph`, plus `assets/icon.png`+`assets/icon.ico`
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
still no automated *release*-build workflow analogous to
`photo_professional_tool`'s `.github/workflows/release.yml` (which builds and
uploads the `.exe` itself) -- see "Known Issues".

## Verification

```bash
pytest
```

covers `retarget/skeleton.py`, `export/blender_pose.py`,
`estimation/landmarker.py` (atomic download logic, with `urlretrieve` mocked),
`imaging.py` (real cv2 encode/decode round trips, including a non-ASCII
filename), and `export/live_link.py` (a real loopback TCP server standing in
for the Blender add-on, checking the exact wire format both sides need to
agree on) -- all pure numpy/scipy/stdlib/cv2, no PySide6/mediapipe/Blender
needed to run them. `tests/test_main_window_helpers.py` additionally covers
`main_window._nearest_person_index()` (the click-to-pick math) but guards its
PySide6 import with `pytest.importorskip`, so it runs locally and skips
cleanly in CI's lighter venv. Confirmed locally: installing only
`requirements-test.txt` into a fresh venv reproduces CI's exact result (20
passed, 1 skipped).

Everything else has been checked manually rather than by automated test,
since it needs a real display/GPU/Blender install that isn't available in
this dev environment:

- The GUI's imports and `MainWindow` construction were smoke-tested (`QApplication` +
  `MainWindow()` + `processEvents()`, no crash, `isVisible()` true) after every
  round of GUI changes in this project -- this proves the widget tree builds,
  not that solving/exporting/clicking a real photo works end to end.
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
  `closeEvent`. `detect_poses()` (the old one-shot function) still exists as a
  thin convenience wrapper around a throwaway `PoseEstimator` for callers that
  really do only solve once.
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

## Known Issues

Ranked roughly by how much they affect whether an exported pose is usable:

- **Twist is not solved.** Only the direction a bone points (swing) is derived
  from landmark positions; rotation around a bone's own long axis (forearm
  pronation, wrist twist, ankle roll) is left at identity in
  `retarget/skeleton.py::solve_bone_rotations()`. Expect to hand-correct
  forearms/hands/feet in Blender -- the same category of fix PoseForge's own
  "Before you buy" section warns about for DAZ. The new per-bone
  `low_confidence` flag (see "Recently Fixed") helps surface *which* bones are
  most likely to need this, but a confident-looking bone can still have wrong
  twist -- confidence is about landmark visibility, not twist accuracy.
- **MediaPipe's world-landmark axis convention is a best-effort guess, not a
  verified fact.** Isolated in `_mediapipe_world_to_engine()` in
  `estimation/landmarker.py` specifically so it's the one place to fix if an
  exported pose comes out upside-down or mirrored against the source photo --
  this has not been checked against real MediaPipe output in this dev
  environment (no working display was used to run the live solve loop
  end-to-end against a real photo).
- **The Blender add-on -- both bone-name matching and the new Live Link
  server -- is unverified against a real Blender install.**
  `NAME_CANDIDATES_TEMPLATES` in `import_pose.py` guesses at common Rigify
  naming patterns (`DEF-`, `ORG-`, `_fk`, `_ik`) but was never loaded into
  Blender or run against an actual Rigify-generated armature; the Live Link
  server's `bpy.app.timers`-based non-blocking accept loop is standard for
  Blender add-ons but likewise only checked by code review, not a real run.
  Check the add-on's unmatched-bone warning after your first real import and
  extend the template list if your rig's names don't match.
- **No code signing** on the built `.exe` -- expect SmartScreen to flag it as
  unrecognized on a machine other than the one it was built on.
- **No automated release-build workflow** -- `.github/workflows/ci.yml` only
  runs tests, unlike `photo_professional_tool`'s `.github/workflows/release.yml`
  which builds and uploads the `.exe` itself. Rebuild locally with
  `pyinstaller PoseCraft.spec` for now.

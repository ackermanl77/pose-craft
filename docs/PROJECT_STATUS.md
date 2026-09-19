# Project Status

## Architecture

```text
Photo (loaded via imaging.imread_unicode, unicode-path-safe;
       imaging.assess_photo_quality flags low resolution before solving)
        |
        v
PoseEstimator (pose_craft/estimation/landmarker.py)
  -- wraps one cached MediaPipe PoseLandmarker instance, multi-person via num_poses
        |  landmarks converted to the engine frame (Y-up, +X right, +Z toward camera);
        |  each DetectedPerson also carries 2D image-space landmarks + per-landmark visibility
        v
retarget/skeleton.py -- 15-bone humanoid hierarchy, T-pose rest directions
  -- per-bone shortest-arc swing quaternion, world -> parent-relative
  -- plus solved twist for hand.L/hand.R (from the index/pinky lateral spread) --
     every other bone's twist stays at identity, no independent reference exists
  -- bone_confidence() derives a per-bone score from landmark visibility
        v
export/blender_pose.py -- axis-converts Y-up -> Blender Z-up, writes JSON
  -- low-confidence bones flagged "low_confidence": true
        |
        +---------------------------------------------+
        v  (file)                                      v  (live)
pose_craft_pose .json file            export/live_link.py: length-prefixed
        |                              JSON over TCP to host:port (UI-configurable,
        |                              default 127.0.0.1:8642)
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
`MainWindow._estimator` and reused for every later solve in the same session,
rather than rebuilding MediaPipe's graph every click. Because that estimator
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
`max_people_spin` control (1-8, default 4) -- since MediaPipe bakes `num_poses`
into the landmarker graph at creation time, changing it closes and discards
the cached `PoseEstimator` so the next solve rebuilds one with the new count,
instead of silently keeping the old limit. `gui/photo_view.py`'s
`PhotoView` keeps the full-resolution `QPixmap` so it can
rescale cleanly on window resize, and turns a click into a normalized
image-space point that `main_window.py` matches against each detected
person's 2D centroid (`_nearest_person_index`) to drive the person picker.
`gui/viewport_3d.py` is a thin wrapper around `pyqtgraph.opengl.GLViewWidget`
that draws `retarget.skeleton.joint_positions()` as line segments + points and
auto-frames the camera to the skeleton's own bounding box on every update
(`_compute_camera_frame`) -- it has no other state of its own beyond the
current skeleton, `main_window.py` always calls `clear_skeleton()` before
`show_skeleton()` on every update rather than diffing.

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
| Unicode-safe image loading + pre-solve photo quality checks | `pose_craft/imaging.py` |
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
`estimation/landmarker.py` (atomic download logic and the progress-callback
contract, with `urlretrieve` mocked), `imaging.py` (real cv2 encode/decode
round trips, including a non-ASCII filename), and `export/live_link.py` (a
real loopback TCP server standing in for the Blender add-on, checking the
exact wire format both sides need to agree on) -- all pure numpy/scipy/stdlib/cv2,
no PySide6/mediapipe/Blender needed to run them. `tests/test_main_window_helpers.py`
additionally covers `main_window._nearest_person_index()` (the click-to-pick
math) and the mirror/confidence consistency fix (see "Recently Fixed"), and
`tests/test_viewport_camera.py` covers the auto-framing math
(`_compute_camera_frame`); both guard their PySide6 import with
`pytest.importorskip`, so they run locally and skip cleanly in CI's lighter
venv. Confirmed locally: installing only `requirements-test.txt` into a fresh
venv reproduces CI's exact result (28 passed, 2 skipped).

Everything else has been checked manually rather than by automated test,
since it needs a real display/GPU/Blender install that isn't available in
this dev environment:

- The GUI's imports and `MainWindow` construction were smoke-tested (`QApplication` +
  `MainWindow()` + `processEvents()`, no crash, `isVisible()` true) after every
  round of GUI changes in this project -- this proves the widget tree builds,
  not that solving/exporting/clicking a real photo works end to end.
- The threading fixes (button-disabling during a solve, `closeEvent` waiting
  for in-flight workers, `SendToBlenderWorker` not blocking the UI thread)
  were each checked with a scripted integration test using a real Qt event
  loop, a fake slow `PoseEstimator`/`send_pose_document`, and timed
  assertions -- not part of the `pytest` suite, since they depend on real
  background-thread timing rather than being pure functions. These are one-off
  scripts run during development, not something that re-runs on every change;
  a regression here wouldn't be caught automatically.
- The `max_people` mismatch handling (closing and discarding a cached
  `PoseEstimator` whose `num_poses` no longer matches the spin box) was
  checked the same scripted-integration way: a fake estimator confirms
  `close()` is actually called and the new `SolveWorker` receives
  `estimator=None` to force a rebuild. That particular run happened to also
  exercise a real `PoseEstimator`/MediaPipe detection end to end (the pose
  model was already cached from earlier manual testing in this session) with
  the custom `max_people=6`, with no crash -- a real, if incidental, live
  smoke test of the whole detection path, not just the mocked one.
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
  segment has in the 33-point set. **Fixed**: `retarget/skeleton.py` now
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
  time, changing the value closes and discards the cached `PoseEstimator` so
  the next solve rebuilds one with the new count instead of keeping the old
  limit silently.
- **Live Link's host/port were hardcoded.** No way to work around
  `127.0.0.1:8642` being unavailable. **Fixed**: added editable host/port
  fields in the UI, defaulting to `export.live_link.DEFAULT_HOST`/`DEFAULT_PORT`;
  `SendToBlenderWorker` now takes them as constructor arguments instead of
  always using the module defaults.
- **The 3D viewport's camera was fixed** (`distance=3.0`, a fixed center),
  so an unusually large/small person or an extreme pose (crouching, arms flung
  wide) could render partially out of view. **Fixed**: `viewport_3d.py` now
  computes the skeleton's actual bounding sphere on every `show_skeleton()`
  call and reframes distance + center to it (`_compute_camera_frame`), while
  leaving elevation/azimuth alone so a manual camera rotation isn't reset on
  every solve. Verified with new tests in `test_viewport_camera.py`.

## Known Issues

Ranked roughly by how much they affect whether an exported pose is usable:

- **Twist is only solved for the hands.** Rotation around a bone's own long
  axis needs a second landmark spanning that axis to measure -- MediaPipe's
  index/pinky knuckles provide that for `hand.L`/`hand.R` (see "Recently
  Fixed"), but forearm pronation, upper-arm rotation, thigh rotation, and
  ankle roll all stay at identity in `retarget/skeleton.py::solve_bone_rotations()`,
  same as before. Expect to hand-correct those in Blender -- the same category
  of fix PoseForge's own "Before you buy" section warns about for DAZ. The
  per-bone `low_confidence` flag doesn't help here either way -- it's about
  landmark visibility, not about whether twist could be solved at all.
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

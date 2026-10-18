# PoseCraft

A photo-to-pose desktop tool, built as an MVP inspired by [PoseForge](https://poseforge.gumroad.com/l/poseforge)
(a commercial photo -> DAZ Studio pose tool). PoseCraft targets **Blender**
instead of DAZ Studio: load a photo, click **Solve Pose**, click a person in
the photo (or pick one from the dropdown) to preview their estimated 3D
skeleton, then either export a pose file or send it straight to Blender over
the Live Link.

## Current Status

- MVP stage: the full pipeline (photo or video -> pose estimation
  (MediaPipe's BlazePose models, plus the optional hand model, on ONNX
  Runtime -- DirectML or CUDA on a GPU, else the CPU) -> bone retargeting ->
  JSON/BVH export or the live Blender link -> Blender add-on 0.8.0, or a DAZ
  Studio pose preset for Genesis 9) is
  implemented, and all three build flavours build and launch. The add-on has
  been run headless in a real Blender 5.2.2 (PoseCraft-shaped, Mixamo-style
  and generated Rigify rigs, the BVH importer, a real Live Link server); its
  panel UI, and the accuracy of a solve on real photos and footage, have not
  been checked end to end -- see `docs/PROJECT_STATUS.md`'s "Verification"
  and "Known Issues".
- `docs/PROJECT_STATUS.md` is the authoritative reference for architecture,
  entry points, and known issues -- this README only covers day-to-day
  commands.
- New to the app itself? [docs/USER_GUIDE.md](docs/USER_GUIDE.md) walks
  through every screen and control, feature by feature, for both the desktop
  app and the Blender add-on.

## Install

The pins are known to work on Python 3.14 (the dev venv and the release
build) and 3.12 (CI's test job). If `pip install` can't find a wheel for your
interpreter, use one of those (`py -0` on Windows lists what's installed):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs `onnxruntime-gpu`, which also runs on the CPU, so
this alone works without a GPU.

Two GPU options, each in its own venv (both install the same `onnxruntime`
module, so they can't be combined):

- `requirements-dml.txt` -- DirectML, any DirectX 12 GPU (NVIDIA, AMD, Intel).
  Measured on an RTX 5080: ~4-8 ms per solve, within 0.02 px of the CPU, and
  a much smaller build. See the file for the install steps.
- `requirements-gpu.txt` -- CUDA 13 + cuDNN as pip packages, so an NVIDIA GPU
  (RTX 20-series or newer) only needs a current driver. ~9-39 ms per solve.
  TF32 math is switched off, so results match the CPU (with it on, landmarks
  moved up to 12 px).

Without either everything still works, on the CPU. The **Use GPU** checkbox
under Solve Pose toggles it, and the status bar says which device each solve
actually ran on ("GPU (DirectML)", "GPU (CUDA)" or "CPU").

The GPU is only used when it is a real, working GPU. DirectML runs on the
hardware DirectX 12 GPU with the most dedicated memory (the discrete card on a
laptop with integrated graphics too) and never on a software renderer such as
Microsoft Basic Render Driver. When the model loads, each GPU model is also run
once on a probe image and compared with the CPU; if a GPU fails, returns
invalid values or differs by more than 1% of an output's range, PoseCraft uses
the CPU instead and the status bar says why ("GPU not used: ...", kept in its
tooltip). **Help > Diagnostics** lists the ONNX Runtime version and providers, the
DirectX 12 GPUs, the loaded model's device, the fallback reason and the log file,
with a Copy button for bug reports.

`requirements.txt` is pinned to known-working versions; if you deliberately
want the latest of everything instead, drop the `==` pins.

## Run

```bash
python -m pose_craft.app
```

Load a photo (PNG/JPEG/BMP/WebP/TIFF -- or drag it onto the window; a banner
warns if its resolution is likely to give a worse result -- see the tips under
the Load Photo button), optionally raise "Max people to detect" above its
default of 4 and tick "Group photo" for a wide group shot (if someone in a hard
photo isn't found, lower **Advanced > Detection threshold**), then click **Solve
Pose** (F5). A source install downloads MediaPipe's pinned pose model to
`~/.cache/pose_craft/models` on first run (SHA-256-checked, then converted to
ONNX next to it -- needs network access once, with a progress bar and
Cancel); a release build ships the models. Once the model is on disk the app
loads it in the background at startup, so the first solve is quick. Every
detected person's skeleton is drawn directly on the photo -- the selected
person in a bright color (low-confidence bones, likely occluded in the photo,
dashed in a warning color), everyone else dimmed. Click the person you want
in the photo, pick them from the dropdown, or use Page Up/Down. A misplaced
joint can be dragged on the photo (its limb is re-solved), a limb that came
out reaching the wrong way in depth can be flipped from the photo's
right-click menu, and any bone can be rotated by hand in "3. Correct" (all
with undo). Tick **Hands and fingers** before solving to also estimate the
fingers (15 bones per hand, Rigify names; off by default -- the first use
downloads MediaPipe's pinned hand model). **Load Video...** tracks up to "Max
people" people through a clip, each with a stable number (re-identified after
a short loss by where they were and what they look like), and exports an
animated BVH per person, optionally with **Rotation smoothing** (zero-phase,
on by default at 35%; 0% exports the frames as solved) -- or sends the
selected person's animation straight to Blender. Hover the photo for the
bone, joint or person under the mouse (highlighted, named in a tooltip). In the 3D preview, hover a bone for its name, local
rotation and confidence; **View > Photo Backdrop** puts the photo behind the
skeleton. From there:

- **Export DAZ Pose (Genesis 9)...** -- a DAZ Studio `.duf` pose preset for
  Genesis 9, computed from the figure installed in your DAZ content library
  (see the user guide), or
- **Export Pose JSON** / **Export BVH** / **Export All People...** /
  **Export All People (BVH)...** (one file per detected person, to a folder
  you choose, named after the photo, asking before overwriting; the
  mirror checkbox, if used, applies to everyone in a batch export the same way
  it applies to a single person; the BVH stands on the floor). The BVH save
  dialogs (pose and video animation) also offer **glTF binary (.glb)**: a skinned
  skeleton with a small bone mesh and the animation, for Unity, Unreal, Godot,
  three.js and Blender; the CLI and batch take `--format glb`. Or
- **File > Batch Process Folder...** -- every photo in a folder, solved
  with the current detection settings, one `<photo>_person<N>.json` (or
  `.bvh`, `.duf`) per person, with a progress bar and Cancel. Without the GUI:
  `python -m pose_craft.cli --input photos --output poses [--format bvh|duf]
  [--mirror] [--model lite|full|heavy] [--cpu] [--hands] [--group-photo]
  [--detection-threshold 0.3]` (from a source install), or
- **Send to Blender (Live Link)** -- needs Blender running with the add-on's
  Live Link started first (see below). The host/port fields next to the
  button default to `127.0.0.1:8642` and only need changing if that port is
  already in use. **Auto-send on change** (off by default, remembered)
  re-sends the shown pose a short debounce after each change and pauses itself
  if Blender can't be reached. For a video, **Send Animation to Blender** keys
  the selected person's whole track on the rig, resampled to the scene's frame
  rate and moving with the person's steps (add-on 0.8.0; long clips go in
  chunks under the add-on's 256 KB message cap).

Inside Blender, install the add-on zip (the release's `posecraft-blender-addon-<version>.zip`,
or build it with `python scripts/build_addon_zip.py`): on Blender 4.2+ it is an
extension -- Edit > Preferences > Get Extensions > Install from Disk, which shows its
declared network permission (Live Link listens on localhost); on 3.6-4.1 use Edit >
Preferences > Add-ons > Install... with the same zip. Then select an armature and
open the "PoseCraft" tab in the 3D viewport sidebar. From there either **Load
Pose JSON** (pick an exported file) or **Start Live Link** once and then use
PoseCraft's "Send to Blender" button for every pose after that, no file
dialog needed. Bones the exporter flagged low-confidence (likely occluded in
the photo) are called out separately from unmatched bone names in both paths.

If your rig's bone names don't match PoseCraft's automatic guessing (a bone
shows up as "unmatched"), use the panel's **Bone Mapping** box: click
"Initialize/Refresh" for one row per PoseCraft bone name, then either "Auto-Guess
Unset" to fill in what it can (it knows Rigify's `upper_arm_fk.L`-style controls
and Mixamo's `mixamorig:` names), or pick each target bone yourself from the
searchable dropdown; a row's "Extra Bones" field chains more bones (Mixamo's
and Rigify's spines are chained automatically). Saved on the armature, so it's a one-time
setup per rig. For Rigify (and other constraint-driven rigs), "Pose rig
controls" poses the controls that drive the DEF-/ORG- bones, switches the
posed limbs to FK, and solves each bone against its parent as Blender
evaluated it. The panel also has "Keep feet on the floor", "Insert keyframes"
with a frame step, and the Live Link target (captured when Live Link starts).

The app's menus list every keyboard shortcut; the user guide has the table.

## Build the .exe

```bat
:: pre-convert the models into build/models (bundled, no first-run download)
python scripts/convert_models.py
:: dml is the default when unset (needs requirements-dml.txt); or cuda / cpu
set POSECRAFT_FLAVOR=dml
pyinstaller PoseCraft.spec
```

To Authenticode-sign the result: `set CODESIGN_THUMBPRINT=<SHA-1 thumbprint>` (a
code-signing certificate in the Windows store), then `python scripts/sign_build.py`
(signtool from the Windows SDK, timestamped by `CODESIGN_TIMESTAMP_URL`, default DigiCert).

A per-user installer (no admin rights; `%LOCALAPPDATA%\Programs\PoseCraft`, Start-menu
shortcut, optional desktop shortcut, uninstaller that offers to delete your settings and logs):
`python scripts/build_installer.py [--flavor dml|cpu|cuda] [--sign]` with Inno Setup 6
installed writes `dist/installer/PoseCraft-<version>-<flavor>-setup.exe`; `--sign` signs the
exe, the installer and the uninstaller with the `CODESIGN_THUMBPRINT` certificate.

Produces `dist/PoseCraft/PoseCraft.exe` plus its `_internal/` dependency
folder -- measured on 2026-10-01 (Python 3.14, PyInstaller 6.22.3): DirectML
508 MB, CPU 472 MB, CUDA 1.73 GB -- ship the whole `dist/PoseCraft/` folder, not just
the `.exe`. With `build/models` present the converter (onnx/tflite) is left
out of the build. `.github/workflows/release.yml` does all of this for the
DirectML and CPU flavours on a `v*` tag (or a manual run), zips them and builds
their installers, adds the Blender add-on and writes `SHA256SUMS` -- it hasn't been run yet (waiting
for GitHub's artifact storage quota to reset). See `docs/PROJECT_STATUS.md`'s "Release" section for why onedir
was chosen over `--onefile`. Re-run `python scripts/generate_icon.py` first if
you've changed `assets/icon.png`/`assets/icon.ico`.

## Test

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.venv\Scripts\python.exe -m pytest -q
```

Last run 2026-10-07: **428 passed, 8 skipped** (the skips: the LiteRT
reference test, the real-photo golden test, four opt-in GPU parity tests, the
opt-in real-Blender test and the opt-in real-Blender extension install) -- re-run it rather than trusting this number.

Needs `requirements-test.txt` (numpy/scipy/opencv-python-headless/pytest plus
CPU onnxruntime/onnx/tflite, not the Qt stack or the GPU runtime) -- covers the
bone retargeting math, the Blender axis-conversion math, the atomic
model-download logic, the pose pipeline's pre/post-processing and the model
converter, unicode-safe image loading, the Live Link wire protocol, and a
real-photo end-to-end golden test on five public-domain/CC0 photos in
`tests/data/golden` (it needs the cached "full" model; a second test checks
the up/left/facing axes from the committed goldens without it). `.github/workflows/ci.yml` runs this
same lightweight suite on every push/PR on Ubuntu, the full suite (Qt included,
with coverage) on Windows, and the linters: `python -m ruff check pose_craft tests
scripts` and `python -m mypy` (configured in `pyproject.toml`; mypy covers
`pose_craft/retarget` and `pose_craft/export` so far). With PySide6 installed, run it with
`QT_QPA_PLATFORM=offscreen` to include the GUI flow tests (real Qt event loop,
fake estimators; no pytest-qt needed). Before a release, also run the GPU/CPU
parity tests: `set POSECRAFT_GPU_TESTS=1` then `pytest -m gpu` (needs a GPU
provider and the cached models) and, with Blender installed, the real-Blender
add-on check: set `POSECRAFT_BLENDER` to `blender.exe`, then
`pytest tests/test_real_blender.py` (about 5 s). See `docs/PROJECT_STATUS.md`'s
"Verification" section for what is and isn't covered by an automated test
today.

The 3D preview draws the exported rotations themselves (with the raw
landmarks as faint dots), so what you see is what Blender gets. The app logs
to `%LOCALAPPDATA%\PoseCraft\logs\posecraft.log`.

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for architecture, and its
"Known Issues" section before relying on an exported pose -- a straight limb's
roll is inferred rather than measured, the torso is one bone, the
world-landmark axis convention hasn't been checked end to end against a real
photo in Blender, finger poses and multi-person video tracking haven't been
checked on real footage, and the Blender add-on has been checked headless in
Blender 5.2.2 only (a PoseCraft-shaped, a Mixamo-style and a generated Rigify
rig, the BVH importer and a real Live Link server -- not its panel UI).

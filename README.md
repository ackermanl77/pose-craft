# PoseCraft

A photo-to-pose desktop tool, built as an MVP inspired by [PoseForge](https://poseforge.gumroad.com/l/poseforge)
(a commercial photo -> DAZ Studio pose tool). PoseCraft targets **Blender**
instead of DAZ Studio: load a photo, click **Solve Pose**, click a person in
the photo (or pick one from the dropdown) to preview their estimated 3D
skeleton, then either export a pose file or send it straight to Blender over
the Live Link.

## Current Status

- MVP stage: the full pipeline (photo -> pose estimation (MediaPipe's BlazePose
  models on ONNX Runtime, GPU-accelerated on NVIDIA cards) -> bone
  retargeting -> JSON export or live Blender link -> Blender add-on) is
  implemented and the `.exe` builds and launches, but the Blender add-on and
  the real-photo accuracy of the retargeting have not been verified end to
  end -- see `docs/PROJECT_STATUS.md`'s "Verification" and "Known Issues".
- `docs/PROJECT_STATUS.md` is the authoritative reference for architecture,
  entry points, and known issues -- this README only covers day-to-day
  commands.
- New to the app itself? [docs/USER_GUIDE.md](docs/USER_GUIDE.md) walks
  through every screen and control, feature by feature, for both the desktop
  app and the Blender add-on.

## Install

PySide6 and ONNX Runtime ship wheels that can lag behind the newest CPython
release; if `pip install` can't find a match for your interpreter, use a
3.10-3.12 venv instead (`py -0` on Windows lists what's installed):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # optional: NVIDIA GPU acceleration (CUDA)
```

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
the CPU instead and the status bar says why ("GPU not used: ...").

`requirements.txt` is pinned to known-working versions; if you deliberately
want the latest of everything instead, drop the `==` pins.

## Run

```bash
python -m pose_craft.app
```

Load a photo (PNG/JPEG/BMP/WebP/TIFF -- or drag it onto the window; a banner
warns if its resolution is likely to give a worse result -- see the tips under
the Load Photo button), optionally raise "Max people to detect" above its
default of 4 and tick "Group photo" for a wide group shot, then click **Solve
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
people" people through a clip, each with a stable number, and exports an
animated BVH per person. In the 3D preview, hover a bone for its name, local
rotation and confidence; **View > Photo Backdrop** puts the photo behind the
skeleton. From there:

- **Export Pose JSON** / **Export BVH** / **Export All People...** (one JSON
  per detected person, to a folder you choose, named after the photo; the
  mirror checkbox, if used, applies to everyone in a batch export the same way
  it applies to a single person; the BVH stands on the floor), or
- **Send to Blender (Live Link)** -- needs Blender running with the add-on's
  Live Link started first (see below). The host/port fields next to the
  button default to `127.0.0.1:8642` and only need changing if that port is
  already in use.

Inside Blender, install `pose_craft/export/blender_addon/import_pose.py` as an
add-on (Edit > Preferences > Add-ons > Install...), select an armature, and
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

```bash
python scripts/convert_models.py           # pre-convert the models into build/models (bundled, no first-run download)
set POSECRAFT_FLAVOR=dml   # the default when unset (needs requirements-dml.txt); or cuda / cpu
pyinstaller PoseCraft.spec
```

Produces `dist/PoseCraft/PoseCraft.exe` plus its `_internal/` dependency
folder -- measured on 2026-10-01 (Python 3.14, PyInstaller 6.22.3): DirectML
508 MB, CPU 472 MB, CUDA 1.73 GB -- ship the whole `dist/PoseCraft/` folder, not just
the `.exe`. With `build/models` present the converter (onnx/tflite) is left
out of the build. `.github/workflows/release.yml` does all of this for the
DirectML and CPU flavours on a `v*` tag (or a manual run), zips them with the
Blender add-on and writes `SHA256SUMS` -- it hasn't been run yet. See `docs/PROJECT_STATUS.md`'s "Release" section for why onedir
was chosen over `--onefile`. Re-run `python scripts/generate_icon.py` first if
you've changed `assets/icon.png`/`assets/icon.ico`.

## Test

```bash
pytest
```

Needs `requirements-test.txt` (numpy/scipy/opencv-python-headless/pytest plus
CPU onnxruntime/onnx/tflite, not the Qt stack or the GPU runtime) -- covers the
bone retargeting math, the Blender axis-conversion math, the atomic
model-download logic, the pose pipeline's pre/post-processing and the model
converter, unicode-safe image loading, the Live Link wire protocol, and (when golden
photos and the cached model are present, see `scripts/generate_goldens.py`)
a real-photo end-to-end golden test. `.github/workflows/ci.yml` runs this
same lightweight suite on every push/PR. With PySide6 installed, run it with
`QT_QPA_PLATFORM=offscreen` to include the GUI flow tests (real Qt event loop,
fake estimators; no pytest-qt needed). Before a release, also run the GPU/CPU
parity tests: `set POSECRAFT_GPU_TESTS=1` then `pytest -m gpu` (needs a GPU
provider and the cached models). See `docs/PROJECT_STATUS.md`'s
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
checked on real footage, and the Blender add-on (including the rig-aware
retargeting, Rigify support and the Live Link server) hasn't been run in a
real Blender install.

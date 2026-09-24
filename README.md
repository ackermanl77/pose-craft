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
pip install -r requirements-gpu.txt   # optional: NVIDIA GPU acceleration
```

`requirements-gpu.txt` adds the CUDA 13 runtime and cuDNN as pip packages, so
an NVIDIA GPU (RTX 20-series or newer, including RTX 40/50) only needs a
current driver -- no CUDA Toolkit install. Without it everything still works,
on the CPU. The **Use GPU** checkbox under Solve Pose toggles it, and the
status bar says which device each solve actually ran on.

`requirements.txt` is pinned to known-working versions; if you deliberately
want the latest of everything instead, drop the `==` pins.

## Run

```bash
python -m pose_craft.app
```

Load a photo (a resolution/framing warning pops up if it's likely to give a
worse result -- see the tips under the Load Photo button), optionally raise
"Max people to detect" above its default of 4 for a bigger group photo, then
click **Solve Pose** (downloads MediaPipe's pose model to
`~/.cache/pose_craft/models` on first run and converts it to ONNX next to it -- needs network access once, cached
after that, with a live download percentage in the status bar). Every
detected person's skeleton is drawn directly on the photo -- the selected
person in a bright color (their low-confidence joints, likely occluded in
the photo, shown in a distinct warning color), everyone else dimmed. Either
click the person you want directly in the photo or pick them from the
dropdown. From there:

- **Export Pose JSON** / **Export All People...** (one JSON per detected
  person, to a folder you choose; the mirror checkbox, if used, applies to
  everyone in a batch export the same way it applies to a single person), or
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
Unset" to fill in what it can, or pick each target bone yourself from the
searchable dropdown. Saved on the armature, so it's a one-time setup per rig.

## Build the .exe

```bash
pyinstaller PoseCraft.spec
```

Produces `dist/PoseCraft/PoseCraft.exe` plus its `_internal/` dependency
folder (~480 MB total) -- ship the whole `dist/PoseCraft/` folder, not just
the `.exe`. See `docs/PROJECT_STATUS.md`'s "Release" section for why onedir
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
converter, unicode-safe image loading, and the Live Link wire protocol. `.github/workflows/ci.yml` runs this
same lightweight suite on every push/PR. See `docs/PROJECT_STATUS.md`'s
"Verification" section for what is and isn't covered by an automated test
today.

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for architecture, and its
"Known Issues" section before relying on an exported pose -- twist is only
solved for the hands (everything else stays at identity), the world-landmark axis
convention is an unverified best guess, and the Blender add-on (including the
new Live Link server) hasn't been checked against a real Blender install or rig.

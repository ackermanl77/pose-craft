# PoseCraft

A photo-to-pose desktop tool, built as an MVP inspired by [PoseForge](https://poseforge.gumroad.com/l/poseforge)
(a commercial photo -> DAZ Studio pose tool). PoseCraft targets **Blender**
instead of DAZ Studio: load a photo, click **Solve Pose**, click a person in
the photo (or pick one from the dropdown) to preview their estimated 3D
skeleton, then either export a pose file or send it straight to Blender over
the Live Link.

## Current Status

- MVP stage: the full pipeline (photo -> MediaPipe pose estimation -> bone
  retargeting -> JSON export or live Blender link -> Blender add-on) is
  implemented and the `.exe` builds and launches, but the Blender add-on and
  the real-photo accuracy of the retargeting have not been verified end to
  end -- see `docs/PROJECT_STATUS.md`'s "Verification" and "Known Issues".
- `docs/PROJECT_STATUS.md` is the authoritative reference for architecture,
  entry points, and known issues -- this README only covers day-to-day
  commands.

## Install

MediaPipe and PySide6 ship wheels that can lag behind the newest CPython
release; if `pip install` can't find a match for your interpreter, use a
3.10-3.12 venv instead (`py -0` on Windows lists what's installed):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is pinned to known-working versions; if you deliberately
want the latest of everything instead, drop the `==` pins.

## Run

```bash
python -m pose_craft.app
```

Load a photo, click **Solve Pose** (downloads the MediaPipe model to
`~/.cache/pose_craft/models` on first run -- needs network access once, cached
after that), then either click the person you want directly in the photo or
pick them from the dropdown. From there:

- **Export Pose JSON** / **Export All People...** (one JSON per detected
  person, to a folder you choose), or
- **Send to Blender (Live Link)** -- needs Blender running with the add-on's
  Live Link started first (see below).

Inside Blender, install `pose_craft/export/blender_addon/import_pose.py` as an
add-on (Edit > Preferences > Add-ons > Install...), select an armature, and
open the "PoseCraft" tab in the 3D viewport sidebar. From there either **Load
Pose JSON** (pick an exported file) or **Start Live Link** once and then use
PoseCraft's "Send to Blender" button for every pose after that, no file
dialog needed. Bones the exporter flagged low-confidence (likely occluded in
the photo) are called out separately from unmatched bone names in both paths.

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

Needs `requirements-test.txt` (numpy/scipy/opencv-python-headless/pytest, not
the full mediapipe/Qt stack) -- covers the bone retargeting math, the Blender
axis-conversion math, the atomic model-download logic, unicode-safe image
loading, and the Live Link wire protocol. `.github/workflows/ci.yml` runs this
same lightweight suite on every push/PR. See `docs/PROJECT_STATUS.md`'s
"Verification" section for what is and isn't covered by an automated test
today.

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for architecture, and its
"Known Issues" section before relying on an exported pose -- twist isn't
solved, the MediaPipe axis convention is an unverified best guess, and the
Blender add-on (including the new Live Link server) hasn't been checked
against a real Blender install or rig.

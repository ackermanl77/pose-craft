# PoseCraft

A photo-to-pose desktop tool, built as an MVP inspired by [PoseForge](https://poseforge.gumroad.com/l/poseforge)
(a commercial photo -> DAZ Studio pose tool). PoseCraft targets **Blender** instead
of DAZ Studio: load a photo, click **Solve Pose**, preview the estimated 3D
skeleton, and export a pose file that a companion Blender add-on applies to an
armature's pose bones.

## Pipeline

```
photo --[MediaPipe Pose Landmarker]--> 33 3D landmarks per person
      --[retarget/skeleton.py]--------> per-bone local rotations (swing only)
      --[export/blender_pose.py]------> pose_craft_pose .json (Blender Z-up)
      --[Blender add-on]--------------> applied to an armature's pose bones
```

- `pose_craft/estimation/landmarker.py` -- wraps MediaPipe's `PoseLandmarker` task
  (multi-person capable via `num_poses`), downloads the model weights to
  `~/.cache/pose_craft/models` on first use, and converts MediaPipe's world
  landmarks into this project's engine frame (Y-up, +X right, +Z toward camera).
- `pose_craft/retarget/skeleton.py` -- a small humanoid bone hierarchy (spine,
  neck, head, arms, legs) with a T-pose rest direction per bone. Each bone's
  rotation is the shortest-arc quaternion from its rest direction to its current
  (landmark-derived) direction, made parent-relative. This is the one module
  with unit tests that don't require MediaPipe/Qt/Blender (`tests/test_retarget.py`).
- `pose_craft/export/blender_pose.py` -- converts those rotations into Blender's
  Z-up axis convention and writes/reads the JSON pose format.
- `pose_craft/export/blender_addon/import_pose.py` -- a Blender add-on (install
  it inside Blender itself, not with pip) that reads a PoseCraft JSON file and
  applies it to the selected armature's pose bones, with a fallback for common
  Rigify bone-name variants (`DEF-`, `ORG-`, `_fk`, `_ik`).
- `pose_craft/gui/` -- the PySide6 desktop app: load a photo, run detection on a
  background thread, pick a person when several are detected, preview the
  skeleton in an OpenGL viewport (`pyqtgraph.opengl`), and export.

## Setup

MediaPipe and PySide6 wheels lag behind the newest CPython releases, so use a
3.10-3.12 interpreter for the venv (this machine's default `python` may be
newer -- check with `py -0` on Windows and pick an older one if needed):

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pose_craft.app
```

Run the test suite (only needs numpy/scipy/pytest, not mediapipe/Qt):

```bash
pytest
```

## Known limitations (read before relying on exported poses)

- **Twist is not solved.** Only the direction a bone points (swing) is derived
  from landmark positions; rotation around a bone's own long axis (forearm
  pronation, wrist twist, etc.) is left at identity. Expect to hand-correct
  forearms/hands/feet in Blender, the same category of fix PoseForge's own
  "Before you buy" section warns about for DAZ.
- **MediaPipe world-landmark axis convention is a best-effort guess**, isolated
  in `_mediapipe_world_to_engine` in `landmarker.py`. If an exported pose comes
  out upside-down or mirrored against a real photo, that's the one function to
  adjust -- flip the sign there rather than anywhere else in the pipeline.
- **Bone-name matching in the Blender add-on is heuristic.** It was written
  against common Rigify naming patterns but not validated against a real
  Rigify-generated rig (no Blender install in this dev environment). Check the
  add-on's unmatched-bone warning after the first import on your own rig and
  extend `NAME_CANDIDATES_TEMPLATES` if needed.
- **Multi-person picking** uses MediaPipe's `num_poses` and sorts by horizontal
  position in-frame; there's no click-to-select-in-photo UI yet (PoseForge lets
  you click the person in the image itself).
- No license/activation system, no "Open in Blender" live-apply (export is
  file-based: run PoseCraft, export JSON, switch to Blender, import JSON).

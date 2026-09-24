# PoseCraft User Guide

PoseCraft turns a single photo into a 3D pose you can apply to a character in
Blender. It's a smaller, Blender-focused take on the same idea as
[PoseForge](https://poseforge.gumroad.com/l/poseforge) (which targets DAZ
Studio): load a photo, click **Solve Pose**, pick the person you want, then
either export a pose file or send it straight into a running Blender session.

This guide walks through every screen and every control. For install/build
commands aimed at developers, see the main [README](../README.md); for
architecture, test coverage, and known limitations in full technical detail,
see [PROJECT_STATUS.md](PROJECT_STATUS.md). This guide is the practical,
feature-by-feature version of both.

## Contents

- [Getting the app running](#getting-the-app-running)
- [The main window](#the-main-window)
- [1. Load & Detect](#1-load--detect)
- [2. Select Person](#2-select-person)
- [3. Export](#3-export)
- [4. Blender Live Link](#4-blender-live-link)
- [The 3D preview](#the-3d-preview)
- [Setting up the Blender add-on](#setting-up-the-blender-add-on)
- [The PoseCraft panel in Blender](#the-posecraft-panel-in-blender)
- [Confidence and unmatched bones, explained](#confidence-and-unmatched-bones-explained)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Getting the app running

**Packaged app**: if you have `dist/PoseCraft/PoseCraft.exe`, double-click it
-- no Python install needed. Windows SmartScreen may flag it as unrecognized
the first time (it isn't code-signed); choose "More info" > "Run anyway" if
you trust the source.

**From source**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pose_craft.app
```

Either way, the first time you click **Solve Pose** the app downloads a pose
estimation model (a few MB) to `~/.cache/pose_craft/models` -- this needs
network access once, then it's cached for every run after that.

## The main window

The window has two halves:

- **Left**: a control panel, split into four numbered sections you use in
  order -- *1. Load & Detect*, *2. Select Person*, *3. Export*, *4. Blender
  Live Link* -- with the loaded photo shown above them.
- **Right**: the **3D preview**, a live 3D skeleton view of whichever person
  is currently selected.

A status bar along the bottom reports what's happening -- model download
progress, solve results, export/send confirmations, and errors.

## 1. Load & Detect

**Load Photo...** opens a file picker (PNG, JPG, JPEG, BMP, WEBP). Once a
photo loads:

- The tips list under the button (*"For best results..."*) is always visible
  as a reminder: full body in frame, facing the camera or a slight angle,
  even bright lighting, limbs not hidden behind the body or other people.
  These aren't checked automatically -- there's no reliable way to detect "is
  this person facing sideways" from pixels alone -- so treat them as
  guidance, not a photo requirement.
- If the photo's resolution is below 480px on its shorter side, a popup warns
  you that pose estimation is likely to be less accurate. You can still
  proceed; it's a quality hint, not a hard block.

**Max people to detect** (1-8, default 4) caps how many people the pose model
looks for in one photo. A person beyond this count in a group photo won't be
detected at all -- raise it before solving a bigger group shot. Changing it
takes effect on the next Solve Pose, with no model reload needed.

**Use GPU (NVIDIA CUDA) when available** (on by default) runs the pose model on
an NVIDIA graphics card, which makes each solve roughly twice as fast. If the
computer has no suitable NVIDIA GPU (or the GPU libraries aren't installed),
the app quietly uses the CPU instead -- the status bar after each solve says
which one it used ("Solved 2 pose(s) on GPU"). Changing this checkbox or
**Model quality** reloads the model on the next solve (a brief extra delay).

**Solve Pose** runs the actual detection. On the very first solve of a
session, this also downloads the pose model, and the status bar shows a live
download percentage (the downloaded model is also converted once to the
format the app runs it in -- well under a second). Every solve after that reuses the already-loaded model,
so later clicks are noticeably faster. While a solve is running, **Load
Photo**, **Max people to detect**, **Model quality** and **Use GPU** are disabled -- this is intentional:
swapping the photo or the detection settings mid-solve could let a stale
result land on the wrong photo.

If nobody is detected, a popup says so and you're back to square one with the
same photo (try the framing tips above, or a different photo).

## 2. Select Person

Once a solve finds one or more people, each shows up as "Person 1", "Person
2", etc. in the **Person** dropdown, ordered left-to-right in the photo. Pick
one either from the dropdown, or by clicking that person directly in the
photo -- PoseCraft matches your click to whichever detected person's center
is closest.

The photo itself shows every detected person's 2D skeleton, drawn directly on
top of the image:

- The **selected** person's skeleton is bright cyan with orange joints, drawn
  on top of everyone else.
- Any bone on the selected person that the pose model wasn't confident about
  (likely occluded or hard to see in the photo) is drawn in **orange-red**
  instead of cyan, so you can see at a glance which parts of the pose are
  shakiest before you ever open Blender.
- Everyone **not** currently selected is dimmed gray underneath, so it's
  obvious who you're about to export.

**Mirror left/right (body orientation fix)** swaps which side of the body
feeds which exported bone -- use it if a solved pose comes out with left and
right reversed. It's a property of the *photo/camera orientation*, not of one
person, so when checked it applies consistently to every person in a batch
export too, not just whoever's currently selected. It does **not** change the
skeleton drawn on the photo (checking it doesn't move anyone's arm in the
picture -- it only relabels which side's data goes into the exported bone
names), so don't expect the overlay to visibly change when you toggle it.

## 3. Export

Both buttons only need Load Photo + Solve Pose to have already found at least
one person; the JSON they write includes the solved rotation for every bone,
plus a confidence flag per bone (see
[Confidence and unmatched bones](#confidence-and-unmatched-bones-explained)).

- **Export Pose JSON...** saves just the currently selected person as one
  `.json` file, wherever you choose.
- **Export All People...** saves every detected person as a separate file
  (`pose_person1.json`, `pose_person2.json`, ...) into a folder you choose --
  useful for a group photo where you want every person's pose. Respects the
  Mirror checkbox for everyone, the same as single-person export.

## 4. Blender Live Link

An alternative to file export: send a solved pose straight into a running
Blender session over a local network connection, so you don't need a file
dialog for every attempt.

- **Blender host / port** default to `127.0.0.1` / `8642`. Leave these alone
  unless that port is already in use by something else on your machine --
  and if you do change the port, you must set the *same* port in Blender's
  own PoseCraft panel before starting Live Link there, or the send will fail.
  Changing the host field doesn't make Blender listen anywhere new; see
  [Setting up the Blender add-on](#setting-up-the-blender-add-on).
- **Send to Blender (Live Link)** sends the currently selected person's
  solved pose. This needs Blender open, the add-on installed, and its **Start
  Live Link** button already clicked -- otherwise you'll get a popup
  explaining that and pointing you at the Blender side.

Sending runs in the background, so the app doesn't freeze while it connects
-- useful because the most common failure (Blender's Live Link isn't started
yet) would otherwise mean a multi-second freeze on every attempt.

## The 3D preview

The right-hand panel shows the solved skeleton in 3D, independent of the
photo overlay. It automatically frames itself to the selected person's actual
size and pose on every solve or selection change, so an unusually large,
small, or extreme pose still fits in view. You can still orbit/pan/zoom it
manually with the mouse (standard 3D-viewport drag-to-orbit,
scroll-to-zoom) -- doing so doesn't get reset by a later solve, only the
camera's distance and center re-fit automatically.

## Setting up the Blender add-on

1. In Blender: **Edit > Preferences > Add-ons > Install...**, pick
   `pose_craft/export/blender_addon/import_pose.py`, and enable "PoseCraft
   Pose Importer".
2. A **PoseCraft** tab appears in the 3D viewport's sidebar (press `N` if the
   sidebar isn't open).
3. Select the armature you want to pose, then open that PoseCraft tab.

## The PoseCraft panel in Blender

**Load Pose JSON** applies a `.json` file you exported from the app to the
currently selected armature. Use this for one-off poses.

**Bone Mapping** bridges the gap between PoseCraft's 15 bone names (spine,
neck, head, and left/right upper arm, forearm, hand, thigh, shin, foot) and
whatever your rig actually calls its bones:

1. Click **Initialize/Refresh** once per armature -- this creates one row per
   PoseCraft bone name.
2. Click **Auto-Guess Unset** to try filling blank rows automatically (it
   tries the exact name, then common Rigify-style variants like `DEF-`,
   `ORG-`, `_fk`, `_ik`).
3. For anything still blank, or wrong, use each row's searchable dropdown to
   pick the correct bone on your armature by hand.
4. **Clear All** blanks every row's assignment if you want to start over
   (keeps the row list itself).

This mapping is saved on the armature object, so it's saved with your `.blend`
file and only needs doing once per rig. A row you've explicitly set always
wins over the automatic guess -- even reporting that bone as "unmatched"
rather than silently guessing something else, if the bone you picked has
since been renamed or deleted.

**Live Link**: set the **port** to match the PoseCraft app's (default 8642),
then click **Start Live Link**. The status line underneath shows whether it's
listening, and reports the result of the most recently received pose. Click
**Stop Live Link** to close it. This always listens on `127.0.0.1`
(localhost) only, by design -- there's no setting to expose it to the network,
since that would mean accepting arbitrary bone rotations from anyone able to
reach that port with no authentication at all.

## Confidence and unmatched bones, explained

Every exported pose carries two kinds of "this might need a second look"
information:

- **Low confidence**: MediaPipe (the pose-detection model) scores how
  visible each part of the body was in the photo. A bone built from a
  poorly-visible body part (an arm behind the back, a foot out of frame) gets
  flagged `low_confidence` in the exported JSON, shown as orange-red in the
  app's photo overlay, and called out by name when applied in Blender (both
  via **Load Pose JSON** and **Live Link**).
- **Unmatched**: a PoseCraft bone name that couldn't be resolved to any bone
  on your armature, neither by your manual Bone Mapping nor by the automatic
  guess. Reported by name in Blender; fix it by adding a row for it in Bone
  Mapping.

Neither of these blocks anything from being applied -- they're informational,
so you know which parts of a pose are worth checking by hand.

## Known limitations

Worth knowing before you rely on an exported pose:

- **Only hands solve real "twist"** (rotation around a limb's own length,
  like forearm rotation or wrist twist) -- everywhere else, twist is left
  untouched, because a single photo doesn't give enough information to
  recover it. Expect to hand-correct forearms, upper arms, thighs, and
  ankles for anything beyond the basic pose direction.
- **A solved pose might come out mirrored or upside-down** for certain
  photos -- if that happens, try the **Mirror left/right** checkbox first.
- **The Blender add-on** (bone-name matching, the Bone Mapping UI, and Live
  Link) hasn't been checked against every possible rig -- if something looks
  off, the Bone Mapping panel is exactly the tool for fixing bone-name
  mismatches by hand.
- **DAZ Studio isn't a supported export target** -- PoseCraft only exports to
  Blender.
- The packaged `.exe` isn't code-signed, so expect a Windows SmartScreen
  warning on a machine other than the one it was built on.

## Troubleshooting

| Symptom | Try this |
|---|---|
| "No person was detected in this photo" | Check the framing tips under Load Photo -- full body in frame, decent lighting, not too small a photo. |
| Pose looks mirrored or flipped | Toggle **Mirror left/right** and re-export/re-send. |
| A limb looks twisted the wrong way in Blender | Expected for most limbs (see Known Limitations) -- correct it by hand in Blender; hands are the one part PoseCraft solves twist for. |
| "Unmatched" bone reported in Blender | Open the PoseCraft panel's **Bone Mapping** box and assign that bone manually. |
| "Could not reach Blender's Live Link" | In Blender, open the PoseCraft tab and click **Start Live Link** first; double-check the port matches on both sides. |
| Solve Pose seems stuck on first use | It's downloading the pose model -- check the status bar for a percentage, and that you have network access. |
| A 5th+ person in a group photo isn't detected | Raise **Max people to detect** before clicking Solve Pose. |

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
- [3. Correct](#3-correct)
- [Video](#video)
- [4. Export](#4-export)
- [5. Blender Live Link](#5-blender-live-link)
- [The 3D preview](#the-3d-preview)
- [The pose library](#the-pose-library)
- [Menus and keyboard shortcuts](#menus-and-keyboard-shortcuts)
- [Setting up the Blender add-on](#setting-up-the-blender-add-on)
- [The PoseCraft panel in Blender](#the-posecraft-panel-in-blender)
- [Confidence and unmatched bones, explained](#confidence-and-unmatched-bones-explained)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Getting the app running

**Packaged app**: if you have `dist/PoseCraft/PoseCraft.exe`, double-click it
-- no Python install needed. Windows SmartScreen may flag it as unrecognized
the first time (it isn't code-signed); choose "More info" > "Run anyway" if
you trust the source. The main build is the **DirectML** one (it uses any
DirectX 12 graphics card -- NVIDIA, AMD or Intel -- and falls back to the
CPU); there is also a CPU-only build and a much larger NVIDIA-only CUDA
build. All give the same results.

**From source**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pose_craft.app
```

A packaged release build already contains the pose models, so it needs no
network at all. Run from source, the first time you click **Solve Pose** the
app downloads a pose estimation model (a few MB) to
`~/.cache/pose_craft/models` -- this needs network access once, then it's
cached for every run after that. The download is checked against a pinned
checksum; a corrupted or altered file is rejected and nothing is saved.

Once a model is on disk, the app loads it in the background as soon as it
starts ("Preparing the pose model in the background..."), so even the first
Solve answers quickly -- on a GPU, loading can otherwise take several seconds.

## The main window

The window has three side-by-side panes; drag the dividers between them to
resize:

- **Left**: the control panel, split into numbered sections you use in
  order -- *1. Load & Detect*, *2. Select Person*, *3. Correct*, *4. Export*,
  *5. Blender Live Link* (plus a *Video* section once you load a video). It
  scrolls if the window is short, so the whole app fits on a laptop screen.
- **Middle**: the loaded photo, with every detected person's skeleton drawn
  on it.
- **Right**: the **3D preview** of whichever person is currently selected.
- **Toolbar** (under the menus): Open Photo, Open Video, **Solve**, Export
  Pose JSON, Export BVH and Send to Blender, always in view however far the
  left panel is scrolled.

A status bar along the bottom reports what's happening -- solve results,
export/send confirmations, and errors -- with a progress bar and a **Cancel**
button while something runs in the background. The window remembers its size,
the pane widths, the last folder you used, your recent files and your
settings (model, GPU, host/port, ...) between runs.

You can also drag a photo or video file from Explorer onto the window to open
it, or reopen one from **File > Open Recent**.

## 1. Load & Detect

**Load Photo...** opens a file picker (PNG, JPG, JPEG, BMP, WEBP, TIF, TIFF).
**Load Video...** tracks a person through a clip instead -- see [Video](#video).
Once a photo loads:

- The **Photo tips** toggle under the button (also the button's tooltip)
  shows the "For best results" list: full body in frame, facing the camera or a slight angle,
  even bright lighting, limbs not hidden behind the body or other people.
  These aren't checked automatically -- there's no reliable way to detect "is
  this person facing sideways" from pixels alone -- so treat them as
  guidance, not a photo requirement.
- If the photo's resolution is below 480px on its shorter side, a yellow
  banner above the photo warns that pose estimation is likely to be less
  accurate (click its ✕ to dismiss it). You can still proceed; it's a quality
  hint, not a hard block.

**Max people to detect** (1-8, default 4) caps how many people the pose model
looks for in one photo. A person beyond this count in a group photo won't be
detected at all -- raise it before solving a bigger group shot. Changing it
takes effect on the next Solve Pose, with no model reload needed.

**Model quality** (*lite*, *full* or *heavy*; default *full*, remembered)
picks the pose model: *lite* is the fastest and suits low-end machines,
*heavy* is the most accurate but slowest. A source install downloads each
one the first time you solve with it.

**Use GPU when available** (on by default) runs the pose model on the graphics
card: through DirectML (any DirectX 12 GPU, NVIDIA, AMD or Intel) in a
DirectML build, or CUDA (NVIDIA only) in a CUDA build. Either gives the same
result as the CPU to within a fraction of a pixel, several times faster. The
GPU is used only if the app finds a real graphics card (not Windows' software
renderer) and a quick check when the model loads gives the same results as the
CPU. Otherwise it uses the CPU, and the status bar says why ("Ready -- pose
model loaded on CPU (GPU not used: ...)"). After each solve the status bar says
which one it used ("Solved 2 pose(s) on GPU (DirectML)"). Changing this checkbox or
**Model quality** reloads the model on the next solve (a brief extra delay).

**Group photo (slower)** also searches overlapping quarters of the photo for
people. The detector looks at the whole photo shrunk to a small square, so in
a wide group shot each face can be only a few pixels across and get missed;
the quarters see them bigger. Even with it off, the quarters are tried
automatically when the whole photo finds nobody.

**Hands and fingers** (off by default, remembered) also estimates each
person's fingers: for every hand whose wrist and knuckles are visible, the app
crops the hand from the body pose and runs MediaPipe's hand model on it (a few
milliseconds per hand). Each hand found gets 15 finger bones -- three each for
the thumb and four fingers, named as in Rigify (`thumb.01.L`, `f_index.01.L`
... `f_pinky.03.R`) -- drawn thinner in the photo and the 3D preview and
included in every export (JSON, BVH, Live Link). The hand bone itself is then
measured from the palm, which is more accurate. The status bar says how many
hands were found. The first solve with it on downloads the hand model (about
7.5 MB) in a source install; a release build ships it. With it off, exports
are the same 15-bone body pose as before. Photos only: video stays body-only.

**Solve Pose** (or **F5**) runs the actual detection. On the very first solve, a source install also
downloads the pose model, and the progress bar shows how far along it is
(the downloaded model is also converted once to the
format the app runs it in -- well under a second). **Cancel** (or **Esc**)
stops a download. Every solve after that reuses the already-loaded model,
so later clicks are noticeably faster. While a solve is running, **Load
Photo**, **Load Video**, **Max people to detect**, **Model quality**, **Use GPU**, **Group photo** and **Hands and fingers** are disabled -- this is intentional:
swapping the photo or the detection settings mid-solve could let a stale
result land on the wrong photo.

Very large photos (over 2048 pixels on the long side, typical of modern
cameras) are shrunk to 2048 pixels before detection. That's both faster and
more accurate: the model only ever looks at a small version of the photo, and
shrinking it smoothly first removes the noise and jagged edges that the
model's own sampling would otherwise pick up.

If nobody is detected, a popup says so and you're back to square one with the
same photo (try the framing tips above, or a different photo).

## 2. Select Person

Once a solve finds one or more people, each shows up in the **Person**
dropdown, ordered left-to-right in the photo, with how many of their bones
were seen confidently ("Person 2: 13/15 bones confident"). Pick one from the
dropdown, with **Page Down / Page Up**, or by clicking that person's skeleton
in the photo -- the click goes to whichever drawn skeleton passes nearest it.
A click on empty background (or with the right mouse button) doesn't change
the selection.

The photo itself shows every detected person's 2D skeleton, drawn directly on
top of the image; the legend under the photo shows the same line styles:

- The **selected** person's skeleton is bright cyan with white joints, drawn
  on top of everyone else.
- Any bone on the selected person that the pose model wasn't confident about
  (likely occluded or hard to see in the photo) is drawn **dashed** and in
  orange-red, so you can see at a glance which parts of the pose are
  shakiest before you ever open Blender -- the dashes make it readable
  without relying on colour.
- Everyone **not** currently selected is dimmed gray underneath, so it's
  obvious who you're about to export.
- **Hover** the photo: the bone or joint under the mouse is highlighted in
  yellow (a hovered unselected person turns light blue), and a tooltip names
  it ("Person 2: forearm.L").

**Mirror left/right** (**M**; a body orientation fix) swaps which side of the body
feeds which exported bone and mirrors the pose (a left arm raised becomes a
right arm raised) -- use it if a solved pose comes out with left and
right reversed. It's a property of the *photo/camera orientation*, not of one
person, so when checked it applies consistently to every person in a batch
export too, not just whoever's currently selected. It does **not** change the
skeleton drawn on the photo (checking it doesn't move anyone's arm in the
picture -- it only relabels which side's data goes into the exported bone
names), so don't expect the overlay to visibly change when you toggle it.

**Body orientation fix** turns the whole solved pose if it comes out
upside-down or facing 90 degrees off: *Flip Upside-Down*, *Rotate 90° Left*,
*Rotate 90° Right* (or *Default*, no change). Like Mirror, it applies to the
3D preview and to every export at once.

## 3. Correct

A single photo can't always tell which way a limb points in depth: a forearm
reaching toward the camera and one reaching away by the same angle look the
same, and a joint can simply be detected in the wrong place. Three fixes, all
undoable with **Ctrl+Z** (redo **Ctrl+Y** or **Ctrl+Shift+Z**):

- **Drag a joint**: the selected person's shoulders, elbows, wrists, hips,
  knees and ankles are drawn with yellow rings on the photo. Drag one to where
  the joint really is; when you let go, its limb is re-solved and the 3D
  preview updates. The bone keeps its length and stays on the same side in
  depth (toward or away from the camera) -- only its direction changes; if you
  drag farther than the bone reaches, it lies flat in the photo's plane,
  pointing at where you dropped it. A wrist carries its hand along, an ankle
  its foot. Each drag is one undo step ("Move right elbow"). Photos only.
- **Flip a limb's depth**: right-click an arm or leg bone in the photo and
  choose **Flip depth: forearm.L** (or whichever bone you clicked). That bone
  and everything below it is mirrored front-to-back about its joint -- its
  position in the photo stays the same. Do it again to undo it.
- **Adjust bone**: pick a bone and rotate it about its own axes in degrees,
  like Blender's bone axes: **Y** runs along the bone (a twist), **X** and
  **Z** bend it. Its children follow. **Reset bone** clears one bone's
  adjustment, **Reset all edits** everything for this person.

Corrections belong to one person and go into every export (JSON, BVH, Live
Link, and every person of Export All). In a video they belong to the tracked
person and apply to every frame. A new solve or a new photo clears them.

## Video

**Load Video...** (MP4, MOV, AVI, MKV, WEBM, M4V) tracks up to **Max people
to detect** people through the whole clip, with a progress bar and
**Cancel**. Each person keeps the same number for the whole video ("Person
2" stays Person 2), numbered left to right among the people in the first
frame. Every tracked person is followed from where their pose was in the
previous frame (fast: one model run per person per frame, about 10 ms on a
CPU); the detector runs on the first frame, whenever someone is lost, and
every 10 frames while fewer than the maximum are tracked, to pick up
newcomers. Someone who disappears for up to 30 frames (walking behind
something) gets their number back when they reappear near where they left
and look the same (a small colour histogram of each person's box is compared
too, so two people who cross behind something are less likely to come back
with swapped numbers); after longer they get a new number. With Max people set to 1 only the
clearest person is tracked, as before. Each person's landmarks are smoothed
over time on their own (a One Euro filter: strong smoothing while the body
is still, little lag when it moves fast).

The **Video** section then shows a frame slider: scrub to any frame to see
it, its skeletons and the selected person's 3D preview; the selected person
stays selected while you scrub (as long as they're in the frame). Every
export, correction and Live Link send works on the frame you're looking at.
**Export Animation BVH...** writes the selected person's tracked frames as one
animated BVH (frames where they weren't found are skipped): each frame stands
on the floor, and while a foot is planted the root moves so that foot stays
put, which is what moves a walk forward. **Export All People (BVH)...** writes
one such file per tracked person (`clip_person1.bvh`, `clip_person2.bvh`, ...)
into a folder, asking before it overwrites anything.

**Rotation smoothing** (0-100%, default 35%, remembered) additionally smooths
the solved joint rotations of each tracked person (quaternion slerp, run
forwards and backwards so the motion doesn't lag the video; across frames
where the person was lost it catches up instead of blending the two sides).
The preview, both BVH exports, Live Link and Send Animation all use it; 0%
keeps every frame exactly as solved.

**Send Animation to Blender** sends the selected person's whole track over
Live Link: Blender keys the person's track on the target rig, starting at
Blender's current frame, resampled to the scene's frame rate -- every scene
frame gets one key, the pose at that moment of the clip, blended from the
frames around it (a 30 fps clip in a 24 fps scene keeps its length and no
frame is keyed twice). Frames where the person was lost get no key. With the
add-on's **Root motion (animations)** option (on by default) the rig also
travels with the person -- planted feet drive it, scaled to the rig's leg
length; **Keep feet on the floor** still sets the height. Each frame goes through the same bone mapping,
rest-pose matching and Rigify handling as a single pose, and the whole
animation is one undo step in Blender. Long clips are sent in several parts
(at most 30 frames and about 200 KB each); the status bar says how many
frames were keyed and on which Blender frames. Needs add-on 0.7.0 or newer --
an older one replies "update the add-on"; resampling and root motion need 0.8.0.

## 4. Export

All three buttons only need Load Photo + Solve Pose to have already found at
least one person. The JSON includes the solved rotation for every bone, plus a
confidence flag per bone (see
[Confidence and unmatched bones](#confidence-and-unmatched-bones-explained))
and how high the hips sit above the floor in this pose.
If a file can't be written (a read-only folder, a file open elsewhere), a
popup says which file and why. File names default to the photo's name plus
the person, like `IMG_0042_person2.json`.

- **Export Pose JSON...** (**Ctrl+S**) saves just the currently selected person as one
  `.json` file, wherever you choose.
- **Export All People...** saves every detected person as a separate file
  (`IMG_0042_person1.json`, `IMG_0042_person2.json`, ...) into a folder you
  choose -- useful for a group photo where you want every person's pose. It
  asks before overwriting files that are already there. Respects the
  Mirror checkbox for everyone, the same as single-person export.
- **Export DAZ Pose (Genesis 9)...** saves the selected person as a DAZ Studio
  pose preset (`.duf`) for **Genesis 9**: in DAZ Studio, select a Genesis 9
  figure and load the file (double-click it in the Content Library, or File >
  Merge). It sets every bone, like DAZ's own presets, and lowers the hip for a
  crouch. PoseCraft reads the figure's joints from your DAZ content library
  (`data/DAZ 3D/Genesis 9/Base/Genesis9.dsf`, found in DAZ Studio's default
  library folders, or set `POSECRAFT_DAZ_LIBRARY`), so Genesis 9 must be
  installed. Genesis 9's A-pose arms are handled; DAZ's joint limits still
  apply, and the status bar names any bone the pose pushes past them (DAZ
  stops it at the limit). Genesis 8 isn't supported yet. Batch: `--format duf`.
- **Export BVH...** (**Ctrl+Shift+S**) saves the selected person as a single-frame `.bvh` file
  (Y up, facing +Z), for tools that read BVH rather than PoseCraft's JSON. The
  skeleton stands on the floor (Y = 0), so a crouch comes out lowered.
- **Export All People (BVH)...** does the same for every detected person
  (`IMG_0042_person1.bvh`, ...), into a folder you choose, asking before
  overwriting.

With **Hands and fingers** on, a person whose hands were found also has 15
finger bones per hand in the JSON (same format, same keys per bone) and the
BVH (finger joints under each hand joint). Without them, both files are
exactly the 15-bone body pose.

### Skeletons and FBX

The **Skeleton** list in the Export group chooses the bone names, hierarchy and rest pose of
BVH, glTF and FBX files: PoseCraft's own (Rigify names), **Mixamo** (`mixamorig:Hips`,
`Spine`/`Spine1`/`Spine2`, shoulders), the **Unreal Engine 5 Mannequin** (`root`, `pelvis`,
`spine_01`-`05`, clavicles, a 45° A-pose rest) or **HumanIK** (MotionBuilder, Maya). The pose is
the same on every skeleton -- only the rig it is written on changes -- so Unity and Unreal can
import it without going through Blender. The choice is remembered, and the batch (and the
command line's `--skeleton`) uses it too.

When Blender is installed, the save dialogs also offer **FBX through Blender**: PoseCraft writes
the glTF and a hidden Blender converts it with its own FBX exporter (a few seconds). Blender is
found under Program Files, or set `POSECRAFT_BLENDER` to its `blender.exe`.

### Importing an animation from another tool

**File > Import Animation (BVH)** opens a BVH from ComfyUI's SAM 3D Body, a mocap library,
Mixamo, Rokoko or PoseCraft itself as a tracked person: scrub it, fix it, smooth it, export it on
another skeleton or send it to Blender like a tracked video. Any skeleton works as long as its
hands, feet and head are named (hand/wrist, foot/ankle, head, with a left/right side); the
file's own up axis and rest pose are worked out from the hips and head. Root motion is
re-estimated from the feet, as for a video.

## 5. Blender Live Link

An alternative to file export: send a solved pose straight into a running
Blender session over a local network connection, so you don't need a file
dialog for every attempt.

- **Blender host / port** default to `127.0.0.1` / `8642`. Leave these alone
  unless that port is already in use by something else on your machine --
  and if you do change the port, you must set the *same* port in Blender's
  own PoseCraft panel before starting Live Link there, or the send will fail.
  Changing the host field doesn't make Blender listen anywhere new; see
  [Setting up the Blender add-on](#setting-up-the-blender-add-on).
- **Send to Blender (Live Link)** (**Ctrl+B**) sends the currently selected person's
  solved pose. This needs Blender open, the add-on installed, and its **Start
  Live Link** button already clicked -- otherwise you'll get a popup
  explaining that and pointing you at the Blender side.

- **Auto-send on change** (off by default, remembered with its **Debounce**,
  default 350 ms) re-sends the shown pose once changes settle -- a new
  selection, video frame, correction, mirror or orientation change -- so a
  burst of changes (scrubbing, a spin box held down) becomes one send, and
  only one send is ever in flight. Nothing is sent without a selected person.
  If Blender can't be reached, auto-send pauses itself (the status bar says
  so, no popup); a manual **Send to Blender**, or ticking the box again,
  resumes it. Closing the app cancels a pending send.
- **Send Animation to Blender** (in the Video section) -- see [Video](#video).

Sending runs in the background, so the app doesn't freeze while it connects
-- useful because the most common failure (Blender's Live Link isn't started
yet) would otherwise mean a multi-second freeze on every attempt.

Blender replies with what it did, and the status bar shows it: for example
"Applied 13/15 bones to 'Armature' (unmatched: foot.L, foot.R)". If Blender
received the pose but couldn't apply it (no target armature, say), a popup
says why. With an add-on older than 0.4.0, the status bar instead says the
pose was sent but not confirmed -- update the add-on to get the reply.

## The 3D preview

The right-hand panel shows the pose **as it will be exported**: the skeleton
is built from the exported bone rotations themselves, the same way Blender or
a BVH reader will build it, sized to the person's own proportions. The raw
detected body points are drawn as faint gray dots behind it. Where the two
agree, the export reproduces the photo; where the skeleton leaves the dots,
that's what the export will get wrong, so check it before exporting. It is
independent of the photo overlay.

The buttons in its top-right corner change the view -- **Front**, **Side**,
**Top**, **3/4** (the default angle), **Fit** (frame the skeleton) and zoom
**−** / **+** -- the same views as the View menu and its keys (1, 3, 7, 0, F).
The small gizmo in the bottom-left corner shows which way the X (red), Y
(green) and Z (blue, up) axes point from the current angle.

- The skeleton **stands on the grid**: the grid is the floor, at the pose's
  lowest point.
- Bones are drawn as **tapered octahedra** (like Blender's), which turn with
  each bone's roll, with a line through each; a low-confidence bone's line is
  **dashed**.
- A small **XYZ gizmo** at the floor's centre: red X is the subject's left,
  green Y is behind them, blue Z is up.
- **View** menu: **Front** (**1**), **Side** (**3**), **Top** (**7**),
  **Reset Camera** (**0**), **Frame Skeleton** (**F**) and **Bone Labels**
  (**L**, each bone's name at its middle).
- **Hover a bone** (with no mouse button held) for a small label with its
  name, its rotation relative to its parent as X/Y/Z degrees about its own
  axes (Y along the bone -- the same axes as **Adjust bone** and Blender's
  bone axes), and its confidence. **View > Hover Info** (**H**) turns it off.
- **View > Photo Backdrop** (**Shift+B**, remembered) shows the photo (or the
  current video frame) as a plane behind the skeleton, scaled and placed so
  the person in it lines up with the faint raw-landmark dots -- a quick way to
  compare the 3D result with the picture from the front view. It follows the
  mirror checkbox and the body orientation fix.
- **View > View from Photo Camera** (**5**) turns the backdrop on and looks at
  it straight on from far away with a narrow field of view -- the way the pose
  was fitted to the photo -- so the skeleton sits exactly over the person in
  the picture. Any other view key (1/3/7/0) goes back to the normal lens.
- **View > Backdrop opacity** (a slider in the View menu, remembered) fades
  the photo from barely visible to opaque.

It automatically frames itself to the selected person's actual
size and pose on every solve or selection change, so an unusually large,
small, or extreme pose still fits in view. You can still orbit/pan/zoom it
manually with the mouse (standard 3D-viewport drag-to-orbit,
scroll-to-zoom) -- doing so doesn't get reset by a later solve, only the
camera's distance and center re-fit automatically.

## The pose library

Poses you like can be kept and reused without the photo:

- **Pose > Add to Pose Library...** (**Ctrl+Shift+L**) saves the selected
  person as you see them -- your corrections, the Mirror checkbox and the body
  orientation fix included -- with a thumbnail of the photo and a name.
- **Pose > Pose Library...** (**Ctrl+Shift+P**) shows the saved poses as
  thumbnails. Type in the search box (names, tags and photo names) or pick a
  tag to filter. Select a pose to rename it or give it tags (comma separated,
  then **Save Name & Tags**).
- **Export...** writes the pose as Pose JSON, BVH, glTF (`.glb`) or a DAZ
  Studio Genesis 9 preset (`.duf`) -- exactly what exporting from the photo
  gave. **Send to Blender** sends it over Live Link like the current pose.
- **Mirror Copy** adds the left/right mirrored version as a new pose.
- **Blend Two...** (select two poses with Ctrl+click) mixes them: the slider
  goes from all of the first pose to all of the second, bone by bone. A blend
  keeps the tags both poses share. Blends can be exported, sent and blended
  again, but not mirrored -- mirror the poses before blending them.
- **Delete** removes the selected poses after asking.

The library is a folder per pose under `%LOCALAPPDATA%\PoseCraft\pose_library`
(set `POSECRAFT_LIBRARY_DIR` to keep it elsewhere, e.g. in a synced folder).

**Send Library to Blender** (in the pose library) makes a Blender **Pose Library** asset from
each selected pose -- or all of them -- on the Live Link target armature, with the pose's
thumbnail as its preview and its tags as asset tags; blend and flip them from Blender's own Pose
Library. In Blender, **Import PoseCraft Library as Pose Assets** in the PoseCraft panel does the
same from a saved `posecraft_library.json`. Needs add-on 0.9.0.

## Menus and keyboard shortcuts

| Key | Does |
|---|---|
| Ctrl+O / Ctrl+Shift+O | Open a photo / a video |
| F5 or Ctrl+Enter | Solve Pose |
| Esc | Cancel a running solve or video |
| Page Down / Page Up | Next / previous person |
| M | Mirror left/right |
| Ctrl+Z / Ctrl+Y | Undo / redo a correction |
| Ctrl+S / Ctrl+Shift+S | Export Pose JSON / Export BVH |
| Ctrl+B | Send to Blender |
| 1 / 3 / 7 / 0 | Front / side / top / default 3D view |
| F / L | Frame the skeleton / toggle bone labels |
| H / Shift+B | Toggle the 3D hover info / the photo backdrop |
| 5 | View from the photo's camera (backdrop and skeleton overlaid) |
| Ctrl+Shift+L / Ctrl+Shift+P | Add the pose to the pose library / open the library |

Most control labels have an underlined letter: Alt plus that letter jumps to
it. The control with keyboard focus gets a yellow outline, and every control
has an accessible name for screen readers (not yet checked with a real
screen reader). **File > Open Recent** lists the last eight photos and videos.

## Setting up the Blender add-on

1. In Blender: **Edit > Preferences > Add-ons > Install...** (in Blender
   4.2 and later: the **v** menu at the top right > **Install from Disk...**),
   pick `pose_craft/export/blender_addon/import_pose.py` (or the
   `posecraft-blender-addon-<version>.zip` from a release), and enable
   "PoseCraft Pose Importer". The current add-on is version 0.9.0 (Blender
   3.6 or newer; checked in Blender 5.2.2). After updating PoseCraft, install
   the matching add-on too -- **Send Animation to Blender** needs 0.7.0 or
   newer, **Send Library to Blender** 0.9.0.
2. A **PoseCraft** tab appears in the 3D viewport's sidebar (press `N` if the
   sidebar isn't open).
3. Select the armature you want to pose, then open that PoseCraft tab.

## The PoseCraft panel in Blender

**Load Pose JSON** applies a `.json` file you exported from the app to the
currently selected armature. Use this for one-off poses.

**Match rig rest pose** (on by default) aims each bone of your rig at the
direction PoseCraft solved, measured against your rig's *own* rest pose and
bone hierarchy. That makes A-pose rigs, sloped foot bones, and rigs with extra
bones (a Mixamo "Hips" bone above the legs, clavicles between the spine and
the arms) come out right. Bones PoseCraft has no counterpart for keep their
current pose. Turn it off to apply PoseCraft's rotations exactly as exported
(this assumes a T-pose rig shaped like PoseCraft's skeleton, and is also what
happens with files from PoseCraft versions before this option existed).

**Pose rig controls** (on by default) is for rigs driven by constraints,
like a generated Rigify rig. Such a rig deforms with `DEF-` bones that copy
`ORG-` bones, which copy the FK or IK controls -- posing a `DEF-`/`ORG-` bone
directly does nothing. With this on, a mapped bone whose rotation comes from
a constraint is replaced by the bone that actually drives it (Rigify's FK
control, e.g. `upper_arm_fk.L` for `DEF-upper_arm.L`), the arms and legs being
posed are switched to FK (Rigify's **IK-FK** slider set to 1 on
`upper_arm_parent.L` / `thigh_parent.L`), and each bone is aimed against its
parent's pose as Blender actually evaluated it, constraints included -- so the
helper (`MCH-`) bones Rigify puts between the torso and the limbs don't throw
the limbs off. Rigify's FK spine (`spine_fk` and `.001`-`.003`) is chained
automatically. Switch your limbs back to IK yourself afterwards if you want
to keep animating with IK.

**Keep feet on the floor** (off by default): after posing, the rig's root bone
is moved up or down so the pose's lowest point sits where the rig's rest pose
had its feet -- a crouch or a sit comes out lowered instead of hanging at
standing height. It moves the top bone of the rig's hierarchy (Mixamo's Hips,
Rigify's root).

**Insert keyframes** keys every applied bone at the current frame (it also
happens whenever Blender's own auto-keying is on), and **Then advance** steps
the current frame forward by that many frames after each keyed pose -- send a
series of photos to block out an animation. Each applied bone keeps its own
rotation mode (Euler, axis-angle or quaternion), so Euler rigs and their
existing keys aren't disturbed.

**Bone Mapping** bridges the gap between PoseCraft's 15 bone names (spine,
neck, head, and left/right upper arm, forearm, hand, thigh, shin, foot), plus
its 30 finger bone names, and whatever your rig actually calls its bones:

1. Click **Initialize/Refresh** once per armature -- this creates one row per
   PoseCraft bone name.
2. Click **Auto-Guess Unset** to try filling blank rows automatically (it
   tries the exact name, then Rigify controls like `upper_arm_fk.L`, then
   `DEF-`/`ORG-` bones, then Mixamo names like `mixamorig:LeftArm`;
   finger bones match a Rigify rig's finger controls by name, and Mixamo's
   `LeftHandIndex1`-style bones).
3. For anything still blank, or wrong, use each row's searchable dropdown to
   pick the correct bone on your armature by hand.
   The text field next to it, **Extra Bones**, takes more bones continuing a
   chain, comma-separated (for example `Spine1, Spine2` after `Spine`): the
   rotation is spread evenly over the whole chain. Mixamo's three spine bones
   and Rigify's four FK spine controls are chained like this automatically.
4. **Clear All** blanks every row's assignment if you want to start over
   (keeps the row list itself).

This mapping is saved on the armature object, so it's saved with your `.blend`
file and only needs doing once per rig. A row you've explicitly set always
wins over the automatic guess -- even reporting that bone as "unmatched"
rather than silently guessing something else, if the bone you picked has
since been renamed or deleted.

**Live Link**: select the armature, set the **port** to match the PoseCraft app's (default 8642),
then click **Start Live Link**. The armature becomes the **Live Link target**
(shown in the panel; pick another there any time), so you can select other
objects while you work and poses still go to the rig. Each received pose is
its own undo step (**Ctrl+Z** in Blender); an animation from **Send Animation
to Blender** is keyed frame by frame (whatever **Insert keyframes** says, and
without **Then advance**), the current frame is put back afterwards, and the
whole animation is a single undo step. Receiving never stalls Blender:
a sender that stops halfway is dropped after two seconds. The status line
underneath shows whether it's
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
  flagged `low_confidence` in the exported JSON, drawn dashed in orange-red in the
  app's photo overlay and 3D preview, and called out by name when applied in Blender (both
  via **Load Pose JSON** and **Live Link**).
- **Out of frame**: a bone reaching a body part MediaPipe places outside the
  photo (feet cut off at the bottom edge, say) is drawn dotted and flagged
  `"out_of_frame": true` in the JSON -- its pose is a guess, not merely
  hidden. Dragging that joint into place on the photo clears the flag.
- **Advanced** (under Hands and fingers, collapsed by default): **Detection
  threshold** (0.5) -- lower it (to 0.3, say) when a person in a hard photo
  isn't found, raise it to drop doubtful detections; **Low confidence below**
  (0.5) -- the visibility under which bones are flagged. Both are remembered.
- **Unmatched**: a PoseCraft bone name that couldn't be resolved to any bone
  on your armature, neither by your manual Bone Mapping nor by the automatic
  guess. Reported by name in Blender; fix it by adding a row for it in Bone
  Mapping.

Neither of these blocks anything from being applied -- they're informational,
so you know which parts of a pose are worth checking by hand.

## Known limitations

Worth knowing before you rely on an exported pose:

- **Twist on a straight limb is a guess.** PoseCraft solves the body's turn,
  head turns, foot angles and roll, and the roll of the upper arms and thighs
  from how the elbow or knee bends. A nearly straight arm or leg has no bend
  to measure its roll from, so it keeps the roll of the torso; forearm twist
  comes from the hand (half each). Check those in Blender.
- **One spine bone.** PoseCraft's torso is a single bone, so a twist between
  the shoulders and the hips is split evenly between the spine and the limbs.
- **A solved pose might come out mirrored or upside-down** for certain
  photos -- if that happens, try the **Mirror left/right** checkbox first,
  then the **Body orientation fix**.
- **A limb can come out reaching the wrong way in depth** (toward the camera
  instead of away) -- right-click it in the photo and flip its depth.
- **Fingers are optional and photo-only.** With **Hands and fingers** off,
  hands are one bone each. With it on, a hand that's mostly hidden or out of
  frame isn't searched, a finger's twist isn't estimated (fingers bend and
  spread; the thumb's roll follows the palm), and video stays body-only. Real
  photos haven't been checked against known finger poses yet.
- **Video identities can still swap** when two similarly dressed people
  cross closely or one is hidden for more than about a second (appearance only
  helps re-identify someone after they were lost, not while two people
  overlap); the root moves horizontally only through planted feet (a jump or
  a slide doesn't travel).
- **The Blender add-on** has been checked in Blender 5.2.2 on a few rigs
  (PoseCraft's own shape, a Mixamo-style rig, a freshly generated Rigify
  Human rig) and with Live Link, but not against every possible rig or
  Blender version -- if something looks off, the Bone Mapping panel is
  exactly the tool for fixing bone-name mismatches by hand.
- **DAZ Studio export is Genesis 9 only** and needs Genesis 9 installed in
  your DAZ content library; DAZ drives the twist bones itself, and a pose past
  Genesis 9's joint limits stops at the limit.
- The packaged `.exe` isn't code-signed, so expect a Windows SmartScreen
  warning on a machine other than the one it was built on.

## Troubleshooting

| Symptom | Try this |
|---|---|
| "No person was detected in this photo" | Check the framing tips under Load Photo -- full body in frame, decent lighting, not too small a photo. |
| Pose looks mirrored or flipped | Toggle **Mirror left/right**; if it's upside-down or turned sideways, use **Body orientation fix**. Then re-export/re-send. |
| One arm or leg reaches the wrong way (toward vs away from the camera) | Right-click that bone in the photo > **Flip depth**. |
| People in a wide group photo are missing | Tick **Group photo (slower)** and solve again (and raise **Max people to detect** if needed). |
| A limb looks twisted the wrong way in Blender | Most likely a straight arm or leg (see Known Limitations) -- correct it by hand in Blender. Check the 3D preview first: it shows what the export contains. |
| Arms come out too low (or bones point oddly) on your rig | Make sure **Match rig rest pose** is on in Blender's PoseCraft panel, and the file came from this version of PoseCraft. |
| "Unmatched" bone reported in Blender | Open the PoseCraft panel's **Bone Mapping** box and assign that bone manually. |
| "Could not reach Blender's Live Link" | In Blender, open the PoseCraft tab and click **Start Live Link** first; double-check the port matches on both sides. |
| Solve Pose seems stuck on first use | It's downloading the pose model -- check the progress bar, and that you have network access; **Cancel** stops it. |
| "doesn't match its pinned SHA-256" | The model download was corrupted (or the file on the server changed); try again on a stable connection. |
| A 5th+ person in a group photo isn't detected | Raise **Max people to detect** before clicking Solve Pose. |

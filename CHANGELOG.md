# Changelog

All notable changes to PoseCraft. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased] -- 0.1.0, the first release

### Added
- Photo, video and webcam-recording input; BlazePose landmarks on ONNX Runtime (DirectML,
  CUDA or CPU), with optional finger tracking and 52 ARKit-style face expression weights.
- Bone retargeting onto real rigs (Rigify, Mixamo-style and skeleton presets), joint dragging
  on the photo, multi-person video with re-identification, rotation smoothing and root motion.
- Export to JSON, BVH, glTF (.glb), FBX and a DAZ Genesis 9 pose preset; a pose library;
  folder batch processing (GUI and CLI); a live link to Blender.
- The Blender add-on (0.10.0), also as a Blender 4.2+ extension; it keys face expressions
  onto matching shape keys.
- Help > Diagnostics, a per-user installer, Authenticode signing and a self-test of every
  release build.
- A once-per-launch update check against the project's GitHub Releases (update.json).
- Dependabot, and a `pip-audit` job in CI.
- The MIT license, `THIRD_PARTY_NOTICES.md` (shipped in the build and the installer),
  `SECURITY.md` and a privacy note in the README.

### Fixed
- The Live Link server bounds connections, accepts and reads per timer tick.

from __future__ import annotations

import ast
from pathlib import Path

from pose_craft.export.blender_pose import FORMAT_VERSION
from pose_craft.retarget.skeleton import BONE_NAMES

ADDON_PATH = Path(__file__).resolve().parent.parent / "pose_craft" / "export" / "blender_addon" / "import_pose.py"


def _extract_top_level_constant(name: str) -> object:
    """Parses import_pose.py's source (rather than importing it -- it needs `bpy`,
    which isn't installed in this test environment) to pull out a module-level
    literal constant, so tests here can catch it drifting without needing Blender.
    """
    tree = ast.parse(ADDON_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {ADDON_PATH}")


def test_addon_bone_names_match_skeleton_bone_names():
    """POSECRAFT_BONE_NAMES is a hand-maintained mirror of skeleton.BONE_NAMES
    (the add-on can't import pose_craft, since it runs inside Blender's own
    Python) -- this is the regression guard for it drifting out of sync, which
    would otherwise silently break the manual bone-mapping UI's row list.
    """
    assert _extract_top_level_constant("POSECRAFT_BONE_NAMES") == BONE_NAMES


def test_addon_max_supported_format_version_matches_export_format_version():
    """MAX_SUPPORTED_FORMAT_VERSION is a hand-maintained mirror of
    blender_pose.FORMAT_VERSION, for the same reason POSECRAFT_BONE_NAMES
    mirrors BONE_NAMES -- catches the add-on silently falling behind (and
    rejecting every file/Live Link document) after a real format bump.
    """
    assert _extract_top_level_constant("MAX_SUPPORTED_FORMAT_VERSION") == FORMAT_VERSION


def test_addon_source_is_syntactically_valid():
    # Cheap sanity check that complements py_compile in dev: a parse failure here
    # would also break every other test in this file with a less obvious traceback.
    ast.parse(ADDON_PATH.read_text(encoding="utf-8"))

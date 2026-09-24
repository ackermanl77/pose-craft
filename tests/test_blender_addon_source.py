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


def _load_pure_functions(*names: str) -> dict:
    """Compiles just the named pure (bpy-free) functions plus the module-level
    literal constants they use out of the add-on's source, since the module
    itself can't be imported without Blender."""
    tree = ast.parse(ADDON_PATH.read_text(encoding="utf-8"))
    namespace: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            for target in node.targets:
                namespace[target.id] = value
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {f.name for f in functions} == set(names)
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(ADDON_PATH), "exec"), namespace)
    return namespace


def test_rigify_candidates_put_the_fk_tag_before_the_side_suffix():
    candidates = _load_pure_functions("bone_name_candidates")["bone_name_candidates"]
    arm = candidates("upper_arm.L")
    assert "upper_arm_fk.L" in arm
    assert "upper_arm.L_fk" not in arm
    # Control bones are preferred over constraint-driven DEF-/ORG- bones.
    assert arm.index("upper_arm_fk.L") < arm.index("DEF-upper_arm.L") < arm.index("ORG-upper_arm.L")
    assert candidates("shin.R")[:2] == ["shin.R", "shin_fk.R"]
    assert candidates("spine")[:2] == ["spine", "spine_fk"]


def test_mixamo_candidates_cover_every_bone():
    candidates = _load_pure_functions("bone_name_candidates")["bone_name_candidates"]
    assert "mixamorig:LeftArm" in candidates("upper_arm.L")
    assert "mixamorig:RightUpLeg" in candidates("thigh.R")
    for name in BONE_NAMES:
        assert any(c.startswith("mixamorig:") for c in candidates(name)), name
        assert len(set(candidates(name))) == len(candidates(name))


class _Quat:
    """Minimal stand-in for mathutils.Quaternion (`@` and .inverted())."""

    def __init__(self, rotation):
        self.r = rotation

    def __matmul__(self, other):
        return _Quat(self.r * other.r)

    def inverted(self):
        return _Quat(self.r.inv())


def test_armature_delta_to_basis_reproduces_the_armature_space_delta_down_a_chain():
    """Simulates Blender's pose evaluation (pose = parent_pose @
    parent_rest^-1 @ rest @ basis, rotation part only) for a two-bone chain
    whose rest frames are NOT aligned with the armature axes, and checks that
    the converted basis rotations give each bone the world-space delta the
    exported parent-relative rotations describe."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    to_basis = _load_pure_functions("armature_delta_to_basis")["armature_delta_to_basis"]
    rest_parent = Rotation.from_euler("xyz", [10, -70, 35], degrees=True)
    rest_child = Rotation.from_euler("xyz", [-40, 20, 95], degrees=True)
    delta_parent = Rotation.from_rotvec([0.3, -0.2, 0.9])  # exported local (root: == world delta)
    delta_child = Rotation.from_rotvec([-0.5, 0.4, 0.1])  # exported, relative to the parent

    basis_parent = to_basis(_Quat(rest_parent), _Quat(delta_parent)).r
    basis_child = to_basis(_Quat(rest_child), _Quat(delta_child)).r
    pose_parent = rest_parent * basis_parent
    pose_child = pose_parent * rest_parent.inv() * rest_child * basis_child

    np.testing.assert_allclose((pose_parent * rest_parent.inv()).as_matrix(), delta_parent.as_matrix(), atol=1e-9)
    np.testing.assert_allclose(
        (pose_child * rest_child.inv()).as_matrix(), (delta_parent * delta_child).as_matrix(), atol=1e-9
    )

"""Rigify-style rigs in the add-on (B5): constraint-driven DEF-/ORG- bones are
redirected to the controls that drive them, the limbs being posed are switched
to FK, and every bone is solved against its parent's pose as evaluated with
constraints (MCH- parents that follow other bones). Blender isn't installed,
so the add-on's pure functions run against a small simulation of Blender's
pose evaluation with Copy Transforms constraints -- the same approach as
test_blender_addon_source.py, and just as unverified in a real Blender."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).parent))
from test_blender_addon_source import (  # noqa: E402
    _align,
    _load_pure_functions,
    _posed_document,
    _Quat,
    _quat_wxyz,
    _rest_with_direction,
)

_FUNCS = (
    "bone_name_candidates", "is_constraint_driven", "constraint_driver", "control_bone_for",
    "rigify_fk_switches", "rig_pose_basis", "rig_pose_basis_evaluated", "bones_parents_first",
    "rigify_chain_extras",
)
Y = np.array([0.0, 1.0, 0.0])
_S = 2 ** -0.5


def _copy(target, influence=1.0):
    return {"type": "COPY_TRANSFORMS", "subtarget": target, "influence": influence, "mute": False}


# A cut-down generated Rigify rig in Blender armature axes. name: (parent,
# rest direction, roll, constraints, PoseCraft bone the FK control stands for).
# "ik_fk" in a constraint's influence means "driven by upper_arm_parent.L's IK_FK".
def _rigify_like_rig():
    return {
        "root": (None, (0, 1, 0), 0, [], None),
        "torso": ("root", (0, 0, 1), 0, [], None),
        "spine_fk": ("torso", (0, 0.1, 1), 5, [], "spine"),
        "ORG-spine": ("torso", (0, 0.1, 1), 5, [_copy("spine_fk")], None),
        # The arm's parent MCH follows the torso through a constraint, not the hierarchy.
        "MCH-upper_arm_parent.L": ("root", (0, 0.1, 1), 5, [_copy("ORG-spine")], None),
        "upper_arm_parent.L": ("root", (1, 0, 0), 0, [], None),  # holds IK_FK
        "upper_arm_fk.L": ("MCH-upper_arm_parent.L", (_S, 0, -_S), 30, [], "upper_arm.L"),
        "forearm_fk.L": ("upper_arm_fk.L", (_S, -0.1, -_S), -40, [], "forearm.L"),
        "MCH-upper_arm_ik.L": ("MCH-upper_arm_parent.L", (_S, 0, -_S), 30, [], None),
        "MCH-forearm_ik.L": ("MCH-upper_arm_ik.L", (_S, -0.1, -_S), -40, [], None),
        "ORG-upper_arm.L": ("root", (_S, 0, -_S), 30,
                            [_copy("MCH-upper_arm_ik.L"), _copy("upper_arm_fk.L", "ik_fk")], None),
        "ORG-forearm.L": ("ORG-upper_arm.L", (_S, -0.1, -_S), -40,
                          [_copy("MCH-forearm_ik.L"), _copy("forearm_fk.L", "ik_fk")], None),
        "DEF-upper_arm.L": ("root", (_S, 0, -_S), 30, [_copy("ORG-upper_arm.L")], None),
        "DEF-forearm.L": ("DEF-upper_arm.L", (_S, -0.1, -_S), -40, [_copy("ORG-forearm.L")], None),
    }


def _constraints(rig):
    return {
        name: [dict(c, influence=1.0 if c["influence"] == "ik_fk" else c["influence"]) for c in entry[3]]
        for name, entry in rig.items() if entry[3]
    }


class _Rig:
    """Blender's pose evaluation, rotation part: pose = parent_pose @
    parent_rest^-1 @ rest @ basis, then each Copy Transforms constraint blends
    the bone toward its target's pose by its influence."""

    def __init__(self, rig, ik_basis=None):
        self.rig = rig
        self.rest = {n: _rest_with_direction(d, roll) for n, (_p, d, roll, _c, _m) in rig.items()}
        if ik_basis is None:
            ik_basis = Rotation.from_rotvec([0.4, -0.3, 0.2])
        self.basis = {"MCH-upper_arm_ik.L": ik_basis}  # some IK solution the FK pose must override
        self.props = {"upper_arm_parent.L": {"IK_FK": 0.0}}  # Rigify's default: IK

    def pose(self):
        pose = {}
        for name, (parent, _d, _r, constraints, _m) in self.rig.items():
            rotation = self.rest[name] if parent is None else pose[parent] * self.rest[parent].inv() * self.rest[name]
            rotation = rotation * self.basis.get(name, Rotation.identity())
            for c in constraints:
                weight = self.props["upper_arm_parent.L"]["IK_FK"] if c["influence"] == "ik_fk" else c["influence"]
                if weight > 0:
                    both = Rotation.concatenate([rotation, pose[c["subtarget"]]])
                    rotation = Slerp([0.0, 1.0], both)([weight])[0]
            pose[name] = rotation
        return pose

    def evaluate(self, names):
        pose = self.pose()
        return {n: _Quat(pose[n]) for n in names}

    def write(self, name, basis):
        self.basis[name] = basis.r


def _targets_and_deltas(rig_obj, mapping, document):
    """What apply_pose_document computes: per mapped rig bone, PoseCraft's
    world rotation after aligning the rig's rest direction onto PoseCraft's."""
    targets, deltas = {}, {}
    for bone, posecraft in mapping.items():
        entry = document["bones"][posecraft]
        world = _quat_wxyz(entry["world_quaternion_wxyz"])
        deltas[bone] = _Quat(world * _align(rig_obj.rest[bone].apply(Y), entry["rest_direction"]))
        targets[posecraft] = world.apply(entry["rest_direction"])
    return targets, deltas


def _angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def test_deform_and_org_bones_are_redirected_to_the_fk_controls():
    loaded = _load_pure_functions(*_FUNCS)
    rig = _rigify_like_rig()
    constraints, existing = _constraints(rig), set(rig)
    control = loaded["control_bone_for"]
    assert control("DEF-upper_arm.L", constraints, existing) == "upper_arm_fk.L"  # DEF -> ORG -> the FK control
    assert control("ORG-forearm.L", constraints, existing) == "forearm_fk.L"  # FK preferred over the IK MCH
    assert control("upper_arm_fk.L", constraints, existing) == "upper_arm_fk.L"  # a free control stays
    assert control("spine_fk", constraints, existing) == "spine_fk"
    assert loaded["is_constraint_driven"]("DEF-forearm.L", constraints)
    assert not loaded["is_constraint_driven"]("forearm_fk.L", constraints)


def test_a_bone_driven_only_by_tracking_constraints_falls_back_to_its_unprefixed_control():
    loaded = _load_pure_functions(*_FUNCS)
    constraints = {"DEF-shin.L": [{"type": "STRETCH_TO", "subtarget": "MCH-shin_tweak.L", "influence": 1.0, "mute": False}]}
    existing = {"DEF-shin.L", "MCH-shin_tweak.L", "shin_fk.L"}
    assert loaded["control_bone_for"]("DEF-shin.L", constraints, existing) == "shin_fk.L"
    muted = {"DEF-shin.L": [dict(constraints["DEF-shin.L"][0], mute=True)]}
    assert loaded["control_bone_for"]("DEF-shin.L", muted, existing) == "DEF-shin.L"


def test_constraint_driver_stops_on_a_cycle():
    driver = _load_pure_functions(*_FUNCS)["constraint_driver"]
    cyclic = {"a": [_copy("b")], "b": [_copy("a")]}
    assert driver("a", cyclic) == "b"
    assert driver("free", cyclic) == "free"


def test_fk_switches_find_each_limbs_master_control():
    switches = _load_pure_functions(*_FUNCS)["rigify_fk_switches"]
    properties = {"upper_arm_parent.L": {"IK_FK", "IK_Stretch"}, "MCH-thigh_parent.R": {"IK_FK"}, "thigh_parent.L": set()}
    result = switches(["upper_arm_fk.L", "forearm_fk.L", "hand_fk.L", "thigh_fk.R", "shin_fk.L", "spine_fk", "head"], properties)
    assert result == [("upper_arm_parent.L", "IK_FK"), ("MCH-thigh_parent.R", "IK_FK")]
    assert switches(["upper_arm.L"], properties) == []  # not an FK control


def test_rigify_spine_chains_over_its_fk_controls():
    extras = _load_pure_functions(*_FUNCS)["rigify_chain_extras"]
    existing = {"spine_fk", "spine_fk.001", "spine_fk.002", "spine_fk.003"}
    assert extras("spine", "spine_fk", existing) == ["spine_fk.001", "spine_fk.002", "spine_fk.003"]
    assert extras("spine", "mixamorig:Spine", existing) == [] and extras("neck", "spine_fk", existing) == []


def test_a_rigify_like_rig_deforms_onto_the_targets_only_with_all_three_fixes():
    loaded = _load_pure_functions(*_FUNCS)
    rig = _rigify_like_rig()
    document = _posed_document()
    constraints, existing = _constraints(rig), set(rig)
    parents = {name: entry[0] for name, entry in rig.items()}
    order = loaded["bones_parents_first"](parents)

    # The user mapped the deform bones (or the rig only offered DEF-/ORG- names).
    user_mapping = {"spine_fk": "spine", "DEF-upper_arm.L": "upper_arm.L", "DEF-forearm.L": "forearm.L"}

    def deform_errors(sim):
        pose = sim.pose()
        return {
            name: _angle_deg(pose[name].apply(Y), targets[posecraft])
            for name, posecraft in (("DEF-upper_arm.L", "upper_arm.L"), ("DEF-forearm.L", "forearm.L"))
        }

    # Before: posing the DEF bones directly (hierarchy-only solve) changes nothing they show.
    old = _Rig(rig)
    targets, deltas = _targets_and_deltas(old, user_mapping, document)
    rest_q = {n: _Quat(r) for n, r in old.rest.items()}
    for name, basis in loaded["rig_pose_basis"](order, parents, rest_q, {}, deltas).items():
        old.write(name, basis)
    assert max(deform_errors(old).values()) > 20

    # Redirected to the FK controls but left in IK: still the IK pose.
    controls = {loaded["control_bone_for"](b, constraints, existing): p for b, p in user_mapping.items()}
    assert set(controls) == {"spine_fk", "upper_arm_fk.L", "forearm_fk.L"}
    ik = _Rig(rig)
    _targets, deltas = _targets_and_deltas(ik, controls, document)
    loaded["rig_pose_basis_evaluated"](order, parents, rest_q, deltas, ik.evaluate, ik.write)
    assert max(deform_errors(ik).values()) > 20

    # Controls + FK switch, but the hierarchy-only solve: the arm's MCH parent
    # follows the torso through a constraint, so the arm misses.
    hierarchy = _Rig(rig)
    for holder, prop in loaded["rigify_fk_switches"](list(controls), {k: set(v) for k, v in hierarchy.props.items()}):
        hierarchy.props[holder][prop] = 1.0
    _targets, deltas = _targets_and_deltas(hierarchy, controls, document)
    for name, basis in loaded["rig_pose_basis"](order, parents, rest_q, {}, deltas).items():
        hierarchy.write(name, basis)
    assert max(deform_errors(hierarchy).values()) > 5

    # All three: the deform bones land on PoseCraft's targets.
    fixed = _Rig(rig)
    for holder, prop in loaded["rigify_fk_switches"](list(controls), {k: set(v) for k, v in fixed.props.items()}):
        fixed.props[holder][prop] = 1.0
    assert fixed.props["upper_arm_parent.L"]["IK_FK"] == 1.0
    _targets, deltas = _targets_and_deltas(fixed, controls, document)
    loaded["rig_pose_basis_evaluated"](order, parents, rest_q, deltas, fixed.evaluate, fixed.write)
    errors = deform_errors(fixed)
    assert max(errors.values()) < 1e-4, errors
    assert _angle_deg(fixed.pose()["spine_fk"].apply(Y), targets["spine"]) < 1e-4


def test_the_evaluated_solve_equals_the_hierarchy_solve_without_constraints():
    """On a rig with no constraints (the Mixamo-like test rig), solving against
    evaluated parents gives exactly rig_pose_basis's answer."""
    from test_blender_addon_source import _MIXAMO_LIKE_RIG, _evaluate

    loaded = _load_pure_functions(*_FUNCS)
    document = _posed_document()
    rest = {name: _rest_with_direction(d, roll) for name, (_p, d, roll, _m) in _MIXAMO_LIKE_RIG.items()}
    parents = {name: entry[0] for name, entry in _MIXAMO_LIKE_RIG.items()}
    deltas = {}
    for name, (_p, _d, _r, posecraft) in _MIXAMO_LIKE_RIG.items():
        if posecraft is not None:
            entry = document["bones"][posecraft]
            world = _quat_wxyz(entry["world_quaternion_wxyz"])
            deltas[name] = _Quat(world * _align(rest[name].apply(Y), entry["rest_direction"]))
    rest_q = {n: _Quat(r) for n, r in rest.items()}
    expected = loaded["rig_pose_basis"](list(_MIXAMO_LIKE_RIG), parents, rest_q, {}, deltas)

    basis = {}

    def evaluate(names):
        pose = _evaluate(_MIXAMO_LIKE_RIG, rest, basis)
        return {n: _Quat(pose[n]) for n in names}

    def write(name, value):
        basis[name] = value.r

    result = loaded["rig_pose_basis_evaluated"](list(_MIXAMO_LIKE_RIG), parents, rest_q, deltas, evaluate, write)
    assert set(result) == set(expected)
    for name in expected:
        assert (result[name].r * expected[name].r.inv()).magnitude() < 1e-9, name

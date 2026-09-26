"""The URDF adapter for maker_arm_rs.Dynamics (tracking design section 5)."""

import ast
import pathlib
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.mj import scene

ROOT = pathlib.Path(__file__).resolve().parents[1]
DESK_POSE = (0.0, -2.03, 1.71, -1.17, 0.0, 0.0)


def _mj():
    """The vendored URDF as MuJoCo compiles it for the sim, with every term the rigid-body
    model leaves out switched off: no armature, damping, friction or constraints."""
    spec = mujoco.MjSpec.from_string(scene._urdf_with_compiler())
    spec.compiler.fusestatic = False
    spec.body("base_link").quat = scene._mount_quat()
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    m.dof_armature[:] = 0.0
    m.dof_damping[:] = 0.0
    m.dof_frictionloss[:] = 0.0
    d = mujoco.MjData(m)
    for name in urdf_prep.GRIPPER_JOINT_NAMES:
        d.qpos[m.joint(name).qposadr[0]] = dm.JAW_FRACTION * urdf_prep.gripper_travel()
    return m, d


def _addr(m, names, field):
    return [int(getattr(m.joint(n), field)[0]) for n in names]


def _mj_body(m, link):
    """MuJoCo's own inertia for ``link``, reconstructed as a ``dm`` body tuple, in the body frame."""
    bid = m.body(link).id
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, m.body_iquat[bid])
    r = r.reshape(3, 3)
    inertia = r @ np.diag(m.body_inertia[bid]) @ r.T
    return (
        float(m.body_mass[bid]),
        tuple(float(v) for v in m.body_ipos[bid]),
        tuple(tuple(float(v) for v in row) for row in inertia),
    )


def _mujoco_bodies(m, urdf=urdf_prep.VENDORED_URDF):
    """The ``dm.urdf_bodies()`` shape, but every inertia is MuJoCo's own compiled tensor.

    Isolates the adapter's URDF parsing from MuJoCo's inertia compilation: link_007 is
    lumped with the two jaw bodies through the adapter's own ``dm.lump`` and jaw transform,
    exactly as ``dm.arm_bodies`` does, so the only thing this helper changes is the source
    of each body's mass, COM and inertia.
    """
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint")}
    opening = dm.JAW_FRACTION * urdf_prep.gripper_travel(urdf)
    bodies = [_mj_body(m, link) for link in dm.ARM_LINKS[:-1]]
    jaws = [
        dm._jaw_in_link_007(_mj_body(m, link), joints[name], opening)
        for link, name in zip(dm.JAW_LINKS, urdf_prep.GRIPPER_JOINT_NAMES)
    ]
    bodies.append(dm.lump([_mj_body(m, dm.ARM_LINKS[-1]), *jaws]))
    return bodies


def _draws(limits, seed=20260925, n=200):
    """``n`` deterministic ``(q, dq, ddq)`` draws, shared by every RNEA parity test."""
    rng = random.Random(seed)
    for _ in range(n):
        q = [rng.uniform(lo, hi) for lo, hi in limits]
        dq = [rng.uniform(-3.0, 3.0) for _ in range(6)]
        ddq = [rng.uniform(-20.0, 20.0) for _ in range(6)]
        yield q, dq, ddq


def _rnea_worst_error(dyn, m, d, qpos, dof, limits):
    """The worst absolute torque gap between ``dyn.inverse`` and ``mj_inverse`` over ``_draws``."""
    worst = 0.0
    for q, dq, ddq in _draws(limits):
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.qpos[qpos] = q
        d.qvel[dof] = dq
        d.qacc[dof] = ddq
        mujoco.mj_inverse(m, d)
        ours = dyn.inverse(q, dq, ddq)
        worst = max(worst, float(np.max(np.abs(np.asarray(ours) - d.qfrc_inverse[dof]))))
    return worst


def test_rnea_matches_mujoco_inverse_dynamics_algorithm():
    """RNEA parity against ``mj_inverse`` when both sides use the same compiled inertias."""
    m, d = _mj()
    qpos = _addr(m, urdf_prep.ARM_JOINT_NAMES, "qposadr")
    dof = _addr(m, urdf_prep.ARM_JOINT_NAMES, "dofadr")
    dyn = dm.dynamics(_mujoco_bodies(m))
    limits = urdf_prep.arm_joint_limits()
    worst = _rnea_worst_error(dyn, m, d, qpos, dof, limits)
    assert worst < 1e-9, worst


def test_rnea_matches_mujoco_inverse_dynamics_adapter():
    """RNEA parity against ``mj_inverse`` for the adapter's own URDF-derived inertias.

    MuJoCo's compiler stores each inertia as principal moments plus a rotation (the scene
    also sets balanceinertia), so its tensor differs from the URDF's by up to about 1e-4
    relative per element. The measured worst torque difference is about 5.7e-8 N m.
    """
    m, d = _mj()
    qpos = _addr(m, urdf_prep.ARM_JOINT_NAMES, "qposadr")
    dof = _addr(m, urdf_prep.ARM_JOINT_NAMES, "dofadr")
    dyn = dm.dynamics(dm.urdf_bodies())
    limits = urdf_prep.arm_joint_limits()
    worst = _rnea_worst_error(dyn, m, d, qpos, dof, limits)
    assert worst < 1e-6, worst


def test_subtree_mass_and_com_match_mujoco():
    m, d = _mj()
    qpos = _addr(m, urdf_prep.ARM_JOINT_NAMES, "qposadr")
    d.qpos[qpos] = [0.3, -1.5, 1.2, -0.6, 0.4, -0.8]
    mujoco.mj_forward(m, d)
    bodies = dm.urdf_bodies()
    for k, link in enumerate(dm.ARM_LINKS):
        bid = m.body(link).id
        mass = sum(b[0] for b in bodies[k:])
        assert abs(mass - m.body_subtreemass[bid]) < 1e-12
        first = np.zeros(3)
        for i in range(k, 6):
            xb = m.body(dm.ARM_LINKS[i]).id
            com = d.xpos[xb] + d.xmat[xb].reshape(3, 3) @ np.asarray(bodies[i][1])
            first += bodies[i][0] * com
        assert np.max(np.abs(first / mass - d.subtree_com[bid])) < 1e-9


def test_urdf_gravity_at_the_desk_pose_matches_the_design_numbers():
    g = dm.dynamics(dm.urdf_bodies()).gravity(DESK_POSE)
    assert abs(abs(g[1]) - 4.76) < 0.01 and abs(abs(g[2]) - 2.91) < 0.01, g


def test_scaled_scales_mass_and_inertia_only():
    b = dm.urdf_bodies()
    s = dm.scaled(b, 1.3)
    for (m0, c0, i0), (m1, c1, i1) in zip(b, s):
        assert abs(m1 - 1.3 * m0) < 1e-15 and c1 == c0
        assert all(abs(i1[r][c] - 1.3 * i0[r][c]) < 1e-15 for r in range(3) for c in range(3))


def test_the_adapter_imports_only_the_stdlib_and_maker_arm_rs():
    allowed = {
        "__future__",
        "json",
        "math",
        "copy",
        "xml.etree.ElementTree",
        "pathlib",
        "collections.abc",
        "maker_arm_rs",
        "maker_arm_lab.assets.urdf_prep",
    }
    for rel in (
        "source/maker_arm_lab/core/dynamics_model.py",
        "source/maker_arm_lab/assets/urdf_prep.py",
    ):
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module}
            else:
                continue
            assert names <= allowed, (rel, names)

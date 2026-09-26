"""The URDF adapter for maker_arm_rs.Dynamics (tracking design section 5)."""

import ast
import pathlib
import random

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


def test_rnea_matches_mujoco_inverse_dynamics():
    m, d = _mj()
    qpos = _addr(m, urdf_prep.ARM_JOINT_NAMES, "qposadr")
    dof = _addr(m, urdf_prep.ARM_JOINT_NAMES, "dofadr")
    dyn = dm.dynamics(dm.urdf_bodies())
    limits = urdf_prep.arm_joint_limits()
    rng = random.Random(20260925)
    worst = 0.0
    for _ in range(200):
        q = [rng.uniform(lo, hi) for lo, hi in limits]
        dq = [rng.uniform(-3.0, 3.0) for _ in range(6)]
        ddq = [rng.uniform(-20.0, 20.0) for _ in range(6)]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.qpos[qpos] = q
        d.qvel[dof] = dq
        d.qacc[dof] = ddq
        mujoco.mj_inverse(m, d)
        ours = dyn.inverse(q, dq, ddq)
        worst = max(worst, float(np.max(np.abs(np.asarray(ours) - d.qfrc_inverse[dof]))))
    assert worst < 1e-9, worst


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

"""MS7d L6 and L7: the pick-point force from the arm torques (core.torque), the hold gate and
the MuJoCo member behind them, on the three-part heap with cameras off."""

import random

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import hold, loop, torque
from test_heap_loop import SPREAD, _env_and_rig
from test_hold import CLOSE, W, _grasp
from test_mj_heap_oracle import _place

QUIET = {"torque": 0.0, "torque_k": 0.0}
JAC = torch.tensor(
    [
        [0.0, -0.21, -0.12, 0.0, -0.05, 0.0],
        [0.28, 0.0, 0.0, 0.07, 0.0, 0.01],
        [0.0, 0.25, 0.18, 0.0, 0.06, 0.0],
    ],
    dtype=torch.float64,
)
"""A made-up pick point Jacobian of full row rank, m per rad."""


def test_the_least_squares_force_is_the_force_applied_at_the_pick_point():
    f = torch.tensor([3.0, -2.0, 12.0], dtype=torch.float64)
    assert torch.allclose(torque.pick_force(JAC, JAC.T @ f), f)


def _feed(sensor, steps):
    """(force applied at the pick point, in the window) per step; the forces read back."""
    return [
        sensor.update(
            JAC.T @ torch.tensor(f, dtype=JAC.dtype), torch.zeros(6), torch.zeros(6), JAC, w
        )
        for f, w in steps
    ]


def test_a_press_fades_into_the_rolling_tare_and_stays_against_the_snap_tare():
    free, press = (0.0, 0.0, 0.0), (0.0, 0.0, 20.0)
    sensor = torque.ForceSensor(1, 0, QUIET)
    # No tare before 15 steps of history, then free motion reads 0.
    assert _feed(sensor, [(press, False)] * 3 + [(free, False)] * 17) == [0.0] * 20
    # Outside the window a press reads its 20 N while t-15 to t-6 are free, half at 5 of 10
    # pressed tare steps, and 0 from 6 on.
    out = _feed(sensor, [(press, False)] * 12)
    assert out[:6] == pytest.approx([20.0] * 6)
    assert out[10] == pytest.approx(10.0) and out[11] == pytest.approx(0.0)
    # Free again, then into the window with the press: the snap tare of the 10 free steps
    # before the entry keeps it visible for as long as it lasts.
    _feed(sensor, [(free, False)] * 20)
    assert _feed(sensor, [(press, True)] * 30) == pytest.approx([20.0] * 30)
    # Out of the window the rolling tare takes over, which the press already filled.
    assert _feed(sensor, [(press, False)]) == pytest.approx([0.0])
    # A window entered with less than 10 steps of history reads 0 in that window.
    fresh = torque.ForceSensor(1, 0, QUIET)
    assert _feed(fresh, [(free, False)] * 9 + [(press, True)] * 5) == [0.0] * 14


def test_the_sensor_draws_12_at_reset_and_6_per_step_whatever_the_scales():
    states = []
    for scales in (QUIET, None, {"torque": 4.0, "torque_k": 2.0}):
        sensor = torque.ForceSensor(7, 3, scales)
        _feed(sensor, [((0.0, 0.0, 1.0), k % 2 == 0) for k in range(25)])
        states.append(sensor.rng.getstate())
    assert states[0] == states[1] == states[2]
    fresh = random.Random("torque:7:3")
    for _ in range(12 + 6 * 25):
        fresh.gauss(0.0, 1.0)
    assert fresh.getstate() == states[0]


def _stall(entries):
    """The teacher's grasp on a 2x4, then the stall at its width: one entry gate per step."""
    d = hold.HoldDetector()
    steps = _grasp() + [(W, CLOSE, 0.02)] * (len(entries) - len(_grasp()))
    rows = []
    for (measured, commanded, z), entry in zip(steps, entries):
        d.update(measured, commanded, z, entry)
        rows.append((d.holding, d.started, d.blocked))
    return d, rows


def test_a_hold_over_the_force_bar_does_not_enter_and_enters_once_the_force_drops():
    n = len(_grasp()) + 12
    _, open_rows = _stall([True] * n)
    first = next(t for t, row in enumerate(open_rows) if row[0])
    d, rows = _stall([True] * first + [False] * 6 + [True] * (n - first - 6))
    # The gate refuses the confirmed run once, keeps counting it, and the hold enters on the
    # first open step.
    assert [t for t, row in enumerate(rows) if row[2]] == [first]
    assert not any(row[0] for row in rows[: first + 6]) and rows[first + 6][:2] == (True, True)
    # A hold that began is never ended by the force, and the lift still latches.
    for k in range(1, 8):
        d.update(W, CLOSE, 0.02 + 0.011 * k, False)
        assert d.holding and not d.ended
    assert d.lifted
    # Closed on every step, the hold never enters.
    assert not any(row[0] for row in _stall([False] * n)[1])


def _trace(forces):
    """One cycle of the teacher's grasp and stall, a 9-column trace row per step."""
    steps = _grasp() + [(W, CLOSE, 0.02)] * (len(forces) - len(_grasp()))
    return [[0, t, 1, g, c, z, 3, 0.0, f] for t, ((g, c, z), f) in enumerate(zip(steps, forces))]


def test_cycle_rows_gate_entry_on_the_force_column_like_the_live_detector():
    forces = [0.0] * len(_grasp()) + [12.0] * 8 + [0.0] * 12
    trace = _trace(forces)
    open_at = hold.cycle_rows(trace)[0]["first_hold_at"]
    assert hold.cycle_rows([row[:8] for row in trace]) == hold.cycle_rows(trace)  # old traces
    for bar in (5.0, 8.0, 12.0, 13.0):
        entries = [f < bar for f in forces]
        _, rows = _stall(entries)
        live = next(t for t, row in enumerate(rows) if row[0])
        assert hold.cycle_rows(trace, force_max=bar)[0]["first_hold_at"] == live, bar
        assert (live > open_at) is (bar <= 12.0), bar


def test_the_env_member_reads_the_contact_torques_and_the_pick_jacobian():
    """The constraint torque equals J^T f summed over the arm's contacts (the QDD spike's
    cross-check), and it is zero in free motion. A finger pressed into the desk: the force the
    least-squares image reads at the pick point is the contact force, give or take 10 %."""
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    shapes = [tuple(t.shape) for t in env.arm_torques(0)]
    assert shapes == [(6,), (6,), (6,), (3, 6)]
    assert torch.equal(env.arm_torques(0)[0], torch.zeros(6, dtype=torch.float64))
    q = env.scene["robot"].data.joint_pos[0, :6].double()
    down = loop.snapped_arm(rig.ik, q, (0.0, 0.1, -float(rig.pick_point(0)[2]) - 0.01))
    action = torch.zeros(1, 7)
    action[0, :6], action[0, 6] = down.float(), 0.05
    for _ in range(10):
        env.step(action)
    m, d, h = env.models[0], env.datas[0], env.handles
    arm = {b for b in range(m.nbody) if _under(m, b, h.base_body)}
    dofs, jacp, f6 = list(h.joint_dof[:6]), np.zeros((3, m.nv)), np.zeros(6)
    ext, total = np.zeros(6), np.zeros(3)
    for k in range(d.ncon):
        con = d.contact[k]
        mine = [g for g in (con.geom1, con.geom2) if int(m.geom_bodyid[g]) in arm]
        if len(mine) != 1:
            continue
        mujoco.mj_contactForce(m, d, k, f6)
        world = con.frame.reshape(3, 3).T @ f6[:3]  # geom1's force on geom2
        on_arm = world if mine[0] == con.geom2 else -world
        mujoco.mj_jac(m, d, jacp, None, con.pos, int(m.geom_bodyid[mine[0]]))
        ext += jacp[:, dofs].T @ on_arm
        total += on_arm
    constraint, _, _, jac = env.arm_torques(0)
    env.close()
    assert np.linalg.norm(total) > 10.0
    assert np.allclose(constraint.numpy(), ext, atol=1e-6)
    read = float(torque.pick_force(jac, constraint).norm())
    assert read == pytest.approx(np.linalg.norm(total), rel=0.1)


def _under(m, b, base) -> bool:
    while b > 0 and b != base:
        b = int(m.body_parentid[b])
    return b == base

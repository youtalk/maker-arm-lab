"""The heap outer loop on a three-part MuJoCo heap, cameras off (phase B design 5.2)."""

import math
import random

import h5py
import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import dagger_loop, layout, loop, oracle
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import overlay
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_heap_env import _can_render
from test_mj_heap_oracle import PARTS, _place, _step

SPREAD = [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]


def _env_and_rig(seed=11, budget=4, cameras=False, **env_kwargs):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=PARTS,
        cameras=cameras,
        episode_length_s=loop.episode_length_s(budget),
        **env_kwargs,
    )
    env.reset(seed=seed)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl, env.parts, int(env.destination_bins[i]), episode_draw, rng=rng, cycle_budget=budget
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    return env, rig


class _HoldPolicy:
    """Holds the arm still with the jaws open: every cycle ends on its step budget."""

    def __init__(self):
        self.context_dim = 0
        self.policy_id = "fake:hold"
        self.n_action_steps = 1
        self.inference = {}
        self.action_space = "absolute"

    def step(self, state, wrist, overhead, context=None, anchor=None):
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = state[:, :6]
        action[:, 6] = 1.0
        return action


def _rollout(env, rig, policy, budget=4, **kwargs):
    return loop.rollout(
        env,
        rig,
        policy,
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=budget,
        max_wall_minutes=5.0,
        **kwargs,
    )


def test_cycle_budget_and_episode_length_follow_the_gate_rule():
    assert loop.cycle_budget(50) == 65 and loop.cycle_budget(3) == 4
    assert loop.episode_length_s(65) == pytest.approx(65 * loop.CYCLE_STEP_BUDGET / 30.0)


def test_grip_open_flags_read_both_gripper_contracts():
    binary = torch.tensor([1.0, -1.0, 0.0])
    assert loop.grip_open_flags(binary, "binary").tolist() == [True, False, True]
    assert loop.grip_open_flags(binary, "binary", torch.full((3,), 0.5)).tolist() == [
        True,
        False,
        True,
    ]
    assert torch.equal(loop.open_signal(binary, "binary"), binary)
    # A 7.8 mm and a 39.8 mm grasp: each one's landing (w + 8 mm) reads open and its close
    # (w - 6 mm) reads closed, which no single fixed threshold does for both.
    w = torch.tensor([0.0078, 0.0078, 0.0398, 0.0398])
    gaps = torch.tensor([0.0158, 0.0018, 0.0478, 0.0338])
    assert loop.grip_open_flags(gaps, "continuous", w).tolist() == [True, False, True, False]
    assert (loop.open_signal(gaps, "continuous", w) >= 0.0).tolist() == [True, False, True, False]
    # No cycle: a fully open coast reads open, a zero command reads closed.
    idle = torch.tensor([0.1048, 0.0])
    assert loop.grip_open_flags(idle, "continuous").tolist() == [True, False]


def test_the_continuous_rigs_commands_read_open_and_closed_on_the_per_target_threshold():
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    opening = (oracle.Phase.HOVER, oracle.Phase.DESCEND, oracle.Phase.RELEASE)
    closing = (oracle.Phase.CLOSE, oracle.Phase.LIFT)
    seen, target, close_below = set(), None, None
    for _ in range(9000):
        orc = rig.oracles[0]
        if orc is not None and orc.target_brick is not None and orc.target_brick != target:
            # A new cycle: the loop reads its threshold once, on the scene it opens on.
            target = orc.target_brick
            close_below = loop.cycle_close_below(orc, rig.snapshot(0), target)
            assert close_below == pytest.approx(orc.grasp_width)
        goal = rig.goals[0]
        action, out = _step(rig)
        if goal is not None and goal.phase in opening + closing:
            is_open = loop.grip_open_flags(action[:, 6], "continuous", torch.tensor([close_below]))
            assert bool(is_open[0]) == (goal.phase in opening), (goal.phase, float(action[0, 6]))
            seen.add(goal.phase)
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    assert set(opening + closing) <= seen, seen


def test_cycle_end_names_the_three_ends():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=100, stowed_before={2})
    assert loop.cycle_end(cycle, {2}, False, 100) is None
    assert loop.cycle_end(cycle, {0, 2}, False, 100) == ("stowed", [0])
    assert loop.cycle_end(cycle, {2, 1}, False, 100) == ("stowed", [1])  # rung b5: any part
    assert loop.cycle_end(cycle, {2}, False, 100 + loop.CYCLE_STEP_BUDGET) == ("budget", [])
    assert loop.cycle_end(cycle, {2}, True, 100) == ("out_of_reach", [])


def test_a_part_an_earlier_cycle_stowed_never_ends_a_later_cycle():
    # Part 1 was stowed by cycle 1 but still moving when cycle 2 opened, so it is missing from
    # stowed_before; when it settles it must not end cycle 2 as a stow.
    cycle = loop.Cycle(ordinal=2, target=0, start_step=100, stowed_before={2})
    assert loop.cycle_end(cycle, {1, 2}, False, 101, credited={1}) is None
    assert loop.cycle_end(cycle, {0, 1, 2}, False, 101, credited={1}) == ("stowed", [0])


def test_only_a_resting_part_off_the_reach_band_and_off_the_bin_is_out_of_reach():
    bl = layout.DEFAULT_BIN_LAYOUT
    half = (0.05, 0.05)
    cx, cy, _ = layout.heap_bin_center(bl)
    assert not layout.in_reach(cx, cy, bl)
    gap = torch.tensor([0.60, 0.0, 0.006])
    assert loop.resting_out_of_reach(gap, 0.0, False, bl, half)
    assert not loop.resting_out_of_reach(gap, 0.3, False, bl, half)  # carried
    assert not loop.resting_out_of_reach(gap, 0.0, True, bl, half)  # between the jaws
    assert not loop.resting_out_of_reach(torch.tensor([0.60, 0.0, 0.15]), 0.0, False, bl, half)
    assert not loop.resting_out_of_reach(torch.tensor([cx, cy, 0.006]), 0.0, False, bl, half)
    assert not loop.resting_out_of_reach(torch.tensor([0.20, 0.0, 0.006]), 0.0, False, bl, half)


def test_the_oracle_driver_stows_three_spread_parts_through_the_loop():
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    payload = _rollout(env, rig, None)
    env.close()
    ep = payload["episodes"][0]
    assert ep["bricks_stowed"] == 3 and ep["success"] is True, ep
    assert ep["cycles"] == 3 and ep["cycle_outcomes"] == [True, True, True]
    assert ep["cycle_reasons"] == ["stowed"] * 3 and sorted(ep["cycle_targets"]) == [0, 1, 2]
    assert ep["takeovers"] == 3 and ep["takeovers_stowed"] == 3
    assert sorted(ep["stowed_parts"]) == [0, 1, 2]
    assert ep["stowed_families"] == sorted(p.family for p in env.parts)
    assert payload["policy_id"] == "oracle" and payload["cycles_per_stowed"] == pytest.approx(1.0)
    assert payload["gripper"] == "binary" and payload["cycle_budget"] == 4


def test_a_still_policy_spends_its_budget_and_the_failure_memory_excludes_the_cell(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    env, rig = _env_and_rig(budget=9)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _HoldPolicy(), budget=9)
    env.close()
    ep = payload["episodes"][0]
    assert ep["bricks_stowed"] == 0 and ep["success"] is False
    assert set(ep["cycle_reasons"]) == {"budget"} and all(s <= 41 for s in ep["cycle_steps"])
    # Three failures close a cell. Part 0's cell takes three, then parts 1 and 2 share a 5 cm
    # cell that three failures on part 2 close for both: the selector runs dry at six cycles,
    # under the budget of nine.
    assert ep["cycles"] == 6 and ep["done_reason"] == "no_candidate"
    assert ep["cycle_targets"] == [0, 0, 0, 2, 2, 2]
    assert ep["takeovers"] == 0 and payload["policy_id"] == "fake:hold"
    assert payload["stowed_total"] == 0 and payload["cycles_per_stowed"] is None


def test_a_still_policy_under_a_short_budget_ends_on_the_cycle_budget(monkeypatch):
    # The env is built long enough for every cycle to finish, then each cycle gets 40 steps.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    env, rig = _env_and_rig(budget=4)
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _HoldPolicy(), budget=4)
    env.close()
    ep = payload["episodes"][0]
    # Under the selector's natural dry-out of six cycles (the test above).
    assert ep["cycles"] == 4 and ep["done_reason"] == "cycle_budget", ep


class _FramePolicy(_HoldPolicy):
    """Holds still and records, per call, whether the target's pixel is the marker's white."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.hits = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        env, t = self.env, int(self.env.target_part[0])
        d, c = env.datas[0], env._cam_overhead
        u, v = overlay.project_point(
            d.cam_xpos[c],
            d.cam_xmat[c],
            float(env.models[0].cam_fovy[c]),
            224,
            224,
            d.xpos[env.handles.brick_bodies[t]],
        )
        self.hits.append(tuple(overhead[0, round(v), round(u)]) == (255, 255, 255))
        return super().step(state, wrist, overhead, context=context, anchor=anchor)


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_the_policy_reads_the_marker_on_the_loops_target(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 5)
    env, rig = _env_and_rig(budget=2, cameras=True)
    _place(env, SPREAD)
    policy = _FramePolicy(env)
    _rollout(env, rig, policy, budget=2)
    env.close()
    assert len(policy.hits) >= 5 and all(policy.hits), policy.hits


def test_the_marker_follows_the_loops_target():
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    seen = []
    stepped = env.step

    def env_step(action):
        seen.append(int(env.target_part[0]))
        return stepped(action)

    env.step = env_step
    _rollout(env, rig, None)
    env.close()
    assert -1 not in seen[:-1] and set(seen) - {-1} == {0, 1, 2}


class _KickPolicy(_HoldPolicy):
    """Holds still, and on its third step throws the target part off the desk."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.calls = 0

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.calls += 1
        if self.calls == 3:
            m, d, h = self.env.models[0], self.env.datas[0], self.env.handles
            adr = h.brick_qpos[int(self.env.target_part[0])]
            d.qpos[adr : adr + 3] = (0.60, 0.0, 0.006)
            mujoco.mj_forward(m, d)
            self.env._refresh([0], images=False)
        return super().step(state, wrist, overhead, context=context, anchor=anchor)


def test_a_target_that_leaves_reach_ends_its_cycle(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 400)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _KickPolicy(env), budget=3)
    env.close()
    ep = payload["episodes"][0]
    assert ep["cycle_reasons"][0] == "out_of_reach" and ep["cycle_steps"][0] < 10


class _CarryPolicy(_HoldPolicy):
    """Holds the arm still and carries the target, lifted and moving, into the bin."""

    PATH_STEPS = 30

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.calls = 0
        self.path = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.calls += 1
        n = self.calls - 2
        if 0 <= n <= self.PATH_STEPS:
            m, d, h = self.env.models[0], self.env.datas[0], self.env.handles
            b = int(self.env.target_part[0])
            adr, dof = h.brick_qpos[b], h.brick_dof[b]
            if n == 0:
                self.start = np.array(d.qpos[adr : adr + 2])
            cx, cy, cz = layout.heap_bin_center(self.env.bin_layouts[0])
            goal = np.array((cx, cy))
            s = n / self.PATH_STEPS
            xy = self.start + s * (goal - self.start)
            z = cz + (0.08 if n < self.PATH_STEPS else 0.0)  # over the raised bin's walls
            d.qpos[adr : adr + 3] = (xy[0], xy[1], z)
            d.qvel[dof : dof + 6] = 0.0
            if n < self.PATH_STEPS:
                d.qvel[dof : dof + 2] = (goal - self.start) / self.PATH_STEPS * 30.0
            mujoco.mj_forward(m, d)
            self.env._refresh([0], images=False)
            self.path.append(tuple(xy))
        return super().step(state, wrist, overhead, context=context, anchor=anchor)


def test_a_carry_through_out_of_reach_space_ends_as_stowed(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    policy = _CarryPolicy(env)
    payload = _rollout(env, rig, policy, budget=1)
    bl = env.bin_layouts[0]
    env.close()
    assert any(not layout.in_reach(x, y, bl) for x, y in policy.path)
    ep = payload["episodes"][0]
    assert ep["cycle_reasons"] == ["stowed"] and ep["bricks_stowed"] == 1, ep


def test_a_carry_paused_over_the_reach_gap_keeps_its_cycle(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    bl = env.bin_layouts[0]
    cx, _, _ = layout.heap_bin_center(bl)
    x = next(v / 1000 for v in range(250, 450) if not layout.in_reach(v / 1000, 0.0, bl)) + 0.02
    assert x < cx - float(env.stow_half_extents[0]) - 0.02  # in the gap, short of the bin
    pause = range(2, 20)  # 18 steps held still at 0.15 m over the gap, out of the jaws
    calls = [0]
    stepped = env.step

    def env_step(action):
        out = stepped(action)
        calls[0] += 1
        if calls[0] in pause:
            m, d, h = env.models[0], env.datas[0], env.handles
            b = int(env.target_part[0])
            adr, dof = h.brick_qpos[b], h.brick_dof[b]
            d.qpos[adr : adr + 3] = (x, 0.0, 0.15)
            d.qvel[dof : dof + 6] = 0.0
            mujoco.mj_forward(m, d)
            env._refresh([0], images=False)
        return out

    env.step = env_step
    payload = _rollout(env, rig, _HoldPolicy(), budget=1)
    env.close()
    ep = payload["episodes"][0]
    # The pause keeps the cycle, and the part dropped on the desk in the gap then ends it.
    assert ep["cycle_reasons"] == ["out_of_reach"] and ep["cycle_steps"][0] > pause.stop, ep


def test_the_loop_refuses_a_six_brick_env_and_a_context_mismatch():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, None, budget=3)
    env.close()
    assert err.value.reason == "scene"

    wide = _HoldPolicy()
    wide.context_dim = 9
    continuous = _HoldPolicy()
    continuous.gripper = "continuous"
    for policy, reason in ((wide, "context"), (continuous, "gripper")):
        env, rig = _env_and_rig()
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, policy)
        env.close()
        assert err.value.reason == reason


def test_the_loop_refuses_a_selector_with_another_cycle_budget():
    env, rig = _env_and_rig(budget=4)
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, None, budget=5)
    env.close()
    assert err.value.reason == "cycle_budget"


def test_a_still_policy_is_taken_over_on_no_close_and_only_takeovers_are_recorded(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 60)
    env, rig = _env_and_rig(budget=4, export_dir=tmp_path, shard_name="d", export_all=True)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _HoldPolicy(), takeover=loop.TakeoverConfig(random_interval=None))
    env.close()
    ep = payload["episodes"][0]
    assert ep["takeovers"] >= 1 and payload["triggers"].get("no_close", 0) >= 1, payload["triggers"]
    assert ep["bricks_stowed"] >= 1
    with h5py.File(tmp_path / "d.hdf5", "r") as f:
        actor = f["data/demo_0"]["obs/actor"][:]
        marked = f["data/demo_0"]["obs/target_part"][:]
    assert (actor != 0).all(), "a policy frame was recorded"
    assert set(abs(actor).tolist()) <= set(range(1, ep["takeovers"] + 1))
    # One marker, on the loop's target: every recorded frame marks a part a cycle targeted.
    assert set(marked.tolist()) <= set(ep["cycle_targets"]), (marked, ep["cycle_targets"])


def test_a_takeover_keeps_the_marker_on_the_loops_target(monkeypatch):
    # The rig's cycle part differs from the loop's target when it adopts a held part; force
    # that on every takeover step and check the env still marks the loop's target, which is
    # also the part the loop stamped into the frame before the step.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 30)
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    real_actions = rig.actions

    def actions():
        out = real_actions()
        if not rig.idle[0]:
            env.oracle_cycle_brick[0] = (int(env.target_part[0]) + 1) % 3
        return out

    monkeypatch.setattr(rig, "actions", actions)
    pairs = []
    stepped = env.step

    def env_step(action):
        before = int(env.target_part[0])
        out = stepped(action)
        if int(env.oracle_actor[0]) != 0:
            pairs.append((before, int(env.target_part[0])))
        return out

    env.step = env_step
    _rollout(env, rig, _HoldPolicy(), budget=1, takeover=loop.TakeoverConfig(random_interval=None))
    env.close()
    assert pairs and all(a == b for a, b in pairs), pairs


def _resume_then_fail(rig, monkeypatch):
    real_resume = rig.resume

    def resume_then_fail(i, target=None):
        ok = real_resume(i, target=target)
        if ok:
            rig._go_idle(i, False)  # hand back at once as a failed takeover
        return ok

    monkeypatch.setattr(rig, "resume", resume_then_fail)


def test_a_takeover_that_fails_is_negated(tmp_path, monkeypatch):
    env, rig = _env_and_rig(budget=3, export_dir=tmp_path, shard_name="n", export_all=True)
    _place(env, SPREAD)
    _resume_then_fail(rig, monkeypatch)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=3, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    assert payload["takeovers_failed"] == payload["takeovers"] >= 1
    with h5py.File(tmp_path / "n.hdf5", "r") as f:
        actor = f["data/demo_0"]["obs/actor"][:]
    assert (actor < 0).all(), "a failed takeover left a positive-actor frame"


def test_a_takeover_the_time_out_cuts_short_is_failed_and_negated(tmp_path, monkeypatch):
    # 90 steps of episode: the takeover the no-close trigger starts at step 10 is still
    # running when the env times out.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 30)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 10)
    env, rig = _env_and_rig(budget=3, export_dir=tmp_path, shard_name="t", export_all=True)
    _place(env, SPREAD)
    payload = _rollout(
        env,
        rig,
        _HoldPolicy(),
        budget=3,
        takeover=loop.TakeoverConfig(random_interval=None),
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["done_reason"] == "time_out" and ep["takeovers"] == 1, ep
    assert ep["takeovers_failed"] == 1 and ep["takeovers_stowed"] == 0, ep
    with h5py.File(tmp_path / "t.hdf5", "r") as f:
        actor = f["data/demo_0"]["obs/actor"][:]
    assert len(actor) > 10 and (actor == -1).all(), actor


def test_the_half_budget_trigger_fires_without_a_hold(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 10_000)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=3, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    assert payload["triggers"].get("half_budget_no_lift", 0) >= 1, payload["triggers"]


def _half_budget_triggers(monkeypatch, rise):
    """Triggers of a one-cycle episode where the loop reads part 0 held and ``rise`` m above
    its height at the cycle's open on every step after it."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 10_000)
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    _hold_part_zero(monkeypatch)
    real_states = loop._brick_states
    calls = []

    def brick_states(env):
        pos, vel = real_states(env)
        if calls:  # the first call opens the cycle and records its heights
            pos = pos.clone()
            pos[:, 0, 2] += rise
        calls.append(1)
        return pos, vel

    monkeypatch.setattr(loop, "_brick_states", brick_states)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=1, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    return payload["triggers"]


def test_the_half_budget_trigger_fires_on_a_held_part_with_no_lift(monkeypatch):
    triggers = _half_budget_triggers(monkeypatch, rise=0.0)
    assert triggers.get("half_budget_no_lift", 0) == 1, triggers


def test_the_half_budget_trigger_spares_a_held_part_lifted_past_the_rise(monkeypatch):
    triggers = _half_budget_triggers(monkeypatch, rise=oracle.LIFT_MIN_RISE + 0.01)
    assert "half_budget_no_lift" not in triggers, triggers


def test_a_failed_takeover_ends_its_cycle_and_the_episode_cap_holds(monkeypatch):
    # Every 50 open steps trigger and every takeover fails at once. A failed hand-back ends the
    # loop cycle (one takeover per cycle, phase B design 5.5); a factor of 0.5 on a budget of
    # four caps the episode at two takeovers, so the last two cycles run to their step budget.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 50)
    env, rig = _env_and_rig(budget=4)
    _place(env, SPREAD)
    _resume_then_fail(rig, monkeypatch)
    payload = _rollout(
        env,
        rig,
        _HoldPolicy(),
        budget=4,
        takeover=loop.TakeoverConfig(random_interval=None, max_per_episode_factor=0.5),
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["takeovers"] == ep["takeovers_failed"] == 2 and ep["cycles"] == 4, ep
    assert ep["cycle_reasons"] == ["takeover_failed"] * 2 + ["budget"] * 2, ep
    assert all(s <= 52 for s in ep["cycle_steps"][:2]), ep


def test_one_takeover_per_cycle_under_the_default_cap(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 50)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    _resume_then_fail(rig, monkeypatch)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=3, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["takeovers"] == ep["cycles"] == 3, ep
    assert ep["cycle_reasons"] == ["takeover_failed"] * 3, ep


def test_a_declined_takeover_ends_its_cycle_as_a_failure(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 50)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    monkeypatch.setattr(rig, "resume", lambda i, target=None: False)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=3, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["takeovers"] == 0 and payload["takeovers_declined"] >= 1, payload
    assert ep["cycle_reasons"][0] == "takeover_declined", ep
    assert set(ep["cycle_reasons"]) <= {"takeover_declined", "budget"}, ep


def _hold_part_zero(monkeypatch):
    """The loop reads part 0 between the jaws on every step (the rig keeps its own read)."""
    monkeypatch.setattr(
        loop,
        "held_brick_from_forces",
        lambda left, right, min_force: torch.zeros(left.shape[0], dtype=torch.long),
    )


def test_a_decline_on_a_held_part_keeps_its_cycle_with_a_cooldown(monkeypatch):
    # RULING P32 declines most of a correct carry. A random trigger fires on every step and
    # the rig declines each one while the loop reads a held part: the cycle runs on until the
    # episode's time-out (one cycle of budget), the cooldown spaces the declines, and no
    # decline marks the failure memory.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 150)
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    _hold_part_zero(monkeypatch)
    monkeypatch.setattr(rig, "resume", lambda i, target=None: False)
    marks = []
    real_selector = loop._selector

    def selector(rig, i):
        sel = real_selector(rig, i)
        real_mark = sel._mark_failure
        sel._mark_failure = lambda scene, b: (marks.append(b), real_mark(scene, b))[1]
        return sel

    monkeypatch.setattr(loop, "_selector", selector)
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=1, takeover=loop.TakeoverConfig(random_interval=1)
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["cycle_reasons"] == ["episode_end"] and ep["takeovers"] == 0, ep
    # Declines at steps 0, 61 and 122 of 150: one per cooldown of 60 steps.
    assert payload["takeovers_declined"] == 3, payload
    assert marks == [], marks


def test_a_takeover_on_a_target_out_of_reach_is_declined_not_raised():
    env, rig = _env_and_rig(budget=3)
    _place(env, [(0.60, 0.0, 0.006, 0.0)] + SPREAD[1:])
    assert not layout.in_reach(0.60, 0.0, env.bin_layouts[0])
    rig.rearm([0], idle=True)
    ok = rig.resume(0, target=0)
    env.close()
    assert ok is False and rig.idle[0] and rig.takeover_result[0] is None


def test_the_random_trigger_needs_a_policy():
    env, rig = _env_and_rig(budget=3)
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, None, budget=3, takeover=loop.TakeoverConfig(random_interval=300))
    env.close()
    assert err.value.reason == "takeover"


def test_census_step_counts_the_duty_the_hold_and_the_first_close():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    loop.census_step(cycle, grip_open=True, held=-1, rise=0.0, close_cm=9.0)
    loop.census_step(cycle, grip_open=False, held=-1, rise=0.0, close_cm=1.25)
    loop.census_step(cycle, grip_open=False, held=2, rise=0.03, close_cm=0.5)
    loop.census_step(cycle, grip_open=False, held=2, rise=0.07, close_cm=0.5)
    loop.census_step(cycle, grip_open=True, held=-1, rise=0.0, close_cm=30.0)
    assert (cycle.open_steps, cycle.closed_steps, cycle.held_steps) == (2, 3, 2)
    assert cycle.first_close_cm == 1.25 and cycle.max_rise == pytest.approx(0.07)
    assert cycle.holding is False


def _census(**fields):
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    for name, value in fields.items():
        setattr(cycle, name, value)
    return cycle


def test_cycle_class_names_the_last_stage_a_failed_cycle_reached():
    held = loop.CENSUS_MIN_HELD_STEPS
    lifted = oracle.LIFT_MIN_RISE
    assert loop.cycle_class(_census(closed_steps=9), "stowed") == "stowed"
    assert loop.cycle_class(_census(open_steps=800), "budget") == "no_close"
    assert loop.cycle_class(_census(closed_steps=800), "budget") == "no_open"
    assert loop.cycle_class(_census(open_steps=500, closed_steps=300), "budget") == "close_miss"
    # A pinch shorter than the hold window's start is a brush, not a hold.
    brush = _census(open_steps=500, closed_steps=300, held_steps=held - 1, max_rise=lifted)
    assert loop.cycle_class(brush, "budget") == "close_miss"
    low = _census(open_steps=1, closed_steps=1, held_steps=held, max_rise=lifted - 0.001)
    assert loop.cycle_class(low, "budget") == "held_no_lift"
    up = _census(open_steps=1, closed_steps=1, held_steps=held, max_rise=lifted)
    assert loop.cycle_class(up, "budget") == "lifted_lost"
    up.holding = True
    assert loop.cycle_class(up, "budget") == "carrying"


def test_a_still_policy_is_classed_no_close_and_the_teacher_closes_on_its_target(monkeypatch):
    env, rig = _env_and_rig(gripper_mode="continuous")
    _place(env, SPREAD)
    ep = _rollout(env, rig, None)["episodes"][0]
    env.close()
    assert ep["cycle_classes"] == ["stowed"] * 3, ep
    # The teacher's first close lands on the target: well inside the 2 cm close-far rule.
    closes = ep["cycle_first_close_cm"]
    assert all(cm is not None and cm < dagger_loop.CLOSE_FAR_CM for cm in closes), closes

    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    env, rig = _env_and_rig(budget=9)
    _place(env, SPREAD)
    ep = _rollout(env, rig, _HoldPolicy(), budget=9)["episodes"][0]
    env.close()
    assert ep["cycle_classes"] == ["no_close"] * 6 and ep["cycle_first_close_cm"] == [None] * 6


def test_the_next_target_waits_for_the_retreat_or_the_cap():
    at_retreat = torch.tensor(oracle.RETREAT_POINT)
    assert loop.select_ready(at_retreat, 1.0, 0)
    assert loop.select_ready(at_retreat + torch.tensor([0.05, 0.0, 0.0]), 0.9, 10)
    assert not loop.select_ready(at_retreat, 0.3, 10)  # jaws still closed
    assert not loop.select_ready(torch.tensor([0.44, 0.0, 0.30]), 1.0, 10)  # at the bin
    cap = int(loop.SELECT_WAIT_FRACTION * loop.CYCLE_STEP_BUDGET)
    assert cap == 300 and not loop.select_ready(torch.tensor([0.44, 0.0, 0.30]), 0.0, cap - 1)
    assert loop.select_ready(torch.tensor([0.44, 0.0, 0.30]), 0.0, cap)


def test_a_part_inside_the_bin_but_still_moving_ends_the_cycle_and_not_the_metric():
    from maker_arm_lab.core.stowed import per_brick_stowed

    env, _rig = _env_and_rig()
    _place(env, SPREAD)
    cx, cy, cz = layout.heap_bin_center(env.bin_layouts[0])
    m, d, h = env.models[0], env.datas[0], env.handles
    adr = h.brick_qpos[0]
    d.qpos[adr : adr + 3] = (cx, cy, cz)
    d.qvel[h.brick_dof[0] : h.brick_dof[0] + 3] = (0.3, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    env._refresh([0], images=False)
    assert not bool(per_brick_stowed(env)[0, 0])
    assert bool(per_brick_stowed(env, max_speed=math.inf)[0, 0])
    env.close()

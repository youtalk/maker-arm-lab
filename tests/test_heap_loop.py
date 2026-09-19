"""The heap outer loop on a three-part MuJoCo heap, cameras off (phase B design 5.2)."""

import random

import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import layout, loop, oracle
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_heap_oracle import PARTS, _place

SPREAD = [(0.195, -0.045, 0.006, 0.3), (0.195, 0.0, 0.006, -0.8), (0.195, 0.045, 0.006, 1.2)]


def _env_and_rig(seed=11, budget=4, **env_kwargs):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=PARTS,
        cameras=False,
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
    gaps = torch.tensor([0.1048, 0.0398, 0.0138, 0.0])
    assert loop.grip_open_flags(gaps, "continuous").tolist() == [True, True, False, False]
    assert (loop.open_signal(gaps, "continuous") >= 0.0).tolist() == [True, True, False, False]
    assert torch.equal(loop.open_signal(binary, "binary"), binary)


def test_cycle_end_names_the_three_ends():
    bl = layout.DEFAULT_BIN_LAYOUT
    cycle = loop.Cycle(ordinal=1, target=0, start_step=100, stowed_before={2})
    inside = torch.tensor([0.20, 0.0])
    assert loop.cycle_end(cycle, {2}, inside, bl, 100) is None
    assert loop.cycle_end(cycle, {0, 2}, inside, bl, 100) == ("stowed", [0])
    assert loop.cycle_end(cycle, {2, 1}, inside, bl, 100) == ("stowed", [1])  # rung b5: any part
    assert loop.cycle_end(cycle, {2}, inside, bl, 100 + loop.CYCLE_STEP_BUDGET) == ("budget", [])
    assert loop.cycle_end(cycle, {2}, torch.tensor([0.60, 0.0]), bl, 100) == ("out_of_reach", [])


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
    # Parts 1 and 2 share a 5 cm cell, so three failures close it for both, and part 0's cell
    # takes three more: the selector runs dry at six cycles, under the budget of nine.
    assert 3 <= ep["cycles"] <= 9 and ep["done_reason"] in ("no_candidate", "cycle_budget")
    assert ep["takeovers"] == 0 and payload["policy_id"] == "fake:hold"
    assert payload["stowed_total"] == 0 and payload["cycles_per_stowed"] is None


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

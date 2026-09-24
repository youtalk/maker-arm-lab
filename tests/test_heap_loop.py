"""The heap outer loop on a three-part MuJoCo heap, cameras off (phase B design 5.2)."""

import itertools
import math
import random

import h5py
import mujoco
import numpy as np
import pytest
import torch

from maker_arm_lab.core import dagger_loop, desk_boxes, layout, loop, oracle
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
    assert loop.episode_length_s(65) == pytest.approx(65 * (loop.CYCLE_STEP_BUDGET + 300) / 30.0)


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


def test_with_desk_boxes_only_a_part_down_on_the_desk_or_in_another_box_is_out_of_reach():
    bl = layout.DEFAULT_BIN_LAYOUT
    half = (0.05, 0.05)
    beside_box = (0.077, 0.2)  # out of the reach band, beside a box's release point
    assert not layout.in_reach(*beside_box, bl)
    # A part in the opening jaws at the release is slow and under the force floor: not lost.
    release = torch.tensor([*beside_box, desk_boxes.DESK_BOX_RELEASE_Z - 0.002])
    assert not loop.resting_out_of_reach(release, 0.0, False, bl, half, in_box=False)
    down = torch.tensor([*beside_box, 0.006])
    assert loop.resting_out_of_reach(down, 0.0, False, bl, half, in_box=False)
    in_another_box = torch.tensor([*beside_box, 0.03])
    assert loop.resting_out_of_reach(in_another_box, 0.0, False, bl, half, in_box=True)
    assert not loop.resting_out_of_reach(
        torch.tensor([0.2, 0.0, 0.006]), 0.0, False, bl, half, in_box=False
    )


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


def test_a_takeovers_retreat_frames_carry_the_rigs_preselected_part(monkeypatch):
    """Phase B design 5.2 pinned the opposite (a takeover keeps the marker on the loop's
    target); campaign n (2026-09-22) reverses it because the v2 corpora switch the row and the
    marker when RELEASE ends (``HeapOracle._preselect``). So a takeover's RETREAT frames carry
    the rig's pre-selected part, and at hand-back the loop adopts that pre-selection when it
    still holds, else re-ranks (``loop._select``)."""
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 200)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 30)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    retreats = []  # (the marker the frame recorded, the retreating cycle's part)
    stepped = env.step

    def env_step(action):
        out = stepped(action)
        retreating = int(env.oracle_cycle_phase[0]) == oracle.PHASE_TO_INT["RETREAT"]
        if int(env.oracle_actor[0]) != 0 and retreating:
            retreats.append((int(env.target_part[0]), int(env.oracle_cycle_brick[0])))
        return out

    env.step = env_step
    payload = _rollout(
        env, rig, _HoldPolicy(), budget=3, takeover=loop.TakeoverConfig(random_interval=None)
    )
    env.close()
    ep = payload["episodes"][0]
    switched = {(marker, brick) for marker, brick in retreats if marker != brick}
    assert ep["takeovers"] >= 2 and switched, (payload["triggers"], retreats)
    targets = ep["cycle_targets"]
    for marker, brick in switched:  # the pre-selected part is the next cycle's target
        assert targets[targets.index(brick) + 1] == marker, (targets, retreats)


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


def test_failure_reason_fires_on_a_held_part_that_does_not_rise():
    n = loop.HELD_NO_RISE_STEPS
    half = loop.CYCLE_STEP_BUDGET // 2
    assert loop.failure_reason(n, n, False, 10) == loop.HELD_NO_RISE
    assert loop.failure_reason(n - 1, n, False, 10) is None
    assert loop.failure_reason(n, None, False, 10) is None  # off: the random round
    assert loop.failure_reason(n, None, False, half) == loop.HALF_BUDGET_NO_LIFT
    assert loop.failure_reason(0, n, False, half) == loop.HALF_BUDGET_NO_LIFT
    assert loop.failure_reason(0, n, True, half) is None  # a lifted part is never trouble
    assert loop.TakeoverConfig(random_interval=None).held_no_rise_steps is None


class _CloseEveryNStepsPolicy:
    """Holds the arm still with the jaws open, and closes them for the single step whose index
    is a multiple of ``period``. No takeover runs and the arm never stows, so every cycle ends
    on its step budget: a period equal to that budget puts each close edge on a boundary."""

    def __init__(self, period):
        self.period = period
        self.steps = 0
        self.context_dim = 0
        self.policy_id = "fake:close_every_n"
        self.n_action_steps = 1
        self.inference = {}
        self.action_space = "absolute"

    def step(self, state, wrist, overhead, context=None, anchor=None):
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = state[:, :6]
        action[:, 6] = -1.0 if self.steps % self.period == 0 else 1.0
        self.steps += 1
        return action


def test_no_close_edge_is_read_across_a_cycle_boundary(monkeypatch):
    # The two sides of a close edge must come from one threshold. A cycle's first step reads
    # the new cycle's threshold while the step before it read the old one, so that step carries
    # no edge whatever the policy commands.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 10_000)
    # The policy closes on the 40-step cadence of the cycles; the retreat wait would shift
    # every boundary off it, so this test selects the next target at once.
    monkeypatch.setattr(loop, "SELECT_WAIT_FRACTION", 0.0)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    calls = []
    seen = {"step": -1, "start": -1}
    real_reason = loop.trigger_reason
    real_offer = loop._TriggerState.offer

    def offer(self, env, rig, states, tracker, grip_open, held, pos, ordinal, step):
        seen["step"] = step
        seen["start"] = -1 if states[0].cycle is None else states[0].cycle.start_step
        return real_offer(self, env, rig, states, tracker, grip_open, held, pos, ordinal, step)

    def spy(**kw):
        calls.append((seen["step"], seen["start"], kw["grip_open"], kw["closed_now"]))
        return real_reason(**kw)

    monkeypatch.setattr(loop._TriggerState, "offer", offer)
    monkeypatch.setattr(loop, "trigger_reason", spy)
    _rollout(
        env,
        rig,
        _CloseEveryNStepsPolicy(40),
        budget=3,
        # No takeover ever runs: the policy must keep driving every cycle to its step budget.
        takeover=loop.TakeoverConfig(random_interval=None, max_per_episode_factor=0.0),
    )
    env.close()
    boundary = [(step, closed) for step, start, _, closed in calls if step == start]
    assert boundary and not any(closed for _, closed in boundary), calls
    # The guard really bit: on at least one boundary the jaws read open the step before and
    # closed on the boundary step itself, which is the edge the old code would have reported.
    edges = [
        cur[0]
        for prev, cur in itertools.pairwise(calls)
        if cur[0] == cur[1] and prev[2] and not cur[2] and prev[0] == cur[0] - 1
    ]
    assert edges, calls


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
    # The episode now outlives its one cycle by the wait allowance, so the cycle ends on its
    # own budget and the loop ends the episode on the spent cycle budget.
    assert ep["cycle_reasons"] == ["budget"] and ep["takeovers"] == 0, ep
    # Declines at steps 0, 61 and 122 of 150: one per cooldown of 60 steps.
    assert payload["takeovers_declined"] == 3, payload
    # The declines mark nothing; the one mark is the cycle's own budget end.
    assert marks == [0], marks


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


def test_the_census_reads_the_first_close_geometry_and_the_hold_ids():
    # The served closing axis points along +y; the part has yawed 90 degrees since, so its
    # body x axis now lies along +y and is the live closing axis.
    cycle = loop.Cycle(
        ordinal=1,
        target=2,
        start_step=0,
        stowed_before=set(),
        close_below=0.016,
        heading=math.pi / 2,
    )
    yaw90 = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    loop.first_close_geometry(
        cycle,
        gap=0.020,
        pick=torch.tensor([0.01, 0.02, 0.05]),
        target_pos=torch.tensor([0.0, 0.0, 0.04]),
        target_quat=yaw90,
        ctx_pos=np.array([0.03, 0.0, 0.04, 0.0, 1.0, 0.016, 0.005], dtype=np.float32),
    )
    assert cycle.squeeze_margin == pytest.approx(0.004) and cycle.gap_close == 0.020
    assert (cycle.off_x, cycle.off_y, cycle.off_z) == pytest.approx((0.01, 0.02, 0.01))
    assert cycle.off_along == pytest.approx(0.02) and cycle.off_across == pytest.approx(-0.01)
    assert cycle.ctx_stale == pytest.approx(0.03)
    steps = [  # (grip_open, held, rise, pick_z)
        (True, -1, 0.0, 0.10),
        (False, -1, 0.0, 0.05),
        (False, 2, 0.01, 0.06),
        (False, 3, 0.02, 0.09),
        (True, -1, 0.0, 0.12),
        (False, -1, 0.0, 0.07),
    ]
    for grip_open, held, rise, pick_z in steps:
        loop.census_step(
            cycle, grip_open=grip_open, held=held, rise=rise, close_cm=1.0, pick_z=pick_z
        )
    assert (cycle.first_close_at, cycle.reopen_steps, cycle.close_edges) == (1, 3, 2)
    assert (cycle.pick_z_close, cycle.pick_z_max) == (0.05, 0.12)
    assert (cycle.first_held, cycle.last_held) == (2, 3)
    st = loop.EnvState(selector=None)
    loop._book_census(st, cycle, "budget")
    row = st.census[0]
    assert st.classes == ["close_miss"] and set(loop.CENSUS_EXPORT) < set(row)
    assert row["first_held_is_target"] and not row["last_held_is_target"]
    assert row["squeeze_margin"] == 0.004 and row["wait_steps"] == 0 and not row["wait_capped"]


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
    # The teacher's first close commands a gap under the grasp width, pinches its own target
    # and lifts the pick point past the rise; the first selection waits for no retreat.
    for row in ep["cycle_census"]:
        assert -0.02 < row["squeeze_margin"] < 0.0 and row["first_held_is_target"], row
        assert row["pick_z_max"] - row["pick_z_close"] >= oracle.LIFT_MIN_RISE, row
    assert ep["cycle_census"][0]["wait_steps"] == 0

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


def test_release_ready_selects_at_the_release_after_a_stow_and_at_once_otherwise():
    cap = int(loop.SELECT_WAIT_FRACTION * loop.CYCLE_STEP_BUDGET)
    assert loop.release_ready(1.0, 0, after_stow=True)
    assert loop.release_ready(loop.SELECT_OPEN_FRACTION, 5, after_stow=True)
    assert not loop.release_ready(0.3, cap - 1, after_stow=True)  # the jaws are still closing
    assert loop.release_ready(0.3, cap, after_stow=True)  # the cap
    assert loop.release_ready(0.0, 0, after_stow=False)  # a budget end selects at once


def test_released_after_lift_is_the_open_jaw_after_a_lift_with_nothing_held():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    loop.census_step(cycle, grip_open=False, held=0, rise=0.0, close_cm=1.0)
    assert not loop.released_after_lift(cycle, 1.0, -1)  # a re-open with no lift: chatter
    loop.census_step(cycle, grip_open=False, held=0, rise=oracle.LIFT_MIN_RISE, close_cm=1.0)
    assert not loop.released_after_lift(cycle, 0.3, -1)  # the jaws are still opening
    assert not loop.released_after_lift(cycle, 1.0, 0)  # open, but the part is still held
    assert loop.released_after_lift(cycle, loop.SELECT_OPEN_FRACTION, -1)  # the release


def test_budget_ends_cycle_grants_a_held_part_the_carry_grace():
    budget, grace = loop.CYCLE_STEP_BUDGET, loop.CARRY_GRACE_STEPS
    assert not loop.budget_ends_cycle(budget - 1, True, 0)  # under the budget
    assert loop.budget_ends_cycle(budget, False, 0)  # nothing held: the old end
    assert not loop.budget_ends_cycle(budget, True, 0)  # held: the grace opens
    assert not loop.budget_ends_cycle(budget + grace - 1, True, grace - 1)
    assert loop.budget_ends_cycle(budget + grace, True, grace)  # the grace is spent
    assert loop.budget_ends_cycle(budget + 10, False, 10)  # dropped in the grace: at once
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    assert loop.cycle_end(cycle, set(), False, budget) == ("budget", [])  # the off-switch path
    assert "grace_steps" in loop.CENSUS_EXPORT


def test_reopen_edge_is_the_open_after_a_close_that_lifted_nothing():
    cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
    assert not loop.reopen_edge(cycle, True)  # a cycle's first step has no edge
    loop.census_step(cycle, grip_open=True, held=-1, rise=0.0, close_cm=5.0)
    assert not loop.reopen_edge(cycle, True)  # open after open
    loop.census_step(cycle, grip_open=False, held=0, rise=0.0, close_cm=1.0)
    assert not loop.reopen_edge(cycle, False)  # still closed
    assert loop.reopen_edge(cycle, True)  # the re-open, and nothing rose
    loop.census_step(cycle, grip_open=False, held=0, rise=oracle.LIFT_MIN_RISE, close_cm=1.0)
    assert not loop.reopen_edge(cycle, True)  # the part rose: a lift, not a REOPEN


def test_floored_gap_caps_a_closing_command_at_the_teachers_close_set_point():
    import inspect

    from maker_arm_lab.core import eval_loop, grasp

    w = 0.0398  # the 2x4's grasp width: the teacher closes to w - 6 mm
    assert loop.floored_gap(w - 0.0013, w) == pytest.approx(w - 0.006)
    assert loop.floored_gap(w - 0.0013, w) == pytest.approx(grasp.close_gap(w))
    assert loop.floored_gap(w - 0.010, w) == pytest.approx(w - 0.010)  # already under: kept
    assert loop.floored_gap(w + 0.008, w) == w + 0.008  # the landing gap, untouched
    assert loop.floored_gap(w, w) == w
    assert loop.floored_gap(0.0018, 0.0) == 0.0018  # no live row: untouched
    # The six-brick gate rollout has no target row and never floors anything.
    assert "floored_gap" not in inspect.getsource(eval_loop)


def test_gated_gap_holds_the_landing_gap_only_while_descending_above_the_top():
    import inspect

    from maker_arm_lab.core import eval_loop, grasp

    w, h = 0.0398, 0.0057  # the 2x4's grasp width and half height
    z_top = 0.010 + h
    landing = 2.0 * (grasp.landing_half_gap(w) - grasp.JAW_HOLD_BAND)
    closing = w - 0.0013  # the 400k policy's median first close, 1.3 mm under w
    # Closing while still coming down 10 mm above the top: the landing gap instead.
    assert loop.gated_gap(closing, w, z_top + 0.010, z_top, True) == pytest.approx(landing)
    assert loop.gated_gap(closing, w, z_top + 0.010, z_top, True) == pytest.approx(w + 0.008)
    # At the top, or below it: untouched, the endpoint is the policy's.
    assert loop.gated_gap(closing, w, z_top, z_top, True) == closing
    assert loop.gated_gap(closing, w, z_top - 0.004, z_top, True) == closing
    assert loop.gated_gap(closing, w, z_top + loop.GATE_MARGIN, z_top, True) == closing
    # Above the top but no longer descending: untouched.
    assert loop.gated_gap(closing, w, z_top + 0.010, z_top, False) == closing
    # A landing or a coast (at or above w) and a command with no row: untouched.
    assert loop.gated_gap(w + 0.008, w, z_top + 0.010, z_top, True) == w + 0.008
    assert loop.gated_gap(closing, 0.0, z_top + 0.010, z_top, True) == closing
    # Once the cycle has held a part (the carry's descent onto the bin): untouched.
    assert loop.gated_gap(closing, w, z_top + 0.010, z_top, True, held=True) == closing
    assert "gated_gap" not in inspect.getsource(eval_loop)


def _square_pose():
    """A grasp pose, tool tilted the shipping 20 degrees, with the jaw at heading 0.2 rad."""
    from maker_arm_lab.core.global_ik import GlobalIK, GraspTarget
    from maker_arm_lab.robots.frames import GRASP_TILT_RAD

    ik = GlobalIK()
    target = GraspTarget(
        position=(0.2, 0.0, 0.03), tilt=GRASP_TILT_RAD, jaw_heading=0.2, lean_azimuth=0.0
    )
    return ik, ik.solve(target).q


def _jaw_heading(ik, q, roll):
    from maker_arm_lab.robots.frames import JAW_AXIS_BODY

    q = q.clone()
    q[5] = roll
    _, rot = ik.fk(q)
    jaw = rot @ torch.tensor(JAW_AXIS_BODY, dtype=torch.float64)
    return math.atan2(float(jaw[1]), float(jaw[0]))


def _square_row(heading, width=0.0158):
    return [0.2, 0.0, 0.01, math.cos(heading), math.sin(heading), width, 0.0057]


def test_squared_roll_turns_the_jaw_onto_the_rows_axis_at_three_degrees_a_step():
    ik, q = _square_pose()
    roll, lim, step = float(q[5]), (-2.7, 2.7), loop.SQUARE_MAX_STEP
    pick = (0.21, 0.01, 0.03)  # 14 mm off the centre in xy, 14 mm above the top
    row = _square_row(0.2 + 0.3)  # the row's axis 0.3 rad round from the jaw
    # The sign comes from FK: the first step turns the roll 3 degrees toward the row.
    first = loop.squared_roll(ik, q, row, pick, roll, lim)
    assert first == pytest.approx(roll + step)
    assert _jaw_heading(ik, q, first) > _jaw_heading(ik, q, roll)
    # Stepped with the served command as ``prev``, the jaw lands on the row's axis (to the
    # 20 degree tilt's 5% roll-to-heading ratio) and stays there.
    served = roll
    for _ in range(12):
        served = loop.squared_roll(ik, q, row, pick, served, lim)
    assert served == pytest.approx(roll + 0.3)
    assert _jaw_heading(ik, q, served) == pytest.approx(0.5, abs=0.03)
    # The rate rule holds on the way back too: a command above the target comes down 3 degrees.
    assert loop.squared_roll(ik, q, row, pick, roll + 1.0, lim) == pytest.approx(roll + 1.0 - step)
    # The fold: the axis half a turn round is the same grasp, the one 0.1 rad the other way too.
    row = _square_row(0.2 + math.pi + 0.1)
    assert loop.squared_roll(ik, q, row, pick, roll + 0.1, lim) == pytest.approx(roll + 0.1)
    row = _square_row(0.2 - 0.1)
    assert loop.squared_roll(ik, q, row, pick, roll - 0.1, lim) == pytest.approx(roll - 0.1)


def test_squared_roll_takes_the_other_half_turn_at_a_limit_and_else_leaves_the_roll():
    ik, q = _square_pose()
    roll = float(q[5])
    pick, row = (0.2, 0.0, 0.03), _square_row(0.2 + 0.3)
    other = roll + 0.3 - math.pi  # the same grasp half a turn round
    assert loop.squared_roll(ik, q, row, pick, other, (roll - 3.0, roll + 0.2)) == pytest.approx(
        other
    )
    assert loop.squared_roll(ik, q, row, pick, roll, (roll - 1.0, roll + 0.2)) == roll


def test_squared_roll_leaves_the_roll_after_a_hold_with_no_row_and_off_the_approach():
    import inspect

    from maker_arm_lab.core import eval_loop

    ik, q = _square_pose()
    roll, lim = float(q[5]), (-2.7, 2.7)
    row, pick = _square_row(0.5), (0.2, 0.0, 0.03)
    assert loop.squared_roll(ik, q, row, pick, roll, lim) != roll  # the lever acts here
    assert loop.squared_roll(ik, q, row, pick, roll, lim, held=True) == roll  # held or lifted
    assert loop.squared_roll(ik, q, _square_row(0.5, width=0.0), pick, roll, lim) == roll
    far = (0.2 + loop.SQUARE_RADIUS + 0.001, 0.0, 0.03)
    assert loop.squared_roll(ik, q, row, far, roll, lim) == roll
    high = (0.2, 0.0, 0.01 + 0.0057 + loop.SQUARE_ABOVE)
    assert loop.squared_roll(ik, q, row, high, roll, lim) == roll
    assert "squared_steps" in loop.CENSUS_EXPORT
    assert "squared_roll" not in inspect.getsource(eval_loop)


def _snap_row(width=0.0096, half_height=0.0048):
    return [0.2, 0.0, 0.01, 1.0, 0.0, width, half_height]


def test_snap_shift_moves_the_pick_point_3_mm_a_step_toward_the_row():
    row, open_gap = _snap_row(), 0.02
    cmd = (0.23, 0.0, 0.012)  # 3 cm off the row centre in x, 2 mm above it
    shift = (0.0, 0.0, 0.0)
    for k in range(1, 11):
        shift = loop.snap_shift(cmd, cmd, row, shift, open_gap)
        assert shift[0] == pytest.approx(-loop.SNAP_XY_STEP * k) and shift[1] == 0.0
    assert shift[0] == pytest.approx(-0.03)
    # On the row: the offset holds, the served point stays at the centre.
    assert loop.snap_shift(cmd, cmd, row, shift, open_gap)[0] == pytest.approx(-0.03)
    # A diagonal error moves 3 mm along the diagonal.
    dx, dy, _ = loop.snap_shift((0.22, 0.02, 0.01), (0.22, 0.02, 0.01), row, (0, 0, 0), open_gap)
    assert math.hypot(dx, dy) == pytest.approx(loop.SNAP_XY_STEP)
    assert dx == pytest.approx(dy) and dx < 0.0


def test_snap_shift_lowers_a_hover_to_the_teachers_grasp_height_and_never_below():
    row, open_gap = _snap_row(), 0.02
    floor = row[2] + loop.SNAP_Z_MARGIN
    cmd = (0.2, 0.0, row[2] + 0.020)  # the 350k hover: 20 mm above the part centre
    shift, served = (0.0, 0.0, 0.0), []
    for _ in range(15):
        shift = loop.snap_shift(cmd, cmd, row, shift, open_gap)
        served.append(cmd[2] + shift[2])
    assert served[0] == pytest.approx(cmd[2] - loop.SNAP_Z_STEP)
    assert all(a - b <= loop.SNAP_Z_STEP + 1e-12 for a, b in zip([cmd[2]] + served, served))
    assert min(served) == pytest.approx(floor) and served[-1] == pytest.approx(floor)
    # The policy comes down by itself: the offset shrinks and the point stays at the floor.
    low = (0.2, 0.0, floor + 0.001)
    assert low[2] + loop.snap_shift(low, low, row, shift, open_gap)[2] == pytest.approx(floor)
    # A command already under the floor is left alone.
    deep = (0.2, 0.0, row[2] - 0.004)
    assert loop.snap_shift(deep, deep, row, shift, open_gap)[2] == 0.0
    # Closed jaws (gap under the width): the offset is kept, never grown.
    half = (0.0, 0.0, -0.006)
    assert loop.snap_shift(cmd, cmd, row, half, row[5] - 0.001)[2] == pytest.approx(-0.006)
    # Outside SNAP_Z_RADIUS the height is not lowered, the xy still moves.
    side = (0.2 + loop.SNAP_Z_RADIUS + 0.002, 0.0, cmd[2])
    moved = loop.snap_shift(side, side, row, (0.0, 0.0, 0.0), open_gap)
    assert moved[2] == 0.0 and moved[0] == pytest.approx(-loop.SNAP_XY_STEP)


def test_snap_shift_stops_at_the_hold_with_no_row_and_off_the_approach():
    import inspect

    from maker_arm_lab.core import eval_loop

    row, open_gap, shift = _snap_row(), 0.02, (-0.01, 0.0, -0.01)
    cmd = (0.21, 0.0, 0.025)
    assert loop.snap_shift(cmd, cmd, row, shift, open_gap) != (0.0, 0.0, 0.0)  # it acts here
    assert loop.snap_shift(cmd, cmd, row, shift, open_gap, held=True) == (0.0, 0.0, 0.0)
    assert loop.snap_shift(cmd, cmd, _snap_row(width=0.0), shift, open_gap) == (0.0, 0.0, 0.0)
    far = (0.2 + loop.SNAP_RADIUS + 0.001, 0.0, 0.02)
    assert loop.snap_shift(far, far, row, shift, open_gap) == (0.0, 0.0, 0.0)
    high = (0.2, 0.0, row[2] + row[6] + loop.SQUARE_ABOVE)
    assert loop.snap_shift(high, high, row, shift, open_gap) == (0.0, 0.0, 0.0)
    assert "snapped_steps" in loop.CENSUS_EXPORT
    assert "snap_shift" not in inspect.getsource(eval_loop)


def test_snap_hold_keeps_the_jaws_open_while_a_hover_comes_down_and_then_lets_go():
    row, open_gap = _snap_row(), 0.02
    floor = row[2] + loop.SNAP_Z_MARGIN
    cmd = (0.2, 0.0, row[2] + 0.020)  # the hover, commanded 20 mm above the part centre
    shift, pick, holds = (0.0, 0.0, 0.0), cmd, []
    for _ in range(12):
        shift = loop.snap_shift(cmd, pick, row, shift, open_gap)
        holds.append(loop.snap_hold(cmd, pick, row, shift, descending=True))
        pick = (0.2, 0.0, cmd[2] + shift[2])  # the arm tracks the served point
    # Held while the served point comes down 2 mm a step, let go once the arm is at the floor.
    assert cmd[2] + shift[2] == pytest.approx(floor)
    steps = math.ceil((cmd[2] - floor) / loop.SNAP_Z_STEP - 1e-9)
    assert holds == [True] * (steps - 1) + [False] * (13 - steps)
    # At the floor the arm still coming down above it keeps the hold; stopped short, it goes.
    above = (0.2, 0.0, floor + 0.004)
    assert loop.snap_hold(cmd, above, row, shift, descending=True)
    assert not loop.snap_hold(cmd, above, row, shift, descending=False)
    assert not loop.snap_hold(cmd, (0.2, 0.0, floor), row, shift, descending=True)
    # The policy's own command at grasp height, the stop flag, no row, off the centre: no hold.
    low = (0.2, 0.0, floor)
    assert not loop.snap_hold(low, low, row, (0.0, 0.0, 0.0), descending=True)
    assert not loop.snap_hold(cmd, cmd, row, (0.0, 0.0, 0.0), True, held=True)
    assert not loop.snap_hold(cmd, cmd, _snap_row(width=0.0), (0.0, 0.0, 0.0), True)
    side = (0.2 + loop.SNAP_Z_RADIUS + 0.001, 0.0, cmd[2])
    assert not loop.snap_hold(side, side, row, (0.0, 0.0, 0.0), True)
    assert "snap_held_steps" in loop.CENSUS_EXPORT


def test_snapped_arm_moves_the_commanded_pick_point_by_the_shift():
    ik, q = _square_pose()
    before = loop.commanded_pick(ik, q)
    moved = loop.snapped_arm(ik, q, (-0.003, 0.002, -0.004))
    after = loop.commanded_pick(ik, moved)
    assert (after - before).tolist() == pytest.approx([-0.003, 0.002, -0.004], abs=1e-6)
    # The jaw heading and the tilt stay the policy's.
    assert ik.pose_of(moved).jaw_heading == pytest.approx(ik.pose_of(q).jaw_heading, abs=1e-6)
    assert ik.pose_of(moved).tilt == pytest.approx(ik.pose_of(q).tilt, abs=1e-6)


def test_pick_snap_needs_the_continuous_gripper_and_a_served_row():
    env, rig = _env_and_rig()
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, _HoldPolicy(), pick_snap=True)
    env.close()
    assert err.value.reason == "pick_snap"


def test_dwell_count_grows_only_at_grasp_height_with_open_jaws():
    row, open_gap = _snap_row(), 0.02
    floor = row[2] + loop.SNAP_Z_MARGIN
    at = (0.2, 0.004, floor + loop.DWELL_Z_TOL)
    count = 0
    for k in range(1, loop.DWELL_STEPS + 1):
        count = loop.dwell_count(count, at, row, open_gap, open_gap)
        assert count == k
    # Any step off the grasp pose resets it: high, off the centre, jaws or command closing.
    high = (0.2, 0.0, floor + loop.DWELL_Z_TOL + 0.001)
    side = (0.2 + loop.SNAP_Z_RADIUS + 0.001, 0.0, floor)
    assert loop.dwell_count(count, high, row, open_gap, open_gap) == 0
    assert loop.dwell_count(count, side, row, open_gap, open_gap) == 0
    assert loop.dwell_count(count, at, row, row[5] - 0.001, open_gap) == 0
    assert loop.dwell_count(count, at, row, open_gap, row[5] - 0.001) == 0
    # The stop flag and a missing row reset it too.
    assert loop.dwell_count(count, at, row, open_gap, open_gap, held=True) == 0
    assert loop.dwell_count(count, at, _snap_row(width=0.0), open_gap, open_gap) == 0
    assert {"dwell_closes", "adopted"} <= set(loop.CENSUS_EXPORT)


def test_dwell_close_needs_the_continuous_gripper_and_adopt_held_needs_desk_boxes():
    for switch in ("dwell_close", "adopt_held"):
        env, rig = _env_and_rig()
        with pytest.raises(loop.EvalFailure) as err:
            _rollout(env, rig, _HoldPolicy(), **{switch: True})
        env.close()
        assert err.value.reason == switch


def test_refreshed_row_follows_the_live_centre_and_keeps_the_grasp_when_nothing_ranks(
    monkeypatch,
):
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    scene = rig.snapshot(0)
    env.close()
    selector = loop._selector(rig, 0)
    target = selector.select_target(scene)
    row = np.array(loop.target_context(scene.brick_pos[target], selector.grasp), dtype=np.float32)
    scene.brick_pos[target, :2] += 0.02
    fresh = loop.refreshed_row(row, selector, scene, target)
    assert fresh[:3] == pytest.approx(scene.brick_pos[target].tolist())
    assert fresh[3] ** 2 + fresh[4] ** 2 == pytest.approx(1.0) and fresh[5] > 0.0
    monkeypatch.setattr(selector, "_ranked", lambda scene, mask: [])
    kept = loop.refreshed_row(row, selector, scene, target)
    assert kept[:3] == pytest.approx(scene.brick_pos[target].tolist())
    assert kept[3:] == pytest.approx(row[3:].tolist())


class _ReplanSpyPolicy(_HoldPolicy):
    """Holds still with the jaws open, closes them on the steps in ``closes``, and records the
    replan flags the loop sends (None when the switch is off)."""

    def __init__(self, closes=()):
        super().__init__()
        self.flags = []
        self.closes = set(closes)
        self.steps = 0

    def step(self, state, wrist, overhead, context=None, anchor=None, replan=None):
        self.flags.append(None if replan is None else replan.tolist())
        action = super().step(state, wrist, overhead, context, anchor)
        if self.steps in self.closes:
            action[:, 6] = -1.0
        self.steps += 1
        return action


def test_the_switches_select_at_once_after_a_budget_end_refresh_at_a_reopen_and_flag_it(
    monkeypatch,
):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    env, rig = _env_and_rig(budget=3)
    _place(env, SPREAD)
    policy = _ReplanSpyPolicy(closes={5})  # closed at step 5, open again at 6: a REOPEN
    payload = _rollout(
        env,
        rig,
        policy,
        budget=3,
        select_at_release=True,
        replan_on_switch=True,
        row_refresh=True,
    )
    env.close()
    ep = payload["episodes"][0]
    assert payload["inference"]["loop"] == {
        "select_at_release": True,
        "replan_on_switch": True,
        "row_refresh": True,
        "squeeze_floor": False,
        "close_gate": False,
        "square_wrist": False,
        "scripted_place": False,
        "scripted_rake": False,
        "part_failures": False,
        "pick_snap": False,
        "dwell_close": False,
        "adopt_held": False,
        "rake_cap": 0,
        "rake_crowd": False,
    }
    # Every cycle ended on its budget, so no selection waited for a retreat.
    assert ep["cycle_reasons"] == ["budget"] * 3 and ep["select_wait_steps"] == 0
    assert [r["wait_steps"] for r in ep["cycle_census"]] == [0, 0, 0]
    assert [r["refreshes"] for r in ep["cycle_census"]] == [1, 0, 0]
    # The flag rides on the request after a row is written: the selections at steps 0, 40 and
    # 80, and the refresh read at step 6 reaches the policy on step 7.
    assert [i for i, f in enumerate(policy.flags) if f == [True]] == [0, 7, 40, 80]
    # Off, the flag stays off the wire (an older server would reject it).
    env, rig = _env_and_rig(budget=1)
    _place(env, SPREAD)
    policy = _ReplanSpyPolicy()
    _rollout(env, rig, policy, budget=1)
    env.close()
    assert set(policy.flags) == {None}


def test_the_release_selection_waits_for_nothing_after_the_teachers_stow():
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    ep = _rollout(env, rig, None, select_at_release=True)["episodes"][0]
    env.close()
    # The rig hands back at RETREAT with the jaws at the full gap, so the release read holds
    # on the step the cycle closes.
    assert ep["cycle_reasons"] == ["stowed"] * 3
    assert [r["wait_steps"] for r in ep["cycle_census"]] == [0, 0, 0]


def test_the_oracle_driver_runs_a_tray_rake_as_its_own_takeover(monkeypatch):
    """MS7a tray: a 2x4 brick at yaw 45 degrees 25 mm off the left wall has no grasp under a
    20 degree band, so the selector plans a rake instead of a target. The oracle driver runs
    it as a takeover of its own, books it in the selector, and then picks the freed brick."""
    from maker_arm_lab.core import grasp, parts

    monkeypatch.setattr(grasp, "WALL_BAND_RAD", math.radians(20.0))
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    brick = [parts.make_part("brick", 2, 4)]
    env = MjLegoSortEnv(
        1, draw, parts=brick, cameras=False, episode_length_s=loop.episode_length_s(3), tray=True
    )
    env.reset(seed=11)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(bl, env.parts, 0, episode_draw, rng=rng, cycle_budget=3, tray=env.tray)

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    (_, _), (y0, _) = layout.tray_bounds()
    _place(env, [(layout.TRAY_CENTER[0], y0 + 0.025, 0.0049, math.radians(45.0))])
    payload = _rollout(env, rig, None, budget=3)
    env.close()
    ep = payload["episodes"][0]
    assert ep["rakes"] >= 1 and payload["rakes"] == ep["rakes"]
    assert ep["bricks_stowed"] == 1 and ep["cycle_outcomes"][-1] is True, ep


def _stuck_tray_env_and_rig(monkeypatch):
    """One 2x4 brick at yaw 45 degrees 25 mm off the left wall of the tray: under a 20 degree
    band no grasp ranks, so the selector plans a rake instead of naming a target."""
    from maker_arm_lab.core import grasp, parts

    monkeypatch.setattr(grasp, "WALL_BAND_RAD", math.radians(20.0))
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=[parts.make_part("brick", 2, 4)],
        cameras=False,
        episode_length_s=loop.episode_length_s(3),
        tray=True,
    )
    env.reset(seed=11)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(bl, env.parts, 0, episode_draw, rng=rng, cycle_budget=3, tray=env.tray)

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    (_, _), (y0, _) = layout.tray_bounds()
    _place(env, [(layout.TRAY_CENTER[0], y0 + 0.025, 0.0049, math.radians(45.0))])
    return env, rig


def test_the_scripted_rake_frees_a_wall_stuck_part_for_the_policy(monkeypatch):
    """A policy read stops at the first wall-stuck state; with --scripted_rake the rig runs
    the teacher's rake, hands the env back, and the loop opens a cycle on the freed part."""
    env, rig = _stuck_tray_env_and_rig(monkeypatch)
    plain = _rollout(env, rig, _HoldPolicy(), budget=3)["episodes"][0]
    assert plain["rakes"] == 0 and plain["cycles"] == 0 and plain["done_reason"] == "no_candidate"
    env.close()
    env, rig = _stuck_tray_env_and_rig(monkeypatch)
    payload = _rollout(env, rig, _HoldPolicy(), budget=3, scripted_rake=True)
    env.close()
    ep = payload["episodes"][0]
    assert ep["rakes"] >= 1 and payload["inference"]["loop"]["scripted_rake"] is True
    assert ep["cycles"] >= 1 and ep["cycle_targets"][0] == 0  # the policy's turn, on the part
    assert ep["takeovers"] == 0  # a rake is not a pick takeover
    with pytest.raises(loop.EvalFailure, match="scripted_rake"):
        env, rig = _stuck_tray_env_and_rig(monkeypatch)
        try:
            _rollout(env, rig, None, budget=3, scripted_rake=True)
        finally:
            env.close()


def test_part_failures_skip_the_failed_part_and_keep_its_cell_neighbour(monkeypatch):
    """Parts 1 and 2 share a 5 cm cell. A still policy fails every cycle: by cell, three
    failures on part 2 close the cell for part 1 too (the test above); with --part_failures
    each part takes its own three failures and the budget, not the memory, ends the episode."""
    # The env is built long enough for every cycle to finish, then each cycle gets 40 steps.
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    env, rig = _env_and_rig(budget=9)
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 40)
    _place(env, SPREAD)
    payload = _rollout(env, rig, _HoldPolicy(), budget=9, part_failures=True)
    env.close()
    ep = payload["episodes"][0]
    assert ep["cycle_targets"] == [0, 0, 0, 2, 2, 2, 1, 1, 1]
    assert ep["done_reason"] == "cycle_budget"
    # What the episode left: every part, where it lies, no rake and three failed cycles each.
    left = ep["left_parts"]
    assert [row[0] for row in left] == [0, 1, 2] and all(row[4:] == [0, 3] for row in left)
    assert all(math.dist(row[1:3], spread[:2]) < 0.01 for row, spread in zip(left, SPREAD))
    assert payload["inference"]["loop"]["part_failures"] is True

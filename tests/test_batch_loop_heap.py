"""The batch loop's heap additions: cycle actors, negation, flags and stats (design H7, 8)."""

import argparse
import types
from typing import ClassVar

import pytest
import torch

from maker_arm_lab.core import batch_loop, oracle, parts


def _orc(phase, picks, outcomes=()):
    return types.SimpleNamespace(
        phase=phase, pick_order=list(range(picks)), cycle_outcomes=list(outcomes)
    )


def test_cycle_actor_is_the_pick_ordinal_and_a_push_shares_the_next_pick():
    assert batch_loop.cycle_actor(None) == 0
    assert batch_loop.cycle_actor(_orc(oracle.Phase.SETTLE, 0)) == 0
    assert batch_loop.cycle_actor(_orc(oracle.Phase.HOVER, 1)) == 1
    assert batch_loop.cycle_actor(_orc(oracle.Phase.RETREAT, 3)) == 3
    assert batch_loop.cycle_actor(_orc(oracle.Phase.PUSH_SWEEP, 2)) == 3
    assert batch_loop.cycle_actor(_orc(oracle.Phase.DONE, 5)) == 0
    assert batch_loop.cycle_actor(_orc(oracle.Phase.ABANDON, 5)) == 0


def test_negate_new_failures_negates_each_new_false_once():
    calls = []
    recorder = types.SimpleNamespace(negate_actor=lambda i, ordinal: calls.append((i, ordinal)))
    seen = batch_loop.negate_new_failures(recorder, 4, [True, False], 0)
    assert seen == 2 and calls == [(4, 2)]
    seen = batch_loop.negate_new_failures(recorder, 4, [True, False, False], seen)
    assert seen == 3 and calls == [(4, 2), (4, 3)]
    assert batch_loop.negate_new_failures(recorder, 4, [True, False, False], seen) == 3
    assert len(calls) == 2


def _args(**overrides):
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    args = parser.parse_args(["--out_dir", "/tmp/x"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_heap_parts_follow_the_split_and_the_layout_seed():
    assert batch_loop.heap_parts(_args()) is None
    held = batch_loop.heap_parts(_args(heap_k=5, parts_split="heldout", layout_seed=3))
    assert len(held) == 5
    heldout_names = {p.name for p in parts.split(parts.catalog(), 20260915)[1]}
    assert all(p.name in heldout_names for p in held)
    train = batch_loop.heap_parts(_args(heap_k=40, parts_split="train", layout_seed=3))
    assert not {p.name for p in train} & heldout_names
    again = batch_loop.heap_parts(_args(heap_k=40, parts_split="train", layout_seed=3))
    assert [p.name for p in again] == [p.name for p in train]
    other = batch_loop.heap_parts(_args(heap_k=40, parts_split="train", layout_seed=4))
    assert [p.name for p in other] != [p.name for p in train]


def test_heap_episode_length_scales_with_the_smaller_of_k_and_the_budget():
    assert batch_loop.heap_episode_length_s(50, 50) == 1030.0
    assert batch_loop.heap_episode_length_s(10, 50) == 230.0
    assert batch_loop.heap_episode_length_s(200, 50) == 1030.0


def test_run_meta_carries_the_heap_fields():
    args = _args(heap_k=3, parts_split="heldout", cycle_budget=7)
    draw, _bl, noise = batch_loop.build_run(args)
    chosen = parts.catalog()[:3]
    meta = batch_loop.run_meta(
        args,
        draw,
        noise,
        None,
        num_bricks=3,
        export_mode="EXPORT_ALL",
        backend="mujoco",
        task="t",
        parts=chosen,
    )
    assert meta["heap_k"] == 3 and meta["parts_split"] == "heldout" and meta["cycle_budget"] == 7
    assert meta["parts_seed"] == 20260915
    assert meta["parts"] == [p.name for p in chosen]
    assert meta["episode_length_s"] == batch_loop.heap_episode_length_s(3, 7)
    legacy = batch_loop.run_meta(
        args,
        draw,
        noise,
        None,
        num_bricks=6,
        export_mode="EXPORT_SUCCEEDED_ONLY",
        backend="mujoco",
        task="t",
    )
    assert legacy["parts"] is None and legacy["episode_length_s"] == 180.0


def test_run_stats_sums_the_heap_counters():
    episodes = [
        {
            "env": 0,
            "outcome": "time_out",
            "bricks_delivered": 4,
            "cycles": 5,
            "pushes": 1,
            "parts_outside_tray": 1,
        },
        {
            "env": 1,
            "outcome": "success",
            "bricks_delivered": 5,
            "cycles": 5,
            "pushes": 0,
            "parts_outside_tray": 0,
        },
    ]
    counters = {
        "episodes": episodes,
        "abandon_reasons": {},
        "control_steps": 100,
        "clamped_steps": 0,
        "attempts": 2,
        "wall_seconds": 10.0,
        "exported": 2,
        "motion": {},
    }
    stats = batch_loop.run_stats(counters, target_attempts=2, num_envs=2, probe=False, num_bricks=5)
    assert stats["parts_total"] == 10 and stats["parts_stowed"] == 9
    assert stats["parts_outside_tray"] == 1 and stats["cycles"] == 10 and stats["pushes"] == 1
    assert stats["cycles_per_stowed"] == pytest.approx(10 / 9)


def test_run_stats_probe_uses_the_written_count_on_a_heap_run():
    counters = {
        "episodes": [],
        "abandon_reasons": {},
        "control_steps": 10,
        "clamped_steps": 0,
        "attempts": 2,
        "wall_seconds": 10.0,
        "exported": 0,  # no episode was fully stowed
        "exported_written": 2,  # both were written under --export_mode all
        "motion": {},
    }
    heap_stats = batch_loop.run_stats(
        counters, target_attempts=2, num_envs=2, probe=True, num_bricks=5
    )
    assert heap_stats["success_rate"] == pytest.approx(1.0)
    assert heap_stats["extrapolated_50_success_wall_hours"] is not None
    # The six-brick path is unaffected: it still grades on the fully-stowed count.
    six_brick_stats = batch_loop.run_stats(
        counters, target_attempts=2, num_envs=2, probe=True, num_bricks=None
    )
    assert six_brick_stats["success_rate"] == 0.0
    assert six_brick_stats["extrapolated_50_success_wall_hours"] is None


def test_run_meta_carries_the_effective_episode_length_over_the_computed_one():
    args = _args(heap_k=3, parts_split="heldout", cycle_budget=7)
    draw, _bl, noise = batch_loop.build_run(args)
    chosen = parts.catalog()[:3]
    computed = batch_loop.heap_episode_length_s(3, 7)
    meta = batch_loop.run_meta(
        args,
        draw,
        noise,
        None,
        num_bricks=3,
        export_mode="EXPORT_ALL",
        backend="mujoco",
        task="t",
        parts=chosen,
        episode_length_s=600.0,
    )
    # --episode_length_s 600 must land in the sidecar even though it disagrees with the
    # length the heap formula would have picked for this K and budget (I1).
    assert meta["episode_length_s"] == 600.0
    assert meta["episode_length_s"] != computed


def test_heap_k_above_the_int8_cycle_brick_bound_is_rejected():
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    parser.parse_args(["--out_dir", "/tmp/x", "--heap_k", str(batch_loop.MAX_HEAP_K)])
    with pytest.raises(SystemExit):
        parser.parse_args(["--out_dir", "/tmp/x", "--heap_k", str(batch_loop.MAX_HEAP_K + 1)])


class _FakeRecorder:
    """The one method ``run_batch`` calls on a recorder in the heap branch, with the same
    flip-only-matching-frames semantics as ``mj.recorder.ShardRecorder.negate_actor``."""

    def __init__(self, num_envs: int):
        self.actors: list[list[int]] = [[] for _ in range(num_envs)]

    def record(self, i: int, actor: int) -> None:
        self.actors[i].append(actor)

    def negate_actor(self, i: int, ordinal: int) -> None:
        self.actors[i] = [-v if v == ordinal else v for v in self.actors[i]]


class _PushCapOracle:
    """A HeapOracle stand-in that sweeps three push legs, hits ``MAX_PUSHES_IN_A_ROW`` and goes
    idle -- ``pick_order`` and ``cycle_outcomes`` never gain an entry, exactly the shape the
    reviewer reproduced (three push legs under actor 1, both lists empty)."""

    # Phase AT THE START of control step k (0-indexed): three push legs, then idle for good.
    _SCHEDULE: ClassVar[dict] = {
        0: oracle.Phase.PUSH_HOVER,
        1: oracle.Phase.PUSH_DESCEND,
        2: oracle.Phase.PUSH_SWEEP,
    }

    def __init__(self):
        self.pick_order: list[int] = []
        self.cycle_outcomes: list[bool] = []
        self.phase = self._SCHEDULE[0]
        self._step = 0

    def advance(self) -> None:
        self._step += 1
        self.phase = self._SCHEDULE.get(self._step, oracle.Phase.DONE)


class _FakeRig:
    """Enough of ``OracleRig``'s surface for ``run_batch`` to drive one heap env through the
    push-cap-then-timeout scenario, without a real MuJoCo env or physics."""

    def __init__(self, env, orc):
        self.num_envs = 1
        self.env = env
        self.oracles = [orc]
        self.q_cmd = [torch.zeros(6)]

    def step(self) -> dict:
        actor = int(self.env.oracle_actor[0])
        if actor != 0:
            self.env.recorder.record(0, actor)
        self.oracles[0].advance()
        self.env.episode_length_buf[0] += 1
        buf = int(self.env.episode_length_buf[0])
        return {
            "dones": torch.tensor([buf >= self.env.max_episode_length]),
            "abandoned": torch.zeros(1, dtype=torch.bool),
        }

    def episode_stats(self, i: int) -> dict:
        return {"succeeded": False, "abandon": None, "bricks_delivered": 0}

    def rearm(self, env_ids) -> None:
        pass


class _FakeEnv:
    """Enough of ``MjLegoSortEnv``'s duck-typed surface for ``run_batch``'s heap branch."""

    def __init__(self, max_episode_length: int):
        self.parts = [object()]  # heap flag: run_batch reads `env.parts is not None`
        self.episode_length_buf = torch.zeros(1, dtype=torch.int64)
        self.max_episode_length = max_episode_length
        self.device = "cpu"
        self.recorder = _FakeRecorder(1)
        self.recorder_manager = types.SimpleNamespace(
            exported_successful_episode_count=0, exported_episode_count=1
        )
        self.oracle_cycle_brick = torch.tensor([-1])
        self.action_manager = types.SimpleNamespace(
            get_term=lambda name: types.SimpleNamespace(clamped=torch.zeros(1, dtype=torch.bool))
        )

    def close(self) -> None:
        pass

    def reset(self, env_ids) -> None:
        pass


def test_a_trailing_push_leg_with_no_following_pick_is_negated_at_timeout():
    """C1: three push legs hit the push cap (MAX_PUSHES_IN_A_ROW) and the episode goes DONE
    with no pick ever selected -- pick_order and cycle_outcomes both stay empty, so
    negate_new_failures never runs. Only run_batch's own last-step check can still catch it,
    and only if it computes the dangling ordinal instead of trusting the (by-then-idle)
    live `actor`. Without the fix this assertion fails: the three push frames stay positive."""
    env = _FakeEnv(max_episode_length=5)
    orc = _PushCapOracle()
    rig = _FakeRig(env, orc)
    counters = batch_loop.run_batch(env, rig, target_attempts=1, max_wall_minutes=5.0)
    assert orc.pick_order == [] and orc.cycle_outcomes == []
    assert env.recorder.actors[0] == [-1, -1, -1]
    assert counters["attempts"] == 1
    assert counters["episodes"][0]["outcome"] == "time_out"

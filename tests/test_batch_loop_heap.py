"""The batch loop's heap additions: cycle actors, negation, flags and stats (design H7, 8)."""

import argparse
import types

import pytest

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

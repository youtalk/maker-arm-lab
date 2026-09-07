import pytest
import torch

from maker_arm_lab.core.checker import PINCH_MIN_FORCE
from maker_arm_lab.core.gate import (
    EpisodeResult,
    RolloutTracker,
    clamp_arm_actions,
    grade_grasp_attempts,
)

F, T = False, True

# The loose (attribution) threshold the eval script passes in; the strict one is
# PINCH_MIN_FORCE and is not the caller's to choose.
LOOSE = 1.0


def _bool(rows):
    return torch.tensor(rows, dtype=torch.bool)


def _over(rows):
    """(N,6) signed overshoot; `rows` gives each env's overshoot on joint 0."""
    t = torch.zeros(len(rows), 6)
    for i, v in enumerate(rows):
        t[i, 0] = v
    return t


def test_success_finalizes_at_num_bricks_regardless_of_latch():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.latch_pre_step(_bool([[T, F, F, F, F, F], [F, F, F, F, F, F]]))
    tr.record_step(9, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([T, F]))
    r = tr.results[0]
    assert r == EpisodeResult(
        env_index=0, success=True, bricks_stowed=6, num_bricks=6, steps=10, clamp_steps=0
    )
    assert tr.results[1] is None and not tr.all_done


def test_timeout_finalizes_from_the_latch():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    tr.latch_pre_step(_bool([[T, T, T, F, F, F]]))
    tr.record_step(5399, terminated=_bool([F]), truncated=_bool([T]), success=_bool([F]))
    r = tr.results[0]
    assert (r.success, r.bricks_stowed, r.steps) == (False, 3, 5400)
    assert tr.all_done


def test_done_envs_stop_latching_and_counting_clamps():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_clamps(_over([0.2, 0.2]))
    tr.record_step(0, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([F, F]))
    # env 0 is done; later latches/clamps must not touch its result
    tr.latch_pre_step(_bool([[T, T, T, T, T, T], [F, F, F, F, F, F]]))
    tr.record_clamps(_over([0.2, 0.2]))
    tr.record_step(1, terminated=_bool([F, T]), truncated=_bool([F, F]), success=_bool([F, F]))
    assert tr.results[0].bricks_stowed == 0 and tr.results[0].clamp_steps == 1
    assert tr.results[1].clamp_steps == 2
    assert tr.all_done


def test_double_done_is_ignored():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    tr.record_step(3, terminated=_bool([T]), truncated=_bool([F]), success=_bool([T]))
    tr.record_step(7, terminated=_bool([T]), truncated=_bool([F]), success=_bool([F]))
    assert tr.results[0].success is True and tr.results[0].steps == 4


def test_gripper_duty_cycle_accumulates_and_finalizes():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    for g in (1.0, 1.0, -1.0, 0.2, -1.0):
        tr.record_gripper(torch.tensor([g]))
    tr.record_step(4, terminated=_bool([F]), truncated=_bool([T]), success=_bool([F]))
    r = tr.results[0]
    # open (g >= 0): +1, +1, 0.2 -> 3; flips: +1->-1, -1->0.2, 0.2->-1 -> 3
    # (the first step has no predecessor); band (|g| < 0.5): 0.2 -> 1
    assert (r.gripper_open_steps, r.gripper_transitions, r.gripper_band_steps) == (3, 3, 1)


def test_gripper_stats_freeze_for_done_envs():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_gripper(torch.tensor([1.0, 1.0]))
    tr.record_step(0, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([F, F]))
    # env 0 is done; the second sample must not add counts or flips to it
    tr.record_gripper(torch.tensor([-1.0, -1.0]))
    tr.record_step(1, terminated=_bool([F, T]), truncated=_bool([F, F]), success=_bool([F, F]))
    assert (tr.results[0].gripper_open_steps, tr.results[0].gripper_transitions) == (1, 0)
    assert (tr.results[1].gripper_open_steps, tr.results[1].gripper_transitions) == (1, 1)


def test_zero_gripper_counts_as_open():
    # same convention as clamp_arm_actions' docstring and the env: g >= 0 opens
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    tr.record_gripper(torch.tensor([0.0]))
    tr.record_step(0, terminated=_bool([F]), truncated=_bool([T]), success=_bool([F]))
    assert tr.results[0].gripper_open_steps == 1


def test_to_json_round_trips_plain_types():
    r = EpisodeResult(
        env_index=1,
        success=False,
        bricks_stowed=2,
        num_bricks=6,
        steps=100,
        clamp_steps=3,
        gripper_open_steps=61,
        gripper_transitions=12,
        gripper_band_steps=4,
    )
    d = r.to_json()
    assert d == {
        "env_index": 1,
        "success": False,
        "bricks_stowed": 2,
        "num_bricks": 6,
        "steps": 100,
        "clamp_steps": 3,
        "gripper_open_steps": 61,
        "gripper_transitions": 12,
        "gripper_band_steps": 4,
        "first_close_step": -1,
        "wrong_bin_deposits": 0,
        "grasp_attempts": 0,
        "grasp_successes": 0,
        "grasp_successes_strict": 0,
        "grasp_attempts_unresolved": 0,
        "clamp_steps_per_joint": [0] * 6,
        "clamp_overshoot_max_per_joint": [0.0] * 6,
    }
    scalars = {k: v for k, v in d.items() if not isinstance(v, list)}
    assert all(type(v) in (int, bool) for v in scalars.values())
    assert all(type(v) is int for v in d["clamp_steps_per_joint"])
    assert all(type(v) is float for v in d["clamp_overshoot_max_per_joint"])


def test_first_close_step_set_once_on_the_first_close():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    # env 0 closes at call index 2; env 1 never closes.
    for g in ([1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]):
        tr.record_gripper(torch.tensor(g))
    tr.record_step(2, terminated=_bool([F, F]), truncated=_bool([T, T]), success=_bool([F, F]))
    assert tr.results[0].first_close_step == 2
    assert tr.results[1].first_close_step == -1
    assert tr.results[1].gripper_transitions == 0


def test_first_close_step_is_not_overwritten_by_later_closes():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    for g in (-1.0, 1.0, -1.0):  # closes at call 0, reopens, closes again at call 2
        tr.record_gripper(torch.tensor([g]))
    tr.record_step(2, terminated=_bool([T]), truncated=_bool([F]), success=_bool([F]))
    assert tr.results[0].first_close_step == 0


def test_first_close_step_freezes_for_done_envs():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_gripper(torch.tensor([1.0, 1.0]))
    tr.record_step(0, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([F, F]))
    # env 0 is done; a later close must not retroactively set its first_close_step.
    # Asserted on the internal tensor right after the write, not just on the
    # eventually-finalized result: env 0's EpisodeResult is built once (at step
    # 0, above) and never rebuilt, so a later out-of-band write to
    # `_first_close[0]` would be invisible through `tr.results` regardless of
    # whether `record_gripper`'s active-env guard actually fired.
    tr.record_gripper(torch.tensor([-1.0, -1.0]))
    assert tr._first_close[0].item() == -1
    tr.record_step(1, terminated=_bool([F, T]), truncated=_bool([F, F]), success=_bool([F, F]))
    assert tr.results[0].first_close_step == -1
    assert tr.results[1].first_close_step == 1


def test_record_wrong_bin_accumulates_per_env_across_calls():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_wrong_bin(torch.tensor([0]))
    tr.record_wrong_bin(torch.tensor([0]))
    tr.record_step(0, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([F, F]))
    tr.record_step(1, terminated=_bool([F, F]), truncated=_bool([F, T]), success=_bool([F, F]))
    assert tr.results[0].wrong_bin_deposits == 2
    assert tr.results[1].wrong_bin_deposits == 0


def test_record_wrong_bin_handles_duplicate_ids_in_one_call():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_wrong_bin(torch.tensor([0, 0, 1]))
    tr.record_step(0, terminated=_bool([T, T]), truncated=_bool([F, F]), success=_bool([F, F]))
    assert tr.results[0].wrong_bin_deposits == 2
    assert tr.results[1].wrong_bin_deposits == 1


def test_grasp_outcome_splits_graded_attempts_into_successes_and_failures():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0, 1]))
    tr.record_grasp_outcome(torch.tensor([0, 1]), _bool([T, F]), _bool([T, F]))
    tr.record_step(0, terminated=_bool([T, T]), truncated=_bool([F, F]), success=_bool([F, F]))

    held, empty = tr.results[0], tr.results[1]
    assert (held.grasp_attempts, held.grasp_successes, held.grasp_attempts_unresolved) == (1, 1, 0)
    assert (empty.grasp_attempts, empty.grasp_successes, empty.grasp_attempts_unresolved) == (
        1,
        0,
        0,
    )
    assert (held.grasp_successes_strict, empty.grasp_successes_strict) == (1, 0)


def test_a_marginal_pinch_is_a_success_only_at_the_loose_threshold():
    """The threshold-sensitivity read: the two masks disagree on brushing contacts.

    A contact over the loose attribution threshold but under the quality pinch
    threshold is exactly the population a "most gripper cycles grip nothing"
    reading is about, so the strict counter must be able to reject an attempt
    the loose one accepts. (openyam-lab put that at 55 % of cycles -- an OPENYAM
    measurement, quoted for why the pair exists and unmeasured on this arm.)
    Both directions are asserted in one test on purpose: the marginal env alone
    passes against a strict counter wired to a constant zero, and the firm env
    alone passes against one that just reuses `held`.
    """
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0, 1]))
    # env 0 brushed the brick, env 1 has it properly pinched.
    tr.record_grasp_outcome(torch.tensor([0, 1]), _bool([T, T]), _bool([F, T]))
    tr.record_step(0, terminated=_bool([T, T]), truncated=_bool([F, F]), success=_bool([F, F]))

    marginal, firm = tr.results[0], tr.results[1]
    assert (marginal.grasp_attempts, marginal.grasp_successes) == (1, 1)
    assert marginal.grasp_successes_strict == 0
    assert (firm.grasp_successes, firm.grasp_successes_strict) == (1, 1)


def test_ungraded_grasp_attempt_is_reported_unresolved_not_as_a_failure():
    """The wrong-bin counter's structural hole, refused here.

    An attempt whose settle window has not closed when the episode ends was
    neither observed to hold a brick nor observed not to. Counting it in
    `grasp_attempts` would put an ungradeable attempt in the denominator of
    `grasp_successes / grasp_attempts` and bias the rate down -- exactly the
    defect openyam-lab found in its own wrong_bin_deposits (14 of 90 deposits
    structurally ungraded, concentrated on the last brick of every successful
    episode). Those are OPENYAM numbers, carried over as the rationale for this
    counter and unmeasured on the Maker Arm.
    """
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0]))  # graded GRASP_SETTLE_STEPS later
    tr.record_grasp_attempt(torch.tensor([0]))
    tr.record_grasp_outcome(torch.tensor([0]), _bool([T]), _bool([T]))  # only one comes due
    tr.record_step(0, terminated=_bool([F]), truncated=_bool([T]), success=_bool([F]))

    r = tr.results[0]
    assert (r.grasp_attempts, r.grasp_successes) == (1, 1)
    assert r.grasp_attempts_unresolved == 1


def test_grasp_counters_accumulate_duplicate_env_ids_within_one_call():
    # Two attempts of the same env coming due on the same step is the ordinary
    # case for a chattering gripper, and a fancy-index += would keep only the
    # last write.
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0, 0, 1]))
    tr.record_grasp_outcome(torch.tensor([0, 0]), _bool([T, F]), _bool([T, F]))
    tr.record_step(0, terminated=_bool([T, T]), truncated=_bool([F, F]), success=_bool([F, F]))

    assert (tr.results[0].grasp_attempts, tr.results[0].grasp_successes) == (2, 1)
    assert tr.results[0].grasp_successes_strict == 1
    assert tr.results[0].grasp_attempts_unresolved == 0
    assert (tr.results[1].grasp_attempts, tr.results[1].grasp_attempts_unresolved) == (0, 1)


def test_grasp_attempt_on_the_terminating_step_reaches_that_episodes_result():
    """The eval loop's ordering contract: record_* before record_step.

    record_step finalizes each env's EpisodeResult exactly once, on the step it
    terminates. An attempt opened on that same step is only in the result
    because the tracker does not gate these two on ~done -- so this pins that
    the un-gated write is what makes the same-step event land.
    """
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0]))
    tr.record_grasp_outcome(torch.tensor([0]), _bool([T]), _bool([T]))
    tr.record_grasp_attempt(torch.tensor([0]))  # opened on the terminating step
    tr.record_step(7, terminated=_bool([T]), truncated=_bool([F]), success=_bool([T]))

    r = tr.results[0]
    assert (r.grasp_attempts, r.grasp_successes, r.grasp_attempts_unresolved) == (1, 1, 1)


def test_grasp_records_after_termination_cannot_change_a_finalized_result():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    tr.record_grasp_attempt(torch.tensor([0, 1]))
    tr.record_step(0, terminated=_bool([T, F]), truncated=_bool([F, F]), success=_bool([F, F]))
    # env 0 finished; a queue entry outliving its episode still fires (the eval
    # loop keeps stepping finished envs through auto-reset second episodes), and
    # must not retroactively grade env 0's attempt.
    tr.record_grasp_outcome(torch.tensor([0, 1]), _bool([T, T]), _bool([T, T]))
    tr.record_step(1, terminated=_bool([F, T]), truncated=_bool([F, F]), success=_bool([F, F]))

    assert tr.results[0].grasp_attempts_unresolved == 1
    assert (tr.results[0].grasp_attempts, tr.results[0].grasp_successes) == (0, 0)
    assert (tr.results[1].grasp_attempts, tr.results[1].grasp_successes) == (1, 1)
    assert tr.results[1].grasp_attempts_unresolved == 0


def test_record_grasp_outcome_rejects_a_mismatched_held_mask():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    with pytest.raises(ValueError):
        tr.record_grasp_outcome(torch.tensor([0, 1]), _bool([T]), _bool([T, T]))
    with pytest.raises(ValueError):
        tr.record_grasp_outcome(torch.tensor([0, 1]), _bool([T, T]), _bool([T]))


def _mags(per_env_force):
    """(N,1) left/right jaw magnitudes, one brick, `force` newtons on both jaws."""
    col = torch.tensor(per_env_force, dtype=torch.float32).unsqueeze(1)
    return col, col.clone()


def test_grade_grasp_attempts_separates_the_loose_and_strict_thresholds():
    """The whole point of the pair: they must disagree on marginal contacts.

    1.5 N is over the attribution threshold and under the quality one, which is
    the population "gripper cycles that grip nothing" is about. A helper that
    scored both sides at the same threshold -- the one mutation that makes
    grasp_success_rate_strict a silent copy of grasp_success_rate -- reports the
    two masks equal here.
    """
    left, right = _mags([1.5, 5.0, 0.5])

    loose, strict = grade_grasp_attempts(left, right, torch.tensor([0, 1, 2]), LOOSE)

    assert loose.tolist() == [T, T, F]
    assert strict.tolist() == [F, T, F]


def test_grade_grasp_attempts_scores_the_weaker_jaw():
    # A brick brushing one jaw hard and the other barely is not a pinch, so the
    # two magnitude arguments are not interchangeable and neither may be used
    # twice.
    left = torch.tensor([[5.0], [5.0]])
    right = torch.tensor([[5.0], [1.5]])

    loose, strict = grade_grasp_attempts(left, right, torch.tensor([0, 1]), LOOSE)

    assert loose.tolist() == [T, T]
    assert strict.tolist() == [T, F]


def test_grade_grasp_attempts_returns_one_entry_per_attempt_in_env_ids_order():
    # env_ids comes from the pending list: unsorted, and repeatable when one env
    # somehow has two attempts due on the same step. The masks pair positionally
    # with it, since record_grasp_outcome grades one attempt per entry.
    left, right = _mags([0.5, 5.0, 1.5])

    loose, strict = grade_grasp_attempts(left, right, torch.tensor([2, 0, 1, 1]), LOOSE)

    assert loose.tolist() == [T, F, T, T]
    assert strict.tolist() == [F, F, T, T]


def test_grade_grasp_attempts_refuses_a_loose_threshold_that_collapses_the_pair():
    """A caller passing the strict threshold as the loose one gets an error, not a no-op.

    Equal thresholds make grasp_successes_strict a copy of grasp_successes, and
    a silently-identical second metric is worse than no second metric.
    """
    left, right = _mags([5.0])
    for bad in (PINCH_MIN_FORCE, PINCH_MIN_FORCE + 1.0):
        with pytest.raises(ValueError):
            grade_grasp_attempts(left, right, torch.tensor([0]), bad)


def test_grade_grasp_attempts_never_reports_a_strict_success_that_is_not_a_loose_one():
    """`held_brick_from_forces` maxes before thresholding, so strict implies loose.

    grasp_successes_strict <= grasp_successes rests on this, and it holds across
    every combination of jaw forces rather than by convention.
    """
    forces = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 8.7]
    left = torch.tensor([[a] for a in forces for _ in forces])
    right = torch.tensor([[b] for _ in forces for b in forces])
    env_ids = torch.arange(len(forces) ** 2)

    loose, strict = grade_grasp_attempts(left, right, env_ids, LOOSE)

    assert not bool((strict & ~loose).any())
    assert bool(strict.any()) and bool((loose & ~strict).any())


def test_clamp_arm_actions_clamps_and_flags():
    limits = torch.tensor([[[-1.0, 1.0]] * 6, [[-1.0, 1.0]] * 6])
    actions = torch.zeros(2, 6)
    actions[0, 2] = 1.5
    clamped, overshoot = clamp_arm_actions(actions, limits)
    assert clamped[0, 2] == 1.0 and clamped[1].abs().sum() == 0
    assert (overshoot != 0).any(dim=-1).tolist() == [True, False]


def test_clamp_arm_actions_rejects_bad_shapes():
    with pytest.raises(ValueError):
        clamp_arm_actions(torch.zeros(2, 7), torch.zeros(2, 6, 2))


def test_clamp_arm_actions_reports_signed_per_joint_overshoot():
    limits = torch.tensor([[[-1.0, 1.0]] * 6, [[-1.0, 1.0]] * 6])
    actions = torch.zeros(2, 6)
    actions[0, 2] = 1.5  # 0.5 past the upper limit
    actions[1, 4] = -1.25  # 0.25 past the lower limit

    clamped, overshoot = clamp_arm_actions(actions, limits)

    assert clamped[0, 2] == 1.0 and clamped[1, 4] == -1.0
    assert overshoot.shape == (2, 6)
    assert overshoot[0, 2] == pytest.approx(0.5)
    assert overshoot[1, 4] == pytest.approx(-0.25)
    # every other dim is exactly zero -- inside the limits is not "a small overshoot"
    assert overshoot[0].abs().sum() == pytest.approx(0.5)
    assert overshoot[1].abs().sum() == pytest.approx(0.25)


def test_tracker_counts_clamped_steps_per_joint():
    tr = RolloutTracker(num_envs=2, num_bricks=6)
    over = torch.zeros(2, 6)
    over[0, 1] = 0.3
    tr.record_clamps(over)
    over = torch.zeros(2, 6)
    over[0, 1] = 0.1
    over[0, 5] = -0.2
    tr.record_clamps(over)

    tr.record_step(1, _bool([T, T]), _bool([F, F]), _bool([F, F]))

    assert tr.results[0].clamp_steps_per_joint == [0, 2, 0, 0, 0, 1]
    assert tr.results[0].clamp_steps == 2  # still "any joint clamped this step"
    assert tr.results[1].clamp_steps_per_joint == [0] * 6


def test_tracker_keeps_the_worst_overshoot_per_joint():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    over = torch.zeros(1, 6)
    over[0, 1] = 0.3
    tr.record_clamps(over)
    over = torch.zeros(1, 6)
    over[0, 1] = -0.9  # larger in magnitude, opposite limit
    tr.record_clamps(over)

    tr.record_step(1, _bool([T]), _bool([F]), _bool([F]))

    worst = tr.results[0].clamp_overshoot_max_per_joint
    assert worst[1] == pytest.approx(0.9)
    assert worst[0] == pytest.approx(0.0)


def test_episode_result_json_carries_the_per_joint_blocks():
    tr = RolloutTracker(num_envs=1, num_bricks=6)
    over = torch.zeros(1, 6)
    over[0, 3] = 0.4
    tr.record_clamps(over)
    tr.record_step(0, _bool([T]), _bool([F]), _bool([F]))

    d = tr.results[0].to_json()

    assert d["clamp_steps_per_joint"] == [0, 0, 0, 1, 0, 0]
    assert d["clamp_overshoot_max_per_joint"][3] == pytest.approx(0.4)
    assert len(d["clamp_overshoot_max_per_joint"]) == 6

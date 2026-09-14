"""The GRASP and ORACLE parity checks of the MS3 design (section 3), as one CPU test.

The scripted oracle -- the same ``OracleRig`` that generated the MS1 and MS2 corpora on Isaac
-- runs a two-brick episode on the MuJoCo env with cameras off and must stow both bricks: a
real pinch at the measured stall (PINCH_MIN_STALL, PINCH_MIN_FORCE), a lift, a carry, a
release into the right bin, twice. A few seconds on one core. The six-brick episode is the
same code path six times over and is left to ``scripts/mj_generate_batch.py --probe``.
"""

import random

import torch

from maker_arm_lab.core import oracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


def test_oracle_sorts_two_bricks_on_mujoco():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    pinches = []
    steps = 0
    last_lift_pinch = None
    while steps < 4000:
        goal = rig.goals[0]
        in_lift = goal is not None and goal.phase is oracle.Phase.LIFT
        if in_lift and not rig.plans[0]:
            # The lift's walk is done and the brick is in free air: what the rig's own LIFT
            # gate reads on the phase's last step is the last of these samples.
            last_lift_pinch = rig.pinch_evidence(0, rig.oracles[0].target_brick)
        out = rig.step()
        steps += 1
        after = rig.goals[0]
        if in_lift and (after is None or after.phase is not oracle.Phase.LIFT):
            pinches.append(last_lift_pinch)
            last_lift_pinch = None
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    stats = rig.episode_stats(0)
    env.close()
    assert stats["succeeded"] is True, stats
    assert stats["bricks_delivered"] == 2 and stats["retries"] == 0
    assert stats["pick_order"] == [0, 1]
    assert len(pinches) == 2 and all(p["ok"] for p in pinches), pinches
    # The grip Isaac measured was 7.4-7.7 mm of stall at 7.6 N; MuJoCo's contact model lands
    # a little softer (design section 3 records the tuning). Anything under the floors would
    # have failed `ok` above; this pins the magnitude so a silent contact regression shows.
    assert all(0.0055 <= p["stall"] <= 0.0085 for p in pinches), pinches
    assert all(min(p["left"], p["right"]) >= 4.0 for p in pinches), pinches
    assert steps < 2000, steps  # two bricks in well under half the six-brick budget


def _run(rig, steps):
    trace = []
    for _ in range(steps):
        rig.step()
        goal = rig.goals[0]
        trace.append((None if goal is None else goal.phase.name, rig.q_cmd[0].clone()))
    return trace


def test_step_is_actions_then_env_step_then_observe():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    whole = _run(rig, 300)
    env.close()
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    split = []
    for _ in range(300):
        action = rig.actions()
        _, _, terminated, truncated, _ = env.step(action)
        out = rig.observe((terminated | truncated).cpu())
        assert set(out) == {"dones", "abandoned"}
        goal = rig.goals[0]
        split.append((None if goal is None else goal.phase.name, rig.q_cmd[0].clone()))
    env.close()
    assert [p for p, _ in whole] == [p for p, _ in split]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(whole, split))


def test_observe_does_not_touch_the_actions_snapshot():
    """``actions()`` freezes ``_stowed_sample`` and the cycle stash before ``env.step()``.

    Neither value may change again until the next call to ``actions()`` -- that is the whole
    point of sampling them before the step rather than after (see the comments carried over
    under Ruling P15). ``env.step()`` auto-resets a finished env, and ``observe()`` advances
    phases, so if either value were instead read inside ``observe()`` it would silently pick
    up the next episode's scatter or the post-transition phase.

    The run goes all the way to the episode's last step, on purpose: for most of a run,
    nothing about brick poses or the phase changes within a single physics step, so a
    snapshot taken before ``env.step()`` matches a fresh read taken after it even when the
    fresh read runs too late. The one step that always differs is the last one, where
    ``env.step()`` auto-resets a finished env: a fresh post-reset read sees the next
    episode's scatter and phase -1, not the episode that just finished.
    """
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    steps = 0
    while steps < 4000:
        action = rig.actions()
        stowed_snapshot = rig._stowed_sample.clone()
        phase_snapshot = env.oracle_cycle_phase.clone()
        brick_snapshot = env.oracle_cycle_brick.clone()
        _, _, terminated, truncated, _ = env.step(action)
        out = rig.observe((terminated | truncated).cpu())
        steps += 1
        assert torch.equal(rig._stowed_sample, stowed_snapshot)
        assert torch.equal(env.oracle_cycle_phase, phase_snapshot)
        assert torch.equal(env.oracle_cycle_brick, brick_snapshot)
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    assert steps < 4000, steps  # the episode actually ended, so the reset step ran
    env.close()


def test_an_idle_env_coasts_and_stashes_minus_one():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(2, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    rig.rearm([0], idle=True)
    assert rig.idle == [True, False] and rig.goals[0] is None and rig.oracles[0] is None
    q0 = rig.q_arm(0).clone()
    for _ in range(100):
        action = rig.actions()
        assert float(action[0, 6]) == 1.0
        assert torch.allclose(action[0, :6].double(), q0, atol=1e-6)
        assert int(env.oracle_cycle_phase[0]) == -1 and int(env.oracle_cycle_brick[0]) == -1
        _, _, terminated, truncated, _ = env.step(action)
        rig.observe((terminated | truncated).cpu())
    assert rig.goals[0] is None and not bool(rig.abandoned[0])
    assert rig.goals[1] is not None  # env 1 ran its oracle meanwhile
    assert torch.allclose(rig.q_arm(0), q0, atol=0.02)
    env.close()


def test_rearm_reads_each_envs_own_bin_layout():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(2, draw, num_bricks=2, cameras=False, episode_draws=True)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    rig.rearm([0, 1])
    assert rig.oracles[0].bl == env.bin_layouts[0]
    assert rig.oracles[1].bl == env.bin_layouts[1]
    assert rig.oracles[0].bl != rig.oracles[1].bl
    env.close()


def test_oracle_sorts_two_bricks_under_episode_draws():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False, episode_draws=True)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    steps = 0
    while steps < 4000:
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    stats = rig.episode_stats(0)
    env.close()
    assert stats["succeeded"] is True, stats


def _until_takeover_ends(rig, env, budget):
    for _ in range(budget):
        out = rig.step()
        if rig.takeover_result[0] is not None or bool(out["dones"][0]):
            return out
    raise AssertionError("the takeover did not end")


def test_resume_without_a_pinch_hovers_and_stows_one_brick_then_idles():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    rig.rearm([0], idle=True)
    for _ in range(30):
        rig.step()
    assert rig.resume(0) is True
    assert rig.idle[0] is False and rig.goals[0].phase is oracle.Phase.HOVER
    assert rig.single_cycle[0] and rig.takeover_result[0] is None
    out = _until_takeover_ends(rig, env, 2000)
    assert rig.takeover_result[0] is True and rig.idle[0] and rig.goals[0] is None
    assert not bool(out["dones"][0])  # one brick of two: the episode goes on
    assert int(per_brick_stowed(env)[0].sum()) == 1
    assert rig.resume(0) is True
    out = _until_takeover_ends(rig, env, 2000)
    assert bool(out["dones"][0]) and rig.episode_stats(0)["succeeded"] is True
    env.close()


def test_resume_after_a_drop_hovers_again():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    for _ in range(4000):
        rig.step()
        goal = rig.goals[0]
        if goal is not None and goal.phase is oracle.Phase.LIFT and not rig.plans[0]:
            break
    else:
        raise AssertionError("never reached a walked LIFT")
    held = rig.oracles[0].target_brick
    assert rig.pinch_evidence(0, held)["ok"]
    rig._go_idle(0, None)
    for _ in range(10):
        rig.step()  # the coast row keeps the jaws OPEN, so the brick drops back to the desk
    assert rig.resume(0) is True
    assert rig.goals[0].phase is oracle.Phase.HOVER  # dropped brick: nothing pinched now
    env.close()


def test_resume_from_a_pinch_that_holds_enters_lift():
    """The brick is still UNDER LIFT_HEIGHT here -- a pinch the policy has closed on but not
    yet raised, which is the only carry a takeover may enter LIFT from. The carry above the
    clearance is ruling P32's decline; see its own test below."""
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    for _ in range(4000):
        rig.step()
        goal, brick = rig.goals[0], rig.oracles[0].target_brick
        if (
            goal is not None
            and goal.phase is oracle.Phase.LIFT
            and brick is not None
            and rig.pinch_evidence(0, brick)["ok"]
            and float(rig._brick_pos(0, brick)[2]) <= oracle.LIFT_HEIGHT
        ):
            break
    else:
        raise AssertionError("never reached a held LIFT below LIFT_HEIGHT")
    held = brick
    rig.idle[0], rig.goals[0], rig.oracles[0] = True, None, None
    assert rig.resume(0) is True
    assert rig.goals[0].phase is oracle.Phase.LIFT and rig.oracles[0].target_brick == held
    out = _until_takeover_ends(rig, env, 2000)
    assert rig.takeover_result[0] is True and not bool(out["dones"][0])
    env.close()


def test_resume_declines_a_brick_carried_above_lift_height():
    """RULING P32. The ``random`` and ``workspace`` triggers are evaluated without reading
    ``held``, so a takeover can be offered while the policy is carrying a brick aloft. The
    LIFT entry ``resume`` would take reads its target as an absolute clearance of LIFT_HEIGHT
    above the desk, so from a carry at 0.2 m it plans a walk DOWN with the jaws closed, fails
    LIFT's rise gate against the live ``brick_start``, and drops the brick in REOPEN -- after
    which the retry re-picks it, the cycle scores as stowed, and every frame of the descent
    and the deliberate release is kept in the corpus. The takeover must be declined instead:
    the env stays idle, nothing is scored, and a later trigger catches it once the brick is
    back on the desk.
    """
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=2, cameras=False)
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    for _ in range(4000):
        rig.step()
        goal, carried = rig.goals[0], rig.oracles[0].target_brick
        if (
            goal is not None
            and goal.phase in (oracle.Phase.RAISE, oracle.Phase.EXTEND)
            and carried is not None
            and float(rig._brick_pos(0, carried)[2]) > oracle.LIFT_HEIGHT
        ):
            break
    else:
        raise AssertionError("the oracle never carried a brick above LIFT_HEIGHT")
    rig.idle[0], rig.goals[0] = True, None
    assert rig.resume(0) is False
    assert rig.idle[0] is True and rig.goals[0] is None  # never entered LIFT
    assert rig.single_cycle[0] is False
    assert rig.takeover_result[0] is None  # nothing to score: the takeover never started
    env.close()

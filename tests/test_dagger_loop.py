"""The DAgger triggers on scalars, and one mixed rollout on MuJoCo with a policy that holds
still: the no-close trigger hands each brick to the oracle, and only takeover frames land in
the shard."""

import random

import h5py
import numpy as np

from maker_arm_lab.core import dagger_loop as dl
from maker_arm_lab.core import oracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig


def _reason(**overrides):
    base = {
        "grip_open": True,
        "closed_now": False,
        "held": -1,
        "nearest_cm": 10.0,
        "pick_point": (0.25, 0.0, 0.2),
        "since_open": 0,
        "closed_no_pinch": 0,
        "random_draw": 1.0,
        "random_p": 0.0,
    }
    return dl.trigger_reason(**{**base, **overrides})


def test_each_trigger_fires_on_its_condition_only():
    assert _reason() is None
    assert _reason(random_draw=0.001, random_p=1 / 300) == "random"
    assert _reason(since_open=dl.NO_CLOSE_STEPS) == "no_close"
    assert _reason(since_open=dl.NO_CLOSE_STEPS - 1) is None
    assert _reason(grip_open=False, closed_now=True, nearest_cm=2.5) == "close_far"
    assert _reason(grip_open=False, closed_now=True, nearest_cm=-1.0) == "close_far"
    assert _reason(grip_open=False, closed_now=True, nearest_cm=1.5) is None
    assert _reason(grip_open=False, closed_now=True, nearest_cm=9.0, held=0) is None
    assert _reason(grip_open=False, closed_no_pinch=dl.CLOSED_NO_PINCH_STEPS) == "closed_no_pinch"
    assert _reason(grip_open=False, closed_no_pinch=dl.CLOSED_NO_PINCH_STEPS, held=1) is None
    assert _reason(pick_point=(0.6, 0.0, 0.2)) == "workspace"
    assert _reason(pick_point=(0.2, 0.0, -0.1)) == "workspace"
    # Priority pins: a row earlier in trigger_reason's if-chain must win when two rows'
    # conditions are both true at once, so swapping two rows would be caught rather than
    # passing silently (every case above isolates a single condition at a time).
    assert _reason(random_draw=0.001, random_p=1 / 300, since_open=dl.NO_CLOSE_STEPS) == "random"
    assert _reason(since_open=dl.NO_CLOSE_STEPS, pick_point=(0.6, 0.0, 0.2)) == "no_close"
    assert (
        _reason(
            grip_open=False,
            closed_now=True,
            nearest_cm=9.0,
            closed_no_pinch=dl.CLOSED_NO_PINCH_STEPS,
        )
        == "close_far"
    )
    # closed_no_pinch and workspace are not mutually exclusive: an env can sit closed with
    # nothing pinched while its pick point is also outside the box, so this pair needs its own
    # pin (unlike the pairs above, where one row's condition rules the other row's condition
    # out).
    assert (
        _reason(
            grip_open=False,
            closed_no_pinch=dl.CLOSED_NO_PINCH_STEPS,
            pick_point=(0.6, 0.0, 0.2),
        )
        == "closed_no_pinch"
    )
    # close_far and workspace are not mutually exclusive either, for the same reason.
    assert (
        _reason(
            grip_open=False,
            closed_now=True,
            nearest_cm=9.0,
            pick_point=(0.6, 0.0, 0.2),
        )
        == "close_far"
    )


class _HoldPolicy:
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


def test_a_still_policy_is_taken_over_twice_and_only_takeovers_are_recorded(tmp_path):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, num_bricks=2, cameras=False, export_dir=tmp_path, shard_name="d", export_all=True
    )
    env.reset(seed=11)
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    payload = dl.rollout(
        env,
        rig,
        _HoldPolicy(),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        random_interval=None,
        max_wall_minutes=5.0,
    )
    env.close()
    assert payload["takeovers"] == 2 and payload["takeovers_stowed"] == 2
    assert payload["takeovers_failed"] == 0 and payload["triggers"] == {"no_close": 2}
    assert payload["episodes"][0]["success"] is True
    assert payload["random_interval"] is None and payload["policy_id"] == "fake:hold"
    with h5py.File(tmp_path / "d.hdf5", "r") as f:
        demo = f["data/demo_0"]
        actor = demo["obs/actor"][:]
        assert set(actor.tolist()) == {1, 2}
        assert len(actor) < payload["steps_run"] - 2 * dl.NO_CLOSE_STEPS + 10
        assert bool(demo.attrs["success"])


def test_a_time_out_mid_takeover_leaves_no_positive_actor_frame(tmp_path):
    """Ruling P7: the time-out negation must also zero ``env.oracle_actor`` so the very step
    that times out is not recorded at all. Negating alone is not enough -- the env's own
    ``step()`` call still runs after the negation and would stash one more frame under the
    (unchanged) positive ordinal if the actor were left as-is."""
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, num_bricks=2, cameras=False, export_dir=tmp_path, shard_name="t", export_all=True
    )
    env.reset(seed=11)
    # NO_CLOSE_STEPS to trigger the first takeover, plus a short window: nowhere near enough
    # for a full pick-and-place cycle to reach RETREAT, so the time-out lands mid-cycle.
    env.max_episode_length = dl.NO_CLOSE_STEPS + 15
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    payload = dl.rollout(
        env,
        rig,
        _HoldPolicy(),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        random_interval=None,
        max_wall_minutes=5.0,
    )
    env.close()
    assert payload["takeovers"] == 1
    with h5py.File(tmp_path / "t.hdf5", "r") as f:
        demo = f["data/demo_0"]
        actor = demo["obs/actor"][:]
        assert len(actor) > 0
        # The negated frames of the running takeover must show up as -1 (proving negate_actor
        # ran), and none may still be positive (proving the same-step frame was dropped, not
        # just negated one step late).
        assert -1 in actor.tolist()
        assert not any(v > 0 for v in actor.tolist())
        assert not bool(demo.attrs["success"])


class _MaxLenBombPolicy(_HoldPolicy):
    """Wraps ``_HoldPolicy``. Real per-env truncation can only land on the rollout's global last
    step, already negated and zeroed by ruling P7's block before ``dagger_loop``'s ``done_ids``
    branch runs. To reach that branch at all, this policy shrinks ``env.max_episode_length`` a
    few steps past the point a takeover starts, so this env truncates on its own, independent of
    the rollout's ``max_steps`` bound (frozen before this mutation, at the top of ``rollout``)."""

    def __init__(self, env):
        super().__init__()
        self._env = env
        self._armed = False

    def step(self, state, wrist, overhead, context=None, anchor=None):
        if not self._armed and int(self._env.oracle_actor[0]) > 0:
            self._armed = True
            # 5 steps is nowhere near a full pick-and-place cycle (hundreds of steps), so the
            # takeover is still running, and the episode still unsuccessful, when this fires.
            self._env.max_episode_length = int(self._env.episode_length_buf[0]) + 5
        return super().step(state, wrist, overhead, context=context, anchor=anchor)


def test_an_episode_ending_mid_takeover_off_the_final_step_cannot_be_negated(tmp_path):
    """Finding 1: a takeover still running when its env's episode ends can only be negated before
    ``env.step()`` flushes that env's buffer (``mj/env.py``'s ``recorder.end`` call, inside
    ``step()``). By the time ``dagger_loop``'s ``done_ids`` branch runs, the frame is already on
    disk. This is reachable only by forcing this env's own truncation off the rollout's global
    last step, so the test does that with ``_MaxLenBombPolicy`` rather than by editing
    ``dagger_loop.py`` itself."""
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, num_bricks=2, cameras=False, export_dir=tmp_path, shard_name="u", export_all=True
    )
    env.reset(seed=11)
    env.max_episode_length = 3000
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    payload = dl.rollout(
        env,
        rig,
        _MaxLenBombPolicy(env),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        random_interval=None,
        max_wall_minutes=5.0,
    )
    env.close()
    assert payload["takeovers_failed_unnegated"] == 1
    assert payload["episodes"][0]["success"] is False
    with h5py.File(tmp_path / "u.hdf5", "r") as f:
        actor = f["data/demo_0/obs/actor"][:]
        # The branch this test reaches cannot negate: the frame is already committed with a
        # positive actor. This is exactly the contamination the counter exists to surface.
        assert any(v > 0 for v in actor.tolist())


def test_the_takeover_cap_blocks_a_second_takeover(tmp_path, monkeypatch):
    """A still-holding policy would ask for a takeover on every brick; capping at 1 must stop
    the oracle from ever touching the second brick, even though the episode runs long enough
    for that second trigger to fire."""
    monkeypatch.setattr(dl, "MAX_TAKEOVERS", 1)
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, num_bricks=2, cameras=False, export_dir=tmp_path, shard_name="c", export_all=True
    )
    env.reset(seed=11)
    # Generous next to the ~2000 steps two full, uncapped takeovers need (see the sibling
    # test): enough for the first takeover to finish and the second no_close trigger to fire.
    env.max_episode_length = 3600
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    payload = dl.rollout(
        env,
        rig,
        _HoldPolicy(),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        random_interval=None,
        max_wall_minutes=5.0,
    )
    env.close()
    assert payload["takeovers"] == 1
    assert payload["episodes"][0]["success"] is False
    with h5py.File(tmp_path / "c.hdf5", "r") as f:
        actor = f["data/demo_0/obs/actor"][:]
        assert set(actor.tolist()) == {1}


def test_a_declined_takeover_is_not_reoffered_on_the_very_next_step(tmp_path):
    """A decline is a normal outcome (ruling P32, and DONE/ABANDON at entry once the desk is
    clear), and three of the five triggers do not read the step counters -- the ``random`` one
    at p = 1 fires on every step. Every offer rebuilds an ``EnvOracle``, snapshots the scene,
    recomputes the stowed set and wipes ``rig.stats[i]``, so a declined env must be left alone
    for a cooldown instead of being offered again the next step, for the rest of the episode.
    """
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, num_bricks=2, cameras=False, export_dir=tmp_path, shard_name="n", export_all=True
    )
    env.reset(seed=11)
    env.max_episode_length = 300
    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(1))
    calls = []

    def _decline(i):
        calls.append(i)
        return False

    rig.resume = _decline
    payload = dl.rollout(
        env,
        rig,
        _HoldPolicy(),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        random_interval=1.0,  # p = 1.0: the random trigger fires on every single step
        max_wall_minutes=5.0,
    )
    env.close()
    assert payload["takeovers"] == 0 and payload["takeovers_declined"] == len(calls)
    # One offer, then one per cooldown window -- not one per step, which is what the
    # every-step trigger would produce without the cooldown.
    assert 1 <= len(calls) <= payload["steps_run"] // dl.DECLINE_COOLDOWN + 2
    assert len(calls) < payload["steps_run"] // 4

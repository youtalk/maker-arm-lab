"""The eval loop's nearest-brick read with shared and with per-env bin centers, and the
context channel it serves to the policy."""

import random

import numpy as np
import pytest
import torch

from maker_arm_lab.core import layout
from maker_arm_lab.core.eval_loop import EvalFailure, _nearest_brick_cm, rollout
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv


def test_nearest_brick_read_excludes_bricks_inside_each_envs_own_bins():
    shared = layout.bin_interior_centers()
    far = shared.clone()
    far[:, 0] += 0.10
    per_env = torch.stack([shared, far])
    half = layout.bin_half_extents()
    tool = torch.zeros(2, 2)
    held = torch.tensor([-1, -1])
    pos = torch.zeros(2, 1, 3)
    pos[:, 0, :2] = shared[0, :2]  # both envs' only brick sits at env 0's red bin center
    assert _nearest_brick_cm(pos, tool, held, shared, half) == [-1.0, -1.0]
    nearest = _nearest_brick_cm(pos, tool, held, per_env, half)
    assert nearest[0] == -1.0  # env 0's own (shared) bins contain it
    assert abs(nearest[1] - 100.0 * float(shared[0, :2].norm())) < 1e-3  # env 1's bins do not


class _HoldPolicy:
    """Holds the arm still with open jaws; records the context widths it was handed."""

    def __init__(self, context_dim):
        self.context_dim = context_dim
        self.policy_id = "fake:hold"
        self.n_action_steps = 1
        self.inference = {}
        self.widths = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.widths.append(None if context is None else context.shape[1])
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = state[:, :6]
        action[:, 6] = 1.0
        return action


def _env(context):
    draw = sample_run_draw(random.Random(2))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(1, draw, num_bricks=1, cameras=False, episode_length_s=1.0, context=context)
    env.reset(seed=1)
    return env


def test_the_loop_sends_the_served_context_and_checks_its_width():
    env = _env("camera")
    policy = _HoldPolicy(context_dim=6)
    payload = rollout(
        env, policy, run_seed=1, layout_seed=2, permutation=(0, 1, 2, 3), max_wall_minutes=1.0
    )
    assert payload["context"] == "camera" and set(policy.widths) == {6}
    assert payload["spatial_scale"] == 1.0
    assert payload["episode_draws"] is False and payload["frame_jitter"] is False
    env.close()
    env = _env("none")
    with pytest.raises(EvalFailure, match="context"):
        rollout(
            env,
            _HoldPolicy(context_dim=6),
            run_seed=1,
            layout_seed=2,
            permutation=(0, 1, 2, 3),
            max_wall_minutes=1.0,
        )
    env.close()


class _DeltaPrevPolicy(_HoldPolicy):
    """Pretends to be a delta_prev checkpoint: records the anchors the loop sends."""

    action_space = "delta_prev"

    def __init__(self):
        super().__init__(context_dim=0)
        self.anchors = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.anchors.append(None if anchor is None else anchor.copy())
        action = super().step(state, wrist, overhead, context)
        action[:, 0] += 0.01  # the command walks joint 0 by 10 mrad per step
        return action


def test_the_loop_anchors_a_delta_prev_policy_on_the_previous_clamped_command():
    env = _env("none")
    q0 = env.scene["robot"].data.joint_pos[0, :6].numpy().copy()
    policy = _DeltaPrevPolicy()
    rollout(env, policy, run_seed=1, layout_seed=2, permutation=(0, 1, 2, 3), max_wall_minutes=1.0)
    env.close()
    assert len(policy.anchors) >= 2 and policy.anchors[0].shape == (1, 6)
    assert np.allclose(policy.anchors[0][0], q0, atol=1e-6)  # step 0: the measured joints
    # step 1: step 0's command, which is the measured joints plus the 10 mrad walk on joint 0
    assert np.allclose(policy.anchors[1][0, 0], q0[0] + 0.01, atol=1e-6)
    assert np.allclose(policy.anchors[1][0, 1:], q0[1:], atol=1e-6)


def test_an_absolute_policy_gets_no_anchor():
    env = _env("none")
    policy = _AbsolutePolicy()
    rollout(env, policy, run_seed=1, layout_seed=2, permutation=(0, 1, 2, 3), max_wall_minutes=1.0)
    env.close()
    assert set(policy.anchors) == {None}


class _AbsolutePolicy(_HoldPolicy):
    def __init__(self):
        super().__init__(context_dim=0)
        self.anchors = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.anchors.append(anchor)
        return super().step(state, wrist, overhead, context)

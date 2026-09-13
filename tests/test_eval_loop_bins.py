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

    def step(self, state, wrist, overhead, context=None):
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

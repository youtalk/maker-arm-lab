"""The eval loop's nearest-brick read with shared and with per-env bin centers."""

import torch

from maker_arm_lab.core import layout
from maker_arm_lab.core.eval_loop import _nearest_brick_cm


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

"""The target-brick rule (MS4 c2 design 5.3).

One function, computed live on the recorded side and the served side of the privileged
channel, so train and eval see the identical target. It is the choice ``EnvOracle._next_brick``
makes with ``nearest_first_pick_order``: the same candidate filter (not stowed, inside the
top-down x limit of the env's bin layout) and the nearest candidate in the desk plane. A
pinched brick wins, because the oracle carries the brick it holds.
"""

from __future__ import annotations

import torch


def target_brick(
    held: int,
    brick_pos: torch.Tensor,
    tool_xy: torch.Tensor,
    stowed: torch.Tensor,
    x_limit: float,
) -> int:
    """Index of the brick to work on from this state, or -1 when no candidate is left."""
    if held >= 0:
        return int(held)
    candidates = (~stowed) & (brick_pos[:, 0] <= x_limit)
    if not bool(candidates.any()):
        return -1
    dist = (brick_pos[:, :2] - tool_xy).norm(dim=-1)
    dist[~candidates] = float("inf")
    return int(dist.argmin())

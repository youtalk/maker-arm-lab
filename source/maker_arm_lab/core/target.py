"""The target-brick rule (MS4 c2 design 5.3).

One function, computed live on the recorded side and the served side of the privileged
channel, so train and eval see the identical target. It is the choice ``EnvOracle._next_brick``
makes with ``nearest_first_pick_order``: the same candidate filter (not stowed, inside the
top-down x limit of the env's bin layout), measured from the same point (the tool's PICK
POINT, not the tool body), and the nearest candidate in the desk plane. A pinched brick wins,
because the oracle carries the brick it holds.

ONE filter difference remains, and it cannot be closed here: ``_next_brick`` excludes
``self._sorted | stowed``, and ``_sorted`` is the running oracle's own memory of the bricks
IT delivered this episode. This function is stateless -- it is called on envs no oracle is
driving -- so it has only the live ``stowed`` mask. The two disagree exactly when a delivered
brick is momentarily not at rest in its bin (jostled by the next delivery), which is the case
``_sorted`` exists to cover, and only until it settles again. A second, permanent case exists:
a delivered brick knocked clean out of its bin stays outside ``stowed`` for good, so this
function can relabel it the target, while ``_sorted`` still holds it and the running oracle
ignores it for the rest of the episode -- rare, and reachable only on the full-oracle
generation path, since a DAgger takeover re-seeds ``_sorted`` fresh every time it enters.
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

"""Per-brick and per-episode success predicates (design spec section 5.3).

Torch-only: the isaaclab-facing wrappers live in the task's mdp package and
must delegate here, so the gate metric has CPU unit tests.
"""

import torch

PINCH_MIN_FORCE = 2.0
"""Newtons on the WEAKER jaw for a pinch to count as a real, quality grasp.

The codebase's quality pinch gate, shared by every consumer that asks "is this
brick properly gripped": scripts/scripted_grasp_check.py's acceptance, the
lego_sort oracle's runtime grasp check, and core.gate.grade_grasp_attempts'
strict grading. It lives here, beside `held_brick_from_forces`, because it is
that function's `min_force` argument at the three call sites that ask the quality
question -- it used to be three separately-maintained copies of 2.0.

Carried over from the OpenYAM LegoSort checker (openyam-lab M1 Task 10/11,
measured there on OpenYAM's gripper: a settled pinch on a 15.8 mm brick read
8.71/8.71 N with the jaws stalled short of that gripper's closed position,
against <= 3e-5 N on a fully-closed jaw holding nothing). The Maker Arm's
gripper differs; `scripts/scripted_grasp_check.py` (Task D3) re-measures the
pinch criterion on this arm and is the authority for changing this number.

NOT the only force threshold in the codebase, and deliberately so: scripts/
eval_policy.py's HELD_MIN_FORCE is looser (1.0) because it answers a different
question -- which brick was between the jaws mid-transport, where the normal load
dips as the arm decelerates -- and it is single-copy where it is used.
"""


def bricks_at_rest_in_bins(
    brick_pos_w: torch.Tensor,
    brick_lin_vel_w: torch.Tensor,
    bin_centers_w: torch.Tensor,
    bin_half_extents: torch.Tensor,
    brick_color_ids: torch.Tensor,
    max_speed: float = 0.02,
) -> torch.Tensor:
    """(N,B) bool: brick b in env n is at rest inside its correct-color bin.

    The bins are identical in every env, so ``bin_centers_w`` is (4,3) and is
    broadcast across envs. A per-env (N,4,3) form used to be accepted here too,
    but nothing ever produced one -- ``terminations._bin_centers_cached``
    returns ``layout.bin_interior_centers()``, which is (4,3) by construction --
    so that branch shipped unreachable. It is rejected explicitly rather than
    silently accepted, because indexing a (N,4,3) tensor with ``brick_color_ids``
    would select *envs* by colour id and grade every brick against the wrong
    env's bins without raising.

    Args:
        brick_pos_w: (N,B,3) brick centers, world frame.
        brick_lin_vel_w: (N,B,3) brick linear velocities.
        bin_centers_w: (4,3) bin interior AABB centers.
        bin_half_extents: (3,) bin interior half extents.
        brick_color_ids: (B,) long, target bin index per brick.
        max_speed: "at rest" linear speed threshold [m/s].
    """
    if bin_centers_w.dim() != 2:
        raise ValueError(f"bin_centers_w must be (4,3), got {tuple(bin_centers_w.shape)}")
    targets = bin_centers_w[brick_color_ids]  # (B,3)
    inside = ((brick_pos_w - targets).abs() <= bin_half_extents).all(dim=-1)
    at_rest = brick_lin_vel_w.norm(dim=-1) <= max_speed
    return inside & at_rest


def all_bricks_stowed(
    brick_pos_w: torch.Tensor,
    brick_lin_vel_w: torch.Tensor,
    bin_centers_w: torch.Tensor,
    bin_half_extents: torch.Tensor,
    brick_color_ids: torch.Tensor,
    max_speed: float = 0.02,
) -> torch.Tensor:
    """(N,) bool: episode success = every brick stowed (design spec 5.3)."""
    return bricks_at_rest_in_bins(
        brick_pos_w, brick_lin_vel_w, bin_centers_w, bin_half_extents, brick_color_ids, max_speed
    ).all(dim=-1)


def containing_bin(
    brick_pos_w: torch.Tensor,
    brick_lin_vel_w: torch.Tensor,
    bin_centers_w: torch.Tensor,
    bin_half_extents: torch.Tensor,
    max_speed: float = 0.02,
) -> torch.Tensor:
    """(N,B) int64: index (0-3) of the bin each brick is at rest inside, or -1.

    Unlike ``bricks_at_rest_in_bins``, this checks every bin for every brick
    instead of only the brick's correct-color target bin, so it can tell a
    brick resting in the *wrong* bin apart from one that never made it into
    any bin at all.

    Args:
        brick_pos_w: (N,B,3) brick centers, world frame.
        brick_lin_vel_w: (N,B,3) brick linear velocities.
        bin_centers_w: (4,3) bin interior AABB centers.
        bin_half_extents: (3,) bin interior half extents.
        max_speed: "at rest" linear speed threshold [m/s].
    """
    if bin_centers_w.dim() != 2:
        raise ValueError(f"bin_centers_w must be (4,3), got {tuple(bin_centers_w.shape)}")
    pos = brick_pos_w.unsqueeze(2)  # (N,B,1,3)
    bins = bin_centers_w.unsqueeze(0).unsqueeze(0)  # (1,1,4,3)
    inside = ((pos - bins).abs() <= bin_half_extents).all(dim=-1)  # (N,B,4)
    at_rest = brick_lin_vel_w.norm(dim=-1) <= max_speed  # (N,B)
    resting_in_any = inside.any(dim=-1) & at_rest
    bin_idx = inside.long().argmax(dim=-1)
    return torch.where(resting_in_any, bin_idx, torch.full_like(bin_idx, -1))


def held_brick_from_forces(
    left_mag: torch.Tensor,
    right_mag: torch.Tensor,
    min_force: float = 1.0,
) -> torch.Tensor:
    """(N,) int64: index of the brick pinched between the jaws, or -1.

    A brick is "held" when both jaws report contact force against it, so the
    per-brick pinch strength is the weaker of the two -- a brick merely
    brushing one jaw does not count.

    Args:
        left_mag: (N,B) left-jaw contact-force magnitude, per brick.
        right_mag: (N,B) right-jaw contact-force magnitude, per brick.
        min_force: minimum pinch force to count as held [N].
    """
    pinch = torch.minimum(left_mag, right_mag)  # (N,B)
    strongest, brick_idx = pinch.max(dim=-1)
    return torch.where(strongest >= min_force, brick_idx, torch.full_like(brick_idx, -1))

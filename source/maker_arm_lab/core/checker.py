"""Per-brick and per-episode success predicates (design spec section 5.3).

Torch-only: the isaaclab-facing wrappers live in the task's mdp package and
must delegate here, so the gate metric has CPU unit tests.
"""

import torch

from maker_arm_lab.core import layout

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


PINCH_MIN_STALL = min(layout.BRICK_SIZE) / 2.0
"""Metres the jaws must stall SHORT of closed for a pinch to count (0.0048 m).

Derived from the brick, not fitted to a trial. The two jaws open symmetrically about the
tool axis -- `gripper_left_joint`'s URDF axis is +y and `gripper_right_joint`'s is its
negative, both spanning 0 (touching) to GRIPPER_OPEN_POS -- so the gap between the plates is
twice the per-jaw travel that `stall` measures. For the jaws to be spanning the brick AT ALL,
in any orientation, that gap must be at least the brick's smallest dimension:

    gap = 2 * stall >= min(BRICK_SIZE)   =>   stall >= min(BRICK_SIZE) / 2 = 4.8 mm

Below that the plates are closer together than the thinnest cross-section the brick has, so
they cannot be around it: they are on a corner, on a chamfer, or on nothing. That is exactly
the entrainment this gate exists to reject -- fingers closed with the brick merely cammed
into the jaw pocket, which lifts and travels just like a real grasp.

The brick's SMALLEST dimension is the right bound rather than the 15.8 mm width the oracle
actually squares the jaws to. It is the loosest threshold that still proves contact, so it
cannot reject a legitimate grasp that happened to take the brick on another face; a bound
built from the intended width would be stricter but would encode the intent rather than the
geometry.

Replaces an inherited OpenYAM 3.0 mm, which the Task D3 brief named these trials as the
authority to re-validate. The derived floor is STRICTER, not looser. Checked against the
Task D3 part-10 gate afterwards, having been derived first: the six carries stalled
5.797-7.487 mm (against 7.9 mm for an ideal grip on the 15.8 mm width, the difference being
contact compliance), the two dropped trials 0.000 and 0.008 mm, and one entrained trial
2.886 mm. 4.8 mm separates them with 1.0 mm of headroom below the worst carry and 1.9 mm
above the entrainment -- where 3.0 mm had only 0.114 mm of headroom above it.

Single-copy here because it was two: `scripts/scripted_grasp_check.py` and the lego_sort
oracle's runtime grasp check each carried their own 3.0, exactly as PINCH_MIN_FORCE was
three copies of 2.0 before it moved here.
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

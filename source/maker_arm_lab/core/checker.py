"""Per-brick and per-episode success predicates (design spec section 2.2, ported).

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

VALIDATED ON THIS ARM AND CLOSED (Task D3 part 16). It arrived from the
OpenYAM LegoSort checker (openyam-lab M1 Task 10/11, measured there at
8.71/8.71 N on a 15.8 mm brick against <= 3e-5 N on a jaw holding nothing) with
a forward reference to Task D3. D3 is done, and its 10/10 gate is the data that
reference was waiting for: every one of the ten carries read 7.59-7.62 N on the
weaker jaw, symmetric to 0.001, against 0.00 N on the single non-grasp.

The number is KEPT AT 2.0 rather than raised toward the observed 7.6 N, and
that is a decision, not an oversight. Any threshold in (0, 7.5) separates this
data, so the gate does not constrain the choice; what does is what the constant
is FOR. It is a floor that rejects "no contact", and the discriminator between
a real pinch and an entrainment is PINCH_MIN_STALL below, which is derived from
the brick's geometry. Fitting this to ten samples of one brick face would
narrow it to the grip this arm happens to produce on the 15.8 mm width and risk
rejecting a legitimately weaker but real grip on another face -- the same
argument PINCH_MIN_STALL's derivation makes for using min(BRICK_SIZE). 2.0 sits
3.8x below the observed grip and three orders of magnitude above a jaw holding
nothing.

NOT the only force threshold in the codebase, and deliberately so: HELD_MIN_FORCE
below is looser (1.0) because it answers a different question -- which brick was
between the jaws mid-transport, where the normal load dips as the arm decelerates
-- and it now sits beside the function it is an argument to, as one definition.
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


PINCH_MAX_ASYMMETRY = 0.30
"""Largest |L - R| / max(L, R) of filtered finger forces a real pinch may show.

The third of the three pinch thresholds, and single-copy here for the same reason as its
two neighbours: `scripts/scripted_grasp_check.py` and the lego_sort oracle's runtime grasp
check each carried their own 0.30, exactly as PINCH_MIN_FORCE was three copies of 2.0 and
PINCH_MIN_STALL two copies of 3.0 before they moved here. The comment sentence that
introduced all three named them together while only two of them were consolidated.

Carried over from OpenYAM unchanged. Task D3's 10/10 gate on this arm is consistent with it
by a wide margin -- every carry came in at an asymmetry <= 0.001, against this 0.30 bound --
but that is a one-sided check: nothing in D3 produced a lopsided-but-real grip, so the bound
has not been exercised from the side it exists to reject. Treat it as inherited, not
re-measured.
"""


def bin_targets(bin_centers_w: torch.Tensor, brick_color_ids: torch.Tensor) -> torch.Tensor:
    """Each brick's target bin center: (B,3) from (4,3) centers, (N,B,3) from (N,4,3).

    The (4,3) form is one layout shared by every env (the Isaac scene, and a MuJoCo run
    without per-episode draws). The (N,4,3) form is one layout per env (MuJoCo with
    ``--episode_draws``). It is indexed along its LAST bin axis, so env n is graded against
    its own bins and never against env ``color_id``'s, which a plain ``[]`` index would do.
    """
    if bin_centers_w.dim() == 2:
        return bin_centers_w[brick_color_ids]
    if bin_centers_w.dim() == 3:
        return bin_centers_w[:, brick_color_ids]
    raise ValueError(f"bin_centers_w must be (4,3) or (N,4,3), got {tuple(bin_centers_w.shape)}")


def bricks_at_rest_in_bins(
    brick_pos_w: torch.Tensor,
    brick_lin_vel_w: torch.Tensor,
    bin_centers_w: torch.Tensor,
    bin_half_extents: torch.Tensor,
    brick_color_ids: torch.Tensor,
    max_speed: float = 0.02,
) -> torch.Tensor:
    """(N,B) bool: brick b in env n is at rest inside its correct-color bin.

    Args:
        brick_pos_w: (N,B,3) brick centers, world frame.
        brick_lin_vel_w: (N,B,3) brick linear velocities.
        bin_centers_w: (4,3) bin interior AABB centers shared by every env, or (N,4,3)
            centers per env (see ``bin_targets``).
        bin_half_extents: (3,) bin interior half extents.
        brick_color_ids: (B,) long, target bin index per brick.
        max_speed: "at rest" linear speed threshold [m/s].
    """
    targets = bin_targets(bin_centers_w, brick_color_ids)  # (B,3) or (N,B,3)
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
        bin_centers_w: (4,3) shared centers, or (N,4,3) per env.
        bin_half_extents: (3,) bin interior half extents.
        max_speed: "at rest" linear speed threshold [m/s].
    """
    if bin_centers_w.dim() == 2:
        bins = bin_centers_w[None, None]  # (1,1,4,3)
    elif bin_centers_w.dim() == 3:
        bins = bin_centers_w[:, None]  # (N,1,4,3)
    else:
        raise ValueError(
            f"bin_centers_w must be (4,3) or (N,4,3), got {tuple(bin_centers_w.shape)}"
        )
    pos = brick_pos_w.unsqueeze(2)  # (N,B,1,3)
    inside = ((pos - bins).abs() <= bin_half_extents).all(dim=-1)  # (N,B,4)
    at_rest = brick_lin_vel_w.norm(dim=-1) <= max_speed  # (N,B)
    resting_in_any = inside.any(dim=-1) & at_rest
    bin_idx = inside.long().argmax(dim=-1)
    return torch.where(resting_in_any, bin_idx, torch.full_like(bin_idx, -1))


HELD_MIN_FORCE = 1.0
"""Newtons on the weaker jaw for `held_brick_from_forces` to call a brick held: the loose
ATTRIBUTION threshold ("which brick was between the jaws"), deliberately under the quality
gate PINCH_MIN_FORCE = 2.0 that `core.gate.grade_grasp_attempts` scores strictly.

THE ONE DEFINITION, and it lives here because it used to be three. `core.eval_loop`
(attribution), `mj.env` (the privileged channel's target rule) and the oracle runtime's
resume entry (LIFT versus HOVER) each declared their own 1.0 and each named one of the
others as the authority, so retuning it would have left the corpus's labels, the resume
decision and the gate's attribution silently disagreeing. Import it; do not re-declare it.
"""


def held_brick_from_forces(
    left_mag: torch.Tensor,
    right_mag: torch.Tensor,
    min_force: float = HELD_MIN_FORCE,
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

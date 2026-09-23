"""The heap outer loop: a policy drives pick cycles that a scripted selector sets up.

Phase B design 5.2. One cycle: the selector (a ``HeapOracle`` per env, the teacher's own
rule) names the target, the loop writes it into ``env.target_part`` so the overhead marker
moves, the policy drives, and the privileged grader ends the cycle. Three ends: a part that
was not in the bin when the cycle began is inside the bin, moving or not (the target, or any
part the jaws carried, rung b5's rule), the cycle spent ``CYCLE_STEP_BUDGET`` control steps, or the target
came to rest on the desk outside ``layout.in_reach`` and outside the bin, out of the jaws
(knocked out of reach; a carry crosses out-of-reach space in flight and may pause there). A
failed cycle marks its cell in the selector's failure memory. A part an earlier cycle stowed
never ends a later one, so a stowed part that settles late is not credited twice.
The cycle ends on entry into the bin and not at rest (2026-09-22), because the teacher's
corpus switches targets when RETREAT ends, and a part settling in a pile can take hundreds
of steps longer than that; the policy meanwhile saw a stale target and started a pick of its
own. The metric never reads the cycle end: it is the at-rest grader at the episode's end.
There is no reset between cycles. The episode ends when the selector returns None (no
candidate, or the cycle budget is spent) or every part is stowed.

Six protocol switches, all off by default so the old protocol stays reproducible
(2026-09-22): ``select_at_release`` selects the next target when the release opens the jaws
(``release_ready``) instead of at the retreat (``select_ready``), and ends a cycle at the
release of a lifted part wherever it landed (``released_after_lift``), and lets a cycle that
holds a part at its step budget run ``CARRY_GRACE_STEPS`` more (``budget_ends_cycle``); ``replan_on_switch``
sends the server a per-env replan flag on the step a target row is written, so the policy's
chunk is cut at the switch (``policy_server.ActionQueues``); ``row_refresh`` rewrites the row
from the live scene at a re-open that lifted nothing (``reopen_edge``, ``refreshed_row``), the
teacher's REOPEN refresh; ``squeeze_floor`` caps a closing gap command at the teacher's close
set point (``floored_gap``) and ``close_gate`` holds the jaws at the landing gap while the
pick point is still descending onto the target (``gated_gap``), two serving-side gripper
levers applied in that order, gate first; ``square_wrist`` turns the wrist roll command toward
the served row's closing axis on the final approach (``squared_roll``). The payload's
``inference`` block records all six.

With ``policy=None`` the oracle rig drives every cycle through the same takeover entry the
DAgger mode uses (``OracleRig.resume(i, target=...)``): the preflight's driver and the
loop's own end-to-end check. A takeover holds the env until the rig hands it back at
RETREAT, DONE or ABANDON, and the loop grades only while the rig is idle, so one takeover is
one loop cycle.

Like the eval loop this runs one episode per env in lockstep and reads the env through the
duck-typed surface of ``maker_arm_lab.mj.env``. Torch and stdlib only.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import torch

from maker_arm_lab.core import desk_boxes, grasp, layout, oracle
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.dagger_loop import DECLINE_COOLDOWN, trigger_reason
from maker_arm_lab.core.eval_loop import GRASP_HOLD_WINDOW, EvalFailure, _nearest_brick_cm
from maker_arm_lab.core.gate import RolloutTracker, clamp_arm_actions, policy_state
from maker_arm_lab.core.global_ik import fold_to_half_turn
from maker_arm_lab.core.heap_oracle import CANDIDATE_MAX_Z, target_context
from maker_arm_lab.core.stowed import (
    _brick_states,
    env_heap_bin_centers,
    parts_in_desk_boxes,
    per_brick_stowed,
)
from maker_arm_lab.robots.frames import (
    ARM_JOINT_REGEX,
    JAW_AXIS_BODY,
    TOOL_BODY,
    WRIST_ROLL_INDEX,
)

CYCLE_STEP_BUDGET = 800
"""Control steps one policy cycle may take. The phase B preflight (plan Task 15) reads the k
corpus's 1974 teacher cycles (obs/actor over ~/data/maker-arm-ms6/k/corpus): a median of
378.5 steps, p90 572, p95 732, max 1407. 800 is kept because the median is within 10% of 400
and 800 is above the p95."""
CYCLE_BUDGET_FACTOR = 1.3
"""Pick cycles per part an episode may spend: the gate's own cycles-per-stowed rule."""
AT_REST_SPEED = 0.02
"""m/s: the grader's at-rest speed, ``checker.bricks_at_rest_in_bins``'s ``max_speed``."""
IDLE_CLOSE_GAP = grasp.OPEN_HALF_GAP
"""m: the continuous close threshold of an env with no open cycle, half the full jaw gap.

Under the continuous gripper a commanded gap below the grasp width of the cycle's target
reads as a close (``cycle_close_below``). The teacher lands at the matched gap
``2 * (landing_half_gap(w) - JAW_HOLD_BAND)`` = w + 8 mm and closes to
``2 * max(0, w / 2 - CLOSE_SQUEEZE_M)`` = w - 6 mm, so w splits every grasp's two set
points, while one fixed threshold cannot: across the catalog the landings span 15.8 to
47.8 mm and the closes 1.8 to 33.8 mm. An env with no cycle has no w. Half the full gap
keeps a fully open coast reading open and a zero command reading closed. Triggers and
duty-cycle stats read the threshold; the metric never does."""
HALF_BUDGET_NO_LIFT = "half_budget_no_lift"
"""The heap trigger of phase B design 5.5: the cycle is at half its step budget with no lift.
No lift means the jaws hold nothing, or hold a part that has not risen ``oracle.LIFT_MIN_RISE``
above where it was when the cycle opened (``Cycle.start_z``)."""
HELD_NO_RISE = "held_no_rise"
"""The failure trigger of ``--trigger failure`` (campaign n, 2026-09-22): the jaws have held
a part for ``TakeoverConfig.held_no_rise_steps`` steps and it has not risen
``oracle.LIFT_MIN_RISE`` above its height at the cycle's open. Run m's held_no_lift class
(0.278 of its cycles) closes near the desk and lifts the pick point without the part; the
half-budget trigger reaches it 400 steps in, this one at ``HELD_NO_RISE_STEPS``."""
HELD_NO_RISE_STEPS = 60
"""Held steps without a rise before ``HELD_NO_RISE`` fires: ``CLOSED_NO_PINCH_STEPS``'s span."""
CENSUS_MIN_HELD_STEPS = GRASP_HOLD_WINDOW[0]
SELECT_RETREAT_RADIUS = 0.06
"""m: the next target is selected once the pick point is this close to ``oracle.RETREAT_POINT``.

0.06 was widened to 0.10 for a few hours on 2026-09-22 (the 50k policy retreats to about
x 0.31, 5 to 6 cm from the point, so 6 cm catches the retreat on its edge) together with a
0.7 opening and a 150-step cap. Measured on the 200k checkpoint at 50 actions per forward,
draws 0 to 3 of seed 20261300, the wider wait read lower on every finished draw (draw 3: 112
against 151 of 250), so the switch fired mid-retreat, before the state the corpus pairs with a
new target. The constants went back to what the 0.597 read used.

The teacher's corpus switches the target row and the marker when RETREAT ends: the arm at
RETREAT_POINT (plus the episode's via jitter) with the jaws at the full gap, and only then does
the next HOVER start. Selecting the instant the stowed part rests in the bin, as the loop did
until 2026-09-21, hands the policy a new target while it is still at the release pose, a state
no demonstration pairs with a fresh target. On round 1's 700k read, 646 of the 1324
held_no_lift cycles followed a stow and their first close was a median 13 cm from the target,
at the bin."""
SELECT_OPEN_FRACTION = 0.8
"""The measured jaw opening fraction that reads as the retreat's full gap."""
DESK_REST_MAX_Z = desk_boxes.DESK_BOX_RELEASE_Z - oracle.LIFT_MIN_RISE
"""m: with desk boxes, the highest a part off every box may lie to count as knocked out of
reach, a lift's rise under the release. A teacher-driven policy's cycle ended out_of_reach at
its own release, the part in the opening jaws at z 0.098 m over its box (2026-09-23)."""
FRAME_EVERY = 10
"""Steps between two frames ``rollout``'s ``on_frame`` dump receives."""
SELECT_WAIT_FRACTION = 0.375
"""The share of ``CYCLE_STEP_BUDGET`` the loop waits for the retreat at most before it selects
anyway: 300 steps at the 800-step budget, and the teacher's RETREAT converges in 91 to 147 of
its 300 steps. A fraction, so a test that shrinks the budget shrinks the wait with it."""


def select_ready(
    pick_point, open_fraction: float, waited: int, retreat=oracle.RETREAT_POINT
) -> bool:
    """May the loop hand the policy its next target now? Pure. ``pick_point`` is the rig's
    (3,) pick point, ``open_fraction`` the measured jaw opening in [0, 1], ``waited`` the steps
    since the last cycle closed, ``retreat`` the selector's retreat point (desk boxes have
    their own, ``desk_boxes.DESK_BOX_RETREAT_POINT``)."""
    if waited >= int(SELECT_WAIT_FRACTION * CYCLE_STEP_BUDGET):
        return True
    near = float((torch.as_tensor(pick_point) - torch.tensor(retreat)).norm())
    return near <= SELECT_RETREAT_RADIUS and open_fraction >= SELECT_OPEN_FRACTION


def release_ready(open_fraction: float, waited: int, after_stow: bool) -> bool:
    """``select_at_release``: may the loop hand the policy its next target now? Pure.

    After a stow or the release of a lifted part (``after_stow``: the last end is one of
    ``RELEASE_ENDS``) the release is done when the measured jaw opening reads
    ``SELECT_OPEN_FRACTION``, with no retreat radius. The corpus teacher switches the row at
    the end of its RETREAT, but the policy retreats to about x 0.31 and, with the stale row
    and marker on the part now in the bin, never settles inside ``SELECT_RETREAT_RADIUS``:
    on run m's 400k read two thirds of ``select_ready``'s waits ran to the cap, 26 % of all
    steps. The same cap still bounds this wait. After any other end the loop selects at once.
    """
    if not after_stow or waited >= int(SELECT_WAIT_FRACTION * CYCLE_STEP_BUDGET):
        return True
    return open_fraction >= SELECT_OPEN_FRACTION


RELEASE_ENDS = ("stowed", "released")
"""The cycle ends a release makes; ``release_ready`` waits for the jaws' full gap after them."""


def released_after_lift(cycle: Cycle, open_fraction: float, held: int) -> bool:
    """``select_at_release``: did the jaws just release a lifted part? The part rose
    ``oracle.LIFT_MIN_RISE`` between the jaws (``cycle.max_rise``), nothing is held now
    (``held`` < 0, the loop's held read) and the measured opening reads
    ``SELECT_OPEN_FRACTION``. The stowed rule (``cycle_end``) ends the cycle only when the
    part entered the bin; this ends it at the same moment when the part landed anywhere
    else, as the teacher pre-selects when RELEASE ends wherever its part went
    (``HeapOracle._preselect``). A re-open without a lift is jaw chatter or the policy's
    REOPEN, not a release, and keeps the cycle. Pure."""
    return (
        cycle.max_rise >= oracle.LIFT_MIN_RISE
        and held < 0
        and open_fraction >= SELECT_OPEN_FRACTION
    )


CARRY_GRACE_STEPS = 400
"""Control steps ``budget_ends_cycle`` keeps a cycle open past ``CYCLE_STEP_BUDGET`` while
the jaws hold a part."""


def budget_ends_cycle(steps: int, held: bool, grace_used: int) -> bool:
    """``select_at_release``: does the step budget end the cycle now? Pure.

    ``steps`` is the cycle's age, ``held`` the census held read (both jaws press one part at
    or above ``HELD_MIN_FORCE``) and ``grace_used`` the steps the grace has kept this cycle
    open so far. Under the budget the answer is no. At the budget a cycle holding nothing
    ends as before. A cycle holding a part gets ``CARRY_GRACE_STEPS`` more steps, during
    which the normal ends apply (a bin entry, a release after a lift, or the part dropped,
    at which point the budget ends it at once); after the grace it ends held or not.

    The gap: under ``select_at_release`` a budget end selects the next target at once, so a
    cycle that ends on the budget while the jaws still hold a part (the census class
    ``carrying``, or ``held_no_lift`` with the part still held at desk height, mid-retry)
    hands the policy a fresh target row while it carries a part, a state the teacher's
    corpus never contains: every teacher cycle ends with a RELEASE and the row switches only
    then. The T0 100k read (n/gate/sel/t0_100000_v2sq, 2026-09-22) has 40 carrying cycles of
    1289, and the old protocol's reads credited about 16 % of their stowed parts to carries
    that entered the bin after their cycle ended; under the release rule those carries were
    cut short."""
    if steps < CYCLE_STEP_BUDGET:
        return False
    return not held or grace_used >= CARRY_GRACE_STEPS


"""Control steps a part must stay between the jaws to count as held in the cycle census: the
start of the six-brick hold window. A shorter pinch is a brush."""


@dataclass(frozen=True)
class TakeoverConfig:
    """The DAgger mode (phase B design 5.5). ``random_interval`` is the mean control steps
    between random takeovers, None for trouble triggers only. An episode may spend
    ``max_per_episode_factor`` takeovers per cycle of its budget (the design's "two per cycle
    of the budget"); a loop cycle holds at most one takeover, since a takeover's end ends it."""

    random_interval: float | None
    # ``HELD_NO_RISE`` fires after this many held steps without a rise; None leaves it off.
    held_no_rise_steps: int | None = None
    # At the default 2.0 this cap cannot bind: a loop cycle holds at most one takeover and an
    # episode runs at most cycle_budget cycles. It exists to cap rounds that lower it.
    max_per_episode_factor: float = 2.0


def cycle_budget(k: int) -> int:
    """The pick cycles an episode of ``k`` parts may spend (phase B design 5.1: 65 at K=50)."""
    return math.ceil(CYCLE_BUDGET_FACTOR * k)


def episode_length_s(budget: int, control_hz: float = 30.0) -> float:
    """The env's episode length: every cycle of the budget at its full step budget, plus the
    retreat wait at its cap before each. The wait lies outside the cycles, so without its
    allowance the episode's step cap binds before the cycle budget does: on the 400k check of
    run m (2026-09-22), 41 of 60 episodes ended on the step cap at a median of 60 of 65 cycles,
    with a quarter of their steps spent in the wait."""
    return budget * (CYCLE_STEP_BUDGET + int(SELECT_WAIT_FRACTION * CYCLE_STEP_BUDGET)) / control_hz


def grip_open_flags(
    gripper: torch.Tensor, gripper_mode: str, close_below: torch.Tensor | None = None
) -> torch.Tensor:
    """(N,) bool: which raw gripper commands read as an open jaw.

    Binary: at or above zero, the env's contract; ``close_below`` is ignored. Continuous: a
    commanded gap at or above ``close_below``, the (N,) per-env threshold in meters
    (``IDLE_CLOSE_GAP`` for every env when None); under it the jaws are closing onto the part.
    """
    return open_signal(gripper, gripper_mode, close_below) >= 0.0


def open_signal(
    gripper: torch.Tensor, gripper_mode: str, close_below: torch.Tensor | None = None
) -> torch.Tensor:
    """The gripper column shifted so that ``>= 0`` means open in either mode, which is the
    sign convention ``RolloutTracker.record_gripper`` counts duty cycles and flips by.
    ``close_below`` as in ``grip_open_flags``."""
    g = gripper.detach().to(device="cpu", dtype=torch.float32)
    if gripper_mode != "continuous":
        return g
    if close_below is None:
        return g - IDLE_CLOSE_GAP
    return g - close_below.to(device="cpu", dtype=torch.float32)


def floored_gap(gap: float, width: float) -> float:
    """``squeeze_floor``: the continuous gap command after the teacher's close floor. Pure.

    ``width`` is the served target row's grasp width (``target_context``'s sixth column, 0
    with no live row). A command under it means the policy is closing on the part, and it
    is capped at ``grasp.close_gap(width)``, w - 6 mm, the gap the teacher commands at every
    CLOSE. The jaw drive's force is its stall distance times its stiffness: on run m's 400k
    read (305 held_no_lift cycles, 2026-09-22) the first close sat a median 1.3 mm under w,
    about a fifth of the teacher's squeeze, the arm rose 10.8 cm and the part rose 0 mm.
    A command at or above w (a landing, a release, a coast) and any command with no row are
    left alone. The corpus recorder needs no floor: the teacher already commands this exact
    set point (``OracleRig._matched_close_command``), so the floor is a no-op on every
    recorded frame and the served and recorded closes agree by construction."""
    if width <= 0.0 or gap >= width:
        return gap
    return min(gap, grasp.close_gap(width))


GATE_MARGIN = 0.003
"""m: ``close_gate`` holds the jaws open while the pick point is this far above the target's
top (its row centre plus half height). The stows of run m's 400k read land the pick point at
about the centre (off_z quartiles -0.8, +0.6, +3.6 mm), well inside the margin."""
GATE_WINDOW = 5
"""Control steps over which ``close_gate`` reads the pick point's descent."""
GATE_DESCENT = 0.0005
"""m: the pick point must have come down this much over ``GATE_WINDOW`` steps for the gate to
hold. A policy that has stopped descending closes wherever it stopped."""


def gated_gap(
    gap: float, width: float, z_pick: float, z_top: float, descending: bool, held: bool = False
) -> float:
    """``close_gate``: the continuous gap command while the policy is still descending. Pure.

    ``width`` is the served target row's grasp width (0 with no live row), ``z_top`` the row's
    centre plus half height, ``z_pick`` the rig's pick point height, ``descending`` whether
    it came down ``GATE_DESCENT`` over the last ``GATE_WINDOW`` steps and ``held`` whether the
    cycle has already held a part or lifted one. A command under ``width`` before the first
    hold, while the pick point is still ``GATE_MARGIN`` above the top and still coming down,
    is replaced by the teacher's landing gap, ``2 * (landing_half_gap(w) - JAW_HOLD_BAND)``
    = w + 8 mm (``OracleRig.actions``). Every other command is left alone: a close at or
    below the top, a close after the descent stopped (a policy that stops 4 mm high still
    closes there), a landing, a release, any command with no row, and any command once the
    cycle has held a part.

    The hold rule comes from the 500k read of 2026-09-22 (``squeeze_floor`` + ``close_gate``
    without it): 461 of 1000 against 671 with the floor alone, lifted_lost 27 -> 187. The
    served row keeps the target's desk position, so during the carry the arm descends toward
    the bin with a closed command far above the row's top, every gate condition held, and the
    gate commanded the landing gap in flight; the part dropped. The gate now stops at the
    cycle's first held step or first lift, whichever comes first.

    Run m's 400k read (2026-09-22): the held_no_lift cycles (27% of all cycles) are shallow
    pinches, their pick point 3 to 6 mm above the part centre at the first close (off_z
    quartiles +2.8, +4.4, +5.8 mm against -0.8, +0.6, +3.6 mm for the stows), so on a flat
    brick the pads grip the top edge and the studs and the part slides out under the lift.
    A quarter of them close first in mid-air, more than 3 cm above the part, then re-open and
    close again; the squeeze floor moved the read (704 against 642 of 1000 on the 400k
    checkpoint) and left held_no_lift where it was (331 against 321 of about 1236 cycles).
    The corpus recorder needs no gate: the teacher never closes while descending, DESCEND
    ends on a stall and CLOSE begins after it, so the gate is a no-op on recorded frames."""
    if held or width <= 0.0 or gap >= width or not descending or z_pick <= z_top + GATE_MARGIN:
        return gap
    return 2.0 * (grasp.landing_half_gap(width) - grasp.JAW_HOLD_BAND)


SQUARE_RADIUS = 0.040
"""m: ``square_wrist`` acts while the pick point is this close in xy to the row centre.

Read on the T0 250k trace of 2026-09-22 (draws 0 and 1, 271 cycles with a close): the final
descent (the pick point below the row's top plus ``SQUARE_ABOVE``) begins a median 17 to 21
mm from the centre in xy, p75 29 to 37 mm. A 20 mm radius would start the correction after
the descent had begun in half the cycles; 40 mm starts it with the descent in about 80% of
them. The policy's own roll moves little in there (a median 5 to 9 degrees from the radius to
the close), so what the lever adds is the correction, not a race against the policy."""
SQUARE_ABOVE = 0.040
"""m: and while the pick point is below the row's top (centre plus half height) plus this."""
SQUARE_MAX_STEP = math.radians(3.0)
"""rad: the most ``square_wrist`` moves the served roll command in one control step."""


def squared_roll(ik, q, row, pick, prev: float, limits, held: bool = False) -> float:
    """``square_wrist``: the wrist roll command that squares the jaws to the served row. Pure.

    ``q`` is the (6,) arm command the policy served, ``row`` the served target row
    (``target_context``: centre x y z, cos and sin of the closing axis heading, width, half
    height), ``pick`` the rig's pick point, ``prev`` the roll command the env received on the
    step before, ``limits`` the roll joint's (lower, upper) and ``held`` whether the cycle has
    already held a part or lifted one. Returns ``q[WRIST_ROLL_INDEX]`` unchanged with no live
    row (width 0), once the cycle has held or lifted a part, or while the pick point is more
    than ``SQUARE_RADIUS`` from the row centre in xy or ``SQUARE_ABOVE`` above its top.

    Otherwise the jaw heading of ``q`` comes from FK (``JAW_AXIS_BODY`` in the tool frame,
    flattened to the world xy plane), and so does the sign of the roll joint's effect on it
    (a finite difference; the tool points down, so +1 rad of roll turns the jaw about +1
    rad). The error ``fold_to_half_turn(row heading - jaw heading)`` is folded into [-pi/2,
    pi/2) because the parallel jaw is symmetric under a half turn, and the target roll is
    ``q[5] + sign * error``. When that leaves the joint's limits, the other half-turn branch,
    ``error -/+ pi``, is the target if it lies inside; when neither does, the command is left
    alone. Rate rule: the returned command is the target clamped into ``prev`` plus or minus
    ``SQUARE_MAX_STEP``, so the served roll moves at most 3 degrees per step while the lever
    acts. Once the cycle holds or lifts a part the lever stops and the policy's own roll is
    served at once, a step that can be as large as the correction was.

    The T0 250k trace (2026-09-22): 55% of held_no_lift cycles end with one finger pad on top
    of the target, and at the close the jaw is a median 24 degrees off the part's nearest
    horizontal axis (the teacher 2 to 4). Against the row's own heading the policy's jaw sits
    a median 53 to 57 degrees off at the close on held_no_lift and 32 to 47 on stows, so the
    lever turns most grasps onto the row's axis, the axis the teacher closes across."""
    roll = float(q[WRIST_ROLL_INDEX])
    near = math.hypot(float(pick[0]) - float(row[0]), float(pick[1]) - float(row[1]))
    top = float(row[2]) + float(row[6])
    if held or float(row[5]) <= 0.0 or near > SQUARE_RADIUS or float(pick[2]) >= top + SQUARE_ABOVE:
        return roll
    q = torch.as_tensor(q, dtype=torch.float64)
    nudged = q.clone()
    nudged[WRIST_ROLL_INDEX] += 1e-3
    _, rot = ik.fk_batch(torch.stack([q, nudged]))
    jaw = rot @ torch.tensor(JAW_AXIS_BODY, dtype=torch.float64)
    yaw = torch.atan2(jaw[:, 1], jaw[:, 0]).tolist()
    sign = 1.0 if fold_to_half_turn(yaw[1] - yaw[0]) > 0.0 else -1.0
    err = fold_to_half_turn(math.atan2(float(row[4]), float(row[3])) - yaw[0])
    lo, hi = float(limits[0]), float(limits[1])
    for turn in (err, err - math.copysign(math.pi, err)):
        target = roll + sign * turn
        if lo <= target <= hi:
            return min(max(target, prev - SQUARE_MAX_STEP), prev + SQUARE_MAX_STEP)
    return roll


def failure_reason(
    held_no_rise: int, min_no_rise: int | None, lifted: bool, steps_in_cycle: int
) -> str | None:
    """The cycle-state triggers, read after ``trigger_reason`` finds nothing. Pure.

    ``held_no_rise`` is the count of consecutive steps a part sat between the jaws without
    rising ``oracle.LIFT_MIN_RISE`` (``_TriggerState``); at ``min_no_rise`` (None: off) the
    answer is ``HELD_NO_RISE``. Otherwise a cycle at half ``CYCLE_STEP_BUDGET`` with no lifted
    part answers ``HALF_BUDGET_NO_LIFT``."""
    if min_no_rise is not None and held_no_rise >= min_no_rise:
        return HELD_NO_RISE
    if not lifted and steps_in_cycle >= CYCLE_STEP_BUDGET // 2:
        return HALF_BUDGET_NO_LIFT
    return None


def _best_grasp(selector, scene, target: int):
    """The grasp the teacher ranks first on ``target`` alone, the way ``HeapOracle._regrasp``
    and ``aim_hover_at`` rank it, or None when no closing axis lies near the jaw plane."""
    mask = torch.zeros(selector.num_bricks, dtype=torch.bool)
    mask[target] = True
    ranked = selector._ranked(scene, mask)
    return ranked[0] if ranked else None


def cycle_close_below(selector, scene, target: int) -> float:
    """m: the continuous close threshold of a cycle on ``target``, read once when it opens.

    The grasp width the teacher would rank first on the target, the way
    ``HeapOracle.aim_hover_at`` ranks it for a takeover. When no grasp ranks (no closing
    axis near the jaw plane), the part's smallest footprint side: the width any grasp on a
    part lying flat would close across.
    """
    chosen = _best_grasp(selector, scene, target)
    if chosen is not None:
        return float(chosen.width)
    return float(min(selector.parts[target].size[:2]))


def refreshed_row(row, selector, scene, target: int) -> list[float]:
    """``row_refresh``: the target row after the teacher's REOPEN refresh. The part's live
    centre and the grasp re-ranked on it (``target_context``); when nothing ranks any more the
    served grasp columns of ``row`` stay, as ``HeapOracle._regrasp`` keeps its old grasp. Pure."""
    chosen = _best_grasp(selector, scene, target)
    if chosen is not None:
        return target_context(scene.brick_pos[target], chosen)
    return [float(v) for v in scene.brick_pos[target][:3]] + [float(v) for v in row[3:7]]


@dataclass
class Cycle:
    ordinal: int  # 1-based within the episode
    target: int
    start_step: int
    stowed_before: set[int]
    close_below: float = IDLE_CLOSE_GAP  # m, the continuous close threshold (cycle_close_below)
    takeover: int = 0  # the rig's takeover ordinal while the rig holds this cycle, else 0
    # "takeover_failed" or "takeover_declined": the rig's answer ends the cycle as a failure
    # (phase B design 5.5), after the stowed rule had its chance. A decline while the jaws
    # hold a part is not an answer: the cycle runs on (``_take_over``).
    takeover_end: str | None = None
    taken_over: bool = False  # one takeover per loop cycle (phase B design 5.5)
    start_z: torch.Tensor | None = None  # (k,) part heights when the cycle opened, for the lift
    # The census: what the driver did in this cycle. Observation only, never the metric.
    open_steps: int = 0  # steps the gripper command read open
    closed_steps: int = 0  # steps it read closed
    held_steps: int = 0  # steps a part sat between the jaws
    holding: bool = False  # a part sat between the jaws on the last step
    max_rise: float = 0.0  # m, the most a held part rose above its ``start_z``
    # cm, the xy distance from the pick point to the target at the first closed step. The
    # binary teacher flips its command on the approach, so read it on continuous runs only.
    first_close_cm: float | None = None
    # The forensics of the first close (2026-09-22), continuous runs only like first_close_cm.
    # ``first_close_geometry`` fills the geometry once, ``census_step`` the rest every step.
    heading: float | None = None  # rad, the served grasp's closing axis when the cycle opened
    first_close_at: int | None = None  # steps into the cycle at the first closed command
    gap_close: float | None = None  # m, the commanded gap at the first close
    squeeze_margin: float | None = None  # m, ``gap_close`` minus ``close_below``
    off_x: float | None = None  # m, pick point minus target centre at the first close (world)
    off_y: float | None = None
    off_z: float | None = None
    off_along: float | None = None  # m, the same xy offset along the live closing axis
    off_across: float | None = None  # m, and across it (the part's other in-plane axis)
    ctx_stale: float | None = None  # m, the served row's centre to the target's live centre
    pick_z_close: float | None = None  # m, the pick point's z at the first close
    pick_z_max: float | None = None  # m, its highest z from the first close on
    reopen_steps: int | None = None  # steps from the first close to the first re-open
    close_edges: int = 0  # open-to-closed edges of the command within the cycle
    open_prev: bool | None = None
    first_held: int = -1  # the part between the jaws at the first held step
    last_held: int = -1  # and at the last one
    wait_steps: int = 0  # the retreat wait this selection spent (``select_ready``)
    wait_capped: bool = False  # the wait ended on its cap, not on the retreat
    refreshes: int = 0  # target rows ``row_refresh`` rewrote within the cycle
    floored_steps: int = 0  # steps ``squeeze_floor`` changed the gap command
    gated_steps: int = 0  # steps ``close_gate`` changed the gap command
    squared_steps: int = 0  # steps ``square_wrist`` changed the roll command
    grace_steps: int = 0  # steps ``budget_ends_cycle``'s carry grace kept the cycle open


@dataclass
class EnvState:
    selector: object  # the HeapOracle that selects for this env
    cycle: Cycle | None = None
    outcomes: list[bool] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    stowed_parts: list[int] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    first_close_cm: list[float | None] = field(default_factory=list)
    census: list[dict] = field(default_factory=list)  # one ``CENSUS_EXPORT`` row per cycle
    takeovers: int = 0
    takeovers_stowed: int = 0
    takeovers_failed: int = 0
    done_reason: str | None = None
    idle_since: int | None = None  # the step the last cycle closed at, None before the first
    select_wait_steps: int = 0  # steps spent waiting for the retreat before a selection
    preselected: tuple | None = None  # the rig's RELEASE pre-selection a takeover handed back


def _layout_of(rig, i: int):
    layouts = getattr(rig.env, "bin_layouts", None)
    return rig.bl if layouts is None else layouts[i]


def _selector(rig, i: int):
    """A fresh HeapOracle for env ``i``, built the way ``OracleRig.resume`` builds its own."""
    return rig.oracle_factory(i, _layout_of(rig, i), oracle.nominal_draw())


def _select(st: EnvState, scene) -> int | None:
    """The next target: the pre-selection the last takeover's RELEASE left (``preselected``)
    when it still holds (``HeapOracle._revalidated_preselection``), else the selector's own
    ranking of the live scene. Either way the selector books the cycle."""
    sel = st.selector
    if st.preselected is None:
        return sel.select_target(scene)
    sel.next_brick, sel.next_brick_start, sel.next_grasp, sel._next_forced = st.preselected
    st.preselected = None
    return sel._next_brick(scene)


def cycle_end(
    cycle: Cycle,
    stowed_now: set[int],
    resting_out_of_reach: bool,
    step: int,
    credited: set[int] = frozenset(),
) -> tuple[str, list[int]] | None:
    """Why the cycle ends now and the parts it stowed, or None while it runs. Pure.

    ``resting_out_of_reach`` is ``resting_out_of_reach()`` on the target: a carry passes
    through out-of-reach space on its way to the bin, so only a part knocked out and at rest
    ends the cycle. ``credited`` holds the parts earlier cycles of the episode stowed: a part
    still moving in the bin when this cycle opened is missing from ``stowed_before``, and it
    must not end this cycle when it settles.
    """
    new = sorted(stowed_now - cycle.stowed_before - credited)
    if new:
        return "stowed", new
    if cycle.takeover_end is not None:
        return cycle.takeover_end, []
    if step - cycle.start_step >= CYCLE_STEP_BUDGET:
        return "budget", []
    if resting_out_of_reach:
        return "out_of_reach", []
    return None


def census_step(
    cycle: Cycle,
    *,
    grip_open: bool,
    held: int,
    rise: float,
    close_cm: float,
    pick_z: float = 0.0,
) -> None:
    """One control step of the cycle census. ``held`` is the loop's held read (-1 for none),
    ``rise`` the held part's height above where it lay when the cycle opened, ``close_cm``
    the xy distance from the rig's pick point to the target, kept at the first closed step,
    and ``pick_z`` the pick point's height, tracked from the first close on."""
    t = cycle.open_steps + cycle.closed_steps  # this step's index within the cycle
    if not grip_open and cycle.open_prev:
        cycle.close_edges += 1
    cycle.open_prev = grip_open
    if grip_open:
        cycle.open_steps += 1
        if cycle.first_close_at is not None and cycle.reopen_steps is None:
            cycle.reopen_steps = t - cycle.first_close_at
    else:
        cycle.closed_steps += 1
        if cycle.first_close_at is None:
            cycle.first_close_cm = close_cm
            cycle.first_close_at = t
            cycle.pick_z_close = pick_z
    if cycle.first_close_at is not None:
        cycle.pick_z_max = pick_z if cycle.pick_z_max is None else max(cycle.pick_z_max, pick_z)
    cycle.holding = held >= 0
    if cycle.holding:
        if cycle.first_held < 0:
            cycle.first_held = held
        cycle.last_held = held
        cycle.held_steps += 1
        cycle.max_rise = max(cycle.max_rise, rise)


def reopen_edge(cycle: Cycle, grip_open: bool) -> bool:
    """``row_refresh``: is this step the open edge after a close that lifted nothing, the
    policy's REOPEN? Read before ``census_step`` moves ``open_prev``. Pure."""
    return grip_open and cycle.open_prev is False and cycle.max_rise < oracle.LIFT_MIN_RISE


def first_close_geometry(
    cycle: Cycle, *, gap: float, pick, target_pos, target_quat, ctx_pos=None
) -> None:
    """The cycle's first closed command, read once: the commanded ``gap`` (m) against the
    target's grasp width, the pick point's offset from the target's centre in the world and
    along and across the live closing axis (the body axis nearest ``cycle.heading``, signed
    with it), and the served context row's centre ``ctx_pos`` against the live centre."""
    target_pos = torch.as_tensor(target_pos, dtype=torch.float32)
    off = torch.as_tensor(pick, dtype=torch.float32) - target_pos
    cycle.gap_close = gap
    cycle.squeeze_margin = gap - cycle.close_below
    cycle.off_x, cycle.off_y, cycle.off_z = (float(v) for v in off)
    if cycle.heading is not None:
        served = torch.tensor([math.cos(cycle.heading), math.sin(cycle.heading)])
        axes = grasp.body_axes_world(torch.as_tensor(target_quat, dtype=torch.float32))[:, :2]
        axis = axes[int((axes @ served).abs().argmax())]
        axis = axis / axis.norm()
        if float(axis @ served) < 0.0:
            axis = -axis
        cycle.off_along = float(off[:2] @ axis)
        cycle.off_across = float(off[1] * axis[0] - off[0] * axis[1])
    if ctx_pos is not None:
        cycle.ctx_stale = float(
            (torch.as_tensor(ctx_pos[:3], dtype=torch.float32) - target_pos).norm()
        )


CENSUS_EXPORT = (
    "max_rise",
    "held_steps",
    "closed_steps",
    "open_steps",
    "close_below",
    "first_held",
    "last_held",
    "first_close_at",
    "gap_close",
    "squeeze_margin",
    "off_x",
    "off_y",
    "off_z",
    "off_along",
    "off_across",
    "ctx_stale",
    "pick_z_close",
    "pick_z_max",
    "reopen_steps",
    "close_edges",
    "wait_steps",
    "wait_capped",
    "refreshes",
    "floored_steps",
    "gated_steps",
    "squared_steps",
    "grace_steps",
)
"""The ``Cycle`` fields each episode's ``cycle_census`` row exports, plus the two is_target
flags of ``first_held`` and ``last_held``. Observation only, never the metric."""


def cycle_class(cycle: Cycle, reason: str) -> str:
    """The census class of a closed cycle: ``stowed``, or the last stage a failed one reached.

    ``no_close``: the command never read closed. ``no_open``: it never read open.
    ``close_miss``: the jaws closed and held nothing for ``CENSUS_MIN_HELD_STEPS``.
    ``held_no_lift``: a part was held and never rose ``oracle.LIFT_MIN_RISE``. ``carrying``: a
    lifted part was still between the jaws when the cycle ended. ``lifted_lost``: a lifted part
    left the jaws and did not come to rest in the bin. Pure. The grader's ``reason`` stays the
    outcome, and the class only says how far the cycle got."""
    if reason == "stowed":
        return "stowed"
    if cycle.held_steps < CENSUS_MIN_HELD_STEPS:
        if cycle.closed_steps == 0:
            return "no_close"
        return "no_open" if cycle.open_steps == 0 else "close_miss"
    if cycle.max_rise < oracle.LIFT_MIN_RISE:
        return "held_no_lift"
    return "carrying" if cycle.holding else "lifted_lost"


def resting_out_of_reach(pos, speed: float, held: bool, bl, stow_half_xy, in_box=None) -> bool:
    """A part knocked out of reach: at rest (the grader's speed rule), not between the jaws,
    down on the desk or heap (the selector's ``CANDIDATE_MAX_Z``), outside ``layout.in_reach``
    and outside the destination bin's footprint. A slow or paused carry is none of these.

    With desk boxes ``in_box`` says whether the part lies in any box. A part at rest in its own
    box is stowed before this is asked (``cycle_end``), so one in a box here is in another
    box: out of play, like a part out of reach (the class branch names it ``wrong_box``). Off
    the boxes the part must also lie under DESK_REST_MAX_Z: a desk box's release is at
    ``CANDIDATE_MAX_Z``'s own height, where a part the opening jaws hold under the force floor
    and still read as at rest, out of reach beside the box."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if held or speed > AT_REST_SPEED or z > CANDIDATE_MAX_Z:
        return False
    if in_box is not None:
        return bool(in_box) or (z <= DESK_REST_MAX_Z and not layout.in_reach(x, y, bl))
    cx, cy, _ = layout.heap_bin_center(bl)
    in_bin = abs(x - cx) <= float(stow_half_xy[0]) and abs(y - cy) <= float(stow_half_xy[1])
    return not in_bin and not layout.in_reach(x, y, bl)


def _close_cycle(st: EnvState, reason: str, stowed: list[int], step: int, scene) -> None:
    cycle = st.cycle
    ok = reason == "stowed"
    st.outcomes.append(ok)
    st.reasons.append(reason)
    st.steps.append(step - cycle.start_step)
    st.targets.append(cycle.target)
    st.stowed_parts.extend(stowed)
    _book_census(st, cycle, reason)
    if not ok and scene is not None:
        st.selector._mark_failure(scene, cycle.target)
    st.cycle = None


def rollout(
    env,
    rig,
    policy,
    *,
    run_seed: int,
    layout_seed: int,
    permutation: tuple[int, ...],
    cycle_budget: int,
    max_wall_minutes: float,
    takeover=None,
    log=print,
    on_frame=None,
    select_at_release: bool = False,
    replan_on_switch: bool = False,
    row_refresh: bool = False,
    squeeze_floor: bool = False,
    close_gate: bool = False,
    square_wrist: bool = False,
) -> dict:
    """One lockstep rollout of ``env.num_envs`` heap episodes; returns the EVAL payload.

    ``policy`` is a ``policy_client.ServerPolicy`` or None for the oracle driver.
    ``takeover`` is a ``TakeoverConfig`` (the DAgger mode, phase B design 5.5) or None.
    The six switches are the module docstring's; ``replan_on_switch`` needs a policy whose
    ``step`` takes ``replan``, and the flag is left off the wire when the switch is off;
    ``squeeze_floor``, ``close_gate`` and ``square_wrist`` need the continuous gripper and a
    served target row.
    ``on_frame(ordinal, step_in_cycle, wrist, overhead)`` is called with env 0's two policy
    frames every ``FRAME_EVERY`` steps of each of its cycles, for a forensics dump.
    Raises ``EvalFailure`` on a scene, context, action-space or gripper mismatch, the wall
    clock and an incomplete batch, as the eval loop does.
    """
    num_envs = env.num_envs
    k = env.cfg.num_bricks
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    if list(arm_ids) != list(range(6)):
        raise EvalFailure("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    if getattr(env, "parts", None) is None:
        raise EvalFailure("scene", "the heap loop needs a heap env (parts=...), not six bricks")
    context_mode = getattr(env, "context_mode", "none")
    served_width = int(env.obs_buf["context"].shape[1]) if context_mode != "none" else 0
    gripper_mode = getattr(env, "gripper_mode", "binary")
    if policy is not None:
        wants = int(getattr(policy, "context_dim", 0))
        if wants != served_width:
            raise EvalFailure(
                "context",
                f"policy takes a {wants}-wide context, the env serves {served_width} ({context_mode})",
            )
        if getattr(policy, "action_space", "absolute") != "absolute":
            raise EvalFailure("action_space", "the heap loop drives absolute joint targets only")
        served_gripper = getattr(policy, "gripper", "binary")
        if served_gripper != gripper_mode:
            raise EvalFailure(
                "gripper", f"the checkpoint is {served_gripper}, the env serves {gripper_mode}"
            )
    if takeover is not None and policy is None:
        raise EvalFailure("takeover", "the takeover triggers need a policy to take over from")
    ctx_target = getattr(env, "context_target", None)
    # MS7a desk boxes: one destination box per part, and the served box part follows the target.
    per_part = torch.as_tensor(env.destination_bins).dim() == 2
    for name, on in (
        ("squeeze_floor", squeeze_floor),
        ("close_gate", close_gate),
        ("square_wrist", square_wrist),
    ):
        if on and (gripper_mode != "continuous" or ctx_target is None):
            raise EvalFailure(name, "needs the continuous gripper and a served target row")
    limits = robot.data.joint_pos_limits[:, arm_ids, :].to("cpu")
    tracker = RolloutTracker(num_envs, k)
    recorder = getattr(env, "recorder", None)
    rig.rearm(list(range(num_envs)), idle=True)
    env.oracle_actor = torch.zeros(num_envs, dtype=torch.int16)
    env.target_part[:] = -1
    # A takeover's frames follow the rig's marker and row (``env.marker_from_rig``, the env's
    # default): the v2 corpora switch both when RELEASE ends (``HeapOracle._preselect``), so
    # the recorded RETREAT frames carry the pre-selected part, and the loop adopts that
    # pre-selection at hand-back (``_select``). Phase B design 5.2 pinned the loop's target on
    # those frames; campaign n (2026-09-22) reverses it.
    states = [EnvState(selector=_selector(rig, i)) for i in range(num_envs)]
    for st in states:
        if st.selector.cycle_budget != cycle_budget:
            raise EvalFailure(
                "cycle_budget",
                f"the selector spends {st.selector.cycle_budget} cycles, the loop {cycle_budget}",
            )
    ordinal = [0] * num_envs  # the recorder's takeover ordinal per env
    loop_done = torch.zeros(num_envs, dtype=torch.bool)
    triggers = (
        _TriggerState(num_envs, takeover, run_seed, cycle_budget) if takeover is not None else None
    )
    switches = {
        "select_at_release": bool(select_at_release),
        "replan_on_switch": bool(replan_on_switch),
        "row_refresh": bool(row_refresh),
        "squeeze_floor": bool(squeeze_floor),
        "close_gate": bool(close_gate),
        "square_wrist": bool(square_wrist),
    }
    replan = torch.zeros(num_envs, dtype=torch.bool)  # envs whose row changed since the last step
    # The pick point's last GATE_WINDOW + 1 heights per env, the gate's descent read.
    pick_z_hist = [deque(maxlen=GATE_WINDOW + 1) for _ in range(num_envs)]
    # The roll command each env received on the last step, ``square_wrist``'s rate read.
    last_roll = robot.data.joint_pos[:, WRIST_ROLL_INDEX].to("cpu").clone()

    started = time.monotonic()
    max_steps = int(env.max_episode_length)
    step = 0
    while step < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > max_wall_minutes:
            raise EvalFailure("wall_clock", f"exceeded {max_wall_minutes} min at step {step}")
        per_brick = per_brick_stowed(env).to("cpu")
        tracker.latch_pre_step(per_brick)
        in_bin = per_brick_stowed(env, max_speed=math.inf).to("cpu")
        pos, vel = _brick_states(env)
        pos, speed = pos.to("cpu"), vel.norm(dim=-1).to("cpu")
        left_mag = env.scene["finger_left_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        right_mag = env.scene["finger_right_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
        held = held_brick_from_forces(left_mag, right_mag, min_force=HELD_MIN_FORCE).to("cpu")
        openings = policy_state(robot.data.joint_pos)[:, 6].to("cpu")
        boxed = parts_in_desk_boxes(env).to("cpu") if per_part else None
        # 1. Grade and select on the envs the loop holds: episode live, rig idle.
        for i in range(num_envs):
            if bool(tracker.done[i]) or not rig.idle[i]:
                continue
            st = states[i]
            stowed_now = {b for b in range(k) if bool(in_bin[i, b])}
            if st.cycle is not None:
                b = st.cycle.target
                out = resting_out_of_reach(
                    pos[i, b],
                    float(speed[i, b]),
                    int(held[i]) == b,
                    _layout_of(rig, i),
                    env.stow_half_extents,
                    in_box=None if boxed is None else bool(boxed[i, b]),
                )
                end = cycle_end(st.cycle, stowed_now, out, step, set(st.stowed_parts))
                if select_at_release and (end is None or end[0] == "budget"):
                    if released_after_lift(st.cycle, float(openings[i]), int(held[i])):
                        end = ("released", [])
                    elif end is not None and not budget_ends_cycle(
                        step - st.cycle.start_step, int(held[i]) >= 0, st.cycle.grace_steps
                    ):
                        st.cycle.grace_steps += 1
                        end = None
                if end is not None:
                    _close_cycle(st, end[0], end[1], step, rig.snapshot(i))
                    st.idle_since = step
            if st.cycle is not None:
                continue
            if bool(per_brick[i].all()):
                loop_done[i] = True
                st.done_reason = "all_stowed"
                continue
            if st.idle_since is not None:
                opening = float(openings[i])
                if select_at_release:
                    ready = release_ready(
                        opening, step - st.idle_since, st.reasons[-1] in RELEASE_ENDS
                    )
                else:
                    ready = select_ready(
                        rig.pick_point(i).to("cpu"),
                        opening,
                        step - st.idle_since,
                        st.selector.retreat_point,
                    )
                if not ready:
                    st.select_wait_steps += 1
                    continue
            scene = rig.snapshot(i)
            target = _select(st, scene)
            if target is None:
                loop_done[i] = True
                st.done_reason = (
                    "cycle_budget" if st.selector.cycles >= cycle_budget else "no_candidate"
                )
                continue
            waited = 0 if st.idle_since is None else step - st.idle_since
            chosen = getattr(st.selector, "grasp", None)
            st.cycle = Cycle(
                ordinal=len(st.outcomes) + 1,
                target=target,
                start_step=step,
                stowed_before=stowed_now,
                close_below=cycle_close_below(st.selector, scene, target),
                start_z=pos[i, :, 2].clone(),
                heading=None if chosen is None else float(chosen.heading),
                wait_steps=waited,
                wait_capped=waited >= int(SELECT_WAIT_FRACTION * CYCLE_STEP_BUDGET),
            )
            env.target_part[i] = target
            if per_part:
                env.set_context_box(i, int(env.destination_bins[i, target]))
            if ctx_target is not None:
                ctx_target[i] = target_context(scene.brick_pos[target], st.selector.grasp)
            replan[i] = True
            if policy is None:
                _take_over(rig, env, st, i, ordinal, reason="driver")
        close_below = torch.tensor(
            [IDLE_CLOSE_GAP if st.cycle is None else st.cycle.close_below for st in states]
        )
        # 2. The policy's action for the whole batch (its temporal ensemble stays warm).
        policy_action = None
        if policy is not None:
            # The frame the policy reads was rendered unmarked at the end of the last step,
            # and pass 1 may have moved the target since: stamp it now (phase B design 5.2).
            stamp = getattr(env, "stamp_markers", None)
            if stamp is not None:
                stamp()
            state = policy_state(robot.data.joint_pos).to("cpu").numpy()
            wrist = env.obs_buf["images"]["wrist_rgb"].to("cpu").numpy()
            overhead = env.obs_buf["images"]["overhead_rgb"].to("cpu").numpy()
            context = env.obs_buf["context"].to("cpu").numpy() if served_width else None
            c0 = states[0].cycle
            if (
                on_frame is not None
                and c0 is not None
                and (step - c0.start_step) % FRAME_EVERY == 0
            ):
                on_frame(c0.ordinal, step - c0.start_step, wrist[0], overhead[0])
            extra = {"replan": replan.numpy().copy()} if replan_on_switch else {}
            policy_action = torch.from_numpy(
                policy.step(state, wrist, overhead, context=context, **extra)
            )
            replan.zero_()
            if triggers is not None:
                grip_open = grip_open_flags(policy_action[:, 6], gripper_mode, close_below)
                triggers.offer(env, rig, states, tracker, grip_open, held, pos, ordinal, step)
        # 3. Compose: the rig's row where it drives, the policy's where the rig is idle.
        rig_action = rig.actions().to("cpu")
        action_cpu = rig_action.clone()
        picks = [rig.pick_point(i).to(pos.device) for i in range(num_envs)]
        for i in range(num_envs):
            pick_z_hist[i].append(float(picks[i][2]))
        if policy_action is not None:
            idle = torch.tensor(rig.idle)
            action_cpu[idle] = policy_action[idle]
            if close_gate or squeeze_floor or square_wrist:
                for i, st in enumerate(states):
                    if st.cycle is None or not rig.idle[i] or bool(tracker.done[i]):
                        continue
                    row = ctx_target[i]  # CONTEXT_TARGET_NAMES: z at 2, width at 5, half height 6
                    width = float(row[5])
                    gap = float(action_cpu[i, 6])
                    # ``census_step`` runs after this block, so both fields read the steps
                    # before this one: no hold and no lift so far in the cycle.
                    c = st.cycle
                    held_so_far = c.held_steps > 0 or c.max_rise >= oracle.LIFT_MIN_RISE
                    if square_wrist:
                        roll = squared_roll(
                            rig.ik,
                            action_cpu[i, :6],
                            row,
                            picks[i],
                            float(last_roll[i]),
                            limits[i, WRIST_ROLL_INDEX],
                            held=held_so_far,
                        )
                        if roll != float(action_cpu[i, WRIST_ROLL_INDEX]):
                            action_cpu[i, WRIST_ROLL_INDEX] = roll
                            c.squared_steps += 1
                    if close_gate:
                        hist = pick_z_hist[i]
                        descending = len(hist) > GATE_WINDOW and hist[0] - hist[-1] > GATE_DESCENT
                        gated = gated_gap(
                            gap,
                            width,
                            hist[-1],
                            float(row[2]) + float(row[6]),
                            descending,
                            held=held_so_far,
                        )
                        if gated != gap:
                            gap = gated
                            st.cycle.gated_steps += 1
                    if squeeze_floor:
                        floored = floored_gap(gap, width)
                        if floored != gap:
                            gap = floored
                            st.cycle.floored_steps += 1
                    action_cpu[i, 6] = gap
        signal = open_signal(action_cpu[:, 6], gripper_mode, close_below)
        tracker.record_gripper(signal)
        for i, st in enumerate(states):
            if st.cycle is None or bool(tracker.done[i]):
                continue
            c, h = st.cycle, int(held[i])
            pick = picks[i]
            grip_open = bool(signal[i] >= 0.0)
            if row_refresh and rig.idle[i] and reopen_edge(c, grip_open):
                c.refreshes += 1
                if ctx_target is not None:
                    ctx_target[i] = refreshed_row(
                        ctx_target[i], st.selector, rig.snapshot(i), c.target
                    )
                replan[i] = True
            close_cm = 0.0
            if c.first_close_at is None:
                close_cm = 100.0 * float((pick[:2] - pos[i, c.target, :2]).norm())
                if not grip_open:
                    first_close_geometry(
                        c,
                        gap=float(action_cpu[i, 6]),
                        pick=pick,
                        target_pos=pos[i, c.target],
                        target_quat=env.scene[f"brick_{c.target}"].data.root_quat_w[i].to("cpu"),
                        ctx_pos=None if ctx_target is None else ctx_target[i],
                    )
            census_step(
                c,
                grip_open=grip_open,
                held=h,
                rise=float(pos[i, h, 2] - c.start_z[h]) if h >= 0 else 0.0,
                close_cm=close_cm,
                pick_z=float(pick[2]),
            )
        arm_clamped, overshoot = clamp_arm_actions(action_cpu[:, :6], limits)
        tracker.record_clamps(overshoot)
        clamped = torch.cat([arm_clamped, action_cpu[:, 6:7]], dim=-1)
        last_roll = clamped[:, WRIST_ROLL_INDEX].clone()
        if step == max_steps - 1 and recorder is not None:
            # The time-out cuts a running takeover short of RETREAT: negate before the env's
            # step exports the episode (the six-brick DAgger loop's ruling P7).
            for i in range(num_envs):
                if int(env.oracle_actor[i]) > 0:
                    recorder.negate_actor(i, ordinal[i])
                    states[i].takeovers_failed += 1
                    env.oracle_actor[i] = 0
        _, _, terminated, truncated, _ = env.step(clamped.to(env.device))
        dones = (terminated | truncated).cpu()
        rig.observe(dones)
        # 4. Takeovers the rig handed back this step.
        for i in range(num_envs):
            result = rig.takeover_result[i]
            if result is None:
                continue
            st = states[i]
            orc = rig.oracles[i]
            if getattr(orc, "next_brick", None) is not None:
                st.preselected = (
                    orc.next_brick,
                    orc.next_brick_start,
                    orc.next_grasp,
                    orc._next_forced,
                )
            if result:
                st.takeovers_stowed += 1
            else:
                st.takeovers_failed += 1
                if recorder is not None:
                    recorder.negate_actor(i, ordinal[i])
                if st.cycle is not None:
                    # Phase B design 5.5: the loop grades the takeover's cycle like any other,
                    # so a failed hand-back ends it (next pass 1, after the stowed rule).
                    st.cycle.takeover_end = "takeover_failed"
            if st.cycle is not None:
                st.cycle.takeover = 0
            rig.takeover_result[i] = None
            env.oracle_actor[i] = 0
        # 5. Episode ends: the env's own (auto-reset) and the loop's.
        success = env.termination_manager.get_term("success").to("cpu")
        newly_env = dones & ~tracker.done
        tracker.record_step(step, dones | loop_done, truncated.cpu(), success)
        for i in torch.nonzero(newly_env).flatten().tolist():
            st = states[i]
            if int(env.oracle_actor[i]) > 0:
                # A takeover the episode's end cut short: its part is in on success (the
                # six-brick DAgger loop's rule), and its frames are already exported.
                if bool(success[i]):
                    st.takeovers_stowed += 1
                else:
                    st.takeovers_failed += 1
            if st.cycle is not None:
                if bool(success[i]):
                    # Every part is in: the cycle stowed all the parts not credited before.
                    new = set(range(k)) - st.cycle.stowed_before - set(st.stowed_parts)
                    _close_cycle(st, "stowed", sorted(new), step + 1, None)
                else:
                    _close_cycle_unmarked(st, step + 1)
            st.done_reason = st.done_reason or ("all_stowed" if bool(success[i]) else "time_out")
            rig.rearm([i], idle=True)
            env.oracle_actor[i] = 0
        if recorder is not None:
            # An episode the loop ends (selector dry or cycle budget spent) never reaches the
            # env's own end: export its takeover frames here. No takeover runs on a loop-done
            # env (pass 1 sees only idle rigs), and ``end`` is a no-op on an env already ended.
            for i in torch.nonzero(loop_done).flatten().tolist():
                recorder.end(i, states[i].done_reason == "all_stowed")
        loop_done.zero_()
        step += 1
        if step % 500 == 0:
            log(
                f"[loop] step {step}/{max_steps} done={int(tracker.done.sum())}/{num_envs} "
                f"cycles={sum(len(s.outcomes) for s in states)}",
                flush=True,
            )

    wall = time.monotonic() - started
    if not tracker.all_done:
        raise EvalFailure(
            "incomplete", f"{int((~tracker.done).sum())} envs still running at {max_steps}"
        )
    episodes = []
    for i, result in enumerate(tracker.results):
        st = states[i]
        final = tracker.stowed_mask(i)
        episodes.append(
            {
                **result.to_json(),
                "cycles": len(st.outcomes),
                "cycle_outcomes": list(st.outcomes),
                "cycle_reasons": list(st.reasons),
                "cycle_steps": list(st.steps),
                "cycle_targets": list(st.targets),
                "stowed_parts": list(st.stowed_parts),
                "cycle_classes": list(st.classes),
                "cycle_first_close_cm": list(st.first_close_cm),
                "cycle_census": list(st.census),
                "stowed_families": sorted(env.parts[p].family for p in range(k) if final[p]),
                "done_reason": st.done_reason,
                "takeovers": st.takeovers,
                "takeovers_stowed": st.takeovers_stowed,
                "takeovers_failed": st.takeovers_failed,
                "select_wait_steps": st.select_wait_steps,
            }
        )
    steps_total = sum(e["steps"] for e in episodes)
    cycles_total = sum(e["cycles"] for e in episodes)
    stowed_total = sum(e["bricks_stowed"] for e in episodes)
    payload = {
        "run_seed": run_seed,
        "layout_seed": layout_seed,
        "permutation": [int(v) for v in permutation],
        "num_envs": num_envs,
        "num_bricks": k,
        "heap_k": k,
        "parts": [p.name for p in env.parts],
        "cycle_budget": cycle_budget,
        "cycle_step_budget": CYCLE_STEP_BUDGET,
        "policy_id": "oracle" if policy is None else policy.policy_id,
        "n_action_steps": None if policy is None else policy.n_action_steps,
        "inference": {} if policy is None else {**policy.inference, "loop": switches},
        "context": context_mode,
        "gripper": gripper_mode,
        "episode_draws": bool(getattr(env.cfg, "episode_draws", False)),
        "frame_jitter": bool(getattr(env.cfg, "frame_jitter", False)),
        "spatial_scale": float(getattr(env.cfg, "spatial_scale", 1.0)),
        "episodes": episodes,
        "steps_run": step,
        "cycles_total": cycles_total,
        "stowed_total": stowed_total,
        "cycles_per_stowed": (cycles_total / stowed_total) if stowed_total else None,
        "takeovers": sum(e["takeovers"] for e in episodes),
        "takeovers_stowed": sum(e["takeovers_stowed"] for e in episodes),
        "takeovers_failed": sum(e["takeovers_failed"] for e in episodes),
        "triggers": {} if triggers is None else dict(triggers.counts),
        "takeovers_declined": 0 if triggers is None else triggers.declined,
        "clamp_fraction": (
            sum(e["clamp_steps"] for e in episodes) / steps_total if steps_total else 0.0
        ),
        "wall_sec": round(wall, 1),
        "wall_sec_per_episode": round(wall / num_envs, 1),
    }
    return payload


def _book_census(st: EnvState, cycle: Cycle, reason: str) -> None:
    st.classes.append(cycle_class(cycle, reason))
    close_cm = cycle.first_close_cm
    st.first_close_cm.append(None if close_cm is None else round(close_cm, 2))
    row = {name: getattr(cycle, name) for name in CENSUS_EXPORT}
    row["first_held_is_target"] = cycle.first_held == cycle.target
    row["last_held_is_target"] = cycle.last_held == cycle.target
    st.census.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def _close_cycle_unmarked(st: EnvState, step: int) -> None:
    """An episode the env ended (time-out) closes its open cycle without a scene to mark."""
    cycle = st.cycle
    st.outcomes.append(False)
    st.reasons.append("episode_end")
    _book_census(st, cycle, "episode_end")
    st.steps.append(step - cycle.start_step)
    st.targets.append(cycle.target)
    st.cycle = None


def _take_over(
    rig, env, st: EnvState, i: int, ordinal: list[int], *, reason: str, holding: bool = False
) -> bool:
    """Hand env ``i`` to the rig on the loop's target; False when the rig declines.

    A decline with nothing held ends the cycle as a failure (phase B design 5.5: "the cycle
    ends as a failure, and the loop moves on"); ``_close_cycle`` marks the selector's failure
    memory. A decline while the jaws hold a part (``holding``, the loop's held read) keeps the
    six-brick behavior: the cycle runs on and the caller's cooldown spaces the next trigger.
    ``OracleRig.resume`` declines most of a correct carry (RULING P32: a part above
    ``LIFT_HEIGHT - LIFT_MIN_RISE``), and ending the cycle there would spend it and mark the
    failure memory at a point in the air."""
    if rig.resume(i, target=st.cycle.target):
        st.cycle.taken_over = True
        ordinal[i] += 1
        st.takeovers += 1
        st.cycle.takeover = ordinal[i]
        env.oracle_actor[i] = ordinal[i]
        return True
    if not holding:
        st.cycle.taken_over = True
        st.cycle.takeover_end = "takeover_declined"
    return False


class _TriggerState:
    """Per-env trigger counters and the offer of a takeover: the six-brick DAgger loop's logic
    on the loop's cycles (the MS4 four triggers, the random draw, the decline cooldown) plus
    the half-budget trigger (phase B design 5.5). ``offer`` runs once per step before the
    actions are composed, on the envs the policy holds. Caps (design 5.5): one takeover per
    loop cycle, ``max_per_episode_factor * cycle_budget`` per episode."""

    def __init__(self, num_envs: int, config: TakeoverConfig, run_seed: int, cycle_budget: int):
        self.config = config
        self.max_per_episode = math.floor(config.max_per_episode_factor * cycle_budget)
        self.rng = random.Random(run_seed)
        self.random_p = 0.0 if config.random_interval is None else 1.0 / config.random_interval
        self.since_open = [0] * num_envs
        self.closed_no_pinch = [0] * num_envs
        self.held_no_rise = [0] * num_envs  # steps a part sat held without the rise
        self.cooldown = [0] * num_envs
        self.cycle_start = [-1] * num_envs  # the open step of the cycle the last offer saw
        self.grip_open_prev: torch.Tensor | None = None
        self.counts: Counter = Counter()
        self.declined = 0

    def offer(self, env, rig, states, tracker, grip_open, held, pos, ordinal, step) -> None:
        """``grip_open`` is the policy's (N,) open read on the loop's per-env close threshold
        and ``held`` the loop's per-step held read, so the triggers and the grader agree."""
        robot = env.scene["robot"]
        tool = robot.find_bodies(TOOL_BODY)[0][0]
        nearest_cm = _nearest_brick_cm(
            pos,
            robot.data.body_pos_w[:, tool, :2].to("cpu"),
            held,
            env_heap_bin_centers(env, "cpu").to("cpu"),
            torch.as_tensor(env.stow_half_extents).to("cpu"),
        )
        for i in range(env.num_envs):
            st = states[i]
            if bool(tracker.done[i]) or not rig.idle[i] or st.cycle is None:
                continue
            open_i = bool(grip_open[i])
            # A new cycle brings a new close threshold (``cycle_close_below``), so the previous
            # step's open read was taken against another threshold: a close edge across the
            # boundary would compare two different rules. The first step of a cycle has no edge.
            # The reset is unconditional on purpose. Under the binary gripper the rule is the
            # command's sign, which no boundary changes, so there the guard drops one real close
            # edge and with it one close_far opportunity per cycle. That is the cheaper side:
            # closed_no_pinch reads the same stall a few steps later, and threading the gripper
            # mode in here to make the guard exact costs more than the trigger it saves.
            new_cycle = self.cycle_start[i] != st.cycle.start_step
            self.cycle_start[i] = st.cycle.start_step
            if new_cycle:
                self.held_no_rise[i] = 0  # the rise is read against the new cycle's start_z
            closed_now = (
                not new_cycle
                and self.grip_open_prev is not None
                and bool(self.grip_open_prev[i])
                and not open_i
            )
            self.since_open[i] = self.since_open[i] + 1 if open_i else 0
            no_pinch = not open_i and int(held[i]) < 0
            self.closed_no_pinch[i] = self.closed_no_pinch[i] + 1 if no_pinch else 0
            if self.cooldown[i] > 0:
                self.cooldown[i] -= 1
                continue
            reason = trigger_reason(
                grip_open=open_i,
                closed_now=closed_now,
                held=int(held[i]),
                nearest_cm=nearest_cm[i],
                pick_point=rig.pick_point(i).tolist(),
                since_open=self.since_open[i],
                closed_no_pinch=self.closed_no_pinch[i],
                random_draw=self.rng.random(),
                random_p=self.random_p,
            )
            h = int(held[i])
            lifted = h >= 0 and float(pos[i, h, 2] - st.cycle.start_z[h]) >= oracle.LIFT_MIN_RISE
            self.held_no_rise[i] = self.held_no_rise[i] + 1 if h >= 0 and not lifted else 0
            if reason is None:
                reason = failure_reason(
                    self.held_no_rise[i],
                    self.config.held_no_rise_steps,
                    lifted,
                    step - st.cycle.start_step,
                )
            if reason is None or st.cycle.taken_over or st.takeovers >= self.max_per_episode:
                continue
            # Reset on both branches: an answered trigger must not fire again off the same count.
            self.since_open[i] = 0
            self.closed_no_pinch[i] = 0
            self.held_no_rise[i] = 0
            if _take_over(rig, env, st, i, ordinal, reason=reason, holding=int(held[i]) >= 0):
                self.counts[reason] += 1
            else:
                # The decline cooldown stays (phase B design 5.5). A decline on a held part
                # leaves the cycle running and the cooldown spaces the next trigger on it; a
                # decline with nothing held ends the cycle, and the cooldown spaces the next
                # cycle's first trigger from a scene the rig just refused.
                self.declined += 1
                self.cooldown[i] = DECLINE_COOLDOWN
        self.grip_open_prev = grip_open

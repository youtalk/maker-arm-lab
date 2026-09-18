"""Scripted pick-and-place oracle: per-env phase sequencing (spec section 2.2, ported).

Torch + stdlib only -- NO isaaclab imports. The GPU binding that turns Goals
into joint targets lives in tasks/manager_based/lego_sort/oracle_runtime.py,
which plans each Goal as a `core.global_ik` path; this module owns WHAT to do
next, not how to get there.

Release constants: TCP (bin_x - 0.02, y_bin, 0.30), tool tilted from above the
rim. RELEASE_Z, RELEASE_X_SETBACK and the bin geometry they are measured
against were carried over from OpenYAM's LegoSort oracle and are CONFIRMED on
the Maker Arm by Task D4 (`scripts/release_envelope_check.py`), unchanged.
RELEASE_TILT_RAD is the one that had to move: OpenYAM's 0.70 rad put the
release pose 117-160 mm above this arm's reach ceiling, and D4 re-derived it at
1.40 rad. See its own comment for the measurement.
"""

import enum
import math
import random
from dataclasses import dataclass

import torch

from . import checker, layout

# Height the brick is released from, above the bin interior. Carried over from
# OpenYAM and CONFIRMED UNCHANGED on the Maker Arm by Task D4
# (`scripts/release_envelope_check.py`): 40/40 driven pick-and-place trials, every bin at
# the nominal layout and at all four bin-jitter corners, converged EXTEND and stowed.
#
# It survives because it sits near the MIDDLE of the band RELEASE_TILT_RAD opens, and the
# two ends of that band pull in opposite directions (both measured, see below):
#
#   floor    0.257 m -- the jaw plate's far corner hangs JAW_FACE_FAR * sin(tilt) = 61.6 mm
#                       below the tool tip at 1.40 rad, and it has to clear the bin rim at
#                       layout._BIN_FLOOR_TOP_Z + BIN_INTERIOR[2] = 0.195.
#   ceiling  ~0.345 m -- the arm's own reach at the release attitude, at the FARTHEST
#                       release target the jitter can produce (bin_x 0.46, y +-0.23, so
#                       radius 0.497 m). Extrapolated one bin past the sampled table below;
#                       what CONFIRMS the pose is reachable is the driven gate, not this.
#
# So do not move this without re-reading RELEASE_TILT_RAD: lowering it fouls the rim,
# raising it leaves the envelope, and changing the tilt moves both bounds at once.
RELEASE_Z = 0.30

# Tool-axis tilt off world -z held through RAISE, EXTEND, RELEASE and RETREAT.
#
# MEASURED ON THIS ARM by Task D4 (`scripts/release_envelope_check.py`), 2026-09-06.
# OpenYAM's 0.70 rad was carried in and is REFUTED HERE: it does not merely cost margin on
# the Maker Arm, it puts the release pose OUTSIDE the arm's envelope altogether.
#
# What 0.70 did, driven for real at the nominal layout: EXTEND ran its full 300-step budget
# and ended 116.4-160.2 mm short of the release target, at all four bins. The failure has
# none of the usual signatures -- no joint within 0.19 rad of a stop, nothing clamped, the
# commanded joint targets within 0.007 rad of the measured joints, attitude converged to
# 0.13-0.40 deg. The arm simply stops: on two of the four trials the pick point at step 300
# is bit-identical to the pick point at step 200.
#
# The cause is the reach ceiling AT THIS ATTITUDE, and it is a strong function of the tilt.
# Sweeping the vendored URDF's FK over random joint configurations, keeping poses whose tool
# axis is within 3 deg of the exact axis the runtime commands -- (sin t, 0, -cos t), the lean
# azimuth pinned to +x, which is what `global_ik.tool_axis_world` builds today and what
# `desired_quat` built before Task D5, identically -- and with every joint at least 0.15 rad
# from its stops, the highest the PICK POINT can go (m) is:
#
#     radius   t=0.70  t=0.90  t=1.00  t=1.10  t=1.20  t=1.40
#      0.42     0.171   0.243   0.276   0.307   0.338   0.400
#      0.46     0.138   0.216   0.254   0.284   0.316   0.383
#      0.48     0.125   0.190   0.239   0.266   0.308   0.358
#
# The farthest release target the bin jitter can produce is radius 0.497 m, where 0.70 rad
# tops out around 0.12 m against a RELEASE_Z of 0.30 -- which is the 160 mm the live run
# measured, so the sweep and the arm agree. More tilt buys height because at full extension
# the arm is nearly straight and a tool aligned with the forearm costs the wrist far less
# travel than a near-vertical one; 1.20 rad only just reaches 0.30 there, and 1.40 rad clears
# it by roughly 45 mm.
#
# So the tilt is pushed up until the ceiling clears RELEASE_Z with real margin, and then
# stopped. What stops it is NOT the jaw plate: that floor tops out at 0.195 + JAW_FACE_FAR =
# 0.257 m even at a horizontal tool, always under RELEASE_Z. It is the attitude frame.
# The tool axis is (sin t, 0, -cos t) and the jaw axis is constrained to the horizontal
# heading w, and those two directions become PARALLEL at t = pi/2, w = 0 -- there is then no
# jaw axis perpendicular to the tool with heading w, and the commanded attitude is undefined.
# That was a Gram-Schmidt cross product going to zero when `oracle_runtime.desired_quat`
# assembled the frame; since Task D5 it is `core.global_ik`'s jaw-heading residual row going
# singular. Same pose, same bound. 1.40 rad keeps 0.17 rad (9.8 deg) away from it, which the
# driven gate confirms is enough; a value chosen closer to pi/2 for more reach would be
# trading a measured margin for an unmeasured singularity.
#
# Confirmed driven, 2026-09-06: 40/40 trials (5 layouts x 4 bins x 2), EXTEND converging in
# 203-222 steps of its 300-step budget with 1.5-7.9 mm of residual against an 8 mm stop, the
# tightest joint -- link_006_joint on every one of the 40 -- 0.78-1.28 rad from its stop, and
# every brick landing 15.7-42.3 mm from its bin's interior center.
RELEASE_TILT_RAD = 1.40
RELEASE_X_SETBACK = 0.02
RETREAT_POINT = (0.26, 0.0, 0.28)
MAX_GRASP_RETRIES = 2

# Phase budgets, in 30 Hz control steps. HOVER/EXTEND cover the longest legs
# OpenYAM measured (HOME -> bin took 67-226 steps in OpenYAM's Task 9);
# DESCEND relies on its stall detector, CLOSE is OpenYAM's measured 45-step
# jaw stroke + margin. Carried over unchanged, and RE-VALIDATED ON THIS ARM by
# Task D5 (`scripts/oracle_check.py`, 6 six-brick episodes on the global-IK
# planner): every leg that could be planned finished inside its budget with
# room -- HOVER 101-180, DESCEND 83-105, LIFT 109, EXTEND 105-179, RETREAT
# 94-147 -- so DEFAULT_MAX_STEPS' 300 is not tight anywhere. The one phase that
# spends its whole budget is RAISE, and it does so without moving; see
# RAISE_MAX_STEPS below.
SETTLE_STEPS = 240
CLOSE_STEPS = 45
DEFAULT_MAX_STEPS = 300
DESCEND_STALL_STEPS = 12
LIFT_HEIGHT = 0.10

# RAISE's own target height and budget -- OpenYAM MEASURED this (Task 8), not
# the RELEASE_Z a first reading of "raise" suggests.
#
# RAISE_CLEARANCE_Z IS DELIBERATELY NOT LOWERED, AND TASK D5 MEASURED WHY BOTH
# HALVES OF THAT SENTENCE ARE NEEDED. Holding the pick point over the BRICK's own
# xy at this height is out of reach for part of the scatter band whatever the
# tool attitude: sweeping the band (32 positions inside SCATTER_MAX_RADIUS x 4
# jaw headings) against tilts from 30 to 80 deg, the reachable fraction of
# (position, heading) pairs runs 0 % at 30 deg, 49 % at 40, 68 % at 60, 72 % at
# 75 and 66 % at the release tilt, and the band's near corners -- (0.13, +-0.18),
# (0.16, +-0.18) -- are unreachable at every tilt tried (/tmp/d5_raise_tilt.py).
# So no tilt makes this phase plannable band-wide, and the tilt is a WEAK lever on
# RAISE itself even though it is a decisive one on the leg after it (see
# RAISE_TILT_RAD below).
#
# Lowering the height is what would make RAISE plannable, and it is exactly the
# wrong move: about 0.16 m does it, which is BELOW the bin rim
# (layout._BIN_FLOOR_TOP_Z + BIN_INTERIOR[2] = 0.195) that this phase exists to
# clear. That would satisfy the solver by deleting the phase's purpose. The
# residue is handled where it belongs instead: the runtime advances straight to
# EXTEND on a leg it cannot plan, rather than holding still for the budget (see
# oracle_runtime's `_phase_result`).
#
# The rim is then cleared by EXTEND itself WHEN EXTEND PLANS STRAIGHT -- its
# Cartesian leg interpolates the pick point from the lift height to RELEASE_Z, so
# it rises monotonically and is past 0.195 m well before it reaches the bin faces.
# That is a property of the Cartesian planner and NOT of the joint-space fallback
# underneath it (`oracle_runtime._plan_direct`), which reaches the same endpoint by
# interpolating joints and makes no promise about the path between. The fallback is
# counted per episode (`episode_stats()["direct_legs"]`) precisely so this
# distinction is observable rather than assumed.
#
# The rest of this comment is OpenYAM's, carried over unchanged with its arm and
# layout, and it is what the constants were chosen for. Climbing straight to RELEASE_Z
# (0.30, near the reach ceiling) while holding a fixed attitude and NOT
# translating ran joint4 (tilt=0.0, the original design) or joint6
# (tilt=RELEASE_TILT_RAD applied all at once, from a pick whose desk yaw
# happens to be unfavorable) onto its stop before the climb finished, well
# short of a leg EXTEND's own translate+climb+tilt convergence handled
# cleanly (OpenYAM measured: EXTEND alone, from LIFT's pose, reaches the
# release pose in 100-230 steps with real joint margin). RAISE_CLEARANCE_Z
# (0.22) only needs to clear the bin rim
# (layout._BIN_FLOOR_TOP_Z + BIN_INTERIOR[2] = 0.195) before EXTEND starts
# translating over the bins -- it is deliberately NOT RELEASE_Z; EXTEND climbs
# the remaining ~80 mm together with the lateral move and the tilt. Applying
# RELEASE_TILT_RAD here too (not 0.0) still matters: it gives joint4 the same
# margin during this partial climb that EXTEND's own approach needs, and
# avoids a second abrupt tilt jump between RAISE and EXTEND on top of the one
# already coming from LIFT. RAISE_MAX_STEPS (60) is short ON PURPOSE: a real
# reach-ceiling contact (RAISE's own tilt application, at an unfavorable pick
# yaw, OpenYAM measured pinning joint6 within 30-40 steps) must end this phase
# on "budget" -- report() advances to EXTEND regardless of RAISE's outcome --
# before OracleRig's watchdog's 30-consecutive-near-limit trip fires and
# abandons the episode instead.
#
# That budget sizing still holds on this arm, for a different reason worth
# stating: an unreachable RAISE is now not driven at all (the planner finds no
# path and the arm holds), so it ends on "budget" with every joint stationary and
# the watchdog, which trips on joint-limit proximity, cannot fire during it. When
# RAISE IS reachable it walks its planned leg and converges in about 36 steps, at
# 1.6-8.3 mm against RAISE_STOP_POS -- so 60 is not tight from that side either.
RAISE_CLEARANCE_Z = 0.22
RAISE_STOP_POS = 0.02
RAISE_MAX_STEPS = 60

# Tool-axis tilt RAISE climbs at. NOT RELEASE_TILT_RAD, which it used to reuse.
#
# MEASURED ON THIS ARM by Task D5, and the measurement is about the leg AFTER this one.
# RAISE's job is to clear the bin rim; only RELEASE needs the release attitude. Sweeping the
# raise tilt and then asking whether EXTEND can be planned from the pose RAISE leaves --
# the release-envelope check's own pick, an identity-yaw brick at (0.20, 0.0), so the jaw
# heading is pinned near +-90 deg (/tmp/d5_raise_then_extend.py):
#
#     raise tilt   EXTEND to y=+-0.21   EXTEND to y=+-0.07
#      40 deg           OK                   OK
#      50 deg           OK                   OK
#      60 deg           OK                   OK
#      70 deg           FAIL                 OK
#      80.2 deg         FAIL                 FAIL      <- what RELEASE_TILT_RAD gave
#      (no RAISE)       OK                   OK
#
# At the release tilt RAISE leaves the arm in a pose from which NO continuous EXTEND exists
# to any bin, at either heading -- driven, that is exactly what
# scripts/release_envelope_check.py measured under the shipped value: EXTEND ran its full
# 300-step budget and ended 312-315 mm short at both outer bins, with link_003_joint 0.145 rad
# from its stop. 60 deg is the largest tested value that keeps every bin plannable, with the
# cliff at 70; it is also at the practical peak of RAISE's OWN reachability (see
# RAISE_CLEARANCE_Z's comment above), so nothing is traded for it.
#
# What this value is NOT chosen to do is make RAISE plannable band-wide, because no value
# does: the tilt lever moves RAISE's own coverage of the scatter band only from 66 % to 72 %
# of (position, heading) pairs across 30-80 deg, and the band corners are out of reach at
# every tilt. The runtime advances straight to EXTEND on the legs it cannot plan.
RAISE_TILT_RAD = 1.05

PICK_TILT_RAD = 0.0
"""Tool-axis tilt the pick phases hold. MS6 rework rung R5 tried to raise it off zero.

At 0.0 the jaws come down flat, so a part offers only the body axes that are within
`grasp.AXIS_MAX_TILT_RAD` of horizontal, and a lying plate offers its in-plane axes alone.
Raising it moves the reach ceiling, the joint6 budget of `fold_jaw_heading` and the descend
clearance model at the same time, which is why the rung ran last, and why it ran twice.

The ladder read two non-zero values and rejected both, so the constant stays at 0.0.

25.0 deg moved the commanded pose past `robots.frames.GRASP_TILT_DEG` (20.0 deg today), the
runtime's own floor on the hover-approach tilt (`oracle_runtime`'s
`max(goal.tilt_rad, GRASP_TILT_RAD)`, and `_plan_hover_ladder`'s hardcoded use of it). That
moved attitude cost a grasp attempt in the bottom-pick probe and halved the heap teacher's
completed cycles inside a fixed step budget.

20.0 deg equals `GRASP_TILT_RAD` itself, so it never moves the commanded pose -- every
request at or below the runtime's floor lands on the pose it already holds today. That floor
comes from the D3 sweep's 18.42 deg worst case at the outer corners of the scatter band
(`robots/frames.py`'s own comment), rounded up to the nearest whole degree. Holding the pose
did not hold the grasp model steady, though: `grasp.candidate_grasps` measures a candidate
closing axis against the JAW PLANE, and at 20.0 deg that plane is no longer level.
`AXIS_MAX_TILT_RAD` (15.0 deg) was calibrated against a level plane, so tilting the plane
20.0 deg while keeping that same 15.0 deg band refuses every closing axis that lies along the
tilt direction -- a whole orientation class of parts the level-plane model used to accept.
Driven, a three-part heap that stowed completely at 0.0 deg stalled at 20.0 deg:
`sort_attempts` 2 of 3, `retries` 12, `unplanned_legs` 0, so the IK solver could still reach
the part and the grasp model was the one that refused to offer it a grip. The arm never
moved; only the model's own opinion of what it can hold did, so this is a modeling gap, not a
measured hardware limit.

This constant does not go non-zero again until `grasp.AXIS_MAX_TILT_RAD` is re-measured
against a tilted jaw plane at the matching tilt, and that measured-constant change is outside
this rung's scope. Until then 0.0 is the value every test, the bottom-pick probe and the heap
teacher all still pass at, and it is the value that ships.

`core/` never imports `robots.frames` -- it is the ported, arm-agnostic state machine, and
the runtime owning the arm's floor is the whole reason that floor lives there instead of
here -- so this duplicates the number rather than importing it, and the two must be read
together, not assumed to track each other automatically.
"""

_JOINT_LIMIT_GUARD_RAD = 0.05
_JOINT_SPEED_MAX = 6.0

# Margin from each joint6 stop that ``fold_jaw_heading``'s PREDICTED span must
# keep before the near representative is accepted. Deliberately NOT the
# watchdog's 0.05: that band is about a measured joint position, this one is
# about a prediction, and OpenYAM MEASURED (139 fold decisions over an
# instrumented 8-env run, on OpenYAM's arm) that the prediction UNDER-shoots
# the joint6 a leg actually reaches. Over the 15 legs whose predicted span
# landed within 0.30 rad of a stop -- the only ones this band governs -- four
# undershot, by +0.064, +0.082, +0.131 and +0.210 rad; the +0.064 one is a leg
# that was measured to pin joint6 at +2.0944 during RAISE and lose the
# episode to the watchdog while its predicted 2.030 sat "inside" a 0.05 band.
# So a prediction that clears a stop by less than roughly its own error is
# not evidence of clearance.
#
# 0.10 and not more: raising it further started flipping legs where BOTH
# representatives were predicted to violate (measured at 0.15: two more
# flips, one of them decided by a 0.007 rad difference between two losing
# options), which is a coin toss dressed as a decision. Flipping is never a
# downgrade -- the fold only switches when the far representative violates
# strictly less -- so erring toward this band costs at most a wrist
# half-turn at HOVER, with the jaws open and no brick held.
#
# Carried over unchanged from OpenYAM; no task in this plan re-measures this
# magnitude for the Maker Arm's own joint6 servo behavior -- Task D5's
# full-episode runs (`scripts/oracle_check.py`) are what would surface a
# watchdog abandon if it is wrong here.
_FOLD_GUARD_RAD = 0.10


@dataclass(frozen=True)
class OracleNoise:
    """Range config for per-episode noise draws (spec section 2.3, Datagen-v0 DR)."""

    hover_height: tuple[float, float] = (0.10, 0.14)
    lateral_offset: float = 0.003
    yaw_offset: float = 0.0873  # 5 deg
    gain_scale: tuple[float, float] = (0.7, 1.0)
    action_noise_std: float = 0.05  # raw-action units, translation channels
    via_jitter: float = 0.02  # RETREAT via-point jitter, m
    # Corrective approaches (MS3 recipe design 5.1, ported from openyam-lab): a per-pick
    # lateral offset of the HOVER target, radius uniform in this range at a uniform angle,
    # drawn every time HOVER is (re)entered for a target. DESCEND keeps the brick's true
    # pose, so every approach demonstrates nulling a lateral error on camera. (0, 0) is off.
    # Deliberately outside --noise_scale, like randomize_pick_order: a separate mechanism,
    # and folding it in would make every recipe string ambiguous about what it carried.
    hover_jitter: tuple[float, float] = (0.0, 0.0)
    # Pick the next target uniformly at random among the not-yet-stowed bricks instead of
    # taking the lowest index. Off by default, so every MS1 recipe stays byte-identical.
    #
    # ``_next_brick`` scans indices in order and brick color is ``index % 4``, so every
    # demonstration in a lowest-index corpus picks R,B,Y,G,R,B: a fixed color SCHEDULE a
    # policy can follow open-loop without reading which brick it carries. openyam-lab
    # measured exactly that (D2, 120 episodes, 2026-08-15: 8.3% wrong-bin on-schedule vs
    # 42.0% off-schedule) and randomizing the order was its largest gate lever (W4,
    # per-brick 0.372 -> 0.626). With the order randomized there is no schedule to fit, so
    # a policy that delivers correctly can only be conditioning on the carried brick. This
    # removes a shortcut from the data; it adds no signal. Deliberately outside
    # ``--noise_scale``: it is a separate mechanism, and folding it in would make every
    # recipe string ambiguous about whether its episodes carried the schedule.
    randomize_pick_order: bool = False
    # Take the candidate nearest the tool instead, the third order (MS3 visual-inspection
    # plan, lever 5). Mutually exclusive with ``randomize_pick_order``.
    #
    # The randomized order broke the color schedule but left the next target unreadable:
    # the 2026-09-11 probes on the t2 policy found 30-45% of the closes 2 cm or further
    # from any brick sitting BETWEEN two bricks under 10 cm apart, which is what averaging
    # over candidates the image cannot disambiguate looks like. Nearest-first keeps the
    # schedule broken -- the desk scatter is random, so the color order still varies per
    # episode -- and makes the target a function of the scene the cameras see. Distance is
    # measured in the desk plane from ``SceneSnapshot.pick_point``: the bricks all sit at
    # desk height, so the tool's own height is not part of which one is closest. Like the
    # other two orders it is outside ``--noise_scale``.
    nearest_first_pick_order: bool = False


@dataclass(frozen=True)
class EpisodeDraw:
    hover_height: float
    lateral_dx: float
    lateral_dy: float
    yaw_offset: float
    gain_scale: float
    action_noise_std: float
    via_dx: float
    via_dy: float
    # Carried as a RANGE, not a sampled value: the HOVER offset is drawn per PICK (see
    # ``EnvOracle._draw_hover_jitter``), not per episode, so every one of an episode's six
    # approaches demands its own correction.
    hover_jitter_range: tuple[float, float] = (0.0, 0.0)
    # Carried as the FLAG, not as a sampled permutation: the target is drawn per pick from
    # whatever is still unstowed (see ``EnvOracle._next_brick``), so a brick that a retry, a
    # failed grasp or an unreachable pose takes off the table never has to be edited out of
    # a pre-drawn order.
    randomize_pick_order: bool = False
    # Carried as the FLAG for the same reason: the nearest candidate is re-read from the
    # live scene at every pick, never fixed at episode start.
    nearest_first_pick_order: bool = False


def sample_episode_draw(rng: random.Random, noise: OracleNoise) -> EpisodeDraw:
    return EpisodeDraw(
        hover_height=rng.uniform(*noise.hover_height),
        lateral_dx=rng.uniform(-noise.lateral_offset, noise.lateral_offset),
        lateral_dy=rng.uniform(-noise.lateral_offset, noise.lateral_offset),
        yaw_offset=rng.uniform(-noise.yaw_offset, noise.yaw_offset),
        gain_scale=rng.uniform(*noise.gain_scale),
        action_noise_std=noise.action_noise_std,
        via_dx=rng.uniform(-noise.via_jitter, noise.via_jitter),
        via_dy=rng.uniform(-noise.via_jitter, noise.via_jitter),
        hover_jitter_range=noise.hover_jitter,
        randomize_pick_order=noise.randomize_pick_order,
        nearest_first_pick_order=noise.nearest_first_pick_order,
    )


def nominal_draw() -> EpisodeDraw:
    """The draw a DAgger takeover plans with (MS4 c2 design 6.2): the middle hover height,
    no offsets, no noise, no hover jitter, no induction, and the nearest-first pick order the
    corpus recipe uses."""
    return EpisodeDraw(
        hover_height=0.12,
        lateral_dx=0.0,
        lateral_dy=0.0,
        yaw_offset=0.0,
        gain_scale=1.0,
        action_noise_std=0.0,
        via_dx=0.0,
        via_dy=0.0,
        nearest_first_pick_order=True,
    )


@dataclass(frozen=True)
class FailureInduction:
    """Deliberate first-attempt grasp failures, recovered by the existing retry.

    Applied per pick with probability ``prob`` when HOVER selects a new
    target brick. Only the first attempt (``retries_used == 0``) is
    affected; the REOPEN retry runs with clean targeting, which is what
    makes the episode succeed and export.

    Offset-style sabotage (see ``EnvOracle._sabotage_offset``) is drawn
    perpendicular to the jaw closing axis, +-30 deg jitter, not a uniform
    random direction: on OpenYAM's arm, the first live probe found
    uniform-direction offsets almost never caused a failure, because
    OpenYAM's parallel jaws open ~9.5 cm and any offset with a large
    component along the closing axis gets funneled -- swept back into the
    grip as the jaws close. Only offsets along the jaw line (perpendicular to
    closing) reliably miss. The jaw geometry this reasoning depends on is
    OpenYAM's, carried over unvalidated for the Maker Arm's own jaw span.
    """

    prob: float = 0.0
    offset_weight: float = 0.7  # offset style vs noise style
    offset_range: tuple[float, float] = (0.015, 0.025)  # m, radial
    noise_boost: float = 5.0  # action_noise_std multiplier, DESCEND/CLOSE


@dataclass
class SceneSnapshot:
    """Ground-truth state for one env, world frame (env-origin subtracted)."""

    brick_pos: torch.Tensor  # (B, 3)
    brick_yaw: torch.Tensor  # (B,)
    brick_lin_vel: torch.Tensor  # (B, 3)
    pick_point: torch.Tensor  # (3,)
    # (B, 4) wxyz part orientations. None on the six-brick task, which reads only the yaw;
    # the heap teacher's grasp synthesis needs the full rotation (MS6 design 7.1).
    brick_quat: torch.Tensor | None = None


class Phase(enum.Enum):
    SETTLE = enum.auto()
    HOVER = enum.auto()
    DESCEND = enum.auto()
    CLOSE = enum.auto()
    LIFT = enum.auto()
    REOPEN = enum.auto()
    RAISE = enum.auto()
    EXTEND = enum.auto()
    RELEASE = enum.auto()
    RETREAT = enum.auto()
    DONE = enum.auto()
    ABANDON = enum.auto()
    PUSH_HOVER = enum.auto()
    PUSH_DESCEND = enum.auto()
    PUSH_SWEEP = enum.auto()
    TIP_HOVER = enum.auto()
    TIP_DESCEND = enum.auto()
    TIP_SWEEP = enum.auto()


# Explicit pin, NOT Phase.value: the recorded integers are a dataset schema
# that must survive any future reordering of the Phase enum. SETTLE/DONE/ABANDON
# (0/10/11) are re-declared locally in lerobot_stack/convert_to_lerobot.py -- that
# venv cannot import this module, so nothing enforces the two copies agree;
# renumber here and update that file's SETTLE_INT/DONE_INT/ABANDON_INT by hand.
PHASE_TO_INT = {
    "SETTLE": 0,
    "HOVER": 1,
    "DESCEND": 2,
    "CLOSE": 3,
    "LIFT": 4,
    "REOPEN": 5,
    "RAISE": 6,
    "EXTEND": 7,
    "RELEASE": 8,
    "RETREAT": 9,
    "DONE": 10,
    "ABANDON": 11,
    # MS6 heap teacher (design 7.3). Appended, never renumbered: the converter drops 0/10/11
    # by pinned integer and keeps everything else as in-cycle frames.
    "PUSH_HOVER": 12,
    "PUSH_DESCEND": 13,
    "PUSH_SWEEP": 14,
    # MS6 rework section 7. Appended, never renumbered, for the same reason as the pushes.
    "TIP_HOVER": 15,
    "TIP_DESCEND": 16,
    "TIP_SWEEP": 17,
}

PUSH_PHASES = frozenset({Phase.PUSH_HOVER, Phase.PUSH_DESCEND, Phase.PUSH_SWEEP})
TIP_PHASES = frozenset({Phase.TIP_HOVER, Phase.TIP_DESCEND, Phase.TIP_SWEEP})
"""The tip-up: a corrective like a push, and recorded by the same actor ordinal."""


@dataclass
class Goal:
    phase: Phase
    target_pos: torch.Tensor | None
    # Brick long-axis heading; None = hold the current attitude. The runtime
    # squares the jaws across this axis (it folds ``yaw + pi/2``, the actual
    # closing axis, and hands that to its solver as the jaw heading).
    yaw: float | None
    tilt_rad: float  # tool-axis tilt toward +x (0 = straight down)
    gripper: float  # +1 open / -1 close (raw binary action)
    stop_pos: float | None
    stall_steps: int
    max_steps: int
    hold_steps: int
    # Every xy waypoint the heading commanded by ``yaw`` will still be held at,
    # for ``fold_jaw_heading``'s joint6 budget. Only the yaw-setting phases
    # (HOVER/DESCEND) fill this; the runtime turns it into a base-yaw range.
    # It is the oracle's to supply because only the oracle knows the rest of
    # the plan -- which bin this brick is bound for, and that nothing re-yaws
    # between here and RETREAT.
    yaw_hold_xy: tuple[tuple[float, float], ...] = ()
    # Half the target's vertical extent (m). The runtime's descend ladder derives its pick
    # offset from it (core.global_ik.descend_pick_offset). The six-brick task always grasps a
    # resting 2x4; a heap part supplies its own (MS6 design 7.5).
    half_height: float = layout.BRICK_SIZE[2] / 2


@dataclass
class PhaseResult:
    # "converged" | "stalled" | "budget" | "unplanned". The last is the runtime reporting
    # that it could not build a path for this leg at all, so the phase ends at once instead
    # of holding still until its budget expires -- idle frames go into the dataset and teach
    # a policy to pause. Nothing branches on this field except the runtime's own telemetry;
    # `report()` below reads only pinch_ok/lifted.
    outcome: str
    pinch_ok: bool | None = None
    lifted: bool | None = None


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def fold_jaw_heading(
    want: float,
    jaw_yaw: float,
    joint6: float,
    joint6_limits: tuple[float, float],
    base_yaw_range: tuple[float, float] = (0.0, 0.0),
    guard_rad: float = _FOLD_GUARD_RAD,
    *,
    roll_sign: float = -1.0,
    base_sign: float = 1.0,
) -> float:
    """Pick the half-turn representative of ``want`` that joint6 can hold all the way.

    A jaw is symmetric under a half turn, so ``want`` and ``want + pi`` are the
    same physical grasp -- but they are not the same wrist pose, and joint6
    spans only +-2.0944 rad, narrower than the +-pi ambiguity the fold reasons
    over. Choosing purely by "which representative is the wrist already nearer"
    (raw ``wrap_to_pi`` proximity, what this fold did before) picks an
    unreachable heading whenever the wrist starts far enough round that the
    nearer representative sits outside joint6's remaining travel.

    OpenYAM MEASURED this relation on its own arm (two instrumented full
    oracle runs of 13437 and 14920 control steps, ``layout_seed=0 run_seed=0
    num_envs=1``): with the tool axis at tilt 0 the jaw's world heading was

        jaw_yaw = joint1 - joint6 + c,   c = 1.5747 +- 0.0248 rad (n = 4169)

    i.e. ``roll_sign = -1`` and ``base_sign = +1`` in the general form below.
    ``roll_sign`` is d(jaw_yaw)/d(joint6) and ``base_sign`` is
    d(jaw_yaw)/d(joint1), each +-1. The defaults reproduce that OpenYAM
    relation exactly; the Maker Arm runtime passes
    ``robots.frames.WRIST_ROLL_SIGN`` / ``BASE_YAW_SIGN``, measured for this
    arm by ``scripts/scripted_grasp_check.py --signs`` (Task D3). With the
    defaults, holding a commanded heading while the base yaw moves by ``d``
    costs joint6 exactly

        required(d) = joint6 + d - wrap_to_pi(want - jaw_yaw)

    -- byte-for-byte OpenYAM's original formula.

    ``base_yaw_range`` is a RANGE, not a single value, because ``Goal.yaw`` is
    None for every phase from CLOSE through RETREAT: the heading a pick commits
    to at HOVER is still held while EXTEND swings out to the bin and RETREAT
    comes back, and joint6 pays for that whole excursion. OpenYAM found this
    the hard way when it first measured the single-point version -- checking
    only the reach to the brick moved the abandon from the second brick's
    HOVER to the third brick's EXTEND, joint6 pinned at the OTHER (+2.0944)
    stop, same mechanism one phase later.

    The two fatal cases this is pinned against, both OpenYAM measurements from
    those runs, on OpenYAM's arm, under the default signs:

    * ``RETREAT -> HOVER``, brick 2, control step 13380: ``jaw_yaw=3.0434``,
      ``joint6=-1.5179``, ``base_yaw_range=(-0.703, 0.0)``. The nearer
      representative (``want=3.7790``, only 0.736 rad of heading away) needs
      joint6 in ``[-2.957, -2.254]``, a full radian past its -2.0944 stop; the
      far one needs ``[0.184, 0.885]``. Unfixed, joint6 walked onto that stop in
      29 control steps and stayed pinned, and the watchdog abandoned 30 strikes
      later at HOVER step 56 of its 300-step budget.
    * ``HOVER``, brick 3, control step 14402: near needs ``[1.658, 2.123]``, far
      ``[-1.484, -1.019]``. Unfixed, joint6 pinned at +2.0944 during EXTEND.

    RETREAT is what sets the first one up: it deliberately holds the previous
    brick's release heading (see the RETREAT goal's comment), so the wrist
    enters the next HOVER already 1.5 rad round.

    Deliberately conservative -- the near representative is kept unless it is
    predicted to violate ``guard_rad`` AND the far one violates it less, so a
    flip is never a downgrade. OpenYAM measured this, over a 139-fold
    instrumented 8-env run on its own arm: it fired on 7 of 139 decisions;
    every other one was bit-identical to the old proximity fold. The
    prediction is a good estimate, not a bound (the servo overshoots the
    waypoint bearings in transit, and ``c`` shifts ~0.07 rad under
    RELEASE_TILT_RAD), which is what ``_FOLD_GUARD_RAD`` is sized for and why
    this is used only to veto, never to seek.

    What it does NOT fix, OpenYAM measured in the same run: a pose where BOTH
    representatives are out. ``jaw_yaw=3.057``, ``joint6=-1.458``, a -0.876 rad
    base swing to the brick -- near needs ``[-2.328, -1.420]``, and the far one
    is not the mirror image at ``+0.81`` but ``[-5.470, -4.561]``, because the
    attitude servo walks the SHORT way round and for a near-pi heading change
    the short way is the losing direction. That episode still abandons. Fixing
    it needs the wrist re-centred before HOVER (RETREAT holds the previous
    brick's heading by design), not a better fold.

    Args:
        want: Any representative of the desired jaw heading; folded here.
        jaw_yaw: The jaw's current world heading.
        joint6: The wrist-roll joint's current position, rad.
        joint6_limits: ``(lower, upper)`` stops for that joint, rad.
        base_yaw_range: ``(min, max)`` signed joint1 excursion, relative to the
            current pose, over every waypoint this heading will be held across.
        guard_rad: Margin from each stop that counts as a violation. Defaults to
            ``_FOLD_GUARD_RAD`` (0.10), deliberately wider than the watchdog's
            own 0.05 -- see that constant's comment for the measurement, and
            ``tests/test_oracle.py`` for the fatal case 0.05 lets through.
        roll_sign: d(jaw_yaw)/d(joint6), +-1. Defaults to -1.0, the OpenYAM
            relation.
        base_sign: d(jaw_yaw)/d(joint1), +-1. Defaults to +1.0, the OpenYAM
            relation.

    Returns:
        The chosen representative, in the same unwrapped convention the caller
        passed in (callers hold it across phases, so it is not re-wrapped).
    """
    near = want + math.pi * round(wrap_to_pi(jaw_yaw - want) / math.pi)
    far = near + math.pi
    lower, upper = joint6_limits
    # Wrist excursion needed to hold a heading across the base swing: joint6 changes by
    # -base_sign * swing / roll_sign per radian of base yaw.
    k = -base_sign / roll_sign
    swing_lo, swing_hi = sorted((k * base_yaw_range[0], k * base_yaw_range[1]))

    def overshoot(heading: float) -> float:
        held = joint6 + roll_sign * wrap_to_pi(heading - jaw_yaw)
        return max(
            lower + guard_rad - (held + swing_lo), (held + swing_hi) - (upper - guard_rad), 0.0
        )

    near_overshoot = overshoot(near)
    if near_overshoot == 0.0 or overshoot(far) >= near_overshoot:
        return near
    return far


def watchdog_trips(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    limits: torch.Tensor,
    guard_rad: float = _JOINT_LIMIT_GUARD_RAD,
    speed_max: float = _JOINT_SPEED_MAX,
) -> bool:
    """Self-collision is off (OpenYAM's M1 ruling, carried over here via the
    task cfg's ``enabled_self_collisions=False``): abort evidence is
    joint-space only.

    Args:
        joint_pos: (6,) arm joint positions.
        joint_vel: (6,) arm joint velocities.
        limits: (6, 2) lower/upper joint limits.
    """
    near = (joint_pos - limits[:, 0] < guard_rad) | (limits[:, 1] - joint_pos < guard_rad)
    return bool(near.any() or (joint_vel.abs() > speed_max).any())


class EnvOracle:
    """Sequences one env's episode: settle, then per brick pick->place."""

    def __init__(
        self,
        bl: layout.BinLayout,
        num_bricks: int,
        draw: EpisodeDraw,
        induction: FailureInduction | None = None,
        rng: random.Random | None = None,
    ):
        self.bl = bl
        self.num_bricks = num_bricks
        self.draw = draw
        self.phase = Phase.SETTLE
        self.target_brick: int | None = None
        self.retries_used = 0
        self._brick_start: torch.Tensor | None = None
        self._sorted: set[int] = set()
        # Set by `resume()` when it enters LIFT directly; consumed by `next_goal()`'s LIFT
        # branch on the very next call, which is that same entry (see its comment).
        self._resumed_lift = False
        self.induction = induction
        self.rng = rng
        self._induced_style: str | None = None
        self._sabotage_radius = 0.0
        self._sabotage_side = 1.0
        self._sabotage_jitter = 0.0
        self.induced_offset = 0
        self.induced_noise = 0
        self._hover_dx = 0.0
        self._hover_dy = 0.0
        # Corrective approaches this episode actually demonstrated (one per HOVER entry with
        # the knob on), the per-episode counterpart of run_meta's configured range.
        self.hover_jitters = 0
        # Every NEW target this episode selected, in the order it selected them (a retry
        # re-attempts the same brick and is NOT appended). Recorded whether or not the order
        # is randomized: the run_meta flag says what was configured, this says what the
        # episode actually did, and only the second can catch a campaign that advertises a
        # randomized schedule and shipped the fixed one.
        self.pick_order: list[int] = []
        # The width the jaws close across for the current pick (m). The runtime's LIFT gate
        # derives the pinch stall floor from it (core.checker.pinch_min_stall); the six-brick
        # task always squares the jaws to the 2x4's short side. A heap subclass sets it per pick.
        self.grasp_width = layout.BRICK_SIZE[1]
        self.matched_jaw_opening = False
        """Hold the jaws at ``grasp_width``'s own opening on the approach, not fully open.

        Off here, so the six-brick task keeps the fully open descend its gated corpus was
        recorded with. ``HeapOracle`` turns it on (MS6 rung 1)."""
        self.accepts_any_held_part = False
        """May a cycle carry off a part the jaws hold that is not the target?

        Off here: the six-brick task grades by colour, so a brick in the wrong jaws belongs in
        another bin, and the gated corpus was recorded with this LIFT. ``HeapOracle`` turns it
        on, where one bin takes every part. The runtime's LIFT gate reads it
        (``oracle_runtime.OracleRig._lift_result``)."""
        # One entry per completed pick cycle: did the brick reach its bin (RETREAT's read).
        # The heap batch loop negates the recorded frames of a False cycle (MS6 design H7).
        self.cycle_outcomes: list[bool] = []
        if draw.randomize_pick_order and draw.nearest_first_pick_order:
            # Two orders cannot both be the order, and a run_meta advertising both would
            # leave no reader able to say which one the episodes carried.
            raise ValueError("only one pick order may be selected")
        if draw.randomize_pick_order and rng is None:
            # A corpus whose run_meta claims a randomized schedule and whose episodes all
            # pick R,B,Y,G,R,B is worse than a crash.
            raise ValueError("randomize_pick_order requires an rng")
        if draw.hover_jitter_range[1] > 0.0 and rng is None:
            # Silently dropping the jitter would produce a corpus whose run_meta claims
            # corrective approaches and whose episodes contain none. Refuse instead.
            raise ValueError("hover_jitter_range requires an rng")

    def _draw_hover_jitter(self) -> None:
        """Draw this pick's HOVER-only lateral offset, uniform radius in the range.

        Called at every point that (re)sets ``_brick_start``: a fresh target and a retry
        both re-enter HOVER and both deserve a fresh correction to demonstrate. The radius
        is uniform in the range rather than area-uniform, because the quantity swept is the
        LATERAL ERROR the descent has to null, sampled evenly over that error.
        """
        lo, hi = self.draw.hover_jitter_range
        if hi <= 0.0:
            self._hover_dx = self._hover_dy = 0.0
            return
        radius = self.rng.uniform(lo, hi)
        angle = self.rng.uniform(-math.pi, math.pi)
        self._hover_dx = radius * math.cos(angle)
        self._hover_dy = radius * math.sin(angle)
        self.hover_jitters += 1

    @property
    def abandoned(self) -> bool:
        return self.phase == Phase.ABANDON

    def noise_factor(self) -> float:
        """Multiplier on the runtime's per-step action noise, for noise-style
        induced picks: boosted only on the first attempt's approach (DESCEND)
        and grip (CLOSE); the retry runs at nominal noise."""
        if (
            self._induced_style == "noise"
            and self.retries_used == 0
            and self.phase in (Phase.DESCEND, Phase.CLOSE)
        ):
            return self.induction.noise_boost
        return 1.0

    @property
    def sorted_bricks(self) -> frozenset[int]:
        return frozenset(self._sorted)

    def resume(self, scene: SceneSnapshot, held: int) -> None:
        """Enter the machine mid-episode from a state a policy reached (MS4 c2 design 6.2).

        Two rows. A pinched brick resumes at LIFT, with the brick start at its live pose so
        the rise gate measures from now; LIFT's own pinch and rise gate then routes a bad
        pinch into the normal REOPEN retry. Anything else resumes at HOVER with no target,
        and HOVER's entry picks the nearest candidate, or answers DONE or ABANDON.
        """
        self._sorted = self._stowed(scene)
        self.retries_used = 0
        self._induced_style = None
        if held >= 0:
            self.target_brick = held
            self._brick_start = scene.brick_pos[held].clone()
            self.pick_order.append(held)
            self.phase = Phase.LIFT
            # Flags this one LIFT entry for next_goal()'s live-height clamp; see its comment.
            self._resumed_lift = True
        else:
            self.target_brick = None
            self.phase = Phase.HOVER

    # -- brick bookkeeping ---------------------------------------------------

    def _stowed(self, scene: SceneSnapshot) -> set[int]:
        color_ids = torch.arange(self.num_bricks) % len(layout.BRICK_COLOR_ORDER)
        ok = checker.bricks_at_rest_in_bins(
            scene.brick_pos.unsqueeze(0),
            scene.brick_lin_vel.unsqueeze(0),
            layout.bin_interior_centers(self.bl),
            layout.bin_half_extents(),
            color_ids,
        )[0]
        return {i for i in range(self.num_bricks) if bool(ok[i])}

    def _next_brick(self, scene: SceneSnapshot) -> int | None:
        """Select this pick's target among the bricks still on the table.

        Lowest index by default. With ``draw.randomize_pick_order`` the choice is uniform
        over the SAME candidate set, and with ``draw.nearest_first_pick_order`` it is the
        candidate closest to the tool in the desk plane: changing the order must not relax
        the reachability filter or let a stowed brick be picked twice, so the candidate
        list is built first and only the choice among it changes. The choice is per pick
        rather than a permutation fixed at episode start: candidacy is re-evaluated against
        the live scene every time, so a brick that rolls out of reach mid-episode simply
        stops being a candidate.
        """
        limit = layout.top_down_x_limit(self.bl)
        stowed = self._stowed(scene)
        candidates = [
            index
            for index in range(self.num_bricks)
            if index not in self._sorted
            and index not in stowed
            and scene.brick_pos[index, 0].item() <= limit
        ]
        if not candidates:
            return None
        if self.draw.nearest_first_pick_order:
            return min(
                candidates,
                key=lambda i: float((scene.brick_pos[i, :2] - scene.pick_point[:2]).norm()),
            )
        if not self.draw.randomize_pick_order:
            return candidates[0]
        return self.rng.choice(candidates)

    def _release_target(self, brick: int) -> torch.Tensor:
        color = brick % len(layout.BRICK_COLOR_ORDER)
        slot = layout.color_slot(self.bl, color)
        return torch.tensor([self.bl.bin_x - RELEASE_X_SETBACK, self.bl.bin_ys[slot], RELEASE_Z])

    # -- hooks a subclass overrides (MS6 heap teacher) -------------------------

    def _target_yaw(self, scene: SceneSnapshot) -> float:
        """The long-axis heading HOVER and DESCEND square the jaws to, plus the draw offset."""
        return scene.brick_yaw[self.target_brick].item() + self.draw.yaw_offset

    def _target_half_height(self) -> float:
        return layout.BRICK_SIZE[2] / 2

    def _no_target_phase(self, scene: SceneSnapshot) -> Phase:
        """Where HOVER goes when no target is left: done if everything is stowed, else abandon."""
        if self._stowed(scene) | self._sorted >= set(range(self.num_bricks)):
            return Phase.DONE
        return Phase.ABANDON

    # -- goal construction ---------------------------------------------------

    def _goal(
        self,
        phase,
        target,
        yaw,
        tilt,
        gripper,
        stop,
        stall,
        steps,
        hold=0,
        hold_xy=(),
        half_height=None,
    ):
        return Goal(
            phase=phase,
            target_pos=target,
            yaw=yaw,
            tilt_rad=tilt,
            gripper=gripper,
            stop_pos=stop,
            stall_steps=stall,
            max_steps=steps,
            hold_steps=hold,
            yaw_hold_xy=hold_xy,
            half_height=self._target_half_height() if half_height is None else half_height,
        )

    def _sabotage_offset(self, yaw: float) -> tuple[float, float]:
        """Offset along the jaw LINE (heading ``yaw``), so the jaws close beside
        the brick instead of funneling it in.

        ``yaw`` here is the brick's long-axis heading (``Goal.yaw``); the
        runtime squares the jaws to it and commands the CLOSING axis at
        ``yaw + pi/2`` (``oracle_runtime._plan_leg``). The parallel jaws
        open far wider than the sabotage radius, so an offset along the
        closing axis is swept back into the grip -- the sabotage must
        therefore stay ON the jaw line (``yaw``, not ``yaw + pi/2``),
        perpendicular to closing, where the fingers do not sweep (OpenYAM
        measured this on its own arm's first live probe -- 0 retries from
        ~10 induced picks; not re-measured on the Maker Arm)."""
        angle = yaw + (1.0 - self._sabotage_side) * math.pi / 2.0 + self._sabotage_jitter
        return self._sabotage_radius * math.cos(angle), self._sabotage_radius * math.sin(angle)

    def _yaw_hold_xy(self, target: torch.Tensor) -> tuple[tuple[float, float], ...]:
        """Every xy this pick's commanded jaw heading is still held at.

        ``Goal.yaw`` is set at HOVER and DESCEND and then None all the way to
        RETREAT, so the heading chosen here is the heading EXTEND swings out to
        the bin with and RETREAT comes home with. joint6 pays for that whole
        base-yaw excursion (see ``fold_jaw_heading``), so the fold has to see
        the bin and the retreat point, not just the brick.
        """
        release = self._release_target(self.target_brick)
        return (
            (float(target[0]), float(target[1])),
            (float(release[0]), float(release[1])),
            (RETREAT_POINT[0] + self.draw.via_dx, RETREAT_POINT[1] + self.draw.via_dy),
        )

    def descend_goal(self, yaw: float) -> Goal:
        """The DESCEND leg for the current pick: straight down onto the brick.

        ``yaw`` is the brick's long-axis heading plus the episode's yaw offset -- the same
        value the HOVER goal carried, which is how the runtime's HOVER-entry lookahead calls
        this: it plans the descend from a candidate hover pose to decide whether that hover
        is worth walking to (see ``oracle_runtime.OracleRig._plan_leg``), and hands back the
        yaw of the HOVER goal it is planning. Public for that caller; ``next_goal`` uses it
        too, so both ask for the same leg. DESCEND_CLEARANCE_MEASURED is applied by the
        runtime (it owns the robots import); the core target is the brick's resting z.
        """
        target = self._brick_start.clone()
        target[0] += self.draw.lateral_dx
        target[1] += self.draw.lateral_dy
        if self._induced_style == "offset" and self.retries_used == 0:
            sdx, sdy = self._sabotage_offset(yaw)
            target[0] += sdx
            target[1] += sdy
        return self._goal(
            Phase.DESCEND,
            target,
            yaw,
            PICK_TILT_RAD,
            1.0,
            0.0008,
            DESCEND_STALL_STEPS,
            DEFAULT_MAX_STEPS,
            hold_xy=self._yaw_hold_xy(target),
        )

    def extend_goal(self) -> Goal:
        """The EXTEND leg for the brick in the jaws: out to its bin's release pose.

        Public, and separate from ``next_goal``, because the runtime plans this leg twice:
        once for real at EXTEND entry, and once as a LOOKAHEAD at RAISE entry, to refuse a
        RAISE that would leave the arm where no EXTEND can be planned (see
        ``oracle_runtime.OracleRig._plan_leg``). Both must ask for the same leg.
        """
        return self._goal(
            Phase.EXTEND,
            self._release_target(self.target_brick),
            None,
            RELEASE_TILT_RAD,
            -1.0,
            0.008,
            0,
            DEFAULT_MAX_STEPS,
        )

    def next_goal(self, scene: SceneSnapshot) -> Goal | None:
        if self.phase in (Phase.DONE, Phase.ABANDON):
            return None
        if self.phase == Phase.SETTLE:
            return self._goal(
                Phase.SETTLE, None, None, 0.0, 1.0, None, 0, SETTLE_STEPS, hold=SETTLE_STEPS
            )
        if self.phase == Phase.HOVER:
            if self.target_brick is None:
                self.target_brick = self._next_brick(scene)
                self.retries_used = 0
                if self.target_brick is None:
                    # Nothing reachable left: done if everything is stowed,
                    # abandoned if an unreachable/lost brick remains.
                    self.phase = self._no_target_phase(scene)
                    return None
                self.pick_order.append(self.target_brick)
                self._brick_start = scene.brick_pos[self.target_brick].clone()
                self._draw_hover_jitter()
                self._induced_style = None
                self._sabotage_radius = 0.0
                self._sabotage_side = 1.0
                self._sabotage_jitter = 0.0
                if (
                    self.induction is not None
                    and self.induction.prob > 0.0
                    and self.rng.random() < self.induction.prob
                ):
                    if self.rng.random() < self.induction.offset_weight:
                        self._induced_style = "offset"
                        self._sabotage_radius = self.rng.uniform(*self.induction.offset_range)
                        self._sabotage_side = 1.0 if self.rng.random() < 0.5 else -1.0
                        self._sabotage_jitter = self.rng.uniform(-math.pi / 6.0, math.pi / 6.0)
                        self.induced_offset += 1
                    else:
                        self._induced_style = "noise"
                        self.induced_noise += 1
            target = self._brick_start.clone()
            target[0] += self.draw.lateral_dx + self._hover_dx
            target[1] += self.draw.lateral_dy + self._hover_dy
            target[2] += self.draw.hover_height
            yaw = self._target_yaw(scene)
            if self._induced_style == "offset" and self.retries_used == 0:
                sdx, sdy = self._sabotage_offset(yaw)
                target[0] += sdx
                target[1] += sdy
            return self._goal(
                Phase.HOVER,
                target,
                yaw,
                PICK_TILT_RAD,
                1.0,
                0.005,
                0,
                DEFAULT_MAX_STEPS,
                hold_xy=self._yaw_hold_xy(target),
            )
        if self.phase == Phase.DESCEND:
            return self.descend_goal(self._target_yaw(scene))
        if self.phase == Phase.CLOSE:
            return self._goal(
                Phase.CLOSE, None, None, PICK_TILT_RAD, -1.0, None, 0, CLOSE_STEPS, hold=CLOSE_STEPS
            )
        if self.phase == Phase.LIFT:
            target = scene.pick_point.clone()
            if self._resumed_lift:
                # `resume` (MS4 c2 design 6.2) can enter LIFT directly, with the brick
                # already carried PARTWAY up by the policy it took over from -- and only
                # partway: the runtime declines a takeover whose brick is already above
                # LIFT_HEIGHT (ruling P32, `OracleRig.resume`), because from there this
                # clamp would command a DESCENT with the jaws closed, fail the rise gate
                # below and drop the brick into the corpus as an expert demonstration. So
                # this branch only ever lowers a target that is still under the clearance.
                # The plain "live height plus LIFT_HEIGHT" rule below would then ask
                # for a second rise stacked on top of that already-raised height, past the
                # solver's reach. Only THIS entry into LIFT -- the one `resume()` just
                # requested -- gets the clamp, read as an absolute clearance above the desk
                # (layout.py: "desk top surface at z = 0") instead of a further rise. The
                # normal CLOSE->LIFT transition is untouched: the tool is at the brick's
                # near-zero resting height there, by construction, so this branch would
                # resolve to the same number anyway, and a repository-wide switch to
                # "clearance above the desk" as LIFT's rule belongs to its own change.
                target[2] = min(target[2] + LIFT_HEIGHT, LIFT_HEIGHT)
                self._resumed_lift = False  # one-shot: consumed at this LIFT entry
            else:
                target[2] += LIFT_HEIGHT
            return self._goal(
                Phase.LIFT, target, None, PICK_TILT_RAD, -1.0, 0.002, 0, DEFAULT_MAX_STEPS, hold=30
            )
        if self.phase == Phase.REOPEN:
            return self._goal(Phase.REOPEN, None, None, 0.0, 1.0, None, 0, 20, hold=20)
        if self.phase == Phase.RAISE:
            target = scene.pick_point.clone()
            target[2] = RAISE_CLEARANCE_Z
            return self._goal(
                Phase.RAISE,
                target,
                None,
                RAISE_TILT_RAD,
                -1.0,
                RAISE_STOP_POS,
                0,
                RAISE_MAX_STEPS,
            )
        if self.phase == Phase.EXTEND:
            return self.extend_goal()
        if self.phase == Phase.RELEASE:
            return self._goal(
                Phase.RELEASE, None, None, RELEASE_TILT_RAD, 1.0, None, 0, 20, hold=20
            )
        if self.phase == Phase.RETREAT:
            # tilt_rad=RELEASE_TILT_RAD, not 0.0 (level): OpenYAM measured
            # (Task 8) that snapping straight back to level, on top of the
            # position change from the release pose to RETREAT_POINT, was its
            # own joint4/joint6-exhausting abrupt attitude change on that arm
            # -- the same class of problem RAISE's tilt fix above addresses,
            # mirrored. Holding the release tilt through the whole retreat
            # converged cleanly instead; releveling is deferred to the next
            # brick's HOVER (goal.yaw is set there), which starts from
            # RETREAT_POINT -- a far friendlier pose for that change than the
            # release pose is. Carried over unchanged, and CONFIRMED ON THIS
            # ARM by Task D5 (`scripts/oracle_check.py`): RETREAT is the leg
            # nothing had driven before -- release_envelope_check.py stops at
            # RELEASE -- and it converged on RETREAT_POINT in every one of 41
            # legs, 91-147 steps of its 300, 0.7-2.5 mm of residual against a
            # 10 mm stop, with the tightest joint 0.58-0.61 rad from its stop.
            # D4's prediction that RETREAT_POINT was ALSO out of reach at the
            # old 0.70 rad tilt (its sweep put the ceiling at radius 0.26 at
            # 0.242 m, under the 0.28 m target) is consistent with this: at
            # 1.40 rad the same point is reached with room to spare.
            target = torch.tensor(RETREAT_POINT)
            target[0] += self.draw.via_dx
            target[1] += self.draw.via_dy
            return self._goal(
                Phase.RETREAT, target, None, RELEASE_TILT_RAD, 1.0, 0.01, 0, DEFAULT_MAX_STEPS
            )
        raise AssertionError(f"unhandled phase {self.phase}")

    # -- transitions ---------------------------------------------------------

    def report(self, result: PhaseResult, scene: SceneSnapshot) -> None:
        if self.phase == Phase.SETTLE:
            self.phase = Phase.HOVER
        elif self.phase == Phase.HOVER:
            self.phase = Phase.DESCEND
        elif self.phase == Phase.DESCEND:
            # Stall IS the expected outcome: the plates land on the desk.
            self.phase = Phase.CLOSE
        elif self.phase == Phase.CLOSE:
            self.phase = Phase.LIFT
        elif self.phase == Phase.LIFT:
            # MS6 rework section 4: the decision is the measured rise, not the pinch
            # factors. Heap 0 of 2026-09-17 rejected 8 of 26 real holds on the force floor,
            # the asymmetry limit and the stall floor's own boundary, while the part rose
            # 90 mm or more in the jaws. `pinch_ok` stays on the result as telemetry.
            if result.lifted:
                self.phase = Phase.RAISE
            elif self.retries_used < MAX_GRASP_RETRIES:
                self.retries_used += 1
                self.phase = Phase.REOPEN
            else:
                self.phase = Phase.ABANDON
        elif self.phase == Phase.REOPEN:
            self._brick_start = scene.brick_pos[self.target_brick].clone()
            self._draw_hover_jitter()
            self.phase = Phase.HOVER  # same target_brick, fresh brick pose
        elif self.phase == Phase.RAISE:
            self.phase = Phase.EXTEND
        elif self.phase == Phase.EXTEND:
            self.phase = Phase.RELEASE
        elif self.phase == Phase.RELEASE:
            self.phase = Phase.RETREAT
        elif self.phase == Phase.RETREAT:
            # Sorted means ARRIVED, not attempted. A brick that is not in its bin when the
            # arm has retreated was dropped somewhere -- MEASURED on GPU (2026-09-07): an
            # EXTEND with no plan, RELEASE opening over the scatter band, the brick flat on
            # the desk -- and writing it off here left the oracle DONE with a brick on the
            # desk, idling ~1500 control steps to the episode's time_out. Left unsorted, it
            # is picked again by HOVER's selection if it still lies inside the top-down
            # band, and the episode is abandoned at once ("no reachable brick") if not.
            # `_sorted` still records the bricks that did arrive, so a later delivery that
            # jostles one -- momentarily not at rest -- cannot un-sort it.
            arrived = self.target_brick in self._stowed(scene)
            if arrived:
                self._sorted.add(self.target_brick)
            self.cycle_outcomes.append(arrived)
            self.target_brick = None
            self.phase = Phase.HOVER  # HOVER's entry selects the next brick

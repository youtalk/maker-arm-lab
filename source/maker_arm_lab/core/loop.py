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
the served row's closing axis on the final approach (``squared_roll``). A seventh,
``scripted_place`` (MS7a design, the fallback), hands the cycle to the teacher once the policy
holds a part ``SCRIPTED_PLACE_RISE`` above where it lay: the rig resumes at LIFT and carries
the part to its own box (``OracleRig.resume`` with ``start_z``). An eighth, ``scripted_rake``
(MS7a tray), runs the teacher's rake when the selector finds no target but plans a rake off a
tray wall, as the oracle driver always does, and hands the env back to the policy: the policy
never learned to rake, so without it a read ends at the first wall-stuck state. A ninth,
``part_failures`` (MS7a), books a failed cycle against its target part instead of the part's
5 cm cell (``HeapOracle.failure_scope``): a policy that fails a third of its picks closed
every cell of the 13 x 20 cm tray before its cycle budget, and the episode ended
``no_candidate`` with parts in reach. A tenth, ``pick_snap`` (MS7a), moves the commanded pick
point onto the served row on the final approach: its xy toward the row centre and, with the
jaws open over the part, its height down to the teacher's grasp height (``snap_shift``), the
jaws held at the landing gap while it comes down (``snap_hold``). An eleventh,
``dwell_close`` (MS7a), commands the teacher's close when the policy waits at grasp height
with the jaws open for ``DWELL_STEPS`` (``dwell_count``). A twelfth, ``adopt_held``
(MS7a), serves the box and marker of a lifted part that is not the target, as the teacher
does when its LIFT comes up with another part (``HeapOracle.adopt_held_part``). Two rake
settings follow (MS7a rake prototype, 2026-09-24): ``rake_cap`` N lets the selector rake one
part N times instead of ``RAKE_PART_CAP`` and scales the episode's rake cap with it, and
``rake_crowd`` lets a rake start land its tips up to ``grasp.RAKE_CROWD_M`` into a neighbour
when no start is clear. ``park_done`` (MS7a tail forensics, 2026-09-24) holds an env whose
episode has ended at the last command it received: the batch runs in lockstep until its last
episode ends, and without it the policy keeps driving the finished env on the stale row of its
last cycle, a motion the grader never reads. ``rake_turn`` (the same forensics) retries a rake
the rig declined at the teacher's next jaw headings: ``OracleRig.resume`` declines a rake whose
first hover leg does not plan, which the teacher answers by turning the jaws
(``HeapOracle._turn_rake_jaw``), while the loop booked the rake as spent. The payload's
``inference`` block records all sixteen, and ``perceived``, ``noise_scales`` and
``contact_hold`` beside them.

The MS7c hold census rides along, observation only: a ``hold.HoldDetector`` per env reads
the hold and the lift from proprioception alone (the measured jaw gap, the gap the previous
step's command asked for, the FK pick point) on every live step, whoever drives, and each
cycle's census row records what it saw beside the contact truth (``hold_detected``,
``lift_detected``, ``first_hold_at``, ``first_held_at``). The payload's ``hold_agreement`` is
``hold.agreement`` over every cycle (MS7c design section 5). With ``perceived`` off nothing
reads the detector to decide anything, so every switch and every read is what it was without
it.

``perceived`` (MS7c design section 6, off by default) puts one boundary between the simulator
and the loop's decisions: they read a perceived state (``core.perception``) and the hold
detector, never the truth. Three accessors are the only truth reads left: the perception input
at each selection (``perception_truth``), the class of the part between the jaws at a lift
(``wrist_class``, the stand-in for the MS7b wrist classifier) and the metric reads
(``truth_reads``: the grader, the tracker and the census, which stay on the truth so every
number compares with MS7a). A selection ranks the detections; the row, the frozen marker
(``env.marker_points``) and the box of the class perceived overhead come from the target's
detection. The serving rules' stop flag is the detector's hold or lift (``contact_hold``, the
pre-registered fallback, reads the contacts again). A cycle ends when a lifted hold ends and
the measured opening then reads ``SELECT_OPEN_FRACTION`` (``perceived_release``), as ``placed``
if the hold ended over a perceived box footprint and ``dropped`` anywhere else
(``perceived_cycle_end``), or on its budget. A failure is booked at the target's perceived
position (``perception.PositionMemory``), and so is a rake. At the detector's lift the box
follows the class drawn for the part between the jaws (``lift_class_box``), which replaces
``adopt_held``. At the episode's end a placed cycle whose part is stowed counts as a stow. A
selection that finds neither a target nor a rake ends the episode ``no_candidate`` only once
the tool is clear of the tray or the look has waited the retreat wait's cap
(``look_again``): the arm hides the parts under it from the overhead camera, and a real rig
moves it away and looks again before it calls the heap empty. While it waits, the rows and the
marker stay, the policy drives, and every step perceives again.

Nine MS7d switches (MS7d design section 4), all off by default so the MS7c protocol stays
reproducible, and all recorded in the payload's ``inference`` block. Two are loop fixes under
``perceived``: ``box_filter_xy`` (L1) tests only the x and y of a detection against each
perceived box's outer footprint in the selector's candidate filter
(``HeapOracle.box_filter_xy``), and ``release_gap`` (L2) also completes a perceived release
at the gap measured when the hold ended plus ``RELEASE_OPEN_M`` (``perceived_release``).
``rearm_hold`` (L8) clears the serving rules' stop flag after a hold that ends with no lift
(``rearm_stop``). ``box_walls`` (L9) adds the perceived desk boxes' walls to the grasp model
and the rake (``desk_boxes.box_walls``). ``marker_bias`` (L3) takes the gripper marker's
estimate of the camera bias off every detection (``perception.MarkerBias``). ``wrist_fix``
(L4) gives the serving rules the row of a wrist look in the snap window
(``perception.WristCamera``, ``wrist_row``), whose truth read is the decision accessor
``wrist_look_truth``. ``class_views`` (L5) votes on the class at lift with a third view under
a correlated class error (``perception.ClassViews``, ``voted_class``). ``hold_force`` (L6)
lets a hold enter only under a pick-point force bar (``HoldDetector.update``'s ``entry``).
``force_cap`` (L7) holds the arm at its measured pose on a step over the force cap
(``FORCE_CAP_N``). Every cycle's census also records ``target_in_box``, a metric read: whether
the target's true centre lay in a true box region, any box, when it was selected.

``rearm_hold`` (L8, MS7d design section 4.2) gives the serving rules a stop flag of their own
(``Cycle.stop``, ``rearm_stop``): the detector's hold turns it on as before, and a hold that
ends with no lift turns it off, so ``square_wrist``, ``pick_snap``, ``close_gate`` and
``dwell_close`` act again, and a spent ``dwell_close`` may fire once more. The census's
``hold_detected``, ``lift_detected`` and ``first_hold_at`` stay as they were; ``rearms``
counts the clears. It refuses ``contact_hold``, whose flag reads the contacts.

``hold_force`` (L6) and ``force_cap`` (L7) read the pick-point force (``core.torque``) that the
env's arm torques give (``env.arm_torques``, MuJoCo only), with or without ``perceived``. The
member stands for the motors' own torque reports, as the FK pick point stands for the joint
encoders, so under ``perceived`` it is a sensor read and not a truth accessor. L6 lets the hold
detector enter a hold only on a step whose force is under ``HOLD_FORCE_MAX_N``, and never ends
one; L7 replaces the arm command of a step whose force is over ``FORCE_CAP_N`` with the measured
arm joint positions, whoever drives. The payload's ``torque`` block counts what they did
(``TORQUE_COUNTS``).

With ``policy=None`` the oracle rig drives every cycle through the same takeover entry the
DAgger mode uses (``OracleRig.resume(i, target=...)``): the preflight's driver and the
loop's own end-to-end check. A takeover holds the env until the rig hands it back at
RETREAT, DONE or ABANDON, and the loop grades only while the rig is idle, so one takeover is
one loop cycle.

``takeover`` under ``perceived`` is the MS7f wave 3 DAgger mode (MS7f ledger D17): the policy
drives the perceived loop of a read, and on a trigger the teacher takes the cycle over. The
triggers and the teacher read the truth, as the expert of a DAgger corpus does; no decision of
the policy's own cycles does. A rig built with ``perceivers`` (the recorder's MS7c rows) is
handed the loop's perception, so a takeover records the row and marker point the episode's
perception reports, as the teacher corpora do. A takeover that stows ends its cycle
``placed``: the rig's release ends the hold while the loop does not grade, so the detector's
lost hold never reaches the perceived cycle end. Its pre-selection is dropped, and the loop
selects on the perceived scene.

Two MS7f switches (MS7f design section 6), off by default. ``classifier`` (a
``perception.PartClassifier``, under ``perceived``) replaces the class stand-in in each view of the
MS7d vote with the render-trained model on a rendered frame: the narrow overhead view of the
target's detection at selection (``env.narrow_overhead``) sets the box, the wrist frame at the
detector's lift can move it (``lift_class_box``), and with ``class_views`` a disagreement waits
for the wrist frame ``CARRY_VIEW_STEPS`` later, and the category with the largest summed log
probability over the three views takes the box (``summed_class``). The stand-in still draws every
value it draws without the switch, so knob rows stay paired. ``passenger_drop`` (with it) opens
the jaws and ends the cycle ``passenger`` when the lift's wrist frame reads a second part over the
tray interior (``over_tray``); the failure memory books the cycle, and anywhere else the reading
is ignored. The ``perception`` block then carries ``CLASSIFIER_COUNTS``. ``lift_view_steps`` N
(with ``classifier``; 0 is off) reads view 2 N steps after the detector's lift, if the detector
still reads a lift then, and keeps view 3 at ``CARRY_VIEW_STEPS`` after the lift, so a vote
decides on the same step as without the switch. A part that left the jaws before view 2 keeps
the selection's box (``lift_views_lost``). The stand-in still draws at the lift.

``missed_lift`` (MS7f riders, under ``perceived`` with the tray, off by default) reads a carry the
hold detector missed from the arm's own motion. On the F1 100k screen (2026-10-01, 4 heaps of
150) 150 of the 544 boxed parts came from no stowed or wrong_box cycle, and 116 of those were
the target or held part of another cycle. 88 such cycles read no lift, ran to their 800-step
budget, and their part ended in a box (24 a wrong one); on one instrumented heap of 50 such
parts entered the box 100 to 345 steps into the cycle. No wrist read moved the box off the
overhead read, and no release ended the cycle, which ran on with a stale row. A command
that closes over the tray and then carries the pick point out of it ``oracle.LIFT_MIN_RISE``
higher is that lift (``carry_started``): the wrist read runs as at the detector's lift. The jaws
then opening ``RELEASE_OPEN_M`` wider than they held the carry (``carry_lost``) is the release
``perceived_release`` reads at the detector's lost hold: ``placed`` over a perceived box,
``dropped`` anywhere else. If the next selection still sees a part at the target's position
after such an end, the failure memory books it there.
The ``perception`` block then carries ``MISSED_LIFT_COUNTS``.

``cell_failure_cap`` N (MS7f, under ``perceived``; ``CELL_FAILURE_CAP`` is off) lets the perceived
selector's failure memory exclude a part's position after N failed cycles instead of
``CELL_FAILURE_CAP`` (``HeapOracle.cell_failure_cap``). On candidate 2's checked read (F2 150k,
6 of 20 heaps, 2026-10-01) every episode ended ``no_candidate`` with about 16 parts a heap
unboxed, their positions excluded after 3 failures. The teacher keeps the constant.

``carry_votes`` K (MS7f, with ``classifier`` and ``class_views``; 0 is off) reads K more wrist
frames during a vote's carry, on steps spread evenly between view 2 and the carry view
(``carry_vote_steps``), and adds their log probabilities to the vote's sum (``summed_class``).
The vote still decides at the carry view, so the box moves on the same step as without the
switch. A frame due after the part left the jaws is not read (``carry_votes_lost``). Inside the
loop the taken class is right on 1253 of 1505 lifted cycles (0.833, the F1 100k screens,
2026-10-01) against 0.94 on the C1 teacher lifts. The ``perception`` block then carries
``CARRY_VOTE_COUNTS``.

``vote_weights`` (MS7g design section 6.3, with ``classifier`` and ``class_views``; None is off)
replaces the served vote with the fused vote (``fused_class``): every view's log probabilities
times the view's weight, plus a bias per class, the views not yet read left out. Each view sets
the box at its own step with that partial sum: the overhead view at selection, view 2 at the lift
and the carry view ``CARRY_VIEW_STEPS`` later, which the fused vote always reads. A part that
left the jaws before the carry view keeps view 2's box, so ``taken_class`` stays set with
``vote_lost``. The nine values are a maximum-likelihood fit on the frame bank
(``lerobot_stack/vote_bank.py``), a JSON file beside the model. ``read_class`` and
``lift_class`` stay each view's own read. Its vote is pending (``Cycle.vote``) from every lift
to the carry view, so ``carry_guard`` holds the jaws until the carry view sets the box.

``stall_end`` (MS7f, under ``perceived``, off by default) ends a cycle ``stall_end``, a failure
booked as a budget end, once its gripper command has read closed for ``CLOSED_NO_PINCH_STEPS``
steps in a row while the hold detector holds nothing and the cycle has read no lift
(``closed_empty_count``, ``stall_ends``): the DAgger recorder's ``closed_no_pinch`` trigger on
the detector's hold. On the latest checked read (2026-10-02) 162 of 3453 cycles closed on nothing
and sat there to the 800-step budget, 14.6 % of all simulated time.

The MS7g census rides along, observation only (MS7g design 5.1 and 6.2, ``MS7G_CENSUS``): each
cycle row records where the jaws first opened after a lift (``release_place``, a metric read),
the true category of the part held at that lift, the chosen grasp's swept count, the steps of
air stalls over the tray (``AirStall``) and each classifier view's log probabilities. No
decision reads any of it, so every served step is what it was without it.

``carry_guard`` (MS7g design 5.2, under ``perceived``, off by default) keeps a lifted part in the
jaws until it is over its box and hands a still carry to the rig. While the hold detector holds
a part whose hold read a lift (or a ``missed_lift`` carry is under way) and the FK pick point is
not ``GUARD_BOX_MARGIN_M`` inside the interior of the box the loop serves now (``in_box_interior``,
after any late vote), a gripper command that would end the hold is served closed
(``guarded_gap``). While a class vote is pending no box is served, since the vote can still move
it: on the MS7g read g_cg_scr 16 of the 26 class-right wrong-box cycles had a split vote that
overturned a wrong overhead read, and opened over a wrong box, 15 of them with no guard step. A
stall is ``GUARD_STALL_STEPS`` such steps in a row, the vote's steps not counted, or the census's
air stall (``AirStall``) for ``AIR_STALL_STEPS`` steps of the cycle. On a stall with a held part the
loop hands the part to the rig's perceived carry (``_hand_off``, ``OracleRig.resume(carry=...)``),
which carries it to the served box at ``carry_release_z``, releases it there and retreats; the
loop grades the carry as it grades a perceived release (``_end_handoff``). On an air stall with
nothing held it opens the jaws and ends the cycle ``stall_open``, a failure at the target. On
the MS7f checked read f_F6R_chk 17 wrong-box cycles had a vote that corrected the class while
the part fell into the box of the overhead read, and on the video diagnosis traces of
2026-10-03 the policy opened over the tray after 3 to 15 % of its lifts and hovered over the
tray (MS7g design 3.2, M1 and F1). It refuses ``passenger_drop``, which opens over the tray on
purpose. The census rows then carry ``CARRY_GUARD_CENSUS``.

Two MS7g selector switches (MS7g design 5.3 and 5.4), under ``perceived``, off by default and in
the ``inference`` block only when on. ``clear_first`` ranks the perceived selector's grasps that
sweep no standing neighbour before the height band (``grasp.rank_grasps``), so it can change the
target part; the teacher's rank stays. ``overhang_guard`` (with the tray) refuses a part outside
the tray interior under a tray wall's top edge (``grasp.under_overhang``), as neither a target
nor a rake, and keeps the carry grace for a cycle that read a lift (``grace_hold``). The
``perception`` block then carries ``OVERHANG_COUNTS``.

Like the eval loop this runs one episode per env in lockstep and reads the env through the
duck-typed surface of ``maker_arm_lab.mj.env``. Torch and stdlib only.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from functools import partial

import torch

from maker_arm_lab.core import checker, desk_boxes, grasp, layout, oracle, perception
from maker_arm_lab.core.checker import HELD_MIN_FORCE, held_brick_from_forces
from maker_arm_lab.core.dagger_loop import CLOSED_NO_PINCH_STEPS, DECLINE_COOLDOWN, trigger_reason
from maker_arm_lab.core.eval_loop import GRASP_HOLD_WINDOW, EvalFailure, _nearest_brick_cm
from maker_arm_lab.core.gate import RolloutTracker, clamp_arm_actions, policy_state
from maker_arm_lab.core.global_ik import JAW_TIP_REACH, fold_to_half_turn, wrap_to_pi
from maker_arm_lab.core.heap_oracle import (
    CANDIDATE_MAX_Z,
    CELL_FAILURE_CAP,
    MAX_RAKES,
    RAKE_PART_CAP,
    target_context,
)
from maker_arm_lab.core.hold import MIN_EXCESS, HoldDetector, agreement, commanded_gap
from maker_arm_lab.core.parts import BASE_CLASSES, FAMILIES, part_class
from maker_arm_lab.core.quat import quat_apply
from maker_arm_lab.core.stowed import (
    _brick_states,
    env_heap_bin_centers,
    parts_in_desk_boxes,
    parts_resting_box,
    per_brick_stowed,
)
from maker_arm_lab.core.torque import FORCE_CAP_N, HOLD_FORCE_MAX_N, ForceSensor, force_summary
from maker_arm_lab.robots.frames import (
    ARM_JOINT_REGEX,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_POS,
    JAW_AXIS_BODY,
    PICK_OFFSET_MEASURED,
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
SCRIPTED_PLACE_RISE = oracle.LIFT_MIN_RISE
"""m: ``scripted_place`` hands the cycle over once a held part rose this far above where it lay
when the cycle opened: the first hold plus a 5 cm lift (MS7a design, the fallback)."""
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
CONTROL_HZ = 30.0
"""Control steps per simulated second, the env's (``mj.env.CONTROL_HZ``)."""
EVENT_NAMES = ("tray_refill", "box_empty")
"""The between-cycle events of MS7f design sections 5.3 and 5.4 (``EnvState.events``), counted
per episode in the payload's ``events``: 0 while their switches are off. An event takes zero
simulated time (design section 4.1)."""


INTERVAL_NAMES = ("select_to_close", "close_to_lift", "lift_to_release", "release_to_next")
"""MS7f design section 8.1 (S0): the four intervals of a cycle, in control steps, split at the
hold detector's events (``stamp_span``, ``span_intervals``)."""


def stamp_span(span: list, detector: HoldDetector, step: int) -> None:
    """S0: stamp this step's hold detector events on a cycle's span, ``[selection, first close,
    lift, release]`` in absolute steps, None until seen. The first close is the detector's first
    hold (``started``), the lift its ``lift_started`` and the release its ``lost``, a lifted
    hold's end. An event stamps its own slot and every earlier slot still None, so the stamps
    never run backwards: a lift whose hold began before the span stamps the close too, and a
    second hold after a release changes nothing. Pure except for ``span``."""
    for k, fired in enumerate((detector.started, detector.lift_started, detector.lost), 1):
        if fired and span[k] is None:
            for j in range(1, k + 1):
                span[j] = step if span[j] is None else span[j]


def span_intervals(span: list, end: int) -> list[int]:
    """S0: the four ``INTERVAL_NAMES`` intervals of a span that ends at ``end``, the next
    selection or the episode's end. An event never seen takes the next one's time, so a cycle
    that never closed spends all its steps before the close; the four add up to ``end`` minus
    the selection. Pure."""
    times = [span[0], *span[1:], end]
    for k in range(3, 0, -1):
        times[k] = times[k + 1] if times[k] is None else times[k]
    return [times[k + 1] - times[k] for k in range(4)]


def stows_per_hour(stowed: int, steps: int, control_hz: float = CONTROL_HZ) -> float:
    """MS7f design section 4.1: correct stows per simulated hour, 3600 x ``stowed`` / (``steps`` /
    30), where ``steps`` sums every step of every episode, the steps after its last stow too.
    0 with no steps. Pure."""
    return 3600.0 * stowed * control_hz / steps if steps else 0.0


"""Steps between two frames ``rollout``'s ``on_frame`` dump receives."""
STRETCH_MISS_STREAK = 10
"""MS7f stretch rule 3 (MS7f design 10.4): ``stretch`` ends an env's episode after this many
cycles in a row closed with no new correct stow; a person clears the jaws and the tray top and
the loop starts again (``scripts/mj_stretch_run.py``)."""
CARRY_VIEW_STEPS = 15  # INITIAL: the MS7f C1 read measures the carry view at this offset
"""Control steps (0.5 s) from the detector's lift to the wrist frame of MS7f view 3, the third
vote during the carry (MS7f design section 6.3). ``scripts/mj_render_set.py`` records its
carry frames at the same offset."""
MAX_CELL_FAILURE_CAP = 10
"""``cell_failure_cap``'s largest value (MS7f, 2026-10-01)."""
MAX_CARRY_VOTES = 8
"""``carry_votes``'s largest value (MS7f, 2026-10-01): with view 2 at 6 steps after the lift,
the most reads that each get a step of their own before the carry view."""
CARRY_VOTE_COUNTS = ("carry_votes_read", "carry_votes_lost")
"""``carry_votes``'s ``perception`` counts, in the payload only with the switch on: the extra
carry frames read into a vote, and those not read because the part had left the jaws."""
LIFT_VIEW_COUNTS = ("lift_views_lost",)
"""``lift_view_steps``'s ``perception`` count, in the payload only with the switch on: the lifts
whose part left the jaws before view 2's step, so no wrist frame was read.

Why the switch (one instrumented episode, track cls-loop, 2026-10-01: F1 100k at
``--action_warp 1.5``, model_v6a, a 50-part deep-tray heap of seed 20302900 draw 0, 23 lifts):
the policy is still rising fast at the detector's lift, the tool 36 mm higher 3 steps later
and settled by 6. The wrist frame alone reads 20 of 23 at the lift and 22 of 23 at 3, 6 and 9
steps after it; the served rule with view 2 at 6 steps reads 22 of 23 against 21 of 23. On the
150-part screens and the checked read (2586 cycles) the lift read is right on 1297 of 1584
target lifts (0.819; 0.771 over each heap's first 40 cycles, while the deep tray's pile is
high), against C1's 0.903 on teacher lifts over 50-part heaps, and its errors are mostly brick
and special read as each other or as flat."""
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


RELEASE_ENDS = ("stowed", "released", "placed", "dropped", "passenger", "stall_open")
"""The cycle ends a release makes; ``release_ready`` waits for the jaws' full gap after them.
``placed`` and ``dropped`` are ``perceived``'s (``perceived_cycle_end``), ``passenger`` is
``passenger_drop``'s and ``stall_open`` ``carry_guard``'s."""
RELEASE_OPEN_M = 0.010  # INITIAL: the MS7d isolation row R1 measures it
"""m: ``release_gap`` (L2) completes a perceived release once the measured jaw gap is this much
wider than it was when the hold ended (MS7d design section 4.2). The MS7c rule alone waited
for ``SELECT_OPEN_FRACTION`` of the full gap, and 107 of candidate 3's cycles ran to the step
budget in that wait, 101 of them with the part already in a box."""


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


def lift_budget_ends(age: int, lifted: bool, holding: bool, budget: int) -> bool:
    """``lift_budget`` (MS7f design section 8.3, a serving trim): does a ``perceived`` cycle
    that has lifted nothing end now, ``age`` steps after its selection, before
    ``CYCLE_STEP_BUDGET``? Yes at ``budget`` steps or more (0: off) while the hold detector has
    read no lift in the cycle (``lifted``) and holds nothing now (``holding``): a cycle that
    holds a part keeps its row, as the carry grace does. The MS7e official read spent 43 % of
    its steps in failed cycles, most of them run to the 800-step budget. Pure."""
    return 0 < budget <= age and not lifted and not holding


def closed_empty_count(count: int, grip_open: bool, holding: bool) -> int:
    """``stall_end``: the steps in a row a cycle's gripper command has read closed while the
    hold detector holds nothing, the DAgger recorder's ``closed_no_pinch`` count on the
    detector's hold. An open read or a hold starts it again. Pure."""
    return count + 1 if not grip_open and not holding else 0


def stall_ends(closed_empty: int, lifted: bool, holding: bool) -> bool:
    """``stall_end``: does a ``perceived`` cycle whose jaws closed on nothing end now? Yes once
    ``closed_empty`` (``closed_empty_count``) reaches ``CLOSED_NO_PINCH_STEPS`` while the
    detector has read no lift in the cycle (``lifted``) and holds nothing now (``holding``), the
    two reads of ``lift_budget_ends``. Pure."""
    return closed_empty >= CLOSED_NO_PINCH_STEPS and not lifted and not holding


AIR_STALL_STEPS = 60  # INITIAL: the MS7g design's stall (its 5.2); the R0 census reads it
"""Control steps in a row the FK pick point must hover over the tray (``AirStall``). The MS7f
video diagnosis (2026-10-03): the policy stalls at lift height over the tray, about 16 % of the
steps of the diagnosed episodes, and only the step budget ends such a hover."""
AIR_STALL_MIN_Z = 0.06  # INITIAL: the MS7g design's stall (its 5.2); the R0 census reads it
"""m: the lowest pick point height above the desk an air stall counts: the hovers of the MS7f
video diagnosis sit at lift height."""
AIR_STALL_SPEED = 0.0003  # INITIAL: the MS7g design's stall (its 5.2); the R0 census reads it
"""m a step: an air stall's mean pick point speed over ``AIR_STALL_WINDOW`` steps stays under it."""
AIR_STALL_WINDOW = 30
"""Control steps the air stall's mean speed spans (the MS7g design's stall)."""


class AirStall:
    """One env's in-air stall state (MS7g design 5.2, the F1 watchdog of the MS7f video
    diagnosis). ``update`` once per control step with the FK pick point and whether it lies over
    the tray interior (``over_tray``). The state is stalled from the step on which
    ``AIR_STALL_STEPS`` steps in a row had the point over the tray, at least ``AIR_STALL_MIN_Z``
    above the desk and a mean speed (path length over the last ``AIR_STALL_WINDOW`` steps)
    under ``AIR_STALL_SPEED``, until a step breaks one of them; ``entered`` marks the step it
    began. The census counts it (``Cycle.air_stall_steps``), and only ``carry_guard`` acts on it.
    Known geometry and the FK pick point only."""

    def __init__(self):
        self.last: list[float] | None = None
        self.moves: deque = deque(maxlen=AIR_STALL_WINDOW)
        self.run = 0
        self.stalled = self.entered = False

    def update(self, pick, over_tray: bool) -> bool:
        p = [float(v) for v in pick[:3]]
        if self.last is not None:
            self.moves.append(math.dist(p, self.last))
        self.last = p
        slow = len(self.moves) == AIR_STALL_WINDOW
        slow = slow and sum(self.moves) < AIR_STALL_SPEED * AIR_STALL_WINDOW
        self.run = self.run + 1 if over_tray and p[2] >= AIR_STALL_MIN_Z and slow else 0
        was, self.stalled = self.stalled, self.run >= AIR_STALL_STEPS
        self.entered = self.stalled and not was
        return self.stalled


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


def grace_hold(holding: bool, lifted: bool, overhang_guard: bool) -> bool:
    """``perceived``: the held read ``budget_ends_cycle`` keeps a cycle open on, the hold
    detector's hold (``holding``). Under ``overhang_guard`` (MS7g design 5.4) only a cycle that
    read a lift (``lifted``: the detector's lift or the ``missed_lift`` carry) holds: the
    detector reads a clamp on a wall top as a hold, and in the MS7f video diagnosis (draw 19
    cycle 18, 2026-10-03) the grace then kept the arm on such a clamp for 680 more steps.
    Pure."""
    return holding and (lifted or not overhang_guard)


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
    """The pick cycles an episode of ``k`` parts may spend (phase B design 5.1: 65 at K=50, and
    195 at the MS7f heap of 150, MS7f design section 5)."""
    return math.ceil(CYCLE_BUDGET_FACTOR * k)


def episode_length_s(budget: int, control_hz: float = 30.0) -> float:
    """The env's episode length: every cycle of the budget at its full step budget, plus the
    retreat wait at its cap before each. The wait lies outside the cycles, so without its
    allowance the episode's step cap binds before the cycle budget does: on the 400k check of
    run m (2026-09-22), 41 of 60 episodes ended on the step cap at a median of 60 of 65 cycles,
    with a quarter of their steps spent in the wait. It scales with the heap through the
    budget: 2383 s (71,500 steps) at 50 parts, 7150 s (214,500 steps) at 150 (MS7f design
    section 5); the loop ends an episode at its cycle budget or its last candidate long before.
    """
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


SNAP_RADIUS = SQUARE_RADIUS
"""m: ``pick_snap`` acts while the pick point is this close in xy to the row centre (and below
the row's top plus ``SQUARE_ABOVE``), the window ``square_wrist`` uses."""
SNAP_XY_STEP = 0.003
"""m: the most ``pick_snap`` moves the served pick point in xy in one control step."""
SNAP_Z_RADIUS = 0.010
"""m: ``pick_snap`` lowers the pick point only while it is this close in xy to the row centre."""
SNAP_Z_STEP = 0.002
"""m: the most ``pick_snap`` lowers the served pick point in one control step."""
SNAP_Z_MARGIN = GATE_MARGIN
"""m: ``pick_snap`` lowers a commanded pick point to the row centre plus this, never further.

The teacher's DESCEND commands the pick point to the part centre (``EnvOracle.descend_goal``,
``DESCEND_CLEARANCE_MEASURED`` 0), and the rung 1 corpus closes a median 0.2 mm under the row
centre. The forensics of 2026-09-23 (arm B, screen draw 0 of seed 20261700): stows close a
median 0 to 5 mm above the centre; the collapsed episodes of the 350k checkpoint command the
pick point 15 to 25 mm above it, over a part whose top is 4 to 8 mm above it, and close on
air there. The margin is ``close_gate``'s, the stows' upper quartile on run m."""
SNAP_Z_TOL = 0.001
"""m: ``snap_hold`` lets the jaws go once the served pick point is this close to its floor."""
THIN_SNAP_MAX_HALF_HEIGHT = 0.002  # INITIAL: the MS7e ledger R13 rows read the lever
"""m: under ``thin_snap`` a served row whose half height is at most this is a flat part (a
plate or a tile, 3.2 mm tall), and its snap floor is ``DESK_CONTACT_Z``.

The row's half height is the perceived one under ``perceived``: the vertical extent carries
twice ``perception.SIZE_SIGMA_M``, so a plate row reads 1.6 +- 1.0 mm and about two in three
plate rows take the snap, and a brick row (4.8 +- 1.0 mm) about one in 400."""
DESK_CONTACT_Z = JAW_TIP_REACH + PICK_OFFSET_MEASURED[2]
"""m: the pick point's height (2.1 mm) with level jaw tips on the desk, which is z = 0 inside
the tray too: the plates reach ``JAW_TIP_REACH`` past grasp_center, the pick point sits
``PICK_OFFSET_MEASURED`` along the same axis. The teacher's flat DESCEND asks for the tips
``global_ik.FLAT_TIP_PRESS`` under the desk; the snap never takes the served point under the
desk, and a policy command already lower is kept. No press is needed on this gripper: the
lone-part table of 2026-09-29 (``scripts/mj_part_table.py --families plate,tile --trials 12
--deep_box_floor`` at f8faeee) lifts 120 of 120 flat parts with the teacher's tips at the desk
(``FLAT_TIP_PRESS`` 0, 119 stowed), as at 1 mm and 2 mm of press (120 and 120)."""
SNAP_CLOSED_MARGIN = 0.002  # INITIAL: the MS7e flat study's paired row reads it
"""m: under ``snap_closed`` the landing hold ends once the measured jaw gap is under the served
row's width plus this: the jaws closed on the part, whatever the hold detector says.

The MS7e flat study of 2026-09-30. On a flat part the detector (``core/hold.py``) misses real
pinches: the tips sink 1.9 to 2.5 mm into each side of a 7.8 mm wide plate through the soft
contact, so the measured gap creeps past ``hold.STABLE`` or drops under ``hold.MIN_GAP`` before
a lift. With no hold the stop flag stays off, the policy's lift raises its command over the
floor, and ``snap_hold`` forced the jaws open under a closed policy command in 111 of 222 first
flat closes (E2 policy, runs 7000/7001, layouts 8000/8001). The paired read with the rule (4
seeds, 1000 parts an arm, the E2 official protocol): flat stows per cycle 0.497 -> 0.750, other
parts 0.891 -> 0.925, stowed 857 -> 903, cycles per correct stow 1.321 -> 1.145, all 4 pairs
up. The landing gap is the width plus 8 mm (``grasp.landing_half_gap``), so a descent keeps
its hold."""


def snap_floor(row, thin: bool = False) -> float:
    """``pick_snap``'s floor (m): the lowest height it takes the served pick point to. Pure.

    The row centre plus ``SNAP_Z_MARGIN``. With ``thin`` (``thin_snap``, MS7e ledger R13 S2)
    a row whose half height ``row[6]`` is at most ``THIN_SNAP_MAX_HALF_HEIGHT`` snaps to
    ``DESK_CONTACT_Z`` instead. The E1 loss census of 2026-09-29 (screen e_E1_050000_scr): the
    first close on a flat part sits above its top in 0.60 of its close_miss and held_no_lift
    cycles (0.33 of its stows), a median 3.1 mm over the centre, because row z + 3 mm is 1.4 mm
    over a 3.2 mm part's top before the perceived z's bias and noise (2 mm sigma each). The
    desk floor reads no perceived z. A part under the landing spots stops the tips sooner, and
    ``snap_hold`` lets go at a stop short. A slope row keeps the margin: at its centre a slope
    lying on its base stands 7.9 to 9.6 mm, a brick's height, and bricks stow at this margin.
    """
    if thin and float(row[6]) <= THIN_SNAP_MAX_HALF_HEIGHT:
        return DESK_CONTACT_Z
    return float(row[2]) + SNAP_Z_MARGIN


def snap_shift(
    cmd, pick, row, shift, jaw_gap: float, held: bool = False, thin: bool = False
) -> tuple:
    """``pick_snap``: the offset (m, world x y z) the loop adds to the commanded pick point. Pure.

    ``cmd`` is the commanded pick point (FK of the served arm command plus the pick offset),
    ``pick`` the measured one, ``row`` the served target row (``target_context``), ``shift``
    the offset served on the step before, ``jaw_gap`` the measured gap between the jaws and
    ``held`` the stop flag ``square_wrist`` reads (the cycle has held a part or lifted one).
    ``thin`` is ``thin_snap``, which moves the floor of a flat row (``snap_floor``).
    The offset is zero with no live row (width 0), after the stop flag, and while the pick
    point is more than ``SNAP_RADIUS`` from the row centre in xy or ``SQUARE_ABOVE`` above its
    top. Inside that window:

    * xy: the served point moves toward the row centre by at most ``SNAP_XY_STEP``. The
      forensics of 2026-09-23 (arm B 100k): the teacher closes on the row centre, the policy
      a median 9 mm and a p90 45 mm off it on a cycle that follows a stow, and 1 to 3 cm off
      closes stow one cycle in three.
    * z: while the pick point is within ``SNAP_Z_RADIUS`` of the row centre and the jaws are
      open wider than the row's width, the offset goes down by at most ``SNAP_Z_STEP`` a step.
      It never takes the served point under ``snap_floor`` (a command already under that
      height is left alone) and is kept, not grown, while the jaws are closed, so closed tips
      are never driven into the part or the pile.
    """
    if held or float(row[5]) <= 0.0:
        return (0.0, 0.0, 0.0)
    near = math.hypot(float(pick[0]) - float(row[0]), float(pick[1]) - float(row[1]))
    top = float(row[2]) + float(row[6])
    if near > SNAP_RADIUS or float(pick[2]) >= top + SQUARE_ABOVE:
        return (0.0, 0.0, 0.0)
    dx, dy, dz = (float(v) for v in shift)
    ex = float(row[0]) - float(cmd[0]) - dx
    ey = float(row[1]) - float(cmd[1]) - dy
    n = math.hypot(ex, ey)
    if n > SNAP_XY_STEP:
        ex, ey = ex * SNAP_XY_STEP / n, ey * SNAP_XY_STEP / n
    if near <= SNAP_Z_RADIUS and jaw_gap >= float(row[5]):
        dz -= SNAP_Z_STEP
    floor = min(0.0, snap_floor(row, thin) - float(cmd[2]))
    return (dx + ex, dy + ey, min(0.0, max(dz, floor)))


def snap_hold(
    cmd,
    pick,
    row,
    shift,
    descending: bool,
    held: bool = False,
    thin: bool = False,
    jaw_gap: float | None = None,
) -> bool:
    """``pick_snap``: hold the jaws at the landing gap now? Pure.

    ``cmd``, ``pick``, ``row``, ``held`` and ``thin`` as in ``snap_shift``, ``shift`` the offset
    ``snap_shift`` returned for this step and ``descending`` whether the measured pick point
    came down ``GATE_DESCENT`` over the last ``GATE_WINDOW`` steps. True inside the lowering
    window (``SNAP_Z_RADIUS`` of the row centre, under the row's top plus ``SQUARE_ABOVE``,
    before the stop flag) while the policy's own command hovers above the floor
    (``snap_floor``) and the lowering is not done: the served point is still more than
    ``SNAP_Z_TOL`` above the floor, or it is there and the arm is still coming down after it
    and above it. A pick point that stops short (a neighbour under the tips) lets go at once,
    so the hold never outlasts the descent.

    The hover of the 350k checkpoint closes on air 15 to 25 mm above the part centre, and
    holding the jaws open there alone turned each close_miss into a no_close (40 of 44
    cycles, 2026-09-23): the policy commands the close it believes is at grasp height. So
    the jaws stay open only while ``snap_shift`` brings the point down, and the policy's own
    close then acts at the grasp height.

    ``jaw_gap`` is ``snap_closed``'s measured gap between the jaws, None with the switch off.
    A gap under the row's width plus ``SNAP_CLOSED_MARGIN`` ends the hold: the jaws closed on
    the part, and a pinch the hold detector missed must not be opened by the policy's lift.
    """
    if held or float(row[5]) <= 0.0:
        return False
    if jaw_gap is not None and jaw_gap < float(row[5]) + SNAP_CLOSED_MARGIN:
        return False
    near = math.hypot(float(pick[0]) - float(row[0]), float(pick[1]) - float(row[1]))
    top = float(row[2]) + float(row[6])
    if near > SNAP_Z_RADIUS or float(pick[2]) >= top + SQUARE_ABOVE:
        return False
    floor = snap_floor(row, thin)
    if float(cmd[2]) <= floor:
        return False
    if float(cmd[2]) + float(shift[2]) > floor + SNAP_Z_TOL:
        return True
    return descending and float(pick[2]) > floor + SNAP_Z_TOL


def snap_window(pick, row) -> bool:
    """Is the pick point inside ``pick_snap``'s window of the served ``row``: a live row (width
    over 0), within ``SNAP_RADIUS`` of its centre in xy and under its top plus ``SQUARE_ABOVE``?
    ``row`` None is no row. At each entry the force sensor takes its snap tare and
    ``wrist_fix`` a wrist look (``look_due``). Pure."""
    if row is None or float(row[5]) <= 0.0:
        return False
    near = math.hypot(float(pick[0]) - float(row[0]), float(pick[1]) - float(row[1]))
    return near <= SNAP_RADIUS and float(pick[2]) < float(row[2]) + float(row[6]) + SQUARE_ABOVE


def look_due(cycle: Cycle, pick, row) -> bool:
    """``wrist_fix``: does the loop take a wrist look on this step (the MS7d design's L4 model)?

    The look window is ``snap_window`` of the serving row ``row`` at the measured pick point
    ``pick``. A look fires on the step the pick point enters the window, so once per stay: it
    must leave and come back for the next. Pure except for ``cycle.in_window``, the last step's
    read."""
    inside = snap_window(pick, row)
    due = inside and not cycle.in_window
    cycle.in_window = inside
    return due


def wrist_row(row, look, seen_yaw: float) -> list[float]:
    """``wrist_fix``: the serving row after a wrist look that found the target. Pure.

    ``row`` is the target row served at selection (``target_context``), ``look`` the look's
    (x, y, yaw) and ``seen_yaw`` the heading of the target detection's body x axis at
    selection. The centre x and y become the look's, and the closing axis heading (``row[3]``,
    ``row[4]`` = cos, sin) turns by the measured yaw error ``look yaw - seen_yaw``, folded into
    [-pi/4, pi/4): a box part's grasp axes repeat every quarter turn, and the error is a few
    degrees. The z, the width and the half height stay."""
    turn = wrap_to_pi(4.0 * (look[2] - seen_yaw)) / 4.0
    heading = math.atan2(float(row[4]), float(row[3])) + turn
    out = [float(v) for v in row]
    out[0], out[1], out[3], out[4] = look[0], look[1], math.cos(heading), math.sin(heading)
    return out


DWELL_STEPS = 150
"""Control steps the pick point may wait at grasp height with the jaws open before
``dwell_close`` closes them. The rung 2 arm B 300k traces of 2026-09-24 (draws 1 and 3 of
seed 20261700, ``--pick_snap``): a stowed cycle spends a median 38 steps there before its own
close (p95 134); every no_close cycle sits there 400 to 800 steps with its own command at the
grasp pose and its gap command at the landing gap, and the 800-step budget ends it."""
DWELL_Z_TOL = 0.003
"""m: ``dwell_count``'s grasp height: the ``pick_snap`` floor (row centre plus
``SNAP_Z_MARGIN``) plus this."""
CLOSE_LATCH_STEPS = 150
"""Control steps ``dwell_close`` keeps the close command at most, unless the policy's own gap
command closes first."""


def dwell_count(count: int, pick, row, jaw_gap: float, gap_cmd: float, held: bool = False) -> int:
    """``dwell_close``: consecutive steps the arm has waited at grasp height with open jaws. Pure.

    ``pick`` is the measured pick point, ``row`` the served target row, ``jaw_gap`` the
    measured gap between the jaws, ``gap_cmd`` the gap command about to be served and
    ``held`` the stop flag of ``square_wrist``. The count grows while the pick point is within
    ``SNAP_Z_RADIUS`` of the row centre in xy and at most ``DWELL_Z_TOL`` above the
    ``pick_snap`` floor, and the jaws are open and commanded open wider than the row's width;
    any other step resets it, and so do the stop flag and a missing row.
    """
    if held or float(row[5]) <= 0.0:
        return 0
    near = math.hypot(float(pick[0]) - float(row[0]), float(pick[1]) - float(row[1]))
    at = (
        near <= SNAP_Z_RADIUS
        and float(pick[2]) <= float(row[2]) + SNAP_Z_MARGIN + DWELL_Z_TOL
        and jaw_gap >= float(row[5])
        and gap_cmd >= float(row[5])
    )
    return count + 1 if at else 0


def commanded_pick(ik, q) -> torch.Tensor:
    """The pick point of the arm command ``q`` (6,): FK plus the measured pick offset."""
    pos, rot = ik.fk(torch.as_tensor(q, dtype=torch.float64))
    return pos + rot @ torch.tensor(PICK_OFFSET_MEASURED, dtype=torch.float64)


def snapped_arm(ik, q, shift):
    """The arm command ``q`` (6,) with its pick point moved by ``shift`` at the same tilt, jaw
    heading and lean (the pick offset is rigid in the tool frame), or None with no IK
    solution."""
    from maker_arm_lab.core.global_ik import GraspTarget

    q = torch.as_tensor(q, dtype=torch.float64)
    pose = ik.pose_of(q)
    position = tuple(float(p) + float(d) for p, d in zip(pose.position, shift))
    target = GraspTarget(position, pose.tilt, pose.jaw_heading, pose.lean_azimuth)
    sol = ik.solve(target, seed=q, restarts=0)
    return None if sol is None else sol.q


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


def box_wall_refusals(selector, scene, target: int) -> int:
    """``box_walls`` (MS7d L9): the grasps on ``target`` that a desk box wall alone refuses,
    the ones ``grasp.candidate_grasps`` finds against the tray walls and not against every wall
    of the selector. Read at a selection, so a read shows how often the lever turned a grasp
    off a box wall. Pure."""
    args = (
        target,
        selector.sizes[target],
        scene.brick_pos[target],
        scene.brick_quat[target],
        oracle.PICK_TILT_RAD,
        selector.parts[target].closing_axes,
    )
    tray = [w for w in selector.walls if not w.box]
    return len(grasp.candidate_grasps(*args, tray, selector.hold_band)) - len(
        grasp.candidate_grasps(*args, selector.walls, selector.hold_band)
    )


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
    # hold a part is not an answer: the cycle runs on (``_take_over``). "placed": a perceived
    # takeover that stowed (the wave 3 DAgger mode).
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
    snap: tuple = (0.0, 0.0, 0.0)  # m, the pick point offset ``pick_snap`` served last step
    snapped_steps: int = 0  # steps ``pick_snap`` changed the arm command
    snap_held_steps: int = 0  # steps ``pick_snap`` held the jaws at the landing gap
    dwell: int = 0  # consecutive steps at grasp height with open jaws (``dwell_count``)
    latched: int = 0  # ``dwell_close``: 0 not fired, > 0 steps the close has run, -1 spent
    dwell_closes: int = 0  # times ``dwell_close`` fired (at most 1 + ``rearms``)
    adopted: int = -1  # the lifted non-target part ``adopt_held`` served the box of, else -1
    grace_steps: int = 0  # steps ``budget_ends_cycle``'s carry grace kept the cycle open
    # The MS7c hold census: the proprioceptive detector (``hold.HoldDetector``) on this
    # cycle's steps beside the contact truth. Observation only.
    hold_detected: bool = False  # the detector held on at least one step of the cycle
    lift_detected: bool = False  # and read a lift on at least one
    first_hold_at: int | None = None  # steps into the cycle at its first detected hold
    first_held_at: int | None = None  # steps into the cycle at the first contact-held step
    stop: bool = False  # ``rearm_hold``: the serving rules' stop flag (``rearm_stop``)
    rearms: int = 0  # and the times it cleared after a hold with no lift
    # ``perceived``: the target's detection at selection, the failure memory's position.
    target_xy: tuple | None = None  # m, its perceived centre xy
    seen_class: str | None = None  # its class perceived overhead
    seen_size: tuple | None = None  # m, its perceived size (a size-reading classifier's input)
    box: int = -1  # the box served now: the overhead class's, then the class drawn at a lift
    volume: float = 0.0  # m^3, the bounding volume the loop books into ``box`` at a release end
    read_class: str | None = None  # ``classifier``: the class its narrow overhead view read
    read_logp: list | None = None  # and each run class's log probability there
    lift_class: str | None = None  # ``classifier``: the last lift's wrist read
    taken_class: str | None = None  # the class that set the box at that lift or its vote
    vote_lost: bool = False  # the part left the jaws before the vote's carry view
    rider_read: bool = False  # the passenger head read a second part at that lift
    # ``missed_lift``: the lowest FK pick point z of a closed command over the tray, whether
    # ``carry_started`` read a carry from there, the narrowest measured jaw gap of the carry and
    # whether ``carry_lost`` started the cycle's release.
    carry_z: float | None = None
    carried: bool = False
    carry_gap: float | None = None
    carry_released: bool = False
    # ``classifier`` with ``class_views``: a lift that read another class than ``read_class``
    # waits for the carry view, (the lift's class, the true class at the lift, the view's step,
    # the lift's log probabilities).
    vote: tuple | None = None
    # ``classifier``: view 2 is due, (its step, the lift's step, the true class at the lift).
    lift_view: tuple | None = None
    # ``bank_views``: the step of the carry view no vote reads, after a lift that agreed.
    bank_at: int | None = None
    # ``carry_votes``: the steps of the vote's extra carry reads still due, and the log
    # probabilities of those read.
    carry_due: list = field(default_factory=list)
    carry_vote_logp: list = field(default_factory=list)
    # ``perceived_release``: a lifted hold ended over a box (True) or off one (False), and the
    # jaws have not yet opened to ``SELECT_OPEN_FRACTION``; None with no release under way.
    release_over: bool | None = None
    release_gap0: float | None = None  # m, ``release_gap``: the measured jaw gap at that end
    # ``wrist_fix``: the heading of the detection's body x axis at selection, the row the
    # serving rules read after a wrist look (None before the first, ``wrist_row``), and whether
    # the pick point was in the look window on the last step (``look_due``).
    seen_yaw: float = 0.0
    serving_row: list | None = None
    in_window: bool = False
    # The MS7d census: the target's true centre lay in a true box region, any box, at the
    # selection (``TruthReads.boxed``); None without desk boxes. A metric read only.
    target_in_box: bool | None = None
    hold_blocked: int = 0  # MS7d L6: hold entries the force gate refused in the cycle
    closed_empty: int = 0  # ``stall_end``: closed steps in a row with no hold (closed_empty_count)
    # The MS7g census (MS7g design 5.1 and 6.2), observation only. ``classifier``: the
    # last lift's wrist read (view 2) and the carry view after it (view 3), each run category's
    # log probability in the run's order, None for a view not read.
    lift_logp: list | None = None
    carry_logp: list | None = None
    swept: int | None = None  # the swept count of the grasp the selector chose (grasp.Grasp)
    lift_seen: bool = False  # the census read the cycle's first lift (detector or carry)
    lift_part: int = -1  # the contact-held part at that lift
    held_class: str | None = None  # and its true category
    lift_gap: float | None = None  # m, the narrowest measured jaw gap since that lift
    release_place: str | None = None  # where the jaws first opened after it (release_place)
    air_stall_steps: int = 0  # steps in an air stall (AirStall) while the detector holds a part
    air_stall_empty_steps: int = 0  # and while it holds nothing
    air_stalls: int = 0  # stall entries while it holds a part
    # MS7g ``carry_guard`` (None and unexported without it): the steps the guard held an open
    # command closed, those inside the served box's interior but within GUARD_BOX_MARGIN_M of
    # its walls, those inside it while a class vote was pending, the current run of them and
    # the run's fixed gap (``guarded_gap``); the rig carry's outcome ("carrying" while it runs,
    # ``_end_handoff``) and whether the hold ended over a perceived box during it; whether an
    # empty air stall opened the jaws; the selection's perceived scene, the carry's
    # ``takeover_scene``.
    guard_steps: int | None = None
    guard_rim_steps: int = 0
    guard_vote_steps: int = 0
    guard_run: int = 0
    guard_gap: float | None = None
    handoff: str | None = None
    handoff_over: bool | None = None
    stall_open: bool = False
    scene: object = None


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
    rakes: int = 0  # MS7a tray: rakes the oracle driver ran between cycles
    # Parts not in a box when the episode ended, (part, x, y, z, rakes, failed cycles): the
    # read of what the episode left behind, at a wall or elsewhere.
    left: list = field(default_factory=list)
    preselected: tuple | None = None  # the rig's RELEASE pre-selection a takeover handed back
    # ``perceived``: this env's perception, its failure memory and rake counter by position,
    # and the ``PERCEPTION_COUNTS``.
    perceiver: perception.Perceiver | None = None
    marker: perception.MarkerBias | None = None  # ``marker_bias``: the camera bias estimate
    fail_mem: perception.PositionMemory = field(default_factory=perception.PositionMemory)
    rake_mem: perception.PositionMemory = field(default_factory=perception.PositionMemory)
    counts: Counter = field(default_factory=Counter)
    look_since: int | None = None  # the step a streak of empty selections began (look_again)
    wrist: perception.WristCamera | None = None  # ``wrist_fix``: this env's wrist look
    forces: list[float] = field(default_factory=list)  # MS7d: N, each approach step's force
    events: Counter = field(default_factory=Counter)  # MS7f: EVENT_NAMES this episode ran
    span: list | None = None  # S0: the open cycle span's stamps (``stamp_span``)
    intervals: list = field(default_factory=list)  # S0: each closed span's four intervals
    # MS7f --categories: m^3 per desk box, the volume this loop's own cycles released there
    # (``desk_boxes.lighter_box`` picks between a category's two boxes by it), never the truth.
    box_load: list[float] = field(default_factory=list)
    opening: bool = False  # ``passenger_drop``: the jaws are held open until the next selection
    # ``missed_lift``: the perceived target position of a cycle ``carry_lost`` released, which
    # the next selection books a failure at if a detection still lies there.
    recheck: tuple | None = None


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


def perceived_release(
    cycle: Cycle,
    lost: bool,
    regrasped: bool,
    over_now: bool,
    open_fraction: float,
    gap: float | None = None,
) -> bool:
    """``perceived``: does a release complete on this step? The hold detector's lifted hold
    ending (``lost``) starts one and keeps ``over_now``, the ``over_box`` test of the FK pick
    point at that step, in ``cycle.release_over``; a new detected hold (``regrasped``) cancels
    it. It completes once the measured opening reads ``SELECT_OPEN_FRACTION``, the
    proprioceptive read ``released_after_lift`` makes. Ending the cycle on the hold's end
    instead switched the target row while the part was still pinched: the next chunk closed
    the jaws again at about 14 mm and carried wide parts off the box (MS7c ledger,
    2026-09-24). ``gap`` is ``release_gap``'s (None: off), the measured jaw gap: kept in
    ``cycle.release_gap0`` at the hold's end, and a gap ``RELEASE_OPEN_M`` wider than that
    completes the release too. Pure except for ``cycle.release_over`` and ``release_gap0``."""
    if lost:
        cycle.release_over = over_now
        cycle.release_gap0 = gap
    elif regrasped:
        cycle.release_over = None
    if cycle.release_over is None:
        return False
    opened = gap is not None and gap >= cycle.release_gap0 + RELEASE_OPEN_M
    return open_fraction >= SELECT_OPEN_FRACTION or opened


def perceived_cycle_end(cycle: Cycle, step: int, released: bool, over: bool) -> str | None:
    """``perceived``: why the cycle ends now, or None while it runs (MS7c design section 6).

    ``released`` is a completed release (``perceived_release``) and ``over`` whether the FK
    pick point lay over a perceived box (``over_box``) when its hold ended: a release over a
    box ends the cycle ``placed``, anywhere else ``dropped``. Then the rig's
    answer on a takeover and the step budget, as in ``cycle_end``. No truth in-box test ends a
    cycle, and there is no out-of-reach end: the budget ends a cycle whose part left reach.
    Pure."""
    if released:
        return "placed" if over else "dropped"
    if cycle.takeover_end is not None:
        return cycle.takeover_end
    if step - cycle.start_step >= CYCLE_STEP_BUDGET:
        return "budget"
    return None


def carry_started(cycle: Cycle, closed: bool, pick, in_tray: bool) -> bool:
    """``missed_lift``: does a carry the hold detector missed start on this step? A step whose
    gripper command reads ``closed`` with the FK ``pick`` point over the tray interior
    (``in_tray``, ``over_tray``) keeps the lowest such height in ``cycle.carry_z``; the carry
    starts on a step the command still reads closed with the pick point off the tray and
    ``oracle.LIFT_MIN_RISE`` above that height. A retry that reopens over the tray never starts
    one. The loop's own command, the FK point and known geometry only. Pure except for
    ``carry_z``."""
    z = float(pick[2])
    if closed and in_tray:
        cycle.carry_z = z if cycle.carry_z is None else min(cycle.carry_z, z)
        return False
    return (
        closed
        and not in_tray
        and cycle.carry_z is not None
        and z - cycle.carry_z >= oracle.LIFT_MIN_RISE
    )


def carry_lost(cycle: Cycle, gap: float, lifted: bool) -> bool:
    """``missed_lift``: do the jaws let go of a carry the hold detector does not hold? Once the
    cycle carried (``carry_started`` read it, or the detector read a lift and its hold has
    ended), the measured jaw ``gap`` keeps its narrowest value in ``cycle.carry_gap``, and a gap
    ``RELEASE_OPEN_M`` wider than that is the jaws opening on what they carried, which the loop
    hands ``perceived_release`` as the detector's lost hold. While the detector holds a lifted
    part (``lifted``) its own end decides. Proprioception only. Pure except for
    ``carry_gap``."""
    if lifted or not (cycle.carried or cycle.lift_detected):
        return False
    if cycle.carry_gap is None or gap < cycle.carry_gap:
        cycle.carry_gap = gap
        return False
    return gap >= cycle.carry_gap + RELEASE_OPEN_M


def tool_clear_of_tray(xy, lean_deg: float, height: float = layout.TRAY_WALL_HEIGHT) -> bool:
    """``perceived``: does the tool at ``xy`` hide no part in the tray? Its distance to the tray
    interior footprint must be more than ``perception.TOOL_OCCLUSION_RADIUS_M``. The footprint
    is the interior at the rim, the wider of floor and rim: an outward lean of ``lean_deg``
    widens the rim of walls ``height`` high by ``height * tan(lean)`` on every side
    (``layout.tray_rim_growth``). With the tray off it is the heap's drop zone. Known geometry
    and the FK tool point only. Pure."""
    (x0, x1), (y0, y1) = layout.tray_bounds()
    grow = layout.tray_rim_growth(lean_deg, height)
    dx = max(x0 - grow - float(xy[0]), 0.0, float(xy[0]) - x1 - grow)
    dy = max(y0 - grow - float(xy[1]), 0.0, float(xy[1]) - y1 - grow)
    return math.hypot(dx, dy) > perception.TOOL_OCCLUSION_RADIUS_M


def over_tray(xy, lean_deg: float, height: float = layout.TRAY_WALL_HEIGHT) -> bool:
    """``passenger_drop``: does the point ``xy`` lie over the tray interior? The footprint at the
    rim, the wider of floor and rim, as ``tool_clear_of_tray`` reads it (walls ``height``
    high): parts the jaws let go of there fall back into the tray. Known geometry and the FK
    tool point only. Pure."""
    (x0, x1), (y0, y1) = layout.tray_bounds()
    grow = layout.tray_rim_growth(lean_deg, height)
    x, y = float(xy[0]), float(xy[1])
    return x0 - grow <= x <= x1 + grow and y0 - grow <= y <= y1 + grow


def look_again(
    tool_xy, lean_deg: float, streak: int, height: float = layout.TRAY_WALL_HEIGHT
) -> bool:
    """``perceived``: after a selection with neither a target nor a rake, wait and perceive
    again instead of ending the episode ``no_candidate``? Yes while the tool may hide a part
    in the tray (``tool_clear_of_tray``, walls ``height`` high) and the streak of empty
    selections, ``streak`` steps old (0 at its first), is under the retreat wait's cap. Pure."""
    cap = int(SELECT_WAIT_FRACTION * CYCLE_STEP_BUDGET)
    return not tool_clear_of_tray(tool_xy, lean_deg, height) and streak < cap


def over_box(xy, boxes, sizes) -> bool:
    """Does the point ``xy`` lie over a desk box? Each box is an (x, y, yaw), the perceived
    pose under ``perceived``, with the grader's footprint of its interior in ``sizes``
    (``desk_boxes.stow_region``). Pure."""
    regions = [desk_boxes.stow_region(size) for size in sizes]
    frames = [(float(x), float(y), z, float(yaw)) for (x, y, yaw), (z, _) in zip(boxes, regions)]
    if not frames:
        return False
    point = torch.tensor([[[float(xy[0]), float(xy[1]), frames[0][2]]]])
    half = torch.tensor([[h for _, h in regions]])
    return bool(checker.in_boxes(point, torch.tensor([frames]), half).any())


RELEASE_READ_M = 0.004  # INITIAL: the R0 census reads where the jaws open
"""m: the census reads the jaws open once the measured gap is this much wider than its narrowest
since the cycle's first lift (``jaws_opened``). The teacher's RELEASE opens to the landing gap,
the width plus 8 mm (``grasp.landing_half_gap``), and the policy imitates it: on the pinned desk
scene of 2026-10-03 (``tests/test_ms7g_census.py``) the jaws held a brick 15.8 mm across them
at 13.64 mm and opened by 0.7, 2.1, 3.4, 4.4 and 5.3 mm over the release's first five steps and
by 7.2 mm at its eighth, where the episode ended, so ``RELEASE_OPEN_M``'s 10 mm would read it
late or not at all. A slip closes the jaws on the command and reads no open."""


def jaws_opened(cycle: Cycle, gap: float) -> bool:
    """The MS7g census: do the jaws open on this step, the first time since the cycle's first
    lift (``Cycle.lift_seen``)? The measured jaw ``gap`` keeps its narrowest value since the lift
    in ``cycle.lift_gap``, and a gap ``RELEASE_READ_M`` wider than that is the open. Once
    ``release_place`` is read it answers no. Pure except for ``lift_gap``."""
    if not cycle.lift_seen or cycle.release_place is not None:
        return False
    cycle.lift_gap = gap if cycle.lift_gap is None else min(cycle.lift_gap, gap)
    return gap >= cycle.lift_gap + RELEASE_READ_M


RELEASE_PLACES = ("tray", "right_box", "wrong_box", "desk", "rim")
"""The census's ``release_place`` values (MS7g interface section 3)."""


def release_place(xy, boxes, right, lean_deg: float | None, height=layout.TRAY_WALL_HEIGHT) -> str:
    """The census's ``release_place`` (MS7g design 5.1): where the point ``xy`` lies. ``boxes``
    holds each desk box's true (pose, interior size), ``right`` the indices of the boxes of the
    released part's true category, and ``lean_deg`` the tray's lean (None: no tray, walls
    ``height`` high). Over a box's interior, the grader's footprint: ``right_box`` or
    ``wrong_box``. Over its wall: ``rim``. Over the tray interior at its rim (``over_tray``):
    ``tray``; over the tray wall's top, ``layout.TRAY_WALL`` wider again: ``rim``. Anywhere else:
    ``desk``. A metric read. Pure."""
    for m, (pose, size) in enumerate(boxes):
        if desk_boxes.in_outer_footprint(xy, [pose], [size], -desk_boxes.DESK_BOX_WALL):
            return "right_box" if m in right else "wrong_box"
        if desk_boxes.in_outer_footprint(xy, [pose], [size], 0.0):
            return "rim"
    if lean_deg is None:
        return "desk"
    if over_tray(xy, lean_deg, height):
        return "tray"
    (x0, x1), (y0, y1) = layout.tray_bounds()
    grow = layout.tray_rim_growth(lean_deg, height) + layout.TRAY_WALL
    x, y = float(xy[0]), float(xy[1])
    rim = x0 - grow <= x <= x1 + grow and y0 - grow <= y <= y1 + grow
    return "rim" if rim else "desk"


GUARD_BOX_MARGIN_M = 0.010  # INITIAL: the g_cg_scr read's guard_rim_steps say whether it holds
"""m: ``carry_guard`` (MS7g design 5.2) lets the jaws open on a lifted part only with the FK pick
point this far inside the served box's interior walls (``in_box_interior``). A guard step with
the point inside the interior but within the margin counts as one of ``guard_rim_steps``."""
GUARD_STALL_STEPS = 15
"""``carry_guard``: steps in a row the guard holds an open command closed before the loop hands
the carry to the rig (MS7g design 5.2), half a second."""
CARRY_WIDTH_PAD_M = 0.002
"""m: the rig's perceived carry takes the measured jaw gap plus this as its grasp width
(``OracleRig.resume(carry=...)``). Its carry legs then hold ``grasp.close_gap`` of it, 4 mm under
the measured gap and 2 mm past ``hold.MIN_EXCESS``, so the detector keeps the hold, and RELEASE
opens 10 mm over the measured gap, past the widest settle of ``grasp.CLOSE_SQUEEZE_M``'s table
(4.25 mm a side inside the part's width)."""
CARRY_RELEASE_PAD_M = 0.010
"""m: the perceived carry's clearance over the served box's rim (``carry_release_z``)."""
CARRY_RELEASE_MAX_Z = 0.12
"""m: the highest perceived carry release. At 0.12 m the driven desk-box reach already shrinks
(``desk_boxes.DESK_BOX_RELEASE_Z``'s note), and ``_hand_off`` falls back to 0.10 m."""


def in_box_interior(xy, pose, size, margin: float = 0.0) -> bool:
    """Does the point ``xy`` lie over the interior of the box at ``pose`` (x, y, yaw) of
    interior ``size``, at least ``margin`` inside its walls, in the box's own frame? Pure."""
    dx, dy = float(xy[0]) - float(pose[0]), float(xy[1]) - float(pose[1])
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    return (
        abs(c * dx + s * dy) <= size[0] / 2 - margin
        and abs(c * dy - s * dx) <= size[1] / 2 - margin
    )


def guarded_gap(
    gap: float, measured: float, last: float, held: float | None
) -> tuple[float, float | None]:
    """``carry_guard``, on a step it guards (the hold detector holds a part its hold lifted, and
    the FK pick point is not ``GUARD_BOX_MARGIN_M`` inside the served box): the gap command
    served, and the gap the guard run holds. A command that would end the hold, wider than the
    ``measured`` jaw gap less ``hold.MIN_EXCESS``, is served as the run's gap ``held``; the
    run's first such step sets it to the tighter of the ``last`` served command and
    ``grasp.close_gap(measured)``, the teacher's squeeze on the width the jaws hold. Any other
    command passes and ends the run (None). The run's gap stays fixed on purpose: a part that
    slips out then lets the jaws close onto it and the detector's hold end. A gap that followed
    the measured one kept the hold alive on empty jaws, and its stall handed the rig a carry of
    nothing (cg smoke trace, 2026-10-03: a part dropped at step 10757, a handoff at 10771).
    Pure."""
    if gap <= measured - MIN_EXCESS:
        return gap, None
    held = min(last, grasp.close_gap(measured)) if held is None else held
    return held, held


def carry_release_z(start_z: float, size) -> float:
    """m: the pick point height the rig's perceived carry releases at over a box of interior
    ``size``, for a part the loop cannot name. ``start_z`` is the hold detector's pick point
    height when the hold began. The part lay on the desk, the tray floor or a pile then, so it
    reaches at most that far below the pick point while it keeps its pose in the jaws. The rule
    takes the teacher's ``desk_boxes.DESK_BOX_RELEASE_Z`` (0.10 m) and raises it until that depth
    clears the box rim (``desk_boxes.rim_z``, 0.035 to 0.055 m on the drawn boxes) by
    ``CARRY_RELEASE_PAD_M``, to at most ``CARRY_RELEASE_MAX_Z``: a grip 0.035 m up or lower,
    every part lying flat on the floor, releases at 0.10 m over the deepest drawn box, and a grip
    0.05 m up on a pile at 0.115 m. Pure."""
    rise = desk_boxes.rim_z(size) + start_z + CARRY_RELEASE_PAD_M
    return min(max(desk_boxes.DESK_BOX_RELEASE_Z, rise), CARRY_RELEASE_MAX_Z)


def voted_class(views) -> str:
    """``class_views``: the class that two of three views share, the third view's on a split.
    Pure. The loop draws a third view only when the first two disagree, and then this is always
    the third view: it sides with one of the two or splits three ways."""
    first, second, third = views
    return first if first in (second, third) else third


def summed_class(names, *views) -> str:
    """``classifier`` with ``class_views``: the name in ``names`` whose log probability, summed
    over ``views`` (each a list in the order of ``names``), is largest. Pure. On the MS7f C1 read
    (model_v2, 72 heaps, the 2771 lifts with all three views, 2026-09-30) it read 0.888 where
    ``voted_class`` read 0.875: right on 60 lifts the vote got wrong and wrong on 24 it got
    right."""
    sums = [sum(v[k] for v in views) for k in range(len(names))]
    return names[max(range(len(names)), key=sums.__getitem__)]


FUSED_VIEWS = ("overhead", "lift", "carry")
"""``vote_weights``' views, in the order ``fused_class`` takes their log probabilities."""


def fused_class(names, vote: dict, *views) -> str:
    """``vote_weights`` (MS7g design section 6.3): the name in ``names`` whose ``vote["bias"]``
    plus the sum over the views given of ``vote["weights"][view]`` times its log probability is
    largest. ``views`` are the overhead, lift and carry view's lists in the order of ``names``,
    the views not yet read left out (or None). Pure."""
    sums = [float(vote["bias"][n]) for n in names]
    for view, logp in zip(FUSED_VIEWS, views):
        if logp is not None:
            sums = [t + float(vote["weights"][view]) * v for t, v in zip(sums, logp)]
    return names[max(range(len(names)), key=sums.__getitem__)]


def carry_vote_steps(lift_view: int, carry_view: int, k: int) -> list[int]:
    """``carry_votes``: the steps of ``k`` more wrist reads, spread evenly between view 2's step
    ``lift_view`` and the carry view's ``carry_view``, each strictly between them and on its own
    step when ``k`` is under ``carry_view - lift_view``. Pure."""
    span = carry_view - lift_view
    return [lift_view + span * j // (k + 1) for j in range(1, k + 1)]


def lift_class_box(wrist_cls, classify, class_to_box: dict, served: int, seen=None, third=None):
    """``perceived``: the class-at-lift rule (MS7c design section 6), the stand-in for the
    MS7b wrist classifier, which replaces ``adopt_held``.

    Read on the hold detector's lift in a policy-driven cycle. ``wrist_cls`` is the class of
    the part between the jaws (``wrist_class``; None with nothing there, a false lift),
    ``classify`` draws the perceived class from it (``Perceiver.classify``), ``class_to_box``
    maps a class to its box and ``served`` is the box served now. ``class_views`` (MS7d design,
    the L5 vote) passes ``seen``, the overhead class of the selection, and ``third``, which
    draws a third view of the same part (a second wrist look during the carry,
    ``ClassViews.view``): a wrist class equal to ``seen`` is taken, and any other takes
    ``voted_class`` of the three. Returns the drawn wrist class (None: nothing drawn), the
    class taken and the box to serve (None: keep ``served``)."""
    if wrist_cls is None:
        return None, None, None
    drawn = classify(wrist_cls)
    taken = drawn
    if third is not None and drawn != seen:
        taken = voted_class((seen, drawn, third(wrist_cls)))
    box = class_to_box[taken]
    return drawn, taken, None if box == served else box


def rearm_stop(cycle: Cycle, detector: HoldDetector) -> bool:
    """``rearm_hold`` (L8): update the serving rules' stop flag ``cycle.stop`` from this step's
    ``detector``, and say whether it cleared. The hold census calls it after the serving rules,
    so they read it one step late, as they read ``hold_detected`` and ``lift_detected``.

    A detected hold or lift turns the flag on, as the latched rule does. A hold that ends with
    no lift (``ended`` and not ``lost``: a finger pressed on the desk, a tray wall or a box,
    or a grip that slipped before the lift) turns it off, so ``square_wrist``, ``pick_snap``,
    ``close_gate`` and ``dwell_close`` act again, and a spent ``dwell_close`` (``latched``
    -1) may fire once more, its dwell counted from 0. A close still running keeps running. A
    lifted hold that ends (a release or a drop) keeps the flag. On MS7c candidate 3 the latched
    flag was on with no lift in 503 of 3517 cycles, a median 522 steps before the cycle
    ended, and the policy's own wrist roll was served at once near the ground (MS7d design
    section 2, D8). Pure except for ``cycle.stop``, ``rearms``, ``latched`` and ``dwell``."""
    if cycle.stop and detector.ended and not detector.lost:
        cycle.stop, cycle.dwell, cycle.latched = False, 0, max(cycle.latched, 0)
        cycle.rearms += 1
        return True
    cycle.stop = cycle.stop or detector.holding or detector.lifted
    return False


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
            cycle.first_held_at = t
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
    "snapped_steps",
    "snap_held_steps",
    "dwell_closes",
    "adopted",
    "grace_steps",
    "hold_detected",
    "lift_detected",
    "first_hold_at",
    "first_held_at",
    "target_in_box",
    "rearms",
    "release_place",
    "held_class",
    "swept",
    "air_stall_steps",
    "air_stall_empty_steps",
    "air_stalls",
    "read_logp",
    "lift_logp",
    "carry_logp",
)
"""The ``Cycle`` fields each episode's ``cycle_census`` row exports, plus the two is_target
flags of ``first_held`` and ``last_held``. Observation only, never the metric. The last nine are
the MS7g census's (``MS7G_CENSUS``)."""
MS7G_CENSUS = CENSUS_EXPORT[-9:]
"""The MS7g census fields (MS7g interface section 3), on every row, None where they do not
apply. ``release_place``: where the jaws first opened after the cycle's first lift (the hold
detector's or ``missed_lift``'s carry), at the FK pick point, the point the jaws let go at and
the one ``over_box`` and ``over_tray`` read (``release_place``, graded for the part between the
jaws then, else the part held at the lift, else the target); None if they never opened after a
lift. The jaws open when the measured gap reads ``RELEASE_READ_M`` wider than its narrowest
since the lift (``jaws_opened``): a slip closes the jaws and reads no release, and the read
needs no close threshold, which a grasp across another axis than the served row's misreads
(the cycle's ``close_below``). ``held_class``: the true category of the contact-held part at
that lift. ``swept``: the selector's chosen grasp's swept count at the selection
(``grasp.rank_grasps``). The three ``AirStall`` counts. The three views' log probabilities,
whose category order the payload's ``perception`` block names (``logp_order``)."""


def cycle_class(cycle: Cycle, reason: str) -> str:
    """The census class of a closed cycle: ``stowed``, or the last stage a failed one reached.

    ``no_close``: the command never read closed. ``no_open``: it never read open.
    ``close_miss``: the jaws closed and held nothing for ``CENSUS_MIN_HELD_STEPS``.
    ``held_no_lift``: a part was held and never rose ``oracle.LIFT_MIN_RISE``. ``carrying``: a
    lifted part was still between the jaws when the cycle ended. ``lifted_lost``: a lifted part
    left the jaws and did not come to rest in the bin. Pure. The grader's ``reason`` stays the
    outcome, and the class only says how far the cycle got. With desk boxes a ``lifted_lost``
    cycle becomes ``wrong_box`` at the episode's end when its part rests in another class's box
    (``box_census``): the part is still falling when a release ends the cycle."""
    if reason == "stowed":
        return "stowed"
    if cycle.held_steps < CENSUS_MIN_HELD_STEPS:
        if cycle.closed_steps == 0:
            return "no_close"
        return "no_open" if cycle.open_steps == 0 else "close_miss"
    if cycle.max_rise < oracle.LIFT_MIN_RISE:
        return "held_no_lift"
    return "carrying" if cycle.holding else "lifted_lost"


def box_census(
    parts,
    destination,
    stowed,
    resting_box,
    classes: list[str],
    census: list[dict],
    num_boxes: int,
    class_names=BASE_CLASSES,
    keys=None,
    sets=None,
) -> dict:
    """The desk-box rows of one episode's end (MS7a design, the census). Pure.

    ``destination`` is each part's own box, ``stowed`` the grader's final mask (at rest in its
    own box) and ``resting_box`` the box each part rests in at the end, or -1. A part at rest in
    a box that is not its own is a wrong-box part, and the last ``lifted_lost`` cycle that held
    it becomes ``wrong_box`` in ``classes``, in place (``census`` rows hold ``last_held``).
    Returns the rows: parts stowed and parts in total per class (every one of the active
    ``class_names``), per family of the heap (MS7e design section 8.1) and per box, the
    wrong-box parts (their count, and their indices for the MS7g census's riders) and the
    episode's box count.

    ``keys`` is each part's class (``part_class`` by default; its category under MS7f
    ``--categories``) and ``sets`` each part's boxes (``destination`` alone by default): a part
    at rest in any of them is in its own box, and its box row is the one it rests in.
    """
    keys = keys or [part_class(p) for p in parts]
    sets = sets or [(int(d),) for d in destination]
    wrong = [
        b
        for b in range(len(parts))
        if int(resting_box[b]) >= 0 and int(resting_box[b]) not in sets[b]
    ]
    for b in wrong:
        for j in reversed(range(len(classes))):
            if classes[j] == "lifted_lost" and census[j].get("last_held") == b:
                classes[j] = "wrong_box"
                break
    by_class = {c: {"stowed": 0, "total": 0} for c in class_names}
    present = {p.family for p in parts}
    by_family = {f: {"stowed": 0, "total": 0} for f in FAMILIES if f in present}
    by_box = [{"stowed": 0, "total": 0} for _ in range(num_boxes)]
    for b, part in enumerate(parts):
        box = int(resting_box[b]) if int(resting_box[b]) in sets[b] else sets[b][0]
        rows = (by_class[keys[b]], by_family[part.family], by_box[box])
        for row in rows:
            row["total"] += 1
            row["stowed"] += int(bool(stowed[b]))
    return {
        "stowed_by_class": by_class,
        "stowed_by_family": by_family,
        "stowed_by_box": by_box,
        "wrong_box_parts": len(wrong),
        "wrong_box_part_ids": wrong,
        "box_count": num_boxes,
    }


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


@dataclass
class TruthReads:
    """One step's simulator truth, read at its top. The metrics (the tracker, the grader, the
    census and the box census) read it; the truth path's decisions read the same values, and
    under ``perceived`` no decision does (MS7c design section 6)."""

    per_brick: torch.Tensor  # (N,K) bool: at rest in the own bin or box, the grader's mask
    in_bin: torch.Tensor  # (N,K) bool: in the own bin or box, moving or not
    pos: torch.Tensor  # (N,K,3) part centres
    speed: torch.Tensor  # (N,K) m/s
    quat: torch.Tensor  # (N,K,4) part orientations (w, x, y, z)
    held: torch.Tensor  # (N,) the part both jaws press at HELD_MIN_FORCE, -1 for none
    resting_box: torch.Tensor | None  # (N,K) with desk boxes: the box each part rests in, or -1
    boxed: torch.Tensor | None  # (N,K) with desk boxes: in any box, moving or not, the grader's


def truth_reads(env, per_part: bool) -> TruthReads:
    """The metric reads of one step, every one the loop makes (``TruthReads``)."""
    pos, vel = _brick_states(env)
    left = env.scene["finger_left_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
    right = env.scene["finger_right_brick"].data.force_matrix_w.norm(dim=-1).squeeze(1)
    return TruthReads(
        per_brick=per_brick_stowed(env).to("cpu"),
        in_bin=per_brick_stowed(env, max_speed=math.inf).to("cpu"),
        pos=pos.to("cpu"),
        speed=vel.norm(dim=-1).to("cpu"),
        quat=torch.stack(
            [env.scene[f"brick_{b}"].data.root_quat_w for b in range(env.cfg.num_bricks)], dim=1
        ).to("cpu"),
        held=held_brick_from_forces(left, right, min_force=HELD_MIN_FORCE).to("cpu"),
        resting_box=parts_resting_box(env).to("cpu") if per_part else None,
        boxed=parts_in_desk_boxes(env).to("cpu") if per_part else None,
    )


def part_keys(env) -> tuple[str, ...]:
    """Each part's class as the env sorts it: ``env.part_classes`` (its category under MS7f
    ``--categories``), or ``part_class`` on an env without it."""
    keys = getattr(env, "part_classes", None)
    return tuple(keys) if keys else tuple(part_class(p) for p in env.parts)


def perception_truth(env, rig, i: int) -> perception.Truth:
    """``perceived``: the one truth read of a selection, what ``Perceiver.perceive`` turns into
    what the overhead camera reports. Every part's live centre and orientation, the catalog
    sizes, classes and closing axes, the tool point the arm hides parts around (the FK pick
    point) and the MS7f parked parts (``env.parked``), which the camera never sees."""
    data = [env.scene[f"brick_{b}"].data for b in range(env.cfg.num_bricks)]
    origin = env.scene.env_origins[i]
    parked = getattr(env, "parked", None)  # MS7f: parts off the desk, never detected
    return perception.Truth(
        pos=torch.stack([d.root_pos_w[i] - origin for d in data]).to("cpu"),
        quat=torch.stack([d.root_quat_w[i] for d in data]).to("cpu"),
        sizes=torch.tensor([p.size for p in env.parts]),
        classes=part_keys(env),
        tool_xy=tuple(rig.pick_point(i)[:2].tolist()),
        closing_axes=tuple(p.closing_axes for p in env.parts),
        parked=frozenset() if parked is None else frozenset(parked[i].nonzero().flatten().tolist()),
    )


def wrist_class(env, i: int) -> tuple[str | None, int]:
    """``perceived``: the one truth read a decision makes after a lift, the stand-in for the
    MS7b wrist camera. The class of the part both jaws press (the contacts) and its index, or
    (None, -1). Only ``perception.ClassViews`` reads the index, to find the part's own error
    model; no decision does."""
    left = env.scene["finger_left_brick"].data.force_matrix_w[i : i + 1].norm(dim=-1).squeeze(1)
    right = env.scene["finger_right_brick"].data.force_matrix_w[i : i + 1].norm(dim=-1).squeeze(1)
    b = int(checker.held_brick_from_forces(left, right, min_force=HELD_MIN_FORCE)[0])
    return (None, b) if b < 0 else (part_keys(env)[b], b)


def wrist_look_truth(env, rig, i: int, b: int) -> tuple[float, float, float, bool]:
    """``wrist_fix``: the one truth read of a wrist look, the stand-in for the wrist camera
    (MS7d L4). Part ``b``'s live centre x and y, the heading of its body x axis in the world
    xy plane (``PerceivedState.scene``'s convention), and whether another part covers its
    centre from above (``perception.covered``, the overhead camera's rule). A decision
    accessor like ``perception_truth``, which it reads: the no-truth test leaves both alone."""
    t = perception_truth(env, rig, i)
    pos, quat, sizes = (torch.as_tensor(v, dtype=torch.float32) for v in (t.pos, t.quat, t.sizes))
    x_axis = quat_apply(quat[b], torch.tensor([1.0, 0.0, 0.0]))
    yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))
    return float(pos[b, 0]), float(pos[b, 1]), yaw, perception.covered(pos, quat, sizes, b)


def _wrist_look(st: EnvState, env, rig, i: int, pick, row) -> None:
    """``wrist_fix``: take a wrist look at the cycle's target if ``look_due`` says so, and set
    the serving row from it (``wrist_row``). ``row`` is the target row served at selection. A
    miss or a covered target keeps the serving row as it is."""
    c = st.cycle
    if not look_due(c, pick, row if c.serving_row is None else c.serving_row):
        return
    x, y, yaw, covered = wrist_look_truth(env, rig, i, c.target)
    look = st.wrist.look((x, y), yaw, covered)
    st.counts["wrist_looks"] += 1
    st.counts["wrist_covered"] += covered
    st.counts["wrist_misses"] += look is None and not covered
    if look is None:
        return
    c.serving_row = wrist_row(row, look, c.seen_yaw)
    st.counts["wrist_shift_mm"] += 1e3 * math.hypot(look[0] - row[0], look[1] - row[1])
    turn = math.atan2(c.serving_row[4], c.serving_row[3]) - math.atan2(row[4], row[3])
    st.counts["wrist_turn_deg"] += abs(math.degrees(wrap_to_pi(turn)))


PERCEPTION_COUNTS = (
    "selections",
    "wrong_overhead_class",
    "lifts_classified",
    "wrong_wrist_class",
    "third_views",
    "votes_changed",
    "wrong_voted_class",
    "lifts_empty",
    "look_again_waits",
    "look_again_steps",
    "look_again_found",
    "rearms",
)
"""The payload's ``perception`` counts: targets selected on a detection, the targets whose
overhead class was wrong, the lifts classified, the wrong wrist classes, ``class_views``'s
third views, votes that changed the wrist class and wrong classes taken after the vote (0 with
the switch off, where the wrist class is taken), the lifts with no part between the jaws,
``look_again``'s streaks that waited, steps spent waiting and streaks that ended in a target or
a rake, and ``rearm_hold``'s stop flag clears."""
BOX_WALL_COUNTS = ("box_wall_rakes", "box_wall_refusals")
"""``box_walls``'s counts, in the ``perception`` block only with the switch on: the rakes off a
desk box wall (``Push.box_wall``) and the grasps on the selected targets that a box wall alone
refused (``box_wall_refusals``)."""
MARKER_COUNTS = ("marker_looks", "marker_error_mm")
"""``marker_bias``'s ``perception`` counts, in the payload only with the switch on: the looks at
the gripper marker, one per perceive call, and over the selections the sum of the estimate's xy
distance to the true bias in mm (``MarkerBias.error_mm``, a metric read), so over
``selections`` it is the mean error the selected rows carried."""
WRIST_COUNTS = ("wrist_looks", "wrist_misses", "wrist_covered", "wrist_shift_mm", "wrist_turn_deg")
"""``wrist_fix``'s counts, which join the ``perception`` block only with the switch on: the
wrist looks, those that missed and those whose target another part covered, and over the
looks that found it, the sums of the serving row's xy move (mm) and heading turn (degrees),
both against the row at selection."""
CLASSIFIER_COUNTS = ("passenger_reads", "passenger_drops", "votes_lost")
"""``classifier``'s ``perception`` counts, in the payload only with the switch on: the lifts whose
wrist frame read a passenger, the cycles ``passenger_drop`` ended, and the votes whose carry view
came after the part left the jaws. Under the switch ``third_views``, ``votes_changed`` and
``wrong_voted_class`` count the carry views taken. The passenger head's recall is C1's
(``lerobot_stack/train_part_classifier.py``); the loop reads no passenger truth."""
CLASSIFIER_CENSUS = ("read_class", "lift_class", "taken_class", "vote_lost", "rider_read")
"""``classifier``'s fields in a ``cycle_census`` row: the overhead read that set the box at
selection, the last lift's wrist read, the class that set the box at that lift or at its vote
(None when no lift was read or the vote was lost, so the selection's box stayed), and whether
the vote was lost or the passenger head fired. Observation only."""
CARRY_GUARD_CENSUS = ("guard_steps", "guard_rim_steps", "guard_vote_steps", "handoff", "stall_open")
"""``carry_guard``'s fields in a ``cycle_census`` row, there only with the switch on: the guard's
steps, those over the served box's rim band, those inside the served box while a class vote was
pending (held by the vote rule alone), the rig carry's outcome (None: no handoff;
"placed", "lifted_lost", "failed", "declined", or "carrying" when the episode ended during it)
and whether an empty air stall opened the jaws and failed the cycle."""
MISSED_LIFT_COUNTS = ("missed_lifts", "carried_releases", "carried_rechecks")
"""``missed_lift``'s ``perception`` counts, in the payload only with the switch on: the carries
``carry_started`` read, the cycles a release ``carry_lost`` started ended, and the selections
after one that still saw a part at the target's position and booked a failure there."""
OVERHANG_COUNTS = ("overhang_refusals",)
"""``overhang_guard``'s ``perception`` count, in the payload only with the switch on: over the
selections that opened a cycle, the parts the band refused at that selection
(``HeapOracle.overhang_refused``)."""
TORQUE_COUNTS = ("force_capped", "hold_blocked", "hold_blocked_cycles")
"""The ``torque`` block's counts under ``hold_force`` or ``force_cap``: the steps L7 held the arm,
the hold entries L6 refused (one per confirmed run of the jaw-gap rule) and the cycles with one."""


def _pick_forces(env, sensors, states, rows, picks, done) -> list[float]:
    """MS7d L6 and L7: each live env's pick-point force at the top of the step, N (0 for a done
    env). The window is ``snap_window`` of the row ``pick_snap`` reads while a cycle is open:
    the served row, or after a wrist look ``wrist_fix``'s ``Cycle.serving_row``. The force of
    an approach step (the cycle's open to its first close) joins ``EnvState.forces``."""
    forces = [0.0] * len(sensors)
    for i, sensor in enumerate(sensors):
        if bool(done[i]):
            continue
        c = states[i].cycle
        row = None if c is None or rows is None else rows[i]
        if c is not None and c.serving_row is not None:
            row = c.serving_row
        window = snap_window(picks[i], row)
        forces[i] = sensor.update(*env.arm_torques(i), window)
        if c is not None and c.first_close_at is None:
            states[i].forces.append(forces[i])
    return forces


def _count_blocked(st: EnvState) -> None:
    """MS7d L6: book a hold entry the force gate refused, and its cycle's first."""
    st.counts["hold_blocked"] += 1
    if st.cycle is not None:
        st.cycle.hold_blocked += 1
        st.counts["hold_blocked_cycles"] += st.cycle.hold_blocked == 1


def _perceived_scene(st: EnvState, env, rig, i: int, k: int):
    """``perceived``: perceive env ``i`` and hand the selector the detections: the scene with
    one slot per part (``PerceivedState.scene``), the perceived sizes and closing axes, and the
    failure and rake counts by position. Returns the state and the scene. Under
    ``marker_bias`` every centre is first corrected by the gripper marker's estimate."""
    state = st.perceiver.perceive(perception_truth(env, rig, i))
    if st.marker is not None:
        st.marker.look(rig.pick_point(i).tolist())
        state = st.marker.correct(state)
        st.counts["marker_looks"] += 1
    sel = st.selector
    sel.parts = state.parts(k)
    sel.sizes = torch.tensor([p.size for p in sel.parts], dtype=torch.float32)
    sel.failures = st.fail_mem.assign(state.detections)
    sel.raked = st.rake_mem.assign(state.detections)
    return state, state.scene(k, rig.pick_point(i).to("cpu"))


def _credit_placed(st: EnvState, final) -> None:
    """``perceived``: a ``placed`` cycle whose last contact-held part is stowed in the final
    mask is a stow, its outcome and ``stowed_parts`` with it. A part is credited once, to its
    last placed cycle, and not when the env's success already credited it."""
    last = {}
    for j, (reason, row) in enumerate(zip(st.reasons, st.census)):
        b = row["last_held"]
        if reason == "placed" and b >= 0 and bool(final[b]) and b not in st.stowed_parts:
            last[b] = j
    for j in sorted(last.values()):
        st.outcomes[j] = True
        st.classes[j] = "stowed"
        st.stowed_parts.append(st.census[j]["last_held"])


def _left_parts(selector, pos: torch.Tensor, in_bin: torch.Tensor) -> list:
    """(part, x, y, z, rakes, failed cycles) of every part not in a box, rounded to 0.1 mm."""
    failures = getattr(selector, "failures", {})
    raked = getattr(selector, "raked", {})
    return [
        [b, *(round(float(v), 4) for v in pos[b].tolist()), raked.get(b, 0), failures.get(b, 0)]
        for b in range(pos.shape[0])
        if not bool(in_bin[b])
    ]


def books_failure(ok: bool, target: int, stowed: list[int], book_miss: bool) -> bool:
    """Does a closed cycle book a failure against its target? A failed cycle does. Under
    ``book_miss`` (MS7f H) so does a cycle that stowed only other parts: without it a part the
    teacher cannot lift is selected again after every cycle in which the rig gave it up and
    delivered a neighbour (the stray loop, tables/heap/h0.txt). Pure."""
    return not ok or (book_miss and target not in stowed)


def _close_cycle(
    st: EnvState, reason: str, stowed: list[int], step: int, scene, book_miss: bool = False
) -> None:
    cycle = st.cycle
    ok = reason == "stowed"
    if reason in ("stowed", "released", "placed") and cycle.box >= 0:
        st.box_load[cycle.box] += cycle.volume  # the loop's belief: its part went there
    st.outcomes.append(ok)
    st.reasons.append(reason)
    st.steps.append(step - cycle.start_step)
    st.targets.append(cycle.target)
    st.stowed_parts.extend(stowed)
    _book_census(st, cycle, reason)
    if books_failure(ok, cycle.target, stowed, book_miss) and scene is not None:
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
    on_classify=None,
    select_at_release: bool = False,
    replan_on_switch: bool = False,
    row_refresh: bool = False,
    squeeze_floor: bool = False,
    close_gate: bool = False,
    square_wrist: bool = False,
    scripted_place: bool = False,
    scripted_rake: bool = False,
    part_failures: bool = False,
    pick_snap: bool = False,
    dwell_close: bool = False,
    adopt_held: bool = False,
    rake_cap: int = 0,
    rake_crowd: bool = False,
    park_done: bool = False,
    rake_turn: bool = False,
    hold_trace: list | None = None,
    heldout: frozenset[str] = frozenset(),
    perceived: bool = False,
    noise_scales: dict | None = None,
    contact_hold: bool = False,
    box_filter_xy: bool = False,
    release_gap: bool = False,
    rearm_hold: bool = False,
    box_walls: bool = False,
    marker_bias: bool = False,
    wrist_fix: bool = False,
    class_views: bool = False,
    hold_force: bool = False,
    force_cap: bool = False,
    thin_snap: bool = False,
    snap_closed: bool = False,
    box_empty: bool = False,
    stretch: bool = False,
    book_miss: bool = False,
    lift_budget: int = 0,
    classifier=None,
    passenger_drop: bool = False,
    lift_view_steps: int = 0,
    missed_lift: bool = False,
    cell_failure_cap: int = CELL_FAILURE_CAP,
    carry_votes: int = 0,
    stall_end: bool = False,
    carry_guard: bool = False,
    clear_first: bool = False,
    overhang_guard: bool = False,
    bank_views: bool = False,
    vote_weights: dict | None = None,
) -> dict:
    """One lockstep rollout of ``env.num_envs`` heap episodes; returns the EVAL payload.

    ``policy`` is a ``policy_client.ServerPolicy`` or None for the oracle driver.
    ``takeover`` is a ``TakeoverConfig`` (the DAgger mode, phase B design 5.5) or None.
    The sixteen settings are the module docstring's; ``replan_on_switch`` needs a policy whose
    ``step`` takes ``replan``, and the flag is left off the wire when the switch is off;
    ``squeeze_floor``, ``close_gate``, ``square_wrist`` and ``pick_snap`` need the continuous
    gripper and a served target row.
    ``on_frame(ordinal, step_in_cycle, wrist, overhead)`` is called with env 0's two policy
    frames every ``FRAME_EVERY`` steps of each of its cycles, for a forensics dump.
    ``on_classify(view, i, frame, size, step, cycle, detector)`` is called with every frame the
    ``classifier`` reads, before it reads it: the view ("overhead", "lift", "carry", or "wrist"
    for a ``carry_votes`` read), the env, the frame, the perceived size the read gets, the step,
    the env's ``Cycle`` and ``HoldDetector``. It only records (``--classifier_frames``): nothing
    it returns or changes reaches a decision. ``bank_views`` (MS7g ledger D2, the frame bank) also
    hands it the view "bank": the wrist frame ``CARRY_VIEW_STEPS`` after a lift whose read agreed
    with the overhead read, if the part is still up, the carry view a split vote would read. The
    classifier does not read it, so every served step and decision stay the same.
    ``hold_trace``, when a list, receives one ``hold.TRACE_FIELDS`` row per live env per step,
    the input of ``hold.cycle_rows``'s threshold replay.
    ``perceived`` is the MS7c boundary of the module docstring, with ``noise_scales`` the
    ``perception.parse_scales`` knob scales (None: nominal) and ``contact_hold`` the stop
    flag's contact fallback. It needs a policy, desk boxes, the continuous gripper and a served
    target row, and refuses the switches that read the truth (``scripted_place``,
    ``row_refresh``); ``takeover`` under it is the wave 3 DAgger mode of the module docstring.
    The nine MS7d switches are the module docstring's; all but ``hold_force`` and
    ``force_cap`` need ``perceived``. Those two need the env's ``arm_torques``, and a
    ``hold_trace`` row then carries the force too.
    ``thin_snap`` (MS7e ledger R13 S2) moves ``pick_snap``'s floor of a flat row to the desk
    (``snap_floor``); it needs ``pick_snap``, and the heap client records it in the payload's
    "ms7e" block. ``snap_closed`` (the MS7e flat study, 2026-09-30) ends ``pick_snap``'s
    landing hold once the measured jaws closed on the part (``snap_hold``); it needs
    ``pick_snap`` and is recorded the same way.
    ``heldout`` names the part kinds of the held-out split: each episode's ``stowed_heldout``
    counts its parts of those kinds and the ones stowed (MS7f design section 4.4, the held-out
    row), read on the grader's final mask. The payload also carries ``steps_total``,
    ``stows_per_hour`` (``stows_per_hour``) and the ``events`` of ``EVENT_NAMES``.
    ``box_empty`` (MS7f H2, design 5.4) and an env built with ``tray_refill`` (MS7f
    H1') run a person's events between an env's cycles, before its next selection
    (``between_cycles``); ``box_empty`` needs desk boxes. An env with a deep tray or a refill
    also records the hidden share at each selection (``hidden_share``, MS7f design 5.5).
    ``stretch`` runs two person rules of a stretch run (MS7f design 10.4): between cycles a
    part at rest outside the tray and every box goes back into the tray (rule 2, counted per
    part), and ``STRETCH_MISS_STREAK`` cycles in a row with no new correct stow end the episode
    ``miss_streak`` (rule 3, ``miss_streak``). The person watches the grader.
    ``book_miss`` (MS7f H) books a failure against the target of a cycle that stowed only other
    parts (``books_failure``), on the truth-graded path; a ``--perceived`` cycle is unchanged.
    ``classifier`` (a ``perception.PartClassifier`` or None), ``passenger_drop`` and
    ``lift_view_steps`` are the MS7f switches of the module docstring; all need ``perceived``,
    and the last two ``classifier``; ``lift_view_steps`` is under ``CARRY_VIEW_STEPS``.
    ``missed_lift`` (MS7f riders) is the module docstring's; it needs ``perceived`` and the tray.
    ``cell_failure_cap`` (MS7f) is the module docstring's; it lies in [1,
    ``MAX_CELL_FAILURE_CAP``] and needs ``perceived`` unless it is ``CELL_FAILURE_CAP``.
    ``carry_votes`` (MS7f) is the module docstring's; it needs ``classifier`` and ``class_views``,
    and lies in [0, ``MAX_CARRY_VOTES``] and under ``CARRY_VIEW_STEPS - lift_view_steps``.
    ``stall_end`` (MS7f) is the module docstring's; it needs ``perceived``.
    ``carry_guard`` (MS7g) is the module docstring's; it needs ``perceived`` and refuses
    ``passenger_drop``.
    ``clear_first`` and ``overhang_guard`` (MS7g) are the module docstring's; both need
    ``perceived``, and ``overhang_guard`` the tray.
    ``vote_weights`` (MS7g) is the module docstring's, the vote's JSON as a dict (``weights`` per
    ``FUSED_VIEWS``, ``bias`` per class); it needs ``classifier`` and ``class_views`` and refuses
    ``carry_votes``.
    Raises ``EvalFailure`` on a scene, context, action-space or gripper mismatch, the wall
    clock and an incomplete batch, as the eval loop does.
    """
    num_envs = env.num_envs
    k = env.cfg.num_bricks
    robot = env.scene["robot"]
    arm_ids = robot.find_joints([ARM_JOINT_REGEX])[0]
    if list(arm_ids) != list(range(6)):
        raise EvalFailure("joint_order", f"arm joints are at {list(arm_ids)}, expected 0..5")
    jaw_ids = robot.find_joints(list(GRIPPER_JOINT_NAMES))[0]
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
    if scripted_place and policy is None:
        raise EvalFailure("scripted_place", "the scripted place takes over from a policy")
    if scripted_rake and policy is None:
        raise EvalFailure("scripted_rake", "the scripted rake serves a policy; the driver rakes")
    ctx_target = getattr(env, "context_target", None)
    # MS7a desk boxes: one destination box per part, and the served box part follows the target.
    per_part = torch.as_tensor(env.destination_bins).dim() == 2
    class_names = getattr(env, "class_names", BASE_CLASSES)  # the active class list (MS7e)
    keys = part_keys(env)  # each part's class; the census's held_class reads it too (MS7g)
    if per_part:
        # The loop runs one episode per env: its truth, before an env's own end resets it.
        destination = torch.as_tensor(env.destination_bins).clone().to("cpu")
        class_to_box = [dict(zip(class_names, map(int, row))) for row in env.class_to_box]
        # MS7f --categories (design section 6.2): every box of each class, the first one
        # class_to_box's, and each part's. With the switch off each class and each part has its
        # one box, read where the loop always read them.
        if getattr(env, "categories", False):
            class_boxes = [dict(zip(class_names, row)) for row in env.class_boxes]
            part_sets = [[own[c] for c in keys] for own in class_boxes]
        else:
            class_boxes = [{c: (b,) for c, b in row.items()} for row in class_to_box]
            part_sets = [[(int(b),) for b in row] for row in destination]
        # MS7e: the episode's own box count, the boxes the scene built without the box draw.
        box_count = [
            int(n) for n in getattr(env, "box_count", [len(p) for p in env.desk_box_poses])
        ]
        # MS7e ledger D7: the box draw's extra size-draw rounds, recorded with the draw on.
        rounds = getattr(env, "box_draw_rounds", None) if getattr(env, "box_draw", 0) else None
        # And each drawn box's pose and interior, so a read's census rebuilds the boxes.
        geometry = []
        if rounds is not None:
            tables = (("box_poses", env.desk_box_poses), ("box_sizes", env.desk_box_sizes))
            for i, n in enumerate(box_count):
                rows = {
                    key: [[round(float(v), 4) for v in r] for r in t[i][:n]] for key, t in tables
                }
                geometry.append(rows)
        final_box = torch.full((num_envs, k), -1, dtype=torch.long)
        # The MS7g census's release_place: each box's true pose and interior, a metric read.
        true_boxes = [
            list(zip(env.desk_box_poses[i][:n], env.desk_box_sizes[i][:n]))
            for i, n in enumerate(box_count)
        ]
    if adopt_held and not per_part:
        raise EvalFailure("adopt_held", "needs desk boxes: a part's own box is the one it adopts")
    for name, on in (
        ("squeeze_floor", squeeze_floor),
        ("close_gate", close_gate),
        ("square_wrist", square_wrist),
        ("pick_snap", pick_snap),
        ("dwell_close", dwell_close),
    ):
        if on and (gripper_mode != "continuous" or ctx_target is None):
            raise EvalFailure(name, "needs the continuous gripper and a served target row")
    if thin_snap and not pick_snap:
        raise EvalFailure("thin_snap", "moves pick_snap's floor: it needs pick_snap")
    if snap_closed and not pick_snap:
        raise EvalFailure("snap_closed", "ends pick_snap's landing hold: it needs pick_snap")
    if contact_hold and not perceived:
        raise EvalFailure("contact_hold", "the stop flag's contact fallback needs perceived")
    if box_empty and not per_part:
        raise EvalFailure("box_empty", "empties the desk boxes: it needs desk boxes")
    refill = bool(getattr(env, "tray_refill", False))  # MS7f H1': the env parks the waiting parts
    ms7f_tray = refill or bool(getattr(env, "deep_tray", False))
    if lift_budget and not perceived:
        raise EvalFailure("lift_budget", "reads the hold detector's lift: it needs perceived")
    if stall_end and not perceived:
        raise EvalFailure("stall_end", "reads the hold detector's hold: it needs perceived")
    if carry_guard and not perceived:
        raise EvalFailure("carry_guard", "reads the hold detector and the perceived boxes")
    if carry_guard and passenger_drop:
        raise EvalFailure("carry_guard", "refuses passenger_drop, which opens over the tray")
    if perceived:
        if policy is None:
            raise EvalFailure("perceived", "serves a policy; the oracle driver reads the truth")
        if not per_part or gripper_mode != "continuous" or ctx_target is None:
            raise EvalFailure(
                "perceived", "needs desk boxes, the continuous gripper and a served target row"
            )
        for name, on in (
            ("scripted_place", scripted_place),
            ("row_refresh", row_refresh),
        ):
            if on:
                raise EvalFailure("perceived", f"{name} reads the truth")
    ms7d = {
        "box_filter_xy": bool(box_filter_xy),
        "release_gap": bool(release_gap),
        "rearm_hold": bool(rearm_hold),
        "box_walls": bool(box_walls),
        "marker_bias": bool(marker_bias),
        "wrist_fix": bool(wrist_fix),
        "class_views": bool(class_views),
        "hold_force": bool(hold_force),
        "force_cap": bool(force_cap),
    }
    for name, on in ms7d.items():
        # The hold calibration runs the teacher with --hold_force, and the cap binds every driver.
        if on and not perceived and name not in ("hold_force", "force_cap"):
            raise EvalFailure(name, "changes a perceived decision: it needs perceived")
    if rearm_hold and contact_hold:
        raise EvalFailure("rearm_hold", "clears the detector's flag; contact_hold reads contacts")
    if classifier is not None and not perceived:
        raise EvalFailure("classifier", "reads the perceived views: it needs perceived")
    if passenger_drop and classifier is None:
        raise EvalFailure("passenger_drop", "reads the classifier's passenger head")
    if lift_view_steps and classifier is None:
        raise EvalFailure("lift_view_steps", "delays the classifier's wrist read")
    if not 0 <= lift_view_steps < CARRY_VIEW_STEPS:
        raise EvalFailure("lift_view_steps", f"must lie in [0, {CARRY_VIEW_STEPS})")
    if classifier is not None and not set(class_names) <= set(classifier.categories):
        raise EvalFailure(
            "classifier", f"reads {classifier.categories}; the run sorts by {class_names}"
        )
    tray = bool(getattr(env, "tray", False))
    lean = float(getattr(env, "tray_lean_deg", 0.0)) if tray else 0.0
    wall_height = getattr(env, "tray_wall_height", layout.TRAY_WALL_HEIGHT)  # MS7f H1
    if missed_lift and not (perceived and tray):
        raise EvalFailure(
            "missed_lift", "reads a carry off the tray: it needs perceived and a tray"
        )
    if clear_first and not perceived:
        raise EvalFailure("clear_first", "ranks the perceived selector's grasps: needs perceived")
    if overhang_guard and not (perceived and tray):
        raise EvalFailure(
            "overhang_guard", "reads the tray wall's overhang: it needs perceived and a tray"
        )
    if not 1 <= cell_failure_cap <= MAX_CELL_FAILURE_CAP:
        raise EvalFailure("cell_failure_cap", f"must lie in [1, {MAX_CELL_FAILURE_CAP}]")
    if cell_failure_cap != CELL_FAILURE_CAP and not perceived:
        raise EvalFailure("cell_failure_cap", "sets the perceived selector's cap: needs perceived")
    if bank_views and (classifier is None or on_classify is None):
        raise EvalFailure("bank_views", "hands on_classify a carry view: needs the classifier")
    if vote_weights is not None:
        if classifier is None or not class_views:
            raise EvalFailure("vote_weights", "fuses the classifier's views: needs class_views")
        if carry_votes:
            raise EvalFailure("vote_weights", "fuses three views; carry_votes reads more")
        weights, bias = vote_weights.get("weights", {}), vote_weights.get("bias", {})
        # The frame bank's richer forms (ledger D7: a weight a view and class, or a matrix a
        # view) adopted none, so this serves the nine values only.
        scalar = all(isinstance(w, (int, float)) for w in weights.values())
        if set(weights) != set(FUSED_VIEWS) or not scalar or not set(class_names) <= set(bias):
            raise EvalFailure("vote_weights", f"needs a weight of {FUSED_VIEWS} and a bias a class")
    if carry_votes and not (classifier is not None and class_views):
        raise EvalFailure("carry_votes", "adds reads to the classifier's vote: needs class_views")
    if not 0 <= carry_votes <= MAX_CARRY_VOTES or carry_votes >= CARRY_VIEW_STEPS - lift_view_steps:
        raise EvalFailure(
            "carry_votes",
            f"must lie in [0, {MAX_CARRY_VOTES}] and under {CARRY_VIEW_STEPS} - lift_view_steps",
        )
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
        st.box_load = [0.0] * (len(env.desk_box_poses[0]) if per_part else 0)
    if part_failures:
        for st in states:
            st.selector.failure_scope = "part"
    for st in states:
        if rake_cap:
            st.selector.rake_part_cap = rake_cap
            st.selector.max_rakes = max(MAX_RAKES, MAX_RAKES * rake_cap // RAKE_PART_CAP)
        if rake_crowd:
            st.selector.rake_crowd = grasp.RAKE_CROWD_M
    for st in states:
        if st.selector.cycle_budget != cycle_budget:
            raise EvalFailure(
                "cycle_budget",
                f"the selector spends {st.selector.cycle_budget} cycles, the loop {cycle_budget}",
            )
    scales = {**perception.parse_scales(""), **(noise_scales or {})}  # a knob not named is 1
    wave3 = perceived and takeover is not None  # MS7f wave 3, the module docstring's DAgger mode
    sensors = None  # MS7d L6 and L7: one pick-point force sensor per env
    if hold_force or force_cap:
        if getattr(env, "arm_torques", None) is None:
            name = "hold_force" if hold_force else "force_cap"
            raise EvalFailure(name, "needs the env's arm torques (arm_torques, MuJoCo only)")
        sensors = [ForceSensor(run_seed, i, scales) for i in range(num_envs)]
    if perceived:
        # The episode's perception per env. The box poses are the last truth the setup reads:
        # the perceived ones replace them in the served row, the selector's any-box filter and
        # the cycle end. The selector books failures by position and reads no destination.
        true_classes = list(part_keys(env))  # the metric's wrong-class counts
        poses = env.desk_box_poses.copy()
        drawn = bool(getattr(env, "box_draw", 0))  # MS7e: each box's size is perceived too
        for i, st in enumerate(states):
            st.perceiver = perception.Perceiver(run_seed, i, scales, class_names)
            st.perceiver.reset_episode(poses[i], env.desk_box_sizes[i] if drawn else ())
            if wave3 and getattr(rig, "perceivers", None) is not None:
                rig.perceivers[i] = st.perceiver  # a takeover records this perception
            if marker_bias:
                st.marker = perception.MarkerBias(st.perceiver, run_seed, i)
            if class_views:
                st.perceiver.views = perception.ClassViews(run_seed, i, scales, k, class_names)
            poses[i] = st.perceiver.box_poses
            st.selector.boxes = st.perceiver.box_poses
            st.selector.box_sizes = st.perceiver.box_sizes
            st.selector.failure_scope = "part"
            st.selector.cell_failure_cap = cell_failure_cap
            st.selector.knows_destinations = False
            st.selector.box_filter_xy = box_filter_xy
            st.selector.clear_first = clear_first
            st.selector.overhang_guard = overhang_guard
            if box_walls:
                # L9: the grasp model and the rake also read the perceived boxes' walls.
                boxes, sizes = st.perceiver.box_poses, st.perceiver.box_sizes
                st.selector.walls += desk_boxes.box_walls(boxes, sizes)
            if wrist_fix:
                st.wrist = perception.WristCamera(run_seed, i, scales)
        env.perceived_box_poses = poses
        for i in range(num_envs):
            env.set_context_box(i, 0)  # the reset served box 0's true pose
    ordinal = [0] * num_envs  # the recorder's takeover ordinal per env
    loop_done = torch.zeros(num_envs, dtype=torch.bool)
    triggers = (
        _TriggerState(num_envs, takeover, run_seed, cycle_budget, wave3)
        if takeover is not None
        else None
    )
    switches = {
        "select_at_release": bool(select_at_release),
        "replan_on_switch": bool(replan_on_switch),
        "row_refresh": bool(row_refresh),
        "squeeze_floor": bool(squeeze_floor),
        "close_gate": bool(close_gate),
        "square_wrist": bool(square_wrist),
        "scripted_place": bool(scripted_place),
        "scripted_rake": bool(scripted_rake),
        "part_failures": bool(part_failures),
        "pick_snap": bool(pick_snap),
        "dwell_close": bool(dwell_close),
        "adopt_held": bool(adopt_held),
        "rake_cap": int(rake_cap),
        "rake_crowd": bool(rake_crowd),
        "park_done": bool(park_done),
        "rake_turn": bool(rake_turn),
        "perceived": bool(perceived),
        "noise_scales": perception.format_scales(scales) if perceived else "",
        "contact_hold": bool(contact_hold),
        **ms7d,
    }
    if classifier is not None:
        switches |= {"classifier": classifier.path, "passenger_drop": bool(passenger_drop)}
        if getattr(classifier, "flip", False):
            switches["cls_flip"] = True  # MS7g ledger D10: every view also read mirrored
    if lift_view_steps:
        switches["lift_view_steps"] = int(lift_view_steps)
    if missed_lift:
        switches["missed_lift"] = True
    if cell_failure_cap != CELL_FAILURE_CAP:
        switches["cell_failure_cap"] = int(cell_failure_cap)
    if carry_votes:
        switches["carry_votes"] = int(carry_votes)
    if stall_end:
        switches["stall_end"] = True
    if carry_guard:
        switches["carry_guard"] = True
    if clear_first:
        switches["clear_first"] = True
    if overhang_guard:
        switches["overhang_guard"] = True
    if vote_weights is not None:
        switches["vote_weights"] = {k: vote_weights[k] for k in ("weights", "bias")}
    replan = torch.zeros(num_envs, dtype=torch.bool)  # envs whose row changed since the last step
    # The pick point's last GATE_WINDOW + 1 heights per env, the gate's descent read.
    pick_z_hist = [deque(maxlen=GATE_WINDOW + 1) for _ in range(num_envs)]
    # The roll command each env received on the last step, ``square_wrist``'s rate read.
    last_roll = robot.data.joint_pos[:, WRIST_ROLL_INDEX].to("cpu").clone()
    parked = None  # ``park_done``: the command each env got on the step before
    detectors = [HoldDetector() for _ in range(num_envs)]  # the MS7c hold census
    stalls = [AirStall() for _ in range(num_envs)]  # the MS7g census's air stalls
    # m, the gap each env's last command asked for; before the first, the jaws' own gap.
    gap_cmd = robot.data.joint_pos[:, jaw_ids].sum(dim=-1).tolist()
    # MS7f H2: per part the grade and the box of an emptying.
    emptied_ok = torch.zeros(num_envs, k, dtype=torch.bool)
    emptied_box = torch.full((num_envs, k), -1, dtype=torch.long)
    hidden = [[] for _ in range(num_envs)]  # the hidden share at each selection (design 5.5)
    # ``stretch``: the person interventions per env, and each env's correct stows at the last
    # cycle close and the cycles closed since they last grew (rule 3).
    interventions = [Counter(strays=0, miss_streak=0) for _ in range(num_envs)]
    streaks = [(0, 0)] * num_envs

    started = time.monotonic()
    max_steps = int(env.max_episode_length)
    step = 0
    while step < max_steps and not tracker.all_done:
        if (time.monotonic() - started) / 60.0 > max_wall_minutes:
            raise EvalFailure("wall_clock", f"exceeded {max_wall_minutes} min at step {step}")
        truth = truth_reads(env, per_part)
        per_brick, in_bin, pos, speed, held, boxed = (
            truth.per_brick,
            truth.in_bin,
            truth.pos,
            truth.speed,
            truth.held,
            truth.boxed,
        )
        # A part an emptying graded correct stays stowed (MS7f H2): it is parked, not boxed.
        tracker.latch_pre_step(per_brick | emptied_ok)
        if per_part:
            # Latched like the tracker's mask: the step that ends an episode also resets it.
            live = ~tracker.done
            final_box[live] = torch.where(emptied_box >= 0, emptied_box, truth.resting_box)[live]
        openings = policy_state(robot.data.joint_pos)[:, 6].to("cpu")
        gaps = robot.data.joint_pos[:, jaw_ids].sum(dim=-1).to("cpu")  # m, measured jaw gap
        # The hold detector reads this step's proprioception first: the jaw gap measured now,
        # the gap the last step's command asked for and the FK pick point. ``perceived``
        # decides on its state and events below; otherwise only the census reads it.
        top_picks = [rig.pick_point(i).to("cpu") for i in range(num_envs)]
        forces = [0.0] * num_envs  # N, the pick-point force, read under L6 or L7 only
        if sensors is not None:
            forces = _pick_forces(env, sensors, states, ctx_target, top_picks, tracker.done)
        for i, d in enumerate(detectors):
            if not bool(tracker.done[i]):
                gate = not hold_force or forces[i] < HOLD_FORCE_MAX_N  # L6's entry gate
                d.update(float(gaps[i]), gap_cmd[i], float(top_picks[i][2]), gate)
                if states[i].span is not None:
                    stamp_span(states[i].span, d, step)
                if d.blocked:
                    _count_blocked(states[i])
                c = states[i].cycle
                if c is not None and c.handoff == "carrying" and d.ended and c.handoff_over is None:
                    # ``carry_guard``: where the hold ended during the rig's carry grades it.
                    p = states[i].perceiver
                    c.handoff_over = over_box(top_picks[i][:2], p.box_poses, p.box_sizes)
        # 1. Grade and select on the envs the loop holds: episode live, rig idle.
        for i in range(num_envs):
            if bool(tracker.done[i]) or not rig.idle[i]:
                continue
            st = states[i]
            stowed_now = {b for b in range(k) if bool(in_bin[i, b])}
            if st.cycle is not None and perceived:
                c, d = st.cycle, detectors[i]
                p = st.perceiver
                lost = d.lost
                if missed_lift and not lost and c.release_over is None:
                    lost = c.carry_released = carry_lost(c, float(gaps[i]), d.lifted)
                over = lost and over_box(top_picks[i][:2], p.box_poses, p.box_sizes)
                gap = float(gaps[i]) if release_gap else None
                released = perceived_release(c, lost, d.started, over, float(openings[i]), gap)
                reason = perceived_cycle_end(c, step, released, bool(c.release_over))
                lifted = c.lift_detected or c.carried
                # The carry grace keeps a cycle open while the detector holds.
                held_now = grace_hold(d.holding, lifted, overhang_guard)
                grace = not budget_ends_cycle(step - c.start_step, held_now, c.grace_steps)
                if select_at_release and reason == "budget" and grace:
                    c.grace_steps += 1
                    reason = None
                if reason is None and lift_budget_ends(
                    step - c.start_step, lifted, d.holding, lift_budget
                ):
                    reason = "lift_budget"
                if reason is None and stall_end and stall_ends(c.closed_empty, lifted, d.holding):
                    reason = "stall_end"  # booked as a budget end
                if reason is not None:
                    if reason != "placed":
                        st.fail_mem.book(c.target_xy)  # never the selector's _mark_failure
                    if c.carry_released and reason in ("placed", "dropped"):
                        st.counts["carried_releases"] += 1
                        if reason == "placed":  # ``missed_lift``: look at the target's place
                            st.recheck = c.target_xy
                    _close_cycle(st, reason, [], step, None)
                    st.idle_since = step
            elif st.cycle is not None:
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
                    _close_cycle(st, end[0], end[1], step, rig.snapshot(i), book_miss)
                    st.idle_since = step
            if st.cycle is not None:
                continue
            if stretch and st.idle_since == step:  # a cycle closed on this step
                count = int((per_brick[i] | emptied_ok[i]).sum())
                streaks[i] = miss_streak(count, *streaks[i])
                if streaks[i][1] >= STRETCH_MISS_STREAK:
                    interventions[i]["miss_streak"] += 1
                    loop_done[i] = True
                    st.done_reason = "miss_streak"
                    continue
            if not perceived and bool(per_brick[i].all()):
                loop_done[i] = True
                st.done_reason = "all_stowed"
                continue
            if st.idle_since is not None:
                opening = float(openings[i])
                if select_at_release:
                    # ``release_gap`` ended a perceived release on the opening already: the
                    # wait for the full gap would add back the steps it cut.
                    last = st.reasons[-1]
                    gapped = release_gap and last in ("placed", "dropped")
                    ready = release_ready(
                        opening, step - st.idle_since, last in RELEASE_ENDS and not gapped
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
            if (box_empty or refill or stretch) and between_cycles(
                env,
                i,
                st.events,
                emptied_ok,
                emptied_box,
                box_empty,
                refill,
                interventions[i] if stretch else None,
                (st.box_load, getattr(st.selector, "box_load", None) or []),
            ):
                # The event moved parts: the truth of this step is stale for env i.
                truth = truth_reads(env, per_part)
                per_brick, in_bin, pos, speed, held, boxed = (
                    truth.per_brick,
                    truth.in_bin,
                    truth.pos,
                    truth.speed,
                    truth.held,
                    truth.boxed,
                )
                stowed_now = {b for b in range(k) if bool(in_bin[i, b])}
            if perceived:
                state, scene = _perceived_scene(st, env, rig, i, k)
                if st.recheck is not None:
                    # ``missed_lift``: a part still lies where the carried cycle's target lay,
                    # so the carry held nothing; book it as the budget end would have.
                    near = perception.ASSOCIATION_RADIUS_M
                    if any(math.dist(d.centre[:2], st.recheck) <= near for d in state.detections):
                        st.fail_mem.book(st.recheck)
                        st.selector.failures = st.fail_mem.assign(state.detections)
                        st.counts["carried_rechecks"] += 1
                    st.recheck = None
            else:
                scene = rig.snapshot(i)
            target = _select(st, scene)
            rake = getattr(st.selector, "push", None)
            raking = policy is None or scripted_rake
            rakes = target is None and raking and rake is not None and rake.part is not None
            if st.look_since is not None and (target is not None or rakes):
                st.counts["look_again_found"] += 1
                st.look_since = None
            if rakes:
                # MS7a tray: no grasp ranks and the teacher planned a rake off a wall. The
                # oracle driver (and a policy under ``scripted_rake``) runs it as a takeover of
                # its own, not a cycle, and the selector books it (its per-part and per-episode
                # caps), then selects again.
                st.selector._end_push()
                st.rakes += 1
                st.counts["box_wall_rakes"] += rake.box_wall
                overrides = {}
                if perceived:
                    # Booked where the part was perceived; the rig's takeover reads the hold
                    # detector and this scene, never the truth.
                    st.rake_mem.book(scene.brick_pos[rake.part, :2].tolist())
                    overrides = {"held": detectors[i].holding, "scene": scene}
                # The teacher's jaw headings in its order: along the sweep, across it, held.
                jaws = (rake.jaw, rake.heading + math.pi / 2, None) if rake_turn else (rake.jaw,)
                for jaw in jaws:
                    if rig.resume(i, push=replace(rake, jaw=jaw), **overrides):
                        if perceived:
                            env.marker_points[i] = math.nan  # the truth path's rake unmarks too
                        break
                continue
            if target is None and perceived and st.selector.cycles < cycle_budget:
                streak = 0 if st.look_since is None else step - st.look_since
                if look_again(top_picks[i][:2], lean, streak, wall_height):
                    if st.look_since is None:
                        st.look_since = step
                        st.counts["look_again_waits"] += 1
                    st.counts["look_again_steps"] += 1
                    continue
            if target is None:
                loop_done[i] = True
                st.done_reason = (
                    "cycle_budget" if st.selector.cycles >= cycle_budget else "no_candidate"
                )
                st.left = _left_parts(st.selector, pos[i], in_bin[i])
                continue
            waited = 0 if st.idle_since is None else step - st.idle_since
            chosen = getattr(st.selector, "grasp", None)
            if st.span is not None:  # S0: the last cycle's span ends at this selection
                st.intervals.append(span_intervals(st.span, step))
            st.span = [step, None, None, None]
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
                target_in_box=None if boxed is None else bool(boxed[i, target]),
                swept=None if chosen is None else int(chosen.swept),
            )
            if ms7f_tray:
                hidden[i].append(hidden_share(env, pos[i], truth.quat[i], top_picks[i]))
            if perceived:
                # The row, the frozen marker and the box of the class perceived overhead all
                # come from the target's detection; ``env.target_part`` stays -1.
                seen = next(d for d in state.detections if d.sim_index == target)
                st.cycle.target_xy = seen.centre[:2]
                st.cycle.seen_class = cls = seen.cls
                st.opening = False  # ``passenger_drop`` and ``carry_guard``'s stall_open
                if carry_guard:
                    st.cycle.guard_steps, st.cycle.scene = 0, scene
                if classifier is not None:
                    # MS7f view 1: the narrow overhead view of the detection.
                    frame = env.narrow_overhead(i, seen.centre)
                    if on_classify is not None:
                        on_classify("overhead", i, frame, seen.size, step, st.cycle, detectors[i])
                    logp, _ = classifier.log_probs(frame, class_names, size=seen.size)
                    st.cycle.read_logp = logp
                    st.cycle.read_class = cls = summed_class(class_names, logp)
                    if vote_weights is not None:  # the fused vote's partial sum sets the box
                        cls = fused_class(class_names, vote_weights, logp)
                st.cycle.seen_yaw = float(scene.brick_yaw[target])
                st.cycle.box = desk_boxes.lighter_box(class_boxes[i][cls], st.box_load)
                st.cycle.volume = math.prod(seen.size)
                st.cycle.seen_size = seen.size
                env.marker_points[i] = torch.tensor(seen.centre)
                env.set_context_box(i, st.cycle.box)
                st.counts["selections"] += 1
                st.counts["wrong_overhead_class"] += cls != true_classes[target]
                if box_walls:
                    st.counts["box_wall_refusals"] += box_wall_refusals(st.selector, scene, target)
                if overhang_guard:
                    st.counts["overhang_refusals"] += st.selector.overhang_refused
                if st.marker is not None:
                    st.counts["marker_error_mm"] += st.marker.error_mm()
            else:
                env.target_part[i] = target
                if per_part:
                    st.cycle.box = desk_boxes.lighter_box(part_sets[i][target], st.box_load)
                    st.cycle.volume = math.prod(env.parts[target].size)
                    env.set_context_box(i, st.cycle.box)
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
            if scripted_place:
                for i, st in enumerate(states):
                    c, h = st.cycle, int(held[i])
                    if c is None or c.taken_over or not rig.idle[i] or bool(tracker.done[i]):
                        continue
                    if h >= 0 and float(pos[i, h, 2] - c.start_z[h]) >= SCRIPTED_PLACE_RISE:
                        _take_over(
                            rig,
                            env,
                            st,
                            i,
                            ordinal,
                            reason="scripted_place",
                            holding=True,
                            start_z=c.start_z,
                        )
                        c.taken_over = True  # one scripted place a cycle, taken or declined
            if carry_guard:
                for i, st in enumerate(states):
                    c, d = st.cycle, detectors[i]
                    if c is None or c.handoff is not None or not rig.idle[i] or tracker.done[i]:
                        continue
                    # The stall (MS7g design 5.2): the census's air stall for AIR_STALL_STEPS
                    # steps of this cycle (it ran at least as long as the cycle is old), or a
                    # guard held for GUARD_STALL_STEPS. A held part goes to the rig once a
                    # pending class vote has set its box; with none held the jaws open and the
                    # cycle fails at its target (F1).
                    air = stalls[i].stalled and step - c.start_step >= AIR_STALL_STEPS
                    voting = c.vote is not None or c.lift_view is not None
                    if d.holding and (air or c.guard_run >= GUARD_STALL_STEPS) and not voting:
                        _hand_off(st, rig, i, d, float(gaps[i]))
                    elif air and not d.holding:
                        c.stall_open = True
                        st.fail_mem.book(c.target_xy)
                        _close_cycle(st, "stall_open", [], step, None)
                        st.idle_since, st.opening = step, True
        # 3. Compose: the rig's row where it drives, the policy's where the rig is idle.
        rig_action = rig.actions().to("cpu")
        action_cpu = rig_action.clone()
        picks = [rig.pick_point(i).to(pos.device) for i in range(num_envs)]
        for i in range(num_envs):
            pick_z_hist[i].append(float(picks[i][2]))
        if policy_action is not None:
            idle = torch.tensor(rig.idle)
            action_cpu[idle] = policy_action[idle]
            if close_gate or squeeze_floor or square_wrist or pick_snap or dwell_close or wrist_fix:
                for i, st in enumerate(states):
                    if st.cycle is None or not rig.idle[i] or bool(tracker.done[i]):
                        continue
                    c = st.cycle
                    if wrist_fix:
                        _wrist_look(st, env, rig, i, picks[i], ctx_target[i])
                    # CONTEXT_TARGET_NAMES: z at 2, width at 5, half height 6. ``wrist_fix``
                    # serves the last wrist look's row; the policy keeps ``ctx_target``'s.
                    row = ctx_target[i] if c.serving_row is None else c.serving_row
                    width = float(row[5])
                    gap = float(action_cpu[i, 6])
                    # ``census_step`` and the hold census run after this block, so the fields
                    # read the steps before this one: no hold and no lift so far in the cycle.
                    if rearm_hold:  # it needs perceived and refuses contact_hold
                        held_so_far = c.stop  # the detector's, cleared by a hold with no lift
                    elif perceived and not contact_hold:
                        held_so_far = c.hold_detected or c.lift_detected  # the detector's
                    else:
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
                    hold_open = False
                    if pick_snap:
                        cmd = commanded_pick(rig.ik, action_cpu[i, :6])
                        c.snap = snap_shift(
                            cmd,
                            picks[i],
                            row,
                            c.snap,
                            2.0 * GRIPPER_OPEN_POS * float(openings[i]),
                            held=held_so_far,
                            thin=thin_snap,
                        )
                        hist = pick_z_hist[i]
                        descending = len(hist) > GATE_WINDOW and hist[0] - hist[-1] > GATE_DESCENT
                        measured_gap = 2.0 * GRIPPER_OPEN_POS * float(openings[i])
                        hold_open = snap_hold(
                            cmd,
                            picks[i],
                            row,
                            c.snap,
                            descending,
                            held=held_so_far,
                            thin=thin_snap,
                            jaw_gap=measured_gap if snap_closed else None,
                        )
                        if any(c.snap):
                            snapped = snapped_arm(rig.ik, action_cpu[i, :6], c.snap)
                            if snapped is not None:
                                action_cpu[i, :6] = snapped.to(action_cpu.dtype)
                                c.snapped_steps += 1
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
                    if hold_open:
                        landing = 2.0 * (grasp.landing_half_gap(width) - grasp.JAW_HOLD_BAND)
                        if gap < landing:
                            gap = landing
                            c.snap_held_steps += 1
                    if dwell_close and c.latched > 0:
                        # The close runs until the policy's own gap command closes or the cap.
                        if float(policy_action[i, 6]) < width or c.latched >= CLOSE_LATCH_STEPS:
                            c.latched = -1
                        else:
                            c.latched += 1
                            gap = min(gap, grasp.close_gap(width))
                    elif dwell_close and c.latched == 0:
                        c.dwell = dwell_count(
                            c.dwell,
                            picks[i],
                            row,
                            2.0 * GRIPPER_OPEN_POS * float(openings[i]),
                            gap,
                            held=held_so_far,
                        )
                        if c.dwell >= DWELL_STEPS:
                            c.latched = 1
                            c.dwell_closes += 1
                            gap = min(gap, grasp.close_gap(width))
                            replan[i] = True
                    if squeeze_floor:
                        floored = floored_gap(gap, width)
                        if floored != gap:
                            gap = floored
                            st.cycle.floored_steps += 1
                    action_cpu[i, 6] = gap
        if carry_guard:
            for i, st in enumerate(states):
                c = st.cycle
                if c is None or not rig.idle[i] or bool(tracker.done[i]):
                    continue
                d, p = detectors[i], st.perceiver
                pose, size = p.box_poses[c.box], p.box_sizes[c.box]
                lifted = d.lifted or (c.carried and not c.carry_released)
                # A pending class vote can still move the box: no box is served until it decides.
                voting = c.vote is not None or c.lift_view is not None
                deep = in_box_interior(picks[i][:2], pose, size, GUARD_BOX_MARGIN_M)
                inside = deep and not voting
                guarding = d.holding and lifted and not inside and c.handoff != "declined"
                gap, kept = float(action_cpu[i, 6]), None
                if guarding:
                    gap, kept = guarded_gap(gap, float(gaps[i]), gap_cmd[i], c.guard_gap)
                c.guard_gap = kept
                # A vote's steps are no stall: the policy opens over the box it was served.
                c.guard_run = 0 if kept is None else c.guard_run + (not voting)
                if kept is not None:
                    action_cpu[i, 6] = gap
                    c.guard_steps += 1
                    c.guard_vote_steps += voting and deep
                    c.guard_rim_steps += not voting and in_box_interior(picks[i][:2], pose, size)
        for i, st in enumerate(states):
            if st.opening:
                action_cpu[i, 6] = 2.0 * GRIPPER_OPEN_POS  # ``passenger_drop``: let both go
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
                        target_quat=truth.quat[i, c.target],
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
            if stall_end:
                c.closed_empty = closed_empty_count(c.closed_empty, grip_open, detectors[i].holding)
            if perceived:
                lift_now = rig.idle[i] and detectors[i].lift_started and not c.carried
                if (
                    missed_lift
                    and rig.idle[i]
                    and not (c.carried or c.lift_detected or detectors[i].lifted)
                ):
                    in_tray = over_tray(pick, lean, wall_height)
                    c.carried = lift_now = carry_started(c, not grip_open, pick, in_tray)
                    st.counts["missed_lifts"] += c.carried
                if lift_now:
                    # The class at lift: the box follows the class drawn for the part between
                    # the jaws, the target or not; nothing between them keeps the row.
                    true_cls, part = wrist_class(env, i)
                    views = st.perceiver.views  # ``class_views``: a third view on a disagreement
                    lighter = {
                        c: desk_boxes.lighter_box(own, st.box_load)
                        for c, own in class_boxes[i].items()
                    }
                    drawn, taken, box = lift_class_box(
                        true_cls,
                        partial(st.perceiver.classify, part=part),
                        lighter,
                        c.box,
                        seen=c.seen_class,
                        third=None if views is None else partial(views.view, part),
                    )
                    if classifier is not None:
                        # The stand-in above drew what it draws without the switch; the wrist
                        # frame decides (MS7f view 2), now or ``lift_view_steps`` later.
                        c.lift_view = (step + lift_view_steps, step, true_cls)
                    else:
                        if drawn is None:
                            st.counts["lifts_empty"] += 1
                        else:
                            st.counts["lifts_classified"] += 1
                            st.counts["wrong_wrist_class"] += drawn != true_cls
                            if views is not None:
                                st.counts["third_views"] += drawn != c.seen_class
                                st.counts["votes_changed"] += taken != drawn
                                st.counts["wrong_voted_class"] += taken != true_cls
                        if box is not None:
                            c.box = box
                            env.set_context_box(i, box)
                            replan[i] = True
                if c.lift_view is not None and step >= c.lift_view[0]:
                    _, lift_step, true_cls = c.lift_view
                    c.lift_view = None
                    if not (detectors[i].lifted or c.carried):
                        st.counts["lift_views_lost"] += 1  # the selection's box stays
                        continue
                    lighter = {
                        cls: desk_boxes.lighter_box(own, st.box_load)
                        for cls, own in class_boxes[i].items()
                    }
                    # View 2 decides, or ends the cycle on a passenger.
                    frame = classifier.wrist_frame(env, i)
                    if on_classify is not None:
                        on_classify("lift", i, frame, c.seen_size, step, c, detectors[i])
                    logp, rider = classifier.log_probs(
                        frame, class_names, wrist=True, size=c.seen_size
                    )
                    read = summed_class(class_names, logp)
                    c.lift_class, c.rider_read, c.taken_class = read, bool(rider), None
                    c.lift_logp, c.carry_logp = logp, None  # the MS7g census, view 2's read
                    st.counts["passenger_reads"] += rider
                    if passenger_drop and rider and tray and over_tray(pick, lean, wall_height):
                        st.counts["passenger_drops"] += 1
                        st.fail_mem.book(c.target_xy)
                        _close_cycle(st, "passenger", [], step, None)
                        st.idle_since, st.opening = step, True
                        continue
                    if true_cls is None:
                        st.counts["lifts_empty"] += 1
                    else:
                        st.counts["lifts_classified"] += 1
                        st.counts["wrong_wrist_class"] += read != true_cls
                    # The fused vote always reads the carry view, and view 2's partial sum
                    # sets the box now.
                    split = class_views and (vote_weights is not None or read != c.read_class)
                    c.bank_at = lift_step + CARRY_VIEW_STEPS if bank_views and not split else None
                    if split:
                        c.vote = (read, true_cls, lift_step + CARRY_VIEW_STEPS, logp)
                        c.carry_due = carry_vote_steps(step, c.vote[2], carry_votes)
                        c.carry_vote_logp = []
                    if not split or vote_weights is not None:
                        now = read
                        if vote_weights is not None:
                            now = fused_class(class_names, vote_weights, c.read_logp, logp)
                        _, taken, box = lift_class_box(now, lambda x: x, lighter, c.box)
                        c.taken_class = taken
                        if box is not None:
                            c.box = box
                            env.set_context_box(i, box)
                            replan[i] = True
                if c.vote is not None and c.carry_due and step >= c.carry_due[0]:
                    # ``carry_votes``: one more carry frame for the vote, if the part is still up.
                    c.carry_due.pop(0)
                    if detectors[i].lifted or c.carried:
                        frame = classifier.wrist_frame(env, i)
                        if on_classify is not None:
                            on_classify("wrist", i, frame, c.seen_size, step, c, detectors[i])
                        logp, _ = classifier.log_probs(frame, class_names, True, c.seen_size)
                        c.carry_vote_logp.append(logp)
                        st.counts["carry_votes_read"] += 1
                    else:
                        st.counts["carry_votes_lost"] += 1
                if c.vote is not None and step >= c.vote[2]:
                    # MS7f view 3: the carry frame is the third vote, if the part is still up.
                    read, true_cls, _, lift_logp = c.vote
                    c.vote = None
                    if not (detectors[i].lifted or c.carried):
                        st.counts["votes_lost"] += 1
                        c.vote_lost = True
                        continue
                    frame = classifier.wrist_frame(env, i)
                    if on_classify is not None:
                        on_classify("carry", i, frame, c.seen_size, step, c, detectors[i])
                    lighter = {
                        cls: desk_boxes.lighter_box(own, st.box_load)
                        for cls, own in class_boxes[i].items()
                    }
                    carry_logp, _ = classifier.log_probs(frame, class_names, True, c.seen_size)
                    c.carry_logp = carry_logp  # the MS7g census, view 3's read
                    logps = (c.read_logp, lift_logp, *c.carry_vote_logp, carry_logp)
                    views_logp = [v for v in logps if v]
                    summed = summed_class(class_names, *views_logp)  # no overhead read: two
                    if vote_weights is not None:
                        summed = fused_class(class_names, vote_weights, *logps)
                    _, taken, box = lift_class_box(summed, lambda x: x, lighter, c.box)
                    c.taken_class = taken
                    st.counts["third_views"] += 1
                    st.counts["votes_changed"] += taken != read
                    st.counts["wrong_voted_class"] += true_cls is not None and taken != true_cls
                    if box is not None:
                        c.box = box
                        env.set_context_box(i, box)
                        replan[i] = True
                if c.bank_at is not None and step >= c.bank_at:
                    c.bank_at = None  # ``bank_views``: recorded, never read
                    if detectors[i].lifted or c.carried:
                        frame = classifier.wrist_frame(env, i)
                        on_classify("bank", i, frame, c.seen_size, step, c, detectors[i])
            elif (
                adopt_held
                and per_part
                and rig.idle[i]
                and h >= 0
                and h not in (c.target, c.adopted)
                and float(pos[i, h, 2] - c.start_z[h]) >= oracle.LIFT_MIN_RISE
            ):
                # The teacher's LIFT that comes up with another part carries that part to its
                # own box, and its marker and box row follow the part (adopt_held_part).
                c.adopted = h
                env.target_part[i] = h
                c.box = desk_boxes.lighter_box(part_sets[i][h], st.box_load)
                c.volume = math.prod(env.parts[h].size)
                env.set_context_box(i, c.box)
                replan[i] = True
        # The MS7c hold census: what the detector read at the top of this step (the jaw gap
        # measured then against the command the last step sent, whoever drives) on the cycle
        # open now.
        for i, st in enumerate(states):
            if bool(tracker.done[i]):
                continue
            d, c, h = detectors[i], st.cycle, int(held[i])
            gap, pick_z = float(gaps[i]), float(top_picks[i][2])
            stall = stalls[i]
            stall.update(top_picks[i], tray and over_tray(top_picks[i][:2], lean, wall_height))
            if c is not None:
                if d.holding and c.first_hold_at is None:
                    c.first_hold_at = step - c.start_step
                c.hold_detected = c.first_hold_at is not None
                c.lift_detected = c.lift_detected or d.lifted
                if rearm_hold:
                    st.counts["rearms"] += rearm_stop(c, d)
                # The MS7g census (``MS7G_CENSUS``): the cycle's first lift, the jaws' first
                # open after it, and the air stalls. Observation only.
                if (c.lift_detected or c.carried) and not c.lift_seen:
                    c.lift_seen, c.lift_part = True, h
                    c.held_class = keys[h] if h >= 0 else None
                if jaws_opened(c, gap):
                    part = h if h >= 0 else c.lift_part if c.lift_part >= 0 else c.target
                    c.release_place = release_place(
                        top_picks[i][:2],
                        true_boxes[i] if per_part else (),
                        part_sets[i][part] if per_part else (),
                        lean if tray else None,
                        wall_height,
                    )
                if stall.stalled and d.holding:
                    c.air_stall_steps += 1
                    c.air_stalls += stall.entered
                elif stall.stalled:
                    c.air_stall_empty_steps += 1
            if hold_trace is not None:
                rise = float(pos[i, h, 2] - c.start_z[h]) if h >= 0 and c is not None else 0.0
                ordinal_i = -1 if c is None else c.ordinal
                row = [i, step, ordinal_i, gap, gap_cmd[i], pick_z, h, rise]
                hold_trace.append(row if sensors is None else row + [forces[i]])
        arm_clamped, overshoot = clamp_arm_actions(action_cpu[:, :6], limits)
        tracker.record_clamps(overshoot)
        clamped = torch.cat([arm_clamped, action_cpu[:, 6:7]], dim=-1)
        if park_done:
            # ``tracker.done`` is read before this step: an env keeps the command it got on
            # the last step of its episode for as long as the batch runs.
            if parked is not None:
                clamped[tracker.done] = parked[tracker.done]
            parked = clamped.clone()
        if force_cap:
            for i in range(num_envs):
                if forces[i] > FORCE_CAP_N:
                    # MS7d L7: the arm holds still where it is; the gripper command passes.
                    clamped[i, :6] = robot.data.joint_pos[i, :6]
                    states[i].counts["force_capped"] += 1
        last_roll = clamped[:, WRIST_ROLL_INDEX].clone()
        gap_cmd = [commanded_gap(float(g), gripper_mode) for g in clamped[:, 6]]
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
            if st.cycle is not None and st.cycle.handoff == "carrying":
                _end_handoff(st, step)  # ``carry_guard``: the loop grades its own carry
                rig.takeover_result[i] = None
                continue
            orc = rig.oracles[i]
            if getattr(orc, "next_brick", None) is not None and not wave3:
                st.preselected = (
                    orc.next_brick,
                    orc.next_brick_start,
                    orc.next_grasp,
                    orc._next_forced,
                )
            if result:
                st.takeovers_stowed += 1
                if wave3 and st.cycle is not None:
                    st.cycle.takeover_end = "placed"  # the rig released over its box
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
        was_done = tracker.done.clone()
        tracker.record_step(step, dones | loop_done, truncated.cpu(), success)
        for i in torch.nonzero(newly_env).flatten().tolist():
            st = states[i]
            st.left = _left_parts(st.selector, pos[i], in_bin[i])  # pre-step: the reset ran
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
        for i in torch.nonzero(tracker.done & ~was_done).flatten().tolist():
            st = states[i]
            if st.span is not None:  # S0: the last span ends with the episode, step + 1 steps
                st.intervals.append(span_intervals(st.span, step + 1))
                st.span = None
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
    held = [p for p in range(k) if env.parts[p].name in heldout]
    for i, result in enumerate(tracker.results):
        st = states[i]
        final = tracker.stowed_mask(i)
        if perceived:
            _credit_placed(st, final)
        boxes = {}
        if per_part:
            boxes = box_census(
                env.parts,
                destination[i],
                final,
                final_box[i],
                st.classes,
                st.census,
                box_count[i],
                class_names,
                keys,
                part_sets[i],
            )
            boxes["class_to_box"] = class_to_box[i]
            if getattr(env, "categories", False):
                boxes["class_boxes"] = {c: list(own) for c, own in class_boxes[i].items()}
                boxes["stowed_by_category"] = boxes["stowed_by_class"]
            if rounds is not None:
                boxes["box_draw_rounds"] = int(rounds[i])
                boxes.update(geometry[i])
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
                "stowed_heldout": {
                    "stowed": sum(bool(final[p]) for p in held),
                    "total": len(held),
                },
                "events": {name: st.events[name] for name in EVENT_NAMES},
                "cycle_intervals": list(st.intervals),
                "done_reason": st.done_reason,
                "takeovers": st.takeovers,
                "takeovers_stowed": st.takeovers_stowed,
                "takeovers_failed": st.takeovers_failed,
                "select_wait_steps": st.select_wait_steps,
                "rakes": st.rakes,
                "left_parts": st.left,
                **boxes,
            }
        )
        if ms7f_tray:
            episodes[-1]["hidden_share"] = hidden[i]
        if box_empty:
            # Each emptied part: (part, the box it was emptied from, graded correct).
            episodes[-1]["emptied_parts"] = [
                [b, int(emptied_box[i, b]), bool(emptied_ok[i, b])]
                for b in range(k)
                if int(emptied_box[i, b]) >= 0
            ]
        if stretch:
            episodes[-1]["interventions"] = dict(interventions[i])
        if sensors is not None:
            episodes[-1]["torque"] = {
                **{name: st.counts[name] for name in TORQUE_COUNTS},
                "approach_force": force_summary(st.forces),
            }
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
        "steps_total": steps_total,
        "stows_per_hour": stows_per_hour(stowed_total, steps_total),
        "stowed_heldout": {
            key: sum(e["stowed_heldout"][key] for e in episodes) for key in ("stowed", "total")
        },
        "events": {name: sum(e["events"][name] for e in episodes) for name in EVENT_NAMES},
        "cycle_intervals": interval_sums(episodes),
        **(_sum_box_rows(episodes) if per_part else {}),
        "takeovers": sum(e["takeovers"] for e in episodes),
        "takeovers_stowed": sum(e["takeovers_stowed"] for e in episodes),
        "takeovers_failed": sum(e["takeovers_failed"] for e in episodes),
        "rakes": sum(e["rakes"] for e in episodes),
        "triggers": {} if triggers is None else dict(triggers.counts),
        "takeovers_declined": 0 if triggers is None else triggers.declined,
        **(
            {"takeovers_any_part": sum(st.counts["any_part_takeovers"] for st in states)}
            if wave3
            else {}
        ),
        "clamp_fraction": (
            sum(e["clamp_steps"] for e in episodes) / steps_total if steps_total else 0.0
        ),
        "wall_sec": round(wall, 1),
        "wall_sec_per_episode": round(wall / num_envs, 1),
        "hold_agreement": agreement(row for e in episodes for row in e["cycle_census"]),
        "perception": {
            "perceived": switches["perceived"],
            "noise_scales": switches["noise_scales"],
            "contact_hold": switches["contact_hold"],
            **{
                name: sum(st.counts[name] for st in states)
                for name in PERCEPTION_COUNTS
                + (BOX_WALL_COUNTS if box_walls else ())
                + (MARKER_COUNTS if marker_bias else ())
                + (WRIST_COUNTS if wrist_fix else ())
                + (CLASSIFIER_COUNTS if classifier is not None else ())
                + (LIFT_VIEW_COUNTS if lift_view_steps else ())
                + (MISSED_LIFT_COUNTS if missed_lift else ())
                + (CARRY_VOTE_COUNTS if carry_votes else ())
                + (OVERHANG_COUNTS if overhang_guard else ())
            },
            # The MS7g census: the category order of the cycle rows' log probabilities.
            **({"logp_order": list(class_names)} if classifier is not None else {}),
        },
    }
    if ms7f_tray:
        shares = [v for h in hidden for v in h]
        payload["hidden_share_mean"] = sum(shares) / len(shares) if shares else None
    if sensors is not None:
        payload["torque"] = {
            "hold_force": bool(hold_force),
            "force_cap": bool(force_cap),
            "hold_force_max_n": HOLD_FORCE_MAX_N,
            "force_cap_n": FORCE_CAP_N,
            "torque": scales["torque"],
            "torque_k": scales["torque_k"],
            **{name: sum(st.counts[name] for st in states) for name in TORQUE_COUNTS},
            "approach_force": force_summary([f for st in states for f in st.forces]),
        }
    return payload


def interval_sums(episodes: list[dict]) -> dict:
    """S0 (MS7f design section 8.1): each episode's ``cycle_intervals`` summed per cycle outcome,
    the census class of ``cycle_classes`` (``stowed``, ``wrong_box``, ``held_no_lift`` and the
    rest), with the cycles of each. Pure."""
    by: dict[str, dict] = {}
    for ep in episodes:
        for outcome, four in zip(ep["cycle_classes"], ep["cycle_intervals"]):
            row = by.setdefault(outcome, {"cycles": 0, "steps": [0, 0, 0, 0]})
            row["cycles"] += 1
            row["steps"] = [a + b for a, b in zip(row["steps"], four)]
    return {"names": list(INTERVAL_NAMES), "by_outcome": dict(sorted(by.items()))}


def between_cycles(
    env, i: int, counts, emptied_ok, emptied_box, box_empty, refill, strays=None, loads=()
) -> bool:
    """MS7f: a person's events between env ``i``'s cycles, before its next selection, at zero
    simulated time (MS7f design 4.1, 5.3 and 5.4). ``box_empty`` empties every full box
    (``env.empty_full_boxes``): each of its parts is graded at that moment, once, into
    ``emptied_ok`` and ``emptied_box`` (row ``i``), then parked, and the perceived state never
    sees it again, and every record in ``loads`` (a list of volume per box: the loop's
    ``EnvState.box_load``, a teacher's ``HeapOracle.box_load``) starts that box over at 0.
    ``refill`` refills the tray (``env.refill_tray``). ``counts`` books each event. ``strays``, a stretch run's interventions (MS7f design 10.4 rule 2), first puts every
    part at rest outside the tray and every box back into the tray and books one per part. The
    person reads the simulator; no loop decision reads what they read. Returns whether an event
    moved a part."""
    fired = False
    if strays is not None:
        lost = env.stray_parts(i)
        if lost:
            env.return_to_tray(i, lost)
            strays["strays"] += len(lost)
            fired = True
    if box_empty:
        for box, parts, ok in env.empty_full_boxes(i):
            counts["box_empty"] += 1
            emptied_box[i, parts] = box
            emptied_ok[i, parts] = torch.tensor(ok)
            for load in loads:
                load[box] = 0.0
            fired = True
    if refill and env.refill_tray(i):
        counts["tray_refill"] += 1
        fired = True
    return fired


def miss_streak(count: int, seen: int, misses: int) -> tuple[int, int]:
    """MS7f stretch rule 3: after a cycle closes with ``count`` correct stows in the episode so
    far, the new (correct stows seen, cycles closed since they last grew). A stow settles after
    its cycle closed, so it resets the streak at the next close. Pure."""
    return (count, 0) if count > seen else (seen, misses + 1)


def hidden_share(env, pos, quat, tool) -> float:
    """MS7f design 5.5: the share of the parts in the tray (``layout.in_tray``) that
    the perceived state's two structural rules hide from the overhead camera
    (``perception._hidden``: covered from above, or near the ``tool`` point). ``pos`` and
    ``quat`` are one env's (K, 3) and (K, 4) metric reads. A metric; no decision reads it."""
    lean = float(getattr(env, "tray_lean_deg", 0.0))
    height = getattr(env, "tray_wall_height", layout.TRAY_WALL_HEIGHT)
    idx = [b for b in range(pos.shape[0]) if layout.in_tray(pos[b].tolist(), lean, height)]
    if not idx:
        return 0.0
    sizes = torch.tensor([env.parts[b].size for b in idx])
    seen = perception._hidden(pos[idx], quat[idx], sizes, tuple(tool[:2].tolist()))
    return len(seen) / len(idx)


def _sum_box_rows(episodes: list[dict]) -> dict:
    """The payload's desk-box rows: every episode's class, family and box rows, its parts per
    box count (the episode's own count, MS7e design section 8.1) and its wrong-box parts."""
    by_class: dict[str, dict] = {}
    by_family: dict[str, dict] = {}
    by_count: dict[str, dict] = {}
    by_box: list[dict] = []
    by_category: dict[str, dict] = {}
    for ep in episodes:
        count_rows = {
            str(ep["box_count"]): {"stowed": ep["bricks_stowed"], "total": ep["num_bricks"]}
        }
        for table, rows in (
            (by_class, ep["stowed_by_class"]),
            (by_family, ep["stowed_by_family"]),
            (by_count, count_rows),
            (by_category, ep.get("stowed_by_category", {})),
        ):
            for name, row in rows.items():
                total = table.setdefault(name, {"stowed": 0, "total": 0})
                for key in ("stowed", "total"):
                    total[key] += row[key]
        for m, row in enumerate(ep["stowed_by_box"]):
            if m == len(by_box):
                by_box.append({"stowed": 0, "total": 0})
            for key in ("stowed", "total"):
                by_box[m][key] += row[key]
    return {
        "stowed_by_class": by_class,
        "stowed_by_family": by_family,
        "stowed_by_box_count": dict(sorted(by_count.items(), key=lambda kv: int(kv[0]))),
        "stowed_by_box": by_box,
        "wrong_box_parts": sum(ep["wrong_box_parts"] for ep in episodes),
        "wrong_box_cycles": sum(ep["cycle_classes"].count("wrong_box") for ep in episodes),
        **({"stowed_by_category": by_category} if by_category else {}),
    }


def _book_census(st: EnvState, cycle: Cycle, reason: str) -> None:
    st.classes.append(cycle_class(cycle, reason))
    close_cm = cycle.first_close_cm
    st.first_close_cm.append(None if close_cm is None else round(close_cm, 2))
    row = {name: getattr(cycle, name) for name in CENSUS_EXPORT}
    row["first_held_is_target"] = cycle.first_held == cycle.target
    row["last_held_is_target"] = cycle.last_held == cycle.target
    if cycle.read_logp is not None:  # ``classifier``: the reads behind the cycle's box
        row |= {k: getattr(cycle, k) for k in CLASSIFIER_CENSUS}
    if cycle.guard_steps is not None:  # ``carry_guard``
        row |= {k: getattr(cycle, k) for k in CARRY_GUARD_CENSUS}
    st.census.append({k: _rounded(v) for k, v in row.items()})


def _rounded(v):
    """A census value as the payload records it: a float, also in a list, to 4 decimals."""
    if isinstance(v, list):
        return [_rounded(x) for x in v]
    return round(v, 4) if isinstance(v, float) else v


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
    rig,
    env,
    st: EnvState,
    i: int,
    ordinal: list[int],
    *,
    reason: str,
    holding: bool = False,
    start_z=None,
    any_part: bool = False,
) -> bool:
    """Hand env ``i`` to the rig on the loop's target; False when the rig declines.

    A decline with nothing held ends the cycle as a failure (phase B design 5.5: "the cycle
    ends as a failure, and the loop moves on"); ``_close_cycle`` marks the selector's failure
    memory. A decline while the jaws hold a part (``holding``, the loop's held read) keeps the
    six-brick behavior: the cycle runs on and the caller's cooldown spaces the next trigger.
    ``OracleRig.resume`` declines most of a correct carry (RULING P32: a part above
    ``LIFT_HEIGHT - LIFT_MIN_RISE``), and ending the cycle there would spend it and mark the
    failure memory at a point in the air. ``start_z`` is the scripted place's: the cycle's part
    heights at its open, from which the rig's LIFT measures the held part's rise.

    ``any_part`` is the wave 3 DAgger mode's: when the rig declines the target with nothing held
    (no grasp ranks on it, or no plan reaches it), the teacher works the part it ranks first
    instead, from the state the policy left, and the loop books its target failed where it was
    perceived. The smoke heap of 2026-10-01 (p = 90) declined 8 of its 8 close_miss cycles."""
    lifted = {} if start_z is None else {"start_z": start_z}
    taken = rig.resume(i, target=st.cycle.target, **lifted)
    if not taken and any_part and not holding and rig.resume(i):
        st.fail_mem.book(st.cycle.target_xy)
        st.counts["any_part_takeovers"] += 1
        taken = True
    if taken:
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


def _hand_off(st: EnvState, rig, i: int, detector: HoldDetector, gap: float) -> None:
    """``carry_guard``: hand the part the detector holds to the rig's perceived carry
    (``OracleRig.resume(carry=...)``), over the served box's perceived pose at
    ``carry_release_z`` and, if that does not plan, at the teacher's release height. ``gap`` is
    the measured jaw gap. The cycle's ``handoff`` reads "carrying" until the rig hands the env
    back (``_end_handoff``), or "declined"; one handoff a cycle."""
    c, p = st.cycle, st.perceiver
    x, y, _ = p.box_poses[c.box]
    z = carry_release_z(detector.start_z, p.box_sizes[c.box])
    for height in dict.fromkeys((z, desk_boxes.DESK_BOX_RELEASE_Z)):
        carry = ((x, y, height), gap + CARRY_WIDTH_PAD_M, detector)
        if rig.resume(i, scene=c.scene, carry=carry):
            c.handoff = "carrying"
            return
    c.handoff = "declined"


HANDOFF_ENDS = {
    True: ("placed", "placed"),
    False: ("lifted_lost", "dropped"),
    None: ("failed", "takeover_failed"),
}
"""``carry_guard``: (handoff outcome, cycle reason) by where the hold ended during the rig's carry:
over a perceived box, off every box, or not at all (the rig handed the env back still holding)."""


def _end_handoff(st: EnvState, step: int) -> None:
    """``carry_guard``: the rig handed a carry back. The loop grades it as it grades a perceived
    release (``perceived_cycle_end``), by ``HANDOFF_ENDS``, books any failure at the target's
    perceived position and closes the cycle now."""
    c = st.cycle
    c.handoff, reason = HANDOFF_ENDS[c.handoff_over]
    if reason != "placed":
        st.fail_mem.book(c.target_xy)
    _close_cycle(st, reason, [], step, None)
    st.idle_since = step


class _TriggerState:
    """Per-env trigger counters and the offer of a takeover: the six-brick DAgger loop's logic
    on the loop's cycles (the MS4 four triggers, the random draw, the decline cooldown) plus
    the half-budget trigger (phase B design 5.5). ``offer`` runs once per step before the
    actions are composed, on the envs the policy holds. Caps (design 5.5): one takeover per
    loop cycle, ``max_per_episode_factor * cycle_budget`` per episode."""

    def __init__(
        self,
        num_envs: int,
        config: TakeoverConfig,
        run_seed: int,
        cycle_budget: int,
        any_part: bool = False,
    ):
        self.config = config
        self.any_part = any_part  # ``_take_over``'s, the wave 3 DAgger mode
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
            holding = int(held[i]) >= 0
            if _take_over(
                rig, env, st, i, ordinal, reason=reason, holding=holding, any_part=self.any_part
            ):
                self.counts[reason] += 1
            else:
                # The decline cooldown stays (phase B design 5.5). A decline on a held part
                # leaves the cycle running and the cooldown spaces the next trigger on it; a
                # decline with nothing held ends the cycle, and the cooldown spaces the next
                # cycle's first trigger from a scene the rig just refused.
                self.declined += 1
                self.cooldown[i] = DECLINE_COOLDOWN
        self.grip_open_prev = grip_open

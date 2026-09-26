"""The MS7c hold signal: is a part held, and was it lifted, read from proprioception alone.

MS7c design section 5. A real arm has no contact sensors, while the serving loop reads the
hold from simulator contact forces (``checker.held_brick_from_forces``) and the lift from the
part's true height. This detector reads only what the real arm reports: the measured jaw gap
(the sum of the two jaw joints, the stand-in for the one RS00 gripper motor's opening), the gap
the previous command asked for, and the FK pick point's height. Jaws that close on a part stall
at its width, wider than the command (the teacher commands ``grasp.close_gap(w)``, 6 mm under
the width); jaws that close on nothing converge to the command.

Hold entry: on ``confirm_steps`` steps in a row, the measured gap is at least ``min_excess``
wider than the command, at least ``min_gap`` wide, and spread at most ``stable`` over its last
``stable_steps`` readings (this step's and the ones before it). The stability condition keeps
the closing ramp itself (the env ramps each jaw at 0.1 m/s, several mm per step) from reading
as a hold, and it gates entry only. Exit: the excess or the width condition fails on
``exit_steps`` steps in a row, so a carry that jiggles the gap keeps its hold. Lift: the pick
point rises ``min_rise`` above its height at the step the hold began, latched until the hold
ends. Lost: a hold ends while lifted (a drop in the carry, or the release).

MS7d L6 (``hold_force``) adds an entry gate: on a step whose pick-point force is over the bar
(``core.torque``) the gate refuses a new hold (``blocked``), and the confirm run keeps counting,
so the hold enters on the first step the gate opens with the jaw-gap rule still met. A true
hold that closes with a finger pressed on the floor then enters at the lift, which takes the
finger off. The gate never ends a hold that has begun, and without it the detector is MS7c's.

``agreement`` scores the detector against the contact census per cycle and ``meets_bar`` is the
pre-registered bar. ``cycle_rows`` rebuilds those census rows from a per-step trace
(``loop.rollout(hold_trace=...)``) at any thresholds, for the calibration sweep. The detector
is plain floats.
"""

from __future__ import annotations

import statistics
from collections import deque

from maker_arm_lab.core.eval_loop import GRASP_HOLD_WINDOW
from maker_arm_lab.core.oracle import LIFT_MIN_RISE
from maker_arm_lab.robots.frames import JAW_FULL_GAP

MIN_EXCESS = 0.002  # INITIAL: m, measured over commanded gap; the MS7c calibration reads it
MIN_GAP = 0.003  # INITIAL: m, the narrowest measured gap that reads as a part between the jaws
STABLE = 0.0002  # INITIAL: m, the most the measured gap may spread over STABLE_STEPS readings
STABLE_STEPS = 3  # INITIAL: readings in the stability window, this step's included
CONFIRM_STEPS = 3  # INITIAL: consecutive steps all three entry conditions must hold
EXIT_STEPS = 2  # INITIAL: consecutive steps the excess or the width must fail to end a hold

TRUTH_MIN_HELD_STEPS = GRASP_HOLD_WINDOW[0]
"""Contact-held steps that make a cycle's truth hold: ``loop.CENSUS_MIN_HELD_STEPS``."""
BAR_PRECISION = 0.95
"""The pre-registered bar (MS7c design section 5): precision and recall of both events."""
BAR_RECALL = 0.95
BAR_OFFSET_STEPS = 5
"""The bar's median steps from the first contact-held step to the first detected hold."""

TRACE_FIELDS = (
    "env",
    "step",
    "cycle",
    "measured_gap",
    "commanded_gap",
    "pick_z",
    "held",
    "rise",
    "force",
)
"""One ``hold_trace`` row: the env, the loop step, the cycle ordinal (-1 between cycles), the
measured and commanded gaps (m), the pick point z (m), the contact-held part (-1 for none) and
its rise above the cycle's start height (m, 0.0 with none held or no cycle). The ninth, the
pick-point force (N, ``core.torque``), is there only when the rollout measured it (MS7d L6 or
L7); a row without it has the first eight."""


def commanded_gap(raw: float, gripper_mode: str) -> float:
    """m: the total jaw gap the raw gripper column asks for. Continuous: the column is the gap,
    clipped to [0, ``JAW_FULL_GAP``] as the env clamps each jaw to half of it. Binary: a
    negative command closes to 0, any other opens to the full gap."""
    if gripper_mode != "continuous":
        return 0.0 if raw < 0.0 else JAW_FULL_GAP
    return min(max(raw, 0.0), JAW_FULL_GAP)


class HoldDetector:
    """One env's hold state. ``update`` once per control step; after it, ``holding`` and
    ``lifted`` are the state and ``started``, ``lift_started``, ``ended``, ``lost`` and
    ``blocked`` the events of that step."""

    def __init__(
        self,
        min_excess: float = MIN_EXCESS,
        min_gap: float = MIN_GAP,
        stable: float = STABLE,
        stable_steps: int = STABLE_STEPS,
        confirm_steps: int = CONFIRM_STEPS,
        exit_steps: int = EXIT_STEPS,
        min_rise: float = LIFT_MIN_RISE,
    ):
        self.min_excess = min_excess
        self.min_gap = min_gap
        self.stable = stable
        self.stable_steps = stable_steps
        self.confirm_steps = confirm_steps
        self.exit_steps = exit_steps
        self.min_rise = min_rise
        self.reset()

    def reset(self) -> None:
        """A new episode: no hold and no gap history."""
        self.gaps = deque(maxlen=self.stable_steps)
        self.run = 0  # consecutive steps every entry condition held
        self.fails = 0  # consecutive steps an exit condition failed during the hold
        self.start_z = 0.0  # m, the pick point z at the step the hold began
        self.holding = self.lifted = False
        self.started = self.lift_started = self.ended = self.lost = self.blocked = False

    def update(
        self, measured_gap: float, commanded_gap: float, pick_z: float, entry: bool = True
    ) -> None:
        """One step: ``measured_gap`` read at the top of the step, ``commanded_gap`` the gap the
        previous step's command asked for (the measurement answers it), ``pick_z`` the FK
        pick point height, all in m. ``entry`` is the L6 gate, open by default: closed, it
        refuses a new hold on this step, and ``blocked`` marks the step a run first confirmed
        and was refused."""
        self.started = self.lift_started = self.ended = self.lost = self.blocked = False
        self.gaps.append(measured_gap)
        gripping = measured_gap - commanded_gap >= self.min_excess and measured_gap >= self.min_gap
        if self.holding:
            self.fails = 0 if gripping else self.fails + 1
            if self.fails >= self.exit_steps:
                self.ended, self.lost = True, self.lifted
                self.holding = self.lifted = False
            elif not self.lifted and pick_z - self.start_z >= self.min_rise:
                self.lifted = self.lift_started = True
            return
        full = len(self.gaps) == self.stable_steps
        still = full and max(self.gaps) - min(self.gaps) <= self.stable
        self.run = self.run + 1 if gripping and still else 0
        if self.run >= self.confirm_steps and not entry:
            self.blocked = self.run == self.confirm_steps
        elif self.run >= self.confirm_steps:
            self.holding = self.started = True
            self.run = self.fails = 0
            self.start_z = pick_z


def _rate(hit: int, miss: int) -> float | None:
    return hit / (hit + miss) if hit + miss else None


def agreement(
    rows, min_held_steps: int = TRUTH_MIN_HELD_STEPS, min_rise: float = LIFT_MIN_RISE
) -> dict:
    """The detector against the contact census over per-cycle ``rows`` (``loop.CENSUS_EXPORT``
    rows, or ``cycle_rows``'s). Truth: held is ``held_steps >= min_held_steps``, lifted is
    ``max_rise >= min_rise``; the detector's are ``hold_detected`` and ``lift_detected``.
    Per event: true and false positives, false negatives, true negatives, precision and recall
    (None with an empty denominator). The offset is the median of ``first_hold_at`` minus
    ``first_held_at`` over the cycles that have both. Pure."""
    rows = list(rows)
    out = {"cycles": len(rows)}
    for event, truth_of, seen_key in (
        ("hold", lambda r: r["held_steps"] >= min_held_steps, "hold_detected"),
        ("lift", lambda r: r["max_rise"] >= min_rise, "lift_detected"),
    ):
        pairs = [(bool(truth_of(r)), bool(r[seen_key])) for r in rows]
        tp = sum(t and s for t, s in pairs)
        fp = sum(s and not t for t, s in pairs)
        fn = sum(t and not s for t, s in pairs)
        out[event] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": len(pairs) - tp - fp - fn,
            "precision": _rate(tp, fp),
            "recall": _rate(tp, fn),
        }
    offsets = [
        r["first_hold_at"] - r["first_held_at"]
        for r in rows
        if r["first_hold_at"] is not None and r["first_held_at"] is not None
    ]
    out["offset_cycles"] = len(offsets)
    out["first_hold_offset_median"] = statistics.median(offsets) if offsets else None
    return out


def meets_bar(summary: dict) -> bool:
    """The pre-registered bar: precision and recall at least 0.95 for the hold and the lift,
    and a median first-hold offset of at most 5 steps. A missing rate or offset fails."""
    rates = [summary[e][k] for e in ("hold", "lift") for k in ("precision", "recall")]
    offset = summary["first_hold_offset_median"]
    if offset is None or any(r is None for r in rates):
        return False
    bars = (BAR_PRECISION, BAR_RECALL) * 2
    return all(r >= b for r, b in zip(rates, bars)) and offset <= BAR_OFFSET_STEPS


def cycle_rows(trace, force_max: float | None = None, **thresholds) -> list[dict]:
    """Replay ``HoldDetector(**thresholds)`` over one rollout's ``hold_trace`` rows (lists in
    ``TRACE_FIELDS`` order, in step order) and rebuild each cycle's census row: the truth's
    ``held_steps``, ``max_rise`` (rounded as the census rounds it) and ``first_held_at``, and
    the detector's ``hold_detected``, ``lift_detected`` and ``first_hold_at``. At the default
    thresholds the rows equal the rollout's own census. ``force_max`` (N, None: no gate) gates
    each entry on the rows' force column as L6 does live, so a trace read with ``hold_force``
    replays at another bar; it needs the ninth column. Pure."""
    detectors: dict[int, HoldDetector] = {}
    cycles: dict[tuple[int, int], dict] = {}
    for env, step, ordinal, gap, cmd, pick_z, held, rise, *force in trace:
        d = detectors.setdefault(env, HoldDetector(**thresholds))
        d.update(gap, cmd, pick_z, force_max is None or force[0] < force_max)
        if ordinal < 0:
            continue
        c = cycles.setdefault(
            (env, ordinal),
            {
                "start": step,
                "held_steps": 0,
                "max_rise": 0.0,
                "first_held_at": None,
                "hold_detected": False,
                "lift_detected": False,
                "first_hold_at": None,
            },
        )
        if held >= 0:
            c["held_steps"] += 1
            c["max_rise"] = max(c["max_rise"], rise)
            if c["first_held_at"] is None:
                c["first_held_at"] = step - c["start"]
        if d.holding:
            c["hold_detected"] = True
            if c["first_hold_at"] is None:
                c["first_hold_at"] = step - c["start"]
        c["lift_detected"] = c["lift_detected"] or d.lifted
    rows = []
    for c in cycles.values():
        c.pop("start")
        c["max_rise"] = round(c["max_rise"], 4)
        rows.append(c)
    return rows

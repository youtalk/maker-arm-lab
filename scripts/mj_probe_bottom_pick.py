"""Probe: does a pick that opens the jaws at the BOTTOM clear a wall the shipped descend hits?

The shipped descend comes down with the jaws fully open, so each finger tip lands
``grasp.OPEN_HALF_GAP`` (52.4 mm) out from the part center. A tray wall inside that distance is
a hard stop and the part is unpickable (``docs/ms6-teacher-inspection.md`` section 3.2). This
script holds the jaws just clear of the part while the tips travel down, opens them once the
descend leg has been walked, and lets the shipped CLOSE run unchanged.

Three modes:

* ``--mode wall_table`` repeats section 3.2's read. One 2x4 brick, both closing axes, at the
  same part-center-to-wall distances, with the shipped three-retry ladder.
* ``--mode reach_band`` is the driven reach band of the lever-verification design section 6.1.
  It shares this script because the honest measurement is a driven descend and pick on a lone
  part, which is what this script already does. The comment above ``layout.SCATTER_MAX_RADIUS``
  says that bound is a TRACKING bound and that whoever re-measures it must drive real descends
  and must not query the solver, so this mode reports the joint tracking residual of a driven
  servo and never the solver's own convergence. It uses the SHIPPED descend.
* ``--mode heap`` runs one K = 50 held-out heap through ``scripts/mj_inspect_heap.py``'s own
  driver loop, so it writes the same ``events.jsonl`` and ``scripts/heap_inspect_read.py``
  reads it unchanged. Its seeds are pinned.

    MUJOCO_GL=egl uv run python scripts/mj_probe_bottom_pick.py --mode wall_table \\
        --out_dir /tmp/p0 --distances_mm 70,40

Prints exactly one ``PROBE_OK {json}`` or ``PROBE_FAIL {json}`` line and exits through
``os._exit``. Nothing under ``core/`` or ``source/`` changes: the rig subclass lives here and
is installed the way the inspection tool installs its oracle subclass.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from maker_arm_lab.core import grasp, layout, oracle
from maker_arm_lab.core import parts as _parts
from maker_arm_lab.core.batch_loop import add_batch_arguments, build_run, make_induction
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import (
    GRIPPER_OPEN_ACTION,
    OracleRig,
)

NARROW_MARGIN_M = 0.004
"""The 4 mm of slack the held opening carries over the part and the two tip thicknesses.

The held quantity is the GAP BETWEEN THE JAW INNER FACES, so each face sits
``width / 2 + TIP_THICKNESS + 2 mm`` from the tool axis -- 11.9 mm for the 2x4 brick's narrow
axis against the shipped 52.4 mm. ``grasp.OPEN_HALF_GAP`` names the same quantity halved,
which is why the two numbers are not directly comparable without the factor.
"""

JAW_BAND_M = 0.004
"""How far above the held opening the open/shut command flips.

The env's gripper channel is BINARY by contract (raw action >= 0 opens, < 0 closes) and the
jaw target ramps at the URDF's jaw velocity, about 3.3 mm per control step, so an opening
between the two ends has to be held by a bang-bang command and the jaws coast about one
control step of travel past every flip. Flipping one step of travel ABOVE the opening is what
keeps that coast from taking the jaws under the part's own width; the ramp is its own
hysteresis, so no further band is needed. What this rig emits stays a binary gripper command
the real arm accepts.
"""

JAW_OPEN_STEPS = 20
"""Control steps the descend holds at the pick depth while the jaws open.

The shipped descend converges the step after its leg is walked, which is one control step and
about 3 mm of jaw travel: without the hold the jaws would still be near the held opening when
CLOSE starts, and the probe would not have tested a bottom-open pick at all.
``grasp.OPEN_HALF_GAP`` is roughly 40 mm of travel from the held opening, about 13 steps. The
cap is what keeps a jaw opening INTO a tray wall -- the blockage under test -- from holding
the leg for its whole step budget.
"""

AXIS_TOLERANCE_RAD = math.radians(15.0)
WALL_DISTANCES_MM = (70, 60, 55, 50, 40, 20)
"""Part center to wall face, the rows of the inspection report's section 3.2 table."""
AXIS_HEADINGS = {"x": 0.0, "y": math.pi / 2}
"""World heading of the closing axis under test. ``y`` is the tray's long axis."""
PROBE_PART = "brick_2x4"
"""Section 3.2's part. ``--part`` swaps it, which is how section 3.3's flat parts are checked
against a clear wall: the bottom-open descend must not cost what those fixes bought."""
WALL_EPISODE_LENGTH_S = 120.0
"""Long enough for one cycle plus the shipped three-retry ladder."""

REACH_RADII_M = tuple(round(0.10 + 0.02 * i, 2) for i in range(13))
"""0.10 m to 0.34 m in 0.02 m steps, the radii of design section 6.1."""
REACH_AZIMUTHS_DEG = (-40, -20, 0, 20, 40)
REACH_CONTROL_RADIUS_M = 0.20
"""The control point, well inside ``layout.SCATTER_MAX_RADIUS``, read at every azimuth."""
REACH_EPISODE_LENGTH_S = 120.0

MATCHED_HALF_GAP_M = 0.0175
"""The landing half gap ``--matched_clearance`` gives the ranking, against the shipped 0.0524.

``core.grasp.jaw_clearance`` models the landing spots of a jaw that descends FULLY OPEN. The
bottom-open descend does not, so the teacher keeps rejecting as unpickable exactly the grasps
the new primitive can make, and a heap read taken that way measures the old model rather than
the new pick. This is the number the wall table MEASURED the jaws holding while the tips were
below the 20 mm wall top on the 2x4's narrow axis (`gap_below_wall_mm`, 17.5 mm), not the
11.9 mm the opening formula commands: the bang-bang hold sits about 5.6 mm above its floor and
the model should follow the jaws, not the set point.

One scalar stands in for a width-dependent quantity, and the direction of that approximation
matters for reading any yield gain. Against the formula it is the STRICTER of the two. Against
the real jaws it is exact at 15.8 mm, pessimistic for narrower parts and optimistic for wider
ones, so a part wider than 15.8 mm may be selected on a clearance its jaws would not really
have. `MAX_GRASP_WIDTH` is 45 mm, where the true held gap would be near 26.5 mm.
"""

HEAP_HOVER_JITTER = (0.035, 0.080)
"""The read's corrective-approach band, the MS3 Phase 4 t4 window.

Left at the argparse default of 0.0 the heap mode silently measures a configuration the corpus
does not use. That is not a small difference: heap 0 with the matched clearance model stows 15
at the read's flags and 18 without this band, and the first 67 of 150 was measured without it
against a baseline measured with it.
"""

HEAP_K = 50
HEAP_RUN_SEED = 20261000
HEAP_LAYOUT_SEED = 20261010
HEAP_EPISODE_DRAWS = True
"""The read's flags include ``--episode_draws``, and it is not cosmetic here.

The flag redraws bins, lights and cameras at the episode reset, which consumes the same
``_rng`` the heap drop then draws from, so leaving it off gives a DIFFERENT heap at the same
seeds: the parts are the same 50 and their poses after the settle differ by 100 mm on average
against the inspection's own first scan. With it on they match that scan to 0.007 mm, which is
the log's rounding. Comparing this mode's episode against the inspection's 15 of 50 is only
meaningful with it on. It has no "off" switch, because the mode exists to reproduce the read.
"""

HEAP_READ_FLAGS = (
    ("--heap_k", HEAP_K),
    ("--parts_split", "heldout"),
    ("--run_seed", HEAP_RUN_SEED),
    ("--layout_seed", HEAP_LAYOUT_SEED),
    ("--episode_draws", HEAP_EPISODE_DRAWS),
    ("--nearest_first_pick_order", True),
    ("--frame_jitter", True),
    ("--hover_jitter_min", HEAP_HOVER_JITTER[0]),
    ("--hover_jitter_max", HEAP_HOVER_JITTER[1]),
)
"""The read's flags, as DEFAULTS for the heap mode. A flag the caller passed always wins.

The heap mode's whole job is to reproduce the inspection's K = 50 read, and a default it does
not set is a configuration difference nobody sees. Every entry here was a hole someone fell
into: the seeds made three heaps into one, `episode_draws` made the arrangement a different
arrangement, and the jitter band and the pick order made the yield incomparable to the
baseline it was reported against.
"""


class BottomPickRig(OracleRig):
    """``OracleRig`` whose DESCEND holds a narrow opening, then opens at the pick depth.

    Only the descend leg's gripper command changes. The leg's geometry, every other phase and
    CLOSE itself are the shipped ones, so what this measures is the jaw opening and nothing
    else.
    """

    def __init__(self, *args, bottom_open: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # False runs the shipped descend through this same harness and instrumentation: the
        # control the wall table is read against.
        self.bottom_open = bottom_open
        self._open_hold = [0] * self.num_envs
        self._driven: list[oracle.Phase | None] = [None] * self.num_envs
        self.jaw_log: list[dict] = []
        self.phase_log: list[dict] = []
        self.residuals: list[dict[str, float]] = [{} for _ in range(self.num_envs)]
        self.last_pinch: dict | None = None

    # -- the descend --------------------------------------------------------

    def narrow_gap(self, i: int) -> float:
        """Half the opening the descend holds: the part width, two tip thicknesses and 4 mm."""
        width = float(self.oracles[i].grasp_width)
        return (width + 2.0 * grasp.TIP_THICKNESS + NARROW_MARGIN_M) / 2.0

    def jaw_gap(self, i: int) -> float:
        """The narrower jaw's inner face, m from the tool axis (``grasp.OPEN_HALF_GAP`` open)."""
        return min(self.robot.data.joint_pos[i, self.finger_ids].tolist())

    def actions(self):
        action = super().actions()
        for i in range(self.num_envs):
            goal = self.goals[i]
            if goal is None or goal.phase != oracle.Phase.DESCEND:
                self._open_hold[i] = 0
                continue
            # `super().actions()` has already walked the plan, so an empty one means the leg
            # is done and the tips are at the pick depth: the shipped fully-open command stands.
            descending = bool(self.plans[i])
            target, gap = self.narrow_gap(i), self.jaw_gap(i)
            if descending and self.bottom_open:
                open_jaw = gap <= target + JAW_BAND_M
                action[i, 6] = GRIPPER_OPEN_ACTION if open_jaw else -GRIPPER_OPEN_ACTION
            self.jaw_log.append(
                {
                    "env": i,
                    "descending": descending,
                    "command": float(action[i, 6]),
                    "target": round(target, 4),
                    "gap": round(gap, 4),
                    "pick_z": round(float(self.pick_point(i)[2]), 4),
                }
            )
        self._driven = [None if goal is None else goal.phase for goal in self.goals]
        return action

    def observe(self, dones):
        """Sample the joint tracking residual before the base class advances the phases.

        This is a DRIVEN read and nothing else: ``self._planned[i]`` is the joint target this
        step commanded, and ``robot.data.joint_pos`` is where the servo actually ended up after
        ``env.step``. The solver's own convergence report is not consulted -- the whole point of
        ``layout.SCATTER_MAX_RADIUS`` is that the solver reports a clean solve at poses the arm
        never arrives at. An env that finished this step is skipped, because ``env.step`` has
        already reset it and its joints describe the next episode.
        """
        measured = self.robot.data.joint_pos[:, self.arm_ids]
        for i, phase in enumerate(self._driven):
            if phase is None or bool(dones[i]) or self.abandoned[i]:
                continue
            target = self._planned[i].to(device=measured.device, dtype=measured.dtype)
            gap = float((target - measured[i]).abs().max())
            self.residuals[i][phase.name] = max(self.residuals[i].get(phase.name, 0.0), gap)
        return super().observe(dones)

    # -- instrumentation ----------------------------------------------------

    def pinch_evidence(self, *args, **kwargs) -> dict:
        self.last_pinch = super().pinch_evidence(*args, **kwargs)
        return self.last_pinch

    def _phase_result(self, i: int, goal):
        result = super()._phase_result(i, goal)
        if (
            result is not None
            and self.bottom_open
            and goal.phase == oracle.Phase.DESCEND
            and not self.plans[i]
        ):
            # The descend leg now ends when the tips are at the pick depth AND the jaws have
            # opened; opening them there is the half of the leg this probe replaces.
            self._open_hold[i] += 1
            if (
                self._open_hold[i] <= JAW_OPEN_STEPS
                and self.jaw_gap(i) < grasp.OPEN_HALF_GAP - JAW_BAND_M
            ):
                return None
        if result is not None:
            self.phase_log.append(
                {
                    "env": i,
                    "phase": goal.phase.name,
                    "outcome": result.outcome,
                    "pick_z": round(float(self.pick_point(i)[2]), 4),
                    "gap": round(self.jaw_gap(i), 4),
                    "pinch": self.last_pinch if goal.phase == oracle.Phase.LIFT else None,
                }
            )
        return result


class AxisOracle(HeapOracle):
    """``HeapOracle`` pinned to one closing axis, whatever its clearance.

    The wall table is a FORCED read. Left to itself the teacher picks the axis that is not
    pointing at the wall, and a grasp whose tips land past a wall never clears
    ``grasp.CLEARANCE_MIN`` at all, so neither the shipped selection nor the forced-pick
    fallback would ever attempt the row under test.
    """

    def __init__(self, *args, axis: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    def _ranked(self, scene, mask):
        return [
            g
            for g in super()._ranked(scene, mask)
            if abs(math.remainder(g.heading - self.axis, math.pi)) < AXIS_TOLERANCE_RAD
        ]

    def _next_brick(self, scene) -> int | None:
        if self.cycles >= self.cycle_budget:
            return None
        ranked = self._ranked(scene, self._candidates(scene))
        if not ranked:
            return None
        chosen = ranked[0]
        self.grasp = chosen
        self.grasp_width = chosen.width
        self.grasp_min_size = min(self.parts[chosen.part].size)
        self.cycles += 1
        self.cycle_parts.append(chosen.part)
        return chosen.part


class ReachOracle(oracle.EnvOracle):
    """``EnvOracle`` with the top-down x limit lifted, for the driven reach band.

    ``layout.TOP_DOWN_X_LIMIT`` is a bin-fouling limit, not a reach limit (design section 6.1),
    and the shipped candidate filter drops every brick past it before a descend is ever driven.
    The band is what this mode measures, so that filter cannot be part of the measurement.
    """

    def _next_brick(self, scene) -> int | None:
        done = self._stowed(scene) | self._sorted
        alive = [b for b in range(self.num_bricks) if b not in done]
        return alive[0] if alive else None


def reach_grid(radii_m=None, azimuths_deg=None) -> list[tuple[float, int]]:
    """The (radius m, azimuth deg) points, azimuth-major.

    The defaults are design section 6.1's own grid. ``--radii_mm`` and ``--azimuths_deg``
    override them, which is how the near side is read finer and lower than 6.1 asks for
    without the far read changing.
    """
    radii = REACH_RADII_M if radii_m is None else radii_m
    azimuths = REACH_AZIMUTHS_DEG if azimuths_deg is None else azimuths_deg
    return [(radius, azimuth) for azimuth in azimuths for radius in radii]


def reach_outcome(phases: list[dict], succeeded: bool) -> str:
    """``stowed``, ``lost`` or ``no_solution`` for one grid point.

    A leg the solver could not plan is reported apart from a pick that was driven and failed,
    so the two causes never blur: a point the arm cannot even be ASKED to reach is not evidence
    about tracking. ``succeeded`` wins over an earlier unplanned leg, because a point whose
    retry ladder eventually stowed the brick was reachable after all.
    """
    if succeeded:
        return "stowed"
    approach = ("HOVER", "DESCEND")
    if any(e["outcome"] == "unplanned" for e in phases if e["phase"] in approach):
        return "no_solution"
    return "lost"


def wall_pose(
    axis: str, distance_m: float, half_height: float
) -> tuple[float, float, float, float]:
    """(x, y, z, yaw) putting the part's center ``distance_m`` from a tray wall's inner face.

    The part is laid so the axis under test is its narrow 15.8 mm one on both runs, so the two
    axes differ in the wall they face and in nothing else.
    """
    (_x0, x1), (_y0, y1) = layout.tray_bounds()
    z = half_height + 0.001
    if axis == "y":
        return layout.TRAY_CENTER[0], y1 - distance_m, z, 0.0
    return x1 - distance_m, layout.TRAY_CENTER[1], z, math.pi / 2


def place(env, pose) -> None:
    """Drop the part at ``pose``, settle it, and read it back into the env's surface."""
    m, d, h = env.models[0], env.datas[0], env.handles
    x, y, z, yaw = pose
    adr = h.brick_qpos[0]
    d.qpos[adr : adr + 3] = (x, y, z)
    d.qpos[adr + 3 : adr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    d.qvel[h.brick_dof[0] : h.brick_dof[0] + 6] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=120)
    env._refresh([0], images=False)


def wall_cell(args, axis: str, distance_mm: int) -> dict:
    """One row of the section 3.2 table on one closing axis."""
    draw, bl, noise = build_run(args)
    (part,) = _parts.by_name([args.part])
    env = MjLegoSortEnv(
        1,
        draw,
        num_bricks=1,
        parts=[part],
        cameras=False,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        episode_length_s=args.episode_length_s or WALL_EPISODE_LENGTH_S,
    )
    env.reset(seed=args.run_seed)
    rng = random.Random(args.run_seed)
    induction = make_induction(args)

    def factory(i, layout_i, episode_draw):
        return AxisOracle(
            layout_i,
            env.parts,
            int(env.destination_bins[i]),
            episode_draw,
            induction=induction,
            rng=rng,
            cycle_budget=1,
            axis=AXIS_HEADINGS[axis],
        )

    rig = BottomPickRig(
        env,
        bl,
        noise,
        rng,
        induction=induction,
        transport_speed=args.transport_speed,
        oracle_factory=factory,
        bottom_open=not args.shipped,
    )
    place(env, wall_pose(axis, distance_mm / 1000.0, part.size[2] / 2.0))
    steps = 0
    while steps < int(env.max_episode_length):
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
        if rig.oracles[0].phase == oracle.Phase.DONE:
            break
    stats = rig.episode_stats(0)
    descends = [e for e in rig.phase_log if e["phase"] == "DESCEND"]
    lifts = [e for e in rig.phase_log if e["phase"] == "LIFT"]
    descending = [e for e in rig.jaw_log if e["descending"]]
    # The opening that has to clear the wall is the WIDEST one the jaws hold while the tips are
    # below the wall top, not the narrowest one they reach on the way down.
    below = [e["gap"] for e in descending if e["pick_z"] <= layout.TRAY_WALL_HEIGHT]
    gap_mm = None if not below else round(max(below) * 1000.0, 1)
    env.close()
    pinch = lifts[-1]["pinch"] if lifts else None
    return {
        "part": part.name,
        "axis": axis,
        "distance_mm": distance_mm,
        # Where the jaw inner faces sit as they pass the wall: the shipped descend's fixed
        # 52.4 mm, and this run's. Positive is inside the wall face, as section 3.2 reads it.
        "shipped_tip_mm": round(distance_mm - grasp.OPEN_HALF_GAP * 1000.0, 1),
        "tip_mm": None if gap_mm is None else round(distance_mm - gap_mm, 1),
        "gap_below_wall_mm": gap_mm,
        "leg_end_z_mm": None if not descending else round(descending[-1]["pick_z"] * 1000, 1),
        "descend": None if not descends else descends[-1]["outcome"],
        "phase_end_z_mm": None if not descends else round(descends[-1]["pick_z"] * 1000, 1),
        "lift_stall_mm": None if pinch is None else round(pinch["stall"] * 1000, 1),
        "lift_force_n": None if pinch is None else round(min(pinch["left"], pinch["right"]), 2),
        "retries": stats["retries"],
        "cycles": stats.get("cycles", 0),
        "pick": "stowed" if stats["succeeded"] else "lost",
        "steps": steps,
    }


def reach_cell(args, radius: float, azimuth_deg: int) -> dict:
    """One driven grid point: the shipped descend and pick on a lone brick, no tray.

    The env is the six-brick task with ONE brick, which is the scene that has no tray at all
    (``mj/env.py`` builds the tray only for a heap). So the arm reaches a bare desk, and a wall
    can neither block a descend nor foul a carry.
    """
    draw, bl, noise = build_run(args)
    env = MjLegoSortEnv(
        1,
        draw,
        num_bricks=1,
        cameras=False,
        start_pose_offset_rad=0.05
        if args.start_pose_offset_rad is None
        else args.start_pose_offset_rad,
        episode_length_s=args.episode_length_s or REACH_EPISODE_LENGTH_S,
    )
    env.reset(seed=args.run_seed)
    rng = random.Random(args.run_seed)
    induction = make_induction(args)
    rig = BottomPickRig(
        env,
        bl,
        noise,
        rng,
        induction=induction,
        transport_speed=args.transport_speed,
        oracle_factory=lambda i, layout_i, episode_draw: ReachOracle(
            layout_i, 1, episode_draw, induction=induction, rng=rng
        ),
        # The band is measured against the SHIPPED descend; the bottom-open one is experiment 2.
        bottom_open=False,
    )
    angle = math.radians(azimuth_deg)
    x, y = radius * math.cos(angle), radius * math.sin(angle)
    place(env, (x, y, layout.BRICK_SIZE[2] / 2.0 + 0.001, 0.0))
    # Where the brick ACTUALLY settled. Close to the base the arm's own body occupies the
    # desk, so a requested point can be one the scene refuses to hold; a row whose brick moved
    # is not a reading about the arm's reach and has to say so rather than report a number.
    settled = rig._brick_pos(0, 0).tolist()
    settled_radius = math.hypot(settled[0], settled[1])
    steps = 0
    while steps < int(env.max_episode_length):
        out = rig.step()
        steps += 1
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
        if rig.oracles[0].phase in (oracle.Phase.DONE, oracle.Phase.ABANDON):
            break
    stats = rig.episode_stats(0)
    residuals = dict(rig.residuals[0])
    descends = [e for e in rig.phase_log if e["phase"] == "DESCEND"]
    env.close()
    descend_resid = residuals.get("DESCEND")
    lift_resid = residuals.get("LIFT")
    driven = [r for r in (descend_resid, lift_resid) if r is not None]
    return {
        "radius_m": radius,
        "azimuth_deg": azimuth_deg,
        "x": round(x, 4),
        "y": round(y, 4),
        "settled_radius_m": round(settled_radius, 4),
        "settled_z_m": round(settled[2], 4),
        "placement_drift_mm": round((settled_radius - radius) * 1000.0, 1),
        "outcome": reach_outcome(rig.phase_log, stats["succeeded"]),
        # Radians, commanded joint target against the measured joint position after env.step.
        "descend_resid_rad": None if descend_resid is None else round(descend_resid, 4),
        "lift_resid_rad": None if lift_resid is None else round(lift_resid, 4),
        "resid_rad": None if not driven else round(max(driven), 4),
        "descend": None if not descends else descends[-1]["outcome"],
        "unplanned_legs": stats["unplanned_legs"],
        "retries": stats["retries"],
        "steps": steps,
    }


def reach_band(args) -> dict:
    radii, azimuths = args.radii_m, args.azimuths_deg
    rows = [reach_cell(args, radius, azimuth) for radius, azimuth in reach_grid(radii, azimuths)]
    path = os.path.join(args.out_dir, "reach_band.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    letter = {"stowed": "S", "lost": "L", "no_solution": "N"}
    at = {(r["azimuth_deg"], r["radius_m"]): r for r in rows}
    return {
        "mode": "reach_band",
        "table": path,
        "radii_m": list(radii),
        "azimuths_deg": list(azimuths),
        "rows": len(rows),
        "stowed": sum(1 for r in rows if r["outcome"] == "stowed"),
        "no_solution": sum(1 for r in rows if r["outcome"] == "no_solution"),
        # Rows whose brick did not stay where it was put: a scene refusal, not a reach read.
        "displaced": sum(1 for r in rows if abs(r["placement_drift_mm"]) > 2.0),
        # One string per azimuth, one letter per radius, in the order of `radii_m`.
        "grid": {
            str(a): "".join(letter[r["outcome"]] for r in rows if r["azimuth_deg"] == a)
            for a in azimuths
        },
        # None when the control radius is not one of the radii this run drove.
        "control_resid_rad": {
            str(a): (
                at[(a, REACH_CONTROL_RADIUS_M)]["resid_rad"]
                if (a, REACH_CONTROL_RADIUS_M) in at
                else None
            )
            for a in azimuths
        },
        "max_stowed_radius_m": {
            str(a): max(
                (r["radius_m"] for r in rows if r["azimuth_deg"] == a and r["outcome"] == "stowed"),
                default=None,
            )
            for a in azimuths
        },
        "min_stowed_radius_m": {
            str(a): min(
                (r["radius_m"] for r in rows if r["azimuth_deg"] == a and r["outcome"] == "stowed"),
                default=None,
            )
            for a in azimuths
        },
    }


def wall_table(args) -> dict:
    rows = [
        wall_cell(args, axis, distance_mm)
        for axis in args.axes.split(",")
        for distance_mm in args.distances_mm
    ]
    name = f"wall_table_{args.part}{'_shipped' if args.shipped else ''}.json"
    path = os.path.join(args.out_dir, name)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return {
        "mode": "wall_table",
        "descend": "shipped" if args.shipped else "bottom_open",
        "table": path,
        "rows": len(rows),
        "stowed": sum(1 for r in rows if r["pick"] == "stowed"),
        "outcomes": [[r["axis"], r["distance_mm"], r["pick"], r["descend"]] for r in rows],
    }


def resolved_config(args) -> dict:
    """Everything that decides what the heap episode measured, as the run actually resolved it.

    This exists because a log that cannot say what it ran costs a day. Every number in it was
    once left to an argparse default nobody noticed.
    """
    return {
        "run_seed": args.run_seed,
        "layout_seed": args.layout_seed,
        "parts_seed": args.parts_seed,
        "parts_split": args.parts_split,
        "heap_k": args.heap_k,
        "cycle_budget": args.cycle_budget,
        "episode_draws": bool(args.episode_draws),
        "frame_jitter": bool(args.frame_jitter),
        "nearest_first_pick_order": bool(args.nearest_first_pick_order),
        "randomize_pick_order": bool(args.randomize_pick_order),
        "hover_jitter": [args.hover_jitter_min, args.hover_jitter_max],
        "noise_scale": args.noise_scale,
        "transport_speed": args.transport_speed,
        "matched_clearance": bool(args.matched_clearance),
        "matched_half_gap_m": MATCHED_HALF_GAP_M if args.matched_clearance else None,
        "read_flags": not args.no_read_flags,
    }


def stamp_config(log: str, config: dict) -> None:
    """Put the resolved configuration into the log's ``end`` event, which is its last line.

    Written after ``mj_inspect_heap.run`` rather than inside it, so the inspection tool and the
    log format it owns are untouched and ``scripts/heap_inspect_read.py`` keeps reading both.
    """
    with open(log) as f:
        lines = f.read().splitlines()
    end = json.loads(lines[-1])
    end["config"] = config
    lines[-1] = json.dumps(end)
    with open(log, "w") as f:
        f.writelines(line + "\n" for line in lines)


@contextlib.contextmanager
def matched_clearance():
    """Give ``core.grasp``'s ranking the landing half gap the bottom-open descend holds.

    WHAT THIS REACHES, because a patched constant that leaked would make the run measure more
    than the model. ``OPEN_HALF_GAP`` appears in exactly three expressions, all inside
    ``core.grasp``: the landing spots in ``landing_footprints``, and the two stroke bands in
    ``jaw_clearance``. ``landing_footprints`` and ``sweep_footprints`` are called from nowhere
    but ``jaw_clearance``, and ``jaw_clearance`` from nowhere but ``rank_grasps``. No module
    binds the name by ``from ... import``, so there is no second copy to miss. The push
    planner, the footprint model, ``tray_wall_obstacles``, the oracle's phases and the rig's
    own descend geometry never read it. So this reaches the ranking and stops there.

    ``sweep_footprints``'s ``outer`` default was bound to the shipped value when the module was
    imported, so setting the module global alone would leave the swept count measuring the old
    radius; ``__defaults__`` is patched with it and restored with it.
    """
    saved_gap = grasp.OPEN_HALF_GAP
    saved_defaults = grasp.sweep_footprints.__defaults__
    grasp.OPEN_HALF_GAP = MATCHED_HALF_GAP_M
    grasp.sweep_footprints.__defaults__ = (MATCHED_HALF_GAP_M,)
    try:
        yield MATCHED_HALF_GAP_M
    finally:
        grasp.OPEN_HALF_GAP = saved_gap
        grasp.sweep_footprints.__defaults__ = saved_defaults


def heap(args) -> dict:
    """One K = 50 held-out heap through the inspection tool's driver loop and its log."""
    path = Path(__file__).resolve().with_name("mj_inspect_heap.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OracleRig = BottomPickRig
    patch = matched_clearance() if args.matched_clearance else contextlib.nullcontext(None)
    with patch as half_gap:
        summary = module.run(args)
    config = resolved_config(args)
    stamp_config(summary["events"], config)
    return {"mode": "heap", "matched_half_gap_m": half_gap, "config": config, **summary}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_batch_arguments(parser)
    parser.set_defaults(num_envs=1)
    parser.add_argument("--mode", choices=("wall_table", "reach_band", "heap"), required=True)
    parser.add_argument(
        "--axes", type=str, default="y,x", help="closing axes the wall table runs, in order"
    )
    parser.add_argument(
        "--distances_mm",
        type=str,
        default=",".join(str(d) for d in WALL_DISTANCES_MM),
        help="part center to wall face, the rows of the section 3.2 table",
    )
    parser.add_argument(
        "--radii_mm",
        type=str,
        default=",".join(str(round(r * 1000)) for r in REACH_RADII_M),
        help="reach band only: comma-separated radii in mm; the default is design section "
        "6.1's own grid",
    )
    parser.add_argument(
        "--azimuths_deg",
        type=str,
        default=",".join(str(a) for a in REACH_AZIMUTHS_DEG),
        help="reach band only: comma-separated azimuths in degrees",
    )
    parser.add_argument(
        "--part",
        type=str,
        default=PROBE_PART,
        help="catalog name of the lone part the wall table picks, e.g. plate_2x8",
    )
    parser.add_argument(
        "--no_read_flags",
        action="store_true",
        help="heap mode only: skip the read's default flags entirely and leave every argparse "
        "default alone, so the heap's own flags must then be given explicitly. Needed to "
        "reproduce the archived runs that were measured WITHOUT the hover jitter band: "
        "~/data/maker-arm-ms6/d/probe/iso_nojitter_* and minimal_matched_0.",
    )
    parser.add_argument(
        "--matched_clearance",
        action="store_true",
        help="heap mode only: give the grasp ranking the bottom-open descend's own landing "
        "half gap instead of the fully-open one, so the teacher stops rejecting the grasps "
        "the new primitive can make. Off by default.",
    )
    parser.add_argument(
        "--shipped",
        action="store_true",
        help="wall table only: run the SHIPPED fully-open descend through the same harness, "
        "the control the bottom-open rows are read against",
    )
    parser.add_argument("--video_stride", type=int, default=8, help="control steps per frame")
    parser.add_argument("--video_tail_s", type=float, default=5.0)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args(argv)
    if args.num_envs != 1:
        parser.error("--num_envs must be 1: one part or one heap, one log")
    if any(axis not in AXIS_HEADINGS for axis in args.axes.split(",")):
        parser.error(f"--axes must be drawn from {sorted(AXIS_HEADINGS)}")
    args.distances_mm = [int(d) for d in args.distances_mm.split(",")]
    # Radii are taken in mm so a 10 mm step is an exact integer on the wire; the metre value
    # is what every lookup keys on, so it is rounded once here and never recomputed.
    args.radii_m = [round(int(r) / 1000.0, 4) for r in args.radii_mm.split(",")]
    args.azimuths_deg = [int(a) for a in args.azimuths_deg.split(",")]
    if not args.radii_m or not args.azimuths_deg:
        parser.error("--radii_mm and --azimuths_deg must each name at least one value")
    if args.mode == "heap" and not args.no_read_flags:
        # The pre-registered read (Task 3 brief, step 3) is the DEFAULT heap, not a pin. These
        # were once overwritten unconditionally, so three runs at three seed pairs were three
        # copies of heap 0, down to the push_parts list. Accepting a flag and then discarding
        # it is worse than either refusing it or honoring it, so a flag the caller passed wins.
        given = list(sys.argv[1:] if argv is None else argv)
        for flag, value in HEAP_READ_FLAGS:
            if not any(a == flag or a.startswith(flag + "=") for a in given):
                setattr(args, flag[2:], value)
    if args.mode == "heap" and args.heap_k <= 0:
        parser.error(
            "--heap_k must be positive for the heap mode; --no_read_flags leaves every "
            "argparse default alone, so it must be given with the heap's own flags"
        )
    return args


def main(argv=None) -> int:
    try:
        args = parse_args(argv)
        os.makedirs(args.out_dir, exist_ok=True)
        summary = {"heap": heap, "reach_band": reach_band, "wall_table": wall_table}[args.mode](
            args
        )
    except SystemExit as e:  # argparse
        print("PROBE_FAIL " + json.dumps({"error": f"argparse exit {e.code}"}), flush=True)
        return 2
    except Exception as e:  # noqa: BLE001 -- any failure must become PROBE_FAIL.
        traceback.print_exc()
        print("PROBE_FAIL " + json.dumps({"error": repr(e)}), flush=True)
        return 1
    print("PROBE_OK " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

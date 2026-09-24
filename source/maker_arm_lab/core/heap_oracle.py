"""The heap teacher: ``EnvOracle`` over mixed parts dumped on the desk (MS6 phase A design 7).

What changes against the six-brick oracle, and nothing else:

* the target is the highest part with a clear grasp (``core.grasp``), and the jaws square to
  that grasp's closing axis instead of the brick's long axis;
* every part goes to one destination bin drawn per episode, whatever its color (design H5);
* a 5 cm cell holds a failure count; at three failures the cell is skipped (design 7.2);
* when no grasp clears, the top-ranked grasp is taken anyway as a forced pick (fix F2). The
  push that used to come first, a sweep through the densest cell with the jaws closed
  (design H8), is off: ``MAX_PUSHES_IN_A_ROW`` is 0 and its docstring holds the read;
* a part out of the arm's reach band is counted but never a target (fix F1). There is no
  dish since 2026-09-18, so nothing brings such a part back;
* a LIFT that missed its target but holds another part carries that part off instead of
  dropping it back into the heap (``adopt_held_part``, ``accepts_any_held_part``);
* a part whose best grasp grips less than ``TIP_MIN_GRIP_HEIGHT`` of face is tipped onto an
  edge first, with a short stroke into it, and picked on a later scan (MS6 rework section 7);
* the episode ends DONE, never ABANDON, at the cycle budget or with no candidate left;
* the next target is chosen the moment RELEASE ends, so its row and marker show through
  RETREAT, and HOVER adopts that choice (``preselects_next_target``, 2026-09-22);
* with desk boxes (``boxes``, MS7a design, desk boxes) every part has its own box, the
  release is at the pick tilt over that box's centre with no RAISE, and a part inside any box
  is never a target;
* in the MS7a tray, when no grasp ranks and a candidate lies within ``grasp.RAKE_BAND_M`` of
  a wall, the closed jaws rake it off the wall through the push phases (``_plan_rake``).

The phases, the correctives, the induction and the retry ladder are the base class's.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from maker_arm_lab.core import checker, desk_boxes, grasp, layout, oracle
from maker_arm_lab.core.parts import Part, part_class

CYCLE_BUDGET = 50
CELL_FAILURE_CAP = 3
MAX_PUSHES_IN_A_ROW = 0
"""Pushes with nothing clear before the top-ranked grasp is taken anyway (a forced pick).

0 means the teacher never pushes: every blocked scan is a forced pick at once. The bare-desk
read of 2026-09-18 (~/data/maker-arm-ms6/i/rA, 150 parts) decided it. Pushes took 34 % of
every episode's control steps, 118 pushes against 119 picks, and all three episodes ran out
of time with parts left. The forced picks the pushes delayed stowed 20 of 21. With no dish a
landing spot is blocked only by a neighbor, and the closing stroke sweeps a neighbor toward
the part (``core.grasp``), so the model's refusal is stricter than the physics. 3 was the
value the dish ran; the push machinery stays and this constant turns it back on.
"""
PUSH_BUDGET = CYCLE_BUDGET
MAX_RAKES = 20
"""Rakes one episode may spend (MS7a design, the rake). # INITIAL"""
RAKE_PART_CAP = 2
"""Rakes one part may take before it is left where it lies. # INITIAL"""
CANDIDATE_MAX_Z = 0.10
"""A part above this height sits in a bin or on the shelf, never on the heap."""
PUSH_HOVER_Z = 0.08
PUSH_Z = layout.BRICK_SIZE[2] / 2
"""Pick-point height of the sweep: the plate tips skim the desk at the brick pick offset."""
PUSH_HALF_HEIGHT = layout.BRICK_SIZE[2] / 2
PUSH_SWEEP_STEPS = 150
TIP_MIN_GRIP_HEIGHT = 0.0
"""A part whose best grasp grips less than this (m) of face is tipped up, not picked.

0.0 means the tip-up is DISABLED: ``2.0 * chosen.half_height < TIP_MIN_GRIP_HEIGHT`` and
``half_height`` is never negative, so the trigger never fires and no part is ever diverted
into a tip (MS6 rework section 7). It ships off because the pre-registered ladder read it as
a regression: with the trigger live the teacher stowed 16 of 150 parts against 49 of 150 for
the rung below, and the Wilson interval did not overlap. Two measured causes: the tip fired
54 times across three heaps and succeeded once, and a failed tip calls ``_mark_failure``,
which closes a 50 mm cell and makes ``_candidates`` drop every part in that cell including
bricks, so the tip became the only uncapped source of cell closures in the teacher.

0.005 is the value the ladder measured: a plate is 3.2 mm thick and a brick is 9.6 mm tall,
so the threshold takes the flat parts and leaves the bricks alone. Heap 0 of 2026-09-17:
plates failed 13 of their 24 lifts.

Enabling the tip means setting this back to 0.005. Before that, the tip needs a different
descend geometry (the failed-lift cause above) and must stop feeding the cell failure memory
(the closed-cell cause above) -- both machinery, not this constant.
"""
TIP_STROKE_M = 0.025
TIP_HOVER_Z = PUSH_HOVER_Z
TIP_Z = 0.004
"""Pick-point height of the tip stroke: the closed tips skim under a lying plate's edge."""
TIP_SWEEP_STEPS = PUSH_SWEEP_STEPS
PUSH_HEADING_MIN_HALF_ARC = math.pi / 2
PRESELECT_MOVE_M = 0.01
"""A pre-selected part that moved further than this (m) before HOVER is re-ranked live."""
"""Narrowest half-arc a push heading is drawn from, at the drop disc's edge (``_push_heading``)."""
# Design H6: the bin footprint, and twice the interior height so a pile over the rim counts.
STOW_HALF_EXTENTS = (
    layout.HEAP_BIN_INTERIOR[0] / 2,
    layout.HEAP_BIN_INTERIOR[1] / 2,
    layout.HEAP_BIN_INTERIOR[2] * 2,
)


def target_context(pos, chosen: grasp.Grasp | None) -> list[float]:
    """The target channel row (``mj.context_names.CONTEXT_TARGET_NAMES``): the part centre
    ``pos`` and the grasp the teacher ranks on it, zeros when there is none. The rig and the
    heap loop both build the row here, so a recorded frame and a served one agree."""
    if chosen is None:
        return [0.0] * 7
    x, y, z = (float(v) for v in pos[:3])
    return [
        x,
        y,
        z,
        math.cos(chosen.heading),
        math.sin(chosen.heading),
        float(chosen.width),
        float(chosen.half_height),
    ]


@dataclass(frozen=True)
class Push:
    start: tuple[float, float]
    end: tuple[float, float]
    part: int | None = None  # the raked part; None is the densest-cell push
    heading: float | None = None  # a rake's sweep direction
    jaw: float | None = None  # the closing axis the legs square to; None holds the jaws


@dataclass(frozen=True)
class Tip:
    start: tuple[float, float]
    end: tuple[float, float]
    part: int


class HeapOracle(oracle.EnvOracle):
    def __init__(
        self,
        bl: layout.BinLayout,
        parts: Sequence[Part],
        destination: int,
        draw: oracle.EpisodeDraw,
        induction: oracle.FailureInduction | None = None,
        rng: random.Random | None = None,
        cycle_budget: int = CYCLE_BUDGET,
        boxes: Sequence[Sequence[float]] = (),
        tray: bool = False,
        tray_lean_deg: float = 0.0,
    ):
        """``destination`` is the MS6 bin index, or with ``boxes`` (one (x, y, yaw) per desk
        box) the box index of every part, (K,), or one index for them all."""
        super().__init__(bl, len(parts), draw, induction=induction, rng=rng)
        # MS7a design, the tray: the heap lies inside four walls that ``core.grasp`` ranks
        # against (``grasp.wall_allows``). Off, the heap lies on the bare desk as in MS6.
        self.walls = grasp.tray_walls(tray_lean_deg) if tray else ()
        self.parts = tuple(parts)
        self.boxes = tuple(tuple(float(v) for v in box) for box in boxes)
        if self.boxes:
            dest = torch.as_tensor(destination).long()
            self.part_boxes = (
                dest.expand(len(self.parts)).tolist() if dest.dim() == 0 else dest.tolist()
            )
            self.destination = None
            self.release_tilt = oracle.PICK_TILT_RAD
            self.retreat_point = desk_boxes.DESK_BOX_RETREAT_POINT
            self.raises = False
            self.carry_half_turn = True
        else:
            self.part_boxes = None
            self.destination = int(destination)
        self.sizes = torch.tensor([p.size for p in self.parts], dtype=torch.float32)
        self.cycle_budget = cycle_budget
        self.grasp: grasp.Grasp | None = None
        self.grasp_min_size = min(layout.BRICK_SIZE)
        # MS6 rung 1: `core.grasp` ranks on a landing gap that follows the grasp width, so
        # the jaws have to descend at that opening for the model to describe the arm.
        self.matched_jaw_opening = True
        # Batch picking (owner ruling, 2026-09-18): one bin takes every part, so a LIFT that
        # came up holding a NON-target part has still picked something worth carrying. On the
        # round-dish heaps of 2026-09-17 that was 16 of 154 LIFT reports, 10.4 %, each gripping
        # a non-target part 100-144 mm above the desk, and REOPEN dropped every one of them
        # back into the heap.
        self.accepts_any_held_part = True
        # The protocol lever of 2026-09-22: the next target is chosen when RELEASE ends and
        # held here, apart from the retreating cycle's ``target_brick``, ``grasp`` and
        # ``_brick_start``, until HOVER adopts it (``_preselect``, ``_next_brick``).
        self.preselects_next_target = True
        self.next_brick: int | None = None
        self.next_brick_start: torch.Tensor | None = None
        self.next_grasp: grasp.Grasp | None = None
        self._next_forced = False
        self.push: Push | None = None
        self.tip: Tip | None = None
        self.cycles = 0
        self.pushes = 0
        self.rakes = 0
        self.raked: dict[int, int] = {}
        self.jams = 0
        self.push_takeover = False  # set by aim_push: this oracle runs one push, then idles
        self.tips = 0
        self.tips_ok = 0
        self.forced_picks = 0
        self.held_swaps = 0
        self._give_up = False
        self.pushes_in_a_row = 0
        self.failures: dict = {}
        # "cell": failures count per 5 cm cell (design 7.2), the teacher's rule. "part": per
        # part, so a failing policy excludes the part it keeps missing and not its neighbours
        # (the heap loop's --part_failures switch, MS7a).
        self.failure_scope = "cell"
        # The rake's caps and its start rule; the heap loop's --rake_cap and --rake_crowd
        # raise them on its selectors (MS7a rake prototype), the teacher keeps these.
        self.rake_part_cap = RAKE_PART_CAP
        self.max_rakes = MAX_RAKES
        self.rake_crowd = 0.0
        self.cycle_parts: list[int] = []
        self.out_of_reach = 0
        self._push_rng = rng if rng is not None else random.Random(0)

    # -- grading against the destination -----------------------------------

    def _box_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(1,M,4) grader boxes and their half extents, as ``stowed.env_desk_boxes`` builds."""
        z, half = desk_boxes.stow_region(len(self.boxes))
        boxes = torch.tensor([[(x, y, z, yaw) for x, y, yaw in self.boxes]])
        return boxes, torch.tensor(half)

    def _stowed(self, scene: oracle.SceneSnapshot) -> set[int]:
        if self.boxes:
            boxes, half = self._box_tensor()
            ok = checker.parts_at_rest_in_own_boxes(
                scene.brick_pos.unsqueeze(0),
                scene.brick_lin_vel.unsqueeze(0),
                boxes,
                half,
                torch.tensor([self.part_boxes]),
            )[0]
            return {i for i in range(self.num_bricks) if bool(ok[i])}
        # The MS6 z-centre fix: heap mode has one physical bin, so grading reads
        # layout.heap_bin_center directly instead of the six-brick per-color array.
        ids = torch.zeros(self.num_bricks, dtype=torch.long)
        ok = checker.bricks_at_rest_in_bins(
            scene.brick_pos.unsqueeze(0),
            scene.brick_lin_vel.unsqueeze(0),
            torch.tensor(layout.heap_bin_center(self.bl)).unsqueeze(0),
            torch.tensor(STOW_HALF_EXTENTS),
            ids,
        )[0]
        return {i for i in range(self.num_bricks) if bool(ok[i])}

    def _release_target(self, brick: int) -> torch.Tensor:
        if self.boxes:
            # Over the centre of the part's own box: the level jaws drop it straight down.
            x, y, _ = self.boxes[self.part_boxes[brick]]
            return torch.tensor([x, y, desk_boxes.DESK_BOX_RELEASE_Z])
        return torch.tensor(
            [
                self.bl.bin_x - oracle.RELEASE_X_SETBACK,
                self.bl.bin_ys[self.destination],
                oracle.RELEASE_Z,
            ]
        )

    def _no_target_phase(self, scene: oracle.SceneSnapshot) -> oracle.Phase:
        if self.tip is not None:
            return oracle.Phase.TIP_HOVER
        return oracle.Phase.PUSH_HOVER if self.push is not None else oracle.Phase.DONE

    def standing(self, scene: oracle.SceneSnapshot, part: int, tilt_rad: float = 0.0) -> bool:
        """Is the part's shortest axis within the closing band, so its wide face grips?

        `core.grasp` counts a body axis as a closing axis only within AXIS_MAX_TILT_RAD of the
        jaw plane (`grasp.off_plane_axes`, the same call `candidate_grasps` makes). A plate
        lying flat holds its short axis along the plane's normal, and no tilt short of that
        band makes the wide face a grip surface. ``tilt_rad`` must be the same value
        `_ranked`/`candidate_grasps` picks the grasp against, or a tip could pass a jaw plane
        the pick itself would refuse.
        """
        axes = grasp.body_axes_world(scene.brick_quat[part])
        short = int(torch.as_tensor(self.sizes[part]).argmin())
        off_plane = float(grasp.off_plane_axes(axes, tilt_rad)[short])
        return off_plane <= math.sin(grasp.AXIS_MAX_TILT_RAD)

    # -- selection ------------------------------------------------------------

    @staticmethod
    def cell_of(xy) -> tuple[int, int]:
        return (math.floor(xy[0] / grasp.PUSH_CELL_M), math.floor(xy[1] / grasp.PUSH_CELL_M))

    def _candidates(self, scene: oracle.SceneSnapshot) -> torch.Tensor:
        """(B,) bool: parts still to pick that the arm can reach and whose cell is open."""
        done = self._stowed(scene) | self._sorted
        if self.boxes:
            # A part in any box, its own or another, is out of play: the grasp model knows no
            # walls, and a part in another box is the class branch's wrong_box, not a target.
            boxes, half = self._box_tensor()
            boxed = checker.in_boxes(scene.brick_pos.unsqueeze(0), boxes, half)[0].any(-1)
            done |= {b for b in range(self.num_bricks) if bool(boxed[b])}
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        self.out_of_reach = 0
        for b in range(self.num_bricks):
            x, y, z = scene.brick_pos[b].tolist()
            if b in done or z > CANDIDATE_MAX_Z:
                continue
            if not layout.in_reach(x, y, self.bl):
                self.out_of_reach += 1
                continue
            if self.failures.get(self._failure_key(b, (x, y)), 0) >= CELL_FAILURE_CAP:
                continue
            mask[b] = True
        return mask

    def _ranked(self, scene: oracle.SceneSnapshot, mask: torch.Tensor) -> list[grasp.Grasp]:
        return grasp.rank_grasps(
            self.sizes,
            scene.brick_pos,
            scene.brick_quat,
            mask,
            oracle.PICK_TILT_RAD,
            [p.closing_axes for p in self.parts],
            self.walls,
        )

    def _push_heading(self, center: tuple[float, float]) -> float:
        """A push heading from ``self._push_rng``, biased toward the drop disc's centre.

        Design H8 fixes the push's GEOMETRY and says nothing about the heading's
        distribution, so biasing that distribution is an improvement on H8. A plain uniform
        draw sends about half of all pushes near the disc's edge further out, and with no
        dish a part pushed out of the reach band is lost.

        The draw stays one ``uniform`` call, so a seeded episode stays reproducible. It is
        centred on the heading that points at the disc centre, within a half-arc that shrinks
        toward PUSH_HEADING_MIN_HALF_ARC as the cell nears the edge and opens back to a full
        circle at the centre.
        """
        cx, cy = layout.HEAP_CENTER
        dx, dy = center[0] - cx, center[1] - cy
        radius = math.hypot(dx, dy)
        away = math.atan2(-dy, -dx) if radius > 1e-6 else 0.0
        bias = max(0.0, min(1.0, radius / layout.HEAP_RADIUS))
        half_arc = math.pi - bias * (math.pi - PUSH_HEADING_MIN_HALF_ARC)
        return away + self._push_rng.uniform(-half_arc, half_arc)

    def _choose(self, ranked: list[grasp.Grasp]) -> tuple[grasp.Grasp | None, bool]:
        """HOVER's choice among the ranked grasps, and whether it is forced: the best clear
        grasp, else the top-ranked one once the pushes are spent (fix F2), else None."""
        chosen = grasp.select_grasp(ranked)
        if chosen is not None:
            return chosen, False
        if ranked and self.pushes_in_a_row >= MAX_PUSHES_IN_A_ROW:
            return ranked[0], True
        return None, False

    def _preselect(self, scene: oracle.SceneSnapshot) -> None:
        """RELEASE just ended: choose the next target now, so its row and marker show
        through RETREAT (the same rule the eval loop applies, ``core.loop``).

        The released part is in flight into the bin and not yet at rest, so it is excluded
        by index, not by the stow grade. Nothing is committed: ``cycles``, ``cycle_parts``,
        ``grasp`` and ``target_brick`` still describe the cycle that is retreating, and HOVER
        adopts the choice through ``_next_brick``. A scan that would push or tip is left for
        HOVER to plan, so no push or tip is ever scheduled twice.
        """
        self._clear_preselection()
        if self.cycles >= self.cycle_budget:
            return
        mask = self._candidates(scene)
        if self.target_brick is not None:
            mask[self.target_brick] = False
        if not bool(mask.any()):
            return
        chosen, forced = self._choose(self._ranked(scene, mask))
        if chosen is None or 2.0 * chosen.half_height < TIP_MIN_GRIP_HEIGHT:
            return
        self.next_brick = chosen.part
        self.next_brick_start = scene.brick_pos[chosen.part].clone()
        self.next_grasp = chosen
        self._next_forced = forced

    def _clear_preselection(self) -> None:
        self.next_brick = self.next_brick_start = self.next_grasp = None
        self._next_forced = False

    def _revalidated_preselection(
        self, scene: oracle.SceneSnapshot
    ) -> tuple[grasp.Grasp | None, bool]:
        """The pre-selection at HOVER entry if it still holds: not stowed and still a
        candidate, re-ranked on its live pose when it moved more than PRESELECT_MOVE_M."""
        part = self.next_brick
        if part is None or not bool(self._candidates(scene)[part]):
            return None, False
        if float((scene.brick_pos[part] - self.next_brick_start).norm()) <= PRESELECT_MOVE_M:
            return self.next_grasp, self._next_forced
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        mask[part] = True
        chosen, forced = self._choose(self._ranked(scene, mask))
        if chosen is None or 2.0 * chosen.half_height < TIP_MIN_GRIP_HEIGHT:
            return None, False
        return chosen, forced

    def _next_brick(self, scene: oracle.SceneSnapshot) -> int | None:
        """The highest clear grasp; schedules a push and returns None when nothing clears.

        A pre-selection made when RELEASE ended is adopted when it still holds
        (``_revalidated_preselection``); otherwise the selection runs on the live scene.
        Either way the pre-selection is spent here, so it never outlives one HOVER entry.
        """
        if self.cycles >= self.cycle_budget:
            self._clear_preselection()
            return None
        chosen, forced_pick = self._revalidated_preselection(scene)
        self._clear_preselection()
        if chosen is None:
            mask = self._candidates(scene)
            if not bool(mask.any()):
                return None
            ranked = self._ranked(scene, mask)
            chosen, forced_pick = self._choose(ranked)
            if chosen is None:
                if ranked and self.pushes < PUSH_BUDGET:
                    self.push = self._plan_push(scene, mask)
                elif not ranked and self.walls and self.rakes < self.max_rakes:
                    self.push = self._plan_rake(scene, mask)
                return None
            if 2.0 * chosen.half_height < TIP_MIN_GRIP_HEIGHT:
                # MS6 rework section 7: a flat part offers only its in-plane axes, so the
                # jaws would grip its thickness. Tip it up first and pick it on a later
                # scan. A forced pick that gets tipped instead never happened, so
                # `forced_pick` stays unrecorded until the part is actually picked (below).
                self.tip = self._plan_tip(scene, chosen.part)
                if self.tip is not None:
                    return None
        if forced_pick:
            self.forced_picks += 1
        self.grasp = chosen
        self.grasp_width = chosen.width
        self.grasp_min_size = min(self.parts[chosen.part].size)
        self.cycles += 1
        self.pushes_in_a_row = 0
        self.cycle_parts.append(chosen.part)
        return chosen.part

    def _plan_push(self, scene: oracle.SceneSnapshot, mask: torch.Tensor) -> Push:
        """A sweep through the densest cell of the candidates (design H8)."""
        center = grasp.densest_cell(scene.brick_pos[mask, :2])
        return Push(*grasp.push_segment(center, self._push_heading(center)))

    def _plan_rake(self, scene: oracle.SceneSnapshot, mask: torch.Tensor) -> Push | None:
        """A rake off a wall for the candidate nearest one, or None (MS7a design, the rake).

        Called only when no grasp ranks. A candidate within ``grasp.RAKE_BAND_M`` of a wall
        and under ``RAKE_PART_CAP`` rakes is swept off its nearest wall by one part length,
        its longest side (at least ``grasp.PUSH_HALF_LENGTH_M``), the nearest first; a part
        whose every start lands the tips on a neighbor is skipped (``grasp.rake_segment``).
        """
        near = []
        for b in torch.nonzero(mask).flatten().tolist():
            if self.raked.get(b, 0) >= self.rake_part_cap:
                continue
            xy = tuple(scene.brick_pos[b, :2].tolist())
            dist, wall = min((grasp.wall_distance(xy, w), k) for k, w in enumerate(self.walls))
            if dist < grasp.RAKE_BAND_M:
                near.append((dist, b, self.walls[wall]))
        if not near:
            return None
        prints = {
            b: grasp.footprint(self.sizes[b], scene.brick_pos[b], scene.brick_quat[b])
            for b in range(self.num_bricks)
            if float(scene.brick_pos[b, 2]) <= CANDIDATE_MAX_Z
        }
        for _, b, wall in sorted(near, key=lambda t: (t[0], t[1])):
            length = max(max(self.parts[b].size), grasp.PUSH_HALF_LENGTH_M)
            others = [f for k, f in prints.items() if k != b]
            xy = tuple(scene.brick_pos[b, :2].tolist())
            segment = grasp.rake_segment(
                xy, prints[b], length, wall, self.walls, others, self.rake_crowd
            )
            if segment is not None:
                start, end, heading = segment
                return Push(start, end, part=b, heading=heading, jaw=heading)
        return None

    def _push_yaw(self) -> float | None:
        """Goal.yaw of a push leg: a rake squares the jaws to ``Push.jaw``, a push holds."""
        if self.push is None or self.push.jaw is None:
            return None
        return self.push.jaw - math.pi / 2

    def aim_push(self, push: Push) -> None:
        """Enter the push phases on ``push`` from the HOVER that ``resume`` left pending: the
        heap loop's oracle driver runs a rake its selector planned as a takeover of its own
        (``OracleRig.resume(push=...)``), which ends back at HOVER."""
        self.push = push
        self.push_takeover = True
        self.target_brick = None
        self.phase = oracle.Phase.PUSH_HOVER

    def _turn_rake_jaw(self) -> bool:
        """The rig could not plan the rake's hover at this jaw heading: take the next one.

        The closing axis runs along the sweep first (the finger plates stand parallel to the
        wall), then across it, then the jaws hold the heading they have. Driven on 2026-09-23:
        the first fails to plan off the near and far walls, where it points the closing axis
        along the arm's radius, and the second off the left wall. False when all three failed.
        """
        push = self.push
        if push.jaw is None:
            return False
        along = abs(oracle.wrap_to_pi(push.jaw - push.heading)) < 1e-9
        self.push = replace(push, jaw=push.heading + math.pi / 2 if along else None)
        return True

    def _plan_tip(self, scene: oracle.SceneSnapshot, part: int) -> Tip | None:
        """A short stroke into the part's edge, from outside its footprint toward its centre.

        The stroke runs along the part's SHORT in-plane axis, which is the edge a closed jaw
        can catch, and it ends past the part's centre so the part rolls up onto that edge.
        """
        print_ = grasp.footprint(self.sizes[part], scene.brick_pos[part], scene.brick_quat[part])
        axes = grasp.body_axes_world(scene.brick_quat[part])
        sizes = torch.as_tensor(self.sizes[part])
        order = sorted(range(3), key=lambda k: float(sizes[k]))
        lateral = next((k for k in order if abs(float(axes[k, 2])) < 0.5), order[0])
        heading = math.atan2(float(axes[lateral, 1]), float(axes[lateral, 0]))
        centre = tuple(scene.brick_pos[part, :2].tolist())
        for sign in (1.0, -1.0):
            c, s = sign * math.cos(heading), sign * math.sin(heading)
            offset = 0.5 * float(sizes[lateral]) + grasp.PUSH_START_GAP_M + grasp.TIP_HALF_FACE
            start = grasp.clamp_to_reach((centre[0] + offset * c, centre[1] + offset * s))
            if grasp.polygon_distance(grasp.tip_footprint(start), print_) < 0.0:
                continue
            end = grasp.clamp_to_reach((start[0] - TIP_STROKE_M * c, start[1] - TIP_STROKE_M * s))
            if math.dist(start, end) < 0.5 * TIP_STROKE_M:
                continue
            return Tip(start, end, part)
        return None

    def _regrasp(self, scene: oracle.SceneSnapshot) -> None:
        """After REOPEN: the same part at its live pose, best grasp even under the floor."""
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        mask[self.target_brick] = True
        ranked = self._ranked(scene, mask)
        if ranked:
            self.grasp = ranked[0]
            self.grasp_width = ranked[0].width

    def adopt_held_part(self, part: int) -> None:
        """The LIFT found this part in the jaws, not the target: the rest of the cycle follows it.

        Two things key on the part after LIFT, and both are here: ``target_brick``, which
        RETREAT grades and ``_mark_failure`` reads, and the cycle log the yield reads pair with
        ``cycle_outcomes`` by index (scripts/heap_yield_read.py). RAISE, EXTEND and RELEASE all
        aim at the bin, never at the part, so no leg has to be re-planned. ``pick_order`` is
        left alone: it records what selection CHOSE, and this is not a selection.
        """
        self.target_brick = part
        self.held_swaps += 1
        if self.cycle_parts:  # a resumed takeover enters LIFT with no cycle of its own
            self.cycle_parts[-1] = part

    # -- the outer loop's entry points (phase B design 5.2 and 5.5) ---------------

    def select_target(self, scene: oracle.SceneSnapshot) -> int | None:
        """The outer loop's selector: the teacher's own rule, with the teacher's bookkeeping.

        A returned part starts a cycle: ``cycles`` is bumped and the part joins
        ``cycle_parts``, so the loop and the teacher count cycles the same way. None means no
        candidate or a spent budget, and the loop ends the episode. The machine is not
        entered: ``target_brick`` stays None, so the same oracle can select again. A
        pre-selection a takeover's RELEASE left behind is dropped first: the loop's selector
        always ranks the live scene.
        """
        self._clear_preselection()
        return self._next_brick(scene)

    def resume(self, scene: oracle.SceneSnapshot, held: int) -> None:
        self._clear_preselection()  # a takeover's HOVER is aimed by the loop, not by RELEASE
        super().resume(scene, held)

    def aim_hover_at(self, scene: oracle.SceneSnapshot, target: int) -> bool:
        """Point the HOVER that ``resume`` left pending at the loop's target.

        ``resume`` with no held part enters HOVER with no target, and HOVER's entry would
        select on its own. A takeover works the loop's cycle instead, so the target is set
        here with the grasp the teacher would rank on it. Nominal plan: no hover jitter, no
        induced style. False when no grasp ranks on the part any more (the policy moved it);
        the caller then declines the takeover and the loop ends the cycle as a failure.
        """
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        mask[target] = True
        ranked = self._ranked(scene, mask)
        if not ranked:
            return False
        self.grasp = ranked[0]
        self.grasp_width = ranked[0].width
        self.grasp_min_size = min(self.parts[target].size)
        self.target_brick = target
        self._brick_start = scene.brick_pos[target].clone()
        self._hover_dx = self._hover_dy = 0.0
        self.pick_order.append(target)
        return True

    def _target_yaw(self, scene: oracle.SceneSnapshot) -> float:
        # Goal.yaw is a LONG-AXIS heading; the runtime folds yaw + pi/2 as the closing axis.
        if self.grasp is None:
            return super()._target_yaw(scene)
        return self.grasp.heading - math.pi / 2 + self.draw.yaw_offset

    def _target_half_height(self) -> float:
        return self.grasp.half_height if self.grasp is not None else super()._target_half_height()

    # -- goals ------------------------------------------------------------------

    def next_goal(self, scene: oracle.SceneSnapshot) -> oracle.Goal | None:
        if self.phase == oracle.Phase.HOVER and self.target_brick is None:
            goal = super().next_goal(scene)  # selects, or sets PUSH_HOVER / TIP_HOVER / DONE
            if goal is not None or self.phase not in (
                oracle.Phase.PUSH_HOVER,
                oracle.Phase.TIP_HOVER,
            ):
                return goal
        if self.phase == oracle.Phase.PUSH_HOVER:
            target = torch.tensor([*self.push.start, PUSH_HOVER_Z])
            return self._goal(
                oracle.Phase.PUSH_HOVER,
                target,
                self._push_yaw(),
                0.0,
                -1.0,
                0.01,
                0,
                oracle.DEFAULT_MAX_STEPS,
                hold_xy=(self.push.start, self.push.end) if self.push.jaw is not None else (),
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.PUSH_DESCEND:
            target = torch.tensor([*self.push.start, PUSH_Z])
            return self._goal(
                oracle.Phase.PUSH_DESCEND,
                target,
                self._push_yaw(),
                0.0,
                -1.0,
                0.003,
                oracle.DESCEND_STALL_STEPS,
                oracle.DEFAULT_MAX_STEPS,
                hold_xy=(self.push.start, self.push.end) if self.push.jaw is not None else (),
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.PUSH_SWEEP:
            target = torch.tensor([*self.push.end, PUSH_Z])
            return self._goal(
                oracle.Phase.PUSH_SWEEP,
                target,
                None,
                0.0,
                -1.0,
                0.01,
                0,
                PUSH_SWEEP_STEPS,
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.TIP_HOVER:
            target = torch.tensor([*self.tip.start, TIP_HOVER_Z])
            return self._goal(
                oracle.Phase.TIP_HOVER,
                target,
                None,
                0.0,
                -1.0,
                0.01,
                0,
                oracle.DEFAULT_MAX_STEPS,
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.TIP_DESCEND:
            target = torch.tensor([*self.tip.start, TIP_Z])
            return self._goal(
                oracle.Phase.TIP_DESCEND,
                target,
                None,
                0.0,
                -1.0,
                0.003,
                oracle.DESCEND_STALL_STEPS,
                oracle.DEFAULT_MAX_STEPS,
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.TIP_SWEEP:
            target = torch.tensor([*self.tip.end, TIP_Z])
            return self._goal(
                oracle.Phase.TIP_SWEEP,
                target,
                None,
                0.0,
                -1.0,
                0.01,
                0,
                TIP_SWEEP_STEPS,
                half_height=PUSH_HALF_HEIGHT,
            )
        return super().next_goal(scene)

    # -- transitions ---------------------------------------------------------

    def report(self, result: oracle.PhaseResult, scene: oracle.SceneSnapshot) -> None:
        phase = self.phase
        if phase == oracle.Phase.PUSH_HOVER:
            if result.outcome == "unplanned" and self.push.part is not None:
                if not self._turn_rake_jaw():
                    self._end_push()  # no heading plans: spent, so the part is not retried forever
                return  # PUSH_HOVER again, at the next jaw heading
            self.phase = oracle.Phase.PUSH_DESCEND
            return
        if phase == oracle.Phase.PUSH_DESCEND:
            self.phase = oracle.Phase.PUSH_SWEEP
            return
        if phase == oracle.Phase.PUSH_SWEEP:
            self._end_push()
            return
        if phase == oracle.Phase.TIP_HOVER:
            self.phase = oracle.Phase.TIP_DESCEND
            return
        if phase == oracle.Phase.TIP_DESCEND:
            self.phase = oracle.Phase.TIP_SWEEP
            return
        if phase == oracle.Phase.TIP_SWEEP:
            part = self.tip.part
            self.tips += 1
            # A tip is a corrective like a push, and counts against the same
            # forced-pick grace period: a successful tip costs nothing once the pick
            # it enables resets the counter on the next scan, but a failed tip must
            # count against the policy so a run of correctives on unrelated parts
            # cannot pin the counter at MAX_PUSHES_IN_A_ROW forever.
            self.pushes_in_a_row += 1
            if self.standing(scene, part):
                self.tips_ok += 1
            else:
                self._mark_failure(scene, part)
            self.tip = None
            self.phase = oracle.Phase.HOVER
            return
        failed_lift = phase == oracle.Phase.LIFT and not result.lifted
        if failed_lift and self.retries_used >= oracle.MAX_GRASP_RETRIES:
            # The base class abandons the episode here; a heap marks the cell and moves on,
            # through REOPEN so a part riding in the pocket is dropped here, not carried off
            # by the next transit and released over the wall (K=12 read, 2026-09-16).
            self._mark_failure(scene, self.target_brick)
            self.cycle_outcomes.append(False)
            self._give_up = True
            self.phase = oracle.Phase.REOPEN
            return
        brick = self.target_brick
        if phase == oracle.Phase.REOPEN and self._give_up:
            self._give_up = False
            self.target_brick = None
            self.phase = oracle.Phase.HOVER
            return
        super().report(result, scene)
        if phase == oracle.Phase.RELEASE:
            self._preselect(scene)
        if phase == oracle.Phase.REOPEN:
            self._regrasp(scene)
        if phase == oracle.Phase.RETREAT and brick is not None and brick not in self._sorted:
            self._mark_failure(scene, brick)

    def recover_from_jam(self, scene: oracle.SceneSnapshot) -> bool:
        """The rig's watchdog found the arm jammed: give up the leg, not the episode.

        MS7a tray only; on the bare desk this returns False and the rig abandons the episode as
        in MS6. A jammed pick (DESCEND, CLOSE or LIFT) ends its cycle the way a lift that
        failed its retries does, through REOPEN with the cell marked; a jammed rake or push is
        spent. On the 600-part tray read of 2026-09-23 two of twelve episodes were abandoned
        with 19 and 7 parts still in the tray: the wrist roll 0.02 rad off its stop while a
        forced pick's descend stalled on a neighbor, and a wedge rake at the near wall.
        Anything else (a jam in transit) still abandons.
        """
        if not self.walls:
            return False
        if self.phase in oracle.PUSH_PHASES and self.push is not None:
            self.jams += 1
            self._end_push()
            return True
        picking = (oracle.Phase.DESCEND, oracle.Phase.CLOSE, oracle.Phase.LIFT)
        if self.phase in picking and self.target_brick is not None:
            self.jams += 1
            self._mark_failure(scene, self.target_brick)
            self.cycle_outcomes.append(False)
            self._give_up = True
            self.phase = oracle.Phase.REOPEN
            return True
        return False

    def _end_push(self) -> None:
        """Count the push or rake that just ended and scan again at HOVER."""
        if self.push.part is None:
            self.pushes += 1
            self.pushes_in_a_row += 1
        else:
            self.rakes += 1
            self.raked[self.push.part] = self.raked.get(self.push.part, 0) + 1
        self.push = None
        self.phase = oracle.Phase.HOVER

    def _failure_key(self, brick: int, xy) -> tuple[int, int] | int:
        """Where a failure is booked: the part's 5 cm cell, or the part (``failure_scope``)."""
        return brick if self.failure_scope == "part" else self.cell_of(xy)

    def _mark_failure(self, scene: oracle.SceneSnapshot, brick: int) -> None:
        key = self._failure_key(brick, scene.brick_pos[brick, :2].tolist())
        self.failures[key] = self.failures.get(key, 0) + 1

    def class_to_box(self) -> dict[str, int]:
        """The box of each class the heap holds, read off the per-part destinations."""
        return {part_class(p): b for p, b in zip(self.parts, self.part_boxes)}

    def extra_stats(self) -> dict:
        """Merged into ``OracleRig.episode_stats`` beside the induction counters."""
        stats = {
            "cycles": self.cycles,
            "pushes": self.pushes,
            "rakes": self.rakes,
            "jams": self.jams,
            "tips": self.tips,
            "tips_ok": self.tips_ok,
            "forced_picks": self.forced_picks,
            "held_swaps": self.held_swaps,
            "cycle_parts": list(self.cycle_parts),
            "cycle_outcomes": list(self.cycle_outcomes),
            "cells_excluded": sum(1 for v in self.failures.values() if v >= CELL_FAILURE_CAP),
            "parts_out_of_reach": self.out_of_reach,
        }
        if self.boxes:
            # The episode's class-to-box map and each part's box (MS7a design, classes).
            stats["part_boxes"] = list(self.part_boxes)
            stats["class_to_box"] = self.class_to_box()
        return stats

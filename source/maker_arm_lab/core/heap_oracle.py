"""The heap teacher: ``EnvOracle`` over mixed parts in a tray (MS6 phase A design 7).

What changes against the six-brick oracle, and nothing else:

* the target is the highest part with a clear grasp (``core.grasp``), and the jaws square to
  that grasp's closing axis instead of the brick's long axis;
* every part goes to one destination bin drawn per episode, whatever its color (design H5);
* a 5 cm cell holds a failure count; at three failures the cell is skipped (design 7.2);
* when no grasp clears, a push moves the best wall-blocked part toward the tray center with
  the jaws closed; with no wall-blocked part it sweeps the densest cell (design H8);
* a part whose center is outside the tray is counted but never a target (fix F1);
* after ``MAX_PUSHES_IN_A_ROW`` pushes with nothing clear, the top-ranked grasp whose tips
  land inside the tray is taken anyway as a forced pick (fix F2); with none, the pushes go
  on up to the push budget, so DONE comes only from a budget or an empty set;
* a part whose best grasp grips less than ``TIP_MIN_GRIP_HEIGHT`` of face is tipped onto an
  edge first, with a short stroke into it, and picked on a later scan (MS6 rework section 7);
* the episode ends DONE, never ABANDON, at the cycle budget or with no candidate left.

The phases, the correctives, the induction and the retry ladder are the base class's.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from maker_arm_lab.core import checker, grasp, layout, oracle
from maker_arm_lab.core.parts import Part

CYCLE_BUDGET = 50
CELL_FAILURE_CAP = 3
MAX_PUSHES_IN_A_ROW = 3
PUSH_BUDGET = CYCLE_BUDGET
CANDIDATE_MAX_Z = 0.10
"""A part above this height sits in a bin or on the shelf, never on the heap."""
PUSH_HOVER_Z = 0.08
PUSH_Z = layout.BRICK_SIZE[2] / 2
"""Pick-point height of the sweep: the plate tips skim the desk at the brick pick offset."""
PUSH_HALF_HEIGHT = layout.BRICK_SIZE[2] / 2
PUSH_SWEEP_STEPS = 150
TIP_MIN_GRIP_HEIGHT = 0.005
"""A part whose best grasp grips less than this (m) of face is tipped up, not picked.

A plate is 3.2 mm thick and a brick is 9.6 mm tall, so the threshold takes the flat parts
and leaves the bricks alone. Heap 0 of 2026-09-17: plates failed 13 of their 24 lifts.
"""
TIP_STROKE_M = 0.025
TIP_HOVER_Z = PUSH_HOVER_Z
TIP_Z = 0.004
"""Pick-point height of the tip stroke: the closed tips skim under a lying plate's edge."""
TIP_SWEEP_STEPS = PUSH_SWEEP_STEPS
PUSH_HEADING_MIN_HALF_ARC = math.pi / 2
"""Narrowest half-arc a push heading is drawn from, right at a tray wall (see ``_push_heading``)."""
# Design H6: the bin footprint, and twice the interior height so a pile over the rim counts.
STOW_HALF_EXTENTS = (
    layout.HEAP_BIN_INTERIOR[0] / 2,
    layout.HEAP_BIN_INTERIOR[1] / 2,
    layout.HEAP_BIN_INTERIOR[2] * 2,
)


@dataclass(frozen=True)
class Push:
    start: tuple[float, float]
    end: tuple[float, float]


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
    ):
        super().__init__(bl, len(parts), draw, induction=induction, rng=rng)
        self.parts = tuple(parts)
        self.destination = int(destination)
        self.sizes = torch.tensor([p.size for p in self.parts], dtype=torch.float32)
        self.cycle_budget = cycle_budget
        self.grasp: grasp.Grasp | None = None
        self.grasp_min_size = min(layout.BRICK_SIZE)
        self.push: Push | None = None
        self.tip: Tip | None = None
        self.cycles = 0
        self.pushes = 0
        self.tips = 0
        self.tips_ok = 0
        self.forced_picks = 0
        self.push_parts: list[int] = []
        self._give_up = False
        self.pushes_in_a_row = 0
        self.failures: dict[tuple[int, int], int] = {}
        self.cycle_parts: list[int] = []
        self.outside_tray = 0
        self.walls = grasp.tray_wall_obstacles()
        self._push_rng = rng if rng is not None else random.Random(0)

    # -- grading against the destination -----------------------------------

    def _stowed(self, scene: oracle.SceneSnapshot) -> set[int]:
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
        limit = layout.top_down_x_limit(self.bl)
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        self.outside_tray = 0
        for b in range(self.num_bricks):
            x, y, z = scene.brick_pos[b].tolist()
            if b in done or z > CANDIDATE_MAX_Z:
                continue
            if not layout.inside_tray(x, y):
                self.outside_tray += 1
                continue
            if x > limit or math.hypot(x, y) > layout.SCATTER_MAX_RADIUS:
                continue
            if self.failures.get(self.cell_of((x, y)), 0) >= CELL_FAILURE_CAP:
                continue
            mask[b] = True
        return mask

    def _ranked(self, scene: oracle.SceneSnapshot, mask: torch.Tensor) -> list[grasp.Grasp]:
        return grasp.rank_grasps(self.sizes, scene.brick_pos, scene.brick_quat, mask, self.walls)

    def _push_heading(self, center: tuple[float, float]) -> float:
        """A push heading from ``self._push_rng``, biased away from the dish rim.

        Design H8 fixes the push's GEOMETRY and says nothing about the heading's
        distribution, so biasing that distribution is an improvement on H8. A plain uniform
        draw sends about half of all pushes near the rim further into it.

        The draw stays one ``uniform`` call, so a seeded episode stays reproducible. It is
        centred on the heading that points at the dish centre, within a half-arc that shrinks
        toward PUSH_HEADING_MIN_HALF_ARC as the cell nears the rim and opens back to a full
        circle at the centre.
        """
        cx, cy = layout.TRAY_CENTER
        dx, dy = center[0] - cx, center[1] - cy
        radius = math.hypot(dx, dy)
        away = math.atan2(-dy, -dx) if radius > 1e-6 else 0.0
        bias = max(0.0, min(1.0, radius / layout.TRAY_RADIUS))
        half_arc = math.pi - bias * (math.pi - PUSH_HEADING_MIN_HALF_ARC)
        return away + self._push_rng.uniform(-half_arc, half_arc)

    def _next_brick(self, scene: oracle.SceneSnapshot) -> int | None:
        """The highest clear grasp; schedules a push and returns None when nothing clears."""
        if self.cycles >= self.cycle_budget:
            return None
        mask = self._candidates(scene)
        if not bool(mask.any()):
            return None
        ranked = self._ranked(scene, mask)
        chosen = grasp.select_grasp(ranked)
        forced_pick = False
        if chosen is None:
            if not ranked:
                return None
            # Fix F2: after the pushes, try the best grasp whose tips land inside the tray.
            # A jaw that lands past a wall never picks (wall reads, 2026-09-16).
            inside = [g for g in ranked if g.wall >= 0.0]
            if inside and self.pushes_in_a_row >= MAX_PUSHES_IN_A_ROW:
                chosen = inside[0]
                forced_pick = True
            elif self.pushes < PUSH_BUDGET:
                self.push = self._plan_push(scene, mask, ranked)
                return None
            else:
                return None
        if 2.0 * chosen.half_height < TIP_MIN_GRIP_HEIGHT:
            # MS6 rework section 7: a flat part offers only its in-plane axes, so the jaws
            # would grip its thickness. Tip it up first and pick it on a later scan. A
            # forced pick that gets tipped instead never happened, so `forced_pick` stays
            # unrecorded until the part is actually picked (below).
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

    def _plan_push(self, scene: oracle.SceneSnapshot, mask: torch.Tensor, ranked) -> Push:
        """A push toward the tray center for the best wall-blocked part, or a densest-cell sweep."""
        prints = {
            b: grasp.footprint(self.sizes[b], scene.brick_pos[b], scene.brick_quat[b])
            for b in range(self.num_bricks)
            if float(scene.brick_pos[b, 2]) <= CANDIDATE_MAX_Z
        }
        seen: set[int] = set()
        for g in ranked:
            if g.wall >= 0.0 or g.part in seen:
                continue
            seen.add(g.part)
            others = [f for b, f in prints.items() if b != g.part]
            xy = tuple(scene.brick_pos[g.part, :2].tolist())
            segment = grasp.push_toward_center(xy, prints[g.part], others)
            if segment is not None:
                self.push_parts.append(g.part)
                return Push(*segment)
        center = grasp.densest_cell(scene.brick_pos[mask, :2])
        heading = self._push_heading(center)
        self.push_parts.append(-1)
        return Push(*grasp.push_segment(center, heading))

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
            start = grasp.clamp_to_tray((centre[0] + offset * c, centre[1] + offset * s))
            if grasp.polygon_distance(grasp.tip_footprint(start), print_) < 0.0:
                continue
            end = grasp.clamp_to_tray((start[0] - TIP_STROKE_M * c, start[1] - TIP_STROKE_M * s))
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
                None,
                0.0,
                -1.0,
                0.01,
                0,
                oracle.DEFAULT_MAX_STEPS,
                half_height=PUSH_HALF_HEIGHT,
            )
        if self.phase == oracle.Phase.PUSH_DESCEND:
            target = torch.tensor([*self.push.start, PUSH_Z])
            return self._goal(
                oracle.Phase.PUSH_DESCEND,
                target,
                None,
                0.0,
                -1.0,
                0.003,
                oracle.DESCEND_STALL_STEPS,
                oracle.DEFAULT_MAX_STEPS,
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
            self.phase = oracle.Phase.PUSH_DESCEND
            return
        if phase == oracle.Phase.PUSH_DESCEND:
            self.phase = oracle.Phase.PUSH_SWEEP
            return
        if phase == oracle.Phase.PUSH_SWEEP:
            self.pushes += 1
            self.pushes_in_a_row += 1
            self.push = None
            self.phase = oracle.Phase.HOVER
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
        if phase == oracle.Phase.REOPEN:
            self._regrasp(scene)
        if phase == oracle.Phase.RETREAT and brick is not None and brick not in self._sorted:
            self._mark_failure(scene, brick)

    def _mark_failure(self, scene: oracle.SceneSnapshot, brick: int) -> None:
        cell = self.cell_of(scene.brick_pos[brick, :2].tolist())
        self.failures[cell] = self.failures.get(cell, 0) + 1

    def extra_stats(self) -> dict:
        """Merged into ``OracleRig.episode_stats`` beside the induction counters."""
        return {
            "cycles": self.cycles,
            "pushes": self.pushes,
            "tips": self.tips,
            "tips_ok": self.tips_ok,
            "forced_picks": self.forced_picks,
            "push_parts": list(self.push_parts),
            "cycle_parts": list(self.cycle_parts),
            "cycle_outcomes": list(self.cycle_outcomes),
            "cells_excluded": sum(1 for v in self.failures.values() if v >= CELL_FAILURE_CAP),
            "parts_outside_tray": self.outside_tray,
        }

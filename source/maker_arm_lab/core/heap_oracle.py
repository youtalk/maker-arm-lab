"""The heap teacher: ``EnvOracle`` over mixed parts in a tray (MS6 phase A design 7).

What changes against the six-brick oracle, and nothing else:

* the target is the highest part with a clear grasp (``core.grasp``), and the jaws square to
  that grasp's closing axis instead of the brick's long axis;
* every part goes to one destination bin drawn per episode, whatever its color (design H5);
* a 5 cm cell holds a failure count; at three failures the cell is skipped (design 7.2);
* when no grasp clears, a push sweeps the densest cell with the jaws closed (design H8);
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
CANDIDATE_MAX_Z = 0.10
"""A part above this height sits in a bin or on the shelf, never on the heap."""
PUSH_HOVER_Z = 0.08
PUSH_Z = layout.BRICK_SIZE[2] / 2
"""Pick-point height of the sweep: the plate tips skim the desk at the brick pick offset."""
PUSH_HALF_HEIGHT = layout.BRICK_SIZE[2] / 2
PUSH_SWEEP_STEPS = 150
# Design H6: the bin footprint, and twice the interior height so a pile over the rim counts.
STOW_HALF_EXTENTS = (
    layout.BIN_INTERIOR[0] / 2,
    layout.BIN_INTERIOR[1] / 2,
    layout.BIN_INTERIOR[2] * 2,
)


@dataclass(frozen=True)
class Push:
    start: tuple[float, float]
    end: tuple[float, float]


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
        self.push: Push | None = None
        self.cycles = 0
        self.pushes = 0
        self.pushes_in_a_row = 0
        self.failures: dict[tuple[int, int], int] = {}
        self.cycle_parts: list[int] = []
        self.outside_tray = 0
        self.wall_pos, self.wall_rxy, self.wall_hz = grasp.tray_wall_obstacles()
        self._push_rng = rng if rng is not None else random.Random(0)

    # -- grading against the destination -----------------------------------

    def _stowed(self, scene: oracle.SceneSnapshot) -> set[int]:
        ids = torch.full((self.num_bricks,), self.destination, dtype=torch.long)
        ok = checker.bricks_at_rest_in_bins(
            scene.brick_pos.unsqueeze(0),
            scene.brick_lin_vel.unsqueeze(0),
            layout.bin_interior_centers(self.bl),
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
        return oracle.Phase.PUSH_HOVER if self.push is not None else oracle.Phase.DONE

    # -- selection ------------------------------------------------------------

    @staticmethod
    def cell_of(xy) -> tuple[int, int]:
        return (math.floor(xy[0] / grasp.PUSH_CELL_M), math.floor(xy[1] / grasp.PUSH_CELL_M))

    def _candidates(self, scene: oracle.SceneSnapshot) -> torch.Tensor:
        """(B,) bool: parts still to pick that the arm can reach and whose cell is open."""
        done = self._stowed(scene) | self._sorted
        limit = layout.top_down_x_limit(self.bl)
        (x0, x1), (y0, y1) = layout.tray_bounds()
        mask = torch.zeros(self.num_bricks, dtype=torch.bool)
        self.outside_tray = 0
        for b in range(self.num_bricks):
            x, y, z = scene.brick_pos[b].tolist()
            if b in done or z > CANDIDATE_MAX_Z:
                continue
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                self.outside_tray += 1
            if x > limit or math.hypot(x, y) > layout.SCATTER_MAX_RADIUS:
                continue
            if self.failures.get(self.cell_of((x, y)), 0) >= CELL_FAILURE_CAP:
                continue
            mask[b] = True
        return mask

    def _ranked(self, scene: oracle.SceneSnapshot, mask: torch.Tensor) -> list[grasp.Grasp]:
        return grasp.rank_grasps(
            self.sizes,
            scene.brick_pos,
            scene.brick_quat,
            mask,
            self.wall_pos,
            self.wall_rxy,
            self.wall_hz,
        )

    def _next_brick(self, scene: oracle.SceneSnapshot) -> int | None:
        """The highest clear grasp; schedules a push and returns None when nothing clears."""
        if self.cycles >= self.cycle_budget:
            return None
        mask = self._candidates(scene)
        if not bool(mask.any()):
            return None
        chosen = grasp.select_grasp(self._ranked(scene, mask))
        if chosen is None:
            if self.pushes_in_a_row >= MAX_PUSHES_IN_A_ROW:
                return None
            center = grasp.densest_cell(scene.brick_pos[mask, :2])
            heading = self._push_rng.uniform(-math.pi, math.pi)
            self.push = Push(*grasp.push_segment(center, heading))
            return None
        self.grasp = chosen
        self.grasp_width = chosen.width
        self.cycles += 1
        self.pushes_in_a_row = 0
        self.cycle_parts.append(chosen.part)
        return chosen.part

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
        return self.grasp.heading - math.pi / 2 + self.draw.yaw_offset

    def _target_half_height(self) -> float:
        return self.grasp.half_height if self.grasp is not None else super()._target_half_height()

    # -- goals ------------------------------------------------------------------

    def next_goal(self, scene: oracle.SceneSnapshot) -> oracle.Goal | None:
        if self.phase == oracle.Phase.HOVER and self.target_brick is None:
            goal = super().next_goal(scene)  # selects, or sets PUSH_HOVER / DONE via the hook
            if goal is not None or self.phase != oracle.Phase.PUSH_HOVER:
                return goal
        if self.phase == oracle.Phase.PUSH_HOVER:
            target = torch.tensor([*self.push.start, PUSH_HOVER_Z])
            return self._goal(
                oracle.Phase.PUSH_HOVER, target, None, 0.0, -1.0, 0.01, 0, oracle.DEFAULT_MAX_STEPS
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
        failed_lift = phase == oracle.Phase.LIFT and not (result.pinch_ok and result.lifted)
        if failed_lift and self.retries_used >= oracle.MAX_GRASP_RETRIES:
            # The base class abandons the episode here; a heap marks the cell and moves on.
            self._mark_failure(scene, self.target_brick)
            self.cycle_outcomes.append(False)
            self.target_brick = None
            self.phase = oracle.Phase.HOVER
            return
        brick = self.target_brick
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
            "cycle_parts": list(self.cycle_parts),
            "cycle_outcomes": list(self.cycle_outcomes),
            "cells_excluded": sum(1 for v in self.failures.values() if v >= CELL_FAILURE_CAP),
            "parts_outside_tray": self.outside_tray,
        }

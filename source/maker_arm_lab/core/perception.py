"""The MS7c perceived state: what a real overhead rig could report, and nothing more.

Torch and stdlib only. Behind the heap loop's ``--perceived`` switch, the loop's decisions
(target selection, the target and box rows, the marker, the cycle ends) read a
``PerceivedState`` instead of the simulator truth (MS7c design section 4). In simulation that
state is the truth passed through noise and failure models, one knob each (``KNOBS``). A knob
is a scale on a nominal value: 0 turns that model off and 1 is the nominal. Every nominal is
``# INITIAL:``, and MS5a measures it on real frames.

Two structural rules hold at every scale, zero included: a part whose centre another part
covers from above is not seen, and neither is a part under the tool
(``TOOL_OCCLUSION_RADIUS_M``).

A ``Detection`` keeps the simulator's part index in ``sim_index`` so that the metrics can
match it to the truth. Within one selection the selector sees it only as the slot that
``PerceivedState.scene`` puts the detection in. Nothing that outlives a selection keys on it:
the failure memory and the rake counter live by perceived position (``PositionMemory``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from maker_arm_lab.core import grasp, oracle
from maker_arm_lab.core.parts import BOX_AXES, PART_CLASSES, SLOPE_AXES, Part
from maker_arm_lab.core.quat import quat_apply, quat_mul

KNOBS = ("pos", "heading", "size", "box", "miss", "class")
"""The knob names, in the order ``format_scales`` writes them."""
EXTRA_KNOBS = ("pos_z",)
"""Diagnostic multipliers that ``all`` leaves alone and ``format_scales`` writes only when not
1: ``pos_z`` scales the z axis of the position bias and noise on top of ``pos``, to split the
position cost between xy and z (MS7c ledger, 2026-09-24)."""

# Position: the iPhone wide camera covers the 0.9 m desk with 1920 px, about 0.5 mm per pixel,
# and the tag-board extrinsics add a bias per episode. Both on (x, y, z).
POS_BIAS_SIGMA_M = (0.002, 0.002, 0.002)  # INITIAL: MS5a measures it on real frames
POS_NOISE_SIGMA_M = (0.0015, 0.0015, 0.002)  # INITIAL: MS5a measures it on real frames
# Heading: the principal axis of a mask, per detection.
HEADING_SIGMA_RAD = math.radians(4.0)  # INITIAL: MS5a measures it on real frames
# Size: the mask extent on the two body extents nearest the desk plane (the grasp width), and
# the depth on the one nearest world z at twice this, so the half height carries this sigma.
SIZE_SIGMA_M = 0.001  # INITIAL: MS5a measures it on real frames
# Box: a box fit or tags on the boxes, per box per episode.
BOX_SIGMA_M = 0.003  # INITIAL: MS5a measures it on real frames
BOX_YAW_SIGMA_RAD = math.radians(2.0)  # INITIAL: MS5a measures it on real frames
# Miss: a part the detector drops at one perceive call.
MISS_PROBABILITY = 0.05  # INITIAL: MS5a measures it on real frames
# Class: a wrong class per classification, the MS7b classifier gate of 0.97.
CLASS_ERROR_PROBABILITY = 0.03  # INITIAL: MS5a measures it on real frames
# Structural rule 2: the arm hides a part this close to the tool (xy) from the overhead camera.
TOOL_OCCLUSION_RADIUS_M = 0.040  # INITIAL: MS5a measures it on real frames
# PositionMemory: ten sigmas of the per-detection xy noise.
ASSOCIATION_RADIUS_M = 0.015  # INITIAL: the MS7c reads check it
PARKED_CENTRE = (3.0, 0.0, 1.0)
"""Where ``PerceivedState.scene`` parks a slot with no detection: 3 m out and 1 m up. That is
over ``heap_oracle.CANDIDATE_MAX_Z``, so the selector never takes it; outside every box and the
reach band; and so far from the desk that no landing spot or rake start measures against it."""
_PARKED_PART = Part("undetected", "brick", 0, 0, (0.008, 0.008, 0.008), False, 0.0)


def parse_scales(text: str) -> dict[str, float]:
    """A scale per knob from ``"pos=0,miss=4"``; a knob not named stays at 1.0.

    ``all=V`` sets every knob. Items apply left to right, so ``"all=0,heading=1"`` reads the
    heading alone. An unknown knob, or a value that is not a finite number >= 0, raises
    ValueError (an argparse ``type=`` turns it into a usage error).
    """
    scales = dict.fromkeys(KNOBS + EXTRA_KNOBS, 1.0)
    for item in text.split(","):
        if not item.strip():
            continue
        name, _, value = (s.strip() for s in item.partition("="))
        if name != "all" and name not in scales:
            names = ", ".join(KNOBS + EXTRA_KNOBS)
            raise ValueError(f"unknown perception knob {name!r}; knobs: all, {names}")
        try:
            scale = float(value)
        except ValueError:
            scale = math.nan
        if not (math.isfinite(scale) and scale >= 0.0):
            raise ValueError(f"perception knob {name} needs a finite number >= 0, got {value!r}")
        for knob in KNOBS if name == "all" else (name,):
            scales[knob] = scale
    return scales


def format_scales(scales: dict[str, float]) -> str:
    """The canonical form the payload and the gate cache record: every knob, in ``KNOBS``
    order, then each of ``EXTRA_KNOBS`` that is not 1, the shortest text that
    ``parse_scales`` reads back to the same float."""
    names = KNOBS + tuple(k for k in EXTRA_KNOBS if float(scales.get(k, 1.0)) != 1.0)
    return ",".join(f"{k}={repr(float(scales[k])).removesuffix('.0')}" for k in names)


@dataclass(frozen=True)
class Detection:
    """One perceived part, world frame."""

    centre: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # (w, x, y, z)
    size: tuple[float, float, float]  # extents along the body axes, as ``parts.Part.size``
    cls: str  # a ``parts.PART_CLASSES`` name
    sim_index: int
    """The simulator's part index: only the metrics and the teacher-side recorder may read it;
    decisions never do."""


@dataclass
class PerceivedState:
    detections: list[Detection]
    box_poses: tuple[tuple[float, float, float], ...]  # (x, y, yaw) per desk box, per episode

    def scene(self, num_parts: int, pick_point) -> oracle.SceneSnapshot:
        """The detections as the selector's snapshot (``heap_oracle.HeapOracle``): slot
        ``d.sim_index`` holds detection ``d`` at rest, and every other slot is parked at
        ``PARKED_CENTRE``."""
        pos = torch.tensor([PARKED_CENTRE] * num_parts)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_parts)
        for d in self.detections:
            pos[d.sim_index] = torch.tensor(d.centre)
            quat[d.sim_index] = torch.tensor(d.quat)
        # The rig's convention (``OracleRig._brick_headings``): the heading of body x.
        x_axis = quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]]).expand(num_parts, 3))
        return oracle.SceneSnapshot(
            brick_pos=pos,
            brick_yaw=torch.atan2(x_axis[:, 1], x_axis[:, 0]),
            brick_lin_vel=torch.zeros(num_parts, 3),
            pick_point=torch.as_tensor(pick_point, dtype=torch.float32),
            brick_quat=quat,
        )

    def parts(self, num_parts: int) -> tuple[Part, ...]:
        """One ``Part`` per slot for the selector: the perceived size and the closing axes of
        the perceived class. The other fields are placeholders and carry no truth."""
        out = [_PARKED_PART] * num_parts
        for d in self.detections:
            slope = d.cls == "slope"
            out[d.sim_index] = Part(
                "perceived",
                "slope" if slope else "brick",
                0,
                0,
                d.size,
                False,
                0.0,
                closing_axes=SLOPE_AXES if slope else BOX_AXES,
            )
        return tuple(out)


@dataclass(frozen=True)
class Truth:
    """The simulator state one ``Perceiver.perceive`` call reads, world frame."""

    pos: torch.Tensor  # (K, 3) part centres
    quat: torch.Tensor  # (K, 4) (w, x, y, z)
    sizes: torch.Tensor  # (K, 3) catalog extents along the body axes
    classes: Sequence[str]  # (K,) PART_CLASSES names
    tool_xy: tuple[float, float]  # the tool point the arm hides parts around


def _hidden(pos: torch.Tensor, quat: torch.Tensor, sizes: torch.Tensor, tool_xy) -> set[int]:
    """The parts the two structural rules remove: a part within TOOL_OCCLUSION_RADIUS_M of the
    tool in xy, and a part whose centre lies inside the footprint of a part whose centre is
    higher."""
    radius, _ = grasp.neighbor_extents(sizes, quat)
    centres = pos.tolist()
    hidden = set()
    for b, (x, y, z) in enumerate(centres):
        if math.hypot(x - tool_xy[0], y - tool_xy[1]) < TOOL_OCCLUSION_RADIUS_M:
            hidden.add(b)
            continue
        for c, (cx, cy, cz) in enumerate(centres):
            if cz <= z or math.hypot(x - cx, y - cy) > float(radius[c]):
                continue  # c is the part itself, lower, or too far to reach b's centre
            if grasp.polygon_distance([(x, y)], grasp.footprint(sizes[c], pos[c], quat[c])) < 0:
                hidden.add(b)
                break
    return hidden


class Perceiver:
    """One env's perception: its generator, the episode's biases and the per-call draws.

    The generator is seeded from the run seed and the env index. ``reset_episode`` draws the
    episode's position bias and then each box's bias; every ``perceive`` and ``classify``
    after it draws in call order. A read with the same seeds and knobs repeats exactly.

    The draws are PAIRED across knobs. Every call draws the same values in the same order
    whatever the scales and whatever the structural rules remove: per part 3 position, 1
    heading, 3 size, 1 miss and 2 class draws, per box 3. A scale only multiplies what was
    drawn, so two reads that differ in one knob see the same underlying noise, which is what
    a knob sweep compares on.
    """

    def __init__(self, run_seed: int, env_index: int, scales: dict[str, float] | None = None):
        unknown = set(scales or {}) - set(KNOBS + EXTRA_KNOBS)
        if unknown:
            raise ValueError(f"unknown perception knobs {sorted(unknown)}; knobs: {KNOBS}")
        self.scales = {**dict.fromkeys(KNOBS + EXTRA_KNOBS, 1.0), **(scales or {})}
        self.rng = random.Random(f"perceive:{run_seed}:{env_index}")
        self._bias = (0.0, 0.0, 0.0)
        self._box_poses: tuple[tuple[float, float, float], ...] = ()

    @property
    def box_poses(self) -> tuple[tuple[float, float, float], ...]:
        """The episode's perceived (x, y, yaw) per desk box, fixed until ``reset_episode``."""
        return self._box_poses

    @property
    def bias(self) -> tuple[float, float, float]:
        """The episode's position bias (x, y, z) on every detection, fixed until
        ``reset_episode``."""
        return self._bias

    def _gauss(self) -> float:
        return self.rng.gauss(0.0, 1.0)

    def _pos_axes(self) -> tuple[float, float, float]:
        """The position scale per axis: ``pos`` on x and y, ``pos`` times ``pos_z`` on z."""
        pos = self.scales["pos"]
        return (pos, pos, pos * self.scales["pos_z"])

    def reset_episode(self, true_boxes: Sequence[Sequence[float]]) -> None:
        """Draw the episode's position bias, then each box's bias in box order."""
        box = self.scales["box"]
        self._bias = tuple(
            a * s * self._gauss() for a, s in zip(self._pos_axes(), POS_BIAS_SIGMA_M)
        )
        poses = []
        for x, y, yaw in true_boxes:
            dx, dy, dyaw = self._gauss(), self._gauss(), self._gauss()
            poses.append(
                (
                    float(x) + box * BOX_SIGMA_M * dx,
                    float(y) + box * BOX_SIGMA_M * dy,
                    float(yaw) + box * BOX_YAW_SIGMA_RAD * dyaw,
                )
            )
        self._box_poses = tuple(poses)

    def classify(self, true_cls: str) -> str:
        """One classification: the true class, or at the class knob's rate one of the other
        three, uniformly. Always two draws."""
        fires = self.rng.random() < min(1.0, CLASS_ERROR_PROBABILITY * self.scales["class"])
        pick = self.rng.random()
        if not fires:
            return true_cls
        others = [c for c in PART_CLASSES if c != true_cls]
        return others[int(pick * len(others))]

    def perceive(self, truth: Truth) -> PerceivedState:
        """Every part's draws in index order, then the structural rules and the miss, then the
        detection: the centre plus the episode bias and the noise, the true orientation turned
        about world z by the heading noise, the catalog size plus the size noise, and the true
        class through ``classify``."""
        s = self.scales
        pos = torch.as_tensor(truth.pos, dtype=torch.float32)
        quat = torch.as_tensor(truth.quat, dtype=torch.float32)
        sizes = torch.as_tensor(truth.sizes, dtype=torch.float32)
        hidden = _hidden(pos, quat, sizes, truth.tool_xy)
        miss = min(1.0, MISS_PROBABILITY * s["miss"])
        axes = self._pos_axes()
        detections = []
        for b in range(pos.shape[0]):
            noise = [a * sigma * self._gauss() for a, sigma in zip(axes, POS_NOISE_SIGMA_M)]
            dyaw = s["heading"] * HEADING_SIGMA_RAD * self._gauss()
            dsize = [s["size"] * SIZE_SIGMA_M * self._gauss() for _ in range(3)]
            missed = self.rng.random() < miss
            cls = self.classify(truth.classes[b])
            if missed or b in hidden:
                continue
            dsize[int(grasp.body_axes_world(quat[b])[:, 2].abs().argmax())] *= 2.0
            turn = (math.cos(dyaw / 2.0), 0.0, 0.0, math.sin(dyaw / 2.0))
            detections.append(
                Detection(
                    centre=tuple(t + e + n for t, e, n in zip(pos[b].tolist(), self._bias, noise)),
                    quat=quat_mul(turn, quat[b].tolist()),
                    size=tuple(max(0.0, e + n) for e, n in zip(sizes[b].tolist(), dsize)),
                    cls=cls,
                    sim_index=b,
                )
            )
        return PerceivedState(detections, self._box_poses)


class PositionMemory:
    """Counts kept by perceived position: the failure memory and the rake counter.

    Nothing may key on the simulator's part index across selections, so a record is
    ``[x, y, count]`` at the place the part was last perceived (MS7c design section 6).
    """

    def __init__(self):
        self.records: list[list] = []

    def assign(self, detections: Sequence[Detection]) -> dict[int, int]:
        """Each record's nearest detection within ASSOCIATION_RADIUS_M (xy) takes over its
        count, and the record moves to that detection. When two records pick one detection,
        the nearer wins and the other stays where it is. Returns ``{sim_index: count}``."""
        taken: dict[int, tuple[float, list]] = {}  # detection -> (distance, record)
        for rec in self.records:
            gaps = [
                (math.hypot(d.centre[0] - rec[0], d.centre[1] - rec[1]), k)
                for k, d in enumerate(detections)
            ]
            dist, k = min(gaps, default=(math.inf, -1))
            if dist <= ASSOCIATION_RADIUS_M and (k not in taken or dist < taken[k][0]):
                taken[k] = (dist, rec)
        out = {}
        for k, (_, rec) in taken.items():
            d = detections[k]
            rec[0], rec[1] = d.centre[0], d.centre[1]
            out[d.sim_index] = rec[2]
        return out

    def book(self, xy, n: int = 1) -> None:
        """Add ``n`` to the nearest record within ASSOCIATION_RADIUS_M of ``xy``, or start one."""
        gap = lambda r: math.hypot(r[0] - xy[0], r[1] - xy[1])
        rec = min(self.records, key=gap, default=None)
        if rec is not None and gap(rec) <= ASSOCIATION_RADIUS_M:
            rec[2] += n
        else:
            self.records.append([float(xy[0]), float(xy[1]), n])

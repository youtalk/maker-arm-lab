"""The arithmetic of the perception read: detections against a truth layout, turned into the
knob table (position bias and noise, heading, size, box, miss, class) and judged against the
pre-registered limits (MS5 revision section 7.6). Numpy and the stdlib only."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maker_arm_lab.core import parts as parts_mod
from maker_arm_lab.core import perception as perc
from maker_arm_lab.real.layouts import Layout, TruthBox
from maker_arm_lab.real.rig import label_boxes

MATCH_RADIUS_M = 0.025

LIMITS = {
    "pos_bias_mm": (6.5, 6.5, 6.5),
    "pos_noise_mm": (4.9, 4.9, 6.5),
    "heading_deg": 16.0,
    "size_mm": 4.0,
    "box_mm": 12.0,
    "box_deg": 8.0,
    "miss": 0.20,
    "class": 0.12,
}
NOMINALS = {
    "pos_bias_mm": tuple(1000 * v for v in perc.POS_BIAS_SIGMA_M),
    "pos_noise_mm": tuple(1000 * v for v in perc.POS_NOISE_SIGMA_M),
    "heading_deg": math.degrees(perc.HEADING_SIGMA_RAD),
    "size_mm": 1000 * perc.SIZE_SIGMA_M,
    "box_mm": 1000 * perc.BOX_SIGMA_M,
    "box_deg": math.degrees(perc.BOX_YAW_SIGMA_RAD),
    "miss": perc.MISS_PROBABILITY,
    "class": perc.CLASS_ERROR_PROBABILITY,
}
AXES = "xyz"


@dataclass
class FrameResult:
    layout: Layout
    detections: list[dict]
    boxes: list[list[float]]


def fold_angle(err: float, period: float) -> float:
    """``err`` folded into [-period/2, period/2)."""
    return (err + period / 2) % period - period / 2


def heading_period(name: str) -> float:
    """A quarter turn for a square footprint, a half turn otherwise."""
    size = parts_mod.by_name([name])[0].size
    return math.pi / 2 if math.isclose(size[0], size[1], rel_tol=1e-3) else math.pi


def match_parts(detections, truth, radius_m: float = MATCH_RADIUS_M):
    """(pairs, missed truth, extra detections), greedy by nearest xy first. A merged detection
    of two touching parts pairs with one and leaves the other missed."""
    cands = sorted(
        (math.hypot(d["centre"][0] - t.x, d["centre"][1] - t.y), i, j)
        for i, d in enumerate(detections)
        for j, t in enumerate(truth)
    )
    used_d, used_t, pairs = set(), set(), []
    for dist, i, j in cands:
        if dist > radius_m:
            break
        if i in used_d or j in used_t:
            continue
        used_d.add(i)
        used_t.add(j)
        pairs.append((detections[i], truth[j]))
    missed = [t for j, t in enumerate(truth) if j not in used_t]
    extra = [d for i, d in enumerate(detections) if i not in used_d]
    return pairs, missed, extra


def match_boxes(poses, truth: list[TruthBox]):
    """Detected (x, y, yaw) poses paired with the truth boxes by label order."""
    return list(zip(label_boxes(poses), truth, strict=True))


def _yaw(quat) -> float:
    w, _, _, z = quat
    return 2.0 * math.atan2(z, w)


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def knob_table(frames: list[FrameResult]) -> dict:
    """The knob rows over the sparse and cluster frames. A heap frame (NaN truth positions)
    reports only its detection count against the poured count, under ``counts.heap``. A row
    with no sample reads NaN."""
    pos, head, size, cls_err = [], [], [], []
    box_xy, box_yaw = [], []
    n_parts = n_missed = n_extra = n_frames = 0
    heap = {"detected": 0, "poured": 0}
    for f in frames:
        if f.layout.kind == "heap":
            heap["detected"] += len(f.detections)
            heap["poured"] += len(f.layout.parts)
            continue
        n_frames += 1
        pairs, missed, extra = match_parts(f.detections, f.layout.parts)
        n_parts += len(f.layout.parts)
        n_missed += len(missed)
        n_extra += len(extra)
        for d, t in pairs:
            pos.append([1000 * (d["centre"][i] - (t.x, t.y, t.z)[i]) for i in range(3)])
            head.append(math.degrees(fold_angle(_yaw(d["quat"]) - t.yaw, heading_period(t.name))))
            s = parts_mod.by_name([t.name])[0].size
            truth_size = (*sorted(s[:2], reverse=True), s[2])
            size.extend(1000 * (a - b) for a, b in zip(d["size"], truth_size, strict=True))
            cls_err.append(d["cls"] != t.cls)
        try:
            boxes = match_boxes(f.boxes, f.layout.boxes)
        except ValueError:
            continue  # not four boxes in label order: no box sample
        for (x, y, yaw), t in boxes:
            box_xy.append(1000 * math.hypot(x - t.x, y - t.y))
            box_yaw.append(math.degrees(fold_angle(yaw - t.yaw, math.pi)))
    pos = np.array(pos).reshape(-1, 3)
    nan3 = [math.nan] * 3
    return {
        "pos_bias_mm": pos.mean(axis=0).tolist() if len(pos) else nan3,
        "pos_noise_mm": pos.std(axis=0).tolist() if len(pos) else nan3,
        "heading_deg": _rms(head),
        "size_mm": _rms(size),
        "box_mm": _rms(box_xy),
        "box_deg": _rms(box_yaw),
        "miss": n_missed / n_parts if n_parts else math.nan,
        "class": float(np.mean(cls_err)) if cls_err else math.nan,
        "counts": {
            "frames": n_frames,
            "parts": n_parts,
            "matched": n_parts - n_missed,
            "extra": n_extra,
            "heap": heap,
        },
    }


def verdict(table: dict) -> tuple[bool, list[str]]:
    """(all rows within their limit, the rows over it). A NaN row has no sample and is not over."""
    over = []
    for row, limit in LIMITS.items():
        if isinstance(limit, tuple):
            over += [f"{row}.{AXES[i]}" for i, lim in enumerate(limit) if abs(table[row][i]) > lim]
        elif abs(table[row]) > limit:
            over.append(row)
    return not over, over

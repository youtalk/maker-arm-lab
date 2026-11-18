"""What the iPad shows: its screen geometry, the wrist camera's checkerboard and the truth
layouts of parts. Every image is drawn one to one, so a 100 mm scale bar on it checks with a
ruler. A layout is the truth the sensing reads compare with, in the base frame."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maker_arm_lab.core import desk_boxes
from maker_arm_lab.core import parts as parts_mod
from maker_arm_lab.real.camera_model import Pose
from maker_arm_lab.real.rig import RigConfig, label_boxes


@dataclass(frozen=True)
class Screen:
    """A screen of ``width_px`` x ``height_px`` at ``ppi``. Its frame has the origin at the
    screen centre, x along +u (the long side) and y along -v, in metres. Pixel coordinates use
    the lab convention: the centre of pixel (0, 0) is (0.5, 0.5)."""

    width_px: int
    height_px: int
    ppi: float

    @property
    def mm_per_px(self) -> float:
        return 25.4 / self.ppi

    def to_px(self, xy_m) -> tuple[float, float]:
        s = 1000.0 / self.mm_per_px
        return (self.width_px / 2 + xy_m[0] * s, self.height_px / 2 - xy_m[1] * s)

    def from_px(self, u: float, v: float) -> tuple[float, float]:
        s = self.mm_per_px / 1000.0
        return ((u - self.width_px / 2) * s, (self.height_px / 2 - v) * s)


IPAD_AIR_11 = Screen(2360, 1640, 264.0)


def checker_png(
    cols: int = 9, rows: int = 6, square_px: int = 208, screen: Screen = IPAD_AIR_11
) -> Image.Image:
    """A checkerboard of ``cols`` x ``rows`` inner corners (20.01 mm squares at 208 px on the
    iPad), centred on a white screen, with a 100 mm scale bar under it."""
    img = Image.new("RGB", (screen.width_px, screen.height_px), "white")
    draw = ImageDraw.Draw(img)
    w, h = (cols + 1) * square_px, (rows + 1) * square_px
    left, top = (screen.width_px - w) // 2, (screen.height_px - h) // 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                x, y = left + c * square_px, top + r * square_px
                draw.rectangle([x, y, x + square_px - 1, y + square_px - 1], fill="black")
    bar = round(100.0 / screen.mm_per_px)
    x0, y = (screen.width_px - bar) // 2, (top + h + screen.height_px) // 2
    draw.rectangle([x0, y - 3, x0 + bar - 1, y + 2], fill="black")
    for x in (x0, x0 + bar - 1):
        draw.rectangle([x - 1, y - 12, x + 1, y + 11], fill="black")
    return img


SPARSE_MIN_GAP_M = 0.035
SPARSE_MIN_FOOTPRINT_GAP_M = 0.010
CLUSTER_GAP_M = 0.001
TRAY_EDGE_MARGIN_M = 0.005
TEMPLATE_GREY = (90, 90, 90)  # INITIAL: task 9 measures what reads best on the iPad
DRAW_TRIES = 5000
KINDS = ("sparse", "cluster", "heap")


@dataclass(frozen=True)
class TruthPart:
    name: str
    cls: str
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class TruthBox:
    x: float
    y: float
    yaw: float


@dataclass
class Layout:
    name: str
    kind: str
    parts: list[TruthPart]
    boxes: list[TruthBox]
    frame: str = "nominal"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)  # NaN (a heap's positions) round-trips

    @classmethod
    def from_json(cls, path) -> Layout:
        d = json.loads(Path(path).read_text())
        return cls(
            d["name"],
            d["kind"],
            [TruthPart(**p) for p in d["parts"]],
            [TruthBox(**b) for b in d["boxes"]],
            d.get("frame", "nominal"),
        )


def nominal_screen_pose(rig: RigConfig) -> Pose:
    """The screen frame in the base frame, the iPad at the tray spot: screen x along base +y,
    screen y along base -x, screen z up."""
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return Pose(rot, np.array([rig.tray_centre[0], rig.tray_centre[1], rig.ipad_z]))


def _size(name: str) -> tuple[float, float, float]:
    return parts_mod.by_name([name])[0].size


def _corners(x: float, y: float, yaw: float, size) -> np.ndarray:
    """The footprint's four corners (4, 2) in the base frame; body x is the long side."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2, size[1] / 2
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return local @ np.array([[c, s], [-s, c]]) + (x, y)


def _poly_gap(a: np.ndarray, b: np.ndarray) -> float:
    """The distance between two convex rectangles, 0 when they overlap."""
    axes = []
    for poly in (a, b):
        for i in range(2):
            e = poly[i + 1] - poly[i]
            axes.append(np.array([-e[1], e[0]]) / np.linalg.norm(e))
    spans = [((a @ ax).min(), (a @ ax).max(), (b @ ax).min(), (b @ ax).max()) for ax in axes]
    if all(amax >= bmin and bmax >= amin for amin, amax, bmin, bmax in spans):
        return 0.0

    def seg(p, q, pts):
        d = q - p
        t = np.clip(((pts - p) @ d) / (d @ d), 0.0, 1.0)
        return np.linalg.norm(pts - (p + t[:, None] * d), axis=1).min()

    return min(
        seg(poly[i], poly[(i + 1) % 4], other) for poly, other in ((a, b), (b, a)) for i in range(4)
    )


def _part_corners(p: TruthPart) -> np.ndarray:
    return _corners(p.x, p.y, p.yaw, _size(p.name))


def footprint_gap(a: TruthPart, b: TruthPart) -> float:
    """The distance between two parts' footprints, m; 0 when they overlap."""
    return _poly_gap(_part_corners(a), _part_corners(b))


def footprint_inside_tray(p: TruthPart, rig: RigConfig) -> bool:
    """The footprint lies inside the tray interior by ``TRAY_EDGE_MARGIN_M``."""
    return _inside(p.name, p.x, p.y, p.yaw, rig)


def _truth(name: str, x: float, y: float, yaw: float, rig: RigConfig, on_screen: bool):
    part = parts_mod.by_name([name])[0]
    z = rig.ipad_z + part.size[2] / 2 if on_screen else float("nan")
    return TruthPart(name, parts_mod.part_class(part), x, y, z, yaw)


def _inside(name, x, y, yaw, rig) -> bool:
    d = _corners(x, y, yaw, _size(name)) - rig.tray_centre
    return bool((np.abs(d) <= np.asarray(rig.tray_interior) / 2 - TRAY_EDGE_MARGIN_M).all())


def draw_layout(
    rng: random.Random, kind: str, names: Sequence[str], rig: RigConfig, name: str
) -> Layout:
    """``names`` placed in the tray: far apart (sparse), touching (cluster) or poured by the
    operator (heap, NaN positions). Raises RuntimeError if the parts do not fit."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    boxes = [TruthBox(*b) for b in label_boxes(desk_boxes.sample_boxes(rng, 4))]
    if kind == "heap":
        nan = float("nan")
        return Layout(name, kind, [_truth(n, nan, nan, nan, rig, False) for n in names], boxes)
    cx, cy = rig.tray_centre
    hx, hy = (v / 2 - TRAY_EDGE_MARGIN_M for v in rig.tray_interior)
    placed: list[TruthPart] = []
    for n in names:
        for _ in range(DRAW_TRIES):
            yaw = rng.uniform(0.0, math.pi)
            if kind == "sparse" or not placed:
                x, y = rng.uniform(cx - hx, cx + hx), rng.uniform(cy - hy, cy + hy)
            else:
                anchor = rng.choice(placed)
                x, y = _touch(rng, anchor, n, yaw)
            if not _inside(n, x, y, yaw, rig):
                continue
            cand = _truth(n, x, y, yaw, rig, True)
            if kind == "sparse":
                ok = all(
                    math.dist((x, y), (q.x, q.y)) >= SPARSE_MIN_GAP_M
                    and footprint_gap(cand, q) >= SPARSE_MIN_FOOTPRINT_GAP_M
                    for q in placed
                )
            else:
                ok = all(footprint_gap(cand, q) >= CLUSTER_GAP_M - 1e-9 for q in placed)
            if ok:
                placed.append(cand)
                break
        else:
            raise RuntimeError(f"no room for {n} in {kind} layout {name}")
    return Layout(name, kind, placed, boxes)


def _touch(rng, anchor: TruthPart, name: str, yaw: float) -> tuple[float, float]:
    """The centre where ``name`` at ``yaw`` lies ``CLUSTER_GAP_M`` from ``anchor``, found by
    bisection along a random direction from the anchor's centre."""
    ang = rng.uniform(0.0, 2 * math.pi)
    d = np.array([math.cos(ang), math.sin(ang)])
    size, other = _size(name), _part_corners(anchor)
    lo, hi = 0.0, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        gap = _poly_gap(_corners(anchor.x + mid * d[0], anchor.y + mid * d[1], yaw, size), other)
        lo, hi = (mid, hi) if gap < CLUSTER_GAP_M else (lo, mid)
    return anchor.x + hi * d[0], anchor.y + hi * d[1]


def place_layout(layout: Layout, nominal: Pose, measured: Pose) -> Layout:
    """Every part moved by the screen's measured pose instead of its nominal one. The boxes
    stay: the operator places them with a ruler in the base frame."""
    rel = measured.rot @ nominal.rot.T
    dyaw = math.atan2(rel[1, 0], rel[0, 0])
    moved = []
    for p in layout.parts:
        q = measured.rot @ nominal.to_camera([p.x, p.y, p.z]) + measured.pos
        moved.append(TruthPart(p.name, p.cls, float(q[0]), float(q[1]), float(q[2]), p.yaw + dyaw))
    return Layout(layout.name, layout.kind, moved, layout.boxes, "measured")


def _font(size: int):
    return ImageFont.load_default(size=size)


def template_png(layout: Layout, rig: RigConfig, screen: Screen = IPAD_AIR_11) -> Image.Image:
    """The layout on the screen at one to one: the tray outline, a 100 mm scale bar with 10 mm
    ticks and, for each part, its footprint outline, index, name and a heading arrow."""
    img = Image.new("RGB", (screen.width_px, screen.height_px), TEMPLATE_GREY)
    draw = ImageDraw.Draw(img)
    pose = nominal_screen_pose(rig)

    def px(x, y):
        s = pose.to_camera([x, y, rig.ipad_z])
        return screen.to_px(s[:2])

    cx, cy = rig.tray_centre
    hx, hy = (v / 2 for v in rig.tray_interior)
    tray = [px(cx + sx * hx, cy + sy * hy) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    draw.line(tray + [tray[0]], fill="white", width=2)
    bar = 100.0 / screen.mm_per_px
    x0, y = (screen.width_px - bar) / 2, screen.height_px - 70
    draw.line([(x0, y), (x0 + bar, y)], fill="white", width=4)
    for k in range(11):
        draw.line([(x0 + k * bar / 10, y - 12), (x0 + k * bar / 10, y + 12)], fill="white", width=2)
    draw.text((x0 + bar + 20, y), "100 mm", fill="white", font=_font(28), anchor="lm")
    for i, p in enumerate(layout.parts):
        if not math.isfinite(p.x):
            continue
        pts = [px(*c) for c in _part_corners(p)]
        draw.line(pts + [pts[0]], fill="white", width=3)
        c0 = px(p.x, p.y)
        c1 = px(
            p.x + 0.4 * _size(p.name)[0] * math.cos(p.yaw),
            p.y + 0.4 * _size(p.name)[0] * math.sin(p.yaw),
        )
        draw.line([c0, c1], fill="white", width=3)
        ang = math.atan2(c1[1] - c0[1], c1[0] - c0[0])
        head = [
            (c1[0] - 14 * math.cos(ang + s), c1[1] - 14 * math.sin(ang + s)) for s in (-0.5, 0.5)
        ]
        draw.line([head[0], c1, head[1]], fill="white", width=3)
        draw.text((c0[0] + 8, c0[1] + 12), f"{i} {p.name}", fill="white", font=_font(24))
    return img

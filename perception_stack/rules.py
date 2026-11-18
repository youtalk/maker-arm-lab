"""The perception worker's geometry (MS5 design section 4.2): numpy and the stdlib only.

Depth to world points, blobs above the tray floor, the minimum-area rectangle, a part
detection, the class rule and the box rule. No model and no camera driver lives here, so the
lab's CPU suite tests it (``tests/test_perception_rules.py`` loads it by path).

Cameras are the wire's camera block: ``k`` is row-major 3x3 in the lab convention (the centre
of pixel (0, 0) is (0.5, 0.5)), ``rot`` and ``pos`` are the world from the ROS optical frame
(x right, y down, z forward), and depth is metres along z.
"""

import math
from collections import deque
from collections.abc import Sequence

import numpy as np

PART_CLASSES = ("slope", "tall", "narrow", "wide")
"""``perception_protocol.PART_CLASSES``, pinned by tests/test_perception_rules.py."""

MIN_PART_HEIGHT_M = 0.003  # INITIAL: plan task 9 (the sparse read) measures it
MIN_PART_CELLS = 3  # INITIAL: plan task 9 (the sparse read) measures it
BRICK_HEIGHT_M = 0.0096  # INITIAL: plan task 9 (the sparse read) measures it
TALL_MIN_M = 0.0116  # INITIAL: plan task 9 (the sparse read) measures it
NARROW_MAX_M = 0.012  # INITIAL: plan task 9 (the sparse read) measures it
SLOPE_MIN = 0.5  # INITIAL: plan task 9 (the sparse read) measures it
BOX_MIN_HEIGHT_M = 0.02  # INITIAL: plan task 9 (the sparse read) measures it
BOX_SIZE_TOL_M = 0.015  # INITIAL: plan task 9 (the sparse read) measures it
BOX_DARK_MAX = 80  # INITIAL: plan task 9 (the sparse read) measures it
BOX_WALLS_M = 0.010  # two simulator walls of 5 mm (layout.TRAY_WALL)


def _intrinsics(k):
    k = np.asarray(k, dtype=np.float64).reshape(3, 3)
    return k[0, 0], k[1, 1], k[0, 2], k[1, 2]


def deproject(depth_mm: np.ndarray, k, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 world points of a (H, W) depth in millimetres; NaN where it is 0."""
    fx, fy, cx, cy = _intrinsics(k)
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float64) / 1000.0
    u = np.arange(w)[None, :] + 0.5
    v = np.arange(h)[:, None] + 0.5
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    world = cam @ np.asarray(rot, dtype=np.float64).T + np.asarray(pos, dtype=np.float64)
    world[depth_mm == 0] = np.nan
    return world.astype(np.float32)


def project(points: np.ndarray, k, rot, pos) -> np.ndarray:
    """(N, 2) pixels (u, v) of (N, 3) world points."""
    fx, fy, cx, cy = _intrinsics(k)
    cam = (np.asarray(points, dtype=np.float64) - np.asarray(pos)) @ np.asarray(rot)
    return np.stack([cx + fx * cam[:, 0] / cam[:, 2], cy + fy * cam[:, 1] / cam[:, 2]], axis=-1)


def blobs_above(
    world: np.ndarray,
    floor_z: float,
    region: tuple[float, float, float, float],
    min_height: float = MIN_PART_HEIGHT_M,
    min_cells: int = MIN_PART_CELLS,
) -> list[np.ndarray]:
    """The 4-connected groups of cells more than ``min_height`` above ``floor_z`` inside the xy
    ``region`` (x_min, x_max, y_min, y_max), each as an (n, 2) array of (row, col)."""
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    x0, x1, y0, y1 = region
    with np.errstate(invalid="ignore"):
        mask = (z - floor_z > min_height) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    h, w = mask.shape
    todo = {int(i) for i in np.flatnonzero(mask)}
    blobs = []
    # ponytail: Python BFS on a 256x192 grid, about 30 ms; a vectorized labeling if the perceive
    # bar binds
    while todo:
        seed = todo.pop()
        cells, queue = [seed], deque([seed])
        while queue:
            r, c = divmod(queue.popleft(), w)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (n := nr * w + nc) in todo:
                    todo.remove(n)
                    cells.append(n)
                    queue.append(n)
        if len(cells) >= min_cells:
            idx = np.array(cells)
            blobs.append(np.stack([idx // w, idx % w], axis=-1))
    return blobs


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of (N, 2) points, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted({(float(a), float(b)) for a, b in points})
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while (
                len(out) >= 2
                and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])
                <= 0
            ):
                out.pop()
            out.append(p)
        return out[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def min_area_rect(xy: np.ndarray) -> tuple[float, float, float, float, float]:
    """(centre x, centre y, yaw of the long side in (-pi/2, pi/2], length, width), length >=
    width, of the smallest rectangle around (N, 2) points."""
    hull = _hull(np.asarray(xy, dtype=np.float64))
    best = None
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        if not d.any():  # a lone point
            d = np.array([1.0, 0.0])
        a = math.atan2(d[1], d[0])
        c, s = math.cos(a), math.sin(a)
        u, v = hull @ [c, s], hull @ [-s, c]
        e1, e2 = np.ptp(u), np.ptp(v)
        if best is None or e1 * e2 < best[0] - 1e-15:
            best = (e1 * e2, a, c, s, (u.min() + u.max()) / 2, (v.min() + v.max()) / 2, e1, e2)
    _, a, c, s, mu, mv, e1, e2 = best
    yaw = a if e1 >= e2 else a + math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    if yaw <= -math.pi / 2 + 1e-12:  # the half-open end: (-pi/2, pi/2]
        yaw = math.pi / 2
    return (mu * c - mv * s, mu * s + mv * c, yaw, max(e1, e2), min(e1, e2))


def _height(heights: np.ndarray) -> float:
    valid = heights[np.isfinite(heights)]
    return float(np.percentile(valid, 90)) if valid.size else BRICK_HEIGHT_M


def slope_score(along: np.ndarray, heights: np.ndarray) -> float:
    """|least-squares slope of height against the long-axis coordinate| times the length over
    the height: about 1 for a ramp from the floor to the part's height, 0 for a flat top."""
    ok = np.isfinite(heights)
    if ok.sum() < 2 or np.ptp(along[ok]) == 0:
        return 0.0
    slope = np.polyfit(along[ok], heights[ok], 1)[0]
    return float(abs(slope) * np.ptp(along) / _height(heights))


def classify(size, slope_score: float) -> str:
    """One of ``PART_CLASSES`` from (length, width, height) and the slope score."""
    if slope_score >= SLOPE_MIN:
        return "slope"
    if size[2] >= TALL_MIN_M:
        return "tall"
    return "narrow" if size[1] <= NARROW_MAX_M else "wide"


def part_detection(xy: np.ndarray, heights: np.ndarray, floor_z: float) -> dict:
    """A wire detection from a part's mask: (N, 2) world xy of its cells and (N,) heights above
    ``floor_z`` (NaN where the depth is invalid). The height is clamped to at least
    ``MIN_PART_HEIGHT_M`` so a noisy mask never gives a flat part."""
    cx, cy, yaw, length, width = min_area_rect(xy)
    height = max(_height(heights), MIN_PART_HEIGHT_M)
    along = (xy[:, 0] - cx) * math.cos(yaw) + (xy[:, 1] - cy) * math.sin(yaw)
    size = (length, width, height)
    return {
        "centre": [cx, cy, floor_z + height / 2],
        "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        "size": list(size),
        "cls": classify(size, slope_score(along, heights)),
        "score": float(np.isfinite(heights).mean()),
    }


def box_detections(
    world: np.ndarray,
    rgb_luma: np.ndarray,
    desk_z: float,
    bands: Sequence[tuple[float, float, float, float]],
    interior: tuple[float, float, float],
) -> list[list[float]]:
    """(x, y, yaw) of each box in the side ``bands`` (x_min, x_max, y_min, y_max). A blob more
    than ``BOX_MIN_HEIGHT_M`` above the desk is a box when its rectangle's outer size is the
    ``interior`` plus two walls within ``BOX_SIZE_TOL_M`` and the mean luma of the frame inside
    it is at most ``BOX_DARK_MAX``. Looking only at the bands keeps a dark part in the tray
    from reading as a box."""
    want = sorted((interior[0] + BOX_WALLS_M, interior[1] + BOX_WALLS_M), reverse=True)
    out = []
    for band in bands:
        for cells in blobs_above(world, desk_z, band, BOX_MIN_HEIGHT_M):
            xy = world[cells[:, 0], cells[:, 1], :2]
            cx, cy, yaw, length, width = min_area_rect(xy)
            if max(abs(length - want[0]), abs(width - want[1])) > BOX_SIZE_TOL_M:
                continue
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = world[..., 0] - cx, world[..., 1] - cy
            with np.errstate(invalid="ignore"):
                inside = (np.abs(dx * c + dy * s) <= length / 2) & (
                    np.abs(dy * c - dx * s) <= width / 2
                )
            if inside.any() and rgb_luma[inside].mean() <= BOX_DARK_MAX:
                out.append([cx, cy, yaw])
    return out

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
BOX_WALL_M = (
    0.005  # INITIAL: plan task 9 (the sparse read) measures it; the real boxes have 1-2 mm walls
)
BOX_DEPTH_MIN_CELLS = 3  # INITIAL: plan task 9 (the sparse read) measures it
"""Valid depth cells a box component needs before its depth rim is checked."""
PROMPT_GRID_M = 0.008  # INITIAL: plan task 9 (the cluster read) measures it
"""The xy grid of SAM prompts over a depth blob: one prompt per square it covers."""
PART_MAX_FOOTPRINT_M = (0.0958, 0.0318)
"""The longest and the widest footprint side of a base-class part the layouts accept
(``core.parts.by_name``: brick_1x12 and brick_4x4), pinned by tests/test_perception_rules.py."""
MASK_FOOTPRINT_MARGIN_M = 0.010  # INITIAL: plan task 9 (the cluster read) measures it
MASK_IOU_MAX = 0.5  # INITIAL: plan task 9 (the cluster read) measures it
"""Two masks overlapping by more than this IoU are one part."""


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


def _plane_xy(u, v, k, rot, pos, plane_z) -> np.ndarray:
    """(..., 2) world xy where the rays through pixel positions (u, v) (lab convention, a pixel's
    centre is at +0.5) meet the plane z = ``plane_z``."""
    fx, fy, cx, cy = _intrinsics(k)
    ray = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(np.shape(u))], axis=-1)
    ray = ray @ np.asarray(rot, dtype=np.float64).T
    pos = np.asarray(pos, dtype=np.float64)
    return pos[:2] + ray[..., :2] * ((plane_z - pos[2]) / ray[..., 2:3])


def mask_xy(pixels: np.ndarray, depth_mm_at_pixels: np.ndarray, k, rot, pos, plane_z: float):
    """(N, 2) world xy of the (row, col) ``pixels`` of a mask: deprojected with the depth where
    it is valid, and the ray meeting the plane z = ``plane_z`` (the floor plus the part height)
    where it is 0."""
    pixels = np.asarray(pixels)
    u, v = pixels[:, 1] + 0.5, pixels[:, 0] + 0.5
    on_plane = _plane_xy(u, v, k, rot, pos, plane_z)
    d = np.asarray(depth_mm_at_pixels, dtype=np.float64)
    fx, fy, cx, cy = _intrinsics(k)
    z = d / 1000.0
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    seen = (cam @ np.asarray(rot, dtype=np.float64).T + np.asarray(pos, dtype=np.float64))[:, :2]
    return np.where((d > 0)[:, None], seen, on_plane)


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


def grid_prompts(points: np.ndarray, cell_m: float = PROMPT_GRID_M) -> np.ndarray:
    """(m, 3) prompt points of a blob's (n, 3) world points: in each ``cell_m`` square of the xy
    grid that the blob covers, the blob point nearest the mean of its points in that square. A
    blob point stands on a part, where the mean of two touching parts can fall in their gap."""
    keys = np.floor(points[:, :2] / cell_m).astype(np.int64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.ravel()
    out = []
    for g in range(group.max() + 1):
        pts = points[group == g]
        out.append(pts[np.argmin(((pts[:, :2] - pts[:, :2].mean(axis=0)) ** 2).sum(axis=1))])
    return np.array(out)


def distinct_masks(masks: np.ndarray, scores: Sequence[float], iou_max: float = MASK_IOU_MAX):
    """Indices of the (N, H, W) ``masks`` that are distinct parts: in order of score (on a tie
    the smaller mask, so a lone part outranks a merged pair), a mask is kept unless its IoU with
    a kept mask is over ``iou_max``. Empty masks are dropped."""
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    boxes = []
    for m in masks:
        rows, cols = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        boxes.append((rows[0], rows[-1] + 1, cols[0], cols[-1] + 1) if len(rows) else None)
    kept: list[int] = []
    for i in sorted(range(len(masks)), key=lambda i: (-scores[i], areas[i])):
        if boxes[i] is None:
            continue
        for j in kept:
            r0, r1 = max(boxes[i][0], boxes[j][0]), min(boxes[i][1], boxes[j][1])
            c0, c1 = max(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            if r0 >= r1 or c0 >= c1:
                continue
            inter = np.count_nonzero(masks[i, r0:r1, c0:c1] & masks[j, r0:r1, c0:c1])
            if inter / (areas[i] + areas[j] - inter) > iou_max:
                break
        else:
            kept.append(i)
    return kept


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
    _, _, yaw, length, width = min_area_rect(xy)
    height = max(_height(heights), MIN_PART_HEIGHT_M)
    cx, cy = xy.mean(axis=0)  # the spec's centre is the mask centroid
    along = (xy[:, 0] - cx) * math.cos(yaw) + (xy[:, 1] - cy) * math.sin(yaw)
    size = (length, width, height)
    return {
        "centre": [cx, cy, floor_z + height / 2],
        "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        "size": list(size),
        "cls": classify(size, slope_score(along, heights)),
        "score": float(np.isfinite(heights).mean()),
    }


def _box_walls(xy_rim: np.ndarray, xy_desk: np.ndarray, nadir) -> tuple[float, ...]:
    """(centre x, centre y, yaw, length, width) of a box's outer walls from its dark blocks,
    each projected onto the rim plane (``xy_rim``) and onto the desk (``xy_desk``).

    The rim-plane fit gives the box frame. The camera also sees each outer wall that faces its
    ``nadir`` (its xy), and on the rim plane that wall falls short of the rim towards the nadir.
    So each edge that faces the nadir is taken from the desk projection, where the wall meets the
    desk, and each edge that faces away from the nadir from the rim projection, where the rim is
    the outline."""
    cx, cy, yaw, _, _ = min_area_rect(xy_rim)
    centre = np.array([cx, cy])
    axes = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    rim, desk = (xy_rim - centre) @ axes.T, (xy_desk - centre) @ axes.T
    to_nadir = (np.asarray(nadir, dtype=np.float64) - centre) @ axes.T
    hi = np.where(to_nadir > rim.max(axis=0), desk.max(axis=0), rim.max(axis=0))
    lo = np.where(to_nadir < rim.min(axis=0), desk.min(axis=0), rim.min(axis=0))
    centre = centre + ((hi + lo) / 2) @ axes
    length, width = hi - lo
    if width > length:  # the yaw is the long side's, in (-pi/2, pi/2] as min_area_rect's
        yaw = yaw - math.pi / 2 if yaw > 0 else yaw + math.pi / 2
        length, width = width, length
    return (float(centre[0]), float(centre[1]), yaw, float(length), float(width))


def box_detections(
    rgb_luma: np.ndarray,
    k,
    rot: np.ndarray,
    pos: np.ndarray,
    desk_z: float,
    bands: Sequence[tuple[float, float, float, float]],
    interior: tuple[float, float, float],
    depth_world: np.ndarray | None = None,
    wall_m: float = BOX_WALL_M,
    step: int = 4,
) -> list[list[float]]:
    """(x, y, yaw) of each box: the dark outline in the (H, W) luma frame, against the desk.

    The frame is block-averaged by ``step`` for speed. Dark blocks (at most ``BOX_DARK_MAX``)
    form 4-connected components inside a side band (x_min, x_max, y_min, y_max): a block is in
    the band when its projection onto the desk or onto the rim plane, z = ``desk_z`` +
    ``interior[2]``, is (a box's far rim projects onto the desk past the band). ``_box_walls``
    fits the outer walls. It is a box when its outer size is the interior plus two ``wall_m``
    walls within ``BOX_SIZE_TOL_M``. ``depth_world`` (the ``deproject`` output, optional) is only
    a check: with at least ``BOX_DEPTH_MIN_CELLS`` valid cells in the component, the highest
    must rise more than ``BOX_MIN_HEIGHT_M`` above the desk (the rim is a thin part of the
    component, so the median would read the box floor). With no valid depth the box stands on
    the outline alone. Looking only at the bands keeps a dark part in the tray from reading as
    a box."""
    h, w = (rgb_luma.shape[0] // step) * step, (rgb_luma.shape[1] // step) * step
    blocks = rgb_luma[:h, :w].astype(np.float64).reshape(h // step, step, w // step, step)
    dark = blocks.mean(axis=(1, 3)) <= BOX_DARK_MAX
    v, u = np.mgrid[0 : h // step, 0 : w // step] * step + step / 2
    desk = _plane_xy(u, v, k, rot, pos, desk_z)
    rim = _plane_xy(u, v, k, rot, pos, desk_z + interior[2])
    want = sorted((interior[0] + 2 * wall_m, interior[1] + 2 * wall_m), reverse=True)
    everywhere = (-np.inf, np.inf, -np.inf, np.inf)
    out = []
    for x0, x1, y0, y1 in bands:
        inside = [
            (xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= y0) & (xy[..., 1] <= y1)
            for xy in (desk, rim)
        ]
        flag = np.where(dark & (inside[0] | inside[1]), 1.0, np.nan)
        grid = np.concatenate([desk, flag[..., None]], axis=-1)
        for cells in blobs_above(grid, 0.0, everywhere, 0.5, min_cells=MIN_PART_CELLS):
            r, c = cells[:, 0], cells[:, 1]
            cx, cy, yaw, length, width = _box_walls(rim[r, c], desk[r, c], pos[:2])
            if max(abs(length - want[0]), abs(width - want[1])) > BOX_SIZE_TOL_M:
                continue
            if depth_world is not None:
                z = depth_world[(r * step + step // 2), (c * step + step // 2), 2]
                z = z[np.isfinite(z)]
                if len(z) >= BOX_DEPTH_MIN_CELLS and z.max() - desk_z <= BOX_MIN_HEIGHT_M:
                    continue
            out.append([cx, cy, yaw])
    return out

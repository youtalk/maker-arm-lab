"""Which bricks are stowed, read off a duck-typed env. Torch-only, no isaaclab.

These used to live in ``tasks/manager_based/lego_sort/mdp/terminations.py`` and
``observations.py``. They read the env through ``env.scene[...]``, ``env.scene.env_origins``,
``env.cfg`` and ``env.device`` only, so any backend that exposes those members -- the Isaac
``ManagerBasedRLEnv`` or ``maker_arm_lab.mj.MjLegoSortEnv`` -- is graded by the same code.
The mdp modules re-export them so the Isaac cfgs are unchanged.

``per_brick_stowed`` also reads ``env.num_envs``, and, when present, ``env.destination_bins``
and ``env.stow_half_extents`` (MuJoCo heap mode, MS6). Both are read with ``getattr(env, name,
None)``, so a backend without them keeps grading by color, as before. ``destination_bins``
grades against the one heap bin (``layout.heap_bin_center``), not the six-brick per-color
array; ``HeapOracle._stowed`` in ``core/heap_oracle.py`` reads the same function. A (N, K)
``destination_bins`` is one box per part among the MS7a desk boxes, read from
``env.desk_box_poses`` (``env_desk_boxes``).
"""

from functools import cache

import torch

from maker_arm_lab.core import checker, desk_boxes, layout


def brick_color_ids(num_bricks: int) -> torch.Tensor:
    """Target bin index per brick: brick i gets color i % 4 (BRICK_COLOR_ORDER)."""
    return torch.arange(num_bricks) % len(layout.BRICK_COLOR_ORDER)


@cache
def _bin_centers_cached(bl: layout.BinLayout, device: str) -> torch.Tensor:
    """``layout.bin_interior_centers(bl)`` moved to ``device``, cached per (layout, device).

    Constant for the env's lifetime; ``BinLayout`` is a frozen dataclass of hashables, so it
    is a valid cache key.
    """
    return layout.bin_interior_centers(bl).to(device)


@cache
def _bin_half_extents_cached(device: str) -> torch.Tensor:
    return layout.bin_half_extents().to(device)


def _env_bin_layout(env) -> layout.BinLayout:
    """The env's configured ``BinLayout``, or ``DEFAULT_BIN_LAYOUT``."""
    return getattr(env.cfg, "bin_layout", layout.DEFAULT_BIN_LAYOUT)


def env_bin_centers(env, device: str) -> torch.Tensor:
    """(4,3) bin interior centers, or (N,4,3) when the env carries one layout per env.

    ``env.bin_layouts`` (MuJoCo with per-episode draws) changes at every reset, so it is
    read live and not cached. Every other env has one layout for its lifetime.
    """
    layouts = getattr(env, "bin_layouts", None)
    if layouts is None:
        return _bin_centers_cached(_env_bin_layout(env), device)
    return torch.stack([layout.bin_interior_centers(bl) for bl in layouts]).to(device)


def _heap_bin_centers(bl: layout.BinLayout) -> torch.Tensor:
    """(4,3): the heap centre repeated once per bin slot.

    A heap env has one physical bin. Repeating ``layout.heap_bin_center`` across the (4,3)
    shape keeps a ``destination_bins`` index of 0-3 valid without a special case, and every
    index reads the same, correct heap centre (the MS6 z-centre fix).
    """
    return torch.tensor(layout.heap_bin_center(bl)).unsqueeze(0).repeat(4, 1)


@cache
def _heap_bin_centers_cached(bl: layout.BinLayout, device: str) -> torch.Tensor:
    return _heap_bin_centers(bl).to(device)


def env_heap_bin_centers(env, device: str) -> torch.Tensor:
    """(4,3) heap centers, or (N,4,3) when the env carries one layout per env.

    The heap-mode companion to ``env_bin_centers``: every returned row is
    ``layout.heap_bin_center(bl)`` (see ``_heap_bin_centers``), not the six-brick
    per-color array.
    """
    layouts = getattr(env, "bin_layouts", None)
    if layouts is None:
        return _heap_bin_centers_cached(_env_bin_layout(env), device)
    return torch.stack([_heap_bin_centers(bl) for bl in layouts]).to(device)


@cache
def _brick_color_ids_cached(num_bricks: int, device: str) -> torch.Tensor:
    return brick_color_ids(num_bricks).to(device)


def _brick_states(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,B,3) positions relative to each env's origin and (N,B,3) linear velocities.

    The ``- env.scene.env_origins`` subtraction is what makes the positions comparable to
    ``layout.bin_interior_centers()``, which is expressed in one env's local frame.
    """
    pos, vel = [], []
    for index in range(env.cfg.num_bricks):
        brick = env.scene[f"brick_{index}"]
        pos.append(brick.data.root_pos_w - env.scene.env_origins)
        vel.append(brick.data.root_lin_vel_w)
    return torch.stack(pos, dim=1), torch.stack(vel, dim=1)


def env_desk_boxes(env, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,M,4) grader boxes (x, y, centre z, yaw) and their (N,M,3) half extents, from
    ``env.desk_box_poses`` (N,M,3), each box's interior ``env.desk_box_sizes`` (N,M,3), the MS7a
    box of ``desk_boxes.interior`` without it, and ``desk_boxes.stow_region``."""
    poses = torch.as_tensor(env.desk_box_poses, dtype=torch.float32)
    n, m = poses.shape[:2]
    sizes = getattr(env, "desk_box_sizes", [desk_boxes.interiors(m)] * n)
    regions = [[desk_boxes.stow_region(size) for size in row] for row in sizes]
    z = torch.tensor([[r[0] for r in row] for row in regions], dtype=torch.float32)
    half = torch.tensor([[r[1] for r in row] for row in regions], dtype=torch.float32)
    boxes = torch.cat((poses[..., :2], z.reshape(poses[..., :1].shape), poses[..., 2:]), -1)
    return boxes.to(device), half.reshape(poses.shape).to(device)


def parts_in_desk_boxes(env) -> torch.Tensor:
    """(N,K) bool: the part lies inside a desk box, any box, moving or not."""
    pos, _ = _brick_states(env)
    boxes, half = env_desk_boxes(env, str(env.device))
    return checker.in_boxes(pos, boxes, half).any(dim=-1)


def parts_resting_box(env, max_speed: float = 0.02) -> torch.Tensor:
    """(N,K) long: the desk box each part lies at rest in, its own or not, or -1."""
    pos, vel = _brick_states(env)
    boxes, half = env_desk_boxes(env, str(env.device))
    inside = checker.in_boxes(pos, boxes, half) & (vel.norm(dim=-1) <= max_speed).unsqueeze(-1)
    index = inside.long().argmax(dim=-1)
    return torch.where(inside.any(dim=-1), index, torch.full_like(index, -1))


def per_brick_stowed(env, max_speed: float = 0.02) -> torch.Tensor:
    """(N,B) bool: which bricks are at rest inside their target bin right now.

    ``max_speed`` is the grader's at-rest speed; ``math.inf`` asks only whether the brick is
    inside the bin, moving or not (the heap loop's cycle end, MS6).

    The target is the brick's color bin, or, when the env carries ``destination_bins`` (N,)
    (MuJoCo heap mode, MS6), the one heap bin (``layout.heap_bin_center``, the MS6 z-centre
    fix) for every part of the env. ``stow_half_extents`` (3,) on the env replaces the bin
    interior half extents; the heap's box is taller than the rim so a pile over it still
    counts (design H6). A (N, K) ``destination_bins`` grades part b of env i against its own
    desk box ``destination_bins[i, b]``, in that box's yawed frame (MS7a design, desk boxes).
    """
    pos, vel = _brick_states(env)
    device = str(env.device)
    destination = getattr(env, "destination_bins", None)
    if destination is not None and torch.as_tensor(destination).dim() == 2:
        boxes, half = env_desk_boxes(env, device)
        ids = torch.as_tensor(destination).to(device)
        return checker.parts_at_rest_in_own_boxes(pos, vel, boxes, half, ids, max_speed)
    if destination is None:
        centers = env_bin_centers(env, device)
        ids = _brick_color_ids_cached(env.cfg.num_bricks, device)
    else:
        centers = env_heap_bin_centers(env, device)
        destination = torch.as_tensor(destination).to(device)
        if destination.shape[0] != env.num_envs:
            raise ValueError(
                f"destination_bins has {destination.shape[0]} entries, "
                f"env.num_envs is {env.num_envs}"
            )
        if centers.dim() == 2:
            centers = centers.unsqueeze(0).expand(env.num_envs, -1, -1)
        ids = destination.long().unsqueeze(1).expand(-1, env.cfg.num_bricks)
    half = getattr(env, "stow_half_extents", None)
    half = _bin_half_extents_cached(device) if half is None else torch.as_tensor(half).to(device)
    return checker.bricks_at_rest_in_bins(pos, vel, centers, half, ids, max_speed=max_speed)


def all_bricks_stowed(env) -> torch.Tensor:
    """(N,) bool: episode success, every brick stowed."""
    return per_brick_stowed(env).all(dim=-1)

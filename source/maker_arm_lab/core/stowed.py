"""Which bricks are stowed, read off a duck-typed env. Torch-only, no isaaclab.

These used to live in ``tasks/manager_based/lego_sort/mdp/terminations.py`` and
``observations.py``. They read the env through ``env.scene[...]``, ``env.scene.env_origins``,
``env.cfg`` and ``env.device`` only, so any backend that exposes those members -- the Isaac
``ManagerBasedRLEnv`` or ``maker_arm_lab.mj.MjLegoSortEnv`` -- is graded by the same code.
The mdp modules re-export them so the Isaac cfgs are unchanged.

``per_brick_stowed`` also reads ``env.num_envs``, and, when present, ``env.destination_bins``
and ``env.stow_half_extents`` (MuJoCo heap mode, MS6). Both are read with ``getattr(env, name,
None)``, so a backend without them keeps grading by color, as before.
"""

from functools import cache

import torch

from maker_arm_lab.core import checker, layout


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


def per_brick_stowed(env) -> torch.Tensor:
    """(N,B) bool: which bricks are at rest inside their target bin right now.

    The target is the brick's color bin, or, when the env carries ``destination_bins`` (N,)
    (MuJoCo heap mode, MS6), that one bin for every part of the env. ``stow_half_extents`` (3,)
    on the env replaces the bin interior half extents; the heap's box is taller than the rim so
    a pile over it still counts (design H6).
    """
    pos, vel = _brick_states(env)
    device = str(env.device)
    centers = env_bin_centers(env, device)
    destination = getattr(env, "destination_bins", None)
    if destination is None:
        ids = _brick_color_ids_cached(env.cfg.num_bricks, device)
    else:
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
    return checker.bricks_at_rest_in_bins(pos, vel, centers, half, ids)


def all_bricks_stowed(env) -> torch.Tensor:
    """(N,) bool: episode success, every brick stowed."""
    return per_brick_stowed(env).all(dim=-1)

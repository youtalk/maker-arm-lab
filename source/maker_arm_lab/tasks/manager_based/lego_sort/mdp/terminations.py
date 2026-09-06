"""Success terminations: thin device/tensor adapters over core.checker.

``from __future__ import annotations`` + a ``TYPE_CHECKING`` guard defer the
``ManagerBasedRLEnv`` annotation, mirroring this module's sibling
``observations.py`` (a Task 7 review finding there, for the identical
reason): everything this module actually executes at import time --
``core.checker``, ``core.layout``, ``.observations`` -- is pure torch, so the
CPU test suite can load and exercise it without isaaclab installed.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from maker_arm_lab.core import checker, layout

from .observations import brick_color_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@cache
def _bin_centers_cached(bl: layout.BinLayout, device: str) -> torch.Tensor:
    """``layout.bin_interior_centers(bl)`` moved to ``device``, cached per (layout, device).

    Constant for the env's lifetime; computing and host->device-copying it on
    every step (the termination hot path) is wasted work that grows with M2's
    higher env counts. ``BinLayout`` is a frozen dataclass of hashables, so it
    is a valid cache key.
    """
    return layout.bin_interior_centers(bl).to(device)


@cache
def _bin_half_extents_cached(device: str) -> torch.Tensor:
    """``layout.bin_half_extents()`` moved to ``device``, cached per device."""
    return layout.bin_half_extents().to(device)


def _env_bin_layout(env) -> layout.BinLayout:
    """Get the env's configured ``BinLayout``, or fall back to ``DEFAULT_BIN_LAYOUT``."""
    return getattr(env.cfg, "bin_layout", layout.DEFAULT_BIN_LAYOUT)


@cache
def _brick_color_ids_cached(num_bricks: int, device: str) -> torch.Tensor:
    """``brick_color_ids(num_bricks)`` moved to ``device``, cached per (num_bricks, device)."""
    return brick_color_ids(num_bricks).to(device)


def _brick_states(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,B,3) world positions (relative to each env's own origin) and linear
    velocities of every ``brick_0..brick_{num_bricks-1}`` in the scene.

    The ``- env.scene.env_origins`` subtraction is what makes the returned
    positions comparable to ``layout.bin_interior_centers()``, which are
    expressed in a single env's local frame: without it, every env but env 0
    would be graded against env 0's bin locations instead of its own.
    """
    pos, vel = [], []
    for index in range(env.cfg.num_bricks):
        brick = env.scene[f"brick_{index}"]
        pos.append(brick.data.root_pos_w - env.scene.env_origins)
        vel.append(brick.data.root_lin_vel_w)
    return torch.stack(pos, dim=1), torch.stack(vel, dim=1)  # (N,B,3) x2


def per_brick_stowed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,B) bool: which bricks are at rest inside their correct-color bin right now.

    Thin adapter over ``core.checker.bricks_at_rest_in_bins`` -- see that
    function's docstring for the exact inside-AABB/at-rest predicate. Exported
    (via ``mdp.__init__``) both for ``all_bricks_stowed`` below and for M3's
    per-brick eval metrics.
    """
    pos, vel = _brick_states(env)
    device = str(env.device)
    return checker.bricks_at_rest_in_bins(
        pos,
        vel,
        _bin_centers_cached(_env_bin_layout(env), device),
        _bin_half_extents_cached(device),
        _brick_color_ids_cached(env.cfg.num_bricks, device),
    )


def all_bricks_stowed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N,) bool: episode success -- every brick in ``env`` is stowed (design spec 5.3).

    Wired as the ``success`` termination term in ``LegoSortEnvCfg.terminations``.
    """
    return per_brick_stowed(env).all(dim=-1)

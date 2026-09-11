"""Observation terms for LegoSort. Imports isaaclab; math stays trivial.

``brick_color_ids`` has no isaaclab dependency at all -- it is pure torch --
so its ``ManagerBasedRLEnv`` sibling's type annotation is deferred via
``from __future__ import annotations`` + a ``TYPE_CHECKING`` guard (the idiom
isaaclab_tasks' own ``franka_stack_events.py`` uses for the same reason) so
that importing this *module* does not require isaaclab to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from maker_arm_lab.core.stowed import brick_color_ids  # noqa: F401 -- re-exported

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def brick_poses_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, B*7) pose (pos relative to env origin, quat wxyz) of every brick.

    Despite the ``_w`` suffix -- kept only because it is Isaac Lab's standard
    world-frame naming convention (``root_pos_w``, ``root_quat_w``), and this
    exact name is the brief-mandated interface -- the returned positions are
    **not** world-frame: they are ``root_pos_w - env.scene.env_origins``, i.e.
    relative to each environment's own origin. Do not add ``env_origins`` to
    them again.
    """
    poses = []
    index = 0
    while f"brick_{index}" in env.scene.keys():  # noqa: SIM118 -- InteractiveScene has no plain `in`
        brick = env.scene[f"brick_{index}"]
        pos = brick.data.root_pos_w - env.scene.env_origins
        poses.append(torch.cat([pos, brick.data.root_quat_w], dim=-1))
        index += 1
    return torch.cat(poses, dim=-1)

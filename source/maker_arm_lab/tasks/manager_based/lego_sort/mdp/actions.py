"""Action terms whose joint targets pass through maker_arm_rs's clamp (spec section 4.3).

Both terms leave Isaac Lab's own processing untouched and clamp the *joint-space target*
right before it reaches the articulation, so what the recorder stores as `joint_targets`
(the dataset's action stream) is always post-clamp. The per-env `clamped` flag is exported
by `recorders.ClampRecorder` and aggregated by `scripts/generate_batch.py`.

`apply_actions` runs once per *physics substep* -- `cfg.decimation` times per control step,
via `ManagerBasedEnv`'s `for _ in range(self.cfg.decimation): action_manager.apply_action()`
loop -- while `process_actions` runs exactly once per control step, and `ClampRecorder` reads
`.clamped` exactly once, after the whole substep loop has finished. So `.clamped` must mean
"clamped on any substep this control step", not "clamped on the last substep it happened to
run": both terms reset it to `False` in `process_actions` and OR each substep's flags into it
in `apply_actions` via `clamp_targets_tensor(..., accumulate=True)`.
"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

from maker_arm_lab.core import profile as _profile


def _resolve_profile(cfg_profile: dict | None) -> dict:
    return cfg_profile if cfg_profile is not None else _profile.sim_profile()


class ClampedJointPositionAction(JointPositionAction):
    """Absolute joint-position action with the Rust clamp on the target."""

    cfg: ClampedJointPositionActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.profile = _resolve_profile(cfg.profile)
        # Control-step-scoped, not substep-scoped: see the module docstring. A no-op for this
        # term in practice (the target is substep-invariant here), but kept uniform with
        # ClampedDifferentialIKAction so both terms mean the same thing to ClampRecorder.
        self.clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        self.clamped.zero_()

    def apply_actions(self):
        gripper_open = self._env.action_manager.action[:, -1] >= 0.0
        targets = _profile.clamp_targets_tensor(
            self.processed_actions, gripper_open, self.profile, self.clamped, accumulate=True
        )
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


class ClampedDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Differential-IK action whose solved joint targets pass through the Rust clamp."""

    cfg: ClampedDifferentialIKActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.profile = _resolve_profile(cfg.profile)
        # Control-step-scoped, not substep-scoped: see the module docstring. Load-bearing here,
        # unlike in ClampedJointPositionAction -- `apply_actions` re-solves IK from the current
        # joint state on every physics substep, so the clamp outcome genuinely differs substep
        # to substep (e.g. an early substep over a limit, a later one converged back under it).
        # Overwriting instead of OR-ing would silently lose that early clamp event.
        self.clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions):
        super().process_actions(actions)
        self.clamped.zero_()

    def apply_actions(self):
        super().apply_actions()  # solves IK and writes joint_pos_target for self._joint_ids
        gripper_open = self._env.action_manager.action[:, -1] >= 0.0
        solved = self._asset.data.joint_pos_target[:, self._joint_ids]
        targets = _profile.clamp_targets_tensor(
            solved, gripper_open, self.profile, self.clamped, accumulate=True
        )
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction
    profile: dict | None = None
    """maker_arm_rs profile dict; None resolves to core.profile.sim_profile() at init."""


@configclass
class ClampedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = ClampedDifferentialIKAction
    profile: dict | None = None

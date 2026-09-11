"""Recorder terms for scripted-oracle data generation (spec section 2.3, Datagen-v0).

The dataset's action stream is the applied joint-position target -- what a
joint-position policy must output and what the real robot's CAN interface
consumes -- and it is POST-CLAMP, which the raw action is not. The raw 7-dim
action is still recorded (prebuilt "actions" term): its column 6 is the binary
gripper command, which the conversion combines with the 6 arm joint targets,
i.e. ``joint_targets[:, :6]`` of the 8-column buffer written by
``JointTargetRecorder`` (6 arm joints + 2 mimic-linked gripper jaws --
``obs/joint_pos_abs`` from ``JointStateRecorder`` is the same 8 columns).

Since Task D5 the datagen env is on the JOINT-POSITION action space (see
``datagen.LegoSortDatagenEnvCfg``), so the raw action's first six columns are
now the oracle's pre-clamp joint targets rather than the task-space IK-Rel
command they were when this stack was ported. Nothing here changed for it: both
clamped action terms expose the same ``clamped`` flag with the same semantics
(Task C6), and ``joint_targets`` was always read off the articulation.
"""

from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import (
    InitialStateRecorderCfg,
    PreStepActionsRecorderCfg,
    PreStepFlatPolicyObservationsRecorderCfg,
)
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from maker_arm_lab.core.oracle import PHASE_TO_INT  # noqa: F401 -- dataset schema pin, re-exported


class JointStateRecorder(RecorderTerm):
    """Absolute joint positions pre-step (policy obs `joint_pos` is default-relative)."""

    def record_pre_step(self):
        return "obs/joint_pos_abs", self._env.scene["robot"].data.joint_pos.clone()


class ImagesObsRecorder(RecorderTerm):
    """The vision cfg's `images` obs group (wrist_rgb / overhead_rgb, uint8 HWC).

    No-op when the env's ``cameras`` feature switch is off: ``LegoSortDatagenEnvCfg``
    then sets ``observations.images = None``, and ``ObservationManager._prepare_terms``
    skips a ``None`` group outright (``if group_cfg is None: continue``), so
    ``env.obs_buf`` never gets an ``"images"`` key at all -- not an empty one.
    ``DatagenRecorderManagerCfg`` installs this term unconditionally (``cameras``
    is a per-instance cfg field the class-level recorder cfg cannot see), and
    ``record_pre_step`` fires every control step regardless of
    ``dataset_export_mode``, so without this guard a ``cameras=False`` run --
    Task 8's release-envelope probe, and ``oracle_check.py``'s default -- crashed
    on the very first ``env.step()`` with ``KeyError: 'images'`` (found running
    this task; ``cameras=False`` was never exercised with the recorders before).
    """

    def record_pre_step(self):
        if "images" not in self._env.obs_buf:
            return None, None
        return "obs/images", self._env.obs_buf["images"]


class JointTargetRecorder(RecorderTerm):
    """Applied joint-position targets post-step: the dataset's action stream."""

    def record_post_step(self):
        return "joint_targets", self._env.scene["robot"].data.joint_pos_target.clone()


# PHASE_TO_INT lives in core.oracle (isaaclab-free) so the MuJoCo backend can stamp the
# same schema; re-exported here because this is where the recorders that write it live.


class CyclePhaseRecorder(RecorderTerm):
    """Per-step oracle cycle phase, from the OracleRig stash (int8; -1 means
    goals[i] and oracles[i] are both None -- see oracle_runtime.py's step()).
    With no rig at all the stash attribute is absent, not -1-filled: this term
    then returns (None, None) and the channel is missing from the shard
    entirely.
    """

    def record_pre_step(self):
        phase = getattr(self._env, "oracle_cycle_phase", None)
        if phase is None:
            return None, None
        return "obs/cycle_phase", phase.clone()


class CycleBrickRecorder(RecorderTerm):
    """Per-step oracle target brick index, from the OracleRig stash (int8, -1 = none)."""

    def record_pre_step(self):
        brick = getattr(self._env, "oracle_cycle_brick", None)
        if brick is None:
            return None, None
        return "obs/cycle_brick", brick.clone()


class ClampRecorder(RecorderTerm):
    """Per-step `clamped` flag from the clamped arm action term (spec section 4.3):
    True where maker_arm_rs.clamp_command changed that env's target this step."""

    def record_post_step(self):
        term = self._env.action_manager.get_term("arm_action")
        flag = getattr(term, "clamped", None)
        if flag is None:
            return None, None
        return "obs/clamped", flag.clone()


@configclass
class JointStateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = JointStateRecorder


@configclass
class ImagesObsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ImagesObsRecorder


@configclass
class JointTargetRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = JointTargetRecorder


@configclass
class CyclePhaseRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CyclePhaseRecorder


@configclass
class CycleBrickRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CycleBrickRecorder


@configclass
class ClampRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ClampRecorder


@configclass
class DatagenRecorderManagerCfg(RecorderManagerBaseCfg):
    """Everything the LeRobot conversion needs, nothing heavier."""

    record_initial_state = InitialStateRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_policy_obs = PreStepFlatPolicyObservationsRecorderCfg()
    record_pre_step_joint_state = JointStateRecorderCfg()
    record_pre_step_images = ImagesObsRecorderCfg()
    record_post_step_joint_targets = JointTargetRecorderCfg()
    record_pre_step_cycle_phase = CyclePhaseRecorderCfg()
    record_pre_step_cycle_brick = CycleBrickRecorderCfg()
    record_post_step_clamped = ClampRecorderCfg()

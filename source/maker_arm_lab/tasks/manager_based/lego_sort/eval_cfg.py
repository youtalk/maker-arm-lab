"""Gate-evaluation env: the Datagen scene/DR with the dataset's action space.

Registered as MakerArm-LegoSort-Eval-v0 (design spec section 5.5).
Delta from LegoSortDatagenEnvCfg, and nothing else:

- recorders: back to the empty default. Nothing is exported at eval time;
  scripts/eval_policy.py grades live via mdp.terminations.per_brick_stowed.

The action space USED to be a delta too, and is not any more. This cfg
re-installed the base joint-position ActionsCfg because Datagen inherited the
IK-relative space, so eval and generation would otherwise have differed in the
one dimension that must match. Task D5 moved Datagen itself onto the
joint-position space (the scripted oracle now emits joint targets from
core.global_ik), so the override became a no-op restating what it already
inherits, and it is gone. What it was protecting is unchanged and now holds by
inheritance: absolute joint-position targets for the six arm joints
(ARM_JOINT_REGEX, scale 1.0, use_default_offset False) plus the binary gripper
(raw action >= 0 opens; see the hard-won sign-convention comment in
lego_sort_env_cfg.py), which is exactly the dataset's 7-dim `action` feature on
the same MIT-mode actuator underneath -- the policy needs no adapter.

Everything else -- 180 s episodes, cameras + image observations, run-level
draw fields, reset-level DR event terms, terminations, contact sensors -- is
inherited unchanged, so the eval distribution IS the generation distribution
(design spec section 2). The contact sensors are kept for the same reason: a
future scripts/eval_policy.py (arriving in a later milestone) is expected to
need them to attribute a wrong-bin deposit to a brick, the way openyam-lab's
does. Stripping them here would also fork the scene from the one the dataset
was generated in.
"""

from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.utils import configclass

from .datagen import LegoSortDatagenEnvCfg


@configclass
class LegoSortEvalEnvCfg(LegoSortDatagenEnvCfg):
    """Registered as MakerArm-LegoSort-Eval-v0."""

    def __post_init__(self):
        super().__post_init__()
        # Same class the base ManagerBasedEnvCfg default uses (imported there
        # as DefaultEmptyRecorderManagerCfg): no terms, no export, no-op.
        self.recorders = RecorderManagerBaseCfg()

"""Task-space (differential-IK) action-space variant of LegoSortEnvCfg.

The oracle, the scripted grasp check, and the release-envelope check drive this variant.
Its IK target is the URDF's `grasp_center` frame, so no body offset is needed; the solved
joint targets pass through maker_arm_rs's clamp (mdp.actions). The joint-velocity and
joint-effort variants of openyam-lab are not registered here (spec section 2.3).

Known behaviour of the upstream action term, measured in openyam-lab and unchanged here:
a zero rotation delta does NOT hold the tool orientation (`apply_delta_pose` re-bases the
desired orientation on the measured one every step), so anything that needs a held attitude
must servo the rotation channel from a world-frame axis-angle error --
`core.attitude_hold.AttitudeHold` is that loop.
"""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from maker_arm_lab.core.action_scale import IK_SCALE
from maker_arm_lab.robots import ARM_JOINT_REGEX, TOOL_BODY

from . import mdp
from .lego_sort_env_cfg import LegoSortEnvCfg


@configclass
class LegoSortIKRelEnvCfg(LegoSortEnvCfg):
    """Task-space delta pose + differential IK, target frame `grasp_center`."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = mdp.ClampedDifferentialIKActionCfg(
            asset_name="robot",
            joint_names=[ARM_JOINT_REGEX],
            body_name=TOOL_BODY,
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=IK_SCALE,
        )

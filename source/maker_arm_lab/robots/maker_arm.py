"""ArticulationCfg for the converted Maker Arm USD, driven by the maker_arm_v1 profile."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from maker_arm_lab.actuators import MITModeActuatorCfg
from maker_arm_lab.assets.articulation import GravityCompensatedArticulation
from maker_arm_lab.core import profile as _profile
from maker_arm_lab.robots.frames import *  # noqa: F403 -- one import surface for the arm

_REPO_ROOT = Path(__file__).resolve().parents[3]
MAKER_ARM_USD_PATH = _REPO_ROOT / "assets" / "usd" / "maker_arm.usd"

# Gains and caps come from the Rust profile; see core.profile. No literals here on purpose.
_GAINS = _profile.arm_gains(_profile.load_profile())

# Placeholder from the URDF (README: "Gripper effort 20 N ... are placeholders"). Whether it
# holds a brick through the pinch criterion is measured in scripted_grasp_check (Task D3).
GRIPPER_EFFORT_LIMIT_N = 20.0

MAKER_ARM_CFG = ArticulationCfg(
    class_type=GravityCompensatedArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(MAKER_ARM_USD_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(HOME_JOINT_POS)),
    actuators={
        "arm": MITModeActuatorCfg(
            joint_names_expr=[ARM_JOINT_REGEX],
            stiffness={name: g["kp"] for name, g in _GAINS.items()},
            damping={name: g["kd"] for name, g in _GAINS.items()},
            armature=0.0,
            gravity_compensation=True,
            tau_ff_limit={name: g["tau_max"] for name, g in _GAINS.items()},
            effort_limit={name: g["effort_limit"] for name, g in _GAINS.items()},
            effort_limit_sim={name: g["effort_limit"] for name, g in _GAINS.items()},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(GRIPPER_JOINT_NAMES),
            stiffness=1000.0,
            damping=200.0,
            effort_limit_sim=GRIPPER_EFFORT_LIMIT_N,
        ),
    },
)

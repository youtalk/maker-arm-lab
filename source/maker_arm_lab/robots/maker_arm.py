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

# Raised from the URDF's 20 N by Task D3 step 4.3, which authorises exactly this raise when
# the pinch will not hold. A DOCUMENTED SIM/REAL DIVERGENCE, and a cheap one: the URDF's own
# README calls its 20 N a placeholder, so the number being diverged from was never a
# measurement of the real gripper either.
#
# MEASURED INERT, and that is the useful part. Task D3 part 13 ran the ten-trial gate at 20 N
# and at 40 N with nothing else changed. The two lift slips this raise was aimed at came out
# the same at both caps -- trial 5 is bit-identical (grip 4.78/4.97 N, stall 4.329 mm) and
# trial 9 moves 5.26/5.26 -> 5.35/5.34 N -- so the effort limit is NOT what caps the grip.
# The gripper's implicit PD is: the jaws are commanded to 0.0 (closed) and stall on the brick
# at ~7.6 mm, so stiffness 1000 N/m x 7.6 mm = ~7.6 N of demand against the observed
# ~7.0 N peak. A ~7 N demand never reached the old 20 N cap, let alone this one.
#
# So the lever on a grip that will not survive the lift is `stiffness` below, not this
# constant. 40 N is kept as headroom for that change rather than reverted, because the value
# it replaced was itself an unmeasured placeholder and neither number is load-bearing today.
# Revisit against the real gripper's rated holding force during MA-series bring-up rather
# than treating either as a property of the hardware.
GRIPPER_EFFORT_LIMIT_N = 40.0

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

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

# The URDF's value, kept because it is the vendored one (README: "Gripper effort 20 N ...
# are placeholders"). Task D3 step 4.3 authorises raising this to 40 N when the pinch will
# not hold, and part 12 established that it does not hold: two of ten trials are proper
# pinches at the close that slip during the lift. Part 13 pulled that lever and MEASURED IT
# INERT, so ruling D3-17 put the number back and kept the knowledge here.
#
# The A/B, ten trials at each cap with nothing else changed: the two lift slips came out the
# same at both. Trial 5 is bit-identical (grip 4.78/4.97 N, stall 4.329 mm) and trial 9 moves
# 5.26/5.26 -> 5.35/5.34 N. The effort limit is NOT what caps the grip -- the gripper's
# implicit PD is. The jaws are commanded to 0.0 (closed) and stall on the brick at ~7.6 mm,
# so stiffness 1000 N/m x 7.6 mm = ~7.6 N of demand against the observed ~7.0 N peak. A ~7 N
# demand never reached even this 20 N cap, let alone a 40 N one.
#
# So the lever on a grip that will not survive the lift is `stiffness` below, not this
# constant, and raising this one is a sim/real divergence that buys nothing. Whatever the
# real gripper turns out to hold at, measured during MA-series bring-up, belongs here -- but
# a strengthened-looking constant that changes no outcome does not.
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

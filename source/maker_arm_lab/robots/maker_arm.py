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

# Rotor inertia reflected through the gearbox, for all six arm joints.
#
# Still UNVERIFIED (spec section 7) -- this is NOT a measured rotor inertia. It is a
# stability floor, and it is an arrival re-measurement item: the real value is the motor's
# rotor inertia times the gear ratio squared, which the vendor URDF omits and the SDK does
# not publish. Replace it on MA-series bring-up, and see below for what your measured value
# has to clear.
#
# WHAT IT IS PROTECTING AGAINST. The URDF omits armature and this was 0.0, which is not a
# neutral default. Measured by /tmp/d3_decompose.py on 2026-09-06 (Task D3 part 14),
# instrumented inside MITModeActuator.compute so the law's three terms are read at the
# instant it differences them: at 0.0 every wrist joint sat in a period-2 limit cycle at the
# Nyquist frequency of the 120 Hz physics rate. link_005_joint reversed sign EVERY substep
# at +-4.2 rad/s with the applied torque flipping +-9 N.m, while its mean position error was
# 0.25 mrad -- the torque was almost entirely the damping term reacting to its own last step.
#
# That is the explicit-damping instability, and the bound is
#
#     kd * dt / M_eff  >=  2
#
# It bites the WRIST joints specifically, because they carry almost no link inertia, so
# M_eff is whatever armature says -- and it said zero. For kd = 2.0 (link_005..007) at
# dt = 1/120 the marginal value is kd*dt/2 = 0.00833, and the measured bracket straddles it:
#
#     armature 0.005  -> still chatters (qd -3.52 / +4.28 / -4.03 / +3.38 rad/s per substep)
#     armature 0.01   -> clean          (qd ~0.004 rad/s)
#     armature 0.02   -> clean, shipped (~2.4x margin on the 0.00833 bound)
#
# A MEASURED VALUE MUST STILL CLEAR THAT BOUND. If the real rotor inertia comes in below
# ~0.0083 for the wrist joints, raising the physics rate or lowering kd is the fix -- do not
# simply keep this number, because too LARGE an armature is its own sim/real divergence: it
# silently adds fictitious inertia and slows dynamics the real arm would not have.
#
# What it cost while it was 0.0, which is why every number in this task's parts 11-13 should
# be re-read: the chatter aliased every post-hoc sample from ArticulationData, because
# computed_torque comes from the last of decimation=4 substeps while joint_pos and joint_vel
# are read back after that substep integrated -- opposite phases of a period-2 oscillation.
# With armature set, the same decomposition closes to 5e-6 N.m and max |qd| across the arm
# falls from 4.99 to 0.002 rad/s.
ARM_ARMATURE_UNVERIFIED = True
ARM_ARMATURE = 0.02

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
            armature=ARM_ARMATURE,
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

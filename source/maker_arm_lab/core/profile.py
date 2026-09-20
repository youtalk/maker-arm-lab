"""Adapter over ``maker_arm_rs.profile()`` / ``clamp_command()`` (spec sections 4.3-4.6).

The sim takes its gains, torque caps, and the command clamp from the Rust SDK so the
dataset's actions are, by construction, commands the real arm accepts. There is no fallback
table: a missing ``maker_arm_rs`` fails here, at import.
"""

from collections.abc import Sequence

import maker_arm_rs
import torch

from maker_arm_lab.assets import urdf_prep

ARM_JOINT_COUNT = 6
GRIPPER_INDEX = 6
# Peak torque per RobStride model (maker-arm-protocol t_max): the MIT-mode total torque
# saturates here; the profile's per-joint tau_max caps only the feed-forward term.
MOTOR_RATING_NM = {"RS00": 14.0, "RS02": 17.0}


def load_profile() -> dict:
    return maker_arm_rs.profile()


def _check_uncalibrated(profile: dict) -> None:
    for j in profile["joints"][:ARM_JOINT_COUNT]:
        if j["direction"] != 1.0 or j["offset"] != 0.0:
            raise ValueError(
                f"joint {j['name']} has direction={j['direction']} offset={j['offset']}; the sim "
                "assumes identity motor->URDF mapping (spec section 4.1) -- re-express the "
                "profile in URDF coordinates instead of passing calibrated values here"
            )


def sim_profile(urdf_limits: Sequence[tuple[float, float]] | None = None) -> dict:
    """The v1 profile with the arm joints' limits replaced by the URDF's (spec section 4.3)."""
    p = load_profile()
    if len(p["joints"]) != ARM_JOINT_COUNT + 1:
        raise ValueError(f"expected 7 joints in the profile, got {len(p['joints'])}")
    _check_uncalibrated(p)
    limits = list(urdf_limits) if urdf_limits is not None else urdf_prep.arm_joint_limits()
    if len(limits) != ARM_JOINT_COUNT:
        raise ValueError(f"expected {ARM_JOINT_COUNT} URDF limits, got {len(limits)}")
    for j, (lo, hi) in zip(p["joints"][:ARM_JOINT_COUNT], limits):
        j["q_lo"], j["q_hi"] = float(lo), float(hi)
    return p


def arm_gains(profile: dict) -> dict[str, dict[str, float]]:
    """URDF joint name -> kp / kd / tau_max (feed-forward cap) / effort_limit (motor rating)."""
    out = {}
    for name, j in zip(urdf_prep.ARM_JOINT_NAMES, profile["joints"][:ARM_JOINT_COUNT]):
        out[name] = {
            "kp": float(j["kp"]),
            "kd": float(j["kd"]),
            "tau_max": float(j["tau_max"]),
            "effort_limit": MOTOR_RATING_NM[j["model"]],
        }
    return out


def gripper_angle(profile: dict, open_fraction: float) -> float:
    """Provisional linear map from opening fraction to the profile gripper's motor range.

    Which end is open is unknown until the arm is calibrated (spec section 7); the sim's own
    gripper is driven in metres and does not depend on this. The value only fills the seventh
    clamp slot, where it is in range by construction.
    """
    g = profile["joints"][GRIPPER_INDEX]
    f = float(open_fraction)
    # Endpoint-exact form of the same linear map: `q_lo + f * (q_hi - q_lo)` loses the last ULP
    # at f=1.0 for this profile's endpoints (-2.092, -0.039), so f=0.0/1.0 no longer round-trip
    # to exactly q_lo/q_hi.
    return g["q_lo"] * (1.0 - f) + g["q_hi"] * f


def clamp_joint_targets(
    arm_targets: Sequence[Sequence[float]], gripper_fraction: Sequence[float], profile: dict
) -> tuple[list[list[float]], list[bool]]:
    """Run every env's 6 arm targets through the Rust clamp; return positions and flags.

    ``gripper_fraction`` is the per-env jaw opening as a fraction of the profile's travel.
    The binary path passes bools (True is open, 1.0); the continuous gripper (MS6 phase B
    design 5.4) passes the fraction its gap command maps to.
    """
    if len(arm_targets) != len(gripper_fraction):
        raise ValueError("arm_targets and gripper_fraction must have one entry per env")
    joints = profile["joints"]
    out_pos: list[list[float]] = []
    flags: list[bool] = []
    for targets, fraction in zip(arm_targets, gripper_fraction):
        if len(targets) != ARM_JOINT_COUNT:
            raise ValueError(f"expected {ARM_JOINT_COUNT} arm targets, got {len(targets)}")
        cmds = [
            (float(q), 0.0, float(j["kp"]), float(j["kd"]), 0.0)
            for q, j in zip(targets, joints[:ARM_JOINT_COUNT])
        ]
        g = joints[GRIPPER_INDEX]
        cmds.append((gripper_angle(profile, float(fraction)), 0.0, g["kp"], g["kd"], 0.0))
        clamped_cmds, changed = maker_arm_rs.clamp_command(cmds, profile)
        out_pos.append([c[0] for c in clamped_cmds[:ARM_JOINT_COUNT]])
        flags.append(bool(changed))
    return out_pos, flags


def clamp_targets_tensor(
    targets: torch.Tensor,
    gripper_fraction: torch.Tensor,
    profile: dict,
    out_flags: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    """Batched (N, 6) targets through the Rust clamp; one host sync per call.

    ``gripper_fraction`` is a bool or float tensor, one entry per env.
    By default ``out_flags`` is overwritten with this call's flags. With ``accumulate=True``,
    this call's flags are OR-ed into whatever ``out_flags`` already holds, for callers that
    invoke this once per physics substep but want the flag to record "clamped on any substep
    this control step" rather than just the last one -- see
    ``mdp.actions.ClampedDifferentialIKAction``.
    """
    pos, flags = clamp_joint_targets(targets.tolist(), gripper_fraction.tolist(), profile)
    new_flags = torch.tensor(flags, dtype=torch.bool, device=out_flags.device)
    if accumulate:
        out_flags |= new_flags
    else:
        out_flags.copy_(new_flags)
    return torch.tensor(pos, dtype=targets.dtype, device=targets.device)

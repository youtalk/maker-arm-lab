"""RobStride MIT-mode control law: tau = kp*(q* - q) + kd*(qd* - qd) + tau_ff.

This function is the sim<->real contract point (design spec section 4.4): the
Isaac Lab actuator (maker_arm_lab.actuators.mit_mode) and a future CAN backend
must both reduce to exactly this law. Torch-only on purpose -- CPU unit tests
must not require isaaclab.
"""

import torch


def mit_mode_torque(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_des: torch.Tensor,
    qd_des: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    tau_ff: torch.Tensor,
    effort_limit: torch.Tensor,
) -> torch.Tensor:
    tau = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff
    return torch.clamp(tau, -effort_limit, effort_limit)


def mit_mode_feedforward(
    tau_cmd: torch.Tensor, tau_gravity: torch.Tensor | None = None
) -> torch.Tensor:
    """Compose the single tau_ff a RobStride MIT-mode packet carries.

    A RobStride motor in MIT mode does not compute gravity compensation on-motor:
    the CAN packet has exactly one feed-forward field, and the *host* decides
    what goes in it. This function is that host-side decision, pinned as one
    tested rule so the Isaac Lab actuator and a future CAN backend cannot drift
    apart:

        tau_ff = tau_cmd + tau_gravity

    Sign convention (the part that is easy to get backwards -- an inverted sign
    doubles the sag instead of cancelling it). ``tau_gravity`` is G(q) from the
    equation of motion

        tau = M(q) qdd + C(q, qd) qd + G(q)

    i.e. the generalized torque that must be *applied* to hold the articulation
    static against gravity, not the torque gravity exerts on it. It therefore
    enters tau_ff with a PLUS sign. In simulation this is PhysX's
    ``ArticulationView.get_gravity_compensation_forces()``, whose documented
    convention is exactly that G(q) (see the omni.physics.tensors
    ``inverse_dynamics`` docs shipped with Isaac Sim 5.1.0); on hardware it is
    whatever the host's rigid-body model returns for the same quantity.

    ``tau_gravity=None`` means no gravity compensation, and returns ``tau_cmd``
    unchanged -- the raw-torque passthrough.

    Args:
        tau_cmd: Feed-forward torque the caller asked for. Any shape.
        tau_gravity: G(q), broadcastable to ``tau_cmd``, or None.

    Returns:
        The composed feed-forward torque to pass to :func:`mit_mode_torque`.
    """
    if tau_gravity is None:
        return tau_cmd
    return tau_cmd + tau_gravity

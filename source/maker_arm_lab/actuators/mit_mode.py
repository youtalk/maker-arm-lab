"""Explicit MIT-mode actuator mirroring the RobStride CAN command structure."""

import logging

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from maker_arm_lab.core.mit_law import mit_mode_feedforward, mit_mode_torque

logger = logging.getLogger(__name__)


class MITModeActuator(IdealPDActuator):
    """IdealPDActuator whose torque computation is pinned to core.mit_law.

    The indirection exists so the sim actuator and a future real-robot CAN
    backend share one law (design spec section 5.2). Swap this class, keep
    the interface. This class is NOT behaviorally equivalent to
    IdealPDActuator on two points, both deliberate and both documented here
    (code review correction -- an earlier version of this docstring claimed
    full equivalence, which was false):

    1. None-target substitution. IdealPDActuator.compute
       (isaaclab v2.3.2, actuators/actuator_pd.py:186-201) has no `is None`
       guard on any of control_action.joint_positions/joint_velocities/
       joint_efforts -- it computes `control_action.joint_positions -
       joint_pos` directly and *raises* TypeError if that field is None.
       There is therefore no upstream "None" behavior to replicate: upstream
       never receives None here, because Articulation._apply_actuator_model()
       always builds control_action from the zero-initialized
       _data.joint_pos_target/joint_vel_target buffers (never None, just
       possibly 0.0 for a joint no action term has targeted). That zero-init
       buffer models "a target the caller left unset", which is a different
       case: ArticulationActions' own fields default to None and its docstring
       says "if the actions are not provided, the values are set to None" --
       i.e. the caller is not commanding position/velocity *at all*. For that
       case the correct substitution is the current state (q_des=joint_pos,
       qd_des=joint_vel), which drives the position/velocity error terms to
       exactly zero and leaves tau = tau_ff -- precisely what a RobStride
       MIT-mode packet with kp=kd=0 does. (An earlier version of this method
       substituted zeros instead, which for nonzero kp/kd silently commands
       full-scale torque toward q=0/qd=0 -- e.g. it returned -10.0 N*m, the
       negative effort limit, for a caller asking for +1.0 N*m of pure
       feed-forward. That was wrong and is fixed below.)

       WHO TAKES THIS BRANCH, stated honestly: as of this port, nothing in
       this repo does. Every path through the env goes via Articulation,
       which builds control_action from the zero-initialized
       _data.joint_pos_target/joint_vel_target buffers and so never passes
       None here either. The branch is kept because a direct caller is the
       whole point of this class -- it is what the CAN backend of spec
       section 5.2 will be -- and it is pinned by
       tests/test_mit_mode_actuator.py so it cannot rot while unused.
    2. computed_effort is the demand *unclamped by effort_limit*, applied_effort
       is the clamped one actually sent to the simulation -- matching
       IdealPDActuator's own convention (`computed_effort = raw PD+ff sum`,
       `applied_effort = self._clip_effort(computed_effort)`). mit_mode_torque
       itself only returns the clamped value (its clamp is a pinned interface
       covered by test_effort_clamp_both_signs and must not change), so the
       unclamped demand is computed separately here, only for reporting via
       ArticulationData.computed_torque. Downstream code that detects
       saturation via `computed_torque - applied_torque` depends on this
       distinction; collapsing both to the clamped value (as an earlier
       version of this method did) makes that difference read as zero
       forever. computed_effort is NOT unclamped with respect to tau_ff_limit,
       though: that cap is applied to tau_ff before this sum, so tau_ff_limit
       saturation is already baked into computed_effort and is invisible to
       the `computed_effort - applied_effort` idiom -- only effort_limit
       saturation shows up there. (tests/test_mit_mode_actuator.py's tau_ff_limit
       test asserts computed_effort == 4.0, not the uncapped 5.5, for exactly
       this reason.)

    Gravity compensation enters through :attr:`gravity_torque`, which this
    class only ever *reads*. Filling it is the job of
    :class:`maker_arm_lab.assets.GravityCompensatedArticulation`, which is what
    MAKER_ARM_CFG spawns; under a plain isaaclab ``Articulation`` the buffer
    stays zero and the law is bit-for-bit what it was before gravity
    compensation existed. That fail-safe is deliberate -- a direct
    ``compute()`` call from a CPU unit test gets the unchanged law.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_ff_limit = self._parse_joint_parameter(self.cfg.tau_ff_limit, float("inf"))
        # G(q) for this actuator's joints, in the same (num_envs, num_joints)
        # layout as computed_effort. Refreshed from PhysX once per physics
        # step by GravityCompensatedArticulation.write_data_to_sim(); zeroed
        # by it when cfg.gravity_compensation is False, so the flag can be
        # flipped at run time for A/B measurement.
        self.gravity_torque = torch.zeros_like(self.computed_effort)
        # Set True by GravityCompensatedArticulation the first time it fills
        # the buffer. If it stays False while cfg.gravity_compensation is True,
        # the articulation is a plain isaaclab Articulation and the flag is a
        # lie -- compute() says so out loud rather than silently doing nothing.
        self.gravity_torque_written = False
        self._warned_no_gravity_source = False

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q_des = control_action.joint_positions
        if q_des is None:
            q_des = joint_pos
        qd_des = control_action.joint_velocities
        if qd_des is None:
            qd_des = joint_vel
        tau_ff = control_action.joint_efforts
        if tau_ff is None:
            tau_ff = torch.zeros_like(joint_pos)
        # Host-side feed-forward composition -- the one field a RobStride
        # MIT-mode CAN packet carries. Gravity is added HERE, not inside
        # mit_mode_torque, so the law's signature and clamping stay exactly as
        # the CPU tests pin them, and so a real host that sends
        # tau_ff = tau_cmd + tau_gravity produces the identical torque.
        tau_ff = mit_mode_feedforward(tau_ff, self.gravity_torque)
        tau_ff = torch.clamp(tau_ff, -self.tau_ff_limit, self.tau_ff_limit)
        if (
            not self._warned_no_gravity_source
            and getattr(self.cfg, "gravity_compensation", False)
            and not self.gravity_torque_written
        ):
            self._warned_no_gravity_source = True
            logger.warning(
                "%s: cfg.gravity_compensation is True but nothing is filling "
                "gravity_torque, so no gravity compensation is being applied. "
                "The articulation must be a "
                "maker_arm_lab.assets.GravityCompensatedArticulation -- set "
                "ArticulationCfg.class_type accordingly (MAKER_ARM_CFG does).",
                type(self).__name__,
            )

        # Unclamped demand, for ArticulationData.computed_torque -- mirrors
        # IdealPDActuator's own computed_effort (pre-saturation). Duplicates
        # mit_mode_torque's arithmetic (minus the clamp) rather than adding a
        # second return value to mit_mode_torque, which is a pinned,
        # CPU-tested interface (test_effort_clamp_both_signs) that must keep
        # its current signature and clamping behavior.
        self.computed_effort = (
            self.stiffness * (q_des - joint_pos) + self.damping * (qd_des - joint_vel) + tau_ff
        )

        tau = mit_mode_torque(
            q=joint_pos,
            qd=joint_vel,
            q_des=q_des,
            qd_des=qd_des,
            kp=self.stiffness,
            kd=self.damping,
            tau_ff=tau_ff,
            effort_limit=self.effort_limit,
        )
        self.applied_effort = tau
        control_action.joint_efforts = tau
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class MITModeActuatorCfg(IdealPDActuatorCfg):
    class_type: type = MITModeActuator

    gravity_compensation: bool = True
    """Add PhysX's G(q) to tau_ff for this actuator's joints. Defaults to True.

    Requires the articulation to be a
    :class:`maker_arm_lab.assets.GravityCompensatedArticulation`; that class is
    what reads this flag (once per physics step, so it can be toggled at run
    time for A/B measurement) and fills
    :attr:`MITModeActuator.gravity_torque`. Set False for an action space whose
    contract is raw torque passthrough.
    """

    tau_ff_limit: dict[str, float] | float | None = None
    """Cap (N·m) on the composed feed-forward ``tau_cmd + tau_gravity`` per joint -- the
    maker_arm_v1 profile's ``tau_max``. ``None`` = no cap. The PD total is still clamped by
    ``effort_limit`` (the motor rating)."""

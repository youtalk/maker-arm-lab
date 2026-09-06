"""CPU-only tests for maker_arm_lab.actuators.mit_mode.MITModeActuator.

``MITModeActuator`` is the sim<->real contract point of design spec section
5.2 -- the class a future RobStride CAN backend replaces. ``mit_mode.py``
imports isaaclab, so it cannot go in the CPU suite directly; this test file is
ported from openyam-lab, where it used the same ``sys.modules`` stand-in trick
as that repo's ``test_lego_sort_terminations.py`` to cover the class on CPU
anyway.

The same trick is used here, scoped tighter. Stand-ins for the four isaaclab
names ``mit_mode.py`` imports are registered in ``sys.modules``, the module is
loaded **by file path** (so ``maker_arm_lab.actuators.__init__`` is never
executed), and the stand-ins are then removed again -- the module has already
bound the names it needs, so nothing else in the suite can see them.

What is stood in for, and what is not. The stand-ins are the two isaaclab base
classes and the ``ArticulationActions`` container: none of them participate in
the arithmetic under test. ``MITModeActuator.compute()`` reads only
``self.stiffness``, ``self.damping``, ``self.effort_limit``,
``self.gravity_torque`` and ``self.cfg``, and the law it applies is the real
``maker_arm_lab.core.mit_law``. ``_clip_effort`` is deliberately absent from the
stand-in base: this class must not call it -- ``mit_mode_torque`` owns the
clamp -- and an edit that reintroduced the upstream call would fail here with
AttributeError rather than passing silently.

Two behaviours are pinned, both of which were review findings in openyam-lab
that a future edit could collapse back with every other gate in this repo
still green:

1. **The None-target substitution.** An earlier version substituted zeros,
   which returned -10 N*m -- the negative effort limit -- for a caller asking
   for +1 N*m of pure feed-forward.
2. **The computed_effort / applied_effort split.** ``computed_effort`` is the
   unclamped demand and ``applied_effort`` the clamped one; collapsing them
   makes downstream saturation detection (``computed_torque -
   applied_torque``) read zero forever.
"""

import dataclasses
import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch

_MIT_MODE_PATH = Path(__file__).resolve().parents[1] / "source/maker_arm_lab/actuators/mit_mode.py"


@dataclasses.dataclass
class ArticulationActions:
    """Stand-in for ``isaaclab.utils.types.ArticulationActions``.

    Field-for-field identical to the upstream dataclass at v2.3.2, including
    the ``None`` defaults -- upstream's own docstring says "If the actions are
    not provided, the values are set to None", which is the state this module's
    first test covers.
    """

    joint_positions: torch.Tensor | None = None
    joint_velocities: torch.Tensor | None = None
    joint_efforts: torch.Tensor | None = None
    joint_indices: torch.Tensor | slice | None = None


class IdealPDActuatorCfg:
    """Stand-in base for the actuator cfg. Carries only what the test sets."""

    joint_names_expr: list[str] | None = None
    stiffness: float | None = None
    damping: float | None = None
    effort_limit: float | None = None


class IdealPDActuator:
    """Stand-in for the isaaclab base class, reduced to what ``compute`` reads.

    Upstream's ``ActuatorBase.__init__`` resolves the gains out of the cfg
    against the articulation's joint names and stores them as ``(num_envs,
    num_joints)`` tensors; that resolution is upstream's contract, not this
    class's, so it is short-circuited here by taking the resolved tensors
    directly. ``computed_effort`` must exist before ``MITModeActuator.__init__``
    runs -- it sizes ``gravity_torque`` off it with ``torch.zeros_like``.
    """

    def __init__(
        self,
        cfg: IdealPDActuatorCfg,
        *,
        num_envs: int = 1,
        num_joints: int = 6,
        stiffness: float = 0.0,
        damping: float = 0.0,
        effort_limit: float = torch.inf,
    ):
        shape = (num_envs, num_joints)
        self.cfg = cfg
        self.stiffness = torch.full(shape, float(stiffness))
        self.damping = torch.full(shape, float(damping))
        self.effort_limit = torch.full(shape, float(effort_limit))
        self.computed_effort = torch.zeros(shape)
        self.applied_effort = torch.zeros(shape)

    def _parse_joint_parameter(self, cfg_value, default_value):
        value = default_value if cfg_value is None else cfg_value
        return torch.full(self.computed_effort.shape, float(value))


def _load_mit_mode():
    """Load ``mit_mode.py`` against the stand-ins, then unregister them."""
    stand_ins = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.actuators": types.ModuleType("isaaclab.actuators"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.types": types.ModuleType("isaaclab.utils.types"),
    }
    stand_ins["isaaclab.actuators"].IdealPDActuator = IdealPDActuator
    stand_ins["isaaclab.actuators"].IdealPDActuatorCfg = IdealPDActuatorCfg
    # The real ``configclass`` turns the cfg into a dataclass with deep-copied
    # nested defaults. Nothing here constructs a nested cfg, so identity is
    # enough and keeps the stand-in honest about doing nothing.
    stand_ins["isaaclab.utils"].configclass = lambda cls: cls
    stand_ins["isaaclab.utils.types"].ArticulationActions = ArticulationActions

    sys.modules.update(stand_ins)
    try:
        spec = importlib.util.spec_from_file_location("_mit_mode_standalone", _MIT_MODE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in stand_ins:
            sys.modules.pop(name, None)


_mit_mode = _load_mit_mode()
MITModeActuator = _mit_mode.MITModeActuator
MITModeActuatorCfg = _mit_mode.MITModeActuatorCfg

# Arbitrary PD-math fixture values, carried over from openyam-lab's DM4310 test
# suite -- not the Maker Arm's real gains or effort limits. See
# test_effort_limit_is_respected_per_joint for the profile's actual per-joint caps.
KP, KD, EFFORT_LIMIT = 800.0, 20.0, 10.0
NUM_JOINTS = 6


def _actuator(gravity_compensation: bool = False, **kwargs) -> MITModeActuator:
    cfg = MITModeActuatorCfg()
    cfg.gravity_compensation = gravity_compensation
    kwargs.setdefault("stiffness", KP)
    kwargs.setdefault("damping", KD)
    kwargs.setdefault("effort_limit", EFFORT_LIMIT)
    return MITModeActuator(cfg, num_joints=NUM_JOINTS, **kwargs)


def _state(pos: float = 0.3, vel: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.full((1, NUM_JOINTS), pos), torch.full((1, NUM_JOINTS), vel)


# --------------------------------------------------------------------------
# Contract 1: None targets mean "not commanding this channel at all"
# --------------------------------------------------------------------------


def test_none_position_and_velocity_targets_leave_pure_feedforward_untouched():
    """kp/kd nonzero, both targets None: the law must reduce to tau = tau_ff.

    This is the RobStride MIT-mode packet with kp=kd=0 in all but name -- the
    caller is commanding torque only -- and substituting the *current state*
    is what drives both error terms to exactly zero. The regression it pins is
    concrete: substituting zeros instead gives
    ``800*(0-0.3) + 20*(0-0.1) + 1.0 = -241 N*m``, clamped to the -10 N*m
    effort limit, for a caller asking for +1 N*m.
    """
    actuator = _actuator()
    joint_pos, joint_vel = _state()
    tau_ff = torch.full((1, NUM_JOINTS), 1.0)

    out = actuator.compute(ArticulationActions(joint_efforts=tau_ff.clone()), joint_pos, joint_vel)

    assert torch.allclose(out.joint_efforts, tau_ff)
    assert torch.allclose(actuator.applied_effort, tau_ff)
    assert not torch.any(actuator.applied_effort <= -EFFORT_LIMIT)


def test_none_effort_target_is_zero_feedforward():
    """Only the effort channel unset: tau_ff is zero, the PD terms still run."""
    actuator = _actuator()
    joint_pos, joint_vel = _state()
    q_des = joint_pos + 0.001

    out = actuator.compute(
        ArticulationActions(joint_positions=q_des, joint_velocities=joint_vel.clone()),
        joint_pos,
        joint_vel,
    )

    # 800 * 0.001 + 20 * 0.0 + 0.0. atol, because kp=800 amplifies the float32
    # representation error of a 1 mrad difference to ~2e-5 N*m.
    assert torch.allclose(out.joint_efforts, torch.full((1, NUM_JOINTS), 0.8), atol=1e-4)


def test_supplied_targets_take_the_ordinary_pd_path():
    """The None branch must not hijack a caller that *is* commanding a target.

    Also pins the return contract the articulation write path depends on: the
    position and velocity fields are cleared and the effort field carries the
    applied torque.
    """
    actuator = _actuator()
    joint_pos, joint_vel = _state()
    action = ArticulationActions(
        joint_positions=joint_pos + 0.001,
        joint_velocities=joint_vel + 0.01,
        joint_efforts=torch.full((1, NUM_JOINTS), 0.5),
    )

    out = actuator.compute(action, joint_pos, joint_vel)

    # 800 * 0.001 + 20 * 0.01 + 0.5
    assert torch.allclose(out.joint_efforts, torch.full((1, NUM_JOINTS), 1.5), atol=1e-4)
    assert out.joint_positions is None
    assert out.joint_velocities is None


# --------------------------------------------------------------------------
# Contract 2: computed_effort is the demand, applied_effort is what is sent
# --------------------------------------------------------------------------


def test_computed_effort_is_unclamped_and_applied_effort_is_clamped():
    """``computed_torque - applied_torque`` is how every saturation figure in
    openyam-lab's milestones was measured; the Maker Arm is expected to reuse
    the same idiom once its own saturation is measured. Collapsing the two
    makes it read zero forever.
    """
    actuator = _actuator()
    joint_pos, joint_vel = _state()
    tau_ff = torch.full((1, NUM_JOINTS), 50.0)

    out = actuator.compute(ArticulationActions(joint_efforts=tau_ff.clone()), joint_pos, joint_vel)

    assert torch.allclose(actuator.computed_effort, tau_ff)
    assert torch.allclose(actuator.applied_effort, torch.full((1, NUM_JOINTS), EFFORT_LIMIT))
    assert torch.allclose(out.joint_efforts, actuator.applied_effort)
    saturation = actuator.computed_effort - actuator.applied_effort
    assert torch.allclose(saturation, torch.full((1, NUM_JOINTS), 40.0))


def test_clamp_is_symmetric_about_zero():
    actuator = _actuator()
    joint_pos, joint_vel = _state()
    tau_ff = torch.full((1, NUM_JOINTS), -50.0)

    out = actuator.compute(ArticulationActions(joint_efforts=tau_ff.clone()), joint_pos, joint_vel)

    assert torch.allclose(out.joint_efforts, torch.full((1, NUM_JOINTS), -EFFORT_LIMIT))


# --------------------------------------------------------------------------
# Gravity compensation enters through tau_ff, and says so when it cannot
# --------------------------------------------------------------------------


def test_gravity_torque_is_added_to_the_feedforward_field():
    """The one CAN field a RobStride packet carries -- host-side composition.

    Sign matters: G(q) is the torque that must be applied to hold the arm, so
    it enters positively. Inverted, it would double the sag.
    """
    actuator = _actuator(gravity_compensation=True)
    actuator.gravity_torque = torch.full((1, NUM_JOINTS), 0.51)
    actuator.gravity_torque_written = True
    joint_pos, joint_vel = _state()

    out = actuator.compute(
        ArticulationActions(joint_efforts=torch.full((1, NUM_JOINTS), 1.0)),
        joint_pos,
        joint_vel,
    )

    assert torch.allclose(out.joint_efforts, torch.full((1, NUM_JOINTS), 1.51))


def test_unwired_gravity_compensation_warns_once(caplog):
    """A cfg claiming compensation under a plain Articulation is a lie, and the
    actuator says so rather than silently doing nothing -- but only once, since
    this runs on the physics hot path.
    """
    actuator = _actuator(gravity_compensation=True)
    joint_pos, joint_vel = _state()

    with caplog.at_level(logging.WARNING, logger=_mit_mode.logger.name):
        for _ in range(3):
            actuator.compute(
                ArticulationActions(joint_efforts=torch.zeros(1, NUM_JOINTS)),
                joint_pos,
                joint_vel,
            )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "gravity_torque" in warnings[0].getMessage()


def test_no_warning_when_compensation_is_off():
    actuator = _actuator(gravity_compensation=False)
    joint_pos, joint_vel = _state()
    actuator.compute(
        ArticulationActions(joint_efforts=torch.zeros(1, NUM_JOINTS)), joint_pos, joint_vel
    )
    assert actuator._warned_no_gravity_source is False


def test_gravity_torque_buffer_starts_zero_and_matches_effort_shape():
    """The fail-safe: under a plain isaaclab Articulation nothing fills the
    buffer and the law is bit-for-bit what it was before compensation existed.
    """
    actuator = _actuator(gravity_compensation=True, num_envs=4)
    assert actuator.gravity_torque.shape == (4, NUM_JOINTS)
    assert not actuator.gravity_torque.any()
    assert actuator.gravity_torque_written is False


def test_stand_in_base_does_not_provide_clip_effort():
    """Guards the guard: this class must own its clamp via ``mit_mode_torque``.

    If a future edit reintroduces upstream's ``self._clip_effort(...)`` call,
    every test above fails loudly with AttributeError instead of passing.
    """
    assert not hasattr(MITModeActuator, "_clip_effort")


def test_cfg_defaults_to_gravity_compensation_on():
    assert MITModeActuatorCfg.gravity_compensation is True
    assert MITModeActuatorCfg.class_type is MITModeActuator


def test_effort_limit_is_respected_per_joint():
    """The Maker Arm cfg gives RS02 joints 17 N·m and RS00 joints 14 N·m; the
    clamp is per joint.
    """
    cfg = MITModeActuatorCfg()
    cfg.gravity_compensation = False
    actuator = MITModeActuator(cfg, num_joints=6, stiffness=KP, damping=KD)
    actuator.effort_limit = torch.tensor([[14.0, 17.0, 17.0, 14.0, 14.0, 14.0]])
    joint_pos, joint_vel = _state()

    out = actuator.compute(
        ArticulationActions(joint_efforts=torch.full((1, 6), 100.0)), joint_pos, joint_vel
    )

    assert torch.equal(out.joint_efforts, torch.tensor([[14.0, 17.0, 17.0, 14.0, 14.0, 14.0]]))


def test_tau_ff_limit_caps_the_feedforward_before_the_pd_sum():
    """The profile's tau_max caps tau_ff (gravity + command), not the PD total (spec 3.3)."""
    cfg = MITModeActuatorCfg()
    cfg.tau_ff_limit = 4.0
    actuator = MITModeActuator(cfg, num_joints=1, stiffness=0.0, damping=0.0, effort_limit=14.0)
    actuator.gravity_torque[:] = 3.5
    actuator.gravity_torque_written = True
    out = actuator.compute(
        ArticulationActions(joint_efforts=torch.tensor([[2.0]])),
        joint_pos=torch.zeros(1, 1),
        joint_vel=torch.zeros(1, 1),
    )
    assert torch.equal(out.joint_efforts, torch.tensor([[4.0]]))
    assert torch.equal(actuator.computed_effort, torch.tensor([[4.0]]))

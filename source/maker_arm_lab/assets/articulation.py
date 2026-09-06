"""Articulation that feeds PhysX gravity torques into MIT-mode tau_ff."""

from isaaclab.assets import Articulation


class GravityCompensatedArticulation(Articulation):
    """Articulation that refreshes each MIT-mode actuator's gravity torque.

    Why this class exists. ``ActuatorBase.compute()`` receives only
    ``control_action``, ``joint_pos`` and ``joint_vel`` -- it has no handle on
    the articulation, so it cannot ask PhysX for G(q) itself. Something that
    *does* hold the articulation has to push the value down to it. This class
    is that push, and nothing else: it overrides one public method, copies
    ``root_physx_view.get_gravity_compensation_forces()`` into each actuator's
    ``gravity_torque`` buffer, and delegates. The actuator then adds it to
    tau_ff (see :func:`maker_arm_lab.core.mit_law.mit_mode_feedforward`), which
    is exactly where a real RobStride host puts it: the CAN packet's feed-forward
    field. Nothing here has a counterpart the real robot lacks.

    Alternatives that were rejected:

    * Overriding the private ``Articulation._apply_actuator_model()`` and
      adding G(q) to ``control_action.joint_efforts`` before the actuator sees
      it. Marginally purer -- the actuator would stay unaware of gravity, as a
      real motor is -- but it means either copying ~35 lines of upstream
      private code into this repo, or transiently mutating and restoring the
      shared ``_data.joint_effort_target`` buffer around a ``super()`` call.
      Overriding one documented public method is worth the small loss of
      purity.
    * Disabling gravity on the arm bodies (``disable_gravity=True``, which is
      what upstream's own cfg for this asset does). That is not gravity
      compensation, it is deleting gravity: payloads become weightless and the
      sim stops predicting anything about the real arm. ``MAKER_ARM_CFG``
      deliberately keeps ``disable_gravity=False``.
    * An analytical gravity model in ``core/``. The USD's mass/inertia/COM
      values are the ground truth the simulation itself integrates; re-deriving
      them in Python would add a second, silently divergent model.
    * A dedicated zero-dimensional action term that calls
      ``set_joint_effort_target(G(q))`` (the shape of upstream's
      ``PinkInverseKinematicsAction._apply_gravity_compensation``). It has to be
      added to every variant's ``ActionsCfg``, and it collides with
      ``JointEffortAction``, which *overwrites* rather than accumulates into the
      same buffer -- so the two action terms would silently cancel each other
      depending on declaration order.

    Actuators without a ``gravity_torque`` attribute (e.g. the gripper's
    ``ImplicitActuator``, whose PD loop runs inside PhysX where the effort
    target has different semantics) are skipped.
    """

    def write_data_to_sim(self):
        self._refresh_gravity_torque()
        super().write_data_to_sim()

    def _refresh_gravity_torque(self):
        """Copy G(q) into every gravity-compensated actuator's buffer."""
        gravity = None
        for actuator in self.actuators.values():
            buffer = getattr(actuator, "gravity_torque", None)
            if buffer is None:
                continue
            # Read live rather than caching an enabled/disabled list at init,
            # so an A/B measurement can flip the flag on a running env.
            if not getattr(actuator.cfg, "gravity_compensation", False):
                buffer.zero_()
                continue
            if gravity is None:
                # G(q) from tau = M(q)*qdd + C(q,qd)*qd + G(q): the torque that
                # must be APPLIED to hold the articulation static, so it is
                # added to tau_ff with a plus sign. Shape is (num_envs,
                # num_dofs) for a fixed-base articulation and
                # (num_envs, 6 + num_dofs) for a floating-base one, where the
                # leading 6 are the free root -- documented in
                # omni.physics.tensors' ArticulationView.
                gravity = self.root_physx_view.get_gravity_compensation_forces()
                if not self.is_fixed_base:
                    gravity = gravity[:, 6:]
            # PhysX hands back the same cached tensor on every call, so copy
            # into the actuator's own buffer instead of aliasing it.
            buffer[:] = gravity[:, actuator.joint_indices]
            # Tells the actuator a real gravity source exists; without this it
            # warns that its cfg flag is doing nothing.
            actuator.gravity_torque_written = True

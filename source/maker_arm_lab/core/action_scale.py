"""Per-axis scale of the IK-relative action term, carried over from openyam-lab.

Torch-free and isaaclab-free on purpose. This is a constant every consumer
must agree on, and this milestone's rule is that such constants are imported
rather than retyped: it lives here so the env cfg that installs it
(``tasks.manager_based.lego_sort.variants``), the attitude-hold anti-windup
band derived from it (``core.attitude_hold.DEFAULT_MAX_LEAD``) and the CPU
tests that describe the arm through it all read the same object. Before
Task 13's final review the three carried three independent copies, one of
which conceded in a comment that "nothing here can catch the two drifting
apart".

Same reasoning that put ``TOP_DOWN_X_LIMIT`` in ``core.layout`` (Task 10) and
the overhead-camera derivation there (Task 12).
"""

# Per-axis scale for the IK-relative action: (dx, dy, dz, drx, dry, drz) that a
# full-deflection action (raw = +-1) commands in ONE 30 Hz control step.
#
# Carried over from openyam-lab (measured there on the OpenYAM at 30 Hz). On
# the Maker Arm the profile's hold gains are lower, so these per-step steps
# may saturate or under-track; ``scripts/scripted_grasp_check.py`` reports
# per-joint torque saturation and, if it exceeds 10 % on any joint, this
# constant is re-measured with the same 120-step per-axis protocol and the
# result recorded in docs/ms1-maker-arm-sim.md.
IK_SCALE: tuple[float, ...] = (0.002, 0.002, 0.002, 0.01, 0.01, 0.01)

# Rotation entries alone, the slice every rotation consumer wants.
IK_SCALE_ROT: tuple[float, ...] = IK_SCALE[3:6]

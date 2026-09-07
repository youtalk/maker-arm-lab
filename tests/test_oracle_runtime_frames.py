"""The pieces of oracle_runtime that are pure math: the pick-offset inversion the planner
solves its IK targets from, and the leg resolution it cuts its waypoints with. Loaded by
file path with an isaaclab stand-in, like test_mit_mode_actuator.py.

This file used to pin ``canonical_to_body``, which composed the oracle's canonical tool
frame with the arm's body frame for the quaternion the differential-IK attitude servo held.
Task D5 replaced that servo with ``core.global_ik``, which constrains tilt, lean azimuth and
jaw heading as three scalars and needs no rotation matrix assembled, so both the function and
its tests are gone rather than kept dead. What replaced them is below.
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.core.global_ik import JointTrim, tool_axis_world
from maker_arm_lab.robots import frames

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/maker_arm_lab/tasks/manager_based/lego_sort/oracle_runtime.py"
)


def _load():
    names = ["isaaclab", "isaaclab.utils", "isaaclab.utils.math"]
    mods = {n: types.ModuleType(n) for n in names}
    pkg = types.ModuleType("_lego_sort_stub")
    pkg.__path__ = [str(_PATH.parent)]
    stubs = {**mods, "_lego_sort_stub": pkg}
    # oracle_runtime imports `.mdp.recorders` relatively; give it a light stand-in.
    rec = types.ModuleType("_lego_sort_stub.mdp.recorders")
    rec.PHASE_TO_INT = {}
    mdp = types.ModuleType("_lego_sort_stub.mdp")
    mdp.__path__ = []
    stubs.update({"_lego_sort_stub.mdp": mdp, "_lego_sort_stub.mdp.recorders": rec})
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_lego_sort_stub.oracle_runtime", _PATH)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "_lego_sort_stub"
        spec.loader.exec_module(module)
        return module
    finally:
        for n in stubs:
            sys.modules.pop(n, None)
        sys.modules.pop("_lego_sort_stub.oracle_runtime", None)


_rt = _load()


def _tool_frame(tilt: float, heading: float) -> torch.Tensor:
    """The `grasp_center` body rotation the solver's three scalars pin down.

    Built independently of the code under test, from the constraints
    ``core.global_ik.GlobalIK._residual_batch`` actually imposes: the tool axis is
    ``tool_axis_world(tilt, lean_azimuth)``, and the jaw axis is the unit vector
    perpendicular to it whose horizontal heading is ``heading``. Columns are body x, y, z.
    """
    body_z = -tool_axis_world(tilt, _rt.LEAN_AZIMUTH)  # TOOL_AXIS_BODY is body -z
    flat = torch.tensor([math.cos(heading), math.sin(heading), 0.0], dtype=torch.float64)
    body_y = flat - body_z * (flat @ body_z)
    body_y = body_y / body_y.norm()
    body_x = torch.linalg.cross(body_y, body_z)
    return torch.stack([body_x, body_y, body_z], dim=1)


@pytest.mark.parametrize("tilt", [0.0, math.radians(0.5), frames.GRASP_TILT_RAD, 1.40])
@pytest.mark.parametrize("heading", [0.0, 0.9, -2.3])
def test_grasp_center_for_inverts_the_pick_offset(tilt, heading):
    """`grasp_center_for` must be the exact inverse of `OracleRig.pick_point`'s offset.

    That is the whole contract: the planner solves IK for `grasp_center`, but every target
    core.oracle hands it -- the brick, the bin, RETREAT_POINT -- is expressed at the PICK
    point. A sign error here would put the jaws a tool length off the brick, and only a GPU
    run would see it.
    """
    pick = torch.tensor([0.24, -0.06, 0.11], dtype=torch.float64)
    offset = torch.tensor(frames.PICK_OFFSET_MEASURED, dtype=torch.float64)
    center = torch.tensor(
        _rt.grasp_center_for(pick, tilt, heading, float(offset[2])), dtype=torch.float64
    )
    assert torch.allclose(center + _tool_frame(tilt, heading) @ offset, pick, atol=1e-12)


def test_grasp_center_sits_above_the_pick_point():
    """The jaws reach PAST grasp_center along the tool axis, so for a near-level tool the
    frame is the higher of the two -- the direction robots.frames.PICK_OFFSET_MEASURED's
    negative z encodes."""
    pick = torch.tensor([0.24, 0.0, 0.0048], dtype=torch.float64)
    center = _rt.grasp_center_for(pick, math.radians(0.5), 0.0, frames.PICK_OFFSET_MEASURED[2])
    assert center[2] > float(pick[2]) + 0.03


def test_leg_resolution_cuts_by_geometry():
    """A leg is cut fine enough for core.global_ik's continuity cap: a 0.12 m leg at the 3 mm
    nominal spacing is 40 waypoints, well inside DEFAULT_MAX_STEPS' walk budget."""
    count, dwell = _rt.leg_resolution(max_steps=300, gain_scale=1.0, distance=0.12)
    assert count == 40
    assert dwell == _rt.WAYPOINT_HOLD_STEPS


def test_leg_resolution_never_exceeds_the_phase_budget():
    """RAISE is the binding case: 60 steps of budget against a leg that applies the whole of
    RELEASE_TILT_RAD. The walk must fit inside WALK_BUDGET_FRACTION of that budget -- a
    truncated plan would leave EXTEND starting from a pose RAISE never reached."""
    count, dwell = _rt.leg_resolution(
        max_steps=60, gain_scale=1.0, distance=0.10, tilt_change=1.40 - frames.GRASP_TILT_RAD
    )
    assert count * dwell <= int(_rt.WALK_BUDGET_FRACTION * 60)


def test_leg_resolution_slows_the_walk_for_a_low_gain_draw():
    """``EpisodeDraw.gain_scale`` used to scale the servo's gain; on the planner it cuts the
    leg finer, so a slow draw walks the same leg in more waypoints."""
    fast, _ = _rt.leg_resolution(max_steps=300, gain_scale=1.0, distance=0.12)
    slow, _ = _rt.leg_resolution(max_steps=300, gain_scale=0.7, distance=0.12)
    assert slow > fast


def test_leg_resolution_has_a_floor_of_two_waypoints():
    """A leg with no geometry at all (a re-plan onto the pose already commanded) still has a
    start and an end, so the walk is never empty and `step()` always has a waypoint to pop."""
    count, dwell = _rt.leg_resolution(max_steps=20, gain_scale=1.0)
    assert count == 2
    assert dwell >= 1


def test_heading_relaxation_tries_the_held_heading_first():
    """The ladder's first rung must be no relaxation at all.

    That is what makes it inert on every leg that solves as commanded: a carry leg only ever
    moves the jaw heading when holding it yields no walkable path (``_plan_carry``).
    """
    assert _rt.HEADING_RELAXATION_RAD[0] == 0.0


def test_heading_relaxation_widens_symmetrically_and_stops_at_60_degrees():
    """Both signs at each width, widening, and bounded: the measured failures needed up to
    60 deg of relaxation and none needed more, so the ladder is not open-ended."""
    offsets = _rt.HEADING_RELAXATION_RAD[1:]
    assert [round(math.degrees(o), 3) for o in offsets] == [
        -10.0,
        10.0,
        -20.0,
        20.0,
        -30.0,
        30.0,
        -45.0,
        45.0,
        -60.0,
        60.0,
    ]


def test_leg_resolution_counts_a_heading_sweep_like_a_tilt_sweep():
    """A carry leg that relaxes the heading rotates the wrist along the leg, and a leg cut
    for position alone would put that rotation over core.global_ik's continuity cap."""
    plain, _ = _rt.leg_resolution(max_steps=300, gain_scale=1.0, distance=0.01)
    swept, _ = _rt.leg_resolution(
        max_steps=300, gain_scale=1.0, distance=0.01, heading_change=math.radians(60.0)
    )
    assert swept > plain


def test_the_command_path_can_leave_the_joint_box_so_the_clamp_flag_can_fire():
    """`OracleRig.step` must NOT clamp its command to the joint limits, and this is the
    property that makes the dataset's clamp fraction a real criterion.

    The action term clamps the target through maker_arm_rs and RECORDS that it did
    (`obs/clamped`), and `core.profile.sim_profile` hands that clamp the same URDF bounds the
    rig would have used -- so clamping first changes nothing about what the arm is commanded
    and everything about what the dataset can tell you. Task D5 shipped exactly that mistake:
    with a pre-clamp, `ClampedJointPositionAction.clamped` could not fire on oracle-generated
    actions at all, and a measured `clamped_mean = 0.0000` over 4194 samples was a tautology
    rather than evidence of health.

    What this pins is the premise the flag needs: the command path CAN leave the joint box.
    `JointTrim`'s bias has 0.35 rad of authority and winds up against a persistent tracking
    error -- exactly what a leg the arm cannot follow produces -- so a waypoint at a stop plus
    a wound-up trim is outside the box. `tests/test_profile.py` pins the other half, that the
    clamp flags such a target; `scripts/oracle_check.py --clamp_probe` observes the whole
    chain firing on GPU.
    """
    limits = urdf_prep.arm_joint_limits()
    at_stop = torch.tensor([hi for _, hi in limits], dtype=torch.float64)
    trim = JointTrim()
    # A leg the arm lags by 0.2 rad, held long enough for the integrator to reach its clamp.
    for _ in range(500):
        trim.update(at_stop, at_stop - 0.2)
    command = trim.command(at_stop)
    assert float(torch.max(command - at_stop)) == pytest.approx(trim.limit)
    assert any(float(command[i]) > hi for i, (_, hi) in enumerate(limits))

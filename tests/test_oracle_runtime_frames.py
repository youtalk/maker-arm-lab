"""The pieces of oracle_runtime that are pure math: the pick-offset inversion the planner
solves its IK targets from, and the leg resolution it cuts its waypoints with. Loaded by
file path with an isaaclab stand-in, like test_mit_mode_actuator.py.

This file used to pin ``canonical_to_body``, which composed the oracle's canonical tool
frame with the arm's body frame for the quaternion the differential-IK attitude servo held.
Task D5 replaced that servo with ``core.global_ik``, which constrains tilt, lean azimuth and
jaw heading as three scalars and needs no rotation matrix assembled, so both the function and
its tests are gone rather than kept dead. What replaced them is below.
"""

import collections
import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import pytest
import torch

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.core import layout, oracle
from maker_arm_lab.core.global_ik import JointTrim, tool_axis_world
from maker_arm_lab.robots import frames

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/maker_arm_lab/tasks/manager_based/lego_sort/oracle_runtime.py"
)


def _quat_apply(quat, vec):
    """``isaaclab.utils.math.quat_apply`` for the stub: rotate ``vec`` by wxyz ``quat``.

    The standard form, not an approximation -- the tests below drive it with identity
    quaternions, where it is the identity, but writing the real thing means a stub that
    silently rotates nothing cannot be mistaken for one that works.
    """
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    if q.shape[0] == 1 and v.shape[0] > 1:
        q = q.expand(v.shape[0], 4)
    if v.shape[0] == 1 and q.shape[0] > 1:
        v = v.expand(q.shape[0], 3)
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return (v + w * t + torch.linalg.cross(xyz, t, dim=-1)).reshape(vec.shape)


def _load():
    names = ["isaaclab", "isaaclab.utils", "isaaclab.utils.math"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["isaaclab.utils.math"].quat_apply = _quat_apply
    pkg = types.ModuleType("_lego_sort_stub")
    pkg.__path__ = [str(_PATH.parent)]
    stubs = {**mods, "_lego_sort_stub": pkg}
    # oracle_runtime imports `.mdp` and `.mdp.recorders` relatively; give them stand-ins.
    # PHASE_TO_INT is a defaultdict rather than the real schema: the tests here care that
    # `step()` stashes SOMETHING per phase, not what the integers are (mdp/recorders.py owns
    # that, and tests/test_recorders.py pins it).
    rec = types.ModuleType("_lego_sort_stub.mdp.recorders")
    rec.PHASE_TO_INT = collections.defaultdict(lambda: -1)
    mdp = types.ModuleType("_lego_sort_stub.mdp")
    mdp.__path__ = []
    mdp.per_brick_stowed = lambda env: env.stowed
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


# --------------------------------------------------------------------------------------
# A stub env, so `OracleRig.step()` itself can be exercised without Isaac Sim.
#
# It exists for one reason: the tests above pin PREMISES (the trim can leave the joint box,
# the geometry inverts), and a premise is necessary but not sufficient. What the dataset's
# clamp fraction actually depends on is the WIRING -- that `step()` hands the action term a
# command it has not already clamped -- and no test that stops short of calling `step()` can
# see that. Task D5's review demonstrated the gap by reintroducing the pre-clamp and watching
# the whole suite stay green.
# --------------------------------------------------------------------------------------

_ARM = list(frames.ARM_JOINT_NAMES)
_GRIPPER = list(frames.GRIPPER_JOINT_NAMES)
_JOINTS = _ARM + _GRIPPER


class _StubArticulation:
    """Just the ``data`` fields and name lookups ``OracleRig`` reads."""

    def __init__(self, num_envs: int):
        limits = urdf_prep.arm_joint_limits()
        self.joint_names = _JOINTS
        self.data = types.SimpleNamespace(
            joint_pos=torch.zeros(num_envs, len(_JOINTS)),
            joint_vel=torch.zeros(num_envs, len(_JOINTS)),
            joint_pos_limits=torch.tensor(
                [[[lo, hi] for lo, hi in limits] + [[0.0, 0.06]] * len(_GRIPPER)] * num_envs
            ),
            body_pos_w=torch.tensor([[[0.25, 0.0, 0.2]]] * num_envs),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * num_envs),
            computed_torque=torch.zeros(num_envs, len(_JOINTS)),
            applied_torque=torch.zeros(num_envs, len(_JOINTS)),
        )

    def find_bodies(self, name):
        return ([0], [name])

    def find_joints(self, names, preserve_order=False):
        return ([self.joint_names.index(n) for n in names], list(names))


class _StubScene:
    def __init__(self, num_envs: int, num_bricks: int):
        self.env_origins = torch.zeros(num_envs, 3)
        self._robot = _StubArticulation(num_envs)
        contact = types.SimpleNamespace(
            data=types.SimpleNamespace(force_matrix_w=torch.zeros(num_envs, 1, num_bricks, 3))
        )
        self._items = {"robot": self._robot}
        for side in ("left", "right"):
            self._items[f"finger_{side}_brick"] = contact
        for b in range(num_bricks):
            self._items[f"brick_{b}"] = types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=torch.tensor([[0.2 + 0.01 * b, 0.0, 0.005]] * num_envs),
                    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs),
                    root_lin_vel_w=torch.zeros(num_envs, 3),
                )
            )

    def __getitem__(self, key):
        return self._items[key]


class _StubEnv:
    """Captures the action ``OracleRig.step()`` emits, and nothing else moves."""

    def __init__(self, num_envs: int = 1, num_bricks: int = 4):
        self.device = "cpu"
        self.num_envs = num_envs
        self.cfg = types.SimpleNamespace(num_bricks=num_bricks)
        self.scene = _StubScene(num_envs, num_bricks)
        term = types.SimpleNamespace(
            _joint_names=_ARM, clamped=torch.zeros(num_envs, dtype=torch.bool)
        )
        self.action_manager = types.SimpleNamespace(total_action_dim=7, get_term=lambda name: term)
        self.stowed = torch.zeros(num_envs, num_bricks, dtype=torch.bool)
        self.success = torch.zeros(num_envs, dtype=torch.bool)
        self.termination_manager = types.SimpleNamespace(
            active_terms=["success", "time_out"], get_term=lambda name: self.success
        )
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)
        self.last_action = None

    def step(self, action):
        self.last_action = action.clone()
        return (None, None, self.terminated.clone(), torch.zeros_like(self.terminated), None)


def _stub_rig(num_envs: int = 1, num_bricks: int = 4):
    env = _StubEnv(num_envs, num_bricks)
    rig = _rt.OracleRig(
        env,
        layout.DEFAULT_BIN_LAYOUT,
        oracle.OracleNoise(),
        random.Random(0),
    )
    return env, rig


def test_step_hands_the_action_term_a_command_it_has_not_clamped():
    """`OracleRig.step()` must pass its command through UNCLAMPED, so the action term's own
    clamp is the one that acts and, crucially, RECORDS (`obs/clamped`).

    This is the wiring test, and it is the one that goes red if the pre-clamp comes back.
    The rig used to clamp to the articulation's joint limits, and `core.profile.sim_profile`
    hands the Rust clamp those same bounds, so `ClampedJointPositionAction.clamped` could not
    fire on oracle-generated actions at all -- making the spec's clamp-fraction criterion
    (section 4.3) unable to fail. The premise that the command CAN leave the joint box is
    pinned separately above; this pins that `step()` does not undo it.
    """
    env, rig = _stub_rig()
    upper = rig.limits[:, 1].double()
    at_stop = upper.clone()
    rig.q_cmd[0] = at_stop
    # Wind the trim against a persistent tracking error, as a leg the arm lags does.
    for _ in range(500):
        rig.trims[0].update(at_stop, at_stop - 0.2)
    expected = rig.trims[0].command(at_stop)

    rig.step()

    sent = env.last_action[0, :6].double()
    assert torch.allclose(sent, expected, atol=1e-5)
    assert bool((sent > upper + 0.3).all()), "the rig clamped its own command"


def test_episode_stats_freeze_when_the_episode_ends_not_when_they_are_read():
    """The outcome fields must describe the episode that ENDED, not the env's state now.

    `succeeded` and `bricks_delivered` come from env state the next `env.step()` overwrites
    for the whole batch, rearmed or not -- so with more than one env the natural pattern
    (finish env A, keep stepping so env B can finish, then read A) silently reported A's NEXT
    episode. `oracle_check.py` never saw it because it runs one env and reads immediately.
    """
    env, rig = _stub_rig(num_envs=2)
    env.stowed[:] = True
    env.success[:] = True
    env.terminated[0] = True

    rig.step()  # env 0 finishes here; env 1 keeps going

    # The world moves on, as it does while the rest of the batch runs.
    env.stowed[:] = False
    env.success[:] = False
    env.terminated[0] = False
    rig.step()
    rig.step()

    stats = rig.episode_stats(0)
    assert stats["succeeded"] is True
    assert stats["bricks_delivered"] == rig.num_bricks
    # env 1 never finished, so its fields are the live ones and follow the world.
    assert rig.episode_stats(1)["succeeded"] is False
    # ... and rearming env 0 releases the snapshot.
    rig.rearm([0])
    assert rig.episode_stats(0)["succeeded"] is False


def test_a_watchdog_abandon_freezes_the_outcome_on_the_step_it_is_raised():
    """`_freeze_outcomes` must run AFTER `_run_watchdog`, not before it.

    An env the watchdog gives up on is not `done` -- Isaac Lab never terminated it -- so the
    only thing that marks its episode over is `abandoned`, and that flag is raised inside
    `_run_watchdog`, late in the same `step()`. Freezing before the watchdog would miss it
    for that step and snapshot on the NEXT one instead, capturing a success term and a brick
    sample that the intervening `env.step()` has already recomputed for the whole batch.

    The ordering was asserted in a comment on `_freeze_outcomes` and pinned by nothing:
    moving the call earlier left the suite green. `scripts/generate_batch.py` is the first
    caller to run the batch pattern this protects -- abandon env A, keep stepping so env B
    can finish, then read A -- so it is pinned here.
    """
    env, rig = _stub_rig(num_envs=2)
    # The stub's zero pose already sits ON two of the URDF's stops (link_003_joint's upper
    # bound and link_004_joint's lower bound are both 0.0), which is watchdog_trips' own
    # "near a limit" criterion -- so env 0 trips every step and is abandoned after
    # WATCHDOG_TRIP_STRIKES of them. Env 1 is moved to mid-range so it keeps running, which
    # is what makes this the batch pattern rather than a single-env read.
    joint_pos = env.scene._robot.data.joint_pos
    joint_pos[1, rig.arm_ids] = rig.limits.mean(dim=1).float()
    env.stowed[0, :2] = True

    for _ in range(_rt.WATCHDOG_TRIP_STRIKES):
        rig.step()
    assert bool(rig.abandoned[0]), "the watchdog did not abandon env 0"
    assert not bool(rig.abandoned[1])

    # The world moves on, as it does while the rest of the batch runs.
    env.stowed[:] = True
    env.success[:] = True
    rig.step()

    stats = rig.episode_stats(0)
    assert stats["succeeded"] is False
    assert stats["bricks_delivered"] == 2
    assert stats["abandon"] == "watchdog"


# --------------------------------------------------------------------------------------
# RAISE lookahead: a RAISE leg is only worth walking if EXTEND can still be planned from
# where it ends. The poses below were RECORDED on GPU (2026-09-07, instrumented 12-env
# datagen batch, run_seed 3081035816): the arm's measured joint vector at the end of LIFT,
# the RAISE target core.oracle asked for, and the release target EXTEND then aimed at.
# --------------------------------------------------------------------------------------


def _carry_setup(rig, q_lift_end, heading_deg):
    rig.q_cmd[0] = torch.tensor(q_lift_end, dtype=torch.float64)
    rig.commanded_yaw[0] = math.radians(heading_deg)
    rig.commanded_tilt[0] = frames.GRASP_TILT_RAD
    rig.oracles[0].draw = oracle.EpisodeDraw(0.12, 0.0, 0.0, 0.0, 1.0, 0.05, 0.0, 0.0)
    rig.oracles[0].target_brick = 0
    rig.oracles[0].phase = oracle.Phase.RAISE


def _raise_goal(target):
    return oracle.Goal(
        oracle.Phase.RAISE,
        torch.tensor(target),
        None,
        oracle.RAISE_TILT_RAD,
        -1.0,
        oracle.RAISE_STOP_POS,
        0,
        oracle.RAISE_MAX_STEPS,
        0,
    )


def test_raise_is_refused_when_extend_cannot_be_planned_from_where_it_ends():
    """Env 3, brick 0 of the instrumented batch: a near pick (radius 0.155 m, azimuth -11
    deg) bound for the far -y bin. RAISE's Cartesian ladder failed and `_plan_direct` walked
    the arm to a joint-space endpoint 1.25 rad away in base yaw -- legal under
    MAX_LEG_JOINT_JUMP, but a branch from which EXTEND had no continuous path at any heading
    rung, so RELEASE opened the jaws over the scatter band and the episode timed out one
    brick short. From the LIFT pose the same EXTEND plans at the held heading.
    """
    _env, rig = _stub_rig(num_envs=1, num_bricks=6)
    bl = layout.BinLayout(
        bin_x=0.443315375851723,
        bin_ys=(
            -0.22119301103713415,
            -0.07162949025824511,
            0.08437825458493829,
            0.22237051165126664,
        ),
        slot_colors=(0, 3, 1, 2),
    )
    rig.bl = bl
    rig.oracles[0].bl = bl
    _carry_setup(rig, [0.334, -1.08, 1.295, -1.28, -0.113, -0.163], 64.1)

    rig._plan_leg(0, _raise_goal([0.1524, -0.0295, oracle.RAISE_CLEARANCE_Z]))

    assert rig.unplanned[0], "RAISE must be refused: EXTEND is unplannable from its end pose"
    assert rig.plans[0] == []
    assert rig.stats[0]["unplanned_legs"] == 1
    assert rig.stats[0]["raise_vetoed"] == 1
    # And the veto is what saves the brick: EXTEND from the LIFT pose plans.
    extend = rig.oracles[0].extend_goal()
    rig._plan_leg(0, extend)
    assert not rig.unplanned[0]
    assert len(rig.plans[0]) > 0


def test_raise_is_kept_when_extend_plans_from_where_it_ends():
    """Env 3, brick 1 of the same batch: a mid-band pick whose RAISE planned on the
    Cartesian ladder and whose EXTEND then converged live in 171 steps. The lookahead must
    not veto a RAISE that EXTEND can follow.
    """
    _env, rig = _stub_rig(num_envs=1, num_bricks=6)
    bl = layout.BinLayout(
        bin_x=0.443315375851723,
        bin_ys=(
            -0.22119301103713415,
            -0.07162949025824511,
            0.08437825458493829,
            0.22237051165126664,
        ),
        slot_colors=(0, 3, 1, 2),
    )
    rig.bl = bl
    rig.oracles[0].bl = bl
    _carry_setup(rig, [-0.296, -1.363, 1.462, -1.162, 0.101, 1.804], 210.9)
    rig.oracles[0].target_brick = 1

    rig._plan_leg(0, _raise_goal([0.2142, 0.0441, oracle.RAISE_CLEARANCE_Z]))

    assert not rig.unplanned[0]
    assert len(rig.plans[0]) > 0
    assert rig.stats[0]["raise_vetoed"] == 0
    assert rig.stats[0]["unplanned_legs"] == 0


# --------------------------------------------------------------------------------------
# HOVER height ladder: the drawn hover height is a via pose, not a grasp parameter, and
# at the band's edges it is what makes a pick unplannable. Same instrumented batch as above.
# --------------------------------------------------------------------------------------


def _hover_goal(target, yaw):
    return oracle.Goal(
        oracle.Phase.HOVER,
        torch.tensor(target),
        yaw,
        0.0,
        1.0,
        0.005,
        0,
        oracle.DEFAULT_MAX_STEPS,
        0,
    )


def _pick_setup(rig, q_before_hover, brick_xy, brick_index=0):
    rig.q_cmd[0] = torch.tensor(q_before_hover, dtype=torch.float64)
    rig.commanded_tilt[0] = frames.GRASP_TILT_RAD
    rig.oracles[0].draw = oracle.EpisodeDraw(0.12, 0.0, 0.0, 0.0, 1.0, 0.05, 0.0, 0.0)
    rig.oracles[0].target_brick = brick_index
    rig.oracles[0].phase = oracle.Phase.HOVER
    brick = rig.env.scene[f"brick_{brick_index}"]
    brick.data.root_pos_w[0] = torch.tensor([brick_xy[0], brick_xy[1], 0.0048])
    rig.brick_start[0] = rig._brick_pos(0, brick_index)
    rig.oracles[0]._brick_start = rig.brick_start[0].clone()


def test_hover_is_lowered_when_the_drawn_height_has_no_ik_solution():
    """Env 11, brick 1: a band-edge pick (radius 0.274 m, azimuth +40 deg) whose HOVER had
    no solution at the drawn 0.138 m hover height at any restart count, so DESCEND, CLOSE
    and LIFT all ran unplanned, three attempts over, and the episode was abandoned on
    `grasp_retries_exhausted`. At 0.10 m above the brick the same hover solves with 0.34 rad
    of joint margin and the descend that follows plans on its first rung.
    """
    _env, rig = _stub_rig(num_envs=1, num_bricks=6)
    _pick_setup(rig, [0.143, -0.55, 0.957, -0.401, -0.141, -1.321], (0.2133, 0.177), 1)
    yaw = math.radians(428.2) - math.pi / 2
    goal = _hover_goal([0.2133, 0.177, 0.0048 + 0.1334], yaw)

    rig._plan_leg(0, goal)

    assert not rig.unplanned[0]
    assert rig.stats[0]["hover_lowered"] == 1
    assert float(goal.target_pos[2]) == pytest.approx(0.0048 + 0.10)
    # The lowered hover is what the plan actually walks to, so HOVER can still converge.
    end_pick, _ = rig._commanded_pick_pose(0, rig.plans[0][-1])
    assert float(end_pick[2]) == pytest.approx(float(goal.target_pos[2]), abs=1e-3)
    # And DESCEND plans from there.
    rig.q_cmd[0] = rig.plans[0][-1]
    descend = rig.oracles[0].descend_goal(yaw)
    rig._plan_leg(0, descend)
    assert not rig.unplanned[0]


def test_hover_is_lowered_when_no_descend_can_follow_it():
    """Env 5, brick 5: a near pick (radius 0.146 m) whose HOVER solved at the drawn height
    but on a branch with the wrist 0.18 rad from its stop, from which the descend ladder had
    no continuous path -- three times, then abandoned. The ladder must look one leg ahead:
    a hover DESCEND cannot follow is not a hover.
    """
    _env, rig = _stub_rig(num_envs=1, num_bricks=6)
    _pick_setup(rig, [-0.658, -1.163, 1.503, -1.451, 0.212, 2.513], (0.1365, 0.0521), 5)
    yaw = math.radians(-89.9) - math.pi / 2
    goal = _hover_goal([0.1365, 0.0521, 0.0048 + 0.1237], yaw)

    rig._plan_leg(0, goal)

    assert not rig.unplanned[0]
    assert rig.stats[0]["hover_lowered"] == 1
    assert float(goal.target_pos[2]) < 0.0048 + 0.1237
    rig.q_cmd[0] = rig.plans[0][-1]
    rig._plan_leg(0, rig.oracles[0].descend_goal(yaw))
    assert not rig.unplanned[0]


def test_hover_keeps_the_drawn_height_when_it_plans():
    """A mid-band pick from the home pose: the ladder's first rung IS the drawn height, so a
    pick that never needed lowering is bit-for-bit unaffected."""
    _env, rig = _stub_rig(num_envs=1, num_bricks=6)
    home = [frames.HOME_JOINT_POS[n] for n in frames.ARM_JOINT_NAMES]
    _pick_setup(rig, home, (0.22, 0.04), 1)
    goal = _hover_goal([0.22, 0.04, 0.0048 + 0.12], 0.3)

    rig._plan_leg(0, goal)

    assert not rig.unplanned[0]
    assert rig.stats[0]["hover_lowered"] == 0
    assert float(goal.target_pos[2]) == pytest.approx(0.0048 + 0.12)

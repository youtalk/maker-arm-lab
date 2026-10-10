"""MS5 design section 4.3 and section 8: RealEnv's surface, its step and its watchdogs."""

import math

import numpy as np
import pytest
import torch

from maker_arm_lab.assets.urdf_prep import GRIPPER_JOINT_NAMES
from maker_arm_lab.core import cameras
from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.gate import policy_state
from maker_arm_lab.core.global_ik import GlobalIK
from maker_arm_lab.core.profile import gripper_angle, sim_profile
from maker_arm_lab.core.quat import rpy_from_quat
from maker_arm_lab.core.torque import pick_force
from maker_arm_lab.real import camera_model
from maker_arm_lab.real.rig import RigConfig
from maker_arm_lab.robots.frames import ARM_JOINT_REGEX, GRIPPER_OPEN_POS, JAW_FULL_GAP, TOOL_BODY
from real_fakes import HOME, FakeArm, FakeClock, FakeOverhead, FakeWrist, make_env

PERIOD = 1.0 / 30.0


def _action(gap=0.05):
    a = torch.zeros(1, 7)
    a[0, :6] = torch.tensor(HOME)
    a[0, 6] = gap
    return a


def test_the_surface_is_the_mj_envs():
    env, _, _ = make_env()
    robot = env.scene["robot"]
    assert robot.find_joints([ARM_JOINT_REGEX])[0] == [0, 1, 2, 3, 4, 5]
    assert robot.find_joints(list(GRIPPER_JOINT_NAMES))[0] == [6, 7]
    assert robot.find_bodies(TOOL_BODY) == ([0], [TOOL_BODY])
    assert robot.data.joint_pos_limits.shape == (1, 8, 2)
    assert env.obs_buf["images"]["overhead_rgb"].shape == (1, 224, 224, 3)
    assert env.obs_buf["context"].shape == (1, 16) and env.cfg.num_bricks == 6
    assert env.action_manager.total_action_dim == 7
    assert not bool(env.termination_manager.get_term("success")[0])


def test_construction_reads_the_arm_through_fk_and_the_transmission():
    env, _, _ = make_env()
    data = env.scene["robot"].data
    assert data.joint_pos[0, :6].tolist() == pytest.approx(HOME, abs=1e-6)
    assert data.joint_pos[0, 6].item() == pytest.approx(GRIPPER_OPEN_POS, abs=1e-7)
    pos, _ = GlobalIK().fk(torch.tensor(HOME, dtype=torch.float64))
    assert data.body_pos_w[0, 0].tolist() == pytest.approx(pos.tolist(), abs=1e-6)


def test_the_context_serves_the_nominal_camera_and_the_target_row_is_a_view():
    env, _, _ = make_env()
    want = [*cameras.OVERHEAD_CAM_POS, *rpy_from_quat(cameras.OVERHEAD_CAM_ROT)]
    assert env.obs_buf["context"][0, :6].tolist() == pytest.approx(want, abs=1e-6)
    env.context_target[0] = 1.0
    assert env.obs_buf["context"][0, 9:].tolist() == [1.0] * 7
    env.perceived_box_poses = np.array([[[0.1, 0.2, 0.3]] * 4])
    env.set_context_box(0, 2)
    assert env.obs_buf["context"][0, 6:9].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_a_step_pushes_the_clamped_row_one_period_ahead_and_waits_for_the_tick():
    env, arm, clock = make_env()
    t0 = arm.now()
    _, _, terminated, truncated, _ = env.step(_action(0.05))
    due, dt, row = arm.pushes[-1]
    assert dt == pytest.approx(PERIOD) and due == pytest.approx(t0 + PERIOD)
    assert row[:6] == pytest.approx(HOME, abs=1e-6)
    assert row[6] == pytest.approx(gripper_angle(sim_profile(), 0.05 / JAW_FULL_GAP))
    assert clock.sleeps[-1] == pytest.approx(PERIOD)
    assert not bool(terminated[0]) and not bool(truncated[0])
    assert len(env.ticks.ticks) == 1


def test_half_dilation_doubles_the_period():
    env, arm, clock = make_env(dilation=0.5)
    env.step(_action())
    assert arm.pushes[-1][1] == pytest.approx(2 * PERIOD) and clock.sleeps[-1] == pytest.approx(
        2 * PERIOD
    )


def test_the_episode_truncates_at_its_length():
    env, _, _ = make_env(episode_length_s=2 * PERIOD)
    assert not bool(env.step(_action())[3][0])
    assert bool(env.step(_action())[3][0])


def test_a_gripper_angle_past_its_range_keeps_the_state_fraction_in_range():
    g = sim_profile()["joints"][6]
    for angle, want in ((g["q_hi"] + 0.5, 1.0), (g["q_lo"] - 0.5, 0.0)):
        clock = FakeClock()
        env, _, _ = make_env(clock=clock, arm=FakeArm(clock, gripper=angle))
        assert policy_state(env.scene["robot"].data.joint_pos)[0, 6].item() == pytest.approx(want)


def test_the_marker_stamps_at_the_nominal_projection_and_nan_stamps_nothing():
    env, _, _ = make_env()
    before = env.obs_buf["images"]["overhead_rgb"].clone()
    env.stamp_markers()
    assert torch.equal(env.obs_buf["images"]["overhead_rgb"], before)
    point = (0.25, 0.05, 0.0)
    env.marker_points[0] = torch.tensor(point)
    env.stamp_markers()
    u, v = camera_model.project(*camera_model.sim_overhead(), point)
    assert env.obs_buf["images"]["overhead_rgb"][0, int(v), int(u)].tolist() == [255, 255, 255]


def test_arm_torques_read_a_push_at_the_pick_point():
    env, arm, _ = make_env()
    dynamics = dm.dynamics(dm.urdf_bodies())
    arm.tau = tuple(dynamics.gravity(HOME))
    env.step(_action())
    residual, _, _, jac = env.arm_torques(0)
    assert float(pick_force(jac, residual).norm()) < 1e-6
    push = np.array([0.0, 0.0, -10.0])
    arm.tau = tuple(np.array(dynamics.gravity(HOME)) - jac.numpy().T @ push)
    env.step(_action())
    residual, _, _, jac = env.arm_torques(0)
    assert pick_force(jac, residual).numpy() == pytest.approx(push, abs=1e-6)


def test_the_wrist_camera_sits_on_its_mount():
    env, _, _ = make_env()
    header = env.wrist_camera_header()
    tool = env.scene["robot"].data.body_pos_w[0, 0].numpy()
    distance = np.linalg.norm(np.array(header["pose"]["pos"]) - tool)
    assert distance == pytest.approx(math.hypot(0.030, 0.169639 - 0.118), abs=1e-3)
    assert header["size"] == [224, 224]


def _images(env):
    return {k: v.clone() for k, v in env.obs_buf["images"].items()}


def _same_images(env, before):
    return all(torch.equal(env.obs_buf["images"][k], v) for k, v in before.items())


def test_a_stale_overhead_frame_holds_the_arm():
    overhead = FakeOverhead()
    env, arm, _ = make_env(overhead=overhead)
    before = _images(env)
    overhead.age_s = 0.5
    overhead.view = np.full((224, 224, 3), 200, dtype=np.uint8)
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_frame" and arm.holds == 1
    assert _same_images(env, before)
    pushes = len(arm.pushes)
    with pytest.raises(EvalFailure):
        env.step(_action())
    assert len(arm.pushes) == pushes and arm.holds == 1


def test_a_stale_frame_trip_names_the_sources_last_drop():
    """MS5 final review M3: the detail carries why the frames stopped, not "(None)"."""
    overhead = FakeOverhead()
    env, _, _ = make_env(overhead=overhead)
    overhead.age_s, overhead.error = 0.5, "jpeg: UnidentifiedImageError: cannot identify"
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_frame" and "UnidentifiedImageError" in err.value.detail


def test_a_stale_wrist_frame_holds_the_arm_and_the_policy_never_sees_it():
    wrist = FakeWrist()
    env, arm, _ = make_env(wrist=wrist)
    before = _images(env)
    wrist.age_s = 0.5
    wrist.view = np.full((224, 224, 3), 200, dtype=np.uint8)
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_frame" and arm.holds == 1
    assert _same_images(env, before)


def test_a_frame_stale_at_the_policy_call_holds_before_the_policy_reads_it():
    overhead = FakeOverhead()
    env, arm, _ = make_env(overhead=overhead)
    env.step(_action())
    before = _images(env)
    # A perception tick between the step and the policy call lets the frame age past the cap.
    overhead.age_s = 0.5
    overhead.view = np.full((224, 224, 3), 200, dtype=np.uint8)
    with pytest.raises(EvalFailure) as err:
        env.stamp_markers()
    assert err.value.reason == "watchdog_frame" and arm.holds == 1
    assert _same_images(env, before)


class _RecoveringOverhead(FakeOverhead):
    """Stale (0.5 s old) until ``clock`` reaches ``fresh_at``, then fresh, with a new view."""

    def __init__(self, clock):
        super().__init__()
        self.clock, self.fresh_at = clock, -math.inf

    @property
    def age_s(self):
        return 0.5 if self.clock() < self.fresh_at else 0.0

    @age_s.setter
    def age_s(self, value):
        pass


def test_a_stale_frame_pauses_the_run_and_resumes_on_a_fresh_one():
    clock = FakeClock()
    overhead = _RecoveringOverhead(clock)
    env, arm, _ = make_env(clock=clock, overhead=overhead)
    env.step(_action())
    pushes = len(arm.pushes)
    overhead.fresh_at = clock() + 0.4
    overhead.view = np.full((224, 224, 3), 200, dtype=np.uint8)
    env.stamp_markers()
    assert len(arm.pushes) == pushes and arm.holds == 0 and not env.holding
    assert len(env.stale_pauses) == 1 and 0.4 <= env.stale_pauses[0] <= 0.4 + 2 * 0.01
    assert int(env.obs_buf["images"]["overhead_rgb"][0, 0, 0, 0]) == 200
    env.step(_action())  # the pause is no tick's work: no watchdog_tick
    assert len(arm.pushes) == pushes + 1


def test_a_pause_longer_than_the_wait_trips():
    from maker_arm_lab.real.env import STALE_WAIT_S

    clock = FakeClock()
    overhead = _RecoveringOverhead(clock)
    env, arm, _ = make_env(clock=clock, overhead=overhead)
    env.step(_action())
    overhead.fresh_at = clock() + STALE_WAIT_S + 0.5
    with pytest.raises(EvalFailure) as err:
        env.stamp_markers()
    assert err.value.reason == "watchdog_frame" and "pause" in err.value.detail
    assert arm.holds == 1


def test_an_arm_fault_during_a_pause_trips_the_arm_watchdog():
    clock = FakeClock()
    overhead = _RecoveringOverhead(clock)
    env, arm, _ = make_env(clock=clock, overhead=overhead)
    env.step(_action())
    overhead.fresh_at = clock() + 1.0
    arm.state = "fault"
    with pytest.raises(EvalFailure) as err:
        env.stamp_markers()
    assert err.value.reason == "watchdog_arm" and arm.holds == 1


def test_the_policy_call_reads_the_newest_fresh_frame():
    overhead = FakeOverhead()
    env, _, _ = make_env(overhead=overhead)
    env.step(_action())
    overhead.view = np.full((224, 224, 3), 200, dtype=np.uint8)
    env.stamp_markers()
    assert int(env.obs_buf["images"]["overhead_rgb"][0, 0, 0, 0]) == 200


def test_a_tick_over_its_cap_holds_but_a_perception_tick_has_a_second():
    env, arm, clock = make_env()
    env.ticks.note_perception()
    clock.advance(0.5)
    env.step(_action())
    clock.advance(0.2)
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_tick" and arm.holds == 1


def test_a_perception_tick_over_a_second_holds():
    env, arm, clock = make_env()
    env.ticks.note_perception()
    clock.advance(1.2)
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_tick" and arm.holds == 1


def test_an_arm_fault_or_a_stopped_loop_holds_and_never_pushes_again():
    env, arm, _ = make_env()
    arm.state = "fault"
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_arm" and arm.pushes == [] and arm.holds == 1
    env, arm, _ = make_env()
    arm.alive = False
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_arm" and arm.pushes == [] and arm.holds == 1


class _ClocklessArm(FakeArm):
    """Alive and enabled, so ``_check_arm`` passes, but the SDK loop has no clock."""

    def now(self):
        return None


def test_a_loop_clock_that_stops_holds_and_pushes_nothing():
    """MS5 final review M4: the ``now() is None`` branch of ``step``."""
    clock = FakeClock()
    env, arm, _ = make_env(clock=clock, arm=_ClocklessArm(clock))
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_arm" and "stopped ticking" in err.value.detail
    assert arm.pushes == [] and arm.holds == 1


class _FailingPushArm(FakeArm):
    def push(self, due, dt, row):
        raise RuntimeError("the tracking session was held")


def test_a_push_that_raises_holds_the_arm():
    clock = FakeClock()
    env, arm, _ = make_env(clock=clock, arm=_FailingPushArm(clock))
    with pytest.raises(EvalFailure) as err:
        env.step(_action())
    assert err.value.reason == "watchdog_arm" and arm.holds == 1
    assert "RuntimeError" in str(err.value)


def test_a_pick_point_under_the_desk_holds_the_arm():
    with pytest.raises(EvalFailure) as err:
        make_env(rig=RigConfig(desk_z=1.0))
    assert err.value.reason == "watchdog_floor"


def test_class_to_box_must_be_a_permutation_of_the_four_boxes():
    with pytest.raises(ValueError):
        make_env(class_to_box=(0, 0, 1, 2))

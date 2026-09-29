"""MS7a desk boxes on the MuJoCo env: the scene, the per-part destination, the box context
row, the recorded scene and the teacher's release into its own box."""

import json
import math
import random

import h5py
import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, layout, loop, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.core.parts import BASE_CLASSES
from maker_arm_lab.core.stowed import per_brick_stowed
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_heap_loop import SPREAD
from test_mj_heap_oracle import PARTS, _place

FIXED = [(0.08, 0.20, 0.3), (0.10, -0.20, -0.5), (0.20, 0.21, 0.0), (0.20, -0.21, 0.6)]
"""Four box poses in the band, clear of the corridor and of one another."""


def _env(n=1, num_envs=1, **kwargs):
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    return MjLegoSortEnv(num_envs, draw, parts=PARTS, cameras=False, desk_boxes=n, **kwargs)


def _hold(env):
    q = env.scene["robot"].data.joint_pos[:, :6].clone().float()
    return torch.cat([q, torch.ones(env.num_envs, 1)], dim=1)


def _box_floor_xy(env, i, j):
    m, d = env.models[i], env.datas[i]
    return d.geom_xpos[m.geom(f"desk_box_{j}_floor").id][:2]


@pytest.mark.parametrize("n", [1, 4])
def test_desk_boxes_replace_the_shelf_and_stand_at_the_drawn_poses(n):
    env = _env(n, num_envs=2, episode_draws=True)
    env.reset(seed=3)
    m = env.models[0]
    names = {m.geom(g).name for g in range(m.ngeom)}
    assert "shelf" not in names and "heap_bin_floor" not in names
    assert env.destination_bins.shape == (2, len(PARTS))
    for i in range(2):
        expected = desk_boxes.part_destinations(PARTS, env.class_to_box[i])
        assert env.destination_bins[i].tolist() == expected
        assert sorted(env.class_to_box[i]) == ([0, 1, 2, 3] if n == 4 else [0, 0, 0, 0])
    assert env.desk_box_poses.shape == (2, n, 3)
    for i in range(2):
        assert desk_boxes.clear(env.desk_box_poses[i], [desk_boxes.interior(n)] * n)
        for j in range(n):
            assert desk_boxes.in_band(env.desk_box_poses[i, j], n)
            assert np.allclose(_box_floor_xy(env, i, j), env.desk_box_poses[i, j, :2])
        # The heap drops on the tray footprint beside the boxes.
        (x0, x1), (y0, y1) = layout.tray_bounds()
        for b in range(len(PARTS)):
            x, y, _ = env.scene[f"brick_{b}"].data.root_pos_w[i].tolist()
            assert x0 - 0.03 <= x <= x1 + 0.03 and y0 - 0.03 <= y <= y1 + 0.03
    # Box 0 is served until the first target names its own.
    assert np.allclose(env.context_box[0], env.desk_box_poses[0, 0])
    for _ in range(3):
        env.step(_hold(env))
    assert not per_brick_stowed(env).any()
    env.close()


def test_a_seed_reproduces_the_boxes_and_the_heap():
    a, b = _env(4), _env(4)
    a.reset(seed=9)
    b.reset(seed=9)
    assert np.array_equal(a.desk_box_poses, b.desk_box_poses)
    assert np.array_equal(a.class_to_box, b.class_to_box)
    for k in range(len(PARTS)):
        assert torch.equal(
            a.scene[f"brick_{k}"].data.root_pos_w, b.scene[f"brick_{k}"].data.root_pos_w
        )
    a.close()
    b.close()


def test_the_ms6_scene_keeps_its_shelf_bin_and_one_destination_per_env():
    env = _env(0)
    env.reset(seed=3)
    m = env.models[0]
    assert m.geom("shelf").id >= 0 and m.geom("heap_bin_floor").id >= 0 and m.nmocap == 0
    assert env.destination_bins.shape == (1,) and env.desk_box_poses.shape == (1, 0, 3)
    env.close()
    with pytest.raises(ValueError, match="desk_boxes"):
        draw = sample_run_draw(random.Random(5))
        MjLegoSortEnv(1, draw, num_bricks=6, cameras=False, desk_boxes=1)


def test_a_part_in_its_own_box_is_stowed_and_one_in_another_box_is_not():
    env = _env(4)
    env.reset(seed=3)
    env.place_desk_boxes(0, FIXED)
    env.destination_bins[0] = torch.tensor([0, 1, 2])
    floor = desk_boxes.DESK_BOX_WALL + 0.006
    # Part 0 lies in its own box 0, part 1 in box 0 too, not its own.
    _place(
        env,
        [
            (*FIXED[0][:2], floor, 0.0),
            (FIXED[0][0] + 0.02, FIXED[0][1], 0.03, 1.0),
            (0.19, 0.0, 0.006, 0.0),
        ],
    )
    for _ in range(30):
        env.step(_hold(env))
    assert per_brick_stowed(env)[0].tolist() == [True, False, False]
    env.close()


def test_the_served_box_follows_the_rigs_marker_part():
    env = _env(4, context="camera_box_target")
    env.reset(seed=3)
    env.place_desk_boxes(0, FIXED)
    for marker in (1, 2, 0):
        env.oracle_cycle_brick = torch.tensor([marker], dtype=torch.int8)
        env.oracle_marker_part = torch.tensor([marker], dtype=torch.int8)
        env.oracle_cycle_target = torch.zeros(1, 7)
        env.step(_hold(env))
        box = FIXED[int(env.destination_bins[0, marker])]
        assert np.allclose(env.context_box[0], box, atol=1e-6)
        assert np.allclose(env.context_camera_box[0, 6:], box, atol=1e-6)
        assert np.allclose(env.obs_buf["context"][0, 6:9].numpy(), box, atol=1e-6)
    env.close()


def test_the_context_box_serves_the_perceived_pose_once_one_is_set():
    env = _env(4)
    env.reset(seed=3)
    assert env.perceived_box_poses is None
    env.set_context_box(0, 2)
    assert np.allclose(env.context_box[0], env.desk_box_poses[0, 2])
    env.perceived_box_poses = env.desk_box_poses + np.array([0.003, -0.002, 0.03])
    env.set_context_box(0, 2)
    assert np.allclose(env.context_box[0], env.perceived_box_poses[0, 2])
    assert np.allclose(env.context_camera_box_target[0, 6:9], env.perceived_box_poses[0, 2])
    assert not np.allclose(env.context_box[0], env.desk_box_poses[0, 2])
    env.close()


def test_the_recorded_initial_state_carries_the_boxes_and_each_parts_destination(tmp_path):
    env = _env(4, export_dir=tmp_path, export_all=True, episode_length_s=0.2)
    env.reset(seed=3)
    poses = env.desk_box_poses[0].copy()
    class_map = env.class_to_box[0].tolist()
    destination = env.destination_bins[0].tolist()
    for _ in range(8):
        env.step(_hold(env))
    env.close()
    with h5py.File(tmp_path / "shard.hdf5") as f:
        init = f["data/demo_0/initial_state"]
        assert np.allclose(init["desk_box_poses"][0], poses, atol=1e-6)
        assert init["destination_bin"][()].tolist() == destination
        assert init["class_to_box"][()].tolist() == class_map
        assert json.loads(init["class_to_box"].attrs["names"]) == list(BASE_CLASSES)
        assert f["data/demo_0/obs/context_box"].shape[1] == 3


def test_the_eight_classes_size_the_class_map_and_the_recorded_names(tmp_path):
    """MS7e design section 5: under an MS7e switch the env folds the eight classes onto its
    boxes and the shard names them; a round part goes to its own class's box."""
    heap = [*PARTS, parts.make_part("round", 2, 2)]
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=heap,
        cameras=False,
        desk_boxes=4,
        class_names=parts.PART_CLASSES,
        export_dir=tmp_path,
        export_all=True,
        episode_length_s=0.2,
    )
    env.reset(seed=3)
    class_map = env.class_to_box[0].tolist()
    assert len(class_map) == 8 and sorted(class_map) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert env.destination_bins[0].tolist()[-1] == class_map[4]  # round, the fifth class
    for _ in range(8):
        env.step(_hold(env))
    env.close()
    with h5py.File(tmp_path / "shard.hdf5") as f:
        init = f["data/demo_0/initial_state"]
        assert json.loads(init["class_to_box"].attrs["names"]) == list(parts.PART_CLASSES)
        assert init["class_to_box"][()].tolist() == class_map


def _teacher_env_and_rig(n, poses, **env_kwargs):
    env = _env(n, **env_kwargs)
    env.reset(seed=11)
    env.place_desk_boxes(0, poses)
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            cycle_budget=4,
            boxes=env.desk_box_poses[i],
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    return env, rig


def test_the_teacher_carries_each_part_to_its_own_box_without_a_raise():
    env, rig = _teacher_env_and_rig(4, FIXED, gripper_mode="continuous")
    _place(env, SPREAD)
    phases, success = set(), False
    for _ in range(9000):
        out = rig.step()
        if rig.goals[0] is not None:
            phases.add(rig.goals[0].phase)
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            # The last part came to rest in its box: every part is stowed, and the env keeps
            # the terminating step's success until its next step (it has already reset).
            success = bool(env.termination_manager.get_term("success")[0])
            break
    stats = rig.episode_stats(0)
    env.close()
    assert oracle.Phase.RAISE not in phases and oracle.Phase.EXTEND in phases
    assert success and stats["unplanned_legs"] == 0, stats


class _RecordingHold:
    """Holds still and records the served box part and the loop's target on every step."""

    def __init__(self, env):
        self.env = env
        self.context_dim = 16
        self.policy_id = "fake:hold"
        self.n_action_steps = 1
        self.inference = {}
        self.action_space = "absolute"
        self.gripper = env.gripper_mode
        self.seen = []

    def step(self, state, wrist, overhead, context=None, anchor=None):
        self.seen.append((int(self.env.target_part[0]), context[0, 6:9].copy()))
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = state[:, :6]
        action[:, 6] = 1.0
        return action


def test_the_loop_serves_the_box_of_every_target_it_selects(monkeypatch):
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 30)
    env, rig = _teacher_env_and_rig(
        4, FIXED, context="camera_box_target", episode_length_s=loop.episode_length_s(4)
    )
    _place(env, SPREAD)
    destination = env.destination_bins[0].tolist()
    policy = _RecordingHold(env)
    payload = loop.rollout(
        env,
        rig,
        policy,
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=4,
        max_wall_minutes=5.0,
    )
    env.close()
    served = [(t, box) for t, box in policy.seen if t >= 0]
    for target, box in served:
        assert np.allclose(box, FIXED[destination[target]], atol=1e-6)
    # The row switches with the target's class: parts 0 and 1 are wide, part 2 narrow.
    boxes = {destination[t] for t, _ in served}
    assert len(boxes) >= 2, payload["episodes"][0]["cycle_targets"]


def test_recorded_teacher_frames_carry_the_box_of_their_target(tmp_path):
    env, rig = _teacher_env_and_rig(
        4, FIXED, gripper_mode="continuous", export_dir=tmp_path, export_all=True
    )
    _place(env, SPREAD)
    destination = env.destination_bins[0].tolist()
    for _ in range(9000):
        out = rig.step()
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    env.close()
    with h5py.File(tmp_path / "shard.hdf5") as f:
        obs = f["data/demo_0/obs"]
        target, box = obs["target_part"][()], obs["context_box"][()]
        phase, cycle = obs["cycle_phase"][()], obs["cycle_brick"][()]
    marked = target >= 0
    expected = np.array([FIXED[destination[t]] for t in target[marked]])
    assert np.allclose(box[marked], expected, atol=1e-6)
    # The row names the next part's box from the frame RELEASE ends: RETREAT frames of a cycle
    # whose next part has another class already carry that class's box.
    switched = (phase == oracle.PHASE_TO_INT["RETREAT"]) & marked & (target != cycle)
    assert switched.any()
    assert len({destination[t] for t in target[marked]}) >= 2


def test_the_oracle_driver_stows_through_the_loop_into_one_box():
    env, rig = _teacher_env_and_rig(1, [FIXED[1]], episode_length_s=loop.episode_length_s(4))
    _place(env, SPREAD)
    payload = loop.rollout(
        env,
        rig,
        None,
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=4,
        max_wall_minutes=5.0,
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["bricks_stowed"] == 3 and ep["cycle_reasons"] == ["stowed"] * 3, ep
    assert math.isclose(payload["cycles_per_stowed"], 1.0)


def test_box_census_names_the_wrong_box_cycle_and_counts_per_class_and_box():
    heap = [parts.make_part("slope", 1, 2), parts.make_part("brick", 1, 6)]
    heap += [parts.make_part("brick", 2, 3), parts.make_part("brick", 2, 7)]
    classes = ["lifted_lost", "stowed", "lifted_lost", "close_miss", "lifted_lost"]
    census = [{"last_held": b} for b in (1, 2, 3, 0, 3)]
    rows = loop.box_census(
        heap,
        destination=[0, 1, 2, 2],
        stowed=[False, False, True, False],
        resting_box=[-1, 3, 2, 1],  # part 1 and part 3 rest in other boxes
        classes=classes,
        census=census,
        num_boxes=4,
    )
    # Part 1's only lifted_lost cycle and part 3's last one become wrong_box.
    assert classes == ["wrong_box", "stowed", "lifted_lost", "close_miss", "wrong_box"]
    assert rows["wrong_box_parts"] == 2
    assert rows["stowed_by_class"] == {
        "slope": {"stowed": 0, "total": 1},
        "tall": {"stowed": 0, "total": 0},
        "narrow": {"stowed": 0, "total": 1},
        "wide": {"stowed": 1, "total": 2},
    }
    assert rows["stowed_by_box"] == [
        {"stowed": 0, "total": 1},
        {"stowed": 0, "total": 1},
        {"stowed": 1, "total": 2},
        {"stowed": 0, "total": 0},
    ]
    assert rows["stowed_by_family"] == {
        "brick": {"stowed": 1, "total": 3},
        "slope": {"stowed": 0, "total": 1},
    }
    assert rows["box_count"] == 4


def test_a_part_carried_into_another_box_is_a_wrong_box_cycle_and_never_stowed():
    env, rig = _teacher_env_and_rig(4, FIXED, episode_length_s=loop.episode_length_s(3))
    _place(env, SPREAD)
    # The grader's truth sends every part to box 1; the teacher's own map sends them to box 0.
    env.destination_bins[0] = torch.tensor([1, 1, 1])
    env.class_to_box[0] = (1, 1, 1, 1)
    rng = random.Random(1)

    def wrong_teacher(i, bl, episode_draw):
        return HeapOracle(
            bl, env.parts, 0, episode_draw, rng=rng, cycle_budget=3, boxes=env.desk_box_poses[i]
        )

    rig.oracle_factory = wrong_teacher
    payload = loop.rollout(
        env,
        rig,
        None,
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=3,
        max_wall_minutes=5.0,
    )
    env.close()
    ep = payload["episodes"][0]
    assert ep["bricks_stowed"] == 0 and payload["stowed_total"] == 0, ep["cycle_reasons"]
    assert ep["cycle_classes"] == ["wrong_box"] * 3 and ep["wrong_box_parts"] == 3
    assert ep["stowed_by_box"][1] == {"stowed": 0, "total": 3}
    assert ep["stowed_by_class"]["wide"] == {"stowed": 0, "total": 2}
    assert payload["wrong_box_parts"] == 3 and payload["wrong_box_cycles"] == 3
    # MS7e design section 8.1: the rows per family and per box count.
    families = {p.family for p in env.parts}
    assert set(payload["stowed_by_family"]) == families and ep["box_count"] == 4
    assert sum(r["total"] for r in payload["stowed_by_family"].values()) == 3
    assert payload["stowed_by_box_count"] == {"4": {"stowed": 0, "total": 3}}


def _lift_until(rig, env, part, rise):
    """Drive the teacher until ``part`` is held ``rise`` above where it lay."""
    start = float(env.scene[f"brick_{part}"].data.root_pos_w[0, 2])
    for _ in range(3000):
        rig.step()
        z = float(env.scene[f"brick_{part}"].data.root_pos_w[0, 2])
        lifting = rig.goals[0] is not None and rig.goals[0].phase == oracle.Phase.LIFT
        if lifting and z - start >= rise:
            return start
    raise AssertionError("the teacher never lifted the part")


def test_the_rig_carries_on_a_part_the_policy_lifted_when_given_the_start_heights():
    env, rig = _teacher_env_and_rig(4, FIXED, gripper_mode="continuous")
    _place(env, [SPREAD[1], (0.6, 0.3, 0.006, 0.0), (0.6, -0.3, 0.006, 0.0)])
    start = _lift_until(rig, env, 0, loop.SCRIPTED_PLACE_RISE)
    rig._go_idle(0, None)
    assert not rig.resume(0)  # RULING P32: a carry above LIFT_HEIGHT - LIFT_MIN_RISE
    start_z = torch.tensor([start, 0.0, 0.0])
    assert rig.resume(0, start_z=start_z) and rig.goals[0].phase == oracle.Phase.LIFT
    for _ in range(1500):
        rig.step()
        if rig.idle[0]:
            break
    stowed = per_brick_stowed(env)[0].tolist()
    env.close()
    assert rig.takeover_result[0] is True and stowed[0]


class _TeacherPolicy:
    """A policy that is the teacher: a second rig on the same env, whose action is served."""

    def __init__(self, env, factory):
        self.context_dim = 0
        self.policy_id = "fake:teacher"
        self.n_action_steps = 1
        self.inference = {}
        self.action_space = "absolute"
        self.gripper = env.gripper_mode
        self.rig = OracleRig(
            env, env.cfg.bin_layout, oracle.OracleNoise(), random.Random(2), oracle_factory=factory
        )
        self.stepped = False

    def step(self, state, wrist, overhead, context=None, anchor=None):
        if self.stepped:
            self.rig.observe(torch.zeros(state.shape[0], dtype=torch.bool))
        self.stepped = True
        return self.rig.actions().numpy()


@pytest.mark.parametrize("scripted", [False, True])
def test_the_scripted_place_hands_a_lifted_part_to_the_teacher(scripted):
    env, rig = _teacher_env_and_rig(
        4, FIXED, gripper_mode="continuous", episode_length_s=loop.episode_length_s(4)
    )
    _place(env, [SPREAD[1], (0.6, 0.3, 0.006, 0.0), (0.6, -0.3, 0.006, 0.0)])
    payload = loop.rollout(
        env,
        rig,
        _TeacherPolicy(env, rig.oracle_factory),
        run_seed=11,
        layout_seed=5,
        permutation=(0, 1, 2, 3),
        cycle_budget=4,
        max_wall_minutes=5.0,
        scripted_place=scripted,
    )
    env.close()
    ep = payload["episodes"][0]
    assert payload["inference"]["loop"]["scripted_place"] is scripted
    assert ep["takeovers"] == int(scripted)
    # One part in reach, two out of it: one cycle, and the teacher's carry stows it.
    assert ep["cycle_targets"] == [0] and ep["cycle_reasons"] == ["stowed"], ep
    assert ep["bricks_stowed"] == 1

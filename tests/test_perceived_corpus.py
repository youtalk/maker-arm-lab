"""The MS7c corpus recorder (MS7c design section 7.1): the teacher acts on the truth, and the
rig records the target row, the frozen marker and the box row from ``core.perception``."""

import importlib.util
import pathlib
import random

import h5py
import numpy as np
import pytest
import torch

from maker_arm_lab.core import grasp, oracle, perception
from maker_arm_lab.core.heap_oracle import HeapOracle, target_context
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_heap_loop import SPREAD
from test_mj_desk_boxes import FIXED, _env
from test_mj_heap_oracle import _place, _step

RUN_SEED = 7
BATCH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_generate_batch.py"
# Every first grasp is offset, so each cycle fails its LIFT and REOPENs at least once.
INDUCE = oracle.FailureInduction(
    prob=1.0, offset_weight=1.0, offset_range=(0.015, 0.025), noise_boost=5.0
)


def _teacher(scales="", perceived=True, induction=None, **env_kwargs):
    """The desk-box teacher on SPREAD, with a Perceiver at ``scales`` unless ``perceived`` is
    False (the truth recorder)."""
    env = _env(4, gripper_mode="continuous", **env_kwargs)
    env.reset(seed=11)
    env.place_desk_boxes(0, FIXED)
    rng = random.Random(1)

    def factory(i, bl, draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            draw,
            induction=induction,
            rng=rng,
            cycle_budget=4,
            boxes=env.desk_box_poses[i],
        )

    perceivers = None
    if perceived:
        perceivers = [perception.Perceiver(RUN_SEED, 0, perception.parse_scales(scales))]
    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        rng,
        induction=induction,
        oracle_factory=factory,
        perceivers=perceivers,
    )
    _place(env, SPREAD)
    return env, rig


def _record(env, rig, stop=None):
    """Step until the episode ends (or ``stop(rig)``); returns the per-step phase and the
    env's marker point of every frame but the one that ended the episode (its reset clears
    the point)."""
    phases, points = [], []
    for _ in range(9000):
        goal = rig.goals[0]
        phases.append(goal.phase.name if goal is not None else rig.oracles[0].phase.name)
        _, out = _step(rig)
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
        points.append(env.marker_points[0].clone())
        if stop is not None and stop(rig):
            break
    return phases, points


def _frames(tmp_path):
    with h5py.File(tmp_path / "shard.hdf5") as f:
        obs = f["data/demo_0/obs"]
        return obs["context_target"][()], obs["context_box"][()], obs["target_part"][()]


def _same(block, row) -> None:
    """Every row of ``block`` equals ``row``, as the float32 the env and the recorder hold."""
    np.testing.assert_array_equal(block, np.broadcast_to(np.float32(row), block.shape))


def _segments(rows, target):
    """(start, end) of each run of marked frames that carry one row."""
    out = []
    for k in range(len(rows)):
        if target[k] < 0:
            continue
        if out and out[-1][1] == k and np.array_equal(rows[k], rows[k - 1]):
            out[-1][1] = k + 1
        else:
            out.append([k, k + 1])
    return out


def _perceived_grasp(state, part, k, fallback):
    """The grasp the recorder ranks on ``part`` alone in the perceived scene."""
    scene = state.scene(k, (0.0, 0.0, 0.0))
    seen = state.parts(k)
    mask = torch.zeros(k, dtype=torch.bool)
    mask[part] = True
    ranked = grasp.rank_grasps(
        torch.tensor([p.size for p in seen]),
        scene.brick_pos,
        scene.brick_quat,
        mask,
        oracle.PICK_TILT_RAD,
        [p.closing_axes for p in seen],
    )
    return (grasp.select_grasp(ranked) or ranked[0]) if ranked else fallback


def test_the_recorded_row_marker_and_box_replay_from_perceive(tmp_path, monkeypatch):
    truths = []
    real = perception.Perceiver.perceive

    def spy(self, truth):
        truths.append(truth)
        return real(self, truth)

    monkeypatch.setattr(perception.Perceiver, "perceive", spy)
    env, rig = _teacher(induction=INDUCE, export_dir=tmp_path, export_all=True)
    destination = env.destination_bins[0].tolist()
    perceived_boxes = env.perceived_box_poses[0].copy()
    phases, points = _record(env, rig)
    stats = rig.episode_stats(0)
    env.close()
    rows, boxes, target = _frames(tmp_path)
    segments = _segments(rows, target)
    points = torch.stack(points).numpy()

    # One perceive per selection, and one frozen row per selection: the pre-selection's
    # adoption at HOVER and the induced REOPENs take no second perceive.
    assert stats["retries"] >= 1 and stats["perceived_fallbacks"] == 0, stats
    assert len(segments) == len(truths) == stats["perceived_selections"] == 3
    replay = perception.Perceiver(RUN_SEED, 0)
    replay.reset_episode(FIXED)
    for (a, b), truth in zip(segments, truths):
        part = int(target[a])
        assert (target[a:b] == part).all()  # no adoption in this run
        state = replay.perceive(truth)
        found = next(d for d in state.detections if d.sim_index == part)
        chosen = _perceived_grasp(state, part, len(env.parts), None)
        _same(rows[a:b], target_context(found.centre, chosen))
        _same(points[a:b], found.centre)
    # Frozen through the whole cycle, a REOPEN included.
    a, b = segments[0]
    assert {"HOVER", "DESCEND", "CLOSE", "LIFT", "REOPEN", "EXTEND"} <= set(phases[a:b])
    assert np.isnan(points[target[: len(points)] < 0]).all()

    # The box row: the perceived pose (true pose + the episode's bias) of the part's true box,
    # and box 0's perceived pose before the first selection.
    assert np.allclose(replay.box_poses, perceived_boxes)
    marked = target >= 0
    expected = np.float32([perceived_boxes[destination[t]] for t in target[marked]])
    np.testing.assert_array_equal(boxes[marked], expected)
    _same(boxes[: segments[0][0]], perceived_boxes[0])
    true_boxes = np.float32([FIXED[destination[t]] for t in target[marked]])
    assert not np.isclose(boxes[marked], true_boxes, atol=1e-4).all(axis=1).any()


def test_an_adoption_keeps_the_row_and_the_marker_and_switches_the_box():
    env, rig = _teacher()
    orc = rig.oracles[0]
    _record(env, rig, stop=lambda r: r.goals[0].phase == oracle.Phase.CLOSE)
    assert orc.target_brick == 0 and env.destination_bins[0].tolist() == [0, 0, 1]
    row, point = env.context_target[0].copy(), env.marker_points[0].clone()
    assert np.allclose(env.context_box[0], env.perceived_box_poses[0, 0])
    orc.adopt_held_part(2)  # the LIFT found part 2, of another class, in the jaws
    _step(rig)
    assert int(env.target_part[0]) == 2
    np.testing.assert_array_equal(env.context_target[0], row)
    assert torch.equal(env.marker_points[0], point)
    np.testing.assert_array_equal(env.context_box[0], np.float32(env.perceived_box_poses[0, 1]))
    assert rig.stats[0]["perceived_selections"] == 1
    env.close()


def test_a_missed_target_falls_back_to_the_biased_truth_and_the_teachers_grasp():
    env, rig = _teacher(scales="miss=20")  # every part is missed at every perceive
    _record(env, rig, stop=lambda r: int(r.env.target_part[0]) >= 0)
    orc = rig.oracles[0]
    centre = [t + e for t, e in zip(orc._brick_start.tolist(), rig.perceivers[0].bias)]
    assert any(abs(e) > 1e-4 for e in rig.perceivers[0].bias)
    _same(env.context_target[0], target_context(centre, orc.grasp))
    _same(env.marker_points[0].numpy(), centre)
    assert rig.stats[0]["perceived_selections"] == rig.stats[0]["perceived_fallbacks"] == 1
    env.close()


def test_at_zero_scales_the_row_is_the_teachers_and_without_perceivers_nothing_changes(
    tmp_path, monkeypatch
):
    truth_dir, zero_dir = tmp_path / "truth", tmp_path / "zero"
    env, rig = _teacher(perceived=False, export_dir=truth_dir, export_all=True)
    _, points = _record(env, rig)
    assert not hasattr(env, "oracle_marker_point") and env.perceived_box_poses is None
    assert all(not torch.isfinite(p).any() for p in points)
    assert "perceived_selections" not in rig.episode_stats(0)
    env.close()

    hidden = []
    real = perception.Perceiver.perceive

    def spy(self, truth):
        state = real(self, truth)
        hidden.append(len(state.detections) < len(truth.classes))
        return state

    monkeypatch.setattr(perception.Perceiver, "perceive", spy)
    env, rig = _teacher(scales="all=0", export_dir=zero_dir, export_all=True)
    _, points = _record(env, rig)
    env.close()
    points = torch.stack(points).numpy()
    truth_rows, truth_boxes, truth_target = _frames(truth_dir)
    rows, boxes, target = _frames(zero_dir)
    assert np.array_equal(target, truth_target)
    marked = target >= 0
    # The true centre at selection, the teacher's width and half height; the box is the true
    # one. Only the heading may differ, when a structural rule hid a neighbor (the part just
    # released, under the tool) and a tie between two equal widths ranked the other way.
    np.testing.assert_allclose(rows[marked][:, :3], truth_rows[marked][:, :3], atol=1e-5)
    np.testing.assert_array_equal(rows[marked][:, 5:], truth_rows[marked][:, 5:])
    np.testing.assert_array_equal(boxes[marked], truth_boxes[marked])
    for (a, b), hid in zip(_segments(rows, target), hidden):
        if not hid:
            np.testing.assert_allclose(rows[a:b], truth_rows[a:b], atol=1e-5)
        # The frozen marker sits on the row's centre; the truth recorder's follows the part.
        np.testing.assert_array_equal(points[a:b], rows[a:b, :3][: len(points[a:b])])
    assert hidden[0] is False


def _load_batch():
    spec = importlib.util.spec_from_file_location("mj_generate_batch", BATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEAP = ["--out_dir", "/tmp/x", "--heap_k", "3"]


@pytest.mark.parametrize(
    "argv",
    [
        HEAP + ["--perceived"],  # no desk boxes
        ["--out_dir", "/tmp/x", "--desk_boxes", "4", "--perceived"],  # no heap
        HEAP + ["--desk_boxes", "4", "--perceived", "--noise_scales", "pos=-1"],
        HEAP + ["--desk_boxes", "4", "--perceived", "--noise_scales", "focus=2"],
    ],
)
def test_the_batch_script_refuses_a_bad_perceived_run_with_its_fail_token(argv, capsys):
    batch = _load_batch()
    with pytest.raises(SystemExit) as exit_info:
        batch.main(argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("BATCH_FAIL usage: "), out


def test_the_run_meta_records_the_perceived_switch_and_the_canonical_scales():
    batch = _load_batch()
    args = batch.parse_args(
        HEAP + ["--desk_boxes", "4", "--perceived", "--noise_scales", "all=0,heading=2"]
    )
    draw, _, noise = batch.build_run(args)
    meta = batch.run_meta(
        args, draw, noise, None, num_bricks=3, export_mode="EXPORT_ALL", backend="mujoco", task="t"
    )
    assert meta["perceived"] is True
    assert meta["noise_scales"] == "pos=0,heading=2,size=0,box=0,miss=0,class=0"
    off = batch.parse_args(HEAP)
    assert off.noise_scales == dict.fromkeys(perception.KNOBS, 1.0)
    meta = batch.run_meta(
        off, draw, noise, None, num_bricks=3, export_mode="EXPORT_ALL", backend="mujoco", task="t"
    )
    assert meta["perceived"] is False and meta["noise_scales"] is None

"""MS5 design section 10: the unchanged heap loop runs on RealEnv under the MS7d served
switches, and no decision on the rig reads a truth accessor."""

import math
import random

import numpy as np
import pytest
import torch

from helpers import load_script
from maker_arm_lab.core import loop, oracle, perception
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.core.parts import BASE_CLASSES, make_part
from maker_arm_lab.real.perceive import RealPerception, WorkerClient
from maker_arm_lab.real.rig import episode_sheet
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from real_fakes import FakeWorker, make_env
from test_perceived_loop import (
    TRUTH_DECISION_READS,
    _garbage_reads,
    _raises,
    _Still,
    _TruthlessEnv,
)

real_eval_heap = load_script("real_eval_heap")

BUDGET = 3
SERVED = real_eval_heap.SERVED_SWITCHES
"""The switches the rig serves: the eval client's one definition (``SERVED_SWITCHES``)."""
SIZE = make_part("brick", 2, 4).size
SPREAD = [(0.195, -0.045, 0.3), (0.195, 0.0, -0.8), (0.195, 0.045, 1.2)]
SHEET = episode_sheet(20265000, 0)


def _detections():
    return [
        {
            "centre": [x, y, SIZE[2] / 2],
            "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
            "size": list(SIZE),
            "cls": "wide",
            "score": 0.9,
        }
        for x, y, yaw in SPREAD
    ]


@pytest.fixture()
def workers():
    made = []

    def make(**kwargs):
        made.append(FakeWorker(**kwargs))
        return made[-1]

    yield make
    for worker in made:
        worker.close()


def _real_rollout(monkeypatch, workers, poison=False, detections=None):
    torch.manual_seed(0)
    monkeypatch.setattr(loop, "CYCLE_STEP_BUDGET", 60)
    env, arm, _ = make_env(
        class_to_box=SHEET["class_to_box"], episode_length_s=loop.episode_length_s(BUDGET)
    )
    worker = workers(
        boxes=SHEET["box_poses"], detections=_detections() if detections is None else detections
    )
    client = WorkerClient(worker.endpoint)
    env.perception_backend = RealPerception(env, client, env.cfg.num_bricks)
    rng = random.Random(1)

    def factory(i, bl, draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            draw,
            rng=rng,
            cycle_budget=BUDGET,
            boxes=env.desk_box_poses[i],
            tray=True,
            tray_lean_deg=env.tray_lean_deg,
            **real_eval_heap.SERVED_ORACLE,
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    served = []
    stepped = env.step

    def step(action):
        served.append(
            (
                env.context_target[0].copy(),
                env.context_box[0].copy(),
                env.marker_points[0].clone(),
                action[0].clone(),
            )
        )
        return stepped(action)

    env.step = step
    target = env
    if poison:
        gen = torch.Generator().manual_seed(7)
        monkeypatch.setattr(
            loop, "truth_reads", lambda e, per_part: _garbage_reads(e, per_part, gen)
        )
        for name in (*TRUTH_DECISION_READS, "perception_truth", "wrist_class", "wrist_look_truth"):
            monkeypatch.setattr(loop, name, _raises(name))
        rig.snapshot = _raises("rig.snapshot")
        target = _TruthlessEnv(env)
    payload = loop.rollout(
        target,
        rig,
        _Still(),
        run_seed=1,
        layout_seed=0,
        permutation=(0, 1, 2, 3),
        cycle_budget=BUDGET,
        max_wall_minutes=5.0,
        perceived=True,
        noise_scales=perception.parse_scales("all=0"),
        **SERVED,
    )
    client.close()
    return payload, served, env, arm, worker


def test_the_heap_loop_runs_on_the_rig_to_its_cycle_budget(monkeypatch, workers):
    payload, served, env, arm, worker = _real_rollout(monkeypatch, workers)
    ep = payload["episodes"][0]
    assert ep["cycles"] == BUDGET and ep["done_reason"] == "cycle_budget", ep
    seen = payload["perception"]
    assert seen["selections"] == BUDGET and seen["marker_looks"] >= BUDGET, seen
    kinds = worker.kinds()
    assert kinds.count("boxes") == 1 and kinds.count("parts") >= BUDGET and "marker" in kinds
    assert len(arm.pushes) == len(served) == len(env.ticks.ticks) > 0
    assert sum(t.perception for t in env.ticks.ticks) >= BUDGET


def test_each_cycle_serves_its_detection_its_marker_and_its_class_box(monkeypatch, workers):
    _, served, _, _, _ = _real_rollout(monkeypatch, workers)
    wide_box = SHEET["box_poses"][SHEET["class_to_box"][BASE_CLASSES.index("wide")]]
    centres = {tuple(round(v, 4) for v in d["centre"]) for d in _detections()}
    marked = [(row, box, marker) for row, box, marker, _ in served if not marker.isnan().any()]
    assert marked
    for row, box, marker in marked:
        assert tuple(round(v, 4) for v in marker.tolist()) in centres
        assert np.allclose(row[:3], marker.numpy(), atol=1e-6)
        assert np.allclose(box, wide_box, atol=1e-6)


def test_no_decision_on_the_rig_reads_a_truth_accessor(monkeypatch, workers):
    first, served, _, arm, _ = _real_rollout(monkeypatch, workers)
    second, blind, _, blind_arm, _ = _real_rollout(monkeypatch, workers, poison=True)
    assert len(served) == len(blind) > 0
    for s, (a, b) in enumerate(zip(served, blind)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), s
        assert torch.equal(a[2].nan_to_num(), b[2].nan_to_num()), s
        assert torch.equal(a[3], b[3]), s
    assert [p[2] for p in arm.pushes] == [p[2] for p in blind_arm.pushes]
    for key in ("cycle_reasons", "cycle_targets", "cycle_steps", "done_reason"):
        assert first["episodes"][0][key] == second["episodes"][0][key], key


def test_an_empty_scene_ends_the_episode_no_candidate(monkeypatch, workers):
    """Controller ruling 1 / Review Focus 2: with no detections the rig's tool starts at HOME,
    clear of the tray (``loop.tool_clear_of_tray``), so the first selection with nothing to pick
    ends the episode ``no_candidate`` at once, the same rule as the simulator's."""
    payload, _, _, _, worker = _real_rollout(monkeypatch, workers, detections=[])
    ep = payload["episodes"][0]
    assert ep["cycles"] == 0 and ep["done_reason"] == "no_candidate", ep
    assert "parts" in worker.kinds()

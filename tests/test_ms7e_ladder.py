"""MS7e unseen-box ladder, the seen row (design section 8.3): a gate read on a recorded wave's
own seeds draws, with held-out parts, the boxes that wave recorded in its env 0."""

import json
import random
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import ROOT, load_script
from maker_arm_lab.core.batch_loop import heap_classes, heap_parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj.env import MjLegoSortEnv

sys.path.insert(0, str(ROOT / "scripts"))
import fleet
import mj_eval_heap

ge = load_script("run_gate_eval")

RUN0, LAYOUT0 = 20284000, 20285000  # MS7e ledger R1, wave 1


def _ledger(tmp_path: Path, runs: int = 40) -> Path:
    path = tmp_path / "ledger.json"
    records = [
        {"name": f"ms7e_w1_proc_{p}", "seeds": {"run_seed": RUN0 + p, "layout_seed": LAYOUT0 + p}}
        for p in range(runs)
    ]
    path.write_text(json.dumps({"runs": records}))
    return path


def _cfg(tmp_path, *extra, processes=20):
    args = SimpleNamespace(
        ckpt="/ck/pretrained_model",
        threshold=0.85,
        seed=7,
        draws=list(range(processes)),
        split="heldout",
        n_steps=50,
        client_flags="",
    )
    argv = fleet.supervisor_args(args, str(_ledger(tmp_path))) + ["--out_dir", str(tmp_path)]
    return ge._build_cfg(ge._parse_args(argv + list(extra)))


def test_the_seen_row_reads_the_recorded_runs_seeds(tmp_path):
    cfg = _cfg(tmp_path, "--seen_seeds", f"{RUN0},{LAYOUT0}", "--num_envs", "1")
    draws = ge.gate_draws(cfg)
    assert [(d.run_seed, d.layout_seed) for d in draws] == [
        (RUN0 + p, LAYOUT0 + p) for p in range(20)
    ]
    assert {d.permutation for d in draws} == {(0, 1, 2, 3)}  # run_corpus_e.sh's default


def test_with_the_switch_off_the_draws_are_the_seeds_and_a_recorded_pair_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.seen_seeds is None
    assert ge.gate_draws(cfg) == ge.make_eval_draws(20, 7)
    seen = _cfg(tmp_path, "--seen_seeds", f"{RUN0 + 30},{LAYOUT0 + 30}", "--num_envs", "1")
    with pytest.raises(ValueError, match="no recorded run"):  # p 40 .. 49 were never recorded
        ge.gate_draws(seen)
    fresh = _cfg(tmp_path, "--seed", str(RUN0 // 1000))  # run seeds 20284000 + i: wave 1's
    with pytest.raises(ValueError, match="collides"):
        ge.gate_draws(fresh)


def test_the_seen_row_needs_one_env_a_draw(tmp_path):
    with pytest.raises(ValueError, match="--num_envs 1"):
        _cfg(tmp_path, "--seen_seeds", f"{RUN0},{LAYOUT0}")


WAVE = shlex.split(
    "--tray --desk_boxes 8 --tray_lean 30 --box_draw 8 --release_lean --carry_via "
    "--passenger_check --families round,ldraw,flat --deep_box_floor --episode_draws "
    "--frame_jitter --context camera_box_target --gripper continuous --heap_k 6"
)


def _boxes(split: str, num_envs: int, p: int):
    """Each env's (count, poses, sizes) after the first reset of the eval client's env."""
    seeds = ["--run_seed", str(RUN0 + p), "--layout_seed", str(LAYOUT0 + p)]
    args = mj_eval_heap.parse_args(
        WAVE + seeds + ["--parts_split", split, "--num_envs", str(num_envs)]
    )
    draw = sample_run_draw(random.Random(args.layout_seed))
    draw["slot_colors"] = (0, 1, 2, 3)
    parts = heap_parts(args)
    env = MjLegoSortEnv(
        num_envs,
        draw,
        parts=parts,
        cameras=False,
        episode_draws=True,
        frame_jitter=True,
        gripper_mode="continuous",
        desk_boxes=8,
        tray=True,
        tray_lean_deg=30.0,
        class_names=heap_classes(args),
        box_draw=8,
        deep_box_floor=True,
        release_lean=True,
        round_physics=True,
    )
    env.reset(seed=args.run_seed)
    out = []
    for i in range(num_envs):
        n = int(env.box_count[i])
        out.append((n, env.desk_box_poses[i, :n].copy(), env.desk_box_sizes[i, :n].copy()))
    env.close()
    return out


@pytest.mark.parametrize("p", [26, 38])
def test_env_0_draws_the_recorded_boxes_whatever_its_parts(p):
    """Wave 1 recorded three train heaps a process; the read's one env with held-out parts
    draws env 0's boxes, and a second env's boxes follow the heap before it."""
    (n, poses, sizes), *rest = _boxes("train", 3, p)
    ((n1, poses1, sizes1),) = _boxes("heldout", 1, p)
    assert n1 == n and np.array_equal(poses1, poses) and np.array_equal(sizes1, sizes)
    later = _boxes("heldout", 3, p)[1:]
    assert any(not (a[0] == b[0] and np.array_equal(a[2], b[2])) for a, b in zip(rest, later))

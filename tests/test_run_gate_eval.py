"""CPU tests for scripts/run_gate_eval.py: draws, seed disjointness, aggregation, the
reuse of finished draws, and the token contract. The server and the eval processes are
injected fakes; the real ones need a GPU."""

import json
import sys
import threading
import types
from pathlib import Path

import pytest

from helpers import load_script

ge = load_script("run_gate_eval")
gd = load_script("generate_dataset")

CAMPAIGN_SEED = 20260907
SELECTION_SEED = 20260917
OFFICIAL_SEED = 20260927


def _fake_ledger(tmp_path: Path, seed: int = CAMPAIGN_SEED, runs: int = 30) -> Path:
    """A ledger whose seeds are derived exactly as generate_dataset.py derives them."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=1000, seed=seed)
    records = []
    for i in range(runs):
        run_seed, layout_seed = gd._derive_seeds(cfg, i)
        records.append(
            {
                "run_index": i,
                "seeds": {"run_seed": run_seed, "layout_seed": layout_seed},
                "exported": 42,
                "attempts": 50,
            }
        )
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"total_successes": 42 * runs, "runs": records}))
    return path


def _episode(stowed: int, success: bool = False, **extra) -> dict:
    ep = {
        "env_index": 0,
        "success": success,
        "bricks_stowed": stowed,
        "num_bricks": 6,
        "steps": 5400,
        "clamp_steps": 0,
        "grasp_attempts": 0,
        "grasp_successes": 0,
        "grasp_successes_strict": 0,
        "grasp_attempts_unresolved": 0,
        "wrong_bin_deposits": 0,
        "first_close_step": -1,
        "clamp_steps_per_joint": [0] * 6,
    }
    ep.update(extra)
    return ep


def _payload(draw: ge.EvalDraw, episodes: list[dict], checkpoint="ck", **extra) -> dict:
    return {
        "run_seed": draw.run_seed,
        "layout_seed": draw.layout_seed,
        "permutation": list(draw.permutation),
        "num_envs": len(episodes),
        "num_bricks": 6,
        "policy_id": f"act:{checkpoint}+te0.01",
        "n_action_steps": 1,
        "inference": {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "chunk_size": 100},
        "episodes": episodes,
        "releases_queued": 0,
        "wall_sec": 100.0,
        **extra,
    }


# --- draws and seeds ------------------------------------------------------------------------


def test_aggregate_reads_the_close_event_instrument():
    """The confusion names which color went to which bin; the close reads pair each close's
    nearest-brick distance with its hold window by index, so an open window pairs with nothing."""
    draw = ge.EvalDraw(process_index=0, permutation=(0, 1, 2, 3), run_seed=1, layout_seed=2)
    episodes = [
        _episode(
            1,
            releases=[["red", "yellow"], ["red", "red"], ["blue", None]],
            closes_nearest_cm=[0.5, 3.0, 1.0],
            closes_held_window=[True, False],
        ),
        _episode(
            0, releases=[["red", "yellow"]], closes_nearest_cm=[-1.0], closes_held_window=[False]
        ),
    ]
    sec = ge.aggregate_gate([_payload(draw, episodes)], 0.7)["secondary"]
    assert sec["wrong_bin_confusion"] == {"red->yellow": 2}
    assert (sec["closes_graded_window"], sec["closes_within_2cm"]) == (3, 1)
    assert sec["hold_rate_window"] == pytest.approx(1 / 3)
    assert sec["hold_rate_window_within_2cm"] == pytest.approx(1.0)
    legacy = ge.aggregate_gate([_payload(draw, [_episode(0)])], 0.7)["secondary"]
    assert legacy["wrong_bin_confusion"] == {} and legacy["hold_rate_window"] is None


def test_draws_are_deterministic_and_use_the_campaign_seed_formula():
    a = ge.make_eval_draws(10, SELECTION_SEED)
    b = ge.make_eval_draws(10, SELECTION_SEED)
    assert a == b
    assert a[3].run_seed == (SELECTION_SEED * 1000 + 3) % 2**32
    assert a[3].layout_seed == (SELECTION_SEED * 1000 + 500 + 3) % 2**32
    assert all(sorted(d.permutation) == [0, 1, 2, 3] for d in a)
    assert len({d.permutation for d in ge.make_eval_draws(40, SELECTION_SEED)}) > 5


def test_the_three_ms2_seed_sets_are_pairwise_disjoint(tmp_path):
    """Corpus (ledger, up to 1000 runs), selection (60 = 5 x 12), official (120 = 10 x 12).

    The seeds are ten apart, not one: the formula spreads a campaign of R runs over
    seed*1000 + [0, R) and + [500, 500 + R), so a seed one apart collides once R > 500.
    """
    ledger = _fake_ledger(tmp_path, CAMPAIGN_SEED, runs=1000)
    ledger_runs = json.loads(ledger.read_text())["runs"]
    selection = ge.make_eval_draws(5, SELECTION_SEED)
    official = ge.make_eval_draws(10, OFFICIAL_SEED)
    ge.assert_seeds_disjoint(selection, ledger_runs)
    ge.assert_seeds_disjoint(official, ledger_runs)
    ge.assert_seed_sets_disjoint(("selection", selection), ("official", official))


def test_a_seed_collision_with_the_ledger_is_refused(tmp_path):
    ledger = _fake_ledger(tmp_path, CAMPAIGN_SEED, runs=5)
    ledger_runs = json.loads(ledger.read_text())["runs"]
    colliding = ge.make_eval_draws(5, CAMPAIGN_SEED)  # the campaign's own seed
    with pytest.raises(ValueError, match="collides"):
        ge.assert_seeds_disjoint(colliding, ledger_runs)


def test_overlapping_selection_and_official_sets_are_refused():
    a = ge.make_eval_draws(5, SELECTION_SEED)
    with pytest.raises(ValueError, match="appears in both"):
        ge.assert_seed_sets_disjoint(("selection", a), ("official", a))


def test_select_draws_keeps_the_full_lists_indices():
    draws = ge.make_eval_draws(10, SELECTION_SEED)
    assert [d.process_index for d in ge.select_draws(draws, (2, 7))] == [2, 7]
    assert ge.select_draws(draws, None) == draws
    with pytest.raises(ValueError, match="outside"):
        ge.select_draws(draws, (11,))


# --- aggregation --------------------------------------------------------------------------


def test_wilson_interval_matches_the_ms1_report():
    lo, hi = ge.wilson_interval(42, 50)
    assert lo == pytest.approx(0.715, abs=0.001)
    assert hi == pytest.approx(0.917, abs=0.001)
    assert ge.wilson_interval(0, 0) == (0.0, 0.0)
    assert ge.wilson_interval(0, 10)[0] == 0.0


def test_aggregate_gate_grades_per_brick_against_the_threshold():
    draws = ge.make_eval_draws(2, SELECTION_SEED)
    p0 = _payload(draws[0], [_episode(1), _episode(0), _episode(6, success=True)])
    p1 = _payload(draws[1], [_episode(2), _episode(0), _episode(0)])
    results = ge.aggregate_gate([p0, p1], threshold=0.15)
    g = results["gate"]
    assert g["episodes"] == 6 and g["bricks_total"] == 36 and g["bricks_stowed"] == 9
    assert g["per_brick_rate"] == pytest.approx(0.25)
    assert g["passed"] is True
    assert g["wilson95"][0] < 0.25 < g["wilson95"][1]
    assert results["secondary"]["all_six_episode_rate"] == pytest.approx(1 / 6)
    assert results["secondary"]["bricks_stowed_histogram"] == {
        "0": 3,
        "1": 1,
        "2": 1,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 1,
    }
    assert ge.aggregate_gate([p0, p1], threshold=0.30)["gate"]["passed"] is False


def test_aggregate_gate_reports_the_secondaries_without_gating_on_them():
    draws = ge.make_eval_draws(1, SELECTION_SEED)
    eps = [
        _episode(0, grasp_attempts=4, grasp_successes=1, grasp_successes_strict=0, clamp_steps=10),
        _episode(0, wrong_bin_deposits=1, first_close_step=12, clamp_steps=5),
    ]
    results = ge.aggregate_gate([_payload(draws[0], eps, releases_queued=3)], threshold=0.15)
    s = results["secondary"]
    assert s["grasp_success_rate"] == pytest.approx(0.25)
    assert s["grasp_success_rate_strict"] == 0.0
    assert s["wrong_bin_rate"] == pytest.approx(1 / 3)
    assert s["never_closed_episodes"] == 1
    assert s["clamp_steps_total"] == 15
    assert s["clamp_fraction"] == pytest.approx(15 / 10800)
    assert results["gate"]["passed"] is False


def test_aggregate_gate_on_no_episodes_is_a_fail_not_a_crash():
    results = ge.aggregate_gate([], threshold=0.15)
    assert results["gate"]["passed"] is False and results["gate"]["episodes"] == 0


# --- orchestration --------------------------------------------------------------------------


class _FakeServer:
    def __init__(self, log):
        self.log = log
        self.ready = {"inference": {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01}}
        log.append("start")

    def stop(self):
        self.log.append("stop")


def _cfg(tmp_path, ledger, **kw):
    base = {
        "checkpoint": "ck",
        "out_dir": tmp_path / "gate",
        "processes": 3,
        "num_envs": 2,
        "seed": SELECTION_SEED,
        "threshold": 0.15,
        "ledger": ledger,
        "retries": 1,
        # The MS3 protocol, which is what ``_payload`` says the server served with.
        "n_action_steps": 1,
        "temporal_ensemble_coeff": 0.01,
    }
    base.update(kw)
    return ge.GateCfg(**base)


def test_run_gate_runs_every_draw_once_and_writes_results(tmp_path, capsys):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger)
    log = []
    ran = []

    def run_eval(draw):
        ran.append(draw.process_index)
        return _payload(draw, [_episode(2), _episode(0)])

    results = ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert ran == [0, 1, 2]
    assert log == ["start", "stop"]
    assert results["gate"]["episodes"] == 6
    assert results["gate"]["per_brick_rate"] == pytest.approx(6 / 36)
    assert results["gate"]["passed"] is True
    assert results["inference"]["temporal_ensemble_coeff"] == 0.01
    assert results["inference"]["action_warp"] == 1.0  # the fake server predates the warp
    assert results["policy_id"] == "act:ck+te0.01"
    assert len(results["draws"]) == 3
    written = json.loads((cfg.out_dir / "gate_results.json").read_text())
    assert written["gate"] == results["gate"]
    assert all((cfg.out_dir / f"eval_{i:03d}.json").exists() for i in range(3))
    assert "GATE_OK" in capsys.readouterr().out


def test_run_gate_reuses_finished_draws_and_skips_the_server_when_nothing_is_left(tmp_path, capsys):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger)
    log = []
    calls = []

    def run_eval(draw):
        calls.append(draw.process_index)
        if draw.process_index == 1 and len(calls) == 2:
            raise KeyboardInterrupt  # the host sweeps the run mid-ladder
        return _payload(draw, [_episode(0), _episode(0)])

    with pytest.raises(KeyboardInterrupt):
        ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert log == ["start", "stop"]  # the server is stopped on the way out
    assert (cfg.out_dir / "eval_000.json").exists()
    assert not (cfg.out_dir / "eval_001.json").exists()

    results = ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert calls == [0, 1, 1, 2]  # draw 0 was reused, 1 and 2 ran
    assert results["gate"]["episodes"] == 6 and results["gate"]["passed"] is False
    assert "GATE_FAIL" in capsys.readouterr().out

    ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert calls == [0, 1, 1, 2]  # everything reused: no eval ran ...
    assert log == ["start", "stop", "start", "stop"]  # ... and no server was started


def test_run_gate_does_not_reuse_another_checkpoints_or_draws_payload(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1)
    cfg.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    stale = _payload(draw, [_episode(6, success=True)], checkpoint="other")
    (cfg.out_dir / "eval_000.json").write_text(json.dumps(stale))
    ran = []

    def run_eval(d):
        ran.append(d.process_index)
        return _payload(d, [_episode(0), _episode(0)])

    results = ge.run_gate(cfg, run_eval, lambda: _FakeServer([]))
    assert ran == [0]
    assert results["gate"]["bricks_stowed"] == 0

    other_draw = ge.EvalDraw(0, (3, 2, 1, 0), 1, 2)
    (cfg.out_dir / "eval_000.json").write_text(json.dumps(_payload(other_draw, [_episode(6)])))
    ge.run_gate(cfg, run_eval, lambda: _FakeServer([]))
    assert ran == [0, 0]


def test_run_gate_retries_then_raises_and_still_stops_the_server(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, retries=2)
    log = []
    attempts = []

    def run_eval(draw):
        attempts.append(draw.process_index)
        raise ge.EvalProcessError("boom")

    with pytest.raises(ge.GateRunError, match="3 times"):
        ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert attempts == [0, 0, 0]
    assert log == ["start", "stop"]


def test_run_gate_refuses_a_colliding_seed_before_starting_anything(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, seed=CAMPAIGN_SEED)
    log = []
    with pytest.raises(ValueError, match="collides"):
        ge.run_gate(cfg, lambda d: pytest.fail("must not run"), lambda: _FakeServer(log))
    assert log == []


def test_main_maps_outcomes_to_exit_codes_and_tokens(monkeypatch, capsys, tmp_path):
    ledger = _fake_ledger(tmp_path)
    base = [
        "--checkpoint",
        "ck",
        "--out_dir",
        str(tmp_path / "g"),
        "--ledger",
        str(ledger),
        "--threshold",
        "0.15",
        "--seed",
        str(SELECTION_SEED),
        "--processes",
        "1",
    ]
    monkeypatch.setattr(ge, "_start_server", lambda cfg: _FakeServer([]))

    monkeypatch.setattr(
        ge, "_run_eval_subprocess", lambda cfg, d: _payload(d, [_episode(3), _episode(0)])
    )
    assert ge.main(base) == 0
    assert "GATE_OK" in capsys.readouterr().out

    monkeypatch.setattr(
        ge, "_run_eval_subprocess", lambda cfg, d: _payload(d, [_episode(0), _episode(0)])
    )
    assert ge.main(base + ["--out_dir", str(tmp_path / "g2")]) == 1
    assert "GATE_FAIL" in capsys.readouterr().out

    def broken(cfg, d):
        raise ge.EvalProcessError("no token")

    monkeypatch.setattr(ge, "_run_eval_subprocess", broken)
    assert ge.main(base + ["--out_dir", str(tmp_path / "g3"), "--retries", "0"]) == 2
    out = capsys.readouterr()
    assert "GATE_ERROR" in out.out and "failed 1 times" in out.err


def test_commands_carry_the_inference_flags_and_the_draw(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "ledger.json", n_action_steps=1, temporal_ensemble_coeff=0.01)
    server = ge._server_command(cfg)
    assert server[server.index("--temporal_ensemble_coeff") + 1] == "0.01"
    assert server[server.index("--n_action_steps") + 1] == "1"
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    cmd = ge._eval_command(cfg, draw)
    assert cmd[cmd.index("--run_seed") + 1] == str(draw.run_seed)
    assert cmd[cmd.index("--permutation") + 1] == ",".join(str(p) for p in draw.permutation)
    assert cmd[cmd.index("--num_envs") + 1] == "2"
    assert "--enable_cameras" in cmd and "--headless" in cmd


def test_eval_command_mujoco_backend_spawns_the_mujoco_client(tmp_path):
    cfg = ge.GateCfg(
        checkpoint="ck",
        out_dir=tmp_path,
        processes=1,
        num_envs=4,
        seed=7,
        threshold=0.7,
        ledger=tmp_path / "ledger.json",
        backend="mujoco",
    )
    draw = ge.make_eval_draws(1, 7)[0]
    cmd = ge._eval_command(cfg, draw)
    assert cmd[:2] == [ge.MUJOCO_PYTHON, "scripts/mj_eval_policy.py"]
    assert "--headless" not in cmd and cmd[cmd.index("--num_envs") + 1] == "4"
    assert ge._isaaclab_eval_env(cfg)["MUJOCO_GL"]
    isaac = ge._eval_command(
        ge.GateCfg(
            checkpoint="ck",
            out_dir=tmp_path,
            processes=1,
            num_envs=4,
            seed=7,
            threshold=0.7,
            ledger=tmp_path / "ledger.json",
        ),
        draw,
    )
    assert isaac[1:3] == ["-p", "scripts/eval_policy.py"]
    assert cmd[cmd.index("--num_envs") :] == isaac[isaac.index("--num_envs") :]


def test_eval_command_forwards_the_draw_flags_to_the_mujoco_client(tmp_path):
    cfg = ge.GateCfg(
        checkpoint="ck",
        out_dir=tmp_path,
        processes=1,
        num_envs=4,
        seed=7,
        threshold=0.7,
        ledger=tmp_path / "ledger.json",
        backend="mujoco",
        episode_draws=True,
        frame_jitter=True,
        spatial_scale=3.0,
    )
    draw = ge.make_eval_draws(1, 7)[0]
    cmd = ge._eval_command(cfg, draw)
    assert "--episode_draws" in cmd and "--frame_jitter" in cmd
    assert cmd[cmd.index("--spatial_scale") + 1] == "3.0"
    plain = ge._eval_command(
        ge.GateCfg(
            checkpoint="ck",
            out_dir=tmp_path,
            processes=1,
            num_envs=4,
            seed=7,
            threshold=0.7,
            ledger=tmp_path / "ledger.json",
            backend="mujoco",
        ),
        draw,
    )
    assert "--episode_draws" not in plain and "--spatial_scale" not in plain
    args = ge._parse_args(
        [
            "--checkpoint",
            "ck",
            "--out_dir",
            str(tmp_path),
            "--ledger",
            "l",
            "--threshold",
            "0.7",
            "--seed",
            "1",
            "--backend",
            "mujoco",
            "--episode_draws",
            "--spatial_scale",
            "2",
        ]
    )
    assert ge._build_cfg(args).episode_draws is True and ge._build_cfg(args).spatial_scale == 2.0


def test_eval_command_forwards_the_context_to_the_mujoco_client(tmp_path):
    cfg = ge.GateCfg(
        checkpoint="ck",
        out_dir=tmp_path,
        processes=1,
        num_envs=4,
        seed=7,
        threshold=0.7,
        ledger=tmp_path / "ledger.json",
        backend="mujoco",
        context="privileged",
    )
    cmd = ge._eval_command(cfg, ge.make_eval_draws(1, 7)[0])
    assert cmd[cmd.index("--context") + 1] == "privileged"
    args = ge._parse_args(
        [
            "--checkpoint",
            "ck",
            "--out_dir",
            str(tmp_path),
            "--ledger",
            "l",
            "--threshold",
            "0.7",
            "--seed",
            "1",
            "--backend",
            "mujoco",
        ]
    )
    plain = ge._build_cfg(args)
    assert plain.context == "none"
    assert "--context" not in ge._eval_command(plain, ge.make_eval_draws(1, 7)[0])


def test_eval_command_heap_spawns_the_heap_client_with_the_heap_flags(tmp_path):
    cfg = ge.GateCfg(
        checkpoint="ck",
        out_dir=tmp_path,
        processes=1,
        num_envs=5,
        seed=20261400,
        threshold=0.85,
        ledger=tmp_path / "ledger.json",
        backend="mujoco",
        episode_draws=True,
        frame_jitter=True,
        context="camera_box",
        heap_k=50,
        parts_split="heldout",
        cycle_budget=65,
    )
    draw = ge.make_eval_draws(1, 20261400)[0]
    cmd = ge._eval_command(cfg, draw)
    assert cmd[:2] == [ge.MUJOCO_PYTHON, "scripts/mj_eval_heap.py"]
    for flag, value in (
        ("--heap_k", "50"),
        ("--parts_seed", "20260915"),
        ("--parts_split", "heldout"),
        ("--cycle_budget", "65"),
        ("--context", "camera_box"),
    ):
        assert cmd[cmd.index(flag) + 1] == value
    assert "--episode_draws" in cmd and "--frame_jitter" in cmd


def test_aggregate_gate_applies_the_cycles_rule_on_a_heap():
    draw = ge.make_eval_draws(1, 20261400)[0]
    eps = [dict(_episode(0), num_bricks=50, bricks_stowed=45, cycles=55, success=False)]
    results = ge.aggregate_gate([_payload(draw, eps)], threshold=0.85, max_cycles_per_stowed=1.3)
    gate = results["gate"]
    assert gate["per_brick_rate"] == pytest.approx(0.9)
    assert gate["cycles_total"] == 55 and gate["cycles_per_stowed"] == pytest.approx(55 / 45)
    assert gate["passed"] is True
    slow = [dict(eps[0], cycles=65)]
    assert (
        ge.aggregate_gate([_payload(draw, slow)], threshold=0.85, max_cycles_per_stowed=1.3)[
            "gate"
        ]["passed"]
        is False
    )
    # No cycles rule: the six-brick gate reads as before, and cycles are reported as None.
    six = ge.aggregate_gate([_payload(draw, [_episode(0)])], threshold=0.15)["gate"]
    assert six["cycles_per_stowed"] is None and six["max_cycles_per_stowed"] is None


def test_aggregate_gate_refuses_the_cycles_rule_on_an_episode_without_cycles():
    draw = ge.make_eval_draws(1, 20261400)[0]
    heap = dict(_episode(0), num_bricks=50, bricks_stowed=45, cycles=55)
    with pytest.raises(ValueError, match="cycles"):
        ge.aggregate_gate(
            [_payload(draw, [heap, _episode(6)])], threshold=0.85, max_cycles_per_stowed=1.3
        )


def _heap_payload(draw, **extra):
    heap = {
        "heap_k": 50,
        "parts_split": "heldout",
        "parts_seed": 20260915,
        "cycle_budget": 65,
        "gripper": "binary",
    }
    heap.update(extra)
    ep = dict(_episode(0), num_bricks=50, bricks_stowed=45, cycles=55)
    return _payload(draw, [ep], **heap)


def test_cached_payload_matches_the_task_as_well_as_the_draw(tmp_path):
    ledger = _fake_ledger(tmp_path)
    heap_cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50)
    six_cfg = _cfg(tmp_path, ledger, processes=1)
    heap_cfg.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    path = heap_cfg.out_dir / "eval_000.json"

    def cached(cfg, payload):
        path.write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    six = _payload(draw, [_episode(6, success=True)])
    assert cached(six_cfg, six) and not cached(heap_cfg, six)
    assert cached(heap_cfg, _heap_payload(draw)) and not cached(six_cfg, _heap_payload(draw))
    for other in (
        {"heap_k": 20},
        {"parts_split": "train"},
        {"parts_seed": 1},
        {"cycle_budget": 9},
        {"gripper": "continuous"},
        {"tray": True},
    ):
        assert not cached(heap_cfg, _heap_payload(draw, **other)), other
    heap_cfg.cycle_budget = 70
    assert cached(heap_cfg, _heap_payload(draw, cycle_budget=70))
    heap_cfg.gripper = "continuous"
    assert cached(heap_cfg, _heap_payload(draw, cycle_budget=70, gripper="continuous"))
    assert not cached(heap_cfg, _heap_payload(draw, cycle_budget=70, gripper="binary"))


def test_eval_command_forwards_the_gripper_to_the_heap_client(tmp_path):
    cfg = ge.GateCfg(
        checkpoint="ck",
        out_dir=tmp_path,
        processes=1,
        num_envs=5,
        seed=1,
        threshold=0.85,
        ledger=tmp_path / "ledger.json",
        backend="mujoco",
        heap_k=50,
        gripper="continuous",
    )
    cmd = ge._eval_command(cfg, ge.make_eval_draws(1, 1)[0])
    assert cmd[cmd.index("--gripper") + 1] == "continuous"
    assert "--tray" not in cmd


def test_the_tray_reaches_only_the_heap_client_and_its_cache(tmp_path):
    base = ["--checkpoint", "ck", "--out_dir", str(tmp_path), "--ledger", "l"]
    base += ["--threshold", "0.85", "--seed", "1", "--backend", "mujoco", "--tray"]
    cfg = ge._build_cfg(ge._parse_args(base + ["--heap_k", "50"]))
    assert cfg.tray and "--tray" in ge._eval_command(cfg, ge.make_eval_draws(1, 1)[0])
    with pytest.raises(ValueError, match="--tray"):
        ge._build_cfg(ge._parse_args(base))
    cached = _cfg(tmp_path, _fake_ledger(tmp_path), processes=1, backend="mujoco", heap_k=50)
    cached.tray = True
    cached.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    path = cached.out_dir / "eval_000.json"
    path.write_text(json.dumps(_heap_payload(draw, tray=True)))
    assert ge._cached_payload(cached, draw) is not None
    path.write_text(json.dumps(_heap_payload(draw)))
    assert ge._cached_payload(cached, draw) is None


def test_a_heap_gate_defaults_the_cycles_rule_and_refuses_the_isaac_backend(
    monkeypatch, capsys, tmp_path
):
    base = [
        "--checkpoint",
        "ck",
        "--out_dir",
        str(tmp_path),
        "--ledger",
        "l",
        "--threshold",
        "0.85",
        "--seed",
        "1",
        "--heap_k",
        "50",
    ]
    cfg = ge._build_cfg(ge._parse_args(base + ["--backend", "mujoco"]))
    assert cfg.max_cycles_per_stowed == 1.3
    explicit = ge._parse_args(base + ["--backend", "mujoco", "--max_cycles_per_stowed", "2"])
    assert ge._build_cfg(explicit).max_cycles_per_stowed == 2.0
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(base + ["--backend", "isaac"]) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "--backend mujoco" in out.err


@pytest.mark.parametrize(
    "extra",
    [["--backend", "mujoco"], ["--backend", "isaac"], ["--backend", "isaac", "--heap_k", "50"]],
)
def test_a_continuous_gate_refuses_anything_but_the_mujoco_heap(
    monkeypatch, capsys, tmp_path, extra
):
    argv = [
        "--checkpoint",
        "ck",
        "--out_dir",
        str(tmp_path),
        "--ledger",
        "l",
        "--threshold",
        "0.85",
        "--seed",
        "1",
        "--gripper",
        "continuous",
    ]
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(argv + extra) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "--gripper continuous" in out.err
    heap = ["--backend", "mujoco", "--heap_k", "50"]
    assert ge._build_cfg(ge._parse_args(argv + heap)).gripper == "continuous"


def test_cached_payload_matches_the_inference_configuration(tmp_path):
    ledger = _fake_ledger(tmp_path)
    ensemble = _cfg(tmp_path, ledger, processes=1)
    open_loop = _cfg(
        tmp_path, ledger, processes=1, n_action_steps=None, temporal_ensemble_coeff=None
    )
    ensemble.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    path = ensemble.out_dir / "eval_000.json"

    def cached(cfg, payload):
        path.write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    served_ensemble = _payload(draw, [_episode(6, success=True)])
    served_open_loop = _payload(
        draw,
        [_episode(6, success=True)],
        policy_id="act:ck",
        inference={"n_action_steps": 100, "temporal_ensemble_coeff": None, "chunk_size": 100},
    )
    # The MS6 phase B defect: an open-loop read stood in the out_dir of an ensemble read.
    assert cached(ensemble, served_ensemble) and not cached(ensemble, served_open_loop)
    assert cached(open_loop, served_open_loop) and not cached(open_loop, served_ensemble)
    open_loop.n_action_steps = 50
    assert not cached(open_loop, served_open_loop)
    # A payload older than the handshake echo carries no inference key: only a gate that asks
    # for no override reuses it.
    legacy = {k: v for k, v in served_open_loop.items() if k != "inference"}
    open_loop.n_action_steps = None
    assert cached(open_loop, legacy) and not cached(ensemble, legacy)


def test_action_warp_reaches_the_server_splits_the_cache_and_refuses_out_of_range(
    tmp_path, monkeypatch, capsys
):
    ledger = _fake_ledger(tmp_path)
    plain = _cfg(tmp_path, ledger, processes=1, n_action_steps=None, temporal_ensemble_coeff=None)
    assert "--action_warp" not in ge._server_command(plain)  # the unwarped command is unchanged
    warped = _cfg(
        tmp_path,
        ledger,
        processes=1,
        n_action_steps=None,
        temporal_ensemble_coeff=None,
        action_warp=2.0,
    )
    server = ge._server_command(warped)
    assert server[server.index("--action_warp") + 1] == "2.0"
    plain.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "chunk_size": 100}

    def cached(cfg, inference):
        payload = _payload(draw, [_episode(6)], policy_id="act:ck+nas50", inference=inference)
        (plain.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    # A warp read never reuses an unwarped payload, nor the reverse; one that predates the
    # warp ran unwarped.
    assert cached(plain, served) and cached(plain, {**served, "action_warp": 1.0})
    assert not cached(plain, {**served, "action_warp": 2.0})
    assert cached(warped, {**served, "action_warp": 2.0}) and not cached(warped, served)
    assert not cached(warped, {**served, "action_warp": 1.5})
    args = ge._parse_args(_argv(tmp_path) + ["--action_warp", "1.5"])
    assert ge._build_cfg(args).action_warp == 1.5
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    for bad in ("0.5", "3.5"):
        assert ge.main(_argv(tmp_path) + ["--action_warp", bad]) == 2
        out = capsys.readouterr()
        assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "--action_warp" in out.err


def test_ensemble_with_reaches_the_server_and_splits_the_cache(tmp_path):
    ledger = _fake_ledger(tmp_path)
    plain = _cfg(tmp_path, ledger, processes=1, n_action_steps=None, temporal_ensemble_coeff=None)
    assert "--ensemble_with" not in ge._server_command(plain)  # the plain command is unchanged
    ens = _cfg(
        tmp_path,
        ledger,
        processes=1,
        n_action_steps=None,
        temporal_ensemble_coeff=None,
        ensemble_with="/m/b",
    )
    server = ge._server_command(ens)
    assert server[server.index("--ensemble_with") + 1] == "/m/b"
    plain.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "chunk_size": 100}

    def cached(cfg, inference, policy_id="act:ck+nas50"):
        payload = _payload(draw, [_episode(6)], policy_id=policy_id, inference=inference)
        (plain.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    # An ensemble read never reuses a one-checkpoint payload, nor the reverse, nor another
    # member's; its policy id still starts with the primary checkpoint.
    with_b = {**served, "ensemble_with": "/m/b"}
    assert cached(plain, served) and not cached(plain, with_b, "act:ck+nas50+ens:/m/b")
    assert cached(ens, with_b, "act:ck+nas50+ens:/m/b") and not cached(ens, served)
    assert not cached(ens, {**served, "ensemble_with": "/m/c"}, "act:ck+nas50+ens:/m/c")
    args = ge._parse_args(_argv(tmp_path) + ["--ensemble_with", "/m/b"])
    assert ge._build_cfg(args).ensemble_with == "/m/b"
    assert ge._build_cfg(ge._parse_args(_argv(tmp_path))).ensemble_with is None


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--checkpoint",
        "ck",
        "--out_dir",
        str(tmp_path),
        "--ledger",
        "l",
        "--threshold",
        "0.85",
        "--seed",
        "1",
    ]


def test_a_heap_gate_refuses_the_privileged_context(monkeypatch, capsys, tmp_path):
    # The heap client has no privileged channel; the supervisor must say so before it
    # starts a server and spends every draw on the client's own usage error.
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    heap = ["--backend", "mujoco", "--heap_k", "50"]
    assert ge.main(_argv(tmp_path) + heap + ["--context", "privileged"]) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "privileged" in out.err
    assert ge._build_cfg(ge._parse_args(_argv(tmp_path) + heap)).context == "none"


def test_an_argparse_usage_error_still_prints_one_token(capsys, tmp_path):
    # The token contract covers every exit path: argparse's own error() prints none.
    with pytest.raises(SystemExit) as exc:
        ge._parse_args(_argv(tmp_path) + ["--processes", "not_a_number"])
    assert exc.value.code == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines() == ["GATE_ERROR"]
    assert "not_a_number" in out.err


def test_client_flags_reach_the_mujoco_client_and_the_cache_matches_on_the_switches(tmp_path):
    args = ge._parse_args(
        [
            "--checkpoint",
            "ck",
            "--out_dir",
            str(tmp_path / "gate"),
            "--ledger",
            "l",
            "--threshold",
            "0.7",
            "--seed",
            "1",
            "--backend",
            "mujoco",
            "--heap_k",
            "50",
            "--gripper",
            "continuous",
            "--client_flags",
            "--select_at_release --replan_on_switch",
        ]
    )
    cfg = ge._build_cfg(args)
    assert cfg.client_flags == ("--select_at_release", "--replan_on_switch")
    draw = ge.make_eval_draws(1, 1)[0]
    cmd = ge._eval_command(cfg, draw)
    at = cmd.index("--select_at_release")
    assert cmd[at : at + 2] == ["--select_at_release", "--replan_on_switch"]
    cfg.out_dir.mkdir()
    on = {"select_at_release": True, "replan_on_switch": True, "row_refresh": False}
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "chunk_size": 100}

    def write(loop):
        payload = _payload(
            draw,
            [_episode(0)],
            inference=served if loop is None else {**served, "loop": loop},
            heap_k=50,
            parts_split="heldout",
            parts_seed=20260915,
            cycle_budget=65,
            gripper="continuous",
        )
        (cfg.out_dir / "eval_000.json").write_text(json.dumps(payload))

    write(on)
    assert ge._cached_payload(cfg, draw) is not None
    write({**on, "row_refresh": True})
    assert ge._cached_payload(cfg, draw) is None
    write(None)  # a payload from before the switches ran with every switch off
    assert ge._cached_payload(cfg, draw) is None
    cfg.client_flags = ()
    assert ge._cached_payload(cfg, draw) is not None


def test_desk_boxes_reach_the_heap_client_only_and_split_the_cache(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    assert "--desk_boxes" not in ge._eval_command(cfg, draw)  # the MS6 command is unchanged
    cfg.desk_boxes = 4
    cmd = ge._eval_command(cfg, draw)
    assert cmd[1] == "scripts/mj_eval_heap.py" and cmd[cmd.index("--desk_boxes") + 1] == "4"
    cfg.out_dir.mkdir(parents=True)
    path = cfg.out_dir / "eval_000.json"
    for payload_boxes, hit in ((4, True), (1, False), (None, False)):
        extra = {} if payload_boxes is None else {"desk_boxes": payload_boxes}
        path.write_text(json.dumps(_heap_payload(draw, **extra)))
        assert (ge._cached_payload(cfg, draw) is not None) is hit, payload_boxes
    cfg.desk_boxes = 0
    path.write_text(json.dumps(_heap_payload(draw)))  # an MS6 payload predates the key
    assert ge._cached_payload(cfg, draw) is not None


@pytest.mark.parametrize(
    "extra",
    [["--backend", "mujoco"], ["--backend", "isaac", "--heap_k", "50"]],
)
def test_desk_boxes_refuse_anything_but_the_mujoco_heap(monkeypatch, capsys, tmp_path, extra):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path), "--ledger", "l"]
    argv += ["--threshold", "0.85", "--seed", "1", "--desk_boxes", "1"]
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(argv + extra) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "--desk_boxes" in out.err
    heap = ["--backend", "mujoco", "--heap_k", "50"]
    assert ge._build_cfg(ge._parse_args(argv + heap)).desk_boxes == 1


def test_the_gate_reports_desk_box_rows_and_counts_own_box_stows_only():
    draw = ge.make_eval_draws(1, 20261400)[0]
    ep = dict(
        _episode(0),
        num_bricks=4,
        bricks_stowed=2,
        cycles=4,
        cycle_classes=["stowed", "stowed", "wrong_box", "no_close"],
        stowed_by_class={"slope": {"stowed": 2, "total": 3}, "wide": {"stowed": 0, "total": 1}},
        stowed_by_box=[{"stowed": 2, "total": 3}, {"stowed": 0, "total": 1}],
        wrong_box_parts=1,
    )
    results = ge.aggregate_gate([_payload(draw, [ep, ep])], threshold=0.5)
    assert results["gate"]["bricks_stowed"] == 4 and results["gate"]["per_brick_rate"] == 0.5
    rows = results["secondary"]
    assert rows["stowed_by_class"]["slope"] == {"stowed": 4, "total": 6}
    assert rows["stowed_by_box"][1] == {"stowed": 0, "total": 2}
    assert rows["wrong_box_parts"] == 2 and rows["wrong_box_cycles"] == 2
    # An MS7d payload has no family rows, and its box count is the length of its box rows.
    assert rows["stowed_by_family"] == {}
    assert rows["stowed_by_box_count"] == {"2": {"stowed": 4, "total": 8}}
    six = ge.aggregate_gate([_payload(draw, [_episode(0)])], threshold=0.1)["secondary"]
    assert "stowed_by_class" not in six


HEAP = ["--backend", "mujoco", "--heap_k", "50"]
PERCEIVED = HEAP + ["--desk_boxes", "4", "--gripper", "continuous", "--perceived"]
ISAAC = (["--backend", "isaac"], "needs --heap_k > 0 and --backend mujoco")
NO_PERCEIVED = (PERCEIVED[:-1], "needs --perceived")
SWITCH_READS = {
    "scripted_place": (HEAP, [(["--backend", "mujoco"], "needs --heap_k > 0")]),
    "scripted_rake": (HEAP + ["--tray"], [ISAAC]),
    "part_failures": (HEAP, [ISAAC]),
    "pick_snap": (HEAP, [ISAAC]),
    "dwell_close": (HEAP, [ISAAC]),
    "adopt_held": (HEAP, [ISAAC]),
    "park_done": (HEAP + ["--tray", "--desk_boxes", "4"], [ISAAC]),
    "rake_turn": (HEAP + ["--tray", "--desk_boxes", "4"], [ISAAC]),
    "carry_guard": (
        PERCEIVED,
        [NO_PERCEIVED, (PERCEIVED + ["--passenger_drop"], "refuses --passenger_drop")],
    ),
    "clear_first": (PERCEIVED + ["--tray"], [NO_PERCEIVED, ISAAC]),
    "overhang_guard": (
        PERCEIVED + ["--tray"],
        [NO_PERCEIVED, ISAAC, (PERCEIVED, "reads the tray wall's overhang: it needs --tray")],
    ),
}
"""The heap loop's on/off switches that have no test of their own here: the flags of the read each
one needs, and each refusal as the flags of a read and the message after the switch's flag."""


@pytest.mark.parametrize("name", list(SWITCH_READS))
def test_a_loop_switch_reaches_the_heap_client_once_keys_the_cache_and_needs_its_read(
    name, tmp_path, monkeypatch, capsys
):
    read, refusals = SWITCH_READS[name]
    argv = _argv(tmp_path / "gate") + read
    plain = ge._build_cfg(ge._parse_args(argv))
    cfg = ge._build_cfg(ge._parse_args(argv + [f"--{name}"]))
    draw = ge.make_eval_draws(1, 1)[0]
    assert ge._eval_command(cfg, draw).count(f"--{name}") == 1
    assert f"--{name}" not in ge._eval_command(plain, draw)
    off = {n: False for n in ge.LOOP_SWITCHES} | {"rake_cap": 0, "retry_noise_at": 0}
    off |= {"guard_margin": 0.010, "noise_scales": ""}
    if "--perceived" in read:
        off |= {"perceived": True, "noise_scales": ge.noise_scales_key("")}
    assert ge.loop_switches(plain.client_flags) == off
    assert ge.loop_switches(cfg.client_flags) == off | {name: True}
    cfg.out_dir.mkdir()
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    heap = {"tray": cfg.tray, "desk_boxes": cfg.desk_boxes, "gripper": cfg.gripper}

    def hit(gate, loop):
        payload = _heap_payload(draw, inference={**served, "loop": loop}, **heap)
        (cfg.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(gate, draw) is not None

    # A payload that predates the switch, or records it only when on, ran with it off.
    older = {k: v for k, v in off.items() if k != name}
    assert hit(plain, off) and hit(plain, older) and not hit(plain, off | {name: True})
    assert hit(cfg, off | {name: True}) and not hit(cfg, off) and not hit(cfg, older)
    assert not hit(cfg, {})
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    for bad, message in refusals:
        assert ge.main(_argv(tmp_path) + bad + [f"--{name}"]) == 2
        out = capsys.readouterr()
        assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and f"--{name} {message}" in out.err


def test_a_pass_through_the_scripted_place_is_its_own_outcome(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50, threshold=0.5)
    cfg.client_flags = ("--scripted_place",)
    draw = ge.make_eval_draws(1, cfg.seed)[0]

    def server():
        return types.SimpleNamespace(ready={}, stop=lambda: None)

    results = ge.run_gate(cfg, run_eval=lambda d: _heap_payload(d), start_server=server)
    assert results["gate"]["outcome"] == "PASS (scripted place)"
    cfg.client_flags = ()
    (cfg.out_dir / "eval_000.json").unlink()
    results = ge.run_gate(cfg, run_eval=lambda d: _heap_payload(d), start_server=server)
    assert results["gate"]["outcome"] == "PASS" and draw.process_index == 0


def test_the_rake_settings_reach_the_heap_client_only_and_key_the_cache(tmp_path, monkeypatch):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path / "gate"), "--ledger", "l"]
    argv += ["--threshold", "0.7", "--seed", "1"]
    heap = ["--backend", "mujoco", "--heap_k", "50", "--tray", "--desk_boxes", "4"]
    cfg = ge._build_cfg(ge._parse_args(argv + heap + ["--rake_cap", "4", "--rake_crowd"]))
    draw = ge.make_eval_draws(1, 1)[0]
    cmd = ge._eval_command(cfg, draw)
    assert cmd[cmd.index("--rake_cap") + 1] == "4" and "--rake_crowd" in cmd
    found = ge.loop_switches(cfg.client_flags)
    assert found["rake_cap"] == 4 and found["rake_crowd"] is True
    plain = ge._build_cfg(ge._parse_args(argv + heap))
    assert ge.loop_switches(plain.client_flags)["rake_cap"] == 0
    assert "--rake_cap" not in ge._eval_command(plain, draw)
    cfg.out_dir.mkdir()
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    loop = {name: False for name in ge.LOOP_SWITCHES} | {"rake_crowd": True}
    for cap, hit in ((4, True), (2, False), (0, False)):
        payload = _heap_payload(
            draw, tray=True, desk_boxes=4, inference={**served, "loop": loop | {"rake_cap": cap}}
        )
        (cfg.out_dir / "eval_000.json").write_text(json.dumps(payload))
        assert (ge._cached_payload(cfg, draw) is not None) is hit, cap
    for bad in (["--rake_cap", "4"], ["--rake_crowd"]):
        with pytest.raises(ValueError, match="--rake_c"):
            ge._build_cfg(ge._parse_args(argv + ["--backend", "isaac"] + bad))


def test_the_tray_lean_reaches_the_heap_client_and_keys_the_cache(tmp_path):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path / "gate"), "--ledger", "l"]
    argv += ["--threshold", "0.7", "--seed", "1", "--backend", "mujoco", "--heap_k", "50"]
    cfg = ge._build_cfg(ge._parse_args(argv + ["--tray", "--tray_lean", "45"]))
    cmd = ge._eval_command(cfg, ge.make_eval_draws(1, 1)[0])
    assert cmd[cmd.index("--tray_lean") + 1] == "45.0" and "--tray" in cmd
    with pytest.raises(ValueError, match="--tray_lean"):
        ge._build_cfg(ge._parse_args(argv + ["--tray_lean", "45"]))
    cached = _cfg(tmp_path, _fake_ledger(tmp_path), processes=1, backend="mujoco", heap_k=50)
    cached.tray, cached.tray_lean = True, 45.0
    cached.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    path = cached.out_dir / "eval_000.json"
    for lean, hit in ((45.0, True), (30.0, False), (0.0, False)):
        path.write_text(json.dumps(_heap_payload(draw, tray=True, tray_lean=lean)))
        assert (ge._cached_payload(cached, draw) is not None) is hit, lean


def test_the_perceived_flags_reach_the_heap_client_and_key_the_cache(tmp_path):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path / "gate"), "--ledger", "l"]
    argv += ["--threshold", "0.7", "--seed", "1"]
    heap = ["--backend", "mujoco", "--heap_k", "50", "--tray", "--desk_boxes", "4"]
    heap += ["--gripper", "continuous"]
    draw = ge.make_eval_draws(1, 1)[0]
    off = ge._build_cfg(ge._parse_args(argv + heap))
    plain = ge._eval_command(off, draw)
    assert not {"--perceived", "--noise_scales", "--contact_hold"} & set(plain)
    on = ["--perceived", "--noise_scales", "heading=2", "--contact_hold"]
    cfg = ge._build_cfg(ge._parse_args(argv + heap + on))
    cmd = ge._eval_command(cfg, draw)
    canonical = "pos=1,heading=2,size=1,box=1,miss=1,class=1,"
    canonical += "tool=1,wrist=1,wrist_bias=1,wrist_miss=1,rho=1,torque=1,torque_k=1"
    assert "--perceived" in cmd and "--contact_hold" in cmd
    assert cmd[cmd.index("--noise_scales") + 1] == canonical  # forwarded canonical
    found = ge.loop_switches(cfg.client_flags)
    assert found["perceived"] and found["contact_hold"] and found["noise_scales"] == canonical
    nominal = ge._build_cfg(ge._parse_args(argv + heap + ["--perceived"]))
    assert "--noise_scales" not in ge._eval_command(nominal, draw)
    assert ge.loop_switches(nominal.client_flags)["noise_scales"] == ge.noise_scales_key("")
    cfg.out_dir.mkdir()
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    base = {name: False for name in ge.LOOP_SWITCHES}
    read = base | {"perceived": True, "contact_hold": True, "noise_scales": canonical}
    for loop, hit in (
        (read, True),
        (read | {"noise_scales": ge.noise_scales_key("heading=4")}, False),
        (read | {"contact_hold": False}, False),
        (read | {"perceived": False, "noise_scales": ""}, False),
        ({}, False),
    ):
        payload = _heap_payload(
            draw, tray=True, desk_boxes=4, gripper="continuous", inference={**served, "loop": loop}
        )
        (cfg.out_dir / "eval_000.json").write_text(json.dumps(payload))
        assert (ge._cached_payload(cfg, draw) is not None) is hit, loop
        # A gate with the three off reads a payload that predates them as off, and no other.
        assert (ge._cached_payload(off, draw) is not None) is (loop == {}), loop
    for bad, flag in (
        (["--noise_scales", "heading=2"], "--noise_scales"),
        (["--contact_hold"], "--contact_hold"),
        (["--perceived", "--noise_scales", "bogus=1"], "--noise_scales"),
    ):
        with pytest.raises(ValueError, match=flag):
            ge._build_cfg(ge._parse_args(argv + heap + bad))
    with pytest.raises(ValueError, match="--perceived"):
        ge._build_cfg(ge._parse_args(argv + ["--backend", "isaac", "--perceived"]))
    for missing in (heap[:5] + heap[7:], heap[:7]):  # no desk boxes, the binary gripper
        with pytest.raises(ValueError, match="--perceived needs"):
            ge._build_cfg(ge._parse_args(argv + missing + ["--perceived"]))


def test_the_ms7d_switches_and_knobs_reach_the_heap_client_and_key_the_cache(tmp_path):
    argv = ["--checkpoint", "ck", "--out_dir", str(tmp_path / "gate"), "--ledger", "l"]
    argv += ["--threshold", "0.7", "--seed", "1"]
    heap = ["--backend", "mujoco", "--heap_k", "50", "--tray", "--desk_boxes", "4"]
    heap += ["--gripper", "continuous", "--perceived"]
    draw = ge.make_eval_draws(1, 1)[0]
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
    nominal = ge.noise_scales_key("")
    base = {name: False for name in ge.LOOP_SWITCHES} | {"perceived": True, "noise_scales": nominal}
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir()

    def hit(cfg, loop):
        payload = _heap_payload(
            draw, tray=True, desk_boxes=4, gripper="continuous", inference={**served, "loop": loop}
        )
        path.write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    plain = ge._build_cfg(ge._parse_args(argv + heap))
    assert not {f"--{name}" for name in ge.MS7D_LEVERS} & set(ge._eval_command(plain, draw))
    assert hit(plain, base)
    for name in ge.MS7D_LEVERS:
        cfg = ge._build_cfg(ge._parse_args(argv + heap + [f"--{name}"]))
        assert f"--{name}" in ge._eval_command(cfg, draw)
        assert ge.loop_switches(cfg.client_flags) == base | {
            name: True,
            "rake_cap": 0,
            "retry_noise_at": 0,
            "guard_margin": 0.010,
        }
        assert (
            hit(cfg, base | {name: True})
            and not hit(cfg, base)
            and not hit(plain, base | {name: True})
        )
    for knob in ("tool", "wrist", "wrist_bias", "wrist_miss", "rho", "torque", "torque_k"):
        scales = ge.noise_scales_key(f"{knob}=2")
        assert f"{knob}=2" in scales.split(",")
        cfg = ge._build_cfg(ge._parse_args(argv + heap + ["--noise_scales", f"{knob}=2"]))
        cmd = ge._eval_command(cfg, draw)
        assert cmd[cmd.index("--noise_scales") + 1] == scales
        assert hit(cfg, base | {"noise_scales": scales}) and not hit(cfg, base), knob
    # An MS7c payload recorded the six MS7c knobs only: a gate on the MS7d code refuses it.
    assert not hit(plain, base | {"noise_scales": "pos=1,heading=1,size=1,box=1,miss=1,class=1"})
    off = heap[:-1]  # no --perceived
    for name in ge.MS7D_LEVERS:
        with pytest.raises(ValueError, match=f"--{name} needs --heap_k"):
            ge._build_cfg(ge._parse_args(argv + ["--backend", "isaac", f"--{name}"]))
        if name in ("hold_force", "force_cap"):  # the teacher's hold calibration runs L6
            assert f"--{name}" in ge._eval_command(
                ge._build_cfg(ge._parse_args(argv + off + [f"--{name}"])), draw
            )
        else:
            with pytest.raises(ValueError, match=f"--{name} needs --perceived"):
                ge._build_cfg(ge._parse_args(argv + off + [f"--{name}"]))


def test_the_supervisors_copy_of_the_scales_parse_matches_the_perception_module():
    from maker_arm_lab.core import perception

    assert ge.PERCEPTION_KNOBS == perception.KNOBS
    assert ge.PERCEPTION_EXTRA_KNOBS == perception.EXTRA_KNOBS
    texts = ("", "heading=2", "all=0,heading=1", " pos=0.5 , miss=4", "class=1e-3", "all=2.50")
    for text in texts + ("pos_z=0", "all=2,pos_z=0.5", "pos_z=1"):
        expected = perception.format_scales(perception.parse_scales(text))
        assert ge.noise_scales_key(text) == expected, text
    for bad in ("bogus=1", "heading=-1", "pos=nan", "miss=inf", "heading=", "size=x"):
        with pytest.raises(ValueError):
            perception.parse_scales(bad)
        with pytest.raises(ValueError):
            ge.noise_scales_key(bad)


# --- MS7e (MS7e design section 10.2) -----------------------------------------------------------

MS7E_HEAP = ["--backend", "mujoco", "--heap_k", "50"]


def test_the_supervisors_ms7e_copies_match_the_core_modules():
    import argparse

    from maker_arm_lab.core import batch_loop, parts

    assert ge.FAMILY_KEYS == tuple(parts.FAMILY_KEYS)
    assert ge.READY_FAMILIES == parts.READY_FAMILIES
    assert ge.UNBUILT_SWITCHES == batch_loop.UNBUILT_SWITCHES
    assert ge.MS7E_OFF == batch_loop.ms7e_switches(argparse.Namespace())
    for text in ("", "flat", "flat,round", " ldraw , round,flat,round"):
        assert ge.families_key(text) == batch_loop.families_arg(text), text
    for bad in ("cone", "round,plate"):
        with pytest.raises(ValueError):
            ge.families_key(bad)
        with pytest.raises(argparse.ArgumentTypeError):
            batch_loop.families_arg(bad)
    cfg = _cfg(Path(), Path(), families=("round",), box_draw=6, heap_family="round")
    args = argparse.Namespace(families=("round",), box_draw=6, heap_family="round")
    assert ge.ms7e_block(cfg) == batch_loop.ms7e_switches(args)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--families", "round"], "--families round is not built"),
        (["--families", "flat,ldraw"], "--families ldraw,flat is not built"),
        (["--families", "cone"], "unknown family keys"),
        (["--box_draw", "6"], "--box_draw is not built"),
        (["--release_lean"], "--release_lean is not built"),
        (["--box_unseen", "size"], "--box_unseen is not built"),
        (["--heap_family", "flat"], "--heap_family must name one of --families"),
    ],
)
def test_an_unbuilt_ms7e_switch_is_a_gate_error_before_any_server(
    monkeypatch, capsys, tmp_path, extra, message
):
    """The base commit's state, whatever the tracks have built since."""
    monkeypatch.setattr(ge, "READY_FAMILIES", frozenset())
    monkeypatch.setattr(ge, "UNBUILT_SWITCHES", ("box_draw", "release_lean", "box_unseen"))
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(_argv(tmp_path) + MS7E_HEAP + extra) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and message in out.err


def test_the_ms7e_switches_need_the_mujoco_heap(monkeypatch):
    monkeypatch.setattr(ge, "READY_FAMILIES", frozenset(ge.FAMILY_KEYS))
    with pytest.raises(ValueError, match="need --heap_k > 0"):
        ge._build_cfg(ge._parse_args(_argv(Path()) + ["--families", "round"]))


def test_the_supervisor_forwards_every_ms7e_switch_and_its_cache_matches_them(
    tmp_path, monkeypatch
):
    """With every switch built, each reaches the heap client, and the cache refuses a payload
    read with any other value. An MS7d payload (no block) ran with every switch off."""
    monkeypatch.setattr(ge, "READY_FAMILIES", frozenset(ge.FAMILY_KEYS))
    monkeypatch.setattr(ge, "UNBUILT_SWITCHES", ())
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _argv(tmp_path / "gate") + MS7E_HEAP + ["--desk_boxes", "4", "--pick_snap"]
    plain = ge._build_cfg(ge._parse_args(argv))
    assert plain.families == () and ge.ms7e_block(plain) == ge.MS7E_OFF
    switches = ["--families", "flat,round", "--box_draw", "6", "--release_lean", "--carry_via"]
    switches += ["--box_unseen", "size", "--heap_family", "flat", "--passenger_check"]
    switches += ["--thin_snap", "--snap_closed"]
    cfg = ge._build_cfg(ge._parse_args(argv + switches))
    cmd = ge._eval_command(cfg, draw)
    at = cmd.index("--families")
    assert cmd[at : at + 13] == ["--families", "round,flat", *switches[2:]]
    block = ge.ms7e_block(cfg)
    assert block == {
        "families": ["round", "flat"],
        "box_draw": 6,
        "release_lean": True,
        "carry_via": True,
        "box_unseen": "size",
        "heap_family": "flat",
        "passenger_check": True,
        "wall_band_cont": False,  # the continuous gripper's switch, its own test
        "thin_snap": True,
        "snap_closed": True,
    }
    off = set(ge._eval_command(plain, draw))
    ms7e_flags = {"--families", "--box_draw", "--release_lean", "--carry_via", "--passenger_check"}
    assert not (ms7e_flags | {"--thin_snap", "--snap_closed"}) & off
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir(parents=True)

    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "loop": {"pick_snap": True}}

    def hit(gate_cfg, **extra):
        payload = _heap_payload(draw, inference=served, desk_boxes=4, **extra)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate_cfg, draw) is not None

    assert hit(plain) and hit(plain, ms7e=ge.MS7E_OFF) and not hit(plain, ms7e=block)
    # a payload whose block predates --carry_via ran with it off
    assert hit(plain, ms7e={k: v for k, v in ge.MS7E_OFF.items() if k != "carry_via"})
    with pytest.raises(ValueError, match="needs --release_lean"):
        ge._build_cfg(ge._parse_args(argv + ["--carry_via"]))
    no_snap = [a for a in argv if a != "--pick_snap"]
    for flag in ("--thin_snap", "--snap_closed"):
        with pytest.raises(ValueError, match="needs --pick_snap"):
            ge._build_cfg(ge._parse_args(no_snap + [flag]))
    assert hit(cfg, ms7e=block) and not hit(cfg) and not hit(cfg, ms7e=ge.MS7E_OFF)
    for name, other in (
        ("families", ["round"]),
        ("box_draw", 8),
        ("release_lean", False),
        ("carry_via", False),
        ("box_unseen", "rim"),
        ("heap_family", "round"),
        ("passenger_check", False),
        ("thin_snap", False),
        ("snap_closed", False),
    ):
        assert not hit(cfg, ms7e=block | {name: other}), name
    ledger = _fake_ledger(tmp_path)
    boxes = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50, box_draw=6)
    results = ge.run_gate(boxes, lambda d: _heap_payload(d), lambda: _FakeServer([]))
    assert results["ms7e"] == ge.MS7E_OFF | {"box_draw": 6}


def test_the_supervisor_forwards_deep_box_floor_and_its_cache_matches_it(tmp_path):
    """MS7e flat lever 1. A payload that predates the switch ran with it off."""
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _argv(tmp_path / "gate") + MS7E_HEAP + ["--desk_boxes", "4"]
    plain = ge._build_cfg(ge._parse_args(argv))
    deep = ge._build_cfg(ge._parse_args(argv + ["--deep_box_floor"]))
    assert "--deep_box_floor" in ge._eval_command(deep, draw)
    assert "--deep_box_floor" not in ge._eval_command(plain, draw)
    with pytest.raises(ValueError, match="--deep_box_floor"):
        ge._build_cfg(ge._parse_args(_argv(tmp_path) + MS7E_HEAP + ["--deep_box_floor"]))
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir(parents=True)
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}

    def hit(gate_cfg, **extra):
        payload = _heap_payload(draw, inference=served, desk_boxes=4, **extra)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate_cfg, draw) is not None

    assert hit(plain) and not hit(deep)
    assert hit(deep, deep_box_floor=True) and not hit(plain, deep_box_floor=True)


def test_the_supervisor_forwards_wall_band_cont_and_its_cache_matches_it(tmp_path):
    """MS7e ledger R13 S3, on the continuous gripper only. A payload that predates the switch
    ran with it off."""
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _argv(tmp_path / "gate") + MS7E_HEAP + ["--gripper", "continuous"]
    plain = ge._build_cfg(ge._parse_args(argv))
    on = ge._build_cfg(ge._parse_args(argv + ["--wall_band_cont"]))
    assert "--wall_band_cont" in ge._eval_command(on, draw)
    assert "--wall_band_cont" not in ge._eval_command(plain, draw)
    assert ge.ms7e_block(on) == ge.MS7E_OFF | {"wall_band_cont": True}
    with pytest.raises(ValueError, match="needs --gripper continuous"):
        ge._build_cfg(ge._parse_args(_argv(tmp_path) + MS7E_HEAP + ["--wall_band_cont"]))
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir(parents=True)
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}

    def hit(gate_cfg, **extra):
        payload = _heap_payload(draw, inference=served, gripper="continuous", **extra)
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate_cfg, draw) is not None

    old = {k: v for k, v in ge.MS7E_OFF.items() if k != "wall_band_cont"}
    assert hit(plain) and hit(plain, ms7e=old) and not hit(on) and not hit(on, ms7e=old)
    assert hit(on, ms7e=ge.ms7e_block(on)) and not hit(plain, ms7e=ge.ms7e_block(on))


def test_the_gate_sums_the_family_and_box_count_rows():
    draw = ge.make_eval_draws(1, 20261400)[0]

    def ep(families, count, stowed):
        return dict(
            _episode(stowed),
            num_bricks=4,
            cycles=4,
            cycle_classes=[],
            stowed_by_class={"round": {"stowed": 1, "total": 2}},
            stowed_by_family=families,
            stowed_by_box=[{"stowed": stowed, "total": 4}]
            + [{"stowed": 0, "total": 0}] * (count - 1),
            box_count=count,
            wrong_box_parts=0,
        )

    a = ep({"brick": {"stowed": 1, "total": 2}, "round": {"stowed": 1, "total": 2}}, 5, 2)
    b = ep({"round": {"stowed": 3, "total": 4}}, 4, 3)
    rows = ge.aggregate_gate([_payload(draw, [a, b, a])], threshold=0.5)["secondary"]
    assert rows["stowed_by_family"] == {
        "brick": {"stowed": 2, "total": 4},
        "round": {"stowed": 5, "total": 8},
    }
    assert rows["stowed_by_box_count"] == {
        "4": {"stowed": 3, "total": 4},
        "5": {"stowed": 4, "total": 8},
    }
    assert list(rows["stowed_by_box_count"]) == ["4", "5"]
    # The box draw gives each episode its own number of boxes (a crash on 2026-09-29).
    assert rows["stowed_by_box"] == [{"stowed": 7, "total": 12}] + [{"stowed": 0, "total": 0}] * 4


def test_the_supervisors_ms7f_copies_match_the_core_modules():
    import argparse

    from maker_arm_lab.core import batch_loop

    assert ge.MS7F_OFF == batch_loop.MS7F_OFF == batch_loop.ms7f_switches(argparse.Namespace())
    assert ge.MS7F_READY == batch_loop.MS7F_READY


def test_the_supervisor_forwards_categories_and_its_cache_matches_it(tmp_path, capsys):
    draw = ge.make_eval_draws(1, 1)[0]
    argv = _argv(tmp_path / "gate") + MS7E_HEAP + ["--desk_boxes", "8", "--box_draw", "6"]
    plain = ge._build_cfg(ge._parse_args(argv))
    on = ge._build_cfg(ge._parse_args(argv + ["--categories"]))
    assert "--categories" in ge._eval_command(on, draw)
    assert "--categories" not in ge._eval_command(plain, draw)
    assert ge.ms7f_block(plain) == ge.MS7F_OFF and ge.ms7f_block(on)["categories"] is True
    path = tmp_path / "gate" / "eval_000.json"
    path.parent.mkdir(parents=True)

    def hit(gate_cfg, **extra):
        served = {"n_action_steps": 50, "temporal_ensemble_coeff": None}
        payload = _heap_payload(draw, inference=served, desk_boxes=8, **extra)
        payload["ms7e"] = ge.MS7E_OFF | {"box_draw": 6}
        path.write_text(json.dumps(payload))
        return ge._cached_payload(gate_cfg, draw) is not None

    # An MS7e payload (no block) ran with every MS7f switch off.
    assert hit(plain) and hit(plain, ms7f=ge.MS7F_OFF) and not hit(plain, ms7f=ge.ms7f_block(on))
    assert hit(on, ms7f=ge.ms7f_block(on)) and not hit(on) and not hit(on, ms7f=ge.MS7F_OFF)
    no_draw = [a for a in argv if a not in ("--box_draw", "6")]
    with pytest.raises(ValueError, match="needs --box_draw"):
        ge._build_cfg(ge._parse_args(no_draw + ["--categories"]))
    with pytest.raises(ValueError, match="need --heap_k > 0|needs --heap_k > 0"):
        ge._build_cfg(ge._parse_args(_argv(tmp_path) + ["--categories", "--box_draw", "6"]))
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50, box_draw=6)
    cfg.ms7f = {"categories": True}
    results = ge.run_gate(cfg, lambda d: _heap_payload(d), lambda: _FakeServer([]))
    assert results["ms7f"] == ge.MS7F_OFF | {"categories": True}


def test_the_gate_sums_the_throughput_and_the_heldout_row_and_gates_them_only_when_set():
    """MS7f design section 4.2: with both bars unset the verdict is MS7e's; each set bar gates."""
    from maker_arm_lab.core import loop

    assert ge.CONTROL_HZ == loop.CONTROL_HZ and ge.INTERVAL_NAMES == loop.INTERVAL_NAMES
    eps = [
        dict(
            _episode(0),
            num_bricks=150,
            bricks_stowed=140,
            steps=54000,
            cycles=150,
            stowed_heldout={"stowed": 17, "total": 20},
            events={"tray_refill": 2, "box_empty": 1},
            cycle_classes=["stowed", "held_no_lift"],
            cycle_intervals=[[1, 2, 3, 4], [5, 6, 0, 0]],
            stowed_by_category={"brick": {"stowed": 90, "total": 95}},
        ),
        dict(
            _episode(0),
            num_bricks=150,
            bricks_stowed=136,
            steps=54000,
            cycles=150,
            stowed_heldout={"stowed": 16, "total": 20},
            events={"tray_refill": 1, "box_empty": 0},
            cycle_classes=["stowed"],
            cycle_intervals=[[10, 20, 30, 40]],
            stowed_by_category={"brick": {"stowed": 80, "total": 90}},
        ),
    ]
    payload = {"episodes": eps}
    plain = ge.aggregate_gate([payload], 0.9)
    gate = plain["gate"]
    assert gate["passed"] and gate["steps_total"] == 108000
    assert gate["stows_per_hour"] == pytest.approx(3600 * 276 / (108000 / 30))  # 276 an hour
    assert gate["heldout"]["stowed"] == 33 and gate["heldout"]["total"] == 40
    assert gate["heldout"]["rate"] == pytest.approx(0.825) and gate["heldout_passed"]
    assert gate["stows_per_hour_passed"] and gate["min_stows_per_hour"] is None
    assert not ge.aggregate_gate([payload], 0.9, min_stows_per_hour=300)["gate"]["passed"]
    assert ge.aggregate_gate([payload], 0.9, min_stows_per_hour=270)["gate"]["passed"]
    assert not ge.aggregate_gate([payload], 0.9, heldout_threshold=0.85)["gate"]["passed"]
    assert ge.aggregate_gate([payload], 0.9, heldout_threshold=0.80)["gate"]["passed"]
    empty = {"episodes": [dict(e, stowed_heldout={"stowed": 0, "total": 0}) for e in eps]}
    assert not ge.aggregate_gate([empty], 0.9, heldout_threshold=0.0)["gate"]["heldout_passed"]
    rows = plain["secondary"]
    assert rows["events"] == {"box_empty": 1, "tray_refill": 3}
    assert rows["stowed_by_category"] == {"brick": {"stowed": 170, "total": 185}}
    assert rows["cycle_intervals"]["by_outcome"] == loop.interval_sums(eps)["by_outcome"]
    # An MS7e payload has none of the MS7f rows and the same verdict.
    old = {"episodes": [_episode(3, num_bricks=6)]}
    assert "events" not in ge.aggregate_gate([old], 0.4)["secondary"]
    assert ge.aggregate_gate([old], 0.4)["gate"]["passed"]


def test_the_bars_reach_the_gate_results_from_the_command_line(tmp_path):
    argv = _argv(tmp_path) + MS7E_HEAP
    cfg = ge._build_cfg(ge._parse_args(argv))
    assert cfg.min_stows_per_hour is None and cfg.heldout_threshold is None
    more = ["--min_stows_per_hour", "300", "--heldout_threshold", "0.85"]
    cfg = ge._build_cfg(ge._parse_args(argv + more))
    assert cfg.min_stows_per_hour == 300.0 and cfg.heldout_threshold == 0.85
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50)
    cfg.min_stows_per_hour, cfg.heldout_threshold = 300.0, 0.85
    results = ge.run_gate(cfg, lambda d: _heap_payload(d), lambda: _FakeServer([]))
    gate = results["gate"]
    assert gate["min_stows_per_hour"] == 300.0 and gate["heldout_threshold"] == 0.85
    assert not gate["passed"] and not gate["heldout_passed"]  # no held-out part in the read


# --- MS7h design 7.3: the pi0.5 read ---------------------------------------------------------

PI05_HEAP = ["--backend", "mujoco", "--heap_k", "50", "--context", "camera_box_target"]


def test_an_act_read_keeps_its_commands_and_a_pi05_read_starts_one_shared_server(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--n_action_steps", "50", "--action_warp", "1.5"]
    read += ["--port", "7100", "--processes", "2"]
    act = ge._build_cfg(ge._parse_args(read))
    pi05 = ge._build_cfg(ge._parse_args(read + ["--pi05", "--lanes", "2"]))
    assert (act.pi05, act.lanes, pi05.pi05, pi05.lanes) == (False, 1, True, 2)
    assert ge._server_command(act) == [
        "lerobot_stack/.venv/bin/python",
        "lerobot_stack/policy_server.py",
        "--checkpoint",
        "ck",
        "--port",
        "7100",
        "--device",
        "cuda",
        "--n_action_steps",
        "50",
        "--action_warp",
        "1.5",
    ]
    # The checkpoint's own chunk (50) and flow steps (10) serve open loop; the warp is a flag.
    assert ge._pi05_server_command(pi05) == [
        "lerobot_stack/.venv-pi/bin/python",
        "lerobot_stack/pi05_server.py",
        "--checkpoint",
        "ck",
        "--port",
        "7100",
        "--device",
        "cuda",
        "--action_warp",
        "1.5",
    ]
    # Every lane's client is the unchanged one, pointed at the one server.
    for draw in ge.make_eval_draws(2, 1):
        assert ge._eval_command(pi05, draw) == ge._eval_command(act, draw)
        assert "tcp://127.0.0.1:7100" in ge._eval_command(pi05, draw)


def test_the_cache_never_mixes_a_pi05_payload_with_an_act_one(tmp_path):
    ledger = _fake_ledger(tmp_path)
    act = _cfg(tmp_path, ledger, processes=1)
    pi05 = _cfg(tmp_path, ledger, processes=1, pi05=True)
    act.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]

    def cached(cfg, policy_id):
        payload = _payload(draw, [_episode(6)], policy_id=policy_id)
        (act.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert cached(act, "act:ck+te0.01") and not cached(act, "pi05:ck+warp1.5")
    assert cached(pi05, "pi05:ck+warp1.5") and not cached(pi05, "act:ck+te0.01")


def test_the_pi05_sampling_switches_reach_the_server_and_split_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--action_warp", "1.5"]
    plain = ge._build_cfg(ge._parse_args(read))
    switched = ge._build_cfg(
        ge._parse_args(read + ["--pi05_num_steps", "50", "--pi05_noise_scale", "0"])
    )
    assert (plain.pi05_num_steps, plain.pi05_noise_scale) == (None, 1.0)
    assert ge._pi05_server_command(switched) == ge._pi05_server_command(plain) + [
        "--num_steps",
        "50",
        "--noise_scale",
        "0.0",
    ]
    ledger = _fake_ledger(tmp_path)
    default = _cfg(tmp_path, ledger, processes=1, pi05=True)
    steps_noise = _cfg(
        tmp_path, ledger, processes=1, pi05=True, pi05_num_steps=50, pi05_noise_scale=0.0
    )
    default.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, tail, **inference):
        payload = _payload(draw, [_episode(6)], policy_id=f"pi05:ck{tail}")
        payload["inference"] = served | inference
        (default.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    # A read reuses only a payload served with its own flow steps and noise scale.
    assert cached(default, "") and not cached(default, "+noise0.0", noise_scale=0.0)
    assert not cached(default, "+steps50", num_inference_steps=50)
    assert cached(steps_noise, "+steps50+noise0.0", num_inference_steps=50, noise_scale=0.0)
    assert not cached(steps_noise, "+noise0.0", noise_scale=0.0)  # the checkpoint's 10 steps
    assert not cached(steps_noise, "+steps50", num_inference_steps=50)


def test_the_pi05_jaw_noise_scale_reaches_the_server_and_splits_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--pi05_noise_scale", "0.5"]
    plain = ge._build_cfg(ge._parse_args(read))
    jaw = ge._build_cfg(ge._parse_args(read + ["--pi05_jaw_noise_scale", "0.25"]))
    same = ge._build_cfg(ge._parse_args(read + ["--pi05_jaw_noise_scale", "1"]))
    assert (plain.pi05_jaw_noise_scale, same.pi05_jaw_noise_scale) == (1.0, 1.0)
    assert ge._pi05_server_command(same) == ge._pi05_server_command(plain)  # 1.0: the same bytes
    command = ge._pi05_server_command(plain)
    at = command.index("--noise_scale") + 2
    assert (
        ge._pi05_server_command(jaw) == command[:at] + ["--jaw_noise_scale", "0.25"] + command[at:]
    )
    shared = read + ["--pi05_jaw_noise_scale", "0.25", "--pi05_jaw_noise_shared"]
    shared = ge._build_cfg(ge._parse_args(shared))
    assert (plain.pi05_jaw_noise_shared, shared.pi05_jaw_noise_shared) == (False, True)
    assert ge._pi05_server_command(shared) == ge._pi05_server_command(jaw) + ["--jaw_noise_shared"]
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_jaw_noise_scale=0.25)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, **inference):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = served | inference
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert cached(off) and not cached(off, jaw_noise_scale=0.25)
    assert cached(on, jaw_noise_scale=0.25) and not cached(on)
    assert not cached(on, jaw_noise_scale=0.5)
    shared = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_jaw_noise_shared=True)
    assert cached(shared, jaw_noise_shared=True) and not cached(shared)
    assert not cached(off, jaw_noise_shared=True) and not cached(on, jaw_noise_shared=True)
    jaw_abs = ge._build_cfg(ge._parse_args(read + ["--pi05_jaw_noise_abs", "1"]))
    assert (plain.pi05_jaw_noise_abs, jaw_abs.pi05_jaw_noise_abs) == (None, 1.0)
    assert ge._pi05_server_command(jaw_abs) == command + ["--jaw_noise_abs", "1.0"]
    jaw_abs = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_jaw_noise_abs=1.0)
    assert cached(jaw_abs, jaw_noise_abs=1.0) and not cached(jaw_abs)
    assert not cached(off, jaw_noise_abs=1.0) and not cached(jaw_abs, jaw_noise_abs=0.5)


def test_the_pi05_still_rule_reaches_the_server_and_splits_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--pi05_noise_scale", "0"]
    plain = ge._build_cfg(ge._parse_args(read))
    on = ge._build_cfg(ge._parse_args(read + ["--pi05_still_noise_scale", "1"]))
    tuned = read + ["--pi05_still_noise_scale", "0.7", "--pi05_still_rad", "0.1"]
    tuned = ge._build_cfg(ge._parse_args(tuned + ["--pi05_still_open", "0.8"]))
    assert plain.pi05_still_noise_scale is None and ge._pi05_still(plain) == {}
    assert "--still_noise_scale" not in ge._pi05_server_command(plain)  # off: the same bytes
    default = ["--still_noise_scale", "1.0", "--still_rad", "0.15", "--still_open", "0.9"]
    assert ge._pi05_server_command(on) == ge._pi05_server_command(plain) + default
    assert ge._pi05_server_command(tuned)[-6:] == [
        "--still_noise_scale",
        "0.7",
        "--still_rad",
        "0.1",
        "--still_open",
        "0.8",
    ]
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_still_noise_scale=1.0)
    tuned = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_still_noise_scale=0.7)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, **still):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = served | still
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    # A read reuses only a payload served with its own still rule, or none without one.
    rule = {"still_noise_scale": 1.0, "still_rad": 0.15, "still_open": 0.9}
    assert cached(off) and not cached(off, **rule)
    assert cached(on, **rule) and not cached(on) and not cached(tuned, **rule)
    assert not cached(on, **(rule | {"still_rad": 0.1}))
    assert not cached(on, **(rule | {"still_open": 0.8}))


def test_the_pi05_jaw_latch_reaches_the_server_and_splits_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--pi05_still_noise_scale", "1"]
    plain = ge._build_cfg(ge._parse_args(read))
    zero = ge._build_cfg(ge._parse_args(read + ["--pi05_jaw_latch", "0"]))
    latch = ge._build_cfg(ge._parse_args(read + ["--pi05_jaw_latch", "10"]))
    assert (plain.pi05_jaw_latch, zero.pi05_jaw_latch, latch.pi05_jaw_latch) == (0, 0, 10)
    assert ge._pi05_server_command(zero) == ge._pi05_server_command(plain)  # 0: the same bytes
    assert ge._pi05_server_command(latch) == ge._pi05_server_command(plain) + ["--jaw_latch", "10"]
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_jaw_latch=10)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, **inference):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = served | inference
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert cached(off) and not cached(off, jaw_latch=10)
    assert cached(on, jaw_latch=10) and not cached(on) and not cached(on, jaw_latch=5)


def test_the_pi05_retry_noise_scale_reaches_the_server_and_splits_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--pi05_noise_scale", "0"]
    plain = ge._build_cfg(ge._parse_args(read))
    retry = ge._build_cfg(ge._parse_args(read + ["--pi05_retry_noise_scale", "1"]))
    assert (plain.pi05_retry_noise_scale, retry.pi05_retry_noise_scale) == (None, 1.0)
    assert "--retry_noise_scale" not in ge._pi05_server_command(plain)  # off: the same bytes
    tail = ["--retry_noise_scale", "1.0"]
    assert ge._pi05_server_command(retry) == ge._pi05_server_command(plain) + tail
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_retry_noise_scale=1.0)
    zero = _cfg(tmp_path, ledger, processes=1, pi05=True, pi05_retry_noise_scale=0.0)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, **inference):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = served | inference
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert cached(off) and not cached(off, retry_noise_scale=1.0)
    assert cached(on, retry_noise_scale=1.0) and not cached(on)
    assert not cached(on, retry_noise_scale=0.0) and cached(zero, retry_noise_scale=0.0)


def test_the_supervisor_copies_the_still_defaults_of_the_server():
    source = (Path(__file__).parents[1] / "lerobot_stack" / "pi05_server.py").read_text()
    assert f"\nSTILL_RAD = {ge.GateCfg.pi05_still_rad}\n" in source
    assert f"\nSTILL_OPEN = {ge.GateCfg.pi05_still_open}\n" in source


def test_pi05_lanes_share_one_server_and_run_their_draws_at_once(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=4, pi05=True, lanes=2)
    log, ran = [], []
    both = threading.Barrier(2, timeout=10)  # breaks unless two draws run at the same time

    def run_eval(draw):
        both.wait()
        ran.append(draw.process_index)
        return _payload(draw, [_episode(1), _episode(0)])

    results = ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert log == ["start", "stop"] and sorted(ran) == [0, 1, 2, 3]
    assert results["gate"]["episodes"] == 8
    assert all((cfg.out_dir / f"eval_{i:03d}.json").exists() for i in range(4))


def test_a_failed_pi05_lane_starts_no_new_draw_and_still_stops_the_server(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=6, pi05=True, lanes=2, retries=0)
    log, ran = [], set()
    release = threading.Event()
    threading.Timer(1.0, release.set).start()

    def run_eval(draw):
        if draw.process_index == 0:
            raise ge.EvalProcessError("boom")
        ran.add(draw.process_index)
        release.wait(10)
        return _payload(draw, [_episode(0)])

    with pytest.raises(ge.GateRunError, match="process 0 failed 1 times"):
        ge.run_gate(cfg, run_eval, lambda: _FakeServer(log))
    assert log == ["start", "stop"]
    assert 1 in ran and ran <= {1, 2}  # the lane that failed may have started one more draw


def _fake_pi05_server(monkeypatch, code: str) -> None:
    monkeypatch.setattr(ge, "_pi05_server_command", lambda cfg: [sys.executable, "-c", code])


def test_the_pi05_server_logs_to_the_out_dir_and_the_supervisor_waits_for_ready(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path, tmp_path / "l.json", pi05=True, n_action_steps=50)
    cfg.out_dir.mkdir(parents=True)
    ready = {"policy_id": "pi05:ck+warp1.5", "n_action_steps": 50}
    _fake_pi05_server(
        monkeypatch,
        f"import time; print('SERVER_READY ' + {json.dumps(json.dumps(ready))}, flush=True); "
        "print('FORWARD rows=2 ms=200.0', flush=True); time.sleep(120)",
    )
    server = ge._start_pi05_server(cfg)
    assert server.ready == ready
    server.stop()
    assert server._proc.returncode is not None
    assert "SERVER_READY" in (cfg.out_dir / "pi05_server.log").read_text()
    _fake_pi05_server(
        monkeypatch, "print('SERVER_FAIL ValueError: no marker'); raise SystemExit(1)"
    )
    with pytest.raises(ge.EvalProcessError, match="no marker"):
        ge._start_pi05_server(cfg)
    # A checkpoint whose chunk is not the read's 50 actions a forward never serves it.
    ready["n_action_steps"] = 33
    _fake_pi05_server(
        monkeypatch,
        f"import time; print('SERVER_READY ' + {json.dumps(json.dumps(ready))}, flush=True); "
        "time.sleep(120)",
    )
    with pytest.raises(ge.EvalProcessError, match="serves 33 actions"):
        ge._start_pi05_server(cfg)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--lanes", "2"], "with --pi05"),
        (["--pi05", "--lanes", "0"], "--lanes 0"),
        (["--pi05", "--heap_k", "0"], "--pi05 needs"),
        (["--pi05", "--context", "camera_box"], "--pi05 needs"),
        (["--pi05", "--temporal_ensemble_coeff", "0.01"], "no ensemble"),
        (["--pi05", "--ensemble_with", "/m/b"], "no ensemble"),
        (["--pi05_num_steps", "50"], "need --pi05"),
        (["--pi05_noise_scale", "1.0"], "need --pi05"),
        (["--pi05", "--pi05_num_steps", "0"], "--pi05_num_steps must be at least 1"),
        (["--pi05", "--pi05_noise_scale", "-1"], "--pi05_noise_scale must be finite"),
        (["--pi05", "--pi05_noise_scale", "inf"], "--pi05_noise_scale must be finite"),
        (["--pi05_jaw_noise_scale", "1.0"], "need --pi05"),
        (["--pi05_jaw_noise_shared"], "need --pi05"),
        (["--pi05_jaw_latch", "10"], "need --pi05"),
        (["--pi05", "--pi05_jaw_latch", "-1"], "--pi05_jaw_latch must be at least 0"),
        (["--pi05", "--pi05_jaw_noise_scale", "-1"], "--pi05_jaw_noise_scale must be finite"),
        (["--pi05", "--pi05_jaw_noise_scale", "nan"], "--pi05_jaw_noise_scale must be finite"),
        (["--pi05_jaw_noise_abs", "1"], "need --pi05"),
        (["--pi05", "--pi05_jaw_noise_abs", "-1"], "--pi05_jaw_noise_abs must be finite"),
        (
            ["--pi05", "--pi05_jaw_noise_abs", "1", "--pi05_jaw_noise_scale", "0.5"],
            "needs --pi05_jaw",
        ),
        (["--pi05", "--pi05_jaw_noise_abs", "1", "--pi05_jaw_noise_shared"], "needs --pi05_jaw"),
        (["--pi05_still_noise_scale", "1.0"], "need --pi05"),
        (["--pi05_still_open", "0.9"], "need --pi05"),
        (["--pi05", "--pi05_still_rad", "0.1"], "need --pi05_still_noise_scale"),
        (["--pi05", "--pi05_still_noise_scale", "-1"], "--pi05_still_noise_scale must be"),
        (["--pi05", "--pi05_still_noise_scale", "1", "--pi05_still_rad", "0"], "_rad must be"),
        (["--pi05", "--pi05_still_noise_scale", "1", "--pi05_still_open", "2"], "_open must be"),
        (["--pi05_retry_noise_scale", "1"], "need --pi05"),
        (["--pi05", "--pi05_retry_noise_scale", "-1"], "--pi05_retry_noise_scale must be finite"),
        (["--pi05", "--pi05_retry_noise_scale", "inf"], "--pi05_retry_noise_scale must be finite"),
        (["--pi05_rtc", "--infer_delay", "6"], "--pi05_rtc needs --pi05"),
        (["--pi05", "--pi05_rtc"], "--pi05_rtc needs"),
        (["--pi05", "--pi05_rtc", "--infer_delay", "26"], "--pi05_rtc needs"),
        (
            ["--pi05", "--pi05_rtc", "--infer_delay", "6", "--action_warp", "1.5"],
            "--pi05_rtc needs",
        ),
    ],
)
def test_the_pi05_mode_refuses_what_its_server_cannot_serve(
    extra, message, monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(ge, "_start_pi05_server", lambda cfg: pytest.fail("no server"))
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server"))
    assert ge.main(_argv(tmp_path) + PI05_HEAP + extra) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and message in out.err


STRESS_EXAMPLES = [
    [],
    ["--overhead_offset", "0,3,0,0,0,0", "--jpeg_quality", "85"],
    ["--wrist_offset=-1,0,0,0,0,0.01", "--wrist_fov_delta", "2", "--warp_back", "--obs_delay", "3"],
    ["--sparse_gap", "0.05", "--tray", "--blur_sigma", "1.5", "--hard_light"],
]


def test_the_supervisors_stress_copy_matches_the_core_module():
    import argparse

    from maker_arm_lab.mj import stress
    from maker_arm_lab.mj.stress import add_stress_arguments, stress_from_args

    assert ge.STRESS_KNOBS == stress.STRESS_KNOBS
    for flags in STRESS_EXAMPLES:
        parser = argparse.ArgumentParser()
        add_stress_arguments(parser)
        parser.add_argument("--tray", action="store_true")
        assert ge.stress_block(flags) == stress_from_args(parser.parse_known_args(flags)[0]).block()


def test_a_cached_draw_with_another_stress_block_is_not_reused(tmp_path):
    ledger = _fake_ledger(tmp_path)
    cfg = _cfg(tmp_path, ledger, processes=1, backend="mujoco", heap_k=50)
    cfg.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]

    def cached(cfg, **extra):
        (cfg.out_dir / "eval_000.json").write_text(json.dumps(_heap_payload(draw, **extra)))
        return ge._cached_payload(cfg, draw) is not None

    block = {"jpeg_quality": 85}
    assert cached(cfg) and not cached(cfg, stress=block)
    cfg.client_flags = ("--jpeg_quality", "85")
    assert cached(cfg, stress=block) and not cached(cfg)


def test_a_malformed_stress_knob_is_a_gate_error_on_a_resumed_read(tmp_path, monkeypatch, capsys):
    for flags in (["--jpeg_quality", "abc"], ["--overhead_offset", "-3,0,0,0,0,0"]):
        with pytest.raises(ValueError, match=flags[0]):
            ge.stress_block(flags)
    ledger = _fake_ledger(tmp_path)
    out_dir = tmp_path / "gate"
    out_dir.mkdir()
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    (out_dir / "eval_000.json").write_text(json.dumps(_heap_payload(draw)))
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server"))
    argv = ["--checkpoint", "ck", "--out_dir", str(out_dir), "--ledger", str(ledger)]
    argv += ["--threshold", "0.85", "--seed", str(SELECTION_SEED), "--processes", "1"]
    argv += HEAP + ["--client_flags", "--jpeg_quality abc"]
    assert ge.main(argv) == 2
    out = capsys.readouterr()
    tokens = [x for x in out.out.splitlines() if x.startswith("GATE_")]
    assert tokens == ["GATE_ERROR"] and "--jpeg_quality" in out.err


def test_the_supervisor_sends_infer_delay_and_keys_the_cache(tmp_path, monkeypatch, capsys):
    read = _argv(tmp_path) + PI05_HEAP + ["--n_action_steps", "50"]
    plain = ge._build_cfg(ge._parse_args(read))
    act = ge._build_cfg(ge._parse_args(read + ["--infer_delay", "1"]))
    pi05 = ge._build_cfg(ge._parse_args(read + ["--pi05", "--infer_delay", "6"]))
    assert (plain.infer_delay, act.infer_delay, pi05.infer_delay) == (0, 1, 6)
    assert "--infer_delay" not in ge._server_command(plain)  # off: the same bytes
    assert ge._server_command(act) == ge._server_command(plain) + ["--infer_delay", "1"]
    pi05_plain = ge._build_cfg(ge._parse_args(read + ["--pi05"]))
    assert "--infer_delay" not in ge._pi05_server_command(pi05_plain)
    assert ge._pi05_server_command(pi05) == ge._pi05_server_command(pi05_plain) + [
        "--infer_delay",
        "6",
    ]
    ledger = _fake_ledger(tmp_path)
    open_loop = {"n_action_steps": None, "temporal_ensemble_coeff": None}
    off = _cfg(tmp_path, ledger, processes=1, **open_loop)
    on = _cfg(tmp_path, ledger, processes=1, infer_delay=1, **open_loop)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "chunk_size": 100}

    def cached(cfg, **inference):
        payload = _payload(draw, [_episode(6)], policy_id="act:ck+nas50")
        payload["inference"] = served | inference
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    # A payload without the key, every one before the delay, ran without one.
    assert cached(off) and not cached(off, infer_delay=1)
    assert cached(on, infer_delay=1) and not cached(on) and not cached(on, infer_delay=6)
    monkeypatch.setattr(ge, "_start_server", lambda cfg: pytest.fail("no server on a usage error"))
    assert ge.main(_argv(tmp_path) + ["--infer_delay", "-1"]) == 2
    out = capsys.readouterr()
    assert out.out.strip().splitlines()[-1] == "GATE_ERROR" and "--infer_delay" in out.err


def test_the_pi05_rtc_reaches_the_server_and_splits_the_cache(tmp_path):
    read = _argv(tmp_path) + PI05_HEAP + ["--pi05", "--infer_delay", "6"]
    plain = ge._build_cfg(ge._parse_args(read))
    rtc = ge._build_cfg(ge._parse_args(read + ["--pi05_rtc"]))
    assert (plain.pi05_rtc, rtc.pi05_rtc) == (False, True)
    assert "--rtc" not in ge._pi05_server_command(plain)  # off: the same bytes
    assert ge._pi05_server_command(rtc) == ge._pi05_server_command(plain) + ["--rtc"]
    ledger = _fake_ledger(tmp_path)
    off = _cfg(tmp_path, ledger, processes=1, pi05=True, infer_delay=6)
    on = _cfg(tmp_path, ledger, processes=1, pi05=True, infer_delay=6, pi05_rtc=True)
    off.out_dir.mkdir(parents=True)
    draw = ge.make_eval_draws(1, SELECTION_SEED)[0]
    served = {"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "num_inference_steps": 10}

    def cached(cfg, **inference):
        payload = _payload(draw, [_episode(6)], policy_id="pi05:ck")
        payload["inference"] = served | {"infer_delay": 6} | inference
        (off.out_dir / "eval_000.json").write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert cached(off) and not cached(off, rtc=25)
    assert cached(on, rtc=25) and not cached(on) and not cached(on, rtc=10)
    source = (Path(__file__).parents[1] / "lerobot_stack" / "pi05_server.py").read_text()
    assert f"\nRTC_HORIZON = {ge.PI05_RTC_HORIZON}\n" in source

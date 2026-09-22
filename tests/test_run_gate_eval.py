"""CPU tests for scripts/run_gate_eval.py: draws, seed disjointness, aggregation, the
reuse of finished draws, and the token contract. The server and the eval processes are
injected fakes; the real ones need a GPU."""

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_run_gate_eval", Path(__file__).resolve().parents[1] / "scripts/run_gate_eval.py"
)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

_gd_spec = importlib.util.spec_from_file_location(
    "_generate_dataset", Path(__file__).resolve().parents[1] / "scripts/generate_dataset.py"
)
gd = importlib.util.module_from_spec(_gd_spec)
_gd_spec.loader.exec_module(gd)

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

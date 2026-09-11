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

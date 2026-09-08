import importlib.util
import itertools
import json
import re
import subprocess
import threading
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_generate_dataset", Path(__file__).resolve().parents[1] / "scripts/generate_dataset.py"
)
gd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gd)


def _raise(exc: BaseException):
    """Build a run_campaign stand-in that raises `exc` as soon as it is called."""

    def raiser(*_args, **_kwargs):
        raise exc

    return raiser


def _hdf5_path(shard_path: Path) -> Path:
    """Mirrors generate_batch.py's real on-disk convention, verified against actual shard
    artifacts: --shard_name is an extensionless base name, and the subprocess writes
    "<shard_name>.hdf5" under --out_dir (the extensionless base path itself is never created).
    Deliberately reimplemented here rather than calling gd._shard_hdf5_path -- a regression in
    that helper must not be able to hide behind a test fake that calls the same buggy helper."""
    return shard_path.parent / f"{shard_path.name}.hdf5"


def test_permutation_schedule_is_without_replacement_per_cycle():
    schedule = gd.permutation_schedule(seed=7)
    first_cycle = [next(schedule) for _ in range(24)]
    assert len(set(first_cycle)) == 24
    assert sorted(first_cycle) == sorted(itertools.permutations(range(4)))
    second_cycle = [next(schedule) for _ in range(24)]
    assert sorted(second_cycle) == sorted(first_cycle)
    assert first_cycle != second_cycle  # reshuffled, not repeated verbatim


def test_parse_batch_output_extracts_the_contract():
    text = (
        'RUN_META {"bin_x": 0.45}\n'
        'RUN_STATS {"attempts": 50, "wall_sec": 1800.0}\n'
        "BATCH_OK exported=46 attempts=50\n"
    )
    out = gd.parse_batch_output(text)
    assert (out["exported"], out["attempts"]) == (46, 50)
    assert out["meta"]["bin_x"] == 0.45


def test_parse_batch_output_raises_without_token():
    try:
        gd.parse_batch_output("no token here")
        raise AssertionError("should have raised")
    except gd.BatchFailed:
        pass


def test_run_campaign_stops_at_target_and_deletes_shards(tmp_path):
    deleted = []
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=100,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        deleted.append(shard_path)  # convert precedes deletion; record the call
        return {"episodes": 45}

    total = gd.run_campaign(cfg, fake_batch, fake_convert)
    assert total >= 100
    assert len(deleted) == 3  # 45 + 45 + 45 -> crosses 100 on run 3
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["total_successes"] == total
    assert all(not (tmp_path / r["shard"]).exists() for r in ledger["runs"])


def test_run_campaign_resumes_from_ledger(tmp_path):
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=80,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    calls = []

    def fake_batch(run_index, permutation, shard_path):
        calls.append(run_index)
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        return {"episodes": 45}

    gd.run_campaign(cfg, fake_batch, fake_convert)  # runs 0 and 1 -> 90
    first_calls = list(calls)
    gd.run_campaign(cfg, fake_batch, fake_convert)  # already >= target
    assert calls == first_calls  # resume: nothing re-ran


# --- Pipelined generation/conversion ----------------------------------------------------


def test_run_campaign_overlaps_next_batch_with_conversion(tmp_path):
    """Batch 1's run_batch call must genuinely execute concurrently with batch 0's run_convert
    call -- not merely be *called* after it starts, which a fully-serial implementation that
    calls run_batch(k+1) before running convert(k) inline would also satisfy.

    Two-sided handshake: fake_convert(shard_0) announces it has started running
    (convert0_running) and then blocks until fake_batch(1) has itself started
    (batch1_started); fake_batch(1) blocks until convert0_running is set before proceeding to
    set batch1_started. Only genuine concurrency satisfies both waits -- a serial
    implementation where convert(shard_0) is not even *called* until run_batch(1) has already
    returned deadlocks on fake_batch(1)'s wait (convert0_running is never set in time) and
    times out into a loud assertion failure instead of hanging forever.
    """
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=90,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    convert0_running = threading.Event()
    batch1_started = threading.Event()

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        if run_index == 1:
            assert convert0_running.wait(timeout=5), (
                "batch 1 started before convert 0 was even running -- unexpected ordering"
            )
            batch1_started.set()
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        if shard_path.name == "shard_000.hdf5":
            convert0_running.set()
            assert batch1_started.wait(timeout=5), (
                "batch 1 never started while convert 0 was still in flight -- no overlap"
            )
        return {"episodes": 45}

    total = gd.run_campaign(cfg, fake_batch, fake_convert)
    assert total >= 90


def test_run_campaign_never_overlaps_two_conversions(tmp_path):
    """No two run_convert calls may execute concurrently (single LeRobotDataset root)."""
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=150,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        threading.Event().wait(0.02)  # give a concurrent convert time to overlap, if it could
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        threading.Event().wait(0.02)
        with lock:
            active -= 1
        return {"episodes": 45}

    gd.run_campaign(cfg, fake_batch, fake_convert)
    assert max_active <= 1


def test_run_campaign_pipelined_ledger_stays_correct_and_bounds_overshoot(tmp_path):
    """Ledger order/contents match the serial contract; overshoot is bounded to one run, and
    that in-flight run is still converted and counted, never discarded."""
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=100,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    converted_order = []

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        converted_order.append(shard_path.name)
        return {"episodes": 45}

    total = gd.run_campaign(cfg, fake_batch, fake_convert)
    ledger = json.loads((tmp_path / "ledger.json").read_text())

    run_indices = [r["run_index"] for r in ledger["runs"]]
    assert run_indices == sorted(run_indices)  # appended in run order
    assert run_indices == list(range(len(run_indices)))  # no run skipped
    assert converted_order == [r["shard"] for r in ledger["runs"]]  # convert-completion order
    assert all(r["converted"] and r["deleted"] for r in ledger["runs"])
    assert ledger["total_successes"] == total
    assert total >= cfg.target
    # Overshoot is bounded to the one run that was already generating/in-flight when the
    # target was reached -- that run is still converted and counted, not discarded, but no
    # further run is launched once the running total has reached the target.
    assert total - cfg.target < 45
    assert all(not (tmp_path / r["shard"]).exists() for r in ledger["runs"])


def test_run_campaign_convert_failure_is_clean_and_resume_regenerates(tmp_path):
    """A run_convert failure fails the campaign cleanly (ledger has only fully-converted runs
    before the failing one) and a resumed campaign regenerates the failed run from scratch."""
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=150,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    batch_calls = []

    def fake_batch(run_index, permutation, shard_path):
        batch_calls.append(run_index)
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def failing_convert(shard_path):
        if shard_path.name == "shard_001.hdf5":
            raise RuntimeError("convert boom")
        return {"episodes": 45}

    with pytest.raises(RuntimeError, match="convert boom"):
        gd.run_campaign(cfg, fake_batch, failing_convert)

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [r["run_index"] for r in ledger["runs"]] == [0]  # only the fully-converted run
    assert ledger["total_successes"] == 45
    # Run 2 was generated, but the failure surfaced before its convert was ever submitted, so
    # nothing of it can be in the dataset and it is not marked. Only run 1's outcome is unknown.
    assert [p.name for p in sorted(tmp_path.glob("run_*.started"))] == ["run_1.started"]

    def succeeding_convert(shard_path):
        return {"episodes": 45}

    # Run 1's convert RAISED, so whether it committed episodes before failing is unknown, and
    # its start marker is still on disk. Resuming over that would re-convert it (ruling D7-4),
    # so the campaign refuses until an operator settles it.
    with pytest.raises(gd.ResumeMismatch, match="run 1"):
        gd.run_campaign(cfg, fake_batch, succeeding_convert)

    # Once settled, run 1 must be regenerated, not skipped or resumed mid-convert -- it never
    # reached the ledger, so the schedule replays it.
    (tmp_path / "run_1.started").unlink()
    batch_calls.clear()

    total = gd.run_campaign(cfg, fake_batch, succeeding_convert)
    assert batch_calls[0] == 1  # regenerated from scratch, in schedule order
    assert total >= cfg.target
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [r["run_index"] for r in ledger["runs"]] == list(range(len(ledger["runs"])))


def test_run_campaign_batch_failure_still_finalizes_the_pending_convert(tmp_path):
    """Reviewer-reproduced regression: run_batch(2) raising must not silently drop run 1 --
    whose convert was already submitted (and, here, already finished successfully) by the time
    run_batch(2) fails -- from the ledger. Losing it would leave its episodes committed to the
    LeRobotDataset but untracked, so a resume would re-convert (and thus duplicate) them."""
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=200,
        attempts_per_run=50,
        num_envs=8,
        seed=0,
    )
    convert_calls = []

    def fake_batch(run_index, permutation, shard_path):
        if run_index == 2:
            raise RuntimeError("batch 2 boom")
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        convert_calls.append(shard_path.name)
        return {"episodes": 45}

    with pytest.raises(RuntimeError, match="batch 2 boom"):
        gd.run_campaign(cfg, fake_batch, fake_convert)

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    # Run 1's convert had already been submitted (and, in this fake, already completed) by the
    # time run_batch(2) raised -- it must be finalized, not dropped.
    assert [r["run_index"] for r in ledger["runs"]] == [0, 1]
    assert ledger["total_successes"] == 90
    assert convert_calls == ["shard_000.hdf5", "shard_001.hdf5"]
    assert all(not (tmp_path / r["shard"]).exists() for r in ledger["runs"])  # both deleted

    # Resume: run 2 (which never reached the ledger) is regenerated. shard_001 -- already
    # converted and deleted before the crash -- must NOT be re-converted (no duplication).
    convert_calls.clear()

    def resumed_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    total = gd.run_campaign(cfg, resumed_batch, fake_convert)
    assert "shard_001.hdf5" not in convert_calls
    assert total >= cfg.target


# --- Pre-campaign fix: isaaclab.sh subprocess bootstrap needs VIRTUAL_ENV -----------------


def test_isaaclab_batch_env_sets_virtual_env_when_absent(monkeypatch, tmp_path):
    """A bare non-interactive `python3 scripts/generate_dataset.py` (how an unattended
    overnight campaign is launched, e.g. via nohup) has no VIRTUAL_ENV of its own --
    isaaclab.sh's bootstrap needs one to find its bundled python, so the supervisor must set
    it itself, derived from --isaaclab_path."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    isaaclab_path = tmp_path / "IsaacLab"
    cfg = gd.CampaignCfg(
        out_dir=tmp_path / "out",
        dataset_root=tmp_path / "ds",
        isaaclab_path=isaaclab_path,
    )
    env = gd._isaaclab_batch_env(cfg)
    assert env["VIRTUAL_ENV"] == str(isaaclab_path / ".venv")
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_isaaclab_batch_env_does_not_clobber_an_existing_virtual_env(monkeypatch, tmp_path):
    """If the campaign is itself launched from inside an already-activated venv, that venv --
    not --isaaclab_path's -- must win; the supervisor only fills the gap, never overrides."""
    monkeypatch.setenv("VIRTUAL_ENV", "/caller/already/active/venv")
    cfg = gd.CampaignCfg(
        out_dir=tmp_path / "out",
        dataset_root=tmp_path / "ds",
        isaaclab_path=tmp_path / "IsaacLab",
    )
    env = gd._isaaclab_batch_env(cfg)
    assert env["VIRTUAL_ENV"] == "/caller/already/active/venv"


# --- Pre-campaign fix: derived seeds must stay inside numpy's 2**32 range ------------------


def test_derive_seeds_stay_within_numpy_range_for_the_campaign_seed():
    """numpy's np.random.seed (Isaac Lab's configure_seed, reached via env.reset(seed=...))
    requires 0 <= seed < 2**32. Any date-shaped campaign seed overflows the naive
    `seed * 1000 + run_index` formula -- MS1's own acceptance seed gives
    20260905 * 1000 = 20,260,905,000 > 2**32 - 1. That is not hypothetical: the same formula
    crashed run 0 of openyam-lab's campaign launch under --seed 20260807, and its earlier mini
    run used seed 42, far under the boundary, so it never exercised this path."""
    campaign_seed = 20260905
    cfg = gd.CampaignCfg(out_dir=Path("/tmp/x"), dataset_root=Path("/tmp/y"), seed=campaign_seed)
    for run_index in range(1000):
        run_seed, layout_seed = gd._derive_seeds(cfg, run_index)
        assert 0 <= run_seed < 2**32
        assert 0 <= layout_seed < 2**32


def test_derive_seeds_distinct_within_a_campaign_run_range():
    """The modulo can't collide within one campaign's run_index range (0..999): the pre-mod
    values for distinct run_index in that range differ from each other by at most 999 (<<
    2**32), so they can never wrap around to the same post-mod seed. Verified directly for
    MS1's acceptance-run seed rather than just argued."""
    campaign_seed = 20260905
    cfg = gd.CampaignCfg(out_dir=Path("/tmp/x"), dataset_root=Path("/tmp/y"), seed=campaign_seed)
    run_seeds = [gd._derive_seeds(cfg, i)[0] for i in range(1000)]
    layout_seeds = [gd._derive_seeds(cfg, i)[1] for i in range(1000)]
    assert len(set(run_seeds)) == 1000
    assert len(set(layout_seeds)) == 1000


def _cmd_cfg(tmp_path, **kw):
    return gd.CampaignCfg(out_dir=tmp_path / "work", dataset_root=tmp_path / "ds", target=5, **kw)


def test_batch_command_omits_the_distribution_flags_by_default(tmp_path):
    cmd = gd._batch_command(_cmd_cfg(tmp_path), 0, (0, 1, 2, 3), tmp_path / "shard_000")

    assert "--noise_scale" not in cmd
    assert "--start_pose_offset_rad" not in cmd


def test_batch_command_passes_the_widened_distribution_flags(tmp_path):
    cfg = _cmd_cfg(tmp_path, noise_scale=2.0, start_pose_offset_rad=0.3)

    cmd = gd._batch_command(cfg, 0, (0, 1, 2, 3), tmp_path / "shard_000")

    assert cmd[cmd.index("--noise_scale") + 1] == "2.0"
    assert cmd[cmd.index("--start_pose_offset_rad") + 1] == "0.3"


def test_batch_command_omits_induction_flags_by_default(tmp_path):
    cmd = gd._batch_command(_cmd_cfg(tmp_path), 0, (0, 1, 2, 3), tmp_path / "shard_000")
    assert "--induce_prob" not in cmd


def test_batch_command_passes_induction_flags(tmp_path):
    cfg = _cmd_cfg(
        tmp_path,
        induce_prob=0.12,
        induce_offset_weight=0.7,
        induce_offset_min=0.015,
        induce_offset_max=0.025,
        induce_noise_boost=5.0,
    )
    cmd = gd._batch_command(cfg, 0, (0, 1, 2, 3), tmp_path / "shard_000")
    assert cmd[cmd.index("--induce_prob") + 1] == "0.12"
    assert cmd[cmd.index("--induce_offset_weight") + 1] == "0.7"
    assert cmd[cmd.index("--induce_offset_min") + 1] == "0.015"
    assert cmd[cmd.index("--induce_offset_max") + 1] == "0.025"
    assert cmd[cmd.index("--induce_noise_boost") + 1] == "5.0"


def test_build_cfg_carries_the_distribution_flags(tmp_path):
    args = gd._parse_args(
        [
            "--out_dir",
            str(tmp_path / "w"),
            "--dataset_root",
            str(tmp_path / "d"),
            "--noise_scale",
            "2.0",
            "--start_pose_offset_rad",
            "0.3",
        ]
    )
    cfg = gd._build_cfg(args)
    assert cfg.noise_scale == 2.0 and cfg.start_pose_offset_rad == 0.3

    plain = gd._build_cfg(
        gd._parse_args(["--out_dir", str(tmp_path / "w"), "--dataset_root", str(tmp_path / "d")])
    )
    assert plain.noise_scale is None and plain.start_pose_offset_rad is None


def test_convert_command_forwards_split_flags(tmp_path):
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        split_cycles=True,
        sidecar_dir=tmp_path / "sidecars",
    )
    cmd = gd._convert_command(cfg, tmp_path / "shard.hdf5")
    assert "--split_cycles" in cmd
    assert "--sidecar_dir" in cmd and str(tmp_path / "sidecars") in cmd


def test_convert_command_default_omits_split_flags(tmp_path):
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds")
    cmd = gd._convert_command(cfg, tmp_path / "shard.hdf5")
    # Pins the branch's central promise: default (split_cycles=False) is byte-identical
    # to the plain conversion command line, not merely "the two new flags happen to be
    # absent".
    assert cmd == [
        "lerobot_stack/.venv/bin/python",
        "lerobot_stack/convert_to_lerobot.py",
        "--shards",
        str(tmp_path / "shard.hdf5"),
        "--root",
        str(tmp_path / "ds"),
    ]


def test_parse_args_rejects_sidecar_dir_without_split_cycles(tmp_path):
    with pytest.raises(SystemExit):
        gd._parse_args(
            [
                "--out_dir",
                str(tmp_path / "w"),
                "--dataset_root",
                str(tmp_path / "d"),
                "--sidecar_dir",
                str(tmp_path / "sidecars"),
            ]
        )


def test_campaign_target_default_matches_generate_batch_s_extrapolation_target(tmp_path):
    """The supervisor's default --target and generate_batch.py's CAMPAIGN_TARGET_SUCCESSES are
    the same campaign size seen from two sides: the supervisor stops at it, and a --probe run
    extrapolates its wall clock to it. They live in two files because this one is stdlib-only
    and generate_batch.py imports isaaclab at module scope, so the pin is by source text (the
    same technique the LeRobot-side schema pins use). openyam-lab's 600 came across with the
    file; MS1's campaign is an order of magnitude smaller, and a stale default here would
    silently run a 12x campaign for anyone who omitted --target."""
    batch_src = (Path(__file__).resolve().parents[1] / "scripts/generate_batch.py").read_text()
    match = re.search(r"^CAMPAIGN_TARGET_SUCCESSES = (\d+)", batch_src, re.MULTILINE)
    assert match is not None, "generate_batch.py no longer defines CAMPAIGN_TARGET_SUCCESSES"
    batch_target = int(match.group(1))

    assert gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds").target == batch_target
    args = gd._parse_args(["--out_dir", str(tmp_path), "--dataset_root", str(tmp_path / "ds")])
    assert args.target == batch_target


def test_main_emits_campaign_fail_when_the_campaign_dies_of_anything_else(
    monkeypatch, capsys, tmp_path
):
    """Every exit path emits its token. A supervisor that dies without printing CAMPAIGN_FAIL
    looks, to whoever is tailing supervisor.log or grepping for the token, exactly like one
    still running -- and this campaign is launched detached, on a host that sweeps background
    jobs, so "still running" is a live hypothesis rather than an absurd one. Only BatchFailed
    was tokenised before; a convert subprocess dying of an OSError, or the operator pressing
    Ctrl-C, left the log tokenless."""
    monkeypatch.setattr(gd, "run_campaign", _raise(RuntimeError("convert host went away")))

    with pytest.raises(RuntimeError, match="convert host went away"):
        gd.main(["--out_dir", str(tmp_path / "w"), "--dataset_root", str(tmp_path / "d")])

    assert "CAMPAIGN_FAIL" in capsys.readouterr().out


def test_main_emits_campaign_fail_on_keyboard_interrupt(monkeypatch, capsys, tmp_path):
    """KeyboardInterrupt is a BaseException, so `except Exception` does not catch it -- and
    Ctrl-C is the single most likely way an overnight campaign ends early."""
    monkeypatch.setattr(gd, "run_campaign", _raise(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        gd.main(["--out_dir", str(tmp_path / "w"), "--dataset_root", str(tmp_path / "d")])

    assert "CAMPAIGN_FAIL" in capsys.readouterr().out


# --- Ruling D7-1: --resume checks itself against the dataset -------------------------------


def _write_ledger(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps({"total_successes": sum(r["exported"] for r in records), "runs": records})
    )


def _ledger_record(run_index: int, exported: int, converted_total: int | None) -> dict:
    return {
        "run_index": run_index,
        "seeds": {"run_seed": run_index, "layout_seed": 500 + run_index},
        "permutation": [0, 1, 2, 3],
        "exported": exported,
        "attempts": 50,
        "shard": f"shard_{run_index:03d}.hdf5",
        "converted": converted_total is not None,
        "converted_total": converted_total,
        "deleted": True,
    }


def _write_dataset(root: Path, total_episodes: int) -> None:
    """The only part of a LeRobotDataset this module reads: meta/info.json's episode count."""
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "total_episodes": total_episodes})
    )


def _never_called(*_args, **_kwargs):
    raise AssertionError("the campaign must refuse before generating anything")


def test_the_ledger_records_the_converters_own_episode_total(tmp_path):
    """Resume can only check itself against the dataset if it knows what the dataset held when
    the last run was banked. The converter already reports it (`CONVERT_OK ... total=<n>`), and
    its own number is the one to keep: with --split_cycles a demo becomes several episodes, so
    summing `exported` would not equal the dataset's episode count."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        return {"episodes": 45, "total": 45}

    gd.run_campaign(cfg, fake_batch, fake_convert)

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [r["converted_total"] for r in ledger["runs"]] == [45]


def test_resume_refuses_when_the_dataset_holds_episodes_the_ledger_never_recorded(tmp_path):
    """The crash window this guards: the converter committed episodes and the supervisor was
    killed before Ledger.append recorded them. Resuming would regenerate that run index and
    convert it again, appending a second batch to the same dataset root -- and since a
    regenerated run is a fresh draw rather than a replay, the duplicates would not even be
    byte-identical to spot by eye."""
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    _write_dataset(tmp_path / "ds", total_episodes=90)
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)

    with pytest.raises(gd.ResumeMismatch, match="90"):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_resume_refuses_when_the_dataset_lost_episodes_the_ledger_recorded(tmp_path):
    """The mismatch is checked in both directions. A dataset that shrank -- moved, truncated,
    or pointed at the wrong --dataset_root -- is just as wrong as one that grew, and silently
    topping it up to the target would bank a campaign against the wrong data."""
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    _write_dataset(tmp_path / "ds", total_episodes=12)
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)

    with pytest.raises(gd.ResumeMismatch):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_resume_refuses_when_the_dataset_is_gone(tmp_path):
    """A ledger claiming converted runs against a dataset root that does not exist."""
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)

    with pytest.raises(gd.ResumeMismatch):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_resume_refuses_even_when_the_ledger_has_already_reached_the_target(tmp_path):
    """The early return on `total_successes >= target` must not short-circuit the check.
    Reporting a campaign complete over a dataset known to disagree with its own ledger is the
    worst of the outcomes here: nothing downstream would ever look again."""
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    _write_dataset(tmp_path / "ds", total_episodes=90)
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=10)

    with pytest.raises(gd.ResumeMismatch):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_resume_proceeds_when_the_dataset_agrees_with_the_ledger(tmp_path):
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    _write_dataset(tmp_path / "ds", total_episodes=45)
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=80)
    calls = []

    def fake_batch(run_index, permutation, shard_path):
        calls.append(run_index)
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    total = gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 90})

    assert calls == [1]  # resumed at the next index, nothing regenerated
    assert total == 90


def test_main_turns_a_resume_mismatch_into_campaign_fail(monkeypatch, capsys, tmp_path):
    """An operator-facing refusal, not a traceback: exit 1 with the token, like BatchFailed.
    subprocess.run is stubbed to fail loudly rather than merely being expected not to fire --
    without the check this test would otherwise pass for the wrong reason, by reaching a real
    isaaclab.sh that happens to exit non-zero."""
    _write_ledger(tmp_path / "ledger.json", [_ledger_record(0, 45, 45)])
    _write_dataset(tmp_path / "ds", total_episodes=90)
    monkeypatch.setattr(gd.subprocess, "run", _never_called)

    code = gd.main(["--out_dir", str(tmp_path), "--dataset_root", str(tmp_path / "ds"), "--resume"])

    assert code == 1
    assert "CAMPAIGN_FAIL" in capsys.readouterr().out


# --- Rulings D7-2 / D7-3: the child's deadline, and the supervisor's timeout over it --------


def test_batch_command_passes_the_childs_wall_clock_deadline(tmp_path):
    """--max_wall_minutes is passed explicitly on every run. generate_batch.py's own default is
    90, which is under the measured wall clock of a 50-attempt run at 12 envs -- leaving it
    unset would silently truncate the very run the acceptance gate is evaluated on."""
    cmd = gd._batch_command(_cmd_cfg(tmp_path), 0, (0, 1, 2, 3), tmp_path / "shard_000")

    assert cmd[cmd.index("--max_wall_minutes") + 1] == str(gd.BATCH_WALL_MINUTES)


def test_the_subprocess_timeout_is_derived_from_the_childs_own_deadline(monkeypatch, tmp_path):
    """A wedged child is the one failure mode that produces no token and no log line, so the
    supervisor cannot wait on it forever. The timeout is derived from the deadline the child
    was actually given rather than hard-coded, so the two cannot drift apart, and sits above it
    by the startup-plus-shard-flush grace that happens outside the child's own timed loop."""
    cfg = _cmd_cfg(tmp_path, max_wall_minutes=42.0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "BATCH_OK exported=1 attempts=1\n", "")

    monkeypatch.setattr(gd.subprocess, "run", fake_run)
    gd._make_real_run_batch(cfg)(0, (0, 1, 2, 3), cfg.out_dir / "shard_000")

    assert seen["timeout"] == (42.0 + gd.BATCH_TIMEOUT_GRACE_MINUTES) * 60.0
    assert seen["timeout"] > 42.0 * 60.0


def test_a_wedged_child_still_leaves_its_log_behind(monkeypatch, tmp_path):
    """The run whose log you most want is the one that hung. subprocess.run buffers the child's
    output and run_batch writes it only on return, so a timeout would otherwise discard exactly
    that output -- and a wedge leaves no BATCH_FAIL token to read instead."""
    cfg = _cmd_cfg(tmp_path)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], output="last line\n", stderr="!")

    monkeypatch.setattr(gd.subprocess, "run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        gd._make_real_run_batch(cfg)(3, (0, 1, 2, 3), cfg.out_dir / "shard_003")

    assert (cfg.out_dir / "run_3.log").read_text() == "last line\n!"


# --- Ruling D7-4: the run-0 convert-before-ledger window ------------------------------------


def test_a_run_marks_itself_started_before_the_batch_subprocess_runs(tmp_path):
    """The marker has to be on disk before anything the child could commit, not after the
    subprocess returns -- a marker written afterwards would miss exactly the kill it exists to
    detect. Asserted from inside the fake batch, which stands where the child would."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)
    seen = []

    def fake_batch(run_index, permutation, shard_path):
        seen.append((tmp_path / f"run_{run_index}.started").exists())
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 45})

    assert seen == [True]
    assert list(tmp_path.glob("run_*.started")) == []  # and cleared once banked


def test_the_marker_survives_a_failed_ledger_append(monkeypatch, tmp_path):
    """Clearing is tied to the ledger append succeeding, not to the subprocess exiting: the
    whole point is the window between those two. If the append is what fails, the run is
    exactly as un-banked as if the process had been killed there, and the marker must stay."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def boom(self, record):
        raise OSError("ledger write failed")

    monkeypatch.setattr(gd.Ledger, "append", boom)

    with pytest.raises(OSError, match="ledger write failed"):
        gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 45})

    assert (tmp_path / "run_0.started").exists()


def test_resume_refuses_when_a_run_started_and_never_reached_the_ledger(tmp_path):
    """The run-0 window: killed between its convert committing episodes and the first
    Ledger.append, there is no ledger entry to compare a dataset total against, so D7-1's check
    has nothing to work with. The marker is what makes that run visible at all."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_0.started").write_text(
        json.dumps({"run_index": 0, "phase": gd.MARKER_PHASE_CONVERTED})
    )

    with pytest.raises(gd.ResumeMismatch, match="run 0"):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_a_bare_reinvocation_after_an_interrupted_first_run_still_refuses(
    monkeypatch, capsys, tmp_path
):
    """Killed before the first append, ledger.json does not exist -- so main()'s
    `ledger_path.exists() and not --resume` refusal does not fire either, and a bare
    re-invocation would regenerate run 0's identical seeds and shard name over whatever the
    killed run had already committed. No --resume is passed here on purpose."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "run_0.started").write_text(
        json.dumps({"run_index": 0, "phase": gd.MARKER_PHASE_CONVERTED})
    )
    monkeypatch.setattr(gd.subprocess, "run", _never_called)

    code = gd.main(["--out_dir", str(tmp_path / "work"), "--dataset_root", str(tmp_path / "ds")])

    assert code == 1
    assert "CAMPAIGN_FAIL" in capsys.readouterr().out


def test_a_batch_failure_clears_its_own_marker(tmp_path):
    """A run whose batch never returned cannot have reached run_convert, so nothing of it can
    be in the dataset and there is nothing to protect. Leaving a marker there would make every
    routine BATCH_FAIL demand manual intervention before the next resume -- which is how a
    guard gets deleted reflexively instead of read."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)

    seen = []

    def failing_batch(run_index, permutation, shard_path):
        seen.append((tmp_path / f"run_{run_index}.started").exists())
        raise gd.BatchFailed("no BATCH_OK token found")

    with pytest.raises(gd.BatchFailed):
        gd.run_campaign(cfg, failing_batch, _never_called)

    assert seen == [True]  # it was marked started
    assert list(tmp_path.glob("run_*.started")) == []  # and unmarked again


# --- Ruling D7-5: the marker carries its phase ---------------------------------------------


def _marker(tmp_path: Path, run_index: int) -> dict:
    return json.loads((tmp_path / f"run_{run_index}.started").read_text())


def test_a_marker_reads_generating_while_its_batch_runs(tmp_path):
    """The phase a run is in decides whether resuming through it is safe, so it has to be on
    disk from the first write -- before anything the child could commit."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)
    seen = []

    def fake_batch(run_index, permutation, shard_path):
        seen.append(_marker(tmp_path, run_index)["phase"])
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 45})

    assert seen == [gd.MARKER_PHASE_GENERATING]


def test_a_marker_leaves_generating_before_its_convert_can_commit_anything(tmp_path):
    """The window that decides this: a run's convert is submitted to the background executor
    and then runs for the whole of the NEXT run's batch -- hours -- before finalize() collects
    it. If the marker only left `generating` once that result was collected, it would read
    `generating` throughout the entire period its convert was committing episodes, and a resume
    would auto-clear a run that had committed. The phase must turn unsafe before the submit,
    which is why there is a `converting` phase at all."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)
    seen = []

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    def fake_convert(shard_path):
        seen.append(_marker(tmp_path, 0)["phase"])
        return {"episodes": 45, "total": 45}

    gd.run_campaign(cfg, fake_batch, fake_convert)

    assert seen == [gd.MARKER_PHASE_CONVERTING]


def test_a_marker_reads_converted_at_the_moment_of_the_ledger_append(monkeypatch, tmp_path):
    """Brackets the transition on its other side: `converted` is written after the convert
    returns and before the append, so this fails if the write moves later (the append would see
    `converting`) just as the test above fails if it moves earlier."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)
    seen = []
    original_append = gd.Ledger.append

    def spy(self, record):
        seen.append(_marker(tmp_path, record["run_index"])["phase"])
        original_append(self, record)

    monkeypatch.setattr(gd.Ledger, "append", spy)

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 45})

    assert seen == [gd.MARKER_PHASE_CONVERTED]


def test_resume_regenerates_through_a_generating_marker_without_asking(capsys, tmp_path):
    """The benign case, and the likeliest one: swept during the ~2.3 h batch, before any
    convert was submitted. That run committed nothing, so refusing here would train an operator
    to clear markers unread -- which is how a guard stops working."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_0.started").write_text(
        json.dumps({"run_index": 0, "phase": gd.MARKER_PHASE_GENERATING})
    )
    calls = []

    def fake_batch(run_index, permutation, shard_path):
        calls.append(run_index)
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 45, "attempts": 50, "meta": {}, "stats": {}}

    gd.run_campaign(cfg, fake_batch, lambda p: {"episodes": 45, "total": 45})

    assert calls == [0]  # regenerated, not refused
    assert list(tmp_path.glob("run_*.started")) == []
    assert "run 0" in capsys.readouterr().out  # and said so rather than doing it silently


@pytest.mark.parametrize("phase", ["converting", "converted"])
def test_resume_still_refuses_a_marker_whose_convert_may_have_committed(phase, tmp_path):
    """Both unsafe phases keep refusing. This ruling narrows when the guard fires, never what
    it does when it does fire."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_0.started").write_text(json.dumps({"run_index": 0, "phase": phase}))

    with pytest.raises(gd.ResumeMismatch, match="run 0"):
        gd.run_campaign(cfg, _never_called, _never_called)


def test_resume_refuses_a_marker_it_cannot_read(tmp_path):
    """Fails closed. A marker truncated by the kill it is recording, or written by an older
    version, says nothing about whether that run converted -- and 'unreadable' must not be the
    one input that unlocks the auto-clear path."""
    cfg = gd.CampaignCfg(out_dir=tmp_path, dataset_root=tmp_path / "ds", target=200)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "run_0.started").write_text('{"run_index": 0, "phase": "gener')

    with pytest.raises(gd.ResumeMismatch, match="run 0"):
        gd.run_campaign(cfg, _never_called, _never_called)


# --- MS2: randomized pick order and the campaign circuit breaker ---------------------------


def test_batch_command_omits_randomize_pick_order_by_default(tmp_path):
    cmd = gd._batch_command(_cmd_cfg(tmp_path), 0, (0, 1, 2, 3), tmp_path / "shard_000")
    assert "--randomize_pick_order" not in cmd


def test_batch_command_forwards_randomize_pick_order(tmp_path):
    cfg = _cmd_cfg(tmp_path, randomize_pick_order=True)
    cmd = gd._batch_command(cfg, 0, (0, 1, 2, 3), tmp_path / "shard_000")
    assert "--randomize_pick_order" in cmd


def test_build_cfg_carries_randomize_pick_order_and_the_breaker(tmp_path):
    base = ["--out_dir", str(tmp_path / "w"), "--dataset_root", str(tmp_path / "d")]
    cfg = gd._build_cfg(gd._parse_args(base + ["--randomize_pick_order"]))
    assert cfg.randomize_pick_order is True
    assert cfg.breaker_min_rate == 0.70 and cfg.breaker_min_runs == 4

    plain = gd._build_cfg(gd._parse_args(base))
    assert plain.randomize_pick_order is False

    off = gd._build_cfg(gd._parse_args(base + ["--breaker_min_rate", "-1"]))
    assert off.breaker_min_rate is None

    custom = gd._build_cfg(
        gd._parse_args(base + ["--breaker_min_rate", "0.5", "--breaker_min_runs", "2"])
    )
    assert custom.breaker_min_rate == 0.5 and custom.breaker_min_runs == 2


def test_the_ledger_carries_each_runs_meta_and_stats(tmp_path):
    cfg = gd.CampaignCfg(
        out_dir=tmp_path, dataset_root=tmp_path / "ds", target=40, attempts_per_run=50, num_envs=8
    )
    meta = {"randomize_pick_order": True, "noise_scale": 1.0}
    stats = {"raise_vetoed": 7, "hover_lowered": 3, "off_schedule_episodes": 41}

    def fake_batch(run_index, permutation, shard_path):
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": 42, "attempts": 50, "meta": dict(meta), "stats": dict(stats)}

    gd.run_campaign(cfg, fake_batch, lambda _p: {"episodes": 42, "total": 42})
    (record,) = json.loads((tmp_path / "ledger.json").read_text())["runs"]
    assert record["run_meta"] == meta
    assert record["run_stats"] == stats


def _breaker_campaign(tmp_path, rates, min_rate=0.70, min_runs=4, target=10_000):
    """Drive a campaign whose run k exports ``rates[k] * 50``; returns (calls, ledger runs)."""
    cfg = gd.CampaignCfg(
        out_dir=tmp_path,
        dataset_root=tmp_path / "ds",
        target=target,
        attempts_per_run=50,
        num_envs=8,
        breaker_min_rate=min_rate,
        breaker_min_runs=min_runs,
    )
    calls = []

    def fake_batch(run_index, permutation, shard_path):
        calls.append(run_index)
        _hdf5_path(shard_path).write_bytes(b"x")
        return {"exported": round(rates[run_index] * 50), "attempts": 50}

    def fake_convert(_shard_path):
        return {"episodes": 1, "total": 1}

    try:
        gd.run_campaign(cfg, fake_batch, fake_convert)
    except gd.CircuitBreakerTripped as exc:
        return calls, json.loads((tmp_path / "ledger.json").read_text())["runs"], exc
    return calls, json.loads((tmp_path / "ledger.json").read_text())["runs"], None


def test_circuit_breaker_trips_after_four_runs_below_the_floor(tmp_path):
    """Four runs pooled at 0.60 < 0.70: no fifth run is launched, and every completed run,
    including the one whose convert was still pending, is in the ledger before the stop."""
    rates = [0.8, 0.6, 0.5, 0.5, 0.9, 0.9]
    calls, runs, exc = _breaker_campaign(tmp_path, rates)
    assert calls == [0, 1, 2, 3]
    assert [r["run_index"] for r in runs] == [0, 1, 2, 3]
    assert exc is not None and "0.600" in str(exc) and "4 runs" in str(exc)


def test_circuit_breaker_does_not_fire_before_the_minimum_run_count(tmp_path):
    """Three bad runs are not yet a verdict: the fourth must complete first."""
    rates = [0.2, 0.2, 0.2, 0.9, 0.9]
    calls, runs, exc = _breaker_campaign(tmp_path, rates)
    # Run 3 is launched with three bad runs behind it; the pool after it (0.375) trips.
    assert calls == [0, 1, 2, 3]
    assert exc is not None
    assert [r["run_index"] for r in runs] == [0, 1, 2, 3]


def test_circuit_breaker_stays_quiet_at_or_above_the_floor(tmp_path):
    rates = [0.7, 0.7, 0.7, 0.7, 0.7]
    calls, _runs, exc = _breaker_campaign(tmp_path, rates, target=5 * 35)
    assert exc is None
    assert calls == [0, 1, 2, 3, 4]


def test_circuit_breaker_can_be_disabled(tmp_path):
    rates = [0.1] * 6
    calls, _runs, exc = _breaker_campaign(tmp_path, rates, min_rate=None, target=6 * 5)
    assert exc is None
    assert calls == [0, 1, 2, 3, 4, 5]


def test_circuit_breaker_pools_the_resumed_ledger(tmp_path):
    """A resumed campaign reads the pool from the ledger, so a stopped campaign that is
    resumed without a fix stops again immediately, before any GPU time."""
    rates = [0.5, 0.5, 0.5, 0.5]
    calls, runs, exc = _breaker_campaign(tmp_path, rates)
    assert exc is not None and calls == [0, 1, 2, 3]
    _write_dataset(tmp_path / "ds", runs[-1]["converted_total"])  # what a resume checks
    calls2, _runs2, exc2 = _breaker_campaign(tmp_path, rates + [0.9])
    assert calls2 == []
    assert exc2 is not None


def test_main_turns_a_tripped_breaker_into_campaign_fail(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(gd, "_make_real_run_batch", lambda cfg: None)
    monkeypatch.setattr(gd, "_make_real_run_convert", lambda cfg: None)
    monkeypatch.setattr(gd, "run_campaign", _raise(gd.CircuitBreakerTripped("pooled 0.5")))
    rc = gd.main(["--out_dir", str(tmp_path / "w"), "--dataset_root", str(tmp_path / "d")])
    assert rc == 1
    out = capsys.readouterr()
    assert "CAMPAIGN_FAIL" in out.out and "pooled 0.5" in out.err

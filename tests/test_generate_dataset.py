import importlib.util
import itertools
import json
import re
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

    # Resume: convert now succeeds. Run 1 (whose convert raised) must be regenerated, not
    # skipped or resumed mid-convert -- it never reached the ledger, so the schedule replays it.
    batch_calls.clear()

    def succeeding_convert(shard_path):
        return {"episodes": 45}

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

"""CPU tests for scripts/fleet.py: the draw split, the stale payload refusal, the job status
parse and the token contract. Every host here is the local one, so nothing touches the network."""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "scripts/fleet.py"
_spec = importlib.util.spec_from_file_location("_fleet", FLEET)
fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleet)
_ge_spec = importlib.util.spec_from_file_location("_ge", ROOT / "scripts/run_gate_eval.py")
ge = importlib.util.module_from_spec(_ge_spec)
_ge_spec.loader.exec_module(ge)


@pytest.mark.parametrize(
    "n, caps, counts",
    [
        (24, [("c", 12), ("a", 8), ("b", 4)], [12, 8, 4]),
        (24, [("c", 12), ("a", 8), ("b", 6)], [11, 7, 6]),
        (12, [("c", 12), ("a", 8), ("b", 6)], [5, 4, 3]),
        (1, [("c", 12), ("a", 8)], [1, 0]),
    ],
)
def test_split_draws_runs_every_draw_once_in_proportion_to_the_lane_caps(n, caps, counts):
    draws = list(range(n))
    split = fleet.split_draws(draws, caps)
    assert [len(split[name]) for name, _ in caps] == counts
    assert [d for name, _ in caps for d in split[name]] == draws  # contiguous, each once
    assert fleet.split_draws(draws, caps) == split
    total = sum(cap for _, cap in caps)
    assert all(abs(len(split[name]) - n * cap / total) < 1 for name, cap in caps)


def test_lanes_deal_a_hosts_draws_round_robin_up_to_its_cap():
    assert fleet.lanes_of([0, 1, 2], 8) == [[0], [1], [2]]
    assert fleet.lanes_of(list(range(6)), 4) == [[0, 4], [1, 5], [2], [3]]


def test_draw_and_host_specs_parse():
    assert fleet.parse_draws("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    with pytest.raises(ValueError):
        fleet.parse_draws("0-3,2")
    assert fleet.parse_hosts("c,b:4") == [("c", None), ("b", 4)]


def test_a_payload_written_before_the_read_started_is_refused(tmp_path):
    start = time.time()
    for d in (0, 1):
        (tmp_path / f"eval_{d:03d}.json").write_text("{}")
    assert sorted(fleet.fresh_payloads(tmp_path, [0, 1], start - 1)) == [0, 1]
    os.utime(tmp_path / "eval_001.json", (start - 60, start - 60))  # a supervisor cache hit
    with pytest.raises(fleet.FleetError, match="eval_001.json"):
        fleet.fresh_payloads(tmp_path, [0, 1], start)
    with pytest.raises(fleet.FleetError, match=r"missing payloads for draws \[2\]"):
        fleet.fresh_payloads(tmp_path, [0, 2], start - 120)


def test_the_read_protocol_is_a_valid_supervisor_command_line(tmp_path):
    args = SimpleNamespace(
        ckpt="/ck/pretrained_model",
        threshold=0.85,
        seed=20261700,
        draws=[0, 1, 2, 3],
        split="heldout",
        n_steps=50,
        client_flags="--select_at_release --square_wrist",
    )
    argv = fleet.supervisor_args(args, "/l.json") + ["--num_envs", "4", "--out_dir", "/o"]
    cfg = ge._build_cfg(ge._parse_args(argv))
    assert (cfg.processes, cfg.heap_k, cfg.num_envs, cfg.n_action_steps) == (4, 50, 4, 50)
    assert cfg.client_flags == ("--select_at_release", "--square_wrist")


def test_status_reads_the_last_token_line_and_running_without_one():
    text = """== read-a-0
GATE_OK {}
JOB_EXIT rc=0
== crashed
JOB_EXIT rc=137
== train
== early
GATE_ERROR: eval process 3 exited 1
GATE_ERROR
"""
    assert fleet.parse_jobs(text) == {
        "read-a-0": "GATE_OK {}",
        "crashed": "JOB_EXIT rc=137",
        "train": "RUNNING",
        "early": "GATE_ERROR",
    }


def _campaign(tmp_path: Path) -> Path:
    host = {"ssh": None, "clone": str(tmp_path), "worktree": str(tmp_path), "setup": "true"}
    (tmp_path / "hosts.json").write_text(json.dumps({"here": {**host, "lanes": 2}}))
    return tmp_path


def _fleet(*argv: str) -> tuple[list[str], subprocess.CompletedProcess]:
    result = subprocess.run(
        [sys.executable, str(FLEET), *argv], capture_output=True, text=True, timeout=60, check=False
    )
    return [line for line in result.stdout.splitlines() if re.match(fleet.TOKEN_RE, line)], result


@pytest.mark.parametrize(
    "argv, token",
    [
        ([], "FLEET_FAIL"),
        (["sync"], "SYNC_FAIL"),  # argparse: --sha is required
        (["sync", "--sha", "abc"], "SYNC_FAIL"),  # argparse: not a full SHA
        (["read", "--tag", "t"], "READ_FAIL"),
        (["ship", "a"], "SHIP_FAIL"),
        (["status", "--campaign", "/nonexistent"], "STATUS_FAIL"),  # no hosts.json
        (["run", "nohost", "j", "--", "true"], "RUN_FAIL"),
    ],
)
def test_every_failure_prints_exactly_one_token(argv, token, tmp_path):
    tokens, result = _fleet(*argv)
    assert tokens == [token]
    assert result.returncode != 0


def test_run_and_status_on_the_local_host(tmp_path):
    campaign = _campaign(tmp_path)
    tokens, _ = _fleet("run", "here", "ok", "--campaign", str(campaign), "--", "echo", "hi")
    assert tokens[0].startswith("RUN_OK") and len(tokens) == 1
    log = (campaign / "jobs/ok.log").read_text().splitlines()
    assert log[0].startswith("sha ") and log[-2:] == ["hi", "JOB_EXIT rc=0"]
    tokens, result = _fleet("run", "here", "bad", "--campaign", str(campaign), "--", "exit 3")
    assert tokens[0].startswith("RUN_FAIL") and len(tokens) == 1 and result.returncode == 1
    (campaign / "jobs/live.log").write_text("sha x\nstep 10\n")
    tokens, result = _fleet("status", "--campaign", str(campaign))
    assert tokens == ["STATUS_OK"]
    assert "[status] here bad: JOB_EXIT rc=3" in result.stdout
    assert "[status] here live: RUNNING" in result.stdout
    assert "[status] here ok: JOB_EXIT rc=0" in result.stdout


def _read_args(**kw) -> SimpleNamespace:
    base = {
        "ckpt": "/ck/pretrained_model",
        "threshold": 0.85,
        "seed": 20261700,
        "draws": [0, 1, 2],
        "split": "all",
        "n_steps": 50,
        "client_flags": "--select_at_release --square_wrist",
        "tag": "t",
        "extra": ["--num_envs", "1"],
    }
    return SimpleNamespace(**{**base, **kw})


def test_an_act_read_keeps_its_supervisor_flags_and_a_pi05_read_adds_one():
    act = fleet.supervisor_args(_read_args(), "/l.json")
    assert act == fleet.PROTOCOL.split() + [
        "--checkpoint",
        "/ck/pretrained_model",
        "--ledger",
        "/l.json",
        "--threshold",
        "0.85",
        "--seed",
        "20261700",
        "--processes",
        "3",
        "--parts_split",
        "all",
        "--n_action_steps",
        "50",
        "--client_flags",
        "--select_at_release --square_wrist",
    ]
    assert fleet.supervisor_args(_read_args(pi05=False), "/l.json") == act
    pi05 = fleet.supervisor_args(_read_args(pi05=True), "/l.json")
    at = act.index("--checkpoint") + 2
    assert pi05 == act[:at] + ["--pi05"] + act[at:]  # right after the checkpoint
    cfg = ge._build_cfg(ge._parse_args(pi05 + ["--out_dir", "/o", "--lanes", "2"]))
    assert (cfg.pi05, cfg.lanes, cfg.n_action_steps) == (True, 2, 50)
    names = ("num_steps", "noise_scale", "jaw_noise_scale", "still_noise_scale", "still_rad")
    names += ("still_open",)
    unset = _read_args(pi05=True, **{f"pi05_{name}": None for name in names})
    assert fleet.supervisor_args(unset, "/l.json") == pi05


def test_a_pi05_read_forwards_its_sampling_switches_and_an_act_read_refuses_them():
    sampling = {"pi05_num_steps": 50, "pi05_noise_scale": 0.0, "pi05_still_noise_scale": 1.0}
    sampling |= {"pi05_still_rad": 0.1, "pi05_still_open": 0.8, "pi05_jaw_noise_scale": 0.25}
    pi05 = fleet.supervisor_args(_read_args(pi05=True, **sampling), "/l.json")
    at = pi05.index("--pi05") + 1
    assert pi05[at : at + 4] == ["--pi05_num_steps", "50", "--pi05_noise_scale", "0.0"]
    assert pi05[at + 4 : at + 6] == ["--pi05_jaw_noise_scale", "0.25"]  # then the still rule
    assert pi05[at + 6 : at + 12] == [
        "--pi05_still_noise_scale",
        "1.0",
        "--pi05_still_rad",
        "0.1",
        "--pi05_still_open",
        "0.8",
    ]
    cfg = ge._build_cfg(ge._parse_args(pi05 + ["--out_dir", "/o"]))
    assert (cfg.pi05_num_steps, cfg.pi05_noise_scale, cfg.pi05_jaw_noise_scale) == (50, 0.0, 0.25)
    assert (cfg.pi05_still_noise_scale, cfg.pi05_still_rad, cfg.pi05_still_open) == (1.0, 0.1, 0.8)
    for name, value in sampling.items():
        act = fleet.supervisor_args(_read_args(**{name: value}), "/l.json")
        with pytest.raises(ValueError, match="need --pi05"):
            ge._build_cfg(ge._parse_args(act + ["--out_dir", "/o"]))


def test_a_pi05_read_runs_one_supervisor_a_host_and_an_act_read_one_a_lane():
    host = {"device": "cuda"}
    lane = ["B", "--device", "cuda", "--num_envs", "1", "--out_dir", "/o", "--port"]
    act = fleet.host_jobs(_read_args(), ["B"], host, "desk", [0, 1, 2], 2, 7100, "/o")
    assert act == [
        ("read-t-desk-0", lane + ["7100", "--only_processes", "0,2"]),
        ("read-t-desk-1", lane + ["7101", "--only_processes", "1"]),
    ]
    pi05 = fleet.host_jobs(_read_args(pi05=True), ["B"], host, "desk", [0, 1, 2], 2, 7100, "/o")
    assert pi05 == [("read-t-desk-0", lane + ["7100", "--only_processes", "0,1,2", "--lanes", "2"])]
    one = fleet.host_jobs(_read_args(pi05=True), ["B"], host, "desk", [4], 10, 7100, "/o")
    assert one[0][1][-2:] == ["--lanes", "1"]
    for pi05 in (False, True):  # a host with no draw runs nothing
        assert fleet.host_jobs(_read_args(pi05=pi05), ["B"], host, "desk", [], 2, 7100, "/o") == []

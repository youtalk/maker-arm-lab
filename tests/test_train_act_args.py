"""CPU tests for lerobot_stack/train_act.sh's argument handling (MS2 spec section 4).

The script is copied into a temp dir with a fake `.venv/bin/lerobot-train` that records
its argv, so the guards and the flags are exercised without lerobot or a GPU.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "lerobot_stack" / "train_act.sh"


@pytest.fixture()
def rig(tmp_path):
    stack = tmp_path / "lerobot_stack"
    (stack / ".venv" / "bin").mkdir(parents=True)
    shutil.copy(SCRIPT, stack / "train_act.sh")
    fake = stack / ".venv" / "bin" / "lerobot-train"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_ARGV"\n')
    fake.chmod(0o755)
    dataset = tmp_path / "dataset" / "meta"
    dataset.mkdir(parents=True)
    (dataset / "info.json").write_text(json.dumps({"total_episodes": 1000}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"target": 1000, "total_successes": 1000, "runs": []}))
    return {
        "script": stack / "train_act.sh",
        "dataset": tmp_path / "dataset",
        "ledger": ledger,
        "argv": tmp_path / "argv.txt",
        "out": tmp_path / "out",
    }


def _run(rig, **env):
    full = {
        **os.environ,
        "FAKE_ARGV": str(rig["argv"]),
        "WANDB_ENABLE": "false",
        "STEPS": "600000",
        "SAVE_FREQ": "50000",
    }
    full.update({k: str(v) for k, v in env.items()})
    for key in ("DATASET_ROOT", "OUT", "LEDGER", "MIN_EPISODES"):
        if key in env and env[key] is None:
            full.pop(key, None)
    return subprocess.run(
        ["bash", str(rig["script"])], env=full, capture_output=True, text=True, check=False
    )


def _full_env(rig):
    return {"DATASET_ROOT": rig["dataset"], "OUT": rig["out"], "LEDGER": rig["ledger"]}


def test_a_complete_corpus_trains_with_the_flags_forwarded(rig):
    result = _run(rig, **_full_env(rig))
    assert result.returncode == 0, result.stderr
    assert "TRAIN_OK" in result.stdout
    argv = rig["argv"].read_text().splitlines()
    assert f"--dataset.root={rig['dataset']}" in argv
    assert f"--output_dir={rig['out']}" in argv
    assert "--steps=600000" in argv
    assert "--save_freq=50000" in argv
    assert "--policy.type=act" in argv


@pytest.mark.parametrize("missing", ["DATASET_ROOT", "OUT"])
def test_dataset_root_and_out_have_no_defaults(rig, missing):
    env = _full_env(rig)
    env[missing] = None
    result = _run(rig, **env)
    assert result.returncode == 1
    assert "TRAIN_FAIL" in result.stdout
    assert "DATASET_ROOT and OUT are required" in result.stderr
    assert not rig["argv"].exists()


def test_the_ledger_is_required(rig):
    env = _full_env(rig)
    env["LEDGER"] = None
    result = _run(rig, **env)
    assert result.returncode == 1
    assert "LEDGER=" in result.stderr and "TRAIN_FAIL" in result.stdout
    assert not rig["argv"].exists()


def test_a_partial_corpus_is_refused(rig):
    (rig["dataset"] / "meta" / "info.json").write_text(json.dumps({"total_episodes": 999}))
    result = _run(rig, **_full_env(rig))
    assert result.returncode == 1
    assert "999 episodes, below the campaign target 1000" in result.stderr
    assert "TRAIN_FAIL" in result.stdout
    assert not rig["argv"].exists()


def test_a_ledger_without_a_target_needs_an_explicit_minimum(rig):
    rig["ledger"].write_text(json.dumps({"total_successes": 63, "runs": []}))
    refused = _run(rig, **_full_env(rig))
    assert refused.returncode == 1 and "records no target" in refused.stderr

    (rig["dataset"] / "meta" / "info.json").write_text(json.dumps({"total_episodes": 63}))
    ok = _run(rig, MIN_EPISODES=63, **_full_env(rig))
    assert ok.returncode == 0, ok.stderr
    assert "TRAIN_OK" in ok.stdout

    short = _run(rig, MIN_EPISODES=64, **_full_env(rig))
    assert short.returncode == 1 and "below the campaign target 64" in short.stderr

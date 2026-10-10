"""colab_train_pi05.sh and pi05_preflight.py (MS7h recipe 5.6 and 9).

The preflight checks run on stub policies and pipelines. The script runs in a temp dir that
stands in for /content, with shims first on PATH: python3 (the HF checks pass, the trainer is
the stand-in, and the pi05_base download copies FAKE_BASE), nvidia-smi, pip, df (FAKE_DF_KB) and
cp (it logs the Drive checkpoints at each copy to Drive, and fails the first FAKE_DIR/cp_fails
copies). The stand-in trainer is this file run as a script: it parses the trainer's argv with
lerobot's own parser, writes lerobot's checkpoint layout and log lines, and fakes the preflight's
outcome (FAKE_PREFLIGHT).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pi05_preflight
from pi05_preflight import PreflightError, install, patch_max_length, stats_sha

HERE = Path(__file__).resolve().parent
FULL = "All keys loaded successfully!"
STATS = {
    "action": {"mean": [0.1] * 7, "q01": [-1.0] * 7, "q99": [1.0] * 7},
    "observation.state": {"mean": [0.2] * 23, "q01": [-2.0] * 23, "q99": [2.0] * 23},
    "observation.images.wrist": {"mean": [[[0.5]]] * 3},
}


def _preprocessor_json(max_length: int) -> dict:
    tokenizer = {"max_length": max_length, "padding": "max_length", "tokenizer_name": "g/p"}
    return {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "normalizer_processor", "config": {}},
            {"registry_name": "tokenizer_processor", "config": tokenizer},
        ],
    }


def _max_length(model_dir: Path) -> int:
    steps = json.loads((model_dir / "policy_preprocessor.json").read_text())["steps"]
    return next(
        s["config"]["max_length"] for s in steps if s["registry_name"] == "tokenizer_processor"
    )


# --- pi05_preflight.py on stubs ---


def test_patch_replaces_the_file_and_keeps_the_start(tmp_path):
    start, copy = tmp_path / "start", tmp_path / "copy"
    start.mkdir()
    copy.mkdir()
    (start / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(200)))
    (copy / "policy_preprocessor.json").symlink_to(start / "policy_preprocessor.json")
    assert patch_max_length(str(copy), 128) == 200
    assert _max_length(copy) == 128 and not (copy / "policy_preprocessor.json").is_symlink()
    assert _max_length(start) == 200


def test_stats_sha_reads_the_action_and_state_as_float32():
    flat = {f"{k}.{n}": v for k, sub in STATS.items() for n, v in sub.items()}
    base = stats_sha(flat)
    assert stats_sha({k: np.asarray(v, dtype=np.float64) for k, v in flat.items()}) == base
    assert stats_sha({**flat, "observation.images.wrist.mean": [[[0.4]]] * 3}) == base
    assert stats_sha({**flat, "action.mean": [0.1] * 6 + [0.2]}) != base
    assert stats_sha({k: v for k, v in flat.items() if k != "observation.state.q99"}) != base


class NormalizerProcessorStep:
    def __init__(self, flat):
        self.flat = flat

    def state_dict(self):
        return self.flat


class UnnormalizerProcessorStep(NormalizerProcessorStep):
    pass


class TokenizerProcessorStep:
    def __init__(self, max_length):
        self.max_length = max_length


def _preflight_rig(**bad):
    """(policy class, train module, batch) after install(), with one defect from ``bad``."""
    flat = {
        f"{k}.{n}": np.asarray(v, np.float32) for k, sub in STATS.items() for n, v in sub.items()
    }
    norm = {**flat, "action.mean": np.zeros(7, np.float32)} if "stats" in bad else flat
    unnorm = {**flat, "action.q99": np.zeros(7, np.float32)} if "unstats" in bad else flat

    class Policy:
        config = SimpleNamespace(chunk_size=50)

        @classmethod
        def from_pretrained(cls, **kwargs):
            print("Loading model from: x")
            print("Missing keys when loading state dict: 3 keys" if "partial" in bad else FULL)
            return cls()

        def forward(self, batch, reduction="mean"):
            return "stock"

    pre = SimpleNamespace(
        steps=[NormalizerProcessorStep(norm), TokenizerProcessorStep(bad.get("tokens", 128))]
    )
    post = SimpleNamespace(steps=[UnnormalizerProcessorStep(unnorm)])
    module = SimpleNamespace(make_pre_post_processors=lambda *a, **k: (pre, post))
    install(module, Policy, flat, 128, bad.get("jaw"))
    if "stock" not in bad:  # train_pi05.py swaps the forward before lerobot builds anything
        Policy.forward = lambda self, batch, reduction="mean": "masked"
    import torch

    batch = {"action": torch.zeros(4, 50, 7), "index": torch.arange(4)}
    if "nopad" not in bad:
        batch["action_is_pad"] = torch.zeros(4, 50, dtype=torch.bool)
    if "noindex" in bad:
        del batch["index"]
    return Policy, module, batch


def _train_once(policy_cls, module, batch):
    policy = policy_cls.from_pretrained(pretrained_name_or_path="x")
    module.make_pre_post_processors()
    return policy_cls.forward(policy, batch)


def test_a_full_load_with_the_corpus_stats_reaches_the_masked_forward(capsys):
    assert _train_once(*_preflight_rig()) == "masked"
    out = capsys.readouterr().out
    assert FULL in out and "PREFLIGHT_OK full_load=1" in out and "max_length=128" in out
    assert "action_is_pad=(4, 50) pad_share=0.0000" in out
    assert _train_once(*_preflight_rig(jaw="m.npy", tokens=128)) == "masked"


@pytest.mark.parametrize(
    "bad, why",
    [
        ({"partial": 1}, "did not load in full"),
        ({"stats": 1}, "normalizer stats sha"),
        ({"unstats": 1}, "unnormalizer stats sha"),
        ({"tokens": 200}, "the tokenizer pads to 200, want 128"),
        ({"stock": 1}, "stock forward"),
        ({"nopad": 1}, "action_is_pad None at the forward"),
        ({"jaw": "m.npy", "noindex": 1}, "index None at the forward"),
    ],
)
def test_each_preflight_miss_raises_before_the_first_update(bad, why):
    with pytest.raises(PreflightError, match=why):
        _train_once(*_preflight_rig(**bad))


def test_the_preflight_tokens(tmp_path, capsys):
    assert pi05_preflight.main([]) == 2
    assert "TRAIN_FAIL usage" in capsys.readouterr().out
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(200)))
    assert pi05_preflight.main(["patch", str(tmp_path), "128"]) == 0
    assert f"PATCH_OK {tmp_path} max_length 200 -> 128" in capsys.readouterr().out
    np.save(tmp_path / "m.npy", np.arange(3, dtype=np.int32))
    assert pi05_preflight.main(["train", f"--jaw_mask={tmp_path / 'm.npy'}"]) == 3
    assert "PREFLIGHT_FAIL --jaw_mask" in capsys.readouterr().out


# --- colab_train_pi05.sh dry runs with the stand-in trainer ---

SHIMS = {
    "python3": """
case "$1 $2" in *pi05_preflight.py\\ train) shift 2; exec "$FAKE_PY" "$FAKE_STANDIN" "$@" ;; esac
case "$*" in *snapshot_download*)
    echo "$3 $4" >> "$FAKE_DIR/download.log"; mkdir -p "$5"; exec cp -r "$FAKE_BASE"/. "$5" ;;
esac
case "$*" in *huggingface_hub*|*AutoTokenizer*|*"import lerobot"*) exit 0 ;; esac
exec "$FAKE_PY" "$@"
""",
    "nvidia-smi": """
case "$*" in *nounits*) echo 81920 ;; *) echo "NVIDIA A100-SXM4-80GB, 81920 MiB" ;; esac
""",
    "pip": "exit 0\n",
    "df": """
[ -n "${FAKE_DF_KB:-}" ] || PATH=$REAL_PATH exec df "$@"
echo "Filesystem 1024-blocks Used Available Capacity Mounted"
echo "drive 999999999 1 $FAKE_DF_KB 1% /drive"
""",
    "cp": """
dest=${@: -1}
case "$dest" in */.tmp_*)
    n=$(cat "$FAKE_DIR/cp_fails" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$FAKE_DIR/cp_fails"; exit 1; fi
    held=$(cd "$(dirname "$dest")" && ls -d [0-9]* 2>/dev/null | tr '\\n' ' ')
    echo "$(basename "$dest") held: $held" >> "$FAKE_DIR/cp.log" ;;
esac
PATH=$REAL_PATH exec cp "$@"
""",
}


@pytest.fixture()
def rig(tmp_path):
    content = tmp_path / "content"
    drive = content / "drive" / "MyDrive" / "ms7h"
    meta = drive / "dataset" / "meta"
    start = drive / "checkpoints" / "030000" / "pretrained_model"
    base = tmp_path / "pi05_base"  # what the snapshot_download of lerobot/pi05_base writes
    for d in (meta, start, base, tmp_path / "bin", tmp_path / "fake"):
        d.mkdir(parents=True)
    for name in ("colab_train_pi05.sh", "pi05_preflight.py", "train_pi05.py"):
        shutil.copy(HERE / name, drive / name)
    (meta / "info.json").write_text("{}")
    (meta / "stats.json").write_text(json.dumps(STATS))
    (meta / "action_space.json").write_text('{"action_space": "absolute", "gripper": "continuous"}')
    for d in (start, base):
        (d / "model.safetensors").write_bytes(b"\0" * 1000)
        (d / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(200)))
    np.save(drive / "jaw.npy", np.arange(5, dtype=np.int64))
    (drive / "jaw.json").write_text("{}")  # the sidecar train_pi05.py checks
    for name, body in SHIMS.items():
        (tmp_path / "bin" / name).write_text("#!/usr/bin/env bash\n" + body)
        (tmp_path / "bin" / name).chmod(0o755)
    (tmp_path / "uptime").write_text("3600.00 1000.00\n")
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "REAL_PATH": os.environ["PATH"],
        "FAKE_PY": sys.executable,
        "FAKE_STANDIN": __file__,
        "FAKE_DIR": str(tmp_path / "fake"),
        "FAKE_BASE": str(base),
        "CONTENT": str(content),
        "UPTIME_FILE": str(tmp_path / "uptime"),
        "DIR": "ms7h",
        "START": "checkpoints/030000/pretrained_model",
        "RUN_NAME": "checkpoints_ab",
        "JAW_MASK": "jaw.npy",
        "STEPS": "300",
        "SAVE_FREQ": "100",
        "NUM_WORKERS": "0",
        "HF_TOKEN": "fake",
        "STATS_MD5": hashlib.md5((meta / "stats.json").read_bytes()).hexdigest(),
        "FAKE_DF_KB": "200000000",  # 204.8 GB free on the fake Drive
        "POLL_S": "0.2",
        "SYNC_RETRY_S": "5",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    return SimpleNamespace(
        path=tmp_path,
        drive=drive,
        run=drive / "checkpoints_ab",
        out=content / "pi05_checkpoints_ab",
        start=start,
        fake=tmp_path / "fake",
        env=env,
    )


def _run(rig, *args, **env):
    full = {**rig.env, **{k: str(v) for k, v in env.items() if v is not None}}
    for key in [k for k, v in env.items() if v is None]:
        full.pop(key, None)
    script = rig.drive / "colab_train_pi05.sh"
    out = subprocess.run(
        ["bash", str(script), *args],
        env=full,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    lines = out.stdout.splitlines()
    tokens = [line for line in lines if line.startswith("COLAB_TRAIN_")]
    assert len(tokens) == 1 and lines[-1] == tokens[0], out.stdout + out.stderr
    return SimpleNamespace(token=tokens[0], stdout=out.stdout, stderr=out.stderr, rc=out.returncode)


def _calls(rig) -> list:
    path = rig.fake / "cfg.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _held(rig) -> list:
    return sorted(p.name for p in rig.run.iterdir()) if rig.run.exists() else []


def test_smoke_trains_200_steps_from_a_128_token_copy_and_writes_nothing_to_drive(rig):
    r = _run(rig, "--smoke", RUN_NAME=None)
    assert r.token.startswith("COLAB_TRAIN_OK smoke sec_per_step=3.850 checkpoint_gib=")
    assert "loss200=0.032 batch=32 expert_only=false" in r.token
    assert "loss band: step 200 loss 0.032 in [0.015, 0.06]" in r.stdout
    (call,) = _calls(rig)
    assert call["steps"] == 200 and call["warmup"] == 6 and call["decay"] == 200
    assert call["max_length"] == 128 and _max_length(rig.start) == 200
    assert call["jaw_mask"] == str(rig.drive / "jaw.npy")
    assert not rig.run.exists() and not (rig.drive / "action_space.json").exists()


BASE = "lerobot/pi05_base"
BASE_REV = "a538eb273274eb30f126a118f39dbc0ee212c883"


def test_a_smoke_from_pi05_base_downloads_the_pinned_revision_and_reads_its_own_band(rig):
    r = _run(rig, "--smoke", START=BASE, RUN_NAME=None, JAW_MASK=None, FAKE_LOSS="0.092")
    assert r.token.startswith("COLAB_TRAIN_OK smoke") and "loss200=0.092" in r.token
    assert "loss band: step 200 loss 0.092 in [0.046, 0.34]" in r.stdout
    assert (rig.fake / "download.log").read_text() == f"{BASE} {BASE_REV}\n"
    (call,) = _calls(rig)
    assert call["pretrained_path"] == str(rig.path / "content" / "start_model")
    assert call["max_length"] == 128 and call["jaw_mask"] is None
    assert call["warmup"] == 6 and call["decay"] == 200
    assert _max_length(rig.path / "pi05_base") == 200
    r = _run(rig, "--smoke", START=BASE, RUN_NAME=None, JAW_MASK=None)  # a warm start's loss
    assert "loss band: step 200 loss 0.032 outside [0.046, 0.34]" in r.token


def test_a_span_run_on_a_shared_drive_keeps_its_reads_and_resumes_toward_the_same_steps(rig):
    # the owner's 2026-10-09 run in small: DRIVE_ROOT on a Shared drive, pi05_base, SPAN=1,
    # KEEP_EVERY, DRIVE_MIN_GB=0 (df reads the account's quota, here 51 GB)
    shared = rig.path / "content" / "drive" / "Shareddrives" / "lab" / "ms7h"
    shared.parent.mkdir(parents=True)
    shutil.move(rig.drive, shared)
    rig.drive.parent.rmdir()  # no MyDrive at all
    rig.drive, rig.run = shared, shared / "checkpoints_ab"
    env = {"START": BASE, "JAW_MASK": None, "STEPS": 600, "SAVE_FREQ": 100, "SPAN": 1}
    env |= {"KEEP_EVERY": 200, "DRIVE_ROOT": shared.parent, "DRIVE_MIN_GB": 0}
    env |= {"FAKE_DF_KB": 50_000_000, "FAKE_LOSS": "0.120", "FAKE_STEP_S": 0.02}
    (rig.path / "uptime").write_text("83600.00 1.00\n")  # N = 1360 / 3.85 = 353: stop at 300
    r = _run(rig, **env)
    assert "SPAN=1: this session stops at checkpoint 300 of 600" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=600 last=000300 session_end")
    assert (rig.run / "last.txt").read_text().strip() == "000300"
    assert (rig.run / "000300" / "training_state").is_dir()  # resumable
    assert sorted(p.name for p in (rig.run / "keep").iterdir()) == ["000200"]
    time.sleep(2.5)  # 125 more steps had the trainer lived on
    assert not (rig.out / "checkpoints" / "000400").exists()

    shutil.rmtree(rig.path / "content" / "dataset")  # a new VM: the local disk is gone
    shutil.rmtree(rig.out)
    (rig.path / "uptime").write_text("3600.00 1.00\n")
    r = _run(rig, **env)
    assert "resuming checkpoints_ab from checkpoint 000300 toward step 600" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=600 last=000600")
    first, second = _calls(rig)
    assert first["pretrained_path"] == str(rig.path / "content" / "start_model")
    assert first["max_length"] == 128 and first["jaw_mask"] is None and not first["resume"]
    assert second["resume"] is True and second["start"] == 300
    for call in (first, second):  # one schedule: warmup 1000 * 600 / 30000, decay STEPS
        assert (call["steps"], call["warmup"], call["decay"]) == (600, 20, 600)
    assert _held(rig) == ["000600", "keep", "last.txt"]
    assert not (rig.run / "000600" / "training_state").exists()
    keep = rig.run / "keep"
    assert sorted(p.name for p in keep.iterdir()) == ["000200", "000400", "000600"]
    for step in keep.iterdir():
        assert [p.name for p in step.iterdir()] == ["pretrained_model"]
        assert (step / "pretrained_model" / "model.safetensors").is_file()
    assert (shared / "action_space.json").exists()
    assert not (rig.path / "content" / "drive" / "MyDrive").exists()

    r = _run(rig, **env)
    assert r.token.startswith("COLAB_TRAIN_DONE run=checkpoints_ab last=000600")
    assert len(_calls(rig)) == 2 and len(list(keep.iterdir())) == 3


def test_a_fresh_run_passes_every_recipe_flag_and_keeps_one_checkpoint(rig):
    # 23.28 h of uptime: N = (86400 - 83800 - 1440) / 3.85 = 301 steps
    (rig.path / "uptime").write_text("83800.00 1.00\n")
    r = _run(rig, STEPS=None)
    assert "(24 h - uptime 23.28 h - 0.4 h) / 3.85 s) = 301 steps" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=301 last=000301")
    (call,) = _calls(rig)
    want = {
        "resume": False,
        "steps": 301,
        "warmup": 10,  # 1000 * 301 / 30000, as lerobot scaled M2k's 2000 steps to 66
        "decay": 301,
        "peak_lr": 2.5e-5,
        "decay_lr": 2.5e-6,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "train_expert_only": False,
        "chunk_size": 50,
        "n_action_steps": 50,
        "tokenizer_max_length": 128,
        "max_length": 128,
        "batch_size": 32,
        "seed": 1000,
        "jaw_mask": str(rig.run / "jaw_mask.npy"),
    }
    assert {k: call[k] for k in want} == want
    assert call["pretrained_path"] == str(rig.path / "content" / "start_model")
    assert _held(rig) == ["000301", "jaw_mask.json", "jaw_mask.npy", "last.txt"]
    assert sorted(p.name for p in (rig.run / "000301").iterdir()) == ["pretrained_model"]
    held = (rig.fake / "cp.log").read_text().splitlines()
    assert any(not line.endswith("held: ") for line in held)  # room: copy, then delete
    assert (rig.drive / "action_space.json").exists()


def test_a_resumed_session_trains_masked_with_the_run_s_jaw_mask(rig):
    r = _run(rig, FAKE_DIE_AT=250)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and "the trainer exited 137" in r.token
    assert (rig.run / "last.txt").read_text().strip() == "000200"
    assert (rig.run / "000200" / "training_state").is_dir()
    shutil.rmtree(rig.path / "content" / "dataset")  # a new VM: the local disk is gone
    shutil.rmtree(rig.out)
    r = _run(rig, JAW_MASK=None, STEPS=None)
    assert "resuming checkpoints_ab from checkpoint 000200 toward step 300" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=300 last=000300")
    first, second = _calls(rig)
    assert second["resume"] is True and second["start"] == 200 and second["steps"] == 300
    assert second["output_dir"] == str(rig.out) and second["max_length"] == 128
    assert second["jaw_mask"] == str(rig.run / "jaw_mask.npy")
    assert second["dataset_root"] == str(rig.path / "content" / "dataset")
    assert second["warmup"] == first["warmup"] == 10 and second["decay"] == 300
    assert _held(rig) == ["000300", "jaw_mask.json", "jaw_mask.npy", "last.txt"]
    assert not (rig.run / "000300" / "training_state").exists()


def test_a_run_that_reached_its_last_step_does_not_resume(rig):
    _run(rig, STEPS=200)
    (rig.run / "000200" / "training_state").mkdir()  # a final delete that never ran
    r = _run(rig, STEPS=None)
    assert r.token.startswith("COLAB_TRAIN_DONE run=checkpoints_ab last=000200")
    assert len(_calls(rig)) == 1 and not (rig.run / "000200" / "training_state").exists()


@pytest.mark.parametrize(
    "preflight, why",
    [
        ("fail", "PREFLIGHT_FAIL the weights did not load in full"),
        ("silent", "the trainer logged a step with no PREFLIGHT_OK line"),
    ],
)
def test_a_preflight_miss_stops_the_run(rig, preflight, why):
    r = _run(rig, FAKE_PREFLIGHT=preflight, STEPS=600, FAKE_STEP_S=0.01)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and why in r.token
    assert not (rig.out / "checkpoints" / "000600").exists()


@pytest.mark.parametrize("loss", ["0.090", "0.010"])
def test_a_loss_outside_the_band_at_step_200_stops_the_run(rig, loss):
    r = _run(rig, FAKE_LOSS=loss, STEPS=600)
    assert f"loss band: step 200 loss {loss} outside [0.015, 0.06]" in r.token
    assert not (rig.out / "checkpoints" / "000600").exists()


def test_a_drive_under_the_free_floor_stops_before_any_copy(rig):
    r = _run(rig, FAKE_DF_KB=58_000_000)  # 59.4 GB
    assert "Drive has 59 GB free, a run needs 60" in r.token
    assert not _calls(rig) and not (rig.path / "content" / "dataset").exists()


def test_a_short_drive_never_holds_two_checkpoints(rig):
    r = _run(rig, FAKE_DF_KB=500_000, DRIVE_MIN_GB=0)
    assert r.token.startswith("COLAB_TRAIN_OK") and "Drive is short of room" in r.stdout
    held = (rig.fake / "cp.log").read_text().splitlines()
    assert len(held) >= 2 and all(line.endswith("held: ") for line in held), held
    assert _held(rig) == ["000300", "jaw_mask.json", "jaw_mask.npy", "last.txt"]


def test_the_final_sync_retries_then_fails_after_its_window(rig):
    (rig.fake / "cp_fails").write_text("3")
    r = _run(rig, SAVE_FREQ=1000)
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=300 last=000300")
    (rig.fake / "cp_fails").write_text("1000")
    shutil.rmtree(rig.run)  # a new run on a new VM
    shutil.rmtree(rig.out)
    start = time.monotonic()
    r = _run(rig, SAVE_FREQ=1000, SYNC_RETRY_S=1)
    assert "the final Drive sync failed for 1 s" in r.token and time.monotonic() - start < 60
    assert (rig.out / "checkpoints" / "000300").is_dir()


@pytest.mark.parametrize(
    "args, env, why",
    [
        (["--bogus"], {}, "unknown argument --bogus"),
        ([], {"DIR": None}, "set DIR="),
        ([], {"RUN_NAME": None}, "set RUN_NAME="),
        ([], {"START": None}, "set START="),
        ([], {"STATS_MD5": "0" * 32}, "is not the corpus's 00000000"),
        ([], {"STEPS": 100}, "a run needs step 200 of the loss band"),
        ([], {"JAW_MASK": "missing.npy"}, "no jaw mask"),
        ([], {"KEEP_EVERY": 150}, "KEEP_EVERY=150 is not a multiple of the save freq 100"),
        (
            [],
            {"KEEP_EVERY": 100, "FAKE_DF_KB": 70_000_000},  # 71.7 GB: 60 + 3 copies of 10
            "Drive has 71 GB free, a run that keeps 3 more copies needs 90",
        ),
        (
            [],
            {"SPAN": 1, "STEPS": 30000, "SAVE_FREQ": 30000},
            "SPAN=1: no checkpoint past 0 fits this session: N=20000",
        ),
    ],
)
def test_each_usage_stop_prints_one_fail_token(rig, args, env, why):
    r = _run(rig, *args, **env)
    assert r.rc == 1 and r.token.startswith("COLAB_TRAIN_FAIL") and why in r.token


def test_a_jaw_mask_without_its_sidecar_is_refused(rig):
    (rig.drive / "jaw.json").unlink()
    r = _run(rig)
    assert "train_pi05.py reads the mask's sidecar" in r.token and not _calls(rig)


def test_steps_past_the_session_end_are_refused(rig):
    (rig.path / "uptime").write_text("84600.00 1.00\n")  # 23.5 h: N = 360 / 3.85 = 93
    r = _run(rig)
    assert "STEPS=300 does not fit this session: N=93" in r.token


# --- the stand-in trainer ---


def _save(cfg, step: int, model_dir: Path) -> None:
    """lerobot's checkpoint layout: <out>/checkpoints/<step>/{pretrained_model,training_state},
    then the relative `last` link."""
    ckpts = Path(cfg.output_dir) / "checkpoints"
    step_dir = ckpts / f"{step:06d}"
    (step_dir / "pretrained_model").mkdir(parents=True)
    (step_dir / "training_state").mkdir()
    cfg.save_pretrained(step_dir / "pretrained_model")
    shutil.copy(model_dir / "policy_preprocessor.json", step_dir / "pretrained_model")
    (step_dir / "pretrained_model" / "model.safetensors").write_bytes(b"\0" * 1000)
    (step_dir / "training_state" / "step").write_text(str(step))
    last = ckpts / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(step_dir.name)


def standin(args: list[str]) -> int:
    import lerobot.policies  # noqa: F401 -- registers the pi05 config
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    def parse(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    jaw = next((a.split("=", 1)[1] for a in args if a.startswith("--jaw_mask=")), None)
    if jaw is not None:  # train_pi05.py (f68e0c8) load_jaw_starts
        if not Path(jaw).with_suffix(".json").is_file():
            raise SystemExit(f"no sidecar beside {jaw}")
        if not any(a.startswith("--dataset.root=") for a in args):
            raise SystemExit("--jaw_mask needs --dataset.root")
    sys.argv = ["lerobot-train", *[a for a in args if not a.startswith("--jaw_mask=")]]
    cfg = parser.wrap()(parse)()
    model_dir = Path(cfg.policy.pretrained_path)
    tokens = _max_length(model_dir)
    mode = os.environ.get("FAKE_PREFLIGHT", "ok")
    if mode == "fail":
        print("PREFLIGHT_FAIL the weights did not load in full (loads: [False])", flush=True)
        return 3
    if (
        tokens != int(os.environ["PREFLIGHT_MAX_LENGTH"])
        or not Path(os.environ["PREFLIGHT_STATS"]).is_file()
    ):
        print(f"PREFLIGHT_FAIL the tokenizer pads to {tokens}", flush=True)
        return 3
    if jaw is not None:
        pi05_preflight.check_jaw_mask(jaw)
    if mode == "ok":
        print(f"PREFLIGHT_OK full_load=1 max_length={tokens} jaw_mask={jaw}", flush=True)
    start = int((cfg.checkpoint_path / "training_state" / "step").read_text()) if cfg.resume else 0
    record = {
        "resume": cfg.resume,
        "start": start,
        "steps": cfg.steps,
        "warmup": cfg.scheduler.num_warmup_steps,
        "decay": cfg.scheduler.num_decay_steps,
        "peak_lr": cfg.scheduler.peak_lr,
        "decay_lr": cfg.scheduler.decay_lr,
        "dtype": cfg.policy.dtype,
        "gradient_checkpointing": cfg.policy.gradient_checkpointing,
        "train_expert_only": cfg.policy.train_expert_only,
        "chunk_size": cfg.policy.chunk_size,
        "n_action_steps": cfg.policy.n_action_steps,
        "tokenizer_max_length": cfg.policy.tokenizer_max_length,
        "max_length": tokens,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "pretrained_path": str(model_dir),
        "output_dir": str(cfg.output_dir),
        "dataset_root": str(cfg.dataset.root),
        "jaw_mask": jaw,
    }
    with open(Path(os.environ["FAKE_DIR"]) / "cfg.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    loss = os.environ.get("FAKE_LOSS", "0.032")
    die = int(os.environ.get("FAKE_DIE_AT", "0"))
    for step in range(start + 1, cfg.steps + 1):
        time.sleep(float(os.environ.get("FAKE_STEP_S", "0.005")))
        if step % cfg.log_freq == 0:
            print(
                f"INFO 2026-10-08 00:00:00 ot_train.py:451 step:{step} smpl:1K ep:1 epch:0.00"
                f" loss:{loss} grdn:0.200 lr:2.5e-05 updt_s:3.800 data_s:0.050",
                flush=True,
            )
        if step % cfg.save_freq == 0 or step == cfg.steps:
            _save(cfg, step, model_dir)
        if step == die:
            print(f"fake session end at {step}", flush=True)
            return 137
    return 0


if __name__ == "__main__":
    os._exit(standin(sys.argv[1:]))

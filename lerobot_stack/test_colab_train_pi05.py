"""colab_train_pi05.sh and pi05_preflight.py (MS7h recipe 5.6 and 9, MS7k design 6.1 D5).

The preflight checks run on stub policies and pipelines. The script runs in a temp dir that
stands in for /content, with shims first on PATH: python3 (the HF checks pass, the trainer is
the stand-in, the pi05_base download copies FAKE_BASE, and with FAKE_NO_STACK the lerobot check
fails until pip ran), nvidia-smi, pip (it logs its arguments), df (FAKE_DF_KB) and cp (it logs
the Drive checkpoints at each copy to Drive, and fails the first FAKE_DIR/cp_fails copies). The
stand-in trainer is this file run as a script: it parses the trainer's argv with lerobot's own
parser, writes lerobot's checkpoint layout and log lines, and fakes the preflight's outcome
(FAKE_PREFLIGHT). A PEFT run saves an adapter whose config names the base as lerobot's does: a new
wrap names --policy.pretrained_path, and a resume loads the base that the checkpoint's adapter
config names, so the stand-in fails when that base is missing. A LoRA argv goes through
train_pi05.main first, so its lever checks and its --lora expansion run. The LoRA runs need
lerobot 0.6.1 (0.5.1 has no --peft.lora_alpha), so they skip in the 0.5.1 venv.
"""

import hashlib
import importlib.metadata
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
from train_pi05 import LORA_FULL, LORA_TARGETS, SIGLIP_TARGETS

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
    """A tokenizer of one token a word, plus BOS."""

    def __init__(self, max_length):
        self.max_length = max_length
        self.input_tokenizer = lambda text: {"input_ids": ["<bos>", *text.split()]}


def _preflight_rig(tasks=("a task",), **bad):
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
    install(module, Policy, flat, 128, list(tasks))
    if "stock" not in bad:  # train_pi05.py swaps the forward before lerobot builds anything
        Policy.forward = lambda self, batch, reduction="mean": "masked"
    import torch

    batch = {"action": torch.zeros(4, 50, 7)}
    if "nopad" not in bad:
        batch["action_is_pad"] = torch.zeros(4, 50, dtype=torch.bool)
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
    # BOS + "Task: a task, State:" + 23 bins (the last "255;") + "Action:" = 1 + 4 + 23 + 1
    assert "max_length=128 longest=29 action_is_pad" in out


def test_the_longest_prompt_must_stay_under_the_padded_length(capsys):
    fits = ("a", " ".join(["w"] * 98))  # 29 - 2 + 98 = 125 tokens
    assert _train_once(*_preflight_rig(tasks=fits)) == "masked"
    assert "longest=125 " in capsys.readouterr().out
    with pytest.raises(PreflightError, match="the longest prompt takes 128 of 128 tokens"):
        _train_once(*_preflight_rig(tasks=("a", " ".join(["w"] * 101))))


def test_the_preflight_reads_run_a_s_prompts_with_the_codes_flag(monkeypatch):
    from train_pi05 import TASK, prompt

    seen = []
    lerobot = {
        "lerobot.policies.pi05.modeling_pi05": SimpleNamespace(PI05Policy=None),
        "lerobot.scripts": SimpleNamespace(lerobot_train=None),
    }
    for name, module in lerobot.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "argv", list(sys.argv))  # train() sets it for the trainer
    monkeypatch.setattr(pi05_preflight, "install", lambda *a: seen.append(a[-1]))
    monkeypatch.setattr(pi05_preflight.runpy, "run_path", lambda *a, **k: None)
    monkeypatch.setattr(pi05_preflight, "file_stats", lambda path: {})
    monkeypatch.setenv("PREFLIGHT_STATS", "x")
    monkeypatch.setenv("PREFLIGHT_MAX_LENGTH", "200")
    pi05_preflight.train(["--steps=8"])
    pi05_preflight.train(["--mix=/d:0.3", "--prompt_codes"])
    assert seen == [[TASK], [prompt(c) for c in range(12)]]


@pytest.mark.parametrize(
    "bad, why",
    [
        ({"partial": 1}, "did not load in full"),
        ({"stats": 1}, "normalizer stats sha"),
        ({"unstats": 1}, "unnormalizer stats sha"),
        ({"tokens": 200}, "the tokenizer pads to 200, want 128"),
        ({"stock": 1}, "stock forward"),
        ({"nopad": 1}, "action_is_pad None at the forward"),
    ],
)
def test_each_preflight_miss_raises_before_the_first_update(bad, why):
    with pytest.raises(PreflightError, match=why):
        _train_once(*_preflight_rig(**bad))


def test_the_preflight_tokens(tmp_path, capsys, monkeypatch):
    assert pi05_preflight.main([]) == 2
    assert "TRAIN_FAIL usage" in capsys.readouterr().out
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(200)))
    assert pi05_preflight.main(["patch", str(tmp_path), "128"]) == 0
    assert f"PATCH_OK {tmp_path} max_length 200 -> 128" in capsys.readouterr().out

    def miss(argv):
        raise PreflightError("a miss")

    monkeypatch.setattr(pi05_preflight, "train", miss)
    assert pi05_preflight.main(["train"]) == 3
    assert "PREFLIGHT_FAIL a miss" in capsys.readouterr().out


# --- colab_train_pi05.sh dry runs with the stand-in trainer ---

SHIMS = {
    "python3": """
case "$1 $2" in *pi05_preflight.py\\ train) shift 2; exec "$FAKE_PY" "$FAKE_STANDIN" "$@" ;; esac
case "$*" in *snapshot_download*)
    echo "$3 $4" >> "$FAKE_DIR/download.log"; mkdir -p "$5"; exec cp -r "$FAKE_BASE"/. "$5" ;;
esac
case "$*" in *"import lerobot"*)
    [ -z "${FAKE_NO_STACK:-}" ] || [ -s "$FAKE_DIR/pip.log" ] || exit 1; exit 0 ;;
esac
case "$*" in *huggingface_hub*|*AutoTokenizer*) exit 0 ;; esac
exec "$FAKE_PY" "$@"
""",
    "nvidia-smi": """
case "$*" in *nounits*) echo 81920 ;; *) echo "NVIDIA A100-SXM4-80GB, 81920 MiB" ;; esac
""",
    "pip": 'echo "$*" >> "$FAKE_DIR/pip.log"\n',
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
    _helper(drive, "0.5.1")  # the modes without LORA run a53379c's train_pi05.py
    (meta / "info.json").write_text("{}")
    (meta / "stats.json").write_text(json.dumps(STATS))
    (meta / "action_space.json").write_text('{"action_space": "absolute", "gripper": "continuous"}')
    for d in (start, base):
        (d / "model.safetensors").write_bytes(b"\0" * 1000)
        (d / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(200)))
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


def _helper(drive: Path, version: str) -> None:
    """train_pi05.py on the Drive as the script checks it: its version line."""
    text = (HERE / "train_pi05.py").read_text()
    (drive / "train_pi05.py").write_text(
        text.replace('LEROBOT = "0.6.1"', f'LEROBOT = "{version}"')
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
    assert not rig.run.exists() and not (rig.drive / "action_space.json").exists()


BASE = "lerobot/pi05_base"
BASE_REV = "a538eb273274eb30f126a118f39dbc0ee212c883"


def test_a_smoke_from_pi05_base_downloads_the_pinned_revision_and_reads_its_own_band(rig):
    r = _run(rig, "--smoke", START=BASE, RUN_NAME=None, FAKE_LOSS="0.092")
    assert r.token.startswith("COLAB_TRAIN_OK smoke") and "loss200=0.092" in r.token
    assert "loss band: step 200 loss 0.092 in [0.046, 0.34]" in r.stdout
    assert (rig.fake / "download.log").read_text() == f"{BASE} {BASE_REV}\n"
    (call,) = _calls(rig)
    assert call["pretrained_path"] == str(rig.path / "content" / "start_model")
    assert call["max_length"] == 128
    assert call["warmup"] == 6 and call["decay"] == 200
    assert _max_length(rig.path / "pi05_base") == 200
    r = _run(rig, "--smoke", START=BASE, RUN_NAME=None)  # a warm start's loss
    assert "loss band: step 200 loss 0.032 outside [0.046, 0.34]" in r.token


def test_a_span_run_on_a_shared_drive_keeps_its_reads_and_resumes_toward_the_same_steps(rig):
    # the owner's 2026-10-09 run in small: DRIVE_ROOT on a Shared drive, pi05_base, SPAN=1,
    # KEEP_EVERY, DRIVE_MIN_GB=0 (df reads the account's quota, here 51 GB)
    shared = rig.path / "content" / "drive" / "Shareddrives" / "lab" / "ms7h"
    shared.parent.mkdir(parents=True)
    shutil.move(rig.drive, shared)
    rig.drive.parent.rmdir()  # no MyDrive at all
    rig.drive, rig.run = shared, shared / "checkpoints_ab"
    env = {"START": BASE, "STEPS": 600, "SAVE_FREQ": 100, "SPAN": 1}
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
    assert first["max_length"] == 128 and not first["resume"]
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
        "use_peft": False,
        "peft": None,
    }
    assert {k: call[k] for k in want} == want
    assert call["pretrained_path"] == str(rig.path / "content" / "start_model")
    assert _held(rig) == ["000301", "last.txt"]
    assert sorted(p.name for p in (rig.run / "000301").iterdir()) == ["pretrained_model"]
    held = (rig.fake / "cp.log").read_text().splitlines()
    assert any(not line.endswith("held: ") for line in held)  # room: copy, then delete
    assert (rig.drive / "action_space.json").exists()


def test_a_resumed_session_trains_toward_the_run_s_steps(rig):
    r = _run(rig, FAKE_DIE_AT=250)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and "the trainer exited 137" in r.token
    assert (rig.run / "last.txt").read_text().strip() == "000200"
    assert (rig.run / "000200" / "training_state").is_dir()
    shutil.rmtree(rig.path / "content" / "dataset")  # a new VM: the local disk is gone
    shutil.rmtree(rig.out)
    r = _run(rig, STEPS=None)
    assert "resuming checkpoints_ab from checkpoint 000200 toward step 300" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=300 last=000300")
    first, second = _calls(rig)
    assert second["resume"] is True and second["start"] == 200 and second["steps"] == 300
    assert second["output_dir"] == str(rig.out) and second["max_length"] == 128
    assert second["dataset_root"] == str(rig.path / "content" / "dataset")
    assert second["warmup"] == first["warmup"] == 10 and second["decay"] == 300
    assert _held(rig) == ["000300", "last.txt"]
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
    assert _held(rig) == ["000300", "last.txt"]


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
        ([], {"TRAIN_EXTRA": "--close_weight=3,2,1"}, "TRAIN_EXTRA needs LORA"),
        ([], {"DIR": None}, "set DIR="),
        ([], {"RUN_NAME": None}, "set RUN_NAME="),
        ([], {"START": None}, "set START="),
        ([], {"STATS_MD5": "0" * 32}, "is not the corpus's 00000000"),
        ([], {"STEPS": 100}, "a run needs step 200 of the loss band"),
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


def test_steps_past_the_session_end_are_refused(rig):
    (rig.path / "uptime").write_text("84600.00 1.00\n")  # 23.5 h: N = 360 / 3.85 = 93
    r = _run(rig)
    assert "STEPS=300 does not fit this session: N=93" in r.token


# --- the DAgger mode (HD59) and colab_dagger.sh ---


def _dagger_root(rig, path: Path) -> Path:
    """A dagger_vla.py root as far as the scripts read it: the corpus's info and stats."""
    (path / "meta").mkdir(parents=True)
    for name in ("info.json", "stats.json"):
        shutil.copy(rig.drive / "dataset" / "meta" / name, path / "meta" / name)
    return path


def test_a_dagger_smoke_trains_the_mix_at_its_own_recipe(rig):
    dag = _dagger_root(rig, rig.path / "dag")
    env = {"DAGGER": dag, "RUN_NAME": None, "FAKE_LOSS": "0.070"}
    r = _run(rig, "--smoke", **env)
    assert r.token.startswith("COLAB_TRAIN_OK smoke") and "loss200=0.070" in r.token
    assert "loss band: step 200 loss 0.070 in [0.015, 0.10]" in r.stdout
    (call,) = _calls(rig)
    want = {"peak_lr": 1e-5, "decay_lr": 2.5e-6, "warmup": 200, "decay": 200, "batch_size": 32}
    assert {k: call[k] for k in want} == want
    assert call["mix"] == [[str(dag), 0.3]] and not call["prompt_codes"]


def test_a_dagger_run_keeps_every_save_and_resumes_only_with_its_mix(rig):
    dag = _dagger_root(rig, rig.path / "dag")
    env = {"DAGGER": dag, "STEPS": 400, "SAVE_FREQ": 100}
    r = _run(rig, **env, FAKE_DIE_AT=250)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and "the trainer exited 137" in r.token
    assert (rig.run / "mix.txt").read_text() == f"{dag}:0.3|\n"
    shutil.rmtree(rig.out)  # a new VM
    r = _run(rig, **{**env, "DAGGER": None})
    assert f"trains on MIX|PROMPT_CODES '{dag}:0.3|': resume it with the same two" in r.token
    r = _run(rig, **env, DAGGER_SHARE=0.5)
    assert "trains on MIX|PROMPT_CODES" in r.token
    r = _run(rig, **env)
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=400 last=000400")
    first, second = _calls(rig)
    assert (first["warmup"], first["decay"], first["peak_lr"]) == (200, 400, 1e-5)
    assert second["resume"] is True and second["start"] == 200
    assert second["mix"] == [[str(dag), 0.3]]
    keep = sorted(p.name for p in (rig.run / "keep").iterdir())
    assert keep == ["000100", "000200", "000300", "000400"]  # every save, for the reads


@pytest.mark.parametrize(
    "env, why",
    [
        ({"DAGGER": "/nowhere"}, "no /nowhere/meta/info.json"),
        ({"BAD_STATS": 1}, "dag/meta/stats.json md5"),
    ],
)
def test_each_dagger_stop_prints_one_fail_token(rig, env, why):
    dag = _dagger_root(rig, rig.path / "dag")
    if env.pop("BAD_STATS", None):
        (dag / "meta" / "stats.json").write_text("{}")
    r = _run(rig, **{"DAGGER": dag, **env})
    assert r.token.startswith("COLAB_TRAIN_FAIL") and why in r.token and not _calls(rig)


def _mix_roots(rig, codes: bool) -> str:
    """Run A's two relative MIX roots on the Drive, with their prompt codes if ``codes``, and the
    corpus codes file: the MIX value."""
    for name in ("own_a", "takeovers"):
        root = _dagger_root(rig, rig.drive / "ms7i" / name)
        if codes:
            np.save(root / "meta" / "prompt_codes.npy", np.full(5, 4, np.int8))
    np.save(rig.drive / "ms7i" / "corpus_prompt_codes.npy", np.arange(6, dtype=np.int8))
    return "ms7i/own_a:0.3 ms7i/takeovers:0.2"


def test_run_a_copies_each_drive_root_and_the_codes_and_trains_at_200_tokens(rig):
    mix = _mix_roots(rig, codes=True)
    env = {"MIX": mix, "PROMPT_CODES": "ms7i/corpus_prompt_codes.npy", "MAX_LENGTH": 200}
    env |= {"OUT_ROOT": rig.path / "scratch", "FAKE_LOSS": "0.070"}
    r = _run(rig, "--smoke", RUN_NAME=None, **env)
    assert r.token.startswith("COLAB_TRAIN_OK smoke") and "loss200=0.070" in r.token
    (call,) = _calls(rig)
    local = [rig.path / "scratch" / f"mix_ms7i_{n}" for n in ("own_a", "takeovers")]
    assert call["mix"] == [[str(local[0]), 0.3], [str(local[1]), 0.2]] and call["prompt_codes"]
    assert (call["max_length"], call["tokenizer_max_length"], call["preflight_max_length"]) == (
        200,
        200,
        "200",
    )
    for root in local:
        assert (root / ".copied").exists() and (root / "meta" / "prompt_codes.npy").exists()
    codes = rig.path / "content" / "dataset" / "meta" / "prompt_codes.npy"
    assert np.load(codes).tolist() == list(range(6))
    assert not (rig.drive / "dataset" / "meta" / "prompt_codes.npy").exists()  # Drive untouched
    (local[0] / "meta" / "info.json").unlink()  # a session copy stays: no second copy
    _run(rig, "--smoke", RUN_NAME=None, **env)
    assert not (local[0] / "meta" / "info.json").exists()


@pytest.mark.parametrize(
    "env, why",
    [
        ({"MIX": "ms7i/own_a"}, "MIX entry ms7i/own_a: want ROOT:SHARE"),
        ({"MIX": "ms7i/none:0.3"}, "no {drive}/ms7i/none/meta/info.json: MIX root ms7i/none"),
        ({"BAD_STATS": 1}, "{drive}/ms7i/takeovers/meta/stats.json md5"),
        ({"PROMPT_CODES": "ms7i/none.npy"}, "no PROMPT_CODES file {drive}/ms7i/none.npy"),
        ({"NO_CODES": 1}, "no {drive}/ms7i/own_a/meta/prompt_codes.npy: PROMPT_CODES"),
        ({"MIX": None}, "PROMPT_CODES needs MIX"),
        ({"MAX_LENGTH": "2OO"}, "MAX_LENGTH=2OO is not a token count"),
    ],
)
def test_each_mix_stop_prints_one_fail_token_before_any_copy(rig, env, why):
    mix = _mix_roots(rig, codes=not env.pop("NO_CODES", None))
    if env.pop("BAD_STATS", None):
        (rig.drive / "ms7i" / "takeovers" / "meta" / "stats.json").write_text("{}")
    env = {"MIX": mix, "PROMPT_CODES": "ms7i/corpus_prompt_codes.npy", **env}
    r = _run(rig, **env)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and why.format(drive=rig.drive) in r.token
    assert not _calls(rig) and not (rig.path / "content" / "dataset").exists()


def test_a_mix_run_keeps_4k_and_8k_and_resumes_only_with_its_mix_and_codes(rig):
    mix = _mix_roots(rig, codes=True)
    env = {"MIX": mix, "PROMPT_CODES": "ms7i/corpus_prompt_codes.npy", "STEPS": None}
    env |= {"SAVE_FREQ": 2000, "KEEP_EVERY": 4000, "FAKE_STEP_S": 0.0002, "DRIVE_MIN_GB": 0}
    r = _run(rig, **env, FAKE_DIE_AT=5000)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and "the trainer exited 137" in r.token
    assert (rig.run / "mix.txt").read_text() == f"{mix}|ms7i/corpus_prompt_codes.npy\n"
    assert (rig.run / "last.txt").read_text().strip() == "004000"
    shutil.rmtree(rig.out)  # a new VM
    for other in ({"PROMPT_CODES": None}, {"MIX": "ms7i/own_a:0.3"}):
        r = _run(rig, **{**env, **other})
        assert "trains on MIX|PROMPT_CODES" in r.token and r.rc == 1
    r = _run(rig, **env)
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=8000 last=008000")
    first, second = _calls(rig)
    assert (first["steps"], first["warmup"], first["decay"], first["peak_lr"]) == (
        8000,
        200,
        8000,
        1e-5,
    )
    assert second["resume"] is True and second["start"] == 4000 and second["prompt_codes"]
    assert second["mix"] == first["mix"] and len(second["mix"]) == 2
    assert sorted(p.name for p in (rig.run / "keep").iterdir()) == ["004000", "008000"]


def test_prepare_copies_the_dataset_and_trains_nothing(rig):
    local = rig.path / "scratch" / "dataset"
    r = _run(rig, "--prepare", RUN_NAME=None, LOCAL_DATA=local)
    assert r.token == f"COLAB_TRAIN_OK prepared data={local}"
    assert (local / ".copied").exists() and (local / "meta" / "stats.json").exists()
    assert not _calls(rig) and not (rig.path / "content" / "dataset").exists()


def test_the_dataset_copy_never_deletes_the_drive_or_the_corpus(rig):
    corpus, content = rig.drive / "dataset", rig.path / "content"
    env = {"RUN_NAME": None, "DATASET": f"{corpus}/"}  # a tab-completed path
    r = _run(rig, "--prepare", **env)
    assert r.token == f"COLAB_TRAIN_OK prepared data={content / 'dataset'}"
    for local in (corpus, f"{corpus}/", content, rig.drive.parent):
        r = _run(rig, "--prepare", **env, LOCAL_DATA=local)
        assert r.token.startswith("COLAB_TRAIN_FAIL") and f"LOCAL_DATA={local} " in r.token
    assert (corpus / "meta" / "info.json").exists() and (rig.drive / "train_pi05.py").exists()


def _runner(rig, *args, **env):
    full = {**rig.env, **{k: str(v) for k, v in env.items()}}
    full.pop("RUN_NAME")  # the runner's default, checkpoints_dagger
    full.update(DRIVE_ROOT=str(rig.drive.parent), SCRATCH=str(rig.path / "scratch"))
    full.update(PARTS=str(rig.path / "parts"), STEPS="400", SAVE_FREQ="100")
    out = subprocess.run(
        ["bash", str(rig.drive / "colab_dagger.sh"), *map(str, args)],
        env=full,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    lines = out.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("DAGGER_"), out.stdout + out.stderr
    return lines[0]


def _until(rig, step: str, want: str) -> str:
    deadline = time.monotonic() + 120
    while not (line := _runner(rig, step)).startswith(want):
        assert "_FAIL" not in line and time.monotonic() < deadline, line
        time.sleep(0.3)
    return line


def test_the_cli_runner_takes_a_dagger_tar_to_a_kept_run(rig):
    shutil.copy(HERE / "colab_dagger.sh", rig.drive)
    scratch, parts = rig.path / "scratch", rig.path / "parts"
    assert _runner(rig, "status").startswith("DAGGER_STATUS setup=none smoke=none run=none")
    assert "setup has not passed: none" in _runner(rig, "smoke")
    assert _runner(rig, "setup").startswith("DAGGER_SETUP_STARTED")
    assert _until(rig, "setup", "DAGGER_SETUP_OK") == (
        f"DAGGER_SETUP_OK COLAB_TRAIN_OK prepared data={scratch / 'dataset'}"
    )
    # dagger_vla.py's tar of one root, in two parts
    _dagger_root(rig, rig.path / "dagx" / "dag")
    tar = rig.path / "dag.tar"
    subprocess.run(["tar", "-cf", tar, "-C", rig.path / "dagx", "dag"], check=True)
    blob = tar.read_bytes()
    parts.mkdir(exist_ok=True)
    (parts / "dag.tar.part000").write_bytes(blob[:1000])
    (parts / "dag.tar.part001").write_bytes(blob[1000:])
    sha = hashlib.sha256(blob).hexdigest()
    assert "sha256 is" in _runner(rig, "receive", len(blob), "0" * 64)
    line = _runner(rig, "receive", len(blob), sha)
    assert line.startswith(f"DAGGER_RECEIVE_OK root={scratch / 'dagger_dataset'}"), line
    assert not list(parts.iterdir()) and (scratch / "dagger_dataset" / "meta").is_dir()
    assert "the smoke has not passed" in _runner(rig, "start")
    assert _runner(rig, "smoke").startswith("DAGGER_SMOKE_STARTED")
    assert "COLAB_TRAIN_OK smoke" in _until(rig, "smoke", "DAGGER_SMOKE_OK")
    assert _runner(rig, "start").startswith("DAGGER_RUN_STARTED")
    deadline = time.monotonic() + 120
    while "run=ok" not in (status := _runner(rig, "status")):
        assert "run=fail" not in status and time.monotonic() < deadline, status
        time.sleep(0.3)
    assert "last_save=000400 kept=000100,000200,000300,000400" in status
    assert _runner(rig, "start").startswith(
        "DAGGER_RUN_OK COLAB_TRAIN_OK run=checkpoints_dagger steps=400 last=000400"
    )
    smoke, run = _calls(rig)
    for call in (smoke, run):
        assert call["mix"] == [[str(scratch / "dagger_dataset"), 0.3]]
        assert call["dataset_root"] == str(scratch / "dataset") and call["peak_lr"] == 1e-5
    assert call["output_dir"] == str(scratch / "pi05_checkpoints_dagger")


# --- the LoRA mode (MS7k) ---

B3_BASE = "~/data/maker-arm-ms7/j/pi05_base"  # what B3's adapter config names
PALIGEMMA_REV = "35e4f46485b4d07967e7e9935bc3786aad50687c"
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
TORCH = "torch==2.11.0+cu126 torchvision==0.26.0+cu126"
STACK = [
    f"install -q {TORCH} --index-url https://download.pytorch.org/whl/cu126",
    "install -q lerobot[pi,peft,training]==0.6.1 transformers==5.5.4 peft==0.21.2",
]
needs_061 = pytest.mark.skipif(
    importlib.metadata.version("lerobot") != "0.6.1", reason="the LoRA flags parse on 0.6.1"
)


def _lora(rig, rank: int = 128) -> dict:
    """B3's 030000 as the LoRA mode reads it from the Drive, the tokenizer folder, and the 0.6.1
    train_pi05.py next to the script: the env of a run with no HF token."""
    _helper(rig.drive, "0.6.1")
    adapter = rig.drive / "k" / "b3_030000"
    adapter.mkdir(parents=True)
    config = {"base_model_name_or_path": B3_BASE, "r": rank, "lora_alpha": rank}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "adapter_model.safetensors").write_bytes(b"\1" * 1000)
    (adapter / "policy_preprocessor.json").write_text(json.dumps(_preprocessor_json(128)))
    tokenizer = rig.drive / "k" / "paligemma_tokenizer"
    tokenizer.mkdir()
    for name in TOKENIZER_FILES:
        (tokenizer / name).write_text("{}")
    return {
        "LORA": rank,
        "START": "k/b3_030000",
        "TOKENIZER": "k/paligemma_tokenizer",
        "HF_TOKEN": None,
        "FAKE_LOSS": "0.030",
    }


def _base(path: Path) -> str:
    return json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]


@needs_061
def test_a_lora_smoke_continues_the_adapter_offline_on_the_pinned_stack(rig):
    # the operator's layout: the corpus of another Drive folder, read in place
    corpus = rig.path / "content" / "drive" / "MyDrive" / "ms7h_old" / "dataset"
    shutil.move(rig.drive / "dataset", corpus)
    scratch = rig.path / "scratch"
    env = _lora(rig) | {"DATASET": corpus, "OUT_ROOT": scratch, "FAKE_NO_STACK": 1}
    env |= {"DECAY_STEPS": 5000}  # a run's: the smoke keeps a decay of its own 200 steps
    r = _run(rig, "--prepare", RUN_NAME=None, **env)
    assert r.token == f"COLAB_TRAIN_OK prepared data={rig.path / 'content' / 'dataset'}"
    assert (rig.fake / "pip.log").read_text().splitlines() == STACK
    assert (rig.fake / "download.log").read_text() == f"{BASE} {BASE_REV}\n"
    assert (scratch / "pi05_base" / ".done").exists()
    repo = scratch / "hf_hub" / "models--google--paligemma-3b-pt-224"
    assert (repo / "refs" / "main").read_text() == PALIGEMMA_REV
    assert sorted(p.name for p in (repo / "snapshots" / PALIGEMMA_REV).iterdir()) == sorted(
        TOKENIZER_FILES
    )
    assert not _calls(rig)

    r = _run(rig, "--smoke", RUN_NAME=None, **env)
    assert r.token.startswith("COLAB_TRAIN_OK smoke sec_per_step=3.850 checkpoint_gib=")
    assert "loss200=0.030 batch=32 expert_only=false lora=128 save=" in r.token
    assert "loss band: step 200 loss 0.030 in [0.010, 0.06]" in r.stdout
    assert len((rig.fake / "pip.log").read_text().splitlines()) == 2  # installed once
    assert (rig.fake / "download.log").read_text().count("\n") == 1  # the base once a runtime
    (call,) = _calls(rig)
    want = {
        "resume": False,
        "steps": 200,
        "warmup": 200,  # the run's first 200 steps: its warmup, no rescale by lerobot
        "decay": 200,
        "peak_lr": 5e-5,
        "decay_lr": 1e-5,
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
        "train_expert_only": False,
        "tokenizer_max_length": 128,
        "max_length": 128,
        "batch_size": 32,
        "seed": 1001,
        "use_peft": False,  # a new wrap, as Machine A probes start
        "peft": [128, 128],
        "targets": [LORA_TARGETS],
        "full": list(LORA_FULL),
        "pretrained_path": str(scratch / "pi05_base"),
        "base": None,  # no adapter config names the base
        "dataset_root": str(rig.path / "content" / "dataset"),
        "offline": "1",
        "hub_cache": str(scratch / "hf_hub"),
    }
    assert {k: call[k] for k in want} == want
    assert call["argv"][-2:] == ["--lora=128", f"--lora_from={scratch / 'start_model'}"]
    assert _max_length(scratch / "pi05_base") == 128  # the processors come from the base
    for adapter in (rig.drive / "k" / "b3_030000", scratch / "start_model"):
        assert _base(adapter) == B3_BASE  # --lora_from does not read the name, nothing sets it
    save = Path(r.token.split("save=")[1])
    assert save == scratch / "pi05_smoke" / "checkpoints" / "000200"
    sums = (save / "SHA256SUMS").read_text()
    assert "pretrained_model/adapter_model.safetensors" in sums and "training_state" not in sums
    subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=save, check=True)
    assert not (rig.drive / "action_space.json").exists()


@needs_061
def test_a_lora_run_keeps_summed_saves_and_resumes_only_with_its_recipe(rig):
    extra = "--siglip_lora  --overhead_aug=0.5,0.06,3"
    env = _lora(rig) | {"STEPS": 600, "SAVE_FREQ": 100, "KEEP_EVERY": 200, "TRAIN_EXTRA": extra}
    r = _run(rig, **env, FAKE_DIE_AT=350)
    assert r.token.startswith("COLAB_TRAIN_FAIL") and "the trainer exited 137" in r.token
    assert (rig.run / "lora.txt").read_text() == f"128|{' '.join(extra.split())}\n"
    assert (rig.run / "last.txt").read_text().strip() == "000300"
    content = rig.path / "content"
    for path in (rig.out, content / "dataset", content / "pi05_base"):
        shutil.rmtree(path)  # a new VM, whose base copy sits on another disk
    env["OUT_ROOT"] = scratch = rig.path / "scratch"
    for other in ({"TRAIN_EXTRA": "--siglip_lora"}, {"LORA": 64}):
        r = _run(rig, **{**env, **other})
        assert r.rc == 1 and "trains with LORA|TRAIN_EXTRA '128|--siglip_lora --overhead" in r.token
    r = _run(rig, **env)
    assert "resuming checkpoints_ab from checkpoint 000300 toward step 600" in r.stdout
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=600 last=000600")
    first, second = _calls(rig)
    # a fresh run: the patched base, a new wrap and --lora_from, as Machine A probes start
    assert first["argv"][-4:] == [
        "--lora=128",
        f"--lora_from={content / 'start_model'}",
        *extra.split(),
    ]
    assert not any(a.startswith(("--policy.use_peft", "--resume")) for a in first["argv"])
    assert first["pretrained_path"] == str(content / "pi05_base") and first["use_peft"] is False
    # a resume: lerobot's own PEFT load, from the base that the re-pointed adapter config names,
    # with --lora (for --siglip_lora) and the levers again, and no --lora_from
    out = scratch / "pi05_checkpoints_ab"
    assert second["argv"] == [
        f"--config_path={out}/checkpoints/last/pretrained_model/train_config.json",
        "--resume=true",
        f"--output_dir={out}",
        f"--dataset.root={content / 'dataset'}",
        "--lora=128",
        *extra.split(),
    ]
    base = str(scratch / "pi05_base")
    assert second["resume"] is True and second["start"] == 300
    assert second["use_peft"] is True and second["base"] == base
    for call in (first, second):
        assert (call["steps"], call["warmup"], call["decay"]) == (600, 200, 600)
        assert (call["peak_lr"], call["decay_lr"], call["seed"]) == (5e-5, 1e-5, 1001)
        assert call["targets"] == [f"{LORA_TARGETS}|{SIGLIP_TARGETS}"]
    keep = rig.run / "keep"
    assert sorted(p.name for p in keep.iterdir()) == ["000200", "000400", "000600"]
    for step in keep.iterdir():
        assert sorted(p.name for p in step.iterdir()) == ["SHA256SUMS", "pretrained_model"]
        subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=step, check=True)
    assert _base(keep / "000200" / "pretrained_model") == str(content / "pi05_base")
    assert _base(keep / "000600" / "pretrained_model") == base
    assert _held(rig) == ["000600", "keep", "last.txt", "lora.txt"]


@needs_061
def test_an_l1_run_never_resumes(rig):
    env = {**_lora(rig), "TRAIN_EXTRA": "--close_weight=3,2,1"}
    r = _run(rig, **env, FAKE_DIE_AT=250)
    assert "the trainer exited 137" in r.token and (rig.run / "000200").is_dir()
    shutil.rmtree(rig.out)  # a new VM
    r = _run(rig, **env)
    assert r.rc == 1 and "trains with --close_weight, which cannot resume: start it" in r.token
    assert len(_calls(rig)) == 1


@needs_061
def test_a_lora_continuation_that_reads_like_a_fresh_lora_stops_at_step_200(rig):
    # B3 from pi05_base read 0.094 to 0.100 where its summed lr meets this warmup's
    r = _run(rig, **{**_lora(rig), "STEPS": 600, "FAKE_LOSS": "0.095"})
    assert "loss band: step 200 loss 0.095 outside [0.010, 0.06]" in r.token
    assert not (rig.out / "checkpoints" / "000600").exists()


@needs_061
def test_a_lora_run_takes_its_schedule_from_the_env(rig):
    env = {"PEAK_LR": "4e-5", "WARMUP": 100, "DECAY_STEPS": 250, "DECAY_LR": "2e-5"}
    r = _run(rig, **{**_lora(rig), **env})
    assert r.token.startswith("COLAB_TRAIN_OK run=checkpoints_ab steps=300 last=000300")
    (call,) = _calls(rig)
    got = [call[k] for k in ("steps", "warmup", "decay", "peak_lr", "decay_lr")]
    assert got == [300, 100, 250, 4e-5, 2e-5]


@pytest.mark.parametrize(
    "args, env, why",
    [
        ([], {"LORA": "r8"}, "LORA=r8 is not a LoRA rank"),
        (["--expert_only"], {}, "--expert_only: the LoRA recipe trains the VLM's LoRA too"),
        ([], {"DAGGER": "/nowhere"}, "LORA: MIX and DAGGER need lerobot 0.5.1"),
        ([], {"LORA": 64}, "START is a rank 128 adapter, not LORA=64"),
        ([], {"START": BASE}, "LORA: no adapter_model.safetensors in START lerobot/pi05_base"),
        (
            [],
            {"START": "checkpoints/030000/pretrained_model"},
            "LORA: no adapter_model.safetensors in START {drive}/checkpoints/030000",
        ),
        ([], {"TOKENIZER": "k/none"}, "no tokenizer_config.json in TOKENIZER {drive}/k/none"),
        ([], {"DECAY_STEPS": 301}, "DECAY_STEPS=301: want a step count up to STEPS=300"),
        (
            [],
            {
                "TRAIN_EXTRA": "--overhead_aug=0.5,0.06,3 --close_weight=3,2,1",
                "SPAN": 1,
                "STEPS": 20001,
            },
            "--close_weight cannot resume, so STEPS=20001 must fit this session: N=20000",
        ),
        (
            [],
            {"STEPS": 600, "KEEP_EVERY": 200, "FAKE_DF_KB": 61_000_000},  # 62.5 GB: 60 + 3 x 1
            "Drive has 62 GB free, a run that keeps 3 more copies needs 63",
        ),
        ([], {"HELPER": "0.5.1"}, "train_pi05.py is not for lerobot 0.6.1"),
        (
            [],
            {"LORA": None, "START": "checkpoints/030000/pretrained_model"},
            "train_pi05.py is not for lerobot 0.5.1 (without LORA: git show a53379c:",
        ),
    ],
)
def test_each_lora_stop_prints_one_fail_token_before_any_copy(rig, args, env, why):
    lora = _lora(rig)
    if helper := env.pop("HELPER", None):
        _helper(rig.drive, helper)
    r = _run(rig, *args, **{**lora, "FAKE_NO_STACK": 1, **env})
    assert (
        r.rc == 1
        and r.token.startswith("COLAB_TRAIN_FAIL")
        and why.format(drive=rig.drive) in r.token
    )
    assert not _calls(rig) and not (rig.path / "content" / "dataset").exists()
    assert not (rig.fake / "pip.log").exists()  # no install minutes spent


# --- the stand-in trainer ---


def _save(cfg, step: int, model_dir: Path, adapter: dict | None) -> None:
    """lerobot's checkpoint layout: <out>/checkpoints/<step>/{pretrained_model,training_state},
    then the relative `last` link. A PEFT run saves ``adapter`` as its adapter config."""
    ckpts = Path(cfg.output_dir) / "checkpoints"
    step_dir = ckpts / f"{step:06d}"
    (step_dir / "pretrained_model").mkdir(parents=True)
    (step_dir / "training_state").mkdir()
    cfg.save_pretrained(step_dir / "pretrained_model")
    shutil.copy(model_dir / "policy_preprocessor.json", step_dir / "pretrained_model")
    if adapter is not None:  # peft saves the adapter and its config, base name included
        (step_dir / "pretrained_model" / "adapter_config.json").write_text(json.dumps(adapter))
        (step_dir / "pretrained_model" / "adapter_model.safetensors").write_bytes(b"%d" % step)
    else:
        (step_dir / "pretrained_model" / "model.safetensors").write_bytes(b"\0" * 1000)
    (step_dir / "training_state" / "step").write_text(str(step))
    last = ckpts / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(step_dir.name)


def standin(args: list[str]) -> int:
    """train_pi05.py on ``args``: a LoRA argv goes through its main, with lerobot_train.main
    swapped for ``trainer``. The mix modes' argv (0.5.1) goes to ``trainer`` directly."""
    import train_pi05

    if not any(a.startswith(train_pi05.LORA_FLAG) for a in args):
        return trainer(args, args)
    from lerobot.scripts import lerobot_train

    codes = []
    lerobot_train.main = lambda: codes.append(trainer(sys.argv[1:], args))
    train_pi05.main(args)
    return codes[0]


def trainer(argv: list[str], args: list[str]) -> int:
    """lerobot-train on ``argv`` (``args`` as the script passed them)."""
    import lerobot.policies  # noqa: F401 -- registers the pi05 config
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    from train_pi05 import split_mix

    def parse(cfg: TrainPipelineConfig):
        cfg.validate()
        return cfg

    mix, prompts, rest = split_mix(argv)
    sys.argv = ["lerobot-train", *rest]
    cfg = parser.wrap()(parse)()
    model_dir = Path(cfg.policy.pretrained_path)
    use_peft, adapter, base = cfg.policy.use_peft, None, None
    if use_peft:  # make_policy loads the base that the adapter config names, then the adapter
        adapter = json.loads((model_dir / "adapter_config.json").read_text())
        base = adapter["base_model_name_or_path"]
        if not (Path(base) / "model.safetensors").is_file():
            print(f"no model.safetensors in the base {base}", flush=True)
            return 1
    elif cfg.peft is not None:  # wrap_with_peft names the base and sets use_peft in the config
        adapter = {"base_model_name_or_path": str(model_dir), "r": cfg.peft.r}
        cfg.policy.use_peft = True
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
    if mode == "ok":
        print(f"PREFLIGHT_OK full_load=1 max_length={tokens}", flush=True)
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
        "preflight_max_length": os.environ["PREFLIGHT_MAX_LENGTH"],
        "mix": [[root, share] for root, share in mix],
        "prompt_codes": prompts,
        "use_peft": use_peft,
        "peft": None if cfg.peft is None else [cfg.peft.r, cfg.peft.lora_alpha],
        "targets": None if cfg.peft is None else [cfg.peft.target_modules],
        "full": None if cfg.peft is None else cfg.peft.full_training_modules,
        "argv": args,
        "offline": os.environ.get("HF_HUB_OFFLINE"),
        "hub_cache": os.environ.get("HF_HUB_CACHE"),
        "base": base,
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
            _save(cfg, step, model_dir, adapter)
        if step == die:
            print(f"fake session end at {step}", flush=True)
            return 137
    return 0


if __name__ == "__main__":
    os._exit(standin(sys.argv[1:]))

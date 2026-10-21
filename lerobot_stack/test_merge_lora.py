"""merge_lora.py with stand-ins for peft and PI05Policy: no model is loaded."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import merge_lora

BASE = "/base/pi05"


class Policy:
    """A one-weight policy: the chunk is w times the first image's mean, plus the noise."""

    def __init__(self):
        self.w = torch.tensor(2.0)
        self.config = SimpleNamespace(
            image_features={"observation.images.wrist": SimpleNamespace(shape=(3, 4, 4))},
            chunk_size=5,
            max_action_dim=3,
            use_peft=True,
        )

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_action_chunk(self, batch, noise):
        return noise * self.w + batch["observation.images.wrist"].mean()

    def save_pretrained(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "config.json").write_text(json.dumps({"use_peft": self.config.use_peft}))
        save_file({"w": self.w.reshape(1)}, str(Path(out) / "model.safetensors"))


def install(monkeypatch, delta, merged_delta):
    """peft whose adapter adds ``delta`` to w, and whose merge folds in ``merged_delta``."""
    base = Policy()

    class Wrapped:
        def __init__(self):
            self.base_model = SimpleNamespace()
            self.base_model.model = SimpleNamespace(
                predict_action_chunk=lambda batch, noise: (
                    base.predict_action_chunk(batch, noise) + delta * noise
                )
            )

        def to(self, device):
            return self

        def eval(self):
            return self

        def merge_and_unload(self):
            base.w = base.w + merged_delta
            return base

    peft = SimpleNamespace(
        PeftConfig=SimpleNamespace(
            from_pretrained=lambda p: SimpleNamespace(base_model_name_or_path=BASE)
        ),
        PeftModel=SimpleNamespace(from_pretrained=lambda b, p, config: Wrapped()),
    )
    pi05 = SimpleNamespace(PI05Policy=SimpleNamespace(from_pretrained=lambda p: base))
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "lerobot.policies.pi05.modeling_pi05", pi05)


@pytest.fixture
def adapter(tmp_path):
    run, src = tmp_path / "run", tmp_path / "run/checkpoints/010000/pretrained_model"
    src.mkdir(parents=True)
    (run / "action_space.json").write_text(json.dumps({"action_space": "absolute"}))
    (src / "config.json").write_text(json.dumps({"use_peft": True}))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (src / name).write_text("{}")
    (src / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_bytes(b"s")
    (src / "adapter_model.safetensors").write_bytes(b"a")
    return src


def test_merge_copies_the_serving_files_and_clears_use_peft(tmp_path, monkeypatch, adapter, capsys):
    install(monkeypatch, delta=0.5, merged_delta=0.5)
    out = tmp_path / "out"
    assert merge_lora.main(["--adapter", str(adapter), "--out", str(out)]) == 0
    token, _, payload = capsys.readouterr().out.strip().partition(" ")
    assert token == "MERGE_OK" and json.loads(payload)["max_abs_diff"] < 1e-6
    assert json.loads((out / "config.json").read_text())["use_peft"] is False
    names = {p.name for p in out.iterdir()}
    assert names == {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "action_space.json",
    }
    assert json.loads((out / "action_space.json").read_text()) == {"action_space": "absolute"}


def test_a_merge_that_changes_the_output_fails_and_writes_nothing(
    tmp_path, monkeypatch, adapter, capsys
):
    install(monkeypatch, delta=0.5, merged_delta=0.6)
    out = tmp_path / "out"
    assert merge_lora.main(["--adapter", str(adapter), "--out", str(out)]) == 1
    token, _, payload = capsys.readouterr().out.strip().partition(" ")
    assert token == "MERGE_FAIL" and json.loads(payload)["max_abs_diff"] > 1e-3
    assert not out.exists()


def test_a_missing_processor_file_and_a_usage_error_print_the_fail_token(
    tmp_path, monkeypatch, adapter, capsys
):
    install(monkeypatch, delta=0.5, merged_delta=0.5)
    (adapter / "policy_postprocessor.json").unlink()
    assert merge_lora.main(["--adapter", str(adapter), "--out", str(tmp_path / "o")]) == 1
    assert capsys.readouterr().out.startswith("MERGE_FAIL ")
    assert merge_lora.main([]) == 2
    assert capsys.readouterr().out.startswith("MERGE_FAIL ")

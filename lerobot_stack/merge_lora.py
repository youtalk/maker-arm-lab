"""Merge a pi0.5 LoRA save into a full checkpoint that pi05_server.load_pi05 loads unchanged
(MS7j plan, Task 4).

    python merge_lora.py --adapter DIR --out DIR

DIR is a LeRobot 0.6.1 ``pretrained_model`` folder of a PEFT run: adapter_config.json,
adapter_model.safetensors, config.json (use_peft true) and the policy_pre/postprocessor files
with their state files. The base policy comes from the adapter config's
``base_model_name_or_path``. One fixed batch (seed 0, fixed noise) runs through the adapter
model and then the merged model. OUT gets config.json with use_peft false, model.safetensors,
the processor files and a copy of action_space.json. It prints one MERGE_OK or MERGE_FAIL line
and exits through os._exit. A max_abs_diff above 1e-3 is MERGE_FAIL and OUT is not written.
peft and lerobot are imported in main's path, so the module imports in any venv.
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

MAX_DIFF = 1e-3
TOKENS = 16  # prompt length of the check batch


def fixed_batch(config, device):
    """Seed 0 images and prompt tokens, plus the noise: the same bytes for every call."""
    import torch

    gen = torch.Generator().manual_seed(0)
    batch = {
        key: torch.rand((1, *feature.shape), generator=gen).to(device)
        for key, feature in config.image_features.items()
    }
    batch["observation.language.tokens"] = torch.randint(0, 1000, (1, TOKENS), generator=gen)
    batch["observation.language.attention_mask"] = torch.ones(1, TOKENS, dtype=torch.bool)
    for key in ("observation.language.tokens", "observation.language.attention_mask"):
        batch[key] = batch[key].to(device)
    noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=gen)
    return batch, noise.to(device)


def merge(adapter: str, out: str, device: str = "cpu") -> tuple[bool, dict]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from peft import PeftConfig, PeftModel

    from policy_server import _read_action_space_marker

    src = Path(adapter)
    processors = sorted(src.glob("policy_*"))
    if not {"policy_preprocessor.json", "policy_postprocessor.json"} <= {
        p.name for p in processors
    }:
        raise ValueError(f"{adapter} lacks the policy_preprocessor/postprocessor files")
    marker = _read_action_space_marker(adapter)
    if marker is None:
        raise ValueError(f"no action_space.json in {adapter} or its three parents")

    peft_config = PeftConfig.from_pretrained(adapter)
    if not peft_config.base_model_name_or_path:
        raise ValueError(f"{adapter}/adapter_config.json names no base model")
    # The run's config (the dataset's features), as LeRobot's factory loads a PEFT base.
    run_config = PreTrainedConfig.from_pretrained(adapter)
    base = PI05Policy.from_pretrained(peft_config.base_model_name_or_path, config=run_config)
    model = PeftModel.from_pretrained(base, adapter, config=peft_config).to(device).eval()
    batch, noise = fixed_batch(base.config, device)
    with torch.no_grad():
        before = model.base_model.model.predict_action_chunk(batch, noise=noise)
        merged = model.merge_and_unload().eval()  # in place: `before` is taken first
        after = merged.predict_action_chunk(batch, noise=noise)
    diff = float((before - after).abs().max())
    info = {"adapter": adapter, "out": out, "max_abs_diff": diff}
    if diff > MAX_DIFF:
        return False, info
    merged.config.use_peft = False
    merged.save_pretrained(out)
    for path in processors:
        shutil.copy(path, Path(out) / path.name)
    (Path(out) / "action_space.json").write_text(json.dumps(marker))
    return True, info


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    try:
        args = parser.parse_args(argv)
        ok, info = merge(args.adapter, args.out, args.device)
    except SystemExit as exc:  # argparse error or --help
        if exc.code:
            print("MERGE_FAIL " + json.dumps({"error": "usage"}), flush=True)
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 -- any failure must become the fail token
        traceback.print_exc()
        print(
            "MERGE_FAIL " + json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True
        )
        return 1
    print(("MERGE_OK " if ok else "MERGE_FAIL ") + json.dumps(info), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()  # os._exit skips the flush, and --help prints without one
    os._exit(code)

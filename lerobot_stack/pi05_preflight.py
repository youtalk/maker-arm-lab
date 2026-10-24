"""The preflight of a pi0.5 Colab run (MS7h recipe 5.6, review findings F3 and F11).

    python pi05_preflight.py patch <pretrained_model dir> <max_length>
    python pi05_preflight.py train <train_pi05.py arguments>

``patch`` sets the prompt length of the tokenizer step in a copy of the start weights. A warm
start loads the saved ``policy_preprocessor.json``, so ``--policy.tokenizer_max_length`` alone
leaves the start's 200. The file is replaced, never written through, so a symlink in the copy
keeps the file it names.

``train`` runs ``train_pi05.py`` from this file's folder as its ``__main__`` with the arguments
unchanged. Before the first optimizer step it stops the run with ``PREFLIGHT_FAIL <what>``
(exit 3) unless:

- the weights loaded in full: lerobot 0.6.1 only prints a warning on a failed or partial load
  and trains on (PI05Policy.from_pretrained), and prints "All keys loaded successfully!" on a
  full one. Under ``--lora`` (peft) the check covers the base load: the adapter is wrapped on
  after it, in ``wrap_with_peft``, and prints nothing of its own;
- the action and state statistics of the normalizer and the unnormalizer hash-equal those of
  ``PREFLIGHT_STATS`` (the corpus ``meta/stats.json``): a warm start and a resume take them
  from the dataset root (lerobot_train.py). The image statistics are left out: the
  images are not normalized, and ``use_imagenet_stats`` replaces them;
- the tokenizer step pads to ``PREFLIGHT_MAX_LENGTH``, and the longest prompt of the run stays
  under it: lerobot truncates a longer one silently, "Action:" first. The prompts are
  ``train_pi05.TASK``, or with ``--prompt_codes`` the 12 of ``train_pi05.prompt``, each with a
  state of three-digit bins (the longest of [-1, 255]). 2026-10-13, the cached paligemma
  tokenizer: TASK 114 tokens, Run A's longest 124, with BOS;
- the forward that trains is not lerobot's stock one, and its first batch carries
  ``action_is_pad`` of shape (batch, chunk).

It prints one ``PREFLIGHT_OK ...`` line at the first forward. The terminal token is
TRAIN_OK, TRAIN_FAIL or PREFLIGHT_FAIL (PATCH_OK or PATCH_FAIL for ``patch``).
Runs in the pi venv, or on Colab with lerobot[pi]==0.6.1.
"""

import contextlib
import hashlib
import io
import json
import os
import runpy
import sys
import traceback

import numpy as np

FULL_LOAD = "All keys loaded successfully!"
CHECKED = ("action", "observation.state")  # the QUANTILES features; VISUAL is IDENTITY
TOKENIZER = "tokenizer_processor"  # the registry name of lerobot's prompt tokenizer step
FAIL_CODE = 3


class PreflightError(Exception):
    pass


def stats_sha(flat: dict) -> str:
    """sha256 of the action and state entries of a flat {"feature.stat": values} dict, as
    float32: the dtype the normalizer holds them in."""
    digest = hashlib.sha256()
    for key in sorted(k for k in flat if k.rsplit(".", 1)[0] in CHECKED):
        digest.update(key.encode())
        digest.update(np.asarray(flat[key], dtype=np.float32).tobytes())
    return digest.hexdigest()


def file_stats(path: str) -> dict:
    with open(path) as f:
        stats = json.load(f)
    return {f"{key}.{name}": value for key, sub in stats.items() for name, value in sub.items()}


def patch_max_length(model_dir: str, max_length: int) -> int:
    """Set the tokenizer step's max_length in ``model_dir``/policy_preprocessor.json; return
    the old one. Its padding stays "max_length", so the prompt pads to the new length."""
    path = os.path.join(model_dir, "policy_preprocessor.json")
    with open(path) as f:
        doc = json.load(f)
    steps = [s for s in doc["steps"] if s["registry_name"] == TOKENIZER]
    if len(steps) != 1:
        raise ValueError(f"{path} has {len(steps)} {TOKENIZER} steps, want 1")
    old = steps[0]["config"]["max_length"]
    steps[0]["config"]["max_length"] = max_length
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(path + ".tmp", path)
    return old


def longest_prompt(step, tasks: list[str], state_dim: int) -> int:
    """Tokens of the longest of ``tasks`` in lerobot's pi05 prompt (processor_pi05.py), BOS
    included, through the tokenizer of ``step``, the state at three-digit bins."""
    state = " ".join(["255"] * state_dim)
    return max(
        len(step.input_tokenizer(f"Task: {t}, State: {state};\nAction: ")["input_ids"])
        for t in tasks
    )


def install(train_module, policy_cls, stats: dict, max_length: int, tasks: list[str]) -> None:
    """Wrap ``policy_cls.from_pretrained`` and ``train_module.make_pre_post_processors``; ``stats``
    is the flat {"feature.stat": values} of the corpus file (``file_stats``) and ``tasks`` the
    run's task strings. The checks run when the processors are built, and the forward check at
    the first forward."""
    loads = []
    load = policy_cls.from_pretrained
    stock = policy_cls.forward

    def checked_load(*args, **kwargs):
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                return load(*args, **kwargs)
        finally:
            print(out.getvalue(), end="", flush=True)
            loads.append(FULL_LOAD in out.getvalue().splitlines())

    make = train_module.make_pre_post_processors

    def checked_make(*args, **kwargs):
        pre, post = make(*args, **kwargs)
        if loads != [True]:
            raise PreflightError(f"the weights did not load in full (loads: {loads})")
        want = stats_sha(stats)
        norm = _step(pre, "NormalizerProcessorStep").state_dict()
        unnorm = _step(post, "UnnormalizerProcessorStep").state_dict()
        for name, got in (("normalizer", stats_sha(norm)), ("unnormalizer", stats_sha(unnorm))):
            if got != want:
                raise PreflightError(f"{name} stats sha {got[:12]} != corpus {want[:12]}")
        tokenizer = _step(pre, "TokenizerProcessorStep")
        tokens = tokenizer.max_length
        if tokens != max_length:
            raise PreflightError(f"the tokenizer pads to {tokens}, want {max_length}")
        state_dim = np.asarray(stats["observation.state.mean"]).size
        longest = longest_prompt(tokenizer, tasks, state_dim)
        if longest >= tokens:
            raise PreflightError(f"the longest prompt takes {longest} of {tokens} tokens")
        masked = policy_cls.forward
        if masked is stock:
            raise PreflightError("the policy trains with lerobot's stock forward (no mask)")
        first = []

        def checked_forward(self, batch, *a, **k):
            if not first:
                first.append(1)
                pad = batch.get("action_is_pad")
                want_shape = (batch["action"].shape[0], self.config.chunk_size)
                if pad is None or str(pad.dtype) != "torch.bool" or tuple(pad.shape) != want_shape:
                    got = None if pad is None else (str(pad.dtype), tuple(pad.shape))
                    raise PreflightError(f"action_is_pad {got} at the forward, want {want_shape}")
                print(
                    f"PREFLIGHT_OK full_load=1 stats_sha={want[:12]} max_length={tokens}"
                    f" longest={longest}"
                    f" action_is_pad={want_shape} pad_share={float(pad.float().mean()):.4f}",
                    flush=True,
                )
            return masked(self, batch, *a, **k)

        policy_cls.forward = checked_forward
        return pre, post

    policy_cls.from_pretrained = staticmethod(checked_load)
    train_module.make_pre_post_processors = checked_make


def _step(pipeline, name: str):
    found = [s for s in pipeline.steps if type(s).__name__ == name]
    if len(found) != 1:
        raise PreflightError(f"{len(found)} {name} steps in the pipeline, want 1")
    return found[0]


def train(argv: list[str]) -> None:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    from train_pi05 import PHRASES, PROMPTS_FLAG, TASK, prompt  # this file's folder

    stats = file_stats(os.environ["PREFLIGHT_STATS"])
    tasks = [prompt(c) for c in range(3 * len(PHRASES))] if PROMPTS_FLAG in argv else [TASK]
    install(lerobot_train, PI05Policy, stats, int(os.environ["PREFLIGHT_MAX_LENGTH"]), tasks)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pi05.py")
    sys.argv = [script, *argv]
    runpy.run_path(script, run_name="__main__")


def main(argv: list[str]) -> int:
    mode = argv[0] if argv else None
    ok, bad = ("PATCH_OK", "PATCH_FAIL") if mode == "patch" else ("TRAIN_OK", "TRAIN_FAIL")
    try:
        if mode == "patch" and len(argv) == 3:
            old = patch_max_length(argv[1], int(argv[2]))
            print(f"{ok} {argv[1]} max_length {old} -> {argv[2]}", flush=True)
        elif mode == "train":
            train(argv[1:])
            print(ok, flush=True)
        else:
            print(f"{bad} usage: patch <dir> <max_length> | train <args>", flush=True)
            return 2
    except PreflightError as err:
        print(f"PREFLIGHT_FAIL {err}", flush=True)
        return FAIL_CODE
    except BaseException as exc:  # noqa: BLE001 -- any failure must become the fail token
        traceback.print_exc()
        print(f"{bad} {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    os._exit(main(sys.argv[1:]))

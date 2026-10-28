"""lerobot-train for pi0.5 with the loss masked on padded actions, as ACT's loss is.

LeRobot's ``PI05Policy.forward`` averages the flow-matching loss over every action of the
chunk, the padded ones past the episode end (``action_is_pad``) included, and LeRobot pads a
chunk by repeating the episode's last action. ACT's loss multiplies by ``~action_is_pad``. In the
MS7h derived dataset 74.6 % of the episodes end parked at the desk-box retreat point, so the
unmasked loss taught a still chunk there, while the next episode's head taught a leave from the
same observation: V at step 30k parked there with open jaws in 28 % of its cycles (MS7h ledger,
2026-10-07). This runs lerobot-train with ``PI05Policy.forward`` masked.

Run it like lerobot-train, from the pi venv: ``python train_pi05.py --policy.type=pi05 ...``.
It refuses another lerobot version, because ``forward`` is a copy of 0.6.1's. ``--lora=R`` expands
to the ``--peft.*`` flags of ``lora_flags(R)`` (spec section 6.3, B3).

``--mix`` and ``--prompt_codes`` exit on 0.6.1: its ``EpisodeAwareSampler`` draws from
``meta.episodes``, so ``Mixed`` never samples its extra roots. The code stays for reproducibility.

``--mix=ROOT:SHARE`` (0.5.1 only), once a source (the MS7h DAgger round HD59 with one, MS7i Run A and Run B
with two), trains on the corpus (``--dataset.root``) and each ROOT in one mix (``Mixed``): a draw
is a frame of ROOT with probability SHARE and a corpus frame with 1 minus their sum, and the
corpus's metadata, so its normalizer, serves every source. ``--prompt_codes`` (MS7i Run A) reads
each source's ``meta/prompt_codes.npy``, one code a frame by its index, and gives each frame the
task string ``prompt(code)`` of MS7i design rev1 5.2: TASK, the subtask phrase and the advantage
sentence of its tag, which a tagged frame leaves out at the rate DROP. Without the flags nothing
changes.

The MS7k stage 1 levers (spec section 6.2, ledger K11). Each one is off unless its flag is given.
With none of them, main runs exactly as before (ledger K14).

- ``--lora_from=DIR`` continues a LoRA save with a fresh optimizer and schedule (no --resume).
  DIR is the save's ``pretrained_model`` folder. Give it with ``--lora=R`` and with
  ``--policy.pretrained_path`` set to the base, as for a new run. The base then loads through
  lerobot's own path, so the preflight still reads its full-load line. ``wrap_with_peft`` wraps
  a new adapter, and ``load_lora`` copies the saved values into it, the modules_to_save included.
  At step 0 the model is the save's model. ``load_lora`` refuses a save with another rank or
  alpha, a saved tensor that finds no place, and an adapter tensor that stays at its init.
  lerobot's own continuation (``--policy.pretrained_path=DIR --policy.use_peft=true``) has none
  of these checks: PEFT warns only on a missing LoRA tensor, ignores a tensor that finds no
  place, and does not report a missing modules_to_save tensor. It also loads the base from the
  absolute path in DIR's adapter_config.json, which does not exist on Colab.
- ``--siglip_lora`` (L2) adds every SigLIP layer's linears (``SIGLIP_TARGETS``: attention q, k,
  v, out and MLP fc1, fc2, 27 layers, 162 modules) to the ``--lora`` targets. With
  ``--lora_from`` their LoRA keeps PEFT's init (B = 0), so the model at step 0 is still the
  save's. embed_image runs under gradient checkpointing, so the backward pass recomputes the
  SigLIP activations of one camera at a time. The memory probe decides if L2 fits.
  The fallback of ledger K11 needs no code. In 0.6.1, x_t = t * noise + (1 - t) * actions, so
  t = 1 is pure noise and t = 0 is the data, and sampling runs Euler steps from t = 1 to t = 0.
  The time sampler draws t = Beta(alpha, beta) * 0.999 + 0.001, alpha 1.5 and beta 1.0 by
  default: mean 0.6, about 9 % of the draws under t = 0.2. Low noise is small t, so the fallback
  swaps the two: ``--policy.time_sampling_beta_alpha=1.0 --policy.time_sampling_beta_beta=1.5``
  (mean 0.4, about 28 % of the draws under t = 0.2 and 9 % over t = 0.8).
- ``--close_weight=W,BEFORE_S,AFTER_S`` (L1) replaces lerobot's sampler with a seeded
  ``WeightedRandomSampler`` (with replacement, one epoch = the dataset's length, seed = lerobot's
  --seed). Each frame from BEFORE_S seconds before to AFTER_S seconds after a close command, in
  the same episode, has weight W, and every other frame has weight 1 (``close_weights``).
- ``--overhead_aug=P,SHIFT,ROLL_DEG`` (L3) gives a share P of the training samples a random
  shift (up to SHIFT of the frame size on each axis) and roll (up to ROLL_DEG degrees about the
  image center) of the overhead image only (``random_affine``). The wrist image is untouched.
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT = "0.6.1"
LORA_TARGETS = (
    r"model\.paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)"
    r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
)
LORA_FULL = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
LORA_FLAG = "--lora="
MIX_FLAG, PROMPTS_FLAG = "--mix=", "--prompt_codes"
TASK = "pick up the marked part and place it in its box"  # derive_vla_dataset imports it
PHRASES = ("reach the marked part", "close on the part", "carry it to its box", "release it")
"""Run A's subtask phrases (MS7i design rev1 5.2) in the order of core.hold.SUBTASKS (test pins)."""
TAGS = ("", " Advantage: positive", " Advantage: negative")  # by code % 3: untagged, +, -
POSITIVE = 1
DROP = 0.2  # MS7i design rev3 6.1: a tagged frame leaves its advantage sentence out at this rate
CODES = "meta/prompt_codes.npy"  # int8 a frame of a root, by its index: 3 * subtask + tag
LEVERS = ("--lora_from", "--siglip_lora", "--close_weight", "--overhead_aug")
SIGLIP_TARGETS = (
    r"model\.paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder"
    r"\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc(1|2))"
)
JAW_FULL_GAP = 2 * 0.0524125  # the full-open command of the continuous gripper (test pins it)
# A close command: the commanded gap falls by at least CLOSE_DROP from below full open. In the V
# corpus every such fall is 14.0 mm (11.2 mm where the gap stops at 0). A fall from full open is
# the jaw shaping to the part's pick opening, not a close.
CLOSE_DROP = 0.005
OVERHEAD = "observation.images.overhead"


def prompt(code: int, drop: bool = False) -> str:
    """Run A's task string of a prompt code. The server serves prompt(3 * subtask + POSITIVE)."""
    subtask, tag = divmod(int(code), 3)
    return f"{TASK}. {PHRASES[subtask]}." + ("" if drop else TAGS[tag])


def lora_flags(rank: int, siglip: bool = False) -> list[str]:
    """The lerobot-train flags of the LoRA recipe at ``rank``, alpha equal to it; with
    ``siglip``, the SigLIP linears are targets too (L2)."""
    targets = f"{LORA_TARGETS}|{SIGLIP_TARGETS}" if siglip else LORA_TARGETS
    return [
        "--peft.method_type=LORA",
        f"--peft.r={rank}",
        f"--peft.lora_alpha={rank}",
        f"--peft.target_modules={targets}",
        f"--peft.full_training_modules={json.dumps(list(LORA_FULL))}",
    ]


def load_lora(model, adapter_dir, new: str | None = None) -> None:
    """Copy the LoRA save in ``adapter_dir`` into the PEFT ``model`` (``--lora_from``). Every
    saved tensor must find its place, and every trainable tensor of ``model`` must load, except
    those that the regex ``new`` finds in their name: they keep PEFT's init (B = 0)."""
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    saved = json.loads((adapter_dir / "adapter_config.json").read_text())
    peft = model.peft_config["default"]
    # A saved modules_to_save tensor that the run does not save would load into the frozen base.
    have = (saved["r"], saved["lora_alpha"], sorted(saved["modules_to_save"] or []))
    want = (peft.r, peft.lora_alpha, sorted(peft.modules_to_save or []))
    if have != want:
        raise ValueError(f"{adapter_dir}: r, alpha, modules_to_save {have}, the run has {want}")
    weights = load_file(adapter_dir / "adapter_model.safetensors")
    result = set_peft_model_state_dict(model, weights)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = [k for k in result.missing_keys if k in trainable]
    unloaded = [k for k in missing if not (new and re.search(new, k))]
    if result.unexpected_keys or unloaded:
        raise ValueError(
            f"{adapter_dir}: {len(result.unexpected_keys)} saved tensors found no place "
            f"{result.unexpected_keys[:2]}, {len(unloaded)} adapter tensors did not load "
            f"{unloaded[:2]}"
        )
    print(
        f"LORA_FROM {adapter_dir} loaded={len(weights)} new={len(missing)}"
        f" trainable={len(trainable)}",
        flush=True,
    )


def continued(wrap, adapter_dir, new: str | None):
    """``wrap_with_peft`` that then loads the LoRA save in ``adapter_dir`` (``load_lora``)."""

    def wrap_with_peft(self, *args, **kwargs):
        model = wrap(self, *args, **kwargs)
        load_lora(model, adapter_dir, new)
        return model

    return wrap_with_peft


def close_weights(gap, episode, fps: float, weight: float, before_s: float, after_s: float):
    """(weights, close count, episodes with no close) of the frames whose commanded jaw gap is
    ``gap`` and episode ``episode`` (both (N,), in frame order). A close command is a frame whose
    gap falls by CLOSE_DROP or more from a previous frame of its episode below full open. The
    frames from ``before_s`` seconds before to ``after_s`` seconds after a close, in its episode,
    weigh ``weight``, and the others 1. An episode that starts closed shows no close."""
    gap, episode = np.asarray(gap, np.float64), np.asarray(episode)
    n = len(gap)
    first = np.r_[True, episode[1:] != episode[:-1]]
    starts = np.flatnonzero(first)
    ep = np.cumsum(first) - 1  # the episode's rank, frame by frame
    last = np.r_[starts[1:], n] - 1
    prev = np.r_[np.inf, gap[:-1]]
    close = np.flatnonzero(~first & (prev - gap >= CLOSE_DROP) & (prev < JAW_FULL_GAP - CLOSE_DROP))
    lo = np.maximum(close - round(before_s * fps), starts[ep[close]])
    hi = np.minimum(close + round(after_s * fps), last[ep[close]])
    edges = np.zeros(n + 1, np.int64)
    np.add.at(edges, lo, 1)
    np.add.at(edges, hi + 1, -1)
    near = np.cumsum(edges[:-1]) > 0
    none = len(starts) - len(np.unique(ep[close]))
    return np.where(near, weight, 1.0), len(close), none


def corpus_gap(root) -> tuple:
    """(commanded jaw gap, episode index) of every frame of the LeRobot root ``root``, in index
    order, read from its parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    names = json.loads((root / "meta/info.json").read_text())["features"]["action"]["names"]
    files = sorted((root / "data").glob("*/*.parquet"))
    table = pa.concat_tables(
        pq.read_table(f, columns=["index", "episode_index", "action"]) for f in files
    )
    if not np.array_equal(table["index"].to_numpy(), np.arange(len(table))):
        raise ValueError(f"{root}: the frame index is not 0 to N - 1 in file order")
    action = table["action"].combine_chunks()
    action = np.asarray(action.flatten()).reshape(len(action), -1)
    return action[:, names.index("gripper_gap_m")], table["episode_index"].to_numpy()


def close_sampling(train_module, weight: float, before_s: float, after_s: float) -> None:
    """L1: ``train_module`` (lerobot_train) builds its datasets, then the weights of
    ``close_weights``, and samples through a ``WeightedRandomSampler`` in place of its
    ``EpisodeAwareSampler``, seeded with the seed lerobot gives the sampler (--seed)."""
    make = train_module.make_train_eval_datasets

    def make_datasets(cfg):
        dataset, eval_dataset = make(cfg)
        start = time.perf_counter()
        gap, episode = corpus_gap(dataset.root)
        weights, closes, none = close_weights(gap, episode, dataset.fps, weight, before_s, after_s)
        if len(weights) != len(dataset):
            raise ValueError(f"{len(weights)} weights for {len(dataset)} frames: a subset?")
        near = weights != 1.0
        share = weights[near].sum() / weights.sum()
        print(
            f"CLOSE_WEIGHT closes={closes} episodes_without={none} frames={len(weights)}"
            f" window_share={near.mean():.4f} weighted_share={share:.4f}"
            f" seconds={time.perf_counter() - start:.1f}",
            flush=True,
        )
        weights = torch.as_tensor(weights)

        def sampler(*args, seed=0, **kwargs):
            gen = torch.Generator().manual_seed(seed)
            return torch.utils.data.WeightedRandomSampler(
                weights, len(weights), replacement=True, generator=gen
            )

        train_module.EpisodeAwareSampler = sampler
        return dataset, eval_dataset

    train_module.make_train_eval_datasets = make_datasets


def random_affine(img, share: float, shift: float, roll_deg: float, gen):
    """L3 on a (B, C, H, W) batch: a share ``share`` of the samples, drawn from ``gen``, rolls
    up to ``roll_deg`` degrees about the image center and shifts up to ``shift`` of the frame
    size on each axis. The border pixels are replicated: a moved camera sees more desk, never a
    black band that no real frame shows. The other samples come back unchanged."""
    b, _, h, w = img.shape
    u = torch.rand(b, 4, generator=gen, device=img.device)
    on = u[:, 0] < share
    roll = (2 * u[:, 1] - 1) * math.radians(roll_deg)
    move = (2 * u[:, 2:] - 1) * 2 * shift  # affine_grid spans the frame from -1 to 1
    cos, sin = roll.cos(), roll.sin()
    theta = torch.stack(  # a rotation in pixels, written in the grid's [-1, 1] units
        [
            torch.stack([cos, -sin * h / w, move[:, 0]], 1),
            torch.stack([sin * w / h, cos, move[:, 1]], 1),
        ],
        1,
    ).to(img.dtype)
    grid = torch.nn.functional.affine_grid(theta, img.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(img, grid, padding_mode="border", align_corners=False)
    return torch.where(on[:, None, None, None], out, img)


def overhead_aug(forward, share: float, shift: float, roll_deg: float):
    """``forward`` with L3 (``random_affine``) on the overhead image, in training mode only. It
    draws from its own generator, seeded at the first call with torch.initial_seed(): lerobot's
    --seed, as set_seed runs before the first step. The noise and time draws stay as without it."""
    gen = None

    def augmented(self, batch: dict, reduction: str = "mean") -> tuple:
        nonlocal gen
        if self.training:
            img = batch[OVERHEAD]
            if gen is None:
                gen = torch.Generator(img.device).manual_seed(torch.initial_seed())
            batch = {**batch, OVERHEAD: random_affine(img, share, shift, roll_deg, gen)}
        return forward(self, batch, reduction)

    return augmented


def split_levers(argv: list[str]) -> tuple[dict, list[str]]:
    """({flag: value} of the LEVERS flags, argv without them). A flag with no value maps to ""."""
    levers, rest = {}, []
    for a in argv:
        name, _, value = a.partition("=")
        if name in LEVERS:
            levers[name] = value
        else:
            rest.append(a)
    return levers, rest


def lever_values(levers: dict, flag: str, test) -> tuple | None:
    """The three floats of ``flag``, or None without it; SystemExit unless ``test(*values)``."""
    if flag not in levers:
        return None
    try:
        values = tuple(float(v) for v in levers[flag].split(","))
    except ValueError:
        values = ()
    if len(values) != 3 or not test(*values):
        raise SystemExit(f"{flag}={levers[flag]}: want three numbers in range (train_pi05.py)")
    return values


def masked_means(losses: torch.Tensor, is_pad: torch.Tensor) -> tuple:
    """(mean, (B,) per-sample means, (D,) per-dim means) of the (B, T, D) ``losses`` over the
    actions that are not padding ((B, T) ``is_pad``)."""
    keep = (~is_pad).unsqueeze(-1).to(losses.dtype)
    kept = losses * keep
    dims = losses.shape[-1]
    per_sample = kept.sum((1, 2)) / (keep.sum((1, 2)) * dims).clamp_min(1)
    per_dim = kept.sum((0, 1)) / keep.sum().clamp_min(1)
    return kept.sum() / (keep.sum() * dims).clamp_min(1), per_sample, per_dim


def forward(self, batch: dict, reduction: str = "mean") -> tuple:
    """``PI05Policy.forward`` of lerobot 0.6.1 with its loss reduced by ``masked_means``."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = self._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = self.prepare_action(batch)
    noise = self.model.sample_noise(actions.shape, actions.device)  # noise, then time: 0.5.1's
    time = self.model.sample_time(actions.shape[0], actions.device)  # RNG order
    losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)
    losses = losses[:, :, : self.config.output_features[ACTION].shape[0]]
    loss, per_sample, per_dim = masked_means(losses, batch["action_is_pad"])
    loss_dict = {"loss_per_dim": per_dim.detach().cpu().numpy().tolist(), "loss": loss.item()}
    return (per_sample, loss_dict) if reduction == "none" else (loss, loss_dict)


class Mixed(torch.utils.data.Dataset):
    """The corpus, then for each (dataset, share) of ``extras`` a slot of indices for its frames:
    lerobot's shuffled sampler draws every index once an epoch, so a draw is that source's frame
    with probability share. A slot cycles through a source that fits it, so every frame is drawn;
    a source larger than its slot fills it with a fixed random subset of its frames (numpy
    default_rng(0), so a resume and every loader worker map alike), never only its first ones.
    ``meta`` is the corpus's: the policy features and the normalizer statistics. With ``codes``
    (one array a source, corpus first) each frame's task is ``prompt(code)``, and a tagged frame
    drops its advantage sentence with probability DROP, drawn per sample in the loader worker
    (torch seeds each worker from lerobot's --seed)."""

    def __init__(self, corpus, extras: list, codes: list | None = None):
        rest = 1 - sum(s for _, s in extras)
        if not extras or min(s for _, s in extras) <= 0 or not 0 < rest < 1:
            raise ValueError(
                f"mix shares must be > 0 and sum under 1, got {[s for _, s in extras]}"
            )
        self.meta, self.codes = corpus.meta, codes
        self.sources = [corpus, *(d for d, _ in extras)]
        self.sizes = [len(corpus), *(round(len(corpus) * s / rest) for _, s in extras)]
        self.maps = [
            None if size >= len(d) else np.random.default_rng(0).permutation(len(d))[:size]
            for d, size in zip(self.sources, self.sizes)
        ]
        self.num_frames = len(self)
        self.num_episodes = sum(d.num_episodes for d in self.sources)

    def __len__(self) -> int:
        return sum(self.sizes)

    def __getitem__(self, i: int) -> dict:
        k = 0
        while i >= self.sizes[k]:
            i, k = i - self.sizes[k], k + 1
        source, subset = self.sources[k], self.maps[k]
        item = source[i % len(source) if subset is None else int(subset[i])]
        if self.codes is not None:
            code = int(self.codes[k][int(item["index"])])
            item["task"] = prompt(code, code % 3 != 0 and float(torch.rand(())) < DROP)
        return item


def root_datasets(make, cfg, corpus, roots: list) -> list:
    """The datasets ``make`` (lerobot's ``make_dataset``) builds at ``roots``, ``cfg``'s own
    root put back after; one with other features or fps than ``corpus`` is refused. train_masked
    imports it too."""
    main_root, data = cfg.dataset.root, []
    try:
        for root in roots:
            cfg.dataset.root = root
            data.append(make(cfg))
    finally:
        cfg.dataset.root = main_root
    for root, d in zip(roots, data):
        if d.meta.features != corpus.meta.features or d.fps != corpus.fps:
            raise ValueError(f"the mix root {root} has other features or fps than the corpus")
    return data


def mixed_maker(make, mix: list, prompts: bool):
    """``make`` (lerobot's ``make_dataset``) for the corpus and for each (root, share) of
    ``mix``, as one ``Mixed``; with ``prompts``, each source's prompt codes too."""

    def make_dataset(cfg):
        corpus = make(cfg)
        data = root_datasets(make, cfg, corpus, [root for root, _ in mix])
        extras = [(d, share) for d, (_, share) in zip(data, mix)]
        sources = [corpus, *data]
        codes = [load_codes(d) for d in sources] if prompts else None
        out = Mixed(corpus, extras, codes)
        print(
            f"MIX shares={','.join(str(s) for _, s in mix)} frames="
            f"{','.join(str(len(d)) for d in sources)} sizes={','.join(map(str, out.sizes))}"
            f" prompts={int(prompts)} roots={','.join(r for r, _ in mix)}",
            flush=True,
        )
        return out

    return make_dataset


def load_codes(dataset) -> np.ndarray:
    """The int8 prompt codes of ``dataset``'s root, one a frame, each in [0, 3 * len(PHRASES))."""
    path = Path(dataset.root) / CODES
    codes = np.load(path)
    if codes.shape != (dataset.num_frames,) or codes.dtype != np.int8:
        raise ValueError(f"{path}: {codes.dtype} {codes.shape}, want int8 ({dataset.num_frames},)")
    if len(codes) and not 0 <= codes.min() <= codes.max() < 3 * len(PHRASES):
        raise ValueError(f"{path}: codes span [{codes.min()}, {codes.max()}]")
    return codes


def split_mix(argv: list[str]) -> tuple[list, bool, list[str]]:
    """([(root, share)] of the --mix= flags, --prompt_codes given, argv without them)."""
    mix = []
    for a in argv:
        if a.startswith(MIX_FLAG):
            root, _, share = a[len(MIX_FLAG) :].rpartition(":")
            if not root:
                raise SystemExit(f"{a}: want --mix=ROOT:SHARE")
            mix.append((root, float(share)))
    prompts = PROMPTS_FLAG in argv
    if prompts and not mix:
        raise SystemExit(f"{PROMPTS_FLAG} needs --mix")
    rest = [a for a in argv if not a.startswith(MIX_FLAG) and a != PROMPTS_FLAG]
    return mix, prompts, rest


def main(argv: list[str]) -> None:
    import lerobot

    if lerobot.__version__ != LEROBOT:
        raise SystemExit(
            f"train_pi05.py copies lerobot {LEROBOT}'s forward, found {lerobot.__version__}"
        )
    if any(a.startswith(MIX_FLAG) or a == PROMPTS_FLAG for a in argv):
        raise SystemExit(
            f"{MIX_FLAG}/{PROMPTS_FLAG} are not supported on lerobot {LEROBOT}: its "
            "EpisodeAwareSampler draws from meta.episodes, so Mixed never samples its extra roots"
        )
    levers, argv = split_levers(argv)
    close = lever_values(levers, "--close_weight", lambda w, b, a: w > 0 and b >= 0 and a >= 0)
    aug = lever_values(levers, "--overhead_aug", lambda p, s, r: 0 <= p <= 1 and min(s, r) >= 0)
    lora_from, siglip = levers.get("--lora_from"), "--siglip_lora" in levers
    if levers.get("--siglip_lora", "") or lora_from == "":
        raise SystemExit("--siglip_lora takes no value, --lora_from needs one")
    if (lora_from or siglip) and not any(a.startswith(LORA_FLAG) for a in argv):
        raise SystemExit("--lora_from and --siglip_lora need --lora=R")
    if lora_from and any(a.startswith(("--policy.use_peft", "--resume")) for a in argv):
        raise SystemExit("--lora_from starts a new run on the base: no --policy.use_peft, --resume")
    if levers:
        print(f"LEVERS {json.dumps(levers, sort_keys=True)}", flush=True)
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    argv = [
        x
        for a in argv
        for x in (lora_flags(int(a[len(LORA_FLAG) :]), siglip) if a.startswith(LORA_FLAG) else [a])
    ]
    PI05Policy.forward = forward if aug is None else overhead_aug(forward, *aug)
    if lora_from:
        new = SIGLIP_TARGETS if siglip else None
        PI05Policy.wrap_with_peft = continued(PI05Policy.wrap_with_peft, lora_from, new)
    if close:
        close_sampling(lerobot_train, *close)
    sys.argv = [sys.argv[0], *argv]
    lerobot_train.main()


if __name__ == "__main__":
    main(sys.argv[1:])

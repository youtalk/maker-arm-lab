"""lerobot-train for pi0.5 with the loss masked on padded actions, as ACT's loss is.

LeRobot 0.5.1's ``PI05Policy.forward`` averages the flow-matching loss over every action of the
chunk, the padded ones past the episode end (``action_is_pad``) included, and LeRobot pads a
chunk by repeating the episode's last action. ACT's loss multiplies by ``~action_is_pad``. In the
MS7h derived dataset 74.6 % of the episodes end parked at the desk-box retreat point, so the
unmasked loss taught a still chunk there, while the next episode's head taught a leave from the
same observation: V at step 30k parked there with open jaws in 28 % of its cycles (MS7h ledger,
2026-10-07). This runs lerobot-train with ``PI05Policy.forward`` masked.

Run it like lerobot-train, from the pi venv: ``python train_pi05.py --policy.type=pi05 ...``.
It refuses another lerobot version, because ``forward`` is a copy of 0.5.1's.

``--mix=ROOT:SHARE``, once a source (the MS7h DAgger round HD59 with one, MS7i Run A and Run B
with two), trains on the corpus (``--dataset.root``) and each ROOT in one mix (``Mixed``): a draw
is a frame of ROOT with probability SHARE and a corpus frame with 1 minus their sum, and the
corpus's metadata, so its normalizer, serves every source. ``--prompt_codes`` (MS7i Run A) reads
each source's ``meta/prompt_codes.npy``, one code a frame by its index, and gives each frame the
task string ``prompt(code)`` of MS7i design rev1 5.2: TASK, the subtask phrase and the advantage
sentence of its tag, which a tagged frame leaves out at the rate DROP. Without the flags nothing
changes.
"""

import sys
from pathlib import Path

import numpy as np
import torch

LEROBOT = "0.5.1"
MIX_FLAG, PROMPTS_FLAG = "--mix=", "--prompt_codes"
TASK = "pick up the marked part and place it in its box"  # derive_vla_dataset.TASK (test pins)
PHRASES = ("reach the marked part", "close on the part", "carry it to its box", "release it")
"""Run A's subtask phrases (MS7i design rev1 5.2) in the order of core.hold.SUBTASKS (test pins)."""
TAGS = ("", " Advantage: positive", " Advantage: negative")  # by code % 3: untagged, +, -
POSITIVE = 1
DROP = 0.2  # MS7i design rev3 6.1: a tagged frame leaves its advantage sentence out at this rate
CODES = "meta/prompt_codes.npy"  # int8 a frame of a root, by its index: 3 * subtask + tag


def prompt(code: int, drop: bool = False) -> str:
    """Run A's task string of a prompt code. The server serves prompt(3 * subtask + POSITIVE)."""
    subtask, tag = divmod(int(code), 3)
    return f"{TASK}. {PHRASES[subtask]}." + ("" if drop else TAGS[tag])


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
    """``PI05Policy.forward`` of lerobot 0.5.1 with its loss reduced by ``masked_means``."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = self._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    losses = self.model.forward(images, img_masks, tokens, masks, self.prepare_action(batch))
    losses = losses[:, :, : self.config.output_features[ACTION].shape[0]]
    loss, per_sample, per_dim = masked_means(losses, batch["action_is_pad"])
    loss_dict = {"loss_per_dim": per_dim.detach().cpu().numpy().tolist(), "loss": loss.item()}
    return (per_sample, loss_dict) if reduction == "none" else (loss, loss_dict)


class Mixed(torch.utils.data.Dataset):
    """The corpus, then for each (dataset, share) of ``extras`` the indices that spread over its
    frames: lerobot's shuffled sampler draws every index once an epoch, so a draw is that
    source's frame with probability share. ``meta`` is the corpus's: the policy features and the
    normalizer statistics. With ``codes`` (one array a source, corpus first) each frame's task is
    ``prompt(code)``, and a tagged frame drops its advantage sentence with probability DROP, drawn
    per sample in the loader worker (torch seeds each worker from lerobot's --seed)."""

    def __init__(self, corpus, extras: list, codes: list | None = None):
        rest = 1 - sum(s for _, s in extras)
        if not extras or min(s for _, s in extras) <= 0 or not 0 < rest < 1:
            raise ValueError(
                f"mix shares must be > 0 and sum under 1, got {[s for _, s in extras]}"
            )
        self.meta, self.codes = corpus.meta, codes
        self.sources = [corpus, *(d for d, _ in extras)]
        self.sizes = [len(corpus), *(round(len(corpus) * s / rest) for _, s in extras)]
        self.num_frames = len(self)
        self.num_episodes = sum(d.num_episodes for d in self.sources)

    def __len__(self) -> int:
        return sum(self.sizes)

    def __getitem__(self, i: int) -> dict:
        k = 0
        while i >= self.sizes[k]:
            i, k = i - self.sizes[k], k + 1
        source = self.sources[k]
        item = source[i * len(source) // self.sizes[k]]  # spread: no tail left undrawn
        if self.codes is not None:
            code = int(self.codes[k][int(item["index"])])
            item["task"] = prompt(code, code % 3 != 0 and float(torch.rand(())) < DROP)
        return item


def mixed_maker(make, mix: list, prompts: bool):
    """``make`` (lerobot's ``make_dataset``) for the corpus and for each (root, share) of
    ``mix``, as one ``Mixed``; with ``prompts``, each source's prompt codes too."""

    def make_dataset(cfg):
        corpus = make(cfg)
        main_root, extras = cfg.dataset.root, []
        try:
            for root, share in mix:
                cfg.dataset.root = root
                extras.append((make(cfg), share))
        finally:
            cfg.dataset.root = main_root
        for (ds, _), (root, _) in zip(extras, mix):
            if ds.meta.features != corpus.meta.features or ds.fps != corpus.fps:
                raise ValueError(f"the mix root {root} has other features or fps than the corpus")
        sources = [corpus, *(d for d, _ in extras)]
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
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    mix, prompts, argv = split_mix(argv)
    if mix:
        lerobot_train.make_dataset = mixed_maker(lerobot_train.make_dataset, mix, prompts)
    PI05Policy.forward = forward
    sys.argv = [sys.argv[0], *argv]
    lerobot_train.main()


if __name__ == "__main__":
    main(sys.argv[1:])

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

``--dagger_root=DIR [--dagger_share=0.3]`` (the MS7h DAgger round, HD59) trains on the corpus
(``--dataset.root``) and the DAgger dataset at DIR in one mix (``Mixed``): a draw is a DAgger
frame with probability ``share``, and the corpus's metadata, so its normalizer, serves both.
Without the flags nothing changes.
"""

import sys

import torch

LEROBOT = "0.5.1"
MIX_FLAGS = ("--dagger_root=", "--dagger_share=")


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
    """The corpus, then ``extra`` more indices that cycle through the DAgger frames. lerobot's
    shuffled sampler draws every index once an epoch, so a draw is a DAgger frame with probability
    ``share``. ``meta`` is the corpus's: the policy features and the normalizer statistics."""

    def __init__(self, corpus, dagger, share: float):
        if not 0 < share < 1:
            raise ValueError(f"--dagger_share must be in (0, 1), got {share}")
        self.corpus, self.dagger, self.meta = corpus, dagger, corpus.meta
        self.extra = round(len(corpus) * share / (1 - share))
        self.num_frames = len(self)
        self.num_episodes = corpus.num_episodes + dagger.num_episodes

    def __len__(self) -> int:
        return len(self.corpus) + self.extra

    def __getitem__(self, i: int) -> dict:
        n = len(self.corpus)
        return self.corpus[i] if i < n else self.dagger[(i - n) % len(self.dagger)]


def mixed_maker(make, root: str, share: float):
    """``make`` (lerobot's ``make_dataset``) for the corpus and for ``root``, as one ``Mixed``."""

    def make_dataset(cfg):
        corpus = make(cfg)
        main_root = cfg.dataset.root
        cfg.dataset.root = root
        try:
            dagger = make(cfg)
        finally:
            cfg.dataset.root = main_root
        if dagger.meta.features != corpus.meta.features or dagger.fps != corpus.fps:
            raise ValueError(f"the DAgger dataset {root} has other features or fps than the corpus")
        mix = Mixed(corpus, dagger, share)
        print(
            f"DAGGER_MIX share={share} corpus_frames={len(corpus)} dagger_frames={len(dagger)}"
            f" dagger_episodes={dagger.num_episodes} extra={mix.extra} root={root}",
            flush=True,
        )
        return mix

    return make_dataset


def split_mix(argv: list[str]) -> tuple[str | None, float, list[str]]:
    """(the DAgger root or None, its share, argv without the two flags)."""
    found = {f: a[len(f) :] for a in argv for f in MIX_FLAGS if a.startswith(f)}
    root, share = found.get(MIX_FLAGS[0]), found.get(MIX_FLAGS[1])
    if share is not None and root is None:
        raise SystemExit("--dagger_share needs --dagger_root")
    rest = [a for a in argv if not a.startswith(MIX_FLAGS)]
    return root, 0.3 if share is None else float(share), rest


def main(argv: list[str]) -> None:
    import lerobot

    if lerobot.__version__ != LEROBOT:
        raise SystemExit(
            f"train_pi05.py copies lerobot {LEROBOT}'s forward, found {lerobot.__version__}"
        )
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    root, share, argv = split_mix(argv)
    if root is not None:
        lerobot_train.make_dataset = mixed_maker(lerobot_train.make_dataset, root, share)
    PI05Policy.forward = forward
    sys.argv = [sys.argv[0], *argv]
    lerobot_train.main()


if __name__ == "__main__":
    main(sys.argv[1:])

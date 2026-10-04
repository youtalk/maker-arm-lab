"""lerobot-train with an optional frame mask (MS7g design 8, ledger G10).

``--frame_mask=PATH`` (a mask ``.npz`` of reopen_mask.py, built for this ``--dataset.root``)
takes its frames out of training as if the converter had cut the episode before each of them:
a masked frame is never a sample, and a kept frame's action chunk is padded (``action_is_pad``)
from the first masked frame it reaches. ACT's loss ignores padded targets, so no chunk learns a
masked action; dropping the samples alone would leave the 99 frames before each REOPEN
predicting it (chunk_size 100). The mask's episode lengths must equal the dataset's.

``--context_norm=roll_wrapped_norm`` (MS7g D12) trains on the context policy_server.py serves to
a checkpoint whose action_space.json carries ``"context": "roll_wrapped_norm"``: every sample's
camera roll becomes wrap(roll - pi) (``policy_server.wrap_context_roll``), the dataset's context
stats are replaced by the mean, std, min and max of that wrapped column over every frame of the
dataset (masked ones too, like LeRobot's own stats), and the policy maps ENV to MEAN_STD, so the
saved preprocessor normalizes the context at serving too. ``--context_proj`` says what a warm
start does with ``encoder_env_state_input_proj``, whose input changed meaning: ``keep`` its
weights, ``reinit`` them (nn.Linear's own init, which ACT leaves that layer at), or ``fold`` the
new normalization into them (W' = W diag(std), b' = b + W mean), so every dimension but the
roll starts out mapped as before.

Without the switches this is ``lerobot-train``, argument for argument. Run it from the lerobot
venv like lerobot-train: ``python train_masked.py --frame_mask=M.npz --policy.path=... ...``.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from policy_server import CONTEXT_NORM, CONTEXT_ROLL, wrap_context_roll

FLAG = "--frame_mask="
CONTEXT_FLAG = "--context_norm="
PROJ_FLAG = "--context_proj="
PROJ_MODES = ("keep", "reinit", "fold")
ENV = "observation.environment_state"
log = logging.getLogger(__name__)


class MaskedFrames(torch.utils.data.Dataset):
    """``dataset`` without the frames ``drop`` (absolute frame indices); every other attribute
    (``meta``, ``features``, ...) is the dataset's."""

    def __init__(self, dataset, drop: np.ndarray, transform=None):
        self.dataset = dataset
        self.transform = transform  # item -> item, after the chunk padding (None: none)
        self.dropped = np.zeros(dataset.num_frames, dtype=bool)
        self.dropped[drop] = True
        self.keep = np.flatnonzero(~self.dropped)
        self.num_frames = len(self.keep)
        # The action chunk's frame offsets, as LeRobot's get_delta_indices rounds them.
        seconds = (dataset.delta_timestamps or {}).get("action", [])
        self.deltas = np.rint(np.asarray(seconds) * dataset.meta.fps).astype(np.int64)

    def __getattr__(self, name):
        if name == "dataset":  # not set yet (unpickling): never recurse
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, i: int) -> dict:
        item = self.dataset[int(self.keep[i])]
        if len(self.deltas):
            chunk = np.clip(int(item["index"]) + self.deltas, 0, len(self.dropped) - 1)
            hit = np.flatnonzero(self.dropped[chunk])
            if len(hit):
                item["action_is_pad"][hit[0] :] = True
        return item if self.transform is None else self.transform(item)


def wrap_item_roll(item: dict) -> dict:
    item[ENV] = wrap_context_roll(item[ENV])
    return item


def context_stats(root, width: int) -> dict:
    """LeRobot-shaped stats of the wrapped context column over every frame under ``root``."""
    import pyarrow.parquet as pq

    columns = [
        pq.read_table(f, columns=[ENV])[ENV].combine_chunks()
        for f in sorted(Path(root).glob("data/*/*.parquet"))
    ]
    flat = np.concatenate([np.asarray(c.values, dtype=np.float32) for c in columns])
    ctx = wrap_context_roll(flat.reshape(-1, width)).astype(np.float64)
    stats = {"mean": ctx.mean(0), "std": ctx.std(0), "min": ctx.min(0), "max": ctx.max(0)}
    return {**{k: v.astype(np.float32) for k, v in stats.items()}, "count": np.array([len(ctx)])}


def masked(make_dataset, path: str | None, context: str | None = None):
    """``make_dataset`` with its dataset wrapped in the mask at ``path`` (None: no mask) and,
    with ``context``, its context wrapped and its context stats replaced."""

    def make(cfg):
        dataset = make_dataset(cfg)
        drop = np.zeros(0, dtype=np.int64)
        if path is not None:
            with np.load(path) as mask:
                lengths, drop = mask["lengths"], mask["drop"]
            actual = np.asarray(dataset.meta.episodes["length"])
            if not np.array_equal(lengths, actual):
                raise RuntimeError(f"{path} belongs to another dataset than {dataset.root}")
            log.info(f"frame mask {path}: {len(drop)} of {dataset.num_frames} frames out")
        if context is None:
            return MaskedFrames(dataset, drop)
        names = dataset.meta.features[ENV]["names"]
        if names[CONTEXT_ROLL] != "overhead_cam_roll":
            raise RuntimeError(f"{ENV}[{CONTEXT_ROLL}] is {names[CONTEXT_ROLL]}, not the roll")
        dataset.meta.stats[ENV] = context_stats(dataset.root, len(names))
        log.info(f"context {context}: stats {dataset.meta.stats[ENV]}")
        return MaskedFrames(dataset, drop, wrap_item_roll)

    return make


def normalized(make_policy, proj: str):
    """``make_policy`` with ENV mapped to MEAN_STD and the context projection per ``proj``."""
    from lerobot.configs.types import NormalizationMode

    def make(cfg, ds_meta=None, **kwargs):
        cfg.normalization_mapping["ENV"] = NormalizationMode.MEAN_STD
        policy = make_policy(cfg=cfg, ds_meta=ds_meta, **kwargs)
        layer = policy.model.encoder_env_state_input_proj
        if proj == "reinit":
            with torch.random.fork_rng():  # every later draw (batch order, dropout) as in keep
                layer.reset_parameters()
        elif proj == "fold":
            w = layer.weight
            mean, std = (torch.as_tensor(ds_meta.stats[ENV][k]).to(w) for k in ("mean", "std"))
            with torch.no_grad():
                layer.bias += w @ mean
                w *= std
        log.info(f"context projection: {proj}")
        return policy

    return make


def main(argv: list[str]) -> None:
    ours = {
        f: [a[len(f) :] for a in argv if a.startswith(f)] for f in (FLAG, CONTEXT_FLAG, PROJ_FLAG)
    }
    rest = [a for a in argv if not a.startswith(tuple(ours))]
    path, context, proj = (ours[f][-1] if ours[f] else None for f in ours)
    if context not in (None, CONTEXT_NORM):
        raise SystemExit(f"--context_norm takes {CONTEXT_NORM}, got {context}")
    if proj is not None and (context is None or proj not in PROJ_MODES):
        raise SystemExit(f"--context_proj takes {PROJ_MODES} with --context_norm, got {proj}")
    from lerobot.scripts import lerobot_train

    if path is not None or context is not None:
        lerobot_train.make_dataset = masked(lerobot_train.make_dataset, path, context)
    if context is not None:
        lerobot_train.make_policy = normalized(lerobot_train.make_policy, proj or "keep")
    sys.argv = [sys.argv[0], *rest]
    lerobot_train.main()


if __name__ == "__main__":
    main(sys.argv[1:])

"""lerobot-train with an optional frame mask (MS7g design 8, ledger G10).

``--frame_mask=PATH`` (a mask ``.npz`` of reopen_mask.py, built for this ``--dataset.root``)
takes its frames out of training as if the converter had cut the episode before each of them:
a masked frame is never a sample, and a kept frame's action chunk is padded (``action_is_pad``)
from the first masked frame it reaches. ACT's loss ignores padded targets, so no chunk learns a
masked action; dropping the samples alone would leave the 99 frames before each REOPEN
predicting it (chunk_size 100). The mask's episode lengths must equal the dataset's.

Without the switch this is ``lerobot-train``, argument for argument. Run it from the lerobot
venv like lerobot-train: ``python train_masked.py --frame_mask=M.npz --policy.path=... ...``.
"""

import logging
import sys

import numpy as np
import torch

FLAG = "--frame_mask="
log = logging.getLogger(__name__)


class MaskedFrames(torch.utils.data.Dataset):
    """``dataset`` without the frames ``drop`` (absolute frame indices); every other attribute
    (``meta``, ``features``, ...) is the dataset's."""

    def __init__(self, dataset, drop: np.ndarray):
        self.dataset = dataset
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
        if not len(self.deltas):
            return item
        chunk = np.clip(int(item["index"]) + self.deltas, 0, len(self.dropped) - 1)
        hit = np.flatnonzero(self.dropped[chunk])
        if len(hit):
            item["action_is_pad"][hit[0] :] = True
        return item


def masked(make_dataset, path: str):
    """``make_dataset`` with its dataset wrapped in the mask at ``path``."""

    def make(cfg):
        dataset = make_dataset(cfg)
        with np.load(path) as mask:
            lengths, drop = mask["lengths"], mask["drop"]
        actual = np.asarray(dataset.meta.episodes["length"])
        if not np.array_equal(lengths, actual):
            raise RuntimeError(f"{path} belongs to another dataset than {dataset.root}")
        log.info(f"frame mask {path}: {len(drop)} of {dataset.num_frames} frames out")
        return MaskedFrames(dataset, drop)

    return make


def main(argv: list[str]) -> None:
    paths = [a[len(FLAG) :] for a in argv if a.startswith(FLAG)]
    rest = [a for a in argv if not a.startswith(FLAG)]
    from lerobot.scripts import lerobot_train

    if paths:
        lerobot_train.make_dataset = masked(lerobot_train.make_dataset, paths[-1])
    sys.argv = [sys.argv[0], *rest]
    lerobot_train.main()


if __name__ == "__main__":
    main(sys.argv[1:])

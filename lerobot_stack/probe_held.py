"""The realizability probe (MS4 c2 design 6.6): can a student see whether a brick is held?

A small classifier on the wrist image (downsampled to 64 x 64) plus the 7-wide state predicts
held-or-not on corpus frames. Labels come from the lane sidecars' cycle_phase: RAISE, EXTEND
and RELEASE are held, HOVER, DESCEND, CLOSE and RETREAT are free, LIFT and REOPEN are
excluded as ambiguous. The DAgger resume entry branches on exactly this evidence, and round 1
starts only if the probe reads at least 0.90 on held-out lanes.

    lerobot_stack/.venv/bin/python lerobot_stack/probe_held.py \\
        --lanes ~/data/maker-arm-ms4/c1/lane_{0..7} --test_lanes ~/data/maker-arm-ms4/c1/lane_{8,9}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HELD_PHASES = (6, 7, 8)  # RAISE, EXTEND, RELEASE (core.oracle.PHASE_TO_INT)
FREE_PHASES = (1, 2, 3, 9)  # HOVER, DESCEND, CLOSE, RETREAT
THRESHOLD = 0.90


def label_frames(phase: np.ndarray) -> np.ndarray:
    labels = np.full(len(phase), -1, np.int8)
    labels[np.isin(phase, HELD_PHASES)] = 1
    labels[np.isin(phase, FREE_PHASES)] = 0
    return labels


class HeldProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 16 + 7, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, images, state):
        return self.head(torch.cat([self.conv(images), state], dim=1))


def fit(model, images, state, labels, test_images, test_state, test_labels, epochs=3, lr=1e-3):
    """Train on the first triple, return the accuracy on the second."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(labels)
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, 64):
            idx = order[start : start + 64]
            loss = nn.functional.cross_entropy(model(images[idx], state[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(test_images, test_state).argmax(dim=1)
    return float((pred == test_labels).float().mean())


def load_lane(lane: Path, frames: int, rng: np.random.Generator):
    """Sample ``frames`` labeled frames of one lane: (images (n,3,64,64), state (n,7), labels)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("youtalk/maker-arm-legosort", root=lane / "dataset")
    labels_all = []
    for ep in range(ds.meta.total_episodes):
        side = np.load(lane / "sidecars" / f"{ep}.npz")
        labels_all.append(label_frames(side["cycle_phase"]))
    labels_all = np.concatenate(labels_all)
    if len(labels_all) != ds.meta.total_frames:
        raise RuntimeError(
            f"{lane}: {len(labels_all)} sidecar frames, {ds.meta.total_frames} in the dataset"
        )
    keep = np.nonzero(labels_all >= 0)[0]
    pick = np.sort(rng.choice(keep, size=min(frames, len(keep)), replace=False))
    images, state = [], []
    for t in pick:
        item = ds[int(t)]
        images.append(
            nn.functional.avg_pool2d(item["observation.images.wrist"][None], 224 // 64)[0]
        )
        state.append(item["observation.state"])
    return torch.stack(images), torch.stack(state), torch.from_numpy(labels_all[pick]).long()


class TokenArgumentParser(argparse.ArgumentParser):
    """An ArgumentParser whose usage errors still print a token.

    argparse's own ``error()`` writes usage to stderr and exits 2 with nothing on stdout, so a
    missing --lanes or a bad --epochs would look like a hang to the campaign chain that waits
    on this script's token, the same failure mode ``policy_server.py``'s own
    ``TokenArgumentParser`` fixes.
    """

    def error(self, message: str) -> None:
        print(f"PROBE_FAIL usage: {message}", flush=True)
        super().error(message)


def main(argv=None) -> int:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=Path, nargs="+", required=True)
    parser.add_argument("--test_lanes", type=Path, nargs="+", required=True)
    parser.add_argument("--frames_per_lane", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        train = [load_lane(lane, args.frames_per_lane, rng) for lane in args.lanes]
        test = [load_lane(lane, args.frames_per_lane, rng) for lane in args.test_lanes]
        cat = lambda k, parts: torch.cat([p[k] for p in parts]).to(args.device)
        model = HeldProbe().to(args.device)
        acc = fit(
            model,
            cat(0, train),
            cat(1, train),
            cat(2, train),
            cat(0, test),
            cat(1, test),
            cat(2, test),
            epochs=args.epochs,
        )
        held = int(cat(2, test).sum())
        free = int(len(cat(2, test)) - held)
    except Exception as err:  # noqa: BLE001 -- the token contract
        print(f"PROBE_FAIL {type(err).__name__}: {err}", flush=True)
        return 1
    verdict = {"accuracy": round(acc, 4), "held": held, "free": free, "threshold": THRESHOLD}
    print(f"PROBE_OK accuracy={acc:.4f} held={held} free={free} {json.dumps(verdict)}", flush=True)
    return 0 if acc >= THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())

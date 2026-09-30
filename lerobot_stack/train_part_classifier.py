"""The MS7f part classifier (design sections 6.5 and 6.7): train it, export it, read it.

A torchvision ResNet18 with ImageNet weights (ACT's backbone) and two heads, trained on render set
files (``scripts/mj_render_set.py``'s ``render_<seed>.npz``): the category, over ``CATEGORIES``
(``core.parts.PART_CATEGORIES``, pinned as text by ``tests/test_lerobot_schema_pin.py``), on every
frame of a part, and the passenger (a rider, another part that rose with the jaws: the render
set's ``riders``) on the wrist frames of the files that label it. The export is a TorchScript file: (N, 224, 224, 3) uint8 frames and whether they are wrist frames
in, (category logits (N, 6), passenger logit (N,)) out, the ImageNet normalisation and the wrist
frame's crop around the jaws (``WRIST_ROI``) inside, the names in ``categories`` and the
passenger operating point in ``passenger_threshold``. The heap loop loads it with
``torch.jit.load`` on the CPU of the uv venv (``core.perception.PartClassifier``), so this venv's
torchvision never reaches the loop.

    lerobot_stack/.venv/bin/python lerobot_stack/train_part_classifier.py \\
        --train ~/data/maker-arm-ms7/f/tables/cls/rs/render_*.npz \\
        --read ~/data/maker-arm-ms7/f/tables/cls/c1/render_*.npz --out ~/data/.../cls/model_v1

Training drops wrist frames with nothing between the jaws and overhead frames whose camera sees
less than ``MIN_SEEN`` of the part (the arm stands in front of it), and reads the render set's
extra ``wrist`` frames (``--wrist_every``) as wrist frames. Each batch draws a brightness, a
contrast, a shift, a flip and a colour-channel order per frame: the scene paints every part from
the same four colours whatever its category. A tenth of the files
(``--val_fraction``) is kept out for the passenger operating point: the lowest logit threshold
whose false-alarm rate on those wrist frames is at most ``PASSENGER_FALSE_ALARMS``.

``--read`` reads files the training never sees (C1, design section 6.7) with the exported file on
``--read_device`` (the CPU, as served): every frame, argmax over the categories the read's labels
hold, the bars and the confusion matrices, into ``<out>/c1.json``. ``--model FILE`` reads an
existing export and trains nothing.

Prints exactly one ``CLS_OK`` or ``CLS_FAIL`` line on every exit path, argparse errors included,
and exits through ``os._exit``.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple  # noqa: UP035 -- TorchScript reads the typing names

import numpy as np
import torch
import torchvision
from torch import nn

CATEGORIES = ("brick", "special", "flat", "round", "joint", "minifig")
VIEWS = ("overhead", "lift", "carry", "wrist")
WRIST_ROI = (76, 188, 58, 170)  # INITIAL: the C1 training ladder, 2026-09-30
"""Rows and columns of the wrist frame the category and passenger heads read, scaled back up to
224 px: 112 px around the pick point, which projects to (114, 144) at 8.9 cm; the heap under the
jaws fills the rest of the frame. On the preliminary set (36 train heaps read on 4 others, 134
lifts) the lift view read 0.69 on the whole frame, 0.82 on a 144 px crop and 0.85 on this one."""
MIN_SEEN = 0.5  # INITIAL: the share of a part's sample points the overhead camera sees
PASSENGER_FALSE_ALARMS = 0.01  # the operating point's cap, half the C1 bar's 0.02
BARS = {"lift_accuracy": 0.97, "category_accuracy": 0.90, "recall": 0.8, "false_alarms": 0.02}
"""C1 (design section 6.7): the wrist view at the lift reads the right category on at least 0.97
of all parts and on at least 0.90 of each category; the passenger head finds at least 0.8 of the
passengers with at most 0.02 false alarms."""


class PartClassifier(nn.Module):
    """ResNet18 and two heads on uint8 NHWC frames."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float
    wrist_roi: List[int]  # noqa: UP006

    def __init__(self, weights=None):
        super().__init__()
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Identity()
        self.backbone = net
        self.category = nn.Linear(512, len(CATEGORIES))
        self.passenger = nn.Linear(512, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.categories = list(CATEGORIES)
        self.passenger_threshold = 0.0
        self.wrist_roi = list(WRIST_ROI)

    def forward(self, frames: torch.Tensor, wrist: bool) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        x = frames.permute(0, 3, 1, 2).float() / 255.0
        if wrist:
            y0, y1, x0, x1 = self.wrist_roi
            x = nn.functional.interpolate(
                x[:, :, y0:y1, x0:x1], size=(224, 224), mode="bilinear", align_corners=False
            )
        features = self.backbone((x - self.mean) / self.std)
        return self.category(features), self.passenger(features).squeeze(-1)


def load(paths) -> dict[str, np.ndarray]:
    """The render set files as one set of arrays, plus ``file``, each frame's file index. A file
    from before a label (``riders``, ``rider_seen``) reads -1 (unknown) there."""
    parts = [np.load(p) for p in paths]

    def column(z, k):
        return z[k] if k in z.files else np.full(len(z["view"]), -1)

    keys = sorted({k for z in parts for k in z.files} - {"args", "frames"})
    out = {k: np.concatenate([column(z, k) for z in parts]) for k in keys}
    # The frames go straight into one array: a concatenation would hold every frame twice.
    out["frames"] = np.empty((len(out["view"]), 224, 224, 3), dtype=np.uint8)
    at = 0
    for z in parts:
        n = len(z["view"])
        out["frames"][at : at + n] = z["frames"]
        at += n
    out["file"] = np.concatenate([np.full(len(z["view"]), n) for n, z in enumerate(parts)])
    return out


def usable(data: dict) -> np.ndarray:
    """The frames training reads: a part in view (``MIN_SEEN``) or between the jaws."""
    over = data["view"] == "overhead"
    return (data["category"] != "") & (~over | (data["part_seen"] >= MIN_SEEN))


def augment(frames: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per frame on the device: a brightness and contrast draw, a shift of up to 12 px, a
    left-right flip and an order of the colour channels, back to uint8."""
    n = frames.shape[0]
    x = frames.float()
    dev = frames.device
    gain = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen, device=dev)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, generator=gen, device=dev)
    x = ((x - mean) * contrast + mean) * gain
    dx, dy = (int(v) for v in torch.randint(-12, 13, (2,), generator=gen, device=dev))
    x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    flip = torch.rand(n, generator=gen, device=dev) < 0.5
    x[flip] = x[flip].flip(2)
    orders = torch.tensor([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    order = orders.to(dev)[torch.randint(0, 6, (n,), generator=gen, device=dev)]
    x = torch.gather(x, 3, order[:, None, None, :].expand(-1, x.shape[1], x.shape[2], -1))
    return x.clamp(0, 255).round().to(torch.uint8)


def train(model, data, idx, args, device) -> None:
    frames = torch.from_numpy(data["frames"])
    cat = torch.tensor([CATEGORIES.index(c) if c else -1 for c in data["category"].tolist()])
    wrist = torch.from_numpy(data["view"] != "overhead")
    known = torch.from_numpy(data["riders"] >= 0)  # the frames the rider label covers
    riders = torch.from_numpy((data["riders"] > 0).astype(np.float32))
    counts = torch.bincount(cat[idx], minlength=len(CATEGORIES)).float()
    weight = torch.where(counts > 0, counts.sum() / counts.clamp(min=1) / (counts > 0).sum(), 0.0)
    told = (wrist & known)[idx]
    pos = float(riders[idx][told].sum())
    pos_weight = torch.tensor(min(10.0, (int(told.sum()) - pos) / max(pos, 1.0)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(idx) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    order_gen = torch.Generator().manual_seed(args.seed)
    ce = nn.CrossEntropyLoss(weight=weight.to(device))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model.train()
    for epoch in range(args.epochs):
        order = idx[torch.randperm(len(idx), generator=order_gen)]
        total, seen = 0.0, 0
        for start in range(0, len(order), args.batch):
            b = order[start : start + args.batch]
            x = augment(frames[b].to(device, non_blocking=True), gen)
            loss = torch.zeros((), device=device)
            for view, is_wrist in ((~wrist[b], False), (wrist[b], True)):
                if not bool(view.any()):
                    continue
                on = view.to(device)
                amp = device.type == "cuda"
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, rider = model(x[on], is_wrist)
                share = float(view.sum()) / len(b)
                loss = loss + share * ce(logits.float(), cat[b][view].to(device))
                told = known[b][view]
                if is_wrist and bool(told.any()):
                    on_told = told.to(device)
                    loss = loss + bce(rider.float()[on_told], riders[b][view][told].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(b)
            seen += len(b)
        print(f"epoch {epoch} loss {total / seen:.4f}", flush=True)


@torch.no_grad()
def infer(model, frames: np.ndarray, wrist: np.ndarray, device, batch: int = 64):
    """Category logits (N, 6) and passenger logits (N,) of ``frames``, ``wrist`` (N,) saying
    which are wrist frames."""
    model.eval()
    logits = np.zeros((len(frames), len(CATEGORIES)), dtype=np.float32)
    riders = np.zeros(len(frames), dtype=np.float32)
    for is_wrist in (False, True):
        rows = np.flatnonzero(wrist == is_wrist)
        for start in range(0, len(rows), batch):
            at = rows[start : start + batch]
            a, b = model(torch.from_numpy(frames[at]).to(device), is_wrist)
            logits[at], riders[at] = a.float().cpu().numpy(), b.float().cpu().numpy()
    return logits, riders


def operating_point(riders: np.ndarray, truth: np.ndarray) -> float:
    """The lowest logit threshold whose false-alarm rate on ``truth``'s negatives is at most
    ``PASSENGER_FALSE_ALARMS``; +inf with no negatives."""
    negatives = np.sort(riders[~truth])[::-1]
    if not len(negatives):
        return math.inf
    allowed = int(PASSENGER_FALSE_ALARMS * len(negatives))
    return float(np.nextafter(negatives[allowed], np.inf))


def _rate(hit: int, total: int) -> float | None:
    return hit / total if total else None


def read(model, data: dict, device) -> dict:
    """C1 (design section 6.7): the rows, the bars and the confusion matrices."""
    logits, riders = infer(model, data["frames"], data["view"] != "overhead", device)
    labels = data["category"]
    active = [c for c in CATEGORIES if c in set(labels.tolist())]
    cols = [CATEGORIES.index(c) for c in active]
    guess = np.array(active)[logits[:, cols].argmax(axis=1)]
    right = guess == labels
    part = labels != ""
    views = {v: data["view"] == v for v in VIEWS}
    held = data["heldout"].astype(bool)
    visible = ~views["overhead"] | (data["part_seen"] >= MIN_SEEN)

    def row(mask) -> dict:
        m = mask & part
        return {
            "right": int(right[m].sum()),
            "total": int(m.sum()),
            "rate": _rate(right[m].sum(), m.sum()),
        }

    def confusion(mask) -> dict:
        m = mask & part
        return {
            t: {g: int(((labels == t) & (guess == g) & m).sum()) for g in active} for t in active
        }

    lift = views["lift"]
    wrist = views["lift"] | views["carry"]
    flagged = riders >= float(model.passenger_threshold)

    def passenger(mask, label="riders") -> dict:
        truth = data[label] > 0
        mask = mask & (data[label] >= 0)
        pos, neg = mask & truth, mask & ~truth
        return {
            "passengers": int(pos.sum()),
            "found": int((flagged & pos).sum()),
            "recall": _rate((flagged & pos).sum(), pos.sum()),
            "negatives": int(neg.sum()),
            "false_alarms": int((flagged & neg).sum()),
            "false_alarm_rate": _rate((flagged & neg).sum(), neg.sum()),
        }

    out = {
        "frames": len(labels),
        "categories": active,
        "lift": row(lift),
        "lift_by_category": {c: row(lift & (labels == c)) for c in active},
        "lift_heldout": row(lift & held),
        "lift_train_kinds": row(lift & ~held),
        "carry": row(views["carry"]),
        "carry_heldout": row(views["carry"] & held),
        "overhead": row(views["overhead"]),
        "overhead_visible": row(views["overhead"] & visible),
        "overhead_heldout_visible": row(views["overhead"] & visible & held),
        "lift_by_kind": {
            k: row(lift & (data["part"] == k))
            for k in sorted(set(data["part"][lift & part].tolist()))
        },
        "passenger_lift": passenger(lift),
        "passenger_wrist": passenger(wrist),
        "passenger_lift_t2_rule": passenger(lift, "passengers"),
        "riders_in_frame": {
            "lift_frames_with_riders_labelled": int((lift & (data["rider_seen"] >= 0)).sum()),
            "rider_seen_at_all": int((lift & (data["rider_seen"] > 0)).sum()),
            "rider_half_seen": int((lift & (data["rider_seen"] >= 0.5)).sum()),
        },
        "passenger_threshold": float(model.passenger_threshold),
        "confusion_lift": confusion(lift),
        "confusion_overhead_visible": confusion(views["overhead"] & visible),
    }
    rates = [out["lift_by_category"][c]["rate"] for c in active]
    p = out["passenger_lift"]
    out["bars"] = {
        "lift_accuracy": bool((out["lift"]["rate"] or 0.0) >= BARS["lift_accuracy"]),
        "category_accuracy": all(bool((r or 0.0) >= BARS["category_accuracy"]) for r in rates),
        "passenger_recall": bool((p["recall"] or 0.0) >= BARS["recall"]),
        "passenger_false_alarms": bool((p["false_alarm_rate"] or 1.0) <= BARS["false_alarms"]),
    }
    out["pass"] = all(out["bars"].values())
    return out


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("CLS_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--train", nargs="*", default=[], help="render set files to train on")
    parser.add_argument("--read", nargs="*", default=[], help="render set files to read (C1)")
    parser.add_argument("--out", required=True, help="the output directory")
    parser.add_argument("--model", default="", help="read this export instead of training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--read_device", default="cpu", help="the served model runs on the CPU")
    args = parser.parse_args(argv)
    if bool(args.train) == bool(args.model):
        parser.error("give either --train files or an exported --model to read")
    if args.model and not args.read:
        parser.error("--model reads an export: it needs --read files")
    return args


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"out": str(out)}
    path = Path(args.model) if args.model else out / "part_classifier.pt"
    if args.train:
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        data = load(sorted(args.train))
        files = sorted(set(data["file"].tolist()))
        random.Random(args.seed).shuffle(files)
        val_files = set(
            files[: max(1, round(args.val_fraction * len(files)))] if len(files) > 1 else []
        )
        keep = usable(data)
        val = np.isin(data["file"], list(val_files))
        idx = torch.from_numpy(np.flatnonzero(keep & ~val))
        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            if args.weights == "imagenet"
            else None
        )
        model = PartClassifier(weights).to(device)
        started = time.time()
        train(model, data, idx, args, device)
        summary["train_seconds"] = round(time.time() - started, 1)
        wrist = keep & (data["view"] != "overhead")
        wrist &= data["riders"] >= 0
        check = wrist & val if (wrist & val).any() else wrist
        _, riders = infer(model, data["frames"][check], np.ones(check.sum(), bool), device)
        model.passenger_threshold = operating_point(riders, data["riders"][check] > 0)
        summary |= {
            "train_frames": len(idx),
            "val_files": sorted(int(f) for f in val_files),
            "passenger_threshold": model.passenger_threshold,
        }
        model = model.cpu().eval()
        torch.jit.script(model).save(str(path))
    summary["model"] = str(path)
    if args.read:
        served = torch.jit.load(str(path), map_location=args.read_device)
        result = read(served, load(sorted(args.read)), torch.device(args.read_device))
        (out / "c1.json").write_text(json.dumps(result, indent=2))
        summary |= {"c1": str(out / "c1.json"), "pass": result["pass"], "bars": result["bars"]}
        summary["lift"] = result["lift"]
        summary["passenger_lift"] = result["passenger_lift"]
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        print("CLS_OK " + json.dumps(run(args)), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("CLS_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())

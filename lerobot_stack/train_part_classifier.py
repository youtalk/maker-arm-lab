"""The MS7f part classifier (design sections 6.5 and 6.7): train it, export it, read it.

A torchvision ResNet (``--arch``: ResNet18, ACT's backbone, by default) with ImageNet weights and
two heads, trained on render set
files (``scripts/mj_render_set.py``'s ``render_<seed>.npz``): the category, over ``CATEGORIES``
(``core.parts.PART_CATEGORIES``, pinned as text by ``tests/test_lerobot_schema_pin.py``), on every
frame of a part, and the passenger (a rider, another part that rose with the jaws: the render
set's ``riders``) on the wrist frames of the files that label it. The category head also reads
the perceived size of the part (``scripts/render_set_sizes.py``'s ``render_<seed>.size.npy``: the
loop's noisy detection size, sorted, the cycle target's on a wrist frame) through a small MLP
beside the CNN features. The export is a TorchScript file: (N, 224, 224, 3) uint8 frames, whether
they are wrist frames and the (N, 3) sizes in metres in, (category logits (N, 6), passenger logit
(N,)) out, the ImageNet normalisation, the wrist frame's crop around the jaws (``WRIST_ROI``) and
the overhead frame's crop around the target (``OVERHEAD_ROI``) inside, the names in
``categories``, ``size_input`` and the passenger operating point in ``passenger_threshold``. The heap loop loads it with
``torch.jit.load`` on the CPU of the uv venv (``core.perception.PartClassifier``), so this venv's
torchvision never reaches the loop.

    lerobot_stack/.venv/bin/python lerobot_stack/train_part_classifier.py \\
        --train ~/data/maker-arm-ms7/f/tables/cls/rs/render_*.npz \\
        --read ~/data/maker-arm-ms7/f/tables/cls/c1/render_*.npz --out ~/data/.../cls/model_v1

Training drops wrist frames with nothing between the jaws and overhead frames whose camera sees
less than ``MIN_SEEN`` of the part (the arm stands in front of it), and reads the render set's
extra ``wrist`` frames (``--wrist_every``) as wrist frames. Each batch draws a brightness, a
contrast, a shift, a flip and a colour-channel order per frame: the scene paints every part from
the same four colours whatever its category. Against overfitting to the lifts it has seen (v2
read 0.977 on its training heaps and 0.894 on its validation heaps, 2026-09-30) it also erases a
random rectangle (``--erase``), smooths the labels (``--label_smoothing``) and keeps the epoch
with the best validation lift accuracy (``--patience``). ``--cap`` thins each file's overhead and
burst frames so a large render set fits in memory. A tenth of the files (``--val_fraction``) is
kept out for that and for the passenger operating point: the lowest logit threshold whose
false-alarm rate on those wrist frames is at most ``PASSENGER_FALSE_ALARMS``.

``--read`` reads files the training never sees (C1, design section 6.7) with the exported file on
``--read_device`` (the CPU, as served): every frame, argmax over the categories the read's labels
hold, the bars and the confusion matrices, into ``<out>/c1.json``. It also reads the overhead
view by the part's rest pose (``tilt``), the lifts whose part the loop sees in all three views
(``cycles``) under the lift alone, the loop's vote (``loop_vote``) and the summed log
probabilities of the three views, and how much of a missed passenger the lift frame shows.
``--model FILE`` reads an existing export and trains nothing. ``--ensemble FILE ...`` exports
existing exports as one model, the mean of their logits, with ``--flip_average`` also over each
frame mirrored (the test-time flip), and reads that.

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
import zlib
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
OVERHEAD_ROI = (
    56,
    168,
    56,
    168,
)  # INITIAL: the controller's review of the train frames, 2026-09-30
"""Rows and columns of the narrow overhead frame the category head reads, scaled back up to 224
px: the middle 112 px, 56 mm at the camera's 0.5 mm a pixel (design 6.3 fixes the sampling, not
the crop). The whole 112 mm frame shows a dozen parts, and only its centre says which is the
target: on a teacher-driven C1 heap the target's projected centre and the centroid of its own
pixels both sit within 10 px of (112, 112), the aim's perceived-like error."""
LATER_LABELS = {"riders", "rider_seen", "tilt"}  # render set labels an older file does not carry
MIN_SEEN = 0.5  # INITIAL: the share of a part's sample points the overhead camera sees
PASSENGER_FALSE_ALARMS = 0.01  # the operating point's cap, half the C1 bar's 0.02
BARS = {"lift_accuracy": 0.97, "category_accuracy": 0.90, "recall": 0.8, "false_alarms": 0.02}
"""C1 (design section 6.7): the wrist view at the lift reads the right category on at least 0.97
of all parts and on at least 0.90 of each category; the passenger head finds at least 0.8 of the
passengers with at most 0.02 false alarms."""
UPRIGHT_DEG = (
    30.0  # INITIAL: a tilt (the render set's ``tilt``) this close to up or down is not a side
)


class PartClassifier(nn.Module):
    """A ResNet (``arch``, ResNet18 by default) and two heads on uint8 NHWC frames; the crop is
    resized to ``input_px`` a side before the backbone."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float
    wrist_roi: List[int]  # noqa: UP006
    overhead_roi: List[int]  # noqa: UP006
    size_input: bool
    wrist_input: str
    input_px: int

    def __init__(
        self,
        weights=None,
        wrist_crop: bool = False,
        arch: str = "resnet18",
        input_px: int = 224,
    ):
        super().__init__()
        net = getattr(torchvision.models, arch)(weights=weights)
        width = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.size_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.category = nn.Linear(width + 64, len(CATEGORIES))
        self.passenger = nn.Linear(width, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.categories = list(CATEGORIES)
        self.passenger_threshold = 0.0
        # A render set with ``wrist_crop`` holds the fingertip crop at native pixels already.
        self.wrist_roi = [0, 224, 0, 224] if wrist_crop else list(WRIST_ROI)
        self.wrist_input = "crop" if wrist_crop else "frame"
        self.overhead_roi = list(OVERHEAD_ROI)
        self.size_input = True
        self.input_px = input_px

    def forward(
        self, frames: torch.Tensor, wrist: bool, size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        x = frames.permute(0, 3, 1, 2).float() / 255.0
        y0, y1, x0, x1 = self.wrist_roi if wrist else self.overhead_roi
        px = self.input_px
        x = nn.functional.interpolate(
            x[:, :, y0:y1, x0:x1], size=(px, px), mode="bilinear", align_corners=False
        )
        features = self.backbone((x - self.mean) / self.std)
        # The perceived size (N, 3) in metres, sorted, and the view: an unknown size (-1) reads 0.
        # A wrist view reads none: the size is the target's, and one lift in ten holds another.
        if wrist:
            size = torch.full_like(size, -1.0)
        extra = torch.cat(
            [size.float().clamp(min=0.0) * 100.0, torch.full_like(size[:, :1], float(wrist))], 1
        )
        both = torch.cat([features, self.size_net(extra.to(features.dtype))], 1)
        return self.category(both), self.passenger(features).squeeze(-1)


class Ensemble(nn.Module):
    """``--ensemble``: exported classifiers served as one, the mean of their category and
    passenger logits, with ``flip_average`` also over each frame mirrored left to right
    (training's flip; on a crop model's centred crops the same as mirroring the crop). The
    members must share the categories, the wrist input and the size input."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float
    wrist_roi: List[int]  # noqa: UP006
    overhead_roi: List[int]  # noqa: UP006
    size_input: bool
    wrist_input: str
    flip_average: bool

    def __init__(self, members, flip_average: bool = False):
        super().__init__()
        first = members[0]
        for m in members:
            same = (
                list(m.categories) == list(first.categories) and m.wrist_input == first.wrist_input
            )
            if not (same and m.size_input):
                raise ValueError("ensemble members must share categories, wrist and size input")
        self.members = nn.ModuleList(members)
        self.categories = list(first.categories)
        # ponytail: the members' mean operating point, not one fitted to the mean passenger logit
        self.passenger_threshold = float(np.mean([m.passenger_threshold for m in members]))
        self.wrist_roi = list(first.wrist_roi)
        self.overhead_roi = list(first.overhead_roi)
        self.size_input = True
        self.wrist_input = str(first.wrist_input)
        self.flip_average = flip_average

    def forward(
        self, frames: torch.Tensor, wrist: bool, size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        views = [frames, frames.flip(2)] if self.flip_average else [frames]
        logits: List[torch.Tensor] = []  # noqa: UP006
        riders: List[torch.Tensor] = []  # noqa: UP006
        for m in self.members:
            for x in views:
                c, r = m(x, wrist, size)
                logits.append(c)
                riders.append(r)
        return torch.stack(logits).mean(0), torch.stack(riders).mean(0)


def pick(view: np.ndarray, cap: int, name: str) -> np.ndarray:
    """The rows of one render file that ``load`` keeps: all with ``cap`` 0, else every lift and
    carry frame and at most ``cap`` frames of each other view, drawn from the file's name."""
    if not cap:
        return np.arange(len(view))
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    keep = [np.flatnonzero((view == "lift") | (view == "carry"))]
    for v in sorted(set(view.tolist()) - {"lift", "carry"}):
        rows = np.flatnonzero(view == v)
        keep.append(rng.choice(rows, min(cap, len(rows)), replace=False))
    return np.sort(np.concatenate(keep))


def load(paths, cap: int = 0) -> dict[str, np.ndarray]:
    """The render set files as one set of arrays, plus ``file``, each frame's file index. A file
    from before a label (``riders``, ``rider_seen``, ``tilt``) reads -1 (unknown) there. ``cap``
    thins each file's views (``pick``) so a large render set fits in memory."""
    parts = [np.load(p) for p in paths]
    picks = [pick(z["view"], cap, Path(p).name) for p, z in zip(paths, parts)]

    def column(z, k, rows):
        return z[k][rows] if k in z.files else np.full(len(rows), -1)

    crops = {tuple(z["wrist_crop"].tolist()) if "wrist_crop" in z.files else None for z in parts}
    if len(crops - {None}) > 1:
        raise ValueError(f"render files with different wrist crops: {sorted(map(str, crops))}")
    keys = sorted({k for z in parts for k in z.files} - {"args", "frames", "wrist_crop"})
    keys = sorted(set(keys) | LATER_LABELS)
    out = {k: np.concatenate([column(z, k, r) for z, r in zip(parts, picks)]) for k in keys}
    # The frames go straight into one array: a concatenation would hold every frame twice.
    out["frames"] = np.empty((len(out["view"]), 224, 224, 3), dtype=np.uint8)
    at = 0
    for z, rows in zip(parts, picks):
        out["frames"][at : at + len(rows)] = z["frames"][rows] if cap else z["frames"]
        at += len(rows)
    out["file"] = np.concatenate([np.full(len(r), n) for n, r in enumerate(picks)])
    out["wrist_crop"] = next((c for c in crops if c is not None), None)
    out["wrist_mixed"] = len(crops) > 1
    if out["wrist_mixed"]:  # a file from before the native crop gives its overhead frames only
        old = np.concatenate(
            [np.full(len(r), "wrist_crop" not in z.files) for z, r in zip(parts, picks)]
        )
        out["category"] = np.where(old & (out["view"] != "overhead"), "", out["category"])
    sizes = [Path(p).with_name(Path(p).name.removesuffix(".npz") + ".size.npy") for p in paths]
    out["size"] = np.concatenate(
        [
            np.load(f)[r] if f.exists() else np.full((len(r), 3), -1.0, np.float32)
            for f, r in zip(sizes, picks)
        ]
    )
    return out


def usable(data: dict) -> np.ndarray:
    """The frames training reads: a part in view (``MIN_SEEN``) or between the jaws."""
    over = data["view"] == "overhead"
    return (data["category"] != "") & (~over | (data["part_seen"] >= MIN_SEEN))


def augment(frames: torch.Tensor, gen: torch.Generator, erase: float = 0.0) -> torch.Tensor:
    """Per frame on the device: a brightness and contrast draw, a shift of up to 12 px, a
    left-right flip, an order of the colour channels and, at the chance ``erase``, a random
    rectangle of 2 to 20 % of the frame painted one random colour (random erasing), back to
    uint8."""
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
    if erase:
        h, w = x.shape[1], x.shape[2]
        on = torch.rand(n, generator=gen, device=dev) < erase
        area = (0.02 + 0.18 * torch.rand(n, generator=gen, device=dev)) * h * w
        aspect = torch.exp((torch.rand(n, generator=gen, device=dev) - 0.5) * 2.4)
        bh, bw = (area * aspect).sqrt().clamp(1, h), (area / aspect).sqrt().clamp(1, w)
        y0 = torch.rand(n, generator=gen, device=dev) * (h - bh)
        x0 = torch.rand(n, generator=gen, device=dev) * (w - bw)
        ys = torch.arange(h, device=dev)[None, :, None]
        xs = torch.arange(w, device=dev)[None, None, :]
        box = (ys >= y0[:, None, None]) & (ys < (y0 + bh)[:, None, None])
        box = box & (xs >= x0[:, None, None]) & (xs < (x0 + bw)[:, None, None]) & on[:, None, None]
        paint = 255.0 * torch.rand(n, 1, 1, 3, generator=gen, device=dev)
        x = torch.where(box[..., None], paint, x)
    return x.clamp(0, 255).round().to(torch.uint8)


def val_lift_accuracy(model, data, rows, device) -> float:
    """The share of the lift frames among ``rows`` whose category the model reads right."""
    rows = rows[data["view"][rows] == "lift"]
    if not len(rows):
        return float("nan")
    logits, _ = infer(
        model, data["frames"][rows], np.ones(len(rows), bool), data["size"][rows], device
    )
    model.train()
    names = np.array(CATEGORIES)
    return float((names[logits.argmax(1)] == data["category"][rows]).mean())


def train(model, data, idx, args, device, val_rows=None) -> None:
    """``args.epochs`` epochs at most; with ``val_rows`` (the by-heap validation frames) and
    ``args.patience`` it keeps the epoch whose validation lift accuracy is best and stops after
    ``args.patience`` epochs without a better one."""
    frames = torch.from_numpy(data["frames"])
    size = torch.from_numpy(data["size"])
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
    ce = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=args.label_smoothing)
    best, best_state, stale = -1.0, None, 0
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model.train()
    for epoch in range(args.epochs):
        order = idx[torch.randperm(len(idx), generator=order_gen)]
        total, seen = 0.0, 0
        for start in range(0, len(order), args.batch):
            b = order[start : start + args.batch]
            x = augment(frames[b].to(device, non_blocking=True), gen, args.erase)
            loss = torch.zeros((), device=device)
            for view, is_wrist in ((~wrist[b], False), (wrist[b], True)):
                if not bool(view.any()):
                    continue
                on = view.to(device)
                amp = device.type == "cuda"
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits, rider = model(x[on], is_wrist, size[b][view].to(device))
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
        line = f"epoch {epoch} loss {total / seen:.4f}"
        if val_rows is not None and len(val_rows) and args.patience:
            acc = val_lift_accuracy(model, data, val_rows, device)
            line += f" val_lift {acc:.4f}"
            if acc > best:
                best, stale = acc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        print(line, flush=True)
        if args.patience and stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"kept the epoch with val_lift {best:.4f}", flush=True)


@torch.no_grad()
def infer(model, frames: np.ndarray, wrist: np.ndarray, size: np.ndarray, device, batch: int = 64):
    """Category logits (N, 6) and passenger logits (N,) of ``frames``, ``wrist`` (N,) saying
    which are wrist frames and ``size`` (N, 3) the perceived sizes, which an export from before
    the size input (no ``size_input``) does not read."""
    sized = bool(getattr(model, "size_input", False))
    model.eval()
    logits = np.zeros((len(frames), len(CATEGORIES)), dtype=np.float32)
    riders = np.zeros(len(frames), dtype=np.float32)
    for is_wrist in (False, True):
        rows = np.flatnonzero(wrist == is_wrist)
        for start in range(0, len(rows), batch):
            at = rows[start : start + batch]
            x = torch.from_numpy(frames[at]).to(device)
            if sized:
                a, b = model(x, is_wrist, torch.from_numpy(size[at]).to(device))
            else:
                a, b = model(x, is_wrist)
            logits[at], riders[at] = a.float().cpu().numpy(), b.float().cpu().numpy()
    return logits, riders


def cycles(data: dict) -> np.ndarray:
    """(lift, overhead, carry) row triples, one per lift frame with a part between the jaws: the
    overhead frame of that part at its env's latest selection before the lift (the loop's
    ``read_class``, -1 with none: the camera did not see it or another part was the target) and
    the env's next wrist view if it is the carry (-1 when a lift comes first or none does)."""
    view, held, step = data["view"], data["held"], data["step"]
    env = data["file"] * 1_000_000 + data["env"]
    over, carry = view == "overhead", (view == "carry") | (view == "lift")
    rows = []
    for n in np.flatnonzero((view == "lift") & (held >= 0)):
        same = env == env[n]
        seen = np.flatnonzero(same & over & (step <= step[n]))
        o = -1
        if len(seen):
            last = seen[step[seen] == step[seen].max()]
            match = last[held[last] == held[n]]
            o = int(match[0]) if len(match) else -1
        later = np.flatnonzero(same & carry & (step > step[n]))
        c = int(later[step[later].argmin()]) if len(later) else -1
        rows.append((n, o, c if c >= 0 and view[c] == "carry" else -1))
    return np.array(rows, dtype=int).reshape(-1, 3)


def targets(data: dict) -> np.ndarray:
    """Each row's selection target, the part index (``held``) of the first overhead row at its
    env's latest selection step up to the row's own (-1 before any): the target when the camera
    saw it, which the render set views first."""
    view, held, step = data["view"], data["held"], data["step"]
    env = data["file"] * 1_000_000 + data["env"]
    out = np.full(len(view), -1)
    latest: dict[int, tuple[int, int]] = {}  # env -> (selection step, its first row's part)
    for n in range(len(view)):
        e = int(env[n])
        if view[n] == "overhead" and latest.get(e, (None,))[0] != int(step[n]):
            latest[e] = (int(step[n]), int(held[n]))
        out[n] = latest.get(e, (None, -1))[1]
    return out


def loop_vote(over: np.ndarray, lift: np.ndarray, carry: np.ndarray) -> np.ndarray:
    """The heap loop's ``class_views`` rule (``core.loop.lift_class_box``): the lift read when it
    equals the overhead read, else the overhead read if the carry read sides with it, else the
    carry read."""
    return np.where(lift == over, lift, np.where(carry == over, over, carry))


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


def read(model, data: dict, device, keep=None) -> dict:
    """C1 (design section 6.7): the rows, the bars and the confusion matrices. ``keep`` (a
    path) also saves every frame's category and passenger logits there."""
    logits, riders = infer(model, data["frames"], data["view"] != "overhead", data["size"], device)
    if keep is not None:
        np.savez(keep, logits=logits, riders=riders)
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
    target = targets(data)
    on_target = target == data["held"]
    neighbour = (target >= 0) & ~on_target
    flagged = riders >= float(model.passenger_threshold)
    kinds = sorted(set(data["part"][part].tolist()))
    tilt = data["tilt"]
    near = np.minimum(tilt, 180.0 - tilt)
    poses = {
        "upright": (tilt >= 0) & (tilt <= UPRIGHT_DEG),
        "upside_down": (tilt >= 0) & (near <= UPRIGHT_DEG) & (tilt > 90.0),
        "side": (tilt >= 0) & (near > UPRIGHT_DEG),
    }
    trip = cycles(data)
    trip = trip[(trip[:, 1] >= 0) & (trip[:, 2] >= 0)]
    logp = torch.log_softmax(torch.from_numpy(logits[:, cols]), dim=1).numpy()
    names = np.array(active)
    t_lift, t_over, t_carry = (trip[:, k] for k in range(3))
    reads = {
        "lift": guess[t_lift],
        "loop_vote": loop_vote(guess[t_over], guess[t_lift], guess[t_carry]),
        "sum_log_prob": names[(logp[t_over] + logp[t_lift] + logp[t_carry]).argmax(axis=1)],
        "sum_log_prob_wrist": names[(logp[t_lift] + logp[t_carry]).argmax(axis=1)],
    }
    # The loop draws the carry view only when the lift and overhead reads differ.
    split = reads["lift"] != guess[t_over]
    reads["loop_sum"] = np.where(split, reads["sum_log_prob"], reads["lift"])
    t_truth, t_held = labels[t_lift], held[t_lift]
    vote_ok = reads["loop_vote"] == t_truth

    def fused(rule) -> dict:
        ok = reads[rule] == t_truth

        def r(m) -> dict:
            return {
                "right": int(ok[m].sum()),
                "total": int(m.sum()),
                "rate": _rate(ok[m].sum(), m.sum()),
            }

        every = np.ones(len(ok), dtype=bool)
        return {
            "all": r(every),
            "heldout": r(t_held),
            "train_kinds": r(~t_held),
            **{c: r(t_truth == c) for c in active},
            "vs_loop_vote": {
                "better": int((ok & ~vote_ok).sum()),
                "worse": int((~ok & vote_ok).sum()),
            },
        }

    # A lift with no part between the jaws has no passenger: its ``riders`` count the part the
    # jaws lifted without pressing it, which the force read missed.
    missed = lift & part & (data["riders"] > 0) & ~flagged

    def passenger(mask, label="riders") -> dict:
        truth = data[label] > 0
        mask = mask & part & (data[label] >= 0)
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
        # The wrist frames' labels are the part in the jaws; the loop's target may be another.
        "lift_on_target": row(lift & on_target),
        "lift_on_target_heldout": row(lift & on_target & held),
        "lift_on_neighbour": row(lift & neighbour),
        "lift_on_neighbour_heldout": row(lift & neighbour & held),
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
            "lift_frames_with_riders_labelled": int(
                (lift & part & (data["rider_seen"] >= 0)).sum()
            ),
            "rider_seen_at_all": int((lift & part & (data["rider_seen"] > 0)).sum()),
            "rider_half_seen": int((lift & part & (data["rider_seen"] >= 0.5)).sum()),
        },
        "passenger_threshold": float(model.passenger_threshold),
        "confusion_lift": confusion(lift),
        "overhead_visible_by_pose": {
            pose: {c: row(views["overhead"] & visible & m & (labels == c)) for c in active}
            for pose, m in poses.items()
        },
        "overhead_visible_by_kind_pose": {
            k: {
                pose: row(views["overhead"] & visible & m & (data["part"] == k))
                for pose, m in poses.items()
            }
            for k in kinds
        },
        "three_views": {"lifts": int((lift & part).sum()), "with_all_three": len(trip)},
        "fused": {rule: fused(rule) for rule in reads},
        "passenger_misses_in_frame": {
            "misses": int(missed.sum()),
            "labelled": int((missed & (data["rider_seen"] >= 0)).sum()),
            "rider_seen_at_all": int((missed & (data["rider_seen"] > 0)).sum()),
            "rider_half_seen": int((missed & (data["rider_seen"] >= 0.5)).sum()),
        },
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
    parser.add_argument(
        "--cap",
        type=int,
        default=0,
        help="per file, at most this many frames of each view but lift and carry (0: all)",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--val_skip",
        default="",
        help="a file of part names the early stop's "
        "validation leaves out (the render set's extra kinds)",
    )
    parser.add_argument("--erase", type=float, default=0.5, help="random erasing's chance")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="early stopping on the "
        "validation heaps' lift accuracy after this many epochs (0: off)",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--arch", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    parser.add_argument("--input_px", type=int, default=224, help="the backbone's input side")
    parser.add_argument(
        "--ensemble",
        nargs="*",
        default=[],
        help="exports to serve as one (the mean of their logits) instead of training",
    )
    parser.add_argument(
        "--flip_average",
        action="store_true",
        help="with --ensemble, also read each frame mirrored and average (test-time flip)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--read_device", default="cpu", help="the served model runs on the CPU")
    args = parser.parse_args(argv)
    if sum(map(bool, (args.train, args.model, args.ensemble))) != 1:
        parser.error("give either --train files, an exported --model or --ensemble exports")
    if args.flip_average and not args.ensemble:
        parser.error("--flip_average applies to an --ensemble export")
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
        data = load(sorted(args.train), args.cap)
        files = sorted(set(data["file"].tolist()))
        random.Random(args.seed).shuffle(files)
        val_files = set(
            files[: max(1, round(args.val_fraction * len(files)))] if len(files) > 1 else []
        )
        keep = usable(data)
        val = np.isin(data["file"], list(val_files))
        idx = torch.from_numpy(np.flatnonzero(keep & ~val))
        weights = (
            torchvision.models.get_model_weights(args.arch).DEFAULT
            if args.weights == "imagenet"
            else None
        )
        crop = data["wrist_crop"] is not None
        model = PartClassifier(weights, crop, args.arch, input_px=args.input_px).to(device)
        started = time.time()
        skip = Path(args.val_skip).read_text().split() if args.val_skip else []
        val_rows = np.flatnonzero(keep & val & ~np.isin(data["part"], skip))
        train(model, data, idx, args, device, val_rows)
        summary["train_seconds"] = round(time.time() - started, 1)
        wrist = keep & (data["view"] != "overhead")
        wrist &= data["riders"] >= 0
        check = wrist & val if (wrist & val).any() else wrist
        _, riders = infer(
            model, data["frames"][check], np.ones(check.sum(), bool), data["size"][check], device
        )
        model.passenger_threshold = operating_point(riders, data["riders"][check] > 0)
        summary |= {
            "train_frames": len(idx),
            "val_files": sorted(int(f) for f in val_files),
            "passenger_threshold": model.passenger_threshold,
        }
        model = model.cpu().eval()
        torch.jit.script(model).save(str(path))
    if args.ensemble:
        members = [torch.jit.load(m, map_location="cpu") for m in args.ensemble]
        torch.jit.script(Ensemble(members, args.flip_average)).save(str(path))
        summary["members"] = args.ensemble
    summary["model"] = str(path)
    if args.read:
        served = torch.jit.load(str(path), map_location=args.read_device)
        keep = out / "logits.npz"
        data = load(sorted(args.read))
        crop = getattr(served, "wrist_input", "frame") == "crop"
        if data["wrist_mixed"] or crop != (data["wrist_crop"] is not None):
            raise ValueError("the model and the read files take different wrist frames")
        result = read(served, data, torch.device(args.read_device), keep)
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

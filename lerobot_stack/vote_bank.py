"""The MS7g frame bank (design sections 6.1 and 6.3, ledger G8): read a part classifier on the
bank's in-loop frames, fit the fused vote and judge it against the served rule, no simulator run.

The bank files are ``scripts/mj_eval_heap.py --classifier_frames DIR --bank_views`` output
(``policy_<run_seed>_<layout_seed>.npz``, run 20320000 + p): every frame the served classifier read
(views "overhead", "lift", "carry") plus the carry view of each agreed lift ("bank"), with the
simulator's labels. A lift event is one "lift" frame with its cycle's overhead frame and the
"carry" or "bank" frame after it in the same cycle (none: the part left the jaws first). Its
truth is the category of the part both jaws pressed at the lift; an event with none is not read.

Three steps, one call:

1. Read ``--model`` (a TorchScript export, as ``core.perception.PartClassifier`` loads it: wrist
   frames as recorded, the overhead frame, the perceived size beside the file) on every frame, the
   log probability of each of the model's categories. Cached per model and file under ``--out``.
2. Fit the fused vote (design 6.3): the class with the largest ``w_o * overhead + w_l * lift +
   w_c * carry + b[class]``, a missing view dropped, three weights and six biases, by maximum
   likelihood of a multinomial logit on the fit part: even p of the bank and every ``--extra`` file
   (the 51 D24 policy files), train kinds only. ``--vote_out`` gets the nine values as JSON.
3. Judge both rules on the evaluation part (odd p): the final class of each event under the
   served rule (``served_class``, ``core.loop``'s ``class_views`` vote) and the fused rule
   (``fused_class``), on the events with all three views (the decision rows) and on all events.
   Rows: whole, train, held-out and each category. G8 adopts the vote if the whole row gains at
   least 0.01, the held-out row falls by at most 0.005 and no category falls by more than 0.02.

``--payloads`` (the runs' ``--out`` JSON files) checks the replay: per cycle, the served rule on the
cycle's last lift against the census ``taken_class`` (and the raw reads against ``read_class`` and
``lift_class``). A file without the ``cycle`` column (D24) numbers its cycles by overhead frame.

    lerobot_stack/.venv/bin/python lerobot_stack/vote_bank.py --model .../part_classifier.pt \\
        --bank '~/data/maker-arm-ms7/g/tables/vote/bank/*/policy_*.npz' \\
        --extra '~/data/maker-arm-ms7/f/tables/cls/h/rs9_*/policy_*.npz' \\
        --payloads '~/data/maker-arm-ms7/g/tables/vote/bank/*/proc_*.json' \\
        --out /tmp/vb --vote_out .../vote_weights.json

Prints exactly one ``VOTE_OK`` or ``VOTE_FAIL`` line on every exit path, argparse errors included,
and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

BANK_RUN0 = 20320000  # ledger G1: the bank's run seed is BANK_RUN0 + p
VIEWS = ("overhead", "lift", "carry")
CARRY_VIEWS = ("carry", "bank")  # the served carry view and the bank's agreed one


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("VOTE_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def served_class(o: np.ndarray, lift: np.ndarray, carry: np.ndarray | None) -> int:
    """``core.loop``'s served rule on one lift (``class_views`` with the classifier): the lift read
    when it equals the overhead read, else the largest summed log probability of the three views
    (``summed_class``), else, when the part left the jaws before the carry view, the overhead read
    (the selection's box stays). Indices into the categories. Pure."""
    read, lifted = int(np.argmax(o)), int(np.argmax(lift))
    if lifted == read:
        return lifted
    if carry is None:
        return read
    return int(np.argmax(o + lift + carry))


def fused_class(vote: dict, names, o=None, lift=None, carry=None) -> int:
    """The fused vote (design 6.3) as ``core.loop.fused_class`` serves it: the index in ``names``
    of the largest ``sum(weights[view] * logp[view]) + bias[name]`` over the views given (None
    drops out). Pure."""
    score = np.array([vote["bias"][n] for n in names], dtype=np.float64)
    for view, logp in zip(VIEWS, (o, lift, carry)):
        if logp is not None:
            score = score + vote["weights"][view] * np.asarray(logp, dtype=np.float64)
    return int(np.argmax(score))


def events(data: dict) -> list[dict]:
    """The lift events of one file's columns (``view``, ``env``, ``cycle`` or None, ...), in file
    order: row indices ``o`` (the cycle's overhead frame, -1 with none), ``lift`` and ``carry`` (-1
    with none), the cycle ordinal, and whether it is the cycle's last lift."""
    view, env = data["view"], data["env"]
    cycle = data.get("cycle")
    out, overhead, count, last = [], {}, {}, {}  # last: env -> its latest event
    for n, v in enumerate(view):
        e = int(env[n])
        if v == "overhead":
            count[e] = count.get(e, 0) + 1
            overhead[e] = n
        c = int(cycle[n]) if cycle is not None else count.get(e, 0)
        ev = last.get(e)
        ev = ev if ev is not None and ev["cycle"] == c else None
        if v == "lift":
            if ev is not None:
                ev["last"] = False
            o = overhead.get(e, -1)
            o = o if o >= 0 and (cycle is None or int(cycle[o]) == c) else -1
            last[e] = {"env": e, "cycle": c, "o": o, "lift": n, "carry": -1, "last": True}
            out.append(last[e])
        elif v in CARRY_VIEWS and ev is not None and ev["carry"] < 0:
            ev["carry"] = n
    return out


def model_logp(model, frames, wrist, size, names, device, batch: int = 64) -> np.ndarray:
    """Each frame's log probability over ``names`` (a subset of the model's categories, in that
    order), read as ``PartClassifier.log_probs`` reads one frame: the model on uint8 frames, whether
    they are wrist frames and the sorted perceived size, then a log softmax over ``names``."""
    cols = [list(model.categories).index(n) for n in names]
    sized = bool(getattr(model, "size_input", False))
    out = np.zeros((len(frames), len(names)), dtype=np.float32)
    with torch.no_grad():
        for is_wrist in (False, True):
            rows = np.flatnonzero(wrist == is_wrist)
            for start in range(0, len(rows), batch):
                at = rows[start : start + batch]
                x = torch.from_numpy(frames[at]).to(device)
                if sized:
                    logits, _ = model(x, is_wrist, torch.from_numpy(size[at]).float().to(device))
                else:
                    logits, _ = model(x, is_wrist)
                logp = torch.log_softmax(logits[:, cols].float(), dim=1)
                out[at] = logp.cpu().numpy()
    return out


def load_file(path: Path, model, names, cache: Path, device) -> dict:
    """One frames file's label columns, its ``logp`` under the model (cached) and its seeds."""
    z = np.load(path)
    data = {k: z[k] for k in z.files if k not in ("frames", "args", "wrist_crop")}
    data.setdefault("cycle", None)
    args = json.loads(str(z["args"]))
    data["run_seed"], data["layout_seed"] = int(args["run_seed"]), int(args["layout_seed"])
    hit = cache / f"{path.stem}.{len(data['view'])}.npy"
    if hit.exists():
        data["logp"] = np.load(hit)
        return data
    side = path.with_name(path.stem + ".size.npy")
    n = len(data["view"])
    size = np.load(side)[:n] if side.exists() else np.full((n, 3), -1.0, np.float32)
    wrist = data["view"] != "overhead"
    data["logp"] = model_logp(model, z["frames"], wrist, size.astype(np.float32), names, device)
    np.save(hit, data["logp"])
    return data


def table(files: list[dict], names) -> list[dict]:
    """Every lift event of ``files`` with its three log probability rows (None missing), its
    truth (the category index of the part held at the lift, -1 with none), its held-out flag, the
    file's index and the bank's p (None for an extra file)."""
    rows = []
    for f, data in enumerate(files):
        logp, cat, held = data["logp"], data["category"], data["heldout"]
        for ev in events(data):
            truth = str(cat[ev["lift"]])
            rows.append(
                {
                    **ev,
                    "file": f,
                    "p": data["p"],
                    "logp": tuple(
                        None if ev[k] < 0 else logp[ev[k]] for k in ("o", "lift", "carry")
                    ),
                    "truth": names.index(truth) if truth in names else -1,
                    "heldout": bool(held[ev["lift"]]),
                }
            )
    return rows


def fit(rows: list[dict], names) -> dict:
    """The nine values by maximum likelihood: weights from 1 and biases from 0, L-BFGS in float64,
    the biases centred (a softmax reads them only up to a constant). ``rows`` need an overhead and
    a lift view and a truth; a missing carry view drops out of the row's score."""
    o = torch.tensor(np.stack([r["logp"][0] for r in rows]), dtype=torch.float64)
    lift = torch.tensor(np.stack([r["logp"][1] for r in rows]), dtype=torch.float64)
    has = torch.tensor([r["logp"][2] is not None for r in rows], dtype=torch.float64)
    carry = torch.tensor(
        np.stack([np.zeros(len(names)) if r["logp"][2] is None else r["logp"][2] for r in rows]),
        dtype=torch.float64,
    )
    truth = torch.tensor([r["truth"] for r in rows])
    w = torch.ones(3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(len(names), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def nll():
        opt.zero_grad()
        score = w[0] * o + w[1] * lift + w[2] * has[:, None] * carry + b
        loss = torch.nn.functional.cross_entropy(score, truth, reduction="sum")
        loss.backward()
        return loss

    loss = float(opt.step(nll))
    bias = (b - b.mean()).detach().tolist()
    return {
        "weights": dict(zip(VIEWS, (round(float(v), 6) for v in w.detach()))),
        "bias": {n: round(float(v), 6) for n, v in zip(names, bias)},
        "nll_start": round(loss, 3),
        "nll": round(float(nll().detach()), 3),
        "rows": len(rows),
    }


def judge(rows: list[dict], names, vote: dict) -> dict:
    """The decision rows of both rules over ``rows`` (each with a truth): whole, train, held-out
    and each category, as {"right", "total", "rate"} for "served" and "fused"."""
    served = np.array([served_class(*r["logp"]) for r in rows], dtype=int)
    fused = np.array([fused_class(vote, names, *r["logp"]) for r in rows], dtype=int)
    truth = np.array([r["truth"] for r in rows], dtype=int)
    held = np.array([r["heldout"] for r in rows], dtype=bool)
    masks = {"whole": np.ones(len(rows), bool), "train": ~held, "heldout": held}
    masks |= {n: truth == k for k, n in enumerate(names)}
    out = {}
    for key, m in masks.items():
        out[key] = {}
        for rule, guess in (("served", served), ("fused", fused)):
            right, total = int((guess[m] == truth[m]).sum()), int(m.sum())
            out[key][rule] = {
                "right": right,
                "total": total,
                "rate": right / total if total else None,
            }
    return out


def verdict(rows: dict, names) -> dict:
    """G8's three conditions on the decision rows."""

    def gain(key):
        s, f = rows[key]["served"]["rate"], rows[key]["fused"]["rate"]
        return None if s is None else f - s

    falls = {n: gain(n) for n in names if gain(n) is not None}
    checks = {
        "whole_gain": gain("whole"),
        "heldout_gain": gain("heldout"),
        "worst_category_gain": min(falls.values()) if falls else None,
    }
    checks["adopted"] = bool(
        checks["whole_gain"] is not None
        and checks["whole_gain"] >= 0.01
        and (checks["heldout_gain"] is None or checks["heldout_gain"] >= -0.005)
        and all(v >= -0.02 for v in falls.values())
    )
    return checks


def check_replay(files: list[dict], payloads: list[Path], names) -> dict:
    """The served rule replayed on each recorded cycle's last lift against the payload census:
    ``taken_class`` (None after a lost vote or with no lift: the replay then has no carry view),
    and the raw ``read_class`` and ``lift_class``."""
    census = {}
    for path in payloads:
        p = json.loads(path.read_text())
        for ep in p["episodes"]:
            key = (int(p["run_seed"]), int(p["layout_seed"]), int(ep.get("env_index", 0)))
            census[key] = ep["cycle_census"]
    n = {"cycles": 0, "taken": 0, "read": 0, "lift": 0, "lost": 0, "lost_ok": 0, "files": 0}
    for data in files:
        logp, matched = data["logp"], False
        for ev in events(data):
            rows = census.get((data["run_seed"], data["layout_seed"], ev["env"]))
            if not ev["last"] or ev["o"] < 0 or rows is None or ev["cycle"] > len(rows):
                continue
            matched = True
            row = rows[ev["cycle"] - 1]
            o, lift = logp[ev["o"]], logp[ev["lift"]]
            carry = None if ev["carry"] < 0 else logp[ev["carry"]]
            n["read"] += names[int(np.argmax(o))] == row["read_class"]
            n["lift"] += names[int(np.argmax(lift))] == row["lift_class"]
            if row["taken_class"] is None:
                n["lost"] += 1
                n["lost_ok"] += carry is None
                continue
            n["cycles"] += 1
            n["taken"] += names[served_class(o, lift, carry)] == row["taken_class"]
        n["files"] += matched
    n["taken_rate"] = n["taken"] / n["cycles"] if n["cycles"] else None
    return n


def parse_args(argv=None) -> argparse.Namespace:
    parser = _Parser(description=__doc__)
    parser.add_argument("--model", required=True, help="the TorchScript classifier to read")
    parser.add_argument("--bank", action="append", default=[], help="glob of bank files (p)")
    parser.add_argument("--extra", action="append", default=[], help="glob of fit-only files")
    parser.add_argument("--payloads", action="append", default=[], help="glob of run payloads")
    parser.add_argument("--out", required=True, help="dir: the logp cache and summary.json")
    parser.add_argument("--vote_out", default="", help="write the fitted vote's JSON here")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if not args.bank and not args.extra:
        parser.error("--bank and --extra name no file")
    return args


def expand(patterns) -> list[Path]:
    return sorted({Path(p) for g in patterns for p in glob.glob(os.path.expanduser(g))})


def run(args) -> dict:
    torch.backends.cudnn.allow_tf32 = False  # the served read is float32 on the CPU
    torch.backends.cuda.matmul.allow_tf32 = False
    model = torch.jit.load(args.model, map_location=args.device).eval()
    names = list(model.categories)
    tag = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()[:12]
    cache = Path(args.out) / f"logp_{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    bank, extra = expand(args.bank), expand(args.extra)
    if not bank + extra:
        raise FileNotFoundError(f"no file matches {args.bank + args.extra}")
    files = []
    for path in bank + extra:
        data = load_file(path, model, names, cache, args.device)
        data["p"] = data["run_seed"] - BANK_RUN0 if path in bank else None
        files.append(data)
    rows = [r for r in table(files, names) if r["logp"][0] is not None]
    known = [r for r in rows if r["truth"] >= 0]
    fit_rows = [r for r in known if (r["p"] is None or r["p"] % 2 == 0) and not r["heldout"]]
    eval_rows = [r for r in known if r["p"] is not None and r["p"] % 2 == 1]
    three = [r for r in eval_rows if r["logp"][2] is not None]
    vote = fit(fit_rows, names)
    decision = judge(three, names, vote)
    summary = {
        "model": args.model,
        "model_sha": tag,
        "categories": names,
        "files": {"bank": len(bank), "extra": len(extra)},
        "events": {
            "lifts": len(rows),
            "with_truth": len(known),
            "bank_three_views": sum(r["p"] is not None and r["logp"][2] is not None for r in known),
            "fit": len(fit_rows),
            "eval": len(eval_rows),
            "eval_three_views": len(three),
        },
        "vote": vote,
        "rows_three_views": decision,
        "rows_all_lifts": judge(eval_rows, names, vote),
        "verdict": verdict(decision, names),
    }
    payloads = expand(args.payloads)
    if payloads:
        summary["replay"] = check_replay(files, payloads, names)
    out = Path(args.out) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    if args.vote_out:
        values = {k: vote[k] for k in ("weights", "bias")}
        Path(args.vote_out).write_text(json.dumps(values, indent=2) + "\n")
    return {
        "out": str(out),
        "events": summary["events"],
        "weights": vote["weights"],
        "verdict": summary["verdict"],
        "replay_taken_rate": summary.get("replay", {}).get("taken_rate"),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        print("VOTE_OK " + json.dumps(run(args)), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("VOTE_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())

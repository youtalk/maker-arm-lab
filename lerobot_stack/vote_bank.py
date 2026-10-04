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
   log probability of each of the model's categories. ``--members a.pt,b.pt,...`` reads an
   ensemble as ``train_part_classifier.py --ensemble`` exports one (model_v9e is v6a + v8): the
   log softmax of the members' mean category logits. ``--flip`` (MS7g ledger D10) also reads each
   frame mirrored left to right, the frame the classifier is given, and averages the two log
   probabilities a view. Each member's logits are cached per file under ``--cache``.
2. Fit each fused vote form of ``--forms`` by maximum likelihood of a multinomial logit on the fit
   part: even p of the bank and every ``--extra`` file (the 51 D24 policy files), train kinds
   only. Each class k scores ``b[k] + sum_v (W_v logp_v)[k]`` over the views read (a missing view
   drops out), with W_v ``w_v`` times the identity in "nine" (design 6.3: three weights and six
   biases), a diagonal of a weight a view and class in "v2" (MS7g ledger D7: 18 + 6) and a full
   matrix in "v3" (D7: 108 + 6, with an L2 penalty on W picked from ``L2_GRID`` by the mean
   held-out log likelihood of a 5-fold split of the fit part by file). ``--vote_dir`` gets each
   form's JSON (``vote_<form>.json``); ``core.loop``'s ``--vote_weights`` serves the nine values
   only, since D7 adopted neither richer form on v9e (2026-10-03).
3. Judge both rules on the evaluation part (odd p): the final class of each event under the
   served rule (``served_class``, ``core.loop``'s ``class_views`` vote) and each form
   (``fused_class``), on the events with all three views (the decision rows) and on all events.
   Rows: whole, train, held-out and each category. G8 adopts a form if the whole row gains at
   least 0.01, the held-out row falls by at most 0.005 and no category falls by more than 0.02;
   of the D7 forms (v2, v3) that pass, the higher whole row wins.

``--against SUMMARY`` (another model's ``summary.json`` on the same bank) reads G8's classifier
bars per form, this model's fused rows against that one's: the held-out row rises at least 0.03,
the whole row does not fall and no category falls by more than 0.02.

``--payloads`` (the runs' ``--out`` JSON files) checks the replay: per cycle, the served rule on the
cycle's last lift against the census ``taken_class`` (and the raw reads against ``read_class`` and
``lift_class``). A file without the ``cycle`` column (D24) numbers its cycles by overhead frame.

    lerobot_stack/.venv/bin/python lerobot_stack/vote_bank.py --model .../part_classifier.pt \\
        --bank '~/data/maker-arm-ms7/g/tables/vote/bank/*/policy_*.npz' \\
        --extra '~/data/maker-arm-ms7/f/tables/cls/h/rs9_*/policy_*.npz' \\
        --payloads '~/data/maker-arm-ms7/g/tables/vote/bank/*/proc_*.json' \\
        --out /tmp/vb --vote_dir .../votes

Prints exactly one ``VOTE_OK`` or ``VOTE_FAIL`` line on every exit path, argparse errors included,
and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import functools
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
FORMS = ("nine", "v2", "v3")
L2_GRID = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)  # v3's L2 candidates on W (the summed fit loss)
FOLDS = 5


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


def matrices(vote: dict, names) -> tuple[np.ndarray, np.ndarray]:
    """A vote JSON's (3, K, K) view matrices and (K,) biases over ``names``: ``matrix`` (v3: per
    view, per class, per class read) or ``weights`` (per view a number, nine, or per class, v2)."""
    w = np.zeros((len(VIEWS), len(names), len(names)))
    for v, view in enumerate(VIEWS):
        if "matrix" in vote:
            w[v] = [[vote["matrix"][view][n][m] for m in names] for n in names]
        else:
            got = vote["weights"][view]
            w[v] = np.diag([got[n] if isinstance(got, dict) else got for n in names])
    return w, np.array([vote["bias"][n] for n in names], dtype=np.float64)


def fused_class(vote: dict, names, o=None, lift=None, carry=None) -> int:
    """The fused vote (design 6.3, ledger D7's forms) as ``core.loop.fused_class`` serves it: the
    index in ``names`` of the largest ``bias + sum_v W_v logp_v`` over the views given (None drops
    out). Pure."""
    w, score = matrices(vote, names)
    for v, logp in enumerate((o, lift, carry)):
        if logp is not None:
            score = score + w[v] @ np.asarray(logp, dtype=np.float64)
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


def model_logits(model, frames, wrist, size, device, batch: int = 64) -> np.ndarray:
    """Each frame's category logits over the model's categories, read as
    ``PartClassifier.log_probs`` reads one frame: the model on uint8 frames, whether they are wrist
    frames and the sorted perceived size."""
    sized = bool(getattr(model, "size_input", False))
    out = np.zeros((len(frames), len(model.categories)), dtype=np.float32)
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
                out[at] = logits.float().cpu().numpy()
    return out


def combine(logits: list[np.ndarray], cols: list[int]) -> np.ndarray:
    """An ensemble's log probabilities over ``cols``: the log softmax of the members' mean logits
    (``train_part_classifier.Ensemble``), as ``PartClassifier.log_probs`` takes it."""
    mean = torch.from_numpy(np.mean(logits, axis=0)[:, cols]).float()
    return torch.log_softmax(mean, dim=1).numpy()


def _cached(cache: Path, member: Path, path: Path, n: int, mirrored: bool) -> Path:
    return cache / _sha(member) / f"{path.stem}.{n}.{'flip' if mirrored else 'orig'}.npy"


def fill_cache(paths: list[Path], members: list[Path], cache: Path, device, flip: bool) -> None:
    """Each member's logits on each file's frames (and mirrored ones with ``flip``), member by
    member so each export loads once, skipping what the cache holds."""
    for member in members:
        model = None
        for path in paths:
            z = np.load(path)
            n = len(z["view"])
            todo = [m for m in ((False, True) if flip else (False,))]
            todo = [m for m in todo if not _cached(cache, member, path, n, m).exists()]
            if not todo:
                continue
            if model is None:
                model = torch.jit.load(str(member), map_location=device).eval()
            side = path.with_name(path.stem + ".size.npy")
            size = (np.load(side)[:n] if side.exists() else np.full((n, 3), -1.0)).astype(
                np.float32
            )
            frames, wrist = z["frames"], z["view"] != "overhead"
            for mirrored in todo:
                hit = _cached(cache, member, path, n, mirrored)
                hit.parent.mkdir(parents=True, exist_ok=True)
                x = np.ascontiguousarray(frames[:, :, ::-1]) if mirrored else frames
                np.save(hit, model_logits(model, x, wrist, size, device))
        del model


def load_file(path: Path, members: list[Path], names, cache: Path, flip: bool) -> dict:
    """One frames file's label columns, its seeds and its ``logp`` from the members' cached
    logits: the ensemble's log probabilities, with ``flip`` the mean of the frame's and the mirrored
    frame's."""
    z = np.load(path)
    data = {k: z[k] for k in z.files if k not in ("frames", "args", "wrist_crop")}
    data.setdefault("cycle", None)
    args = json.loads(str(z["args"]))
    data["run_seed"], data["layout_seed"] = int(args["run_seed"]), int(args["layout_seed"])
    n = len(data["view"])
    sides = [
        combine(
            [np.load(_cached(cache, m, path, n, mirrored)) for m in members],
            [CATEGORIES_OF[members[0]].index(c) for c in names],
        )
        for mirrored in ((False, True) if flip else (False,))
    ]
    data["logp"] = np.mean(sides, axis=0)
    return data


CATEGORIES_OF: dict = {}  # member path -> its categories, filled by ``run``


@functools.cache
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


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


def _arrays(rows: list[dict], names):
    """(N, 3, K) view log probabilities, a missing carry view zero, and (N,) truths."""
    k = len(names)
    x = np.stack([np.stack([v if v is not None else np.zeros(k) for v in r["logp"]]) for r in rows])
    return torch.tensor(x, dtype=torch.float64), torch.tensor([r["truth"] for r in rows])


def _weights(form: str, p: torch.Tensor, k: int) -> torch.Tensor:
    """The (3, K, K) view matrices of a form's parameters."""
    if form == "nine":
        return p[:, None, None] * torch.eye(k, dtype=p.dtype)
    if form == "v2":
        return torch.diag_embed(p)
    return p


def _fit(x, truth, form: str, k: int, l2: float):
    """Maximum likelihood (plus ``l2`` times the squared view weights) from the served sum's
    weights of 1 and biases of 0, L-BFGS in float64. Returns (view matrices, biases)."""
    start = {"nine": torch.ones(3), "v2": torch.ones(3, k), "v3": torch.eye(k).repeat(3, 1, 1)}
    p = start[form].to(torch.float64).requires_grad_(True)
    b = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p, b], max_iter=1000, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        score = torch.einsum("nvj,vkj->nk", x, _weights(form, p, k)) + b
        out = torch.nn.functional.cross_entropy(score, truth, reduction="sum")
        out = out + l2 * (p**2).sum()
        out.backward()
        return out

    opt.step(loss)
    return _weights(form, p, k).detach(), (b - b.mean()).detach()


def _nll(x, truth, w, b) -> float:
    score = torch.einsum("nvj,vkj->nk", x, w) + b
    return float(torch.nn.functional.cross_entropy(score, truth, reduction="sum"))


def fit(rows: list[dict], names, form: str = "nine") -> dict:
    """A form's vote JSON fitted on ``rows`` (each with an overhead and a lift view and a truth),
    the biases centred (a softmax reads them only up to a constant), with its fit statistics. v3
    picks its L2 from ``L2_GRID`` by the summed held-out log likelihood of ``FOLDS`` folds by
    file."""
    k = len(names)
    x, truth = _arrays(rows, names)
    l2, cv = 0.0, None
    if form == "v3":
        fold = torch.tensor([r["file"] % FOLDS for r in rows])
        cv = {}
        for lam in L2_GRID:
            held = 0.0
            for f in range(FOLDS):
                w, b = _fit(x[fold != f], truth[fold != f], form, k, lam)
                held += _nll(x[fold == f], truth[fold == f], w, b)
            cv[str(lam)] = round(held, 3)
        l2 = min(L2_GRID, key=lambda lam: cv[str(lam)])
    w, b = _fit(x, truth, form, k, l2)
    vote = {"form": form, "bias": {n: round(float(v), 6) for n, v in zip(names, b)}}
    if form == "nine":
        vote["weights"] = {view: round(float(w[v, 0, 0]), 6) for v, view in enumerate(VIEWS)}
    elif form == "v2":
        vote["weights"] = {
            view: {n: round(float(w[v, i, i]), 6) for i, n in enumerate(names)}
            for v, view in enumerate(VIEWS)
        }
    else:
        vote["matrix"] = {
            view: {
                n: {m: round(float(w[v, i, j]), 6) for j, m in enumerate(names)}
                for i, n in enumerate(names)
            }
            for v, view in enumerate(VIEWS)
        }
    w, b = (torch.from_numpy(a) for a in matrices(vote, names))  # the rounded values served
    stats = {"nll": round(_nll(x, truth, w, b), 3), "rows": len(rows)}
    stats |= {"l2": l2, "cv_nll": cv} if cv is not None else {}
    return vote | {"fit": stats}


def judge(rows: list[dict], names, vote: dict) -> dict:
    """The decision rows of both rules over ``rows`` (each with a truth): whole, train, held-out
    and each category, as {"right", "total", "rate"} for "served" and "fused"."""
    served = np.array([served_class(*r["logp"]) for r in rows], dtype=int)
    w, b = matrices(vote, names)
    fused = np.array(
        [
            np.argmax(b + sum(w[v] @ x for v, x in enumerate(r["logp"]) if x is not None))
            for r in rows
        ],
        dtype=int,
    )
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


def classifier_verdict(mine: dict, theirs: dict, names) -> dict:
    """G8's classifier bars: this model's fused rows against another's on the same events."""
    if mine["whole"]["fused"]["total"] != theirs["whole"]["fused"]["total"]:
        raise ValueError("the two summaries read different evaluation events")

    def gain(key):
        a, b = mine[key]["fused"]["rate"], theirs[key]["fused"]["rate"]
        return None if a is None or b is None else a - b

    falls = {n: gain(n) for n in names if gain(n) is not None}
    checks = {
        "whole_gain": gain("whole"),
        "heldout_gain": gain("heldout"),
        "worst_category_gain": min(falls.values()) if falls else None,
    }
    checks["adopted"] = bool(
        checks["heldout_gain"] is not None
        and checks["heldout_gain"] >= 0.03
        and checks["whole_gain"] >= 0.0
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
    parser.add_argument("--model", default="", help="the TorchScript classifier to read")
    parser.add_argument("--members", default="", help="comma list of exports read as one ensemble")
    parser.add_argument("--flip", action="store_true", help="also read each frame mirrored")
    parser.add_argument("--cache", default="", help="dir of the member logits cache (--out)")
    parser.add_argument("--bank", action="append", default=[], help="glob of bank files (p)")
    parser.add_argument("--extra", action="append", default=[], help="glob of fit-only files")
    parser.add_argument("--payloads", action="append", default=[], help="glob of run payloads")
    parser.add_argument("--out", required=True, help="dir: the logp cache and summary.json")
    parser.add_argument("--forms", default=",".join(FORMS), help="comma list of FORMS to fit")
    parser.add_argument("--vote_dir", default="", help="write each form's vote_<form>.json here")
    parser.add_argument("--against", default="", help="another model's summary.json: G8 bars")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if not args.bank and not args.extra:
        parser.error("--bank and --extra name no file")
    if bool(args.model) == bool(args.members):
        parser.error("give either --model or --members")
    args.members = [Path(m).expanduser() for m in (args.members or args.model).split(",")]
    args.forms = args.forms.split(",")
    if not set(args.forms) <= set(FORMS):
        parser.error(f"--forms takes {FORMS}")
    return args


def expand(patterns) -> list[Path]:
    return sorted({Path(p) for g in patterns for p in glob.glob(os.path.expanduser(g))})


def run(args) -> dict:
    torch.backends.cudnn.allow_tf32 = False  # the served read is float32 on the CPU
    torch.backends.cuda.matmul.allow_tf32 = False
    for member in args.members:
        CATEGORIES_OF[member] = list(torch.jit.load(str(member), map_location="cpu").categories)
    names = CATEGORIES_OF[args.members[0]]
    if any(c != names for c in CATEGORIES_OF.values()):
        raise ValueError("the members read different categories or orders")
    tag = "+".join(_sha(m) for m in args.members) + ("+flip" if args.flip else "")
    cache = Path(args.cache or args.out)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    bank, extra = expand(args.bank), expand(args.extra)
    if not bank + extra:
        raise FileNotFoundError(f"no file matches {args.bank + args.extra}")
    files = []
    fill_cache(bank + extra, args.members, cache, args.device, args.flip)
    for path in bank + extra:
        data = load_file(path, args.members, names, cache, args.flip)
        data["p"] = data["run_seed"] - BANK_RUN0 if path in bank else None
        files.append(data)
    rows = [r for r in table(files, names) if r["logp"][0] is not None]
    known = [r for r in rows if r["truth"] >= 0]
    fit_rows = [r for r in known if (r["p"] is None or r["p"] % 2 == 0) and not r["heldout"]]
    eval_rows = [r for r in known if r["p"] is not None and r["p"] % 2 == 1]
    three = [r for r in eval_rows if r["logp"][2] is not None]
    forms = {}
    for form in args.forms:
        vote = fit(fit_rows, names, form)
        decision = judge(three, names, vote)
        forms[form] = {
            "vote": vote,
            "rows_three_views": decision,
            "rows_all_lifts": judge(eval_rows, names, vote),
            "verdict": verdict(decision, names),
        }
    passing = [f for f in ("v2", "v3") if f in forms and forms[f]["verdict"]["adopted"]]
    winner = max(
        passing,
        key=lambda f: forms[f]["rows_three_views"]["whole"]["fused"]["rate"],
        default=None,
    )
    summary = {
        "model": [str(m) for m in args.members],
        "flip": args.flip,
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
        "forms": forms,
        "winner": winner,
    }
    if args.against:
        theirs = json.loads(Path(args.against).read_text())["forms"]
        summary["against"] = {"summary": args.against} | {
            f: classifier_verdict(
                forms[f]["rows_three_views"], theirs[f]["rows_three_views"], names
            )
            for f in forms
            if f in theirs
        }
    payloads = expand(args.payloads)
    if payloads:
        summary["replay"] = check_replay(files, payloads, names)
    out = Path(args.out) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    if args.vote_dir:
        Path(args.vote_dir).mkdir(parents=True, exist_ok=True)
        for form, got in forms.items():
            vote = {k: v for k, v in got["vote"].items() if k != "fit"}
            (Path(args.vote_dir) / f"vote_{form}.json").write_text(
                json.dumps(vote, indent=2) + "\n"
            )
    return {
        "out": str(out),
        "events": summary["events"],
        "verdicts": {f: forms[f]["verdict"]["adopted"] for f in forms},
        "winner": summary["winner"],
        "against": {f: v["adopted"] for f, v in summary.get("against", {}).items() if f in forms},
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

"""``lerobot_stack/vote_bank.py`` (MS7g design sections 6.1 and 6.3): the frame bank tool applies
the loop's served and fused rules, groups a cycle's views into lift events and fits the vote."""

import json
import random
from typing import List  # noqa: UP035 -- TorchScript reads the typing names

import numpy as np
import pytest
import torch

from maker_arm_lab.core import loop
from test_ms7f_stack import ROOT, _load
from test_vote_weights import NAMES, _vote

bank = _load(ROOT / "lerobot_stack" / "vote_bank.py")


def test_the_bank_tool_applies_the_loop_s_rules():
    rng = random.Random(7)
    for _ in range(500):
        views = [[rng.uniform(-6.0, 0.0) for _ in NAMES] for _ in range(3)]
        vote = _vote(
            *(rng.uniform(0.0, 2.0) for _ in range(3)), {n: rng.gauss(0, 1) for n in NAMES}
        )
        for k in (1, 2, 3):
            given = views[:k] + [None] * (3 - k)
            arrays = [None if v is None else np.array(v) for v in given]
            mine = NAMES[bank.fused_class(vote, NAMES, *arrays)]
            assert mine == loop.fused_class(NAMES, vote, *given)
        o, lift, carry = views
        read, lifted = loop.summed_class(NAMES, o), loop.summed_class(NAMES, lift)
        served = lifted if lifted == read else loop.summed_class(NAMES, o, lift, carry)
        assert NAMES[bank.served_class(*map(np.array, views))] == served
        lost = lifted if lifted == read else read  # the part left before the carry view
        assert NAMES[bank.served_class(np.array(o), np.array(lift), None)] == lost


def test_a_lift_event_takes_its_cycle_s_overhead_and_the_carry_view_after_it():
    view = ["overhead", "lift", "bank", "overhead", "lift", "lift", "carry", "overhead", "lift"]
    cycle = [1, 1, 1, 2, 2, 2, 2, 3, 3]
    data = {"view": np.array(view), "env": np.zeros(9, int), "cycle": np.array(cycle)}
    got = [(e["cycle"], e["o"], e["lift"], e["carry"], e["last"]) for e in bank.events(data)]
    want = [(1, 0, 1, 2, True), (2, 3, 4, -1, False), (2, 3, 5, 6, True), (3, 7, 8, -1, True)]
    assert got == want
    data["cycle"] = None  # a D24 file: cycles numbered by overhead frame
    assert [(e["cycle"], e["o"], e["lift"], e["carry"]) for e in bank.events(data)] == [
        w[:4] for w in want
    ]


def test_the_fit_recovers_the_vote_that_drew_the_classes():
    rng = np.random.default_rng(3)
    true = _vote(0.6, 1.4, 0.8, {"brick": 1.0, "joint": -0.5})
    w = np.array([true["weights"][v] for v in bank.VIEWS])
    b = np.array([true["bias"][n] for n in NAMES])
    rows = []
    for _ in range(6000):
        logp = [np.log(rng.dirichlet(np.full(len(NAMES), 0.7))) for _ in range(3)]
        has = rng.random() < 0.8
        score = w[0] * logp[0] + w[1] * logp[1] + (w[2] * logp[2] if has else 0.0) + b
        p = np.exp(score - score.max())
        truth = int(rng.choice(len(NAMES), p=p / p.sum()))
        rows.append({"logp": (logp[0], logp[1], logp[2] if has else None), "truth": truth})
    vote = bank.fit(rows, list(NAMES))
    for v in bank.VIEWS:
        assert vote["weights"][v] == pytest.approx(true["weights"][v], abs=0.12), v
    centred = b - b.mean()
    for n, want in zip(NAMES, centred):
        assert vote["bias"][n] == pytest.approx(want, abs=0.15), n


def test_each_form_scores_with_its_view_matrices():
    """Ledger D7: nine values, a weight a view and class (v2) and a full matrix a view (v3), each
    with a bias a class; a missing view drops out of every form."""
    rng = np.random.default_rng(5)
    k = len(NAMES)
    diag, full, bias = (
        rng.normal(1, 0.3, (3, k)),
        rng.normal(0, 0.5, (3, k, k)),
        rng.normal(0, 1, k),
    )
    v2 = {
        "weights": {v: dict(zip(NAMES, diag[i])) for i, v in enumerate(bank.VIEWS)},
        "bias": dict(zip(NAMES, bias)),
    }
    v3 = {
        "matrix": {
            v: {n: dict(zip(NAMES, full[i, a])) for a, n in enumerate(NAMES)}
            for i, v in enumerate(bank.VIEWS)
        },
        "bias": dict(zip(NAMES, bias)),
    }
    for _ in range(200):
        views = [np.log(rng.dirichlet(np.ones(k))) for _ in range(3)]
        for given in (views, views[:2] + [None]):
            want2 = bias + sum(diag[i] * x for i, x in enumerate(given) if x is not None)
            want3 = bias + sum(full[i] @ x for i, x in enumerate(given) if x is not None)
            assert bank.fused_class(v2, NAMES, *given) == int(np.argmax(want2))
            assert bank.fused_class(v3, NAMES, *given) == int(np.argmax(want3))
            rows = [{"logp": tuple(given), "truth": 0, "heldout": False}]
            judged = bank.judge(rows, list(NAMES), v3)["whole"]["fused"]["right"]
            assert judged == int(np.argmax(want3) == 0)


def test_the_stacker_picks_its_l2_by_folds_of_files():
    rng = np.random.default_rng(9)
    rows = []
    for n in range(600):
        logp = [np.log(rng.dirichlet(np.full(len(NAMES), 0.7))) for _ in range(3)]
        truth = int(np.argmax(logp[0] + logp[1] + rng.gumbel(size=len(NAMES))))
        rows.append({"logp": tuple(logp), "truth": truth, "file": n % 7})
    for form in ("v2", "v3"):
        vote = bank.fit(rows, list(NAMES), form)
        assert set(vote["bias"]) == set(NAMES) and vote["form"] == form
        assert ("matrix" in vote) == (form == "v3") and ("weights" in vote) == (form == "v2")
    assert vote["fit"]["l2"] in bank.L2_GRID and len(vote["fit"]["cv_nll"]) == len(bank.L2_GRID)
    best = min(vote["fit"]["cv_nll"].values())
    assert vote["fit"]["cv_nll"][str(vote["fit"]["l2"])] == best


class _Halves(torch.nn.Module):
    """Category 0's logit is the left half's mean brightness and category 1's the right half's,
    times ``gain``, so a mirrored frame reads the other way round."""

    categories: List[str]  # noqa: UP006
    size_input: bool

    def __init__(self, gain: float):
        super().__init__()
        self.categories = list(NAMES)
        self.size_input = True
        self.gain = gain

    def forward(self, frames: torch.Tensor, wrist: bool, size: torch.Tensor):
        x = frames.float().mean(dim=(1, 3)) / 255.0  # (N, W)
        logits = torch.zeros(frames.shape[0], len(self.categories))
        logits[:, 0] = self.gain * x[:, :112].mean(1)
        logits[:, 1] = self.gain * x[:, 112:].mean(1)
        return logits, torch.zeros(frames.shape[0])


def test_members_and_the_flip_read_as_the_ensemble_export_and_its_mirror(tmp_path):
    """``--members``: the log softmax of the members' mean logits (``Ensemble``); ``--flip``: the
    mean of that on the frame and on the frame mirrored left to right."""
    frames = np.zeros((3, 224, 224, 3), np.uint8)
    for n, left in enumerate((200, 50, 120)):  # each frame's left half
        frames[n, :, :112] = left
    path = tmp_path / "policy_1_2.npz"
    view = np.array(["overhead", "lift", "carry"])
    np.savez(path, frames=frames, args=np.array(json.dumps({"run_seed": 1, "layout_seed": 2})),
             view=view, env=np.zeros(3, int), category=np.array(["brick"] * 3))  # fmt: skip
    members = []
    for gain in (1.0, 3.0):
        members.append(tmp_path / f"m{gain}.pt")
        torch.jit.script(_Halves(gain)).save(str(members[-1]))
    for m in members:
        bank.CATEGORIES_OF[m] = list(NAMES)
    bank.fill_cache([path], members, tmp_path / "cache", "cpu", flip=True)

    def ensemble(x):
        logits = np.mean([m.numpy() for m in (_Halves(g)(torch.from_numpy(x), True, None)[0]
                                               for g in (1.0, 3.0))], axis=0)  # fmt: skip
        return torch.log_softmax(torch.from_numpy(logits), dim=1).numpy()

    plain = bank.load_file(path, members, list(NAMES), tmp_path / "cache", flip=False)["logp"]
    flipped = bank.load_file(path, members, list(NAMES), tmp_path / "cache", flip=True)["logp"]
    assert np.allclose(plain, ensemble(frames), atol=1e-5)
    mirror = ensemble(np.ascontiguousarray(frames[:, :, ::-1]))
    assert np.allclose(flipped, (ensemble(frames) + mirror) / 2.0, atol=1e-5)
    assert not np.allclose(flipped, plain)

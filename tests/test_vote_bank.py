"""``lerobot_stack/vote_bank.py`` (MS7g design sections 6.1 and 6.3): the frame bank tool applies
the loop's served and fused rules, groups a cycle's views into lift events and fits the vote."""

import random

import numpy as np
import pytest

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

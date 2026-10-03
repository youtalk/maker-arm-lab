"""MS7g design section 6.3: the fused vote ``--vote_weights`` (``loop.fused_class``), the loop with
the switch on and off, and its refusals."""

import json

import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, loop, parts
from maker_arm_lab.core.eval_loop import EvalFailure
from test_classifier_loop import SWITCHES, _classifier, _scene, _usage
from test_perceived_loop import _rollout, _served

NAMES = parts.PART_CATEGORIES


def _vote(o=1.0, lift=1.0, carry=1.0, bias=None, names=NAMES) -> dict:
    return {
        "weights": {"overhead": o, "lift": lift, "carry": carry},
        "bias": {n: (bias or {}).get(n, 0.0) for n in names},
    }


def test_the_fused_class_weighs_each_view_given_and_adds_the_bias():
    names = ("a", "b", "c")
    o, lift, carry = [-0.1, -2.0, -3.0], [-3.0, -0.2, -2.0], [-3.0, -0.3, -2.0]
    vote = _vote(names=names)
    assert loop.fused_class(names, vote, o) == "a"
    assert loop.fused_class(names, vote, o, lift) == "b"  # -2.2 against a's -3.1
    assert loop.fused_class(names, vote, o, lift, carry) == "b"
    assert loop.fused_class(names, _vote(o=3.0, names=names), o, lift, carry) == "a"
    assert loop.fused_class(names, _vote(names=names, bias={"c": 9.0}), o, lift, carry) == "c"
    assert loop.fused_class(names, _vote(names=names, bias={"b": 1.0})) == "b"  # no view yet
    assert loop.fused_class(names, vote, None, lift) == "b"  # a missing view drops out


def _run(tmp_path, vote=None):
    """The classifier scene with the served vote: overhead reads slope, the black wrist frames
    wide, so every lift splits and the carry view decides."""
    env, rig, policy = _scene()
    served = _served(env)
    stub = _classifier(tmp_path, "slope", "wide")
    kwargs = {} if vote is None else {"vote_weights": vote}
    payload = _rollout(
        env, rig, policy, perceived=True, classifier=stub, class_views=True, **kwargs, **SWITCHES
    )
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return payload, served, env


def test_off_is_the_served_vote_and_on_fuses_the_views(tmp_path):
    off, off_served, env = _run(tmp_path)
    none, none_served, _ = _run(tmp_path, vote=None)
    assert len(none_served) == len(off_served) > 0
    assert json.dumps(none, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert "vote_weights" not in off["inference"]["loop"]
    rows = [r for e in off["episodes"] for r in e["cycle_census"] if r["lift_class"]]
    assert rows and {r["taken_class"] for r in rows if not r["vote_lost"]} == {"wide"}
    # Three times the overhead weight: slope wins the sum of slope, wide, wide at every view.
    names = parts.BASE_CLASSES
    on, on_served, _ = _run(tmp_path, vote=_vote(o=3.0, names=names))
    assert on["inference"]["loop"]["vote_weights"] == _vote(o=3.0, names=names)
    rows = [r for e in on["episodes"] for r in e["cycle_census"] if r["lift_class"]]
    assert rows and {r["taken_class"] for r in rows} == {"slope"}
    assert all(r["read_class"] == "slope" and r["lift_class"] == "wide" for r in rows)
    lifted = on["perception"]["lifts_classified"] + on["perception"]["lifts_empty"]
    assert on["perception"]["third_views"] + on["perception"]["votes_lost"] == lifted
    slope = on["episodes"][0]["class_to_box"]["slope"]  # the env redraws at its end
    poses = env.perceived_box_poses[0]
    assert all(np.allclose(b, poses[slope]) for _, b, _, _, _ in on_served)
    # The bias alone can move the box at the selection, before any view agrees.
    biased, biased_served, _ = _run(tmp_path, vote=_vote(names=names, bias={"tall": 9.0}))
    tall = biased["episodes"][0]["class_to_box"]["tall"]
    assert any(np.allclose(b, poses[tall]) for _, b, _, _, _ in biased_served)
    assert {r["taken_class"] for e in biased["episodes"] for r in e["cycle_census"]} <= {
        "tall",
        None,
    }


def test_the_switch_needs_the_vote_s_views_and_values(tmp_path, capsys):
    env, rig, policy = _scene()
    stub = _classifier(tmp_path, "slope", "wide")
    vote = _vote(names=parts.BASE_CLASSES)
    per_class = {v: dict.fromkeys(parts.BASE_CLASSES, 1.0) for v in loop.FUSED_VIEWS}
    bad = (
        {"classifier": stub},
        {"classifier": stub, "class_views": True, "carry_votes": 2},
        {"classifier": stub, "class_views": True, "vote_weights": {"weights": {}, "bias": {}}},
        # ledger D7's v2 form: a weight a view and class, which this loop does not serve
        {"classifier": stub, "class_views": True, "vote_weights": vote | {"weights": per_class}},
    )
    for kwargs in bad:
        kwargs = {"vote_weights": vote} | kwargs
        with pytest.raises(EvalFailure, match="vote_weights"):
            _rollout(env, rig, policy, perceived=True, **kwargs, **SWITCHES)
    env.close()
    heap = ["--heap_k", "3", "--desk_boxes", "4", "--box_draw", "4", "--gripper", "continuous"]
    flags = heap + ["--perceived", "--categories", "--classifier", "m.pt"]
    line = _usage("mj_eval_heap", flags + ["--vote_weights", "v.json"], capsys)
    assert line.startswith("EVAL_FAIL ") and "--class_views" in line, line
    missing = str(tmp_path / "missing.json")
    line = _usage("mj_eval_heap", flags + ["--class_views", "--vote_weights", missing], capsys)
    assert line.startswith("EVAL_FAIL ") and "--vote_weights" in line, line


def test_the_carry_guard_holds_a_part_while_the_fused_vote_is_pending(tmp_path, monkeypatch):
    """Every view reads wide, the bricks' class, but a slope bias that the overhead and lift views
    do not outweigh keeps the slope box until the carry view (CARRY_VIEW_STEPS 100 here), and the
    teacher opens over the slope box before it. No view splits, yet the fused vote is pending
    from the lift to the carry view. Off, the part falls into the slope box. With ``carry_guard``
    the guard holds the open while the vote is pending, the carry view moves the box to wide,
    and the still guard hands the part to the rig, which places it there."""
    monkeypatch.setattr(loop, "CARRY_VIEW_STEPS", 100)
    stub = _classifier(tmp_path, "wide", "wide")
    # The view logits differ by 1: slope leads by 2 at the selection and 1 at the lift, and
    # trails by 1 once the carry view adds its weight of 2.
    vote = _vote(carry=2.0, names=parts.BASE_CLASSES, bias={"slope": 3.0})
    episodes = []
    for guard in (False, True):
        env, rig, policy = _scene()
        slope = int(env.class_to_box[0][parts.BASE_CLASSES.index("slope")])
        x, y = (float(v) for v in env.desk_box_poses[0, slope, :2])
        policy.rig.oracles[0].release_point = torch.tensor((x, y, desk_boxes.DESK_BOX_RELEASE_Z))
        payload = _rollout(
            env,
            rig,
            policy,
            perceived=True,
            classifier=stub,
            class_views=True,
            vote_weights=vote,
            carry_guard=guard,
            **SWITCHES,
        )
        env.close()
        episodes.append(payload["episodes"][0])
    off, on = episodes
    assert off["cycle_classes"][0] == "wrong_box"
    assert off["cycle_census"][0]["release_place"] == "wrong_box"
    row = on["cycle_census"][0]
    assert on["cycle_classes"][0] == "stowed" and row["handoff"] == "placed", row
    assert row["read_class"] == row["lift_class"] == row["taken_class"] == "wide"
    assert row["guard_vote_steps"] > 0 and row["guard_rim_steps"] == 0
    assert row["guard_steps"] == row["guard_vote_steps"] + loop.GUARD_STALL_STEPS

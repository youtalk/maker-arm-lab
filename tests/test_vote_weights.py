"""MS7g design section 6.3: the fused vote ``--vote_weights`` (``loop.fused_class``), the loop with
the switch on and off, and its refusals; and ledger D10's classifier flip ``--cls_flip``."""

import json
from typing import List, Tuple  # noqa: UP035 -- TorchScript reads the typing names

import numpy as np
import pytest
import torch

from maker_arm_lab.core import desk_boxes, loop, parts, perception
from maker_arm_lab.core.eval_loop import EvalFailure
from test_classifier_loop import SWITCHES, _classifier, _scene, _usage
from test_ms7f_stack import ROOT, _gate_argv, _load, ge
from test_perceived_loop import _rollout, _served
from test_run_gate_eval import _heap_payload

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


def test_the_client_parses_the_guards_and_the_vote_the_supervisor_forwards(tmp_path):
    """The stack serves ``--vote_weights`` with ``--carry_guard`` and ``--overhang_guard``: the
    supervisor forwards each once and keys its cache on each, and the client parses the command
    it forwards, the nine values included."""
    vote = _vote(names=parts.BASE_CLASSES, bias={"slope": 0.5})
    path = tmp_path / "vote.json"
    path.write_text(json.dumps(vote))
    trio = ["--carry_guard", "--overhang_guard", "--vote_weights", str(path)]
    argv = _gate_argv(tmp_path / "gate") + ["--tray", "--gripper", "continuous", "--perceived"]
    argv += ["--box_draw", "4", "--categories", "--classifier", "m.pt", "--class_views"]
    cfg = ge._build_cfg(ge._parse_args(argv + trio))
    draw = ge.make_eval_draws(1, 1)[0]
    cmd = ge._eval_command(cfg, draw)
    assert all(cmd.count(flag) == 1 for flag in trio), cmd
    assert cmd[cmd.index("--vote_weights") + 1] == str(path)
    client = _load(ROOT / "scripts" / "mj_eval_heap.py")
    args = client.parse_args(cmd[cmd.index("scripts/mj_eval_heap.py") + 1 :])
    assert args.carry_guard and args.overhang_guard and args.class_views and args.vote == vote
    switches = ge.loop_switches(cfg.client_flags)
    ms7f = ge.ms7f_block(cfg)
    assert switches["carry_guard"] and switches["overhang_guard"]
    assert ms7f["vote_weights"] == str(path)
    out = tmp_path / "gate" / "eval_000.json"
    out.parent.mkdir()

    def hit(loop_block, block):
        inference = {"n_action_steps": 50, "temporal_ensemble_coeff": None, "loop": loop_block}
        heap = {"tray": True, "desk_boxes": 4, "gripper": "continuous", "ms7e": ge.ms7e_block(cfg)}
        payload = _heap_payload(draw, inference=inference, ms7f=block, **heap)
        out.write_text(json.dumps(payload))
        return ge._cached_payload(cfg, draw) is not None

    assert hit(switches, ms7f)
    assert not hit(switches, ms7f | {"vote_weights": None})
    for name in ("carry_guard", "overhang_guard"):
        assert not hit(switches | {name: False}, ms7f), name


class _Sided(torch.nn.Module):
    """Class 0's logit is the frame's left half mean brightness and class 1's the right half's: a
    mirrored frame reads the other way round."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float

    def __init__(self):
        super().__init__()
        self.categories = list(parts.BASE_CLASSES)
        self.passenger_threshold = 0.0

    def forward(self, frames: torch.Tensor, wrist: bool) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        x = frames.float().mean(dim=(1, 3)) / 255.0
        logits = torch.zeros(frames.shape[0], len(self.categories))
        logits[:, 0] = 4.0 * x[:, :112].mean(1)
        logits[:, 1] = 4.0 * x[:, 112:].mean(1)
        return logits, torch.full((frames.shape[0],), -1.0)


def test_the_flip_averages_the_frame_s_and_its_mirror_s_log_probabilities(tmp_path):
    path = tmp_path / "sided.pt"
    torch.jit.script(_Sided()).save(str(path))
    names = parts.BASE_CLASSES
    frame = np.zeros((224, 224, 3), np.uint8)
    frame[:, :112] = 200
    plain = perception.PartClassifier(path)
    flipped = perception.PartClassifier(path, flip=True)
    own, _ = plain.log_probs(frame, names, wrist=True)
    mirror, _ = plain.log_probs(np.ascontiguousarray(frame[:, ::-1]), names, wrist=True)
    both, _ = flipped.log_probs(frame, names, wrist=True)
    assert own[0] > own[1] and mirror[1] > mirror[0]
    assert np.allclose(both, (np.array(own) + np.array(mirror)) / 2.0, atol=1e-6)
    assert not plain.flip and flipped.flip


def test_with_a_mirror_symmetric_model_the_flip_changes_no_decision(tmp_path, capsys):
    """The scene's stub reads mean brightness, the same on a frame and its mirror: with the flip
    every served step and payload byte are the run's without it, but the inference block's
    ``cls_flip``, which only the flipped run records."""
    off, off_served, _ = _run(tmp_path)
    env, rig, policy = _scene()
    served = _served(env)
    stub = _classifier(tmp_path, "slope", "wide")
    stub.flip = True
    on = _rollout(env, rig, policy, perceived=True, classifier=stub, class_views=True, **SWITCHES)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        on.pop(key)
    assert (
        on["inference"]["loop"].pop("cls_flip") is True
        and "cls_flip" not in off["inference"]["loop"]
    )
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert len(served) == len(off_served) > 0
    capsys.readouterr()  # the runs' progress lines
    heap = ["--heap_k", "3", "--desk_boxes", "4", "--box_draw", "4", "--gripper", "continuous"]
    line = _usage("mj_eval_heap", heap + ["--perceived", "--categories", "--cls_flip"], capsys)
    assert line.startswith("EVAL_FAIL ") and "--cls_flip" in line, line

"""The heap loop's MS7f classifier switches (MS7f design sections 6.3 to 6.6, 12.2): a stub
TorchScript model drives the box, the passenger response opens the jaws only over the tray
interior, the stand-in's draws stay paired, and no decision reads the truth."""

import importlib.util
import math
from pathlib import Path
from typing import List, Tuple  # noqa: UP035 -- TorchScript reads the typing names

import numpy as np
import pytest
import torch

from maker_arm_lab.core import layout, loop, parts, perception
from maker_arm_lab.core.eval_loop import EvalFailure
from test_mj_heap_oracle import _place
from test_perceived_loop import (
    MS7D_ON,
    SPREAD,
    TRUTH_DECISION_READS,
    _env_and_rig,
    _garbage_reads,
    _raises,
    _rollout,
    _served,
    _Teacher,
    _TruthlessEnv,
)

SWITCHES = {
    "select_at_release": True,
    "squeeze_floor": True,
    "square_wrist": True,
    "pick_snap": True,
    "dwell_close": True,
    "part_failures": True,
    "park_done": True,
}
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BRIGHT = 255  # the stub's overhead frame; a cameras-off env serves black wrist frames


class _Stub(torch.nn.Module):
    """Reads class ``bright`` on a frame brighter than mid grey and class ``dark`` otherwise,
    and a passenger at the fixed logit ``rider`` (the operating point is 0)."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float

    def __init__(self, names, bright: int, dark: int, rider: float):
        super().__init__()
        self.categories = list(names)
        self.passenger_threshold = 0.0
        self.bright, self.dark, self.rider = bright, dark, rider

    def forward(self, frames: torch.Tensor, wrist: bool) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        n = frames.shape[0]
        lit = frames.float().mean() > 127.0
        logits = torch.zeros(n, len(self.categories))
        logits[:, self.bright if bool(lit) else self.dark] = 1.0
        return logits, torch.full((n,), self.rider)


class _SizedStub(torch.nn.Module):
    """A ``size_input`` model: class ``over`` on the overhead view and ``wrist`` on a wrist
    frame, and an error on an unknown size (-1), so a run only ends if every view got one."""

    categories: List[str]  # noqa: UP006
    passenger_threshold: float
    size_input: bool

    def __init__(self, names, over: int, wrist: int):
        super().__init__()
        self.categories = list(names)
        self.passenger_threshold = 0.0
        self.size_input = True
        self.over, self.wrist = over, wrist

    def forward(
        self, frames: torch.Tensor, wrist: bool, size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: UP006
        if bool((size < 0.0).any()):
            raise RuntimeError("no perceived size")
        logits = torch.zeros(frames.shape[0], len(self.categories))
        logits[:, self.wrist if wrist else self.over] = 1.0
        return logits, torch.full((frames.shape[0],), -1.0)


def _classifier(tmp_path, bright="wide", dark="wide", rider=-1.0, names=parts.BASE_CLASSES):
    path = tmp_path / f"stub_{bright}_{dark}_{rider}.pt"
    stub = _Stub(names, names.index(bright), names.index(dark), rider)
    torch.jit.script(stub).save(str(path))
    return perception.PartClassifier(path)


def _scene(tray=False):
    """The three bricks of the perceived teacher scene on the bare desk, or in the tray; the
    cameras are off, so the narrow view is a stand-in frame."""
    env, rig = _env_and_rig(tray=tray)
    _place(env, SPREAD)
    env.narrow_overhead = lambda i, point: np.full((224, 224, 3), BRIGHT, dtype=np.uint8)
    return env, rig, _Teacher(env, rig.oracle_factory)


def test_over_tray_reads_the_interior_at_the_rim():
    (x0, x1), (y0, _) = layout.tray_bounds()
    grow = layout.TRAY_WALL_HEIGHT * math.tan(math.radians(30.0))
    assert loop.over_tray(layout.TRAY_CENTER, 30.0)
    assert loop.over_tray((x1 + 0.9 * grow, y0 - 0.9 * grow), 30.0)
    assert not loop.over_tray((x1 + 0.9 * grow, 0.0), 0.0)
    assert not loop.over_tray((x0 - 1.1 * grow, 0.0), 30.0)
    assert not loop.over_tray((0.12, 0.19), 30.0)  # a desk box


def test_the_switches_need_perceived_and_the_run_s_classes(tmp_path):
    env, rig, policy = _scene()
    stub = _classifier(tmp_path)
    with pytest.raises(EvalFailure, match="classifier"):
        _rollout(env, rig, policy, classifier=stub, **SWITCHES)
    with pytest.raises(EvalFailure, match="passenger_drop"):
        _rollout(env, rig, policy, perceived=True, passenger_drop=True, **SWITCHES)
    other = _classifier(tmp_path, "brick", "brick", names=parts.PART_CATEGORIES)
    with pytest.raises(EvalFailure, match="sorts by"):
        _rollout(env, rig, policy, perceived=True, classifier=other, **SWITCHES)
    env.close()


def test_a_stub_that_reads_one_class_serves_that_class_s_box(tmp_path):
    """The overhead view sets the box and the wrist view keeps it: every served box row is the
    perceived pose of the slope box, though no part of the scene is a slope."""
    env, rig, policy = _scene()
    served = _served(env)
    payload = _rollout(
        env,
        rig,
        policy,
        perceived=True,
        classifier=_classifier(tmp_path, "slope", "slope"),
        **SWITCHES,
    )
    env.close()
    box = int(env.class_to_box[0][parts.BASE_CLASSES.index("slope")])
    pose = env.perceived_box_poses[0][box]
    assert served and all(np.allclose(b, pose, atol=1e-6) for _, b, _, _, _ in served)
    seen = payload["perception"]
    assert seen["wrong_overhead_class"] == seen["selections"] > 0
    assert seen["wrong_wrist_class"] == seen["lifts_classified"] > 0
    assert set(loop.CLASSIFIER_COUNTS) <= set(seen)
    assert payload["inference"]["loop"]["classifier"].endswith(".pt")
    assert "placed" in payload["episodes"][0]["cycle_reasons"]


def test_the_summed_class_takes_the_largest_summed_log_probability():
    names = ("a", "b", "c")
    assert loop.summed_class(names, [-0.1, -2.0, -3.0]) == "a"
    # Two views lean to b, one is sure of a: the vote of three would take b.
    sure, lean = [-0.05, -3.0, -4.0], [-0.9, -0.6, -3.0]
    assert loop.summed_class(names, sure, lean, lean) == "a"
    assert loop.summed_class(names, [-0.7, -0.7, -3.0], [-1.0, -0.5, -3.0]) == "b"


def test_the_carry_view_is_the_third_vote(tmp_path):
    """Overhead reads slope, the lift and the carry read wide: the vote takes wide at the carry
    view, so the box row moves from the slope box to the wide box once a cycle."""
    env, rig, policy = _scene()
    served = _served(env)
    stub = _classifier(tmp_path, "slope", "wide")
    payload = _rollout(
        env, rig, policy, perceived=True, classifier=stub, class_views=True, **SWITCHES
    )
    env.close()
    seen = payload["perception"]
    assert seen["third_views"] > 0 and seen["votes_changed"] == 0
    to_box = env.class_to_box[0]
    slope, wide = (int(to_box[parts.BASE_CLASSES.index(c)]) for c in ("slope", "wide"))
    poses = env.perceived_box_poses[0]
    rows = [b for _, b, _, _, _ in served]
    assert np.allclose(rows[0], poses[slope]) and any(np.allclose(r, poses[wide]) for r in rows)


def test_a_size_reading_model_gets_the_detection_size_in_every_view(tmp_path):
    """Overhead reads slope and both wrist views wide, so every lift waits for the carry view:
    the run ends only if the overhead, lift and carry reads all got a perceived size."""
    env, rig, policy = _scene()
    names = parts.BASE_CLASSES
    path = tmp_path / "sized.pt"
    torch.jit.script(_SizedStub(names, names.index("slope"), names.index("wide"))).save(str(path))
    payload = _rollout(
        env,
        rig,
        policy,
        perceived=True,
        classifier=perception.PartClassifier(path),
        class_views=True,
        **SWITCHES,
    )
    env.close()
    seen = payload["perception"]
    assert seen["third_views"] > 0 and seen["votes_changed"] == 0


def test_the_passenger_response_opens_the_jaws_only_over_the_tray(tmp_path):
    reasons = {}
    for tray, drop in ((True, True), (True, False), (False, True)):
        env, rig, policy = _scene(tray=tray)
        served = _served(env)
        payload = _rollout(
            env,
            rig,
            policy,
            perceived=True,
            classifier=_classifier(tmp_path, rider=1.0),
            passenger_drop=drop,
            **SWITCHES,
        )
        env.close()
        ep, seen = payload["episodes"][0], payload["perception"]
        reasons[tray, drop] = ep["cycle_reasons"]
        assert seen["passenger_reads"] > 0
        assert seen["passenger_drops"] == reasons[tray, drop].count("passenger")
        if (tray, drop) == (True, True):
            # From the drop on, the loop commands the full gap until the next selection.
            first = ep["cycle_steps"][0]
            assert all(float(a[6]) >= 0.1 for *_, a in served[first + 1 : first + 6])
    assert reasons[True, True][0] == "passenger"
    assert "passenger" not in reasons[True, False] + reasons[False, True]


def test_the_stand_in_draws_the_same_under_the_classifier(tmp_path, monkeypatch):
    """The teacher policy ignores the served box, so the switch changes no step of the run; the
    stand-in's generators must then end where they end without it (design section 16)."""
    made = []
    real = perception.Perceiver.__init__

    def spy(self, *args, **kwargs):
        real(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(perception.Perceiver, "__init__", spy)
    ends = []
    for stub in (None, _classifier(tmp_path, "slope", "narrow")):
        env, rig, policy = _scene()
        _rollout(env, rig, policy, perceived=True, classifier=stub, **SWITCHES, **MS7D_ON)
        env.close()
        p = made[-1]
        ends.append((p.rng.getstate(), p.views.rng.getstate()))
    assert ends[0] == ends[1]


def test_no_decision_reads_the_truth_under_the_classifier(tmp_path, monkeypatch):
    """``test_perceived_loop.test_no_decision_reads_the_truth`` with both MS7f switches and the
    MS7d levers on, in the tray with a passenger read at every lift."""
    stub = _classifier(tmp_path, "slope", "wide", rider=1.0)
    kwargs = {"perceived": True, "classifier": stub, "passenger_drop": True} | MS7D_ON | SWITCHES

    env, rig, policy = _scene(tray=True)
    served = _served(env)
    first = _rollout(env, rig, policy, **kwargs)
    env.close()

    env, rig, policy = _scene(tray=True)
    blind = _served(env)
    gen = torch.Generator().manual_seed(7)
    monkeypatch.setattr(
        loop, "truth_reads", lambda env, per_part: _garbage_reads(env, per_part, gen)
    )
    for name in TRUTH_DECISION_READS:
        monkeypatch.setattr(loop, name, _raises(name))
    rig.snapshot = _raises("rig.snapshot")
    second = _rollout(_TruthlessEnv(env), rig, policy, **kwargs)
    env.close()

    assert len(blind) == len(served) > 0
    for s, (a, b) in enumerate(zip(served, blind)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), s
        assert torch.equal(a[4], b[4]), s
    ep, blind_ep = first["episodes"][0], second["episodes"][0]
    for key in ("cycle_reasons", "cycle_targets", "cycle_steps", "done_reason"):
        assert ep[key] == blind_ep[key], key
    assert first["perception"] == second["perception"]
    assert first["perception"]["passenger_drops"] > 0


def _usage(script: str, argv: list[str], capsys) -> str:
    spec = importlib.util.spec_from_file_location(script, SCRIPTS / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as exit_info:
        module.main(argv)
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    return out[0]


def test_the_clients_take_the_switches_only_where_the_loop_serves_them(capsys):
    heap = ["--heap_k", "3", "--desk_boxes", "4", "--box_draw", "4", "--gripper", "continuous"]
    line = _usage("mj_eval_heap", heap + ["--perceived", "--classifier", "m.pt"], capsys)
    assert line.startswith("EVAL_FAIL ") and "needs --perceived and --categories" in line
    line = _usage(
        "mj_eval_heap", heap + ["--perceived", "--categories", "--passenger_drop"], capsys
    )
    assert line.startswith("EVAL_FAIL ") and "it needs --classifier" in line
    for flag in (["--classifier", "m.pt"], ["--passenger_drop"]):
        line = _usage("mj_generate_batch", ["--out_dir", "/tmp/x", "--heap_k", "3"] + flag, capsys)
        assert line.startswith("BATCH_FAIL usage: ") and "the recorder has none" in line

"""``scripts/mj_eval_heap.py --classifier_frames`` (``mj_render_set.PolicyFrames`` on the loop's
``on_classify``): it records exactly the frames the classifier reads, labelled with the
simulator's truth, as a render set file the trainer loads, and the run is byte-equal without it."""

import argparse
import json
import sys
import types

import numpy as np
import pytest

from maker_arm_lab.core import loop, parts
from maker_arm_lab.core.eval_loop import EvalFailure
from maker_arm_lab.core.hold import HoldDetector
from test_classifier_loop import SWITCHES, _classifier, _scene, _usage
from test_ms7f_stack import ROOT, _load
from test_perceived_loop import _rollout, _served
from test_thin_snap import _same

render = _load(ROOT / "scripts" / "mj_render_set.py")
ARGS = argparse.Namespace(run_seed=11, layout_seed=5, families=(), parts_seed=20260915)
HEAP = ["--heap_k", "3", "--desk_boxes", "4", "--box_draw", "4", "--gripper", "continuous"]


@pytest.fixture(autouse=True)
def _cameras_off(monkeypatch):
    """The scenes build no camera to cast ``_seen``'s rays from: every part reads in the open. The
    client imports the patched module by name."""
    monkeypatch.setattr(render, "_seen", lambda env, i, b, cam=None: 1.0)
    monkeypatch.setitem(sys.modules, "mj_render_set", render)


def _trainer(monkeypatch):
    """``lerobot_stack/train_part_classifier.py`` under this venv: ``load`` reads numpy only, so a
    stand-in torchvision (absent here) is enough to import it."""
    monkeypatch.setitem(sys.modules, "torchvision", types.ModuleType("torchvision"))
    return _load(ROOT / "lerobot_stack" / "train_part_classifier.py")


def _run(tmp_path, record: bool, wrist: str = "wide", bank: bool = False):
    """The classifier scene with the vote and two carry reads: overhead reads slope and the black
    wrist frames ``wrist``, so with "wide" every lift reads all four views and with "slope" every
    lift agrees and no vote runs. Returns the payload without its wall clock, the served steps,
    each read's (frame, wrist, size), each recorded read's truth, the recorder and the env."""
    env, rig, policy = _scene()
    served = _served(env)
    stub = _classifier(tmp_path, "slope", wrist)
    given, truth = [], []
    real = stub.log_probs

    def log_probs(frame, active, wrist=False, size=None):
        given.append((np.array(frame), wrist, None if size is None else sorted(size)))
        return real(frame, active, wrist, size)

    stub.log_probs = log_probs
    out = tmp_path / ("bank" if bank else "frames")
    recorder = render.PolicyFrames(env, rig, str(out), ARGS, crop=False)

    def spy(view, i, frame, size, step, cycle, detector):
        truth.append(cycle.target if view == "overhead" else loop.wrist_class(env, i)[1])
        recorder(view, i, frame, size, step, cycle, detector)

    kwargs = {"perceived": True, "classifier": stub, "class_views": True, "carry_votes": 2}
    if record:
        kwargs["on_classify"] = spy
    if bank:
        kwargs["bank_views"] = True
    payload = _rollout(env, rig, policy, **kwargs, **SWITCHES)
    recorder.flush()
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return payload, served, given, truth, recorder, env


def test_recording_changes_no_step_and_no_payload_byte(tmp_path):
    off, off_served, off_given, _, recorder, _ = _run(tmp_path, record=False)
    assert not recorder.out.exists()  # off records nothing
    on, on_served, on_given, _, _, _ = _run(tmp_path, record=True)
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert _same(on_served, off_served)
    assert len(on_given) == len(off_given) > 0


def test_one_frame_per_read_labelled_with_the_truth_that_the_trainer_loads(tmp_path, monkeypatch):
    _, _, given, truth, recorder, env = _run(tmp_path, record=True)
    z = np.load(recorder.out)
    # No crop: the policy's wrist frames. Each frame's cycle is the loop's ordinal.
    assert set(z.files) == {"frames", "args", "cycle", *render.LABELS}
    assert z["cycle"][0] == 1 and (np.diff(z["cycle"]) >= 0).all()
    assert (
        z["cycle"][z["view"] == "overhead"] == np.arange(1, (z["view"] == "overhead").sum() + 1)
    ).all()
    views = z["view"].tolist()
    assert {"overhead", "lift", "carry", "wrist"} <= set(views)
    assert len(views) == len(given) == len(truth)
    assert z["frames"].dtype == np.uint8 and z["frames"].shape == (len(views), 224, 224, 3)
    sizes = np.load(recorder.out.with_name(recorder.out.stem + ".size.npy"))
    for n, (frame, wrist, size) in enumerate(given):
        assert np.array_equal(z["frames"][n], frame), n
        assert wrist == (views[n] != "overhead"), n
        assert np.allclose(sizes[n], size), n
    assert z["held"].tolist() == truth
    heldout = render.heldout_names(ARGS)
    for n, b in enumerate(truth):
        name = env.parts[b].name if b >= 0 else ""
        assert z["part"][n] == name and bool(z["heldout"][n]) == (name in heldout), n
        assert z["category"][n] == (parts.part_category(env.parts[b]) if b >= 0 else ""), n
    over = z["view"] == "overhead"
    assert (z["passengers"][over] == -1).all() and (z["riders"][over] == -1).all()
    assert (z["part_seen"][over] == 1.0).all()
    assert (z["part_seen"][~over] == -1.0).all()
    assert ((0 <= z["riders"][~over]) & (z["riders"][~over] <= z["passengers"][~over])).all()
    assert json.loads(str(z["args"]))["run_seed"] == ARGS.run_seed
    data = _trainer(monkeypatch).load([recorder.out])
    assert data["wrist_crop"] is None and len(data["view"]) == len(views)
    assert data["size"].shape == (len(views), 3)


def test_bank_views_record_the_agreed_carry_view_and_change_no_step_and_no_payload_byte(
    tmp_path,
):
    off, off_served, off_given, _, _, _ = _run(tmp_path, record=True, wrist="slope")
    on, on_served, on_given, truth, recorder, _ = _run(
        tmp_path, record=True, wrist="slope", bank=True
    )
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)
    assert _same(on_served, off_served)
    assert len(on_given) == len(off_given) > 0  # the classifier reads no bank frame
    z = np.load(recorder.out)
    views, steps, cycles = z["view"].tolist(), z["step"].tolist(), z["cycle"].tolist()
    assert "carry" not in views and "wrist" not in views  # every lift agreed: no vote ran
    banks = [n for n, v in enumerate(views) if v == "bank"]
    assert banks and len(views) == len(on_given) + len(banks)
    for n in banks:  # the bank frame is its cycle's, CARRY_VIEW_STEPS after the lift before it
        lift = max(m for m in range(n) if views[m] == "lift")
        assert cycles[lift] == cycles[n] and steps[n] == steps[lift] + loop.CARRY_VIEW_STEPS
    assert z["held"].tolist() == truth  # the held part's labels, as on every wrist view
    assert "bank" not in recorder.summary()  # ARGS carries no bank_views
    recorder.bank = True
    assert recorder.summary()["bank"] == len(banks)


def test_bank_views_need_the_recorder_and_the_classifier(tmp_path, capsys):
    env, rig, policy = _scene()
    stub = _classifier(tmp_path, "slope", "slope")
    for kwargs in ({"classifier": stub}, {"on_classify": lambda *a: None}):
        with pytest.raises(EvalFailure, match="bank_views"):
            _rollout(env, rig, policy, perceived=True, bank_views=True, **kwargs, **SWITCHES)
    env.close()
    flags = HEAP + ["--perceived", "--categories", "--classifier", "m.pt", "--bank_views"]
    line = _usage("mj_eval_heap", flags, capsys)
    assert line.startswith("EVAL_FAIL ") and "needs --classifier_frames" in line, line


def test_a_stopped_run_keeps_the_frames_of_its_last_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "FLUSH_EVERY", 3)
    env, rig, policy = _scene()
    recorder = render.PolicyFrames(env, rig, str(tmp_path), ARGS, crop=False)
    stub = _classifier(tmp_path, "slope", "wide")
    _rollout(env, rig, policy, perceived=True, classifier=stub, on_classify=recorder, **SWITCHES)
    env.close()
    kept = len(np.load(recorder.out)["view"])  # the file as a stopped run would leave it
    assert kept >= 3 and kept % 3 == 0 and kept > len(recorder.frames) - 3


def test_the_client_needs_the_classifier_and_perceived(capsys):
    for extra in ([], ["--perceived", "--categories"]):
        line = _usage("mj_eval_heap", HEAP + extra + ["--classifier_frames", "/tmp/x"], capsys)
        assert line.startswith("EVAL_FAIL ") and "--classifier_frames" in line, line
    line = _usage(
        "mj_eval_heap", HEAP + ["--classifier", "m.pt", "--classifier_frames", "/x"], capsys
    )
    assert line.startswith("EVAL_FAIL ") and "needs --perceived" in line, line


class _Policy:
    def close(self):
        pass


@pytest.mark.parametrize("ends", ["early", "done"])
def test_the_client_writes_the_frames_when_the_run_ends(ends, tmp_path, monkeypatch, capsys):
    """The heap client builds the recorder, hands it to the rollout and writes its file whether
    the rollout returns or fails; only the recording run's payload names it."""
    module = _load(ROOT / "scripts" / "mj_eval_heap.py")
    monkeypatch.setattr(module.policy_client, "ServerPolicy", lambda server: _Policy())
    real_env = module.MjLegoSortEnv
    monkeypatch.setattr(
        module, "MjLegoSortEnv", lambda *a, **k: real_env(*a, **{**k, "cameras": False})
    )
    model = _classifier(tmp_path).path

    def run(env, rig, policy, *, on_classify, **kwargs):
        if on_classify is not None:
            cycle = loop.Cycle(ordinal=1, target=0, start_step=0, stowed_before=set())
            frame = np.full((224, 224, 3), 7, dtype=np.uint8)
            on_classify("overhead", 0, frame, (0.03, 0.01, 0.02), 0, cycle, HoldDetector())
        if ends == "early":
            raise EvalFailure("wall_clock", "stopped")
        return {}

    argv = HEAP + ["--perceived", "--categories", "--classifier", model, "--num_envs", "1"]
    argv += ["--run_seed", "4", "--layout_seed", "9"]
    for record in (False, True):
        extra = ["--classifier_frames", str(tmp_path / "f")] if record else []
        assert module.main(argv + extra, run=run) == (1 if ends == "early" else 0)
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert line.startswith("EVAL_FAIL " if ends == "early" else "EVAL_OK "), line
        if ends == "done":
            payload = json.loads(line.split(" ", 1)[1])
            assert ("classifier_frames" in payload) == record
    assert sorted(f.name for f in (tmp_path / "f").iterdir()) == [
        "policy_4_9.npz",
        "policy_4_9.size.npy",
    ]
    z = np.load(tmp_path / "f" / "policy_4_9.npz")
    assert z["view"].tolist() == ["overhead"] and int(z["held"][0]) == 0
    assert np.allclose(np.load(tmp_path / "f" / "policy_4_9.size.npy"), [[0.01, 0.02, 0.03]])

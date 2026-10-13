"""MS7i ``record_all`` (rev3 design 5.1): the heap loop records every frame of the episode with
its census cycle ordinal, subtask and the server's reply, and changes no served step. MuJoCo
three-part heap with four desk boxes, cameras off."""

import json

import h5py
import numpy as np
import pytest
import torch

from maker_arm_lab.core import dagger_loop, hold, loop
from maker_arm_lab.core.gate import policy_state
from maker_arm_lab.mj.recorder import ShardRecorder
from maker_arm_lab.robots.frames import GRIPPER_OPEN_POS
from test_heap_loop import SPREAD
from test_mj_heap_oracle import _place
from test_ms7f_dagger import ARGV, FAILURE, _wiring
from test_ms7f_stack import ROOT, _load
from test_perceived_loop import CARRY_DROP, _env_and_rig, _rollout, _served, _Still
from test_perceived_loop import _teacher_scene as _teacher
from test_thin_snap import _same

EVAL_HEAP = ROOT / "scripts" / "mj_eval_heap.py"
EXTRAS = ("loop_cycle", "subtask", "policy_action")


def _still(monkeypatch, **env_kwargs):
    """A still policy under the failure triggers: each cycle ends in a takeover."""
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 60)
    env, rig = _env_and_rig(**env_kwargs)
    _place(env, SPREAD)
    return env, rig, _Still(), 4, {"takeover": FAILURE, "select_at_release": True}


def _scene(name, monkeypatch, **env_kwargs):
    """``still``: takeovers; ``teacher``: the teacher as the policy, every cycle lifted and
    released by the policy; ``carry``: its releases away from the boxes, handed to the rig."""
    if name == "still":
        return _still(monkeypatch, **env_kwargs)
    env, rig, policy, budget, switches = _teacher(monkeypatch, **env_kwargs)
    if name == "carry":  # test_perceived_loop._carry_scene
        policy.rig.oracles[0].release_point = torch.tensor(CARRY_DROP)
        switches["carry_guard"] = True
    return env, rig, policy, budget, switches


def _run(name, monkeypatch, tmp_path, on: bool):
    """The scene's perceived rollout recorded to ``tmp_path/<name>_<on>.hdf5``: the payload
    without its wall clock, what env 0 was served, the shard path and the rig's IK."""
    path = tmp_path / f"{name}_{int(on)}"
    env, rig, policy, budget, switches = _scene(
        name,
        monkeypatch,
        export_dir=path.parent,
        shard_name=path.name,
        export_all=True,
        record_all=on,
    )
    served = _served(env)
    payload = _rollout(env, rig, policy, budget=budget, perceived=True, record_all=on, **switches)
    env.close()
    for key in ("wall_sec", "wall_sec_per_episode"):
        payload.pop(key)
    return json.loads(json.dumps(payload, sort_keys=True)), served, path.with_suffix(".hdf5"), rig


def _replayed(demo, ik) -> list[int]:
    """``hold.subtask_ids`` from what the shard records, as an offline pass reads it."""
    q = torch.from_numpy(demo["obs/joint_pos_abs"][:])
    frac = policy_state(q)[:, 6].tolist()
    gap = [2.0 * GRIPPER_OPEN_POS * f for f in frac]
    first = float(q[0, 6] + q[0, 7])  # the loop's first command: the jaws' own gap
    cmds = demo["actions"][:, 6].tolist()
    prev = [first] + [hold.commanded_gap(c, "continuous") for c in cmds[:-1]]
    pick_z = [float(loop.commanded_pick(ik, row[:6])[2]) for row in q.numpy()]
    width = demo["obs/context_target"][:, 5].tolist()
    return hold.subtask_ids(gap, prev, pick_z, width, demo["obs/loop_cycle"][:].tolist())


def _check(payload, path, ik):
    """Every frame is a loop step, every census row joins its frames, and the recorded subtask
    is the offline replay on every policy frame."""
    ep = payload["episodes"][0]
    rows = ep["cycle_census"]
    with h5py.File(path, "r") as f:
        assert list(f["data"]) == ["demo_0"]
        demo = f["data/demo_0"]
        n = demo.attrs["num_samples"]
        assert n == ep["steps"] and demo.attrs["env_index"] == 0
        cycle = demo["obs/loop_cycle"][:]
        subtask = demo["obs/subtask"][:]
        actor = demo["obs/actor"][:]
        phase = demo["obs/cycle_phase"][:]
        served = demo["obs/policy_action"][:]
        assert (cycle.dtype, subtask.dtype, served.shape) == (np.int16, np.int8, (n, 7))
        replay = _replayed(demo, ik)
    positive = cycle[cycle > 0]
    assert (np.diff(positive) >= 0).all() and set(positive) <= set(range(1, len(rows) + 1))
    for j, row in enumerate(rows):
        frames = np.flatnonzero(cycle == j + 1)
        assert len(frames) == row["open_steps"] + row["closed_steps"], (j, row)
        assert len(frames) == 0 or frames[-1] - frames[0] + 1 == len(frames), j
        assert {"box", "last_held_at", "release_at"} <= set(row)
    assert len(ep["final_box"]) == payload["heap_k"]
    policy = phase == -1
    assert (actor[policy] == 0).all() and policy.any()
    assert not served[~policy].any() and served[policy].any()
    mine = actor == 0
    assert (subtask[mine] == np.asarray(replay)[mine]).all()
    assert set(subtask.tolist()) <= set(range(len(hold.SUBTASKS)))
    return ep, cycle, subtask, actor, phase


def test_a_teacher_policys_frames_join_their_rows_and_replay_their_subtasks(monkeypatch, tmp_path):
    payload, _, path, rig = _run("teacher", monkeypatch, tmp_path, True)
    ep, cycle, subtask, actor, phase = _check(payload, path, rig.ik)
    assert (actor == 0).all() and (phase == -1).all()  # the policy composed every frame
    # Each lifted cycle reads carry and then release on its own frames.
    rows = ep["cycle_census"]
    lifted = [j for j, r in enumerate(rows) if r["max_rise"] >= hold.LIFT_MIN_RISE]
    assert lifted, rows
    for j in lifted:
        names = {hold.SUBTASKS[s] for s in subtask[cycle == j + 1]}
        assert {"reach", "grasp", "carry"} <= names, (j, names)
        assert rows[j]["release_at"] is not None and rows[j]["last_held_at"] is not None


def test_takeovers_keep_their_actor_and_the_switch_changes_no_step(monkeypatch, tmp_path):
    on, on_served, on_path, rig = _run("still", monkeypatch, tmp_path, True)
    off, off_served, off_path, _ = _run("still", monkeypatch, tmp_path, False)
    ep, _, _, actor, phase = _check(on, on_path, rig.ik)
    assert ep["takeovers"] >= 1 and set(np.abs(actor[actor != 0])) <= set(
        range(1, ep["takeovers"] + 1)
    )
    assert (phase[actor != 0] >= 0).all()
    # Off: the same served steps and payload bytes but the extras, and the stock shard.
    assert _same(on_served, off_served)
    for row in on["episodes"][0]["cycle_census"]:
        for key in loop.OWN_FRAMES_CENSUS:
            row.pop(key)
    on["episodes"][0].pop("final_box")
    assert on == off
    with h5py.File(on_path, "r") as f_on, h5py.File(off_path, "r") as f_off:
        a, b = f_on["data/demo_0"], f_off["data/demo_0"]
        assert "env_index" not in b.attrs and not set(EXTRAS) & set(b["obs"])
        kept = a["obs/actor"][:] != 0
        assert b.attrs["num_samples"] == kept.sum()
        for key in ("actions", "joint_targets"):
            assert np.array_equal(a[key][:][kept], b[key][:])
        for key in b["obs"]:
            assert np.array_equal(a["obs"][key][:][kept], b["obs"][key][:]), key


def test_a_handoffs_rig_frames_are_policy_actor_frames_with_a_phase(monkeypatch, tmp_path):
    payload, _, path, rig = _run("carry", monkeypatch, tmp_path, True)
    ep, cycle, _, actor, phase = _check(payload, path, rig.ik)
    rows = ep["cycle_census"]
    handed = [j for j, r in enumerate(rows) if r["handoff"] in ("placed", "lifted_lost", "failed")]
    assert handed, rows
    for j in handed:
        frames = cycle == j + 1
        assert ((phase >= 0) & frames).any() and (actor[frames] == 0).all()


def test_record_all_refuses_an_env_without_it_and_the_recorder_skips_a_closed_episode(tmp_path):
    env, rig = _env_and_rig()
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, _Still(), perceived=True, record_all=True)
    env.close()
    assert err.value.reason == "record_all"
    rec = ShardRecorder(tmp_path / "s.hdf5", 1, export_all=True, cameras=False)
    rec.pre_step(0, np.zeros(8), np.zeros(7), -1, -1, None, None)  # never begun: no frame
    rec.post_step(0, np.zeros(8), False)
    assert rec.end(0, True) is False
    rec.close()


def test_the_client_wires_record_all_and_refuses_it_without_dagger_dir(
    tmp_path, monkeypatch, capsys
):
    seen = _wiring(monkeypatch, ARGV + ["--dagger_dir", str(tmp_path), "--record_all"])
    assert seen["env"]["record_all"] is True and seen["rollout"]["record_all"] is True
    off = _wiring(monkeypatch, ARGV + ["--dagger_dir", str(tmp_path)])
    assert off["env"]["record_all"] is False and off["rollout"]["record_all"] is False
    capsys.readouterr()
    module = _load(EVAL_HEAP)
    with pytest.raises(SystemExit) as exit_info:
        module.main(ARGV + ["--record_all"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and "--dagger_dir" in out[0]

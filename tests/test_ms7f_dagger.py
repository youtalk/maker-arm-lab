"""MS7f wave 3 (ledger D17): the served heap loop hands failing cycles to the teacher under
``perceived`` and records only the takeover frames (``scripts/mj_eval_heap.py --dagger_dir``).
Off, the client builds the env, the rig and the rollout exactly as before."""

import random

import h5py
import numpy as np
import pytest

from helpers import ROOT, load_script
from maker_arm_lab.core import dagger_loop, loop, oracle, perception
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_heap_loop import SPREAD
from test_mj_heap_oracle import _place
from test_perceived_loop import _env_and_rig, _rollout, _Still

EVAL_HEAP = ROOT / "scripts" / "mj_eval_heap.py"
FAILURE = loop.TakeoverConfig(random_interval=None, held_no_rise_steps=loop.HELD_NO_RISE_STEPS)


def test_a_perceived_heap_hands_a_still_policy_to_the_teacher_and_records_only_its_frames(
    tmp_path, monkeypatch
):
    """The jaws never close, so ``no_close`` fires; the teacher stows the target and the cycle
    ends ``placed`` at the hand-back. The shard holds takeover frames only, and every row they
    carry is one the rig built from the loop's own perception."""
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 60)
    env, rig = _env_and_rig(export_dir=tmp_path, shard_name="d", export_all=True)
    _place(env, SPREAD)
    spare = perception.Perceiver(11, 0, None, env.class_names)
    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        random.Random(1),
        oracle_factory=rig.oracle_factory,
        perceivers=[spare],
    )
    rows = set()
    built = rig._perceived_row

    def spy(*args):
        row, point = built(*args)
        rows.add(tuple(np.float32(row).tolist()))
        return row, point

    monkeypatch.setattr(rig, "_perceived_row", spy)
    payload = _rollout(env, rig, _Still(), perceived=True, takeover=FAILURE, select_at_release=True)
    env.close()
    ep = payload["episodes"][0]
    assert rig.perceivers[0] is not spare  # the loop's perception records
    assert payload["triggers"].get("no_close", 0) >= 1, payload["triggers"]
    assert ep["takeovers_stowed"] >= 1 and "placed" in ep["cycle_reasons"], ep
    with h5py.File(tmp_path / "d.hdf5", "r") as f:
        actor = f["data/demo_0"]["obs/actor"][:]
        recorded = f["data/demo_0"]["obs/context_target"][:]
    assert (actor != 0).all(), "a policy frame was recorded"
    assert set(abs(actor).tolist()) <= set(range(1, ep["takeovers"] + 1))
    kept = {tuple(r.tolist()) for r in recorded[actor > 0]}
    assert kept and kept <= rows, (kept - rows, rows)


def test_a_declined_target_hands_the_teacher_the_part_it_ranks_first(monkeypatch):
    """When the teacher ranks no grasp on the loop's target, the wave 3 takeover works the
    teacher's own first part, and the loop books its target failed."""
    monkeypatch.setattr(dagger_loop, "NO_CLOSE_STEPS", 60)
    monkeypatch.setattr(HeapOracle, "aim_hover_at", lambda self, scene, target: False)
    env, rig = _env_and_rig()
    _place(env, SPREAD)
    payload = _rollout(env, rig, _Still(), perceived=True, takeover=FAILURE, select_at_release=True)
    env.close()
    ep = payload["episodes"][0]
    assert payload["takeovers_any_part"] == ep["takeovers"] >= 1, payload["triggers"]
    assert ep["takeovers_stowed"] >= 1 and "placed" in ep["cycle_reasons"], ep
    assert "takeover_declined" not in ep["cycle_reasons"], ep


def test_takeover_under_perceived_is_no_longer_refused_but_scripted_place_still_is():
    env, rig = _env_and_rig()
    with pytest.raises(loop.EvalFailure) as err:
        _rollout(env, rig, _Still(), perceived=True, takeover=FAILURE, scripted_place=True)
    env.close()
    assert err.value.reason == "perceived" and "scripted_place" in err.value.detail


ARGV = [
    "--num_envs",
    "1",
    "--heap_k",
    "3",
    "--parts_split",
    "train",
    "--run_seed",
    "11",
    "--layout_seed",
    "5",
    "--desk_boxes",
    "4",
    "--gripper",
    "continuous",
    "--context",
    "camera_box_target",
    "--perceived",
    "--max_wall_minutes",
    "5",
]


def _wiring(monkeypatch, argv) -> dict:
    """What the heap client hands the env, the rig and the rollout, with cameras off and a
    stand-in server and rollout."""
    module = load_script(EVAL_HEAP)
    seen = {}
    real_env, real_rig = module.MjLegoSortEnv, module.OracleRig

    def env(*args, **kwargs):
        seen["env"] = kwargs
        return real_env(*args, **{**kwargs, "cameras": False})

    def rig(*args, **kwargs):
        seen["rig"] = kwargs
        return real_rig(*args, **kwargs)

    def rollout(env, rig, policy, **kwargs):
        seen["rollout"] = kwargs
        return {}

    class Server:
        def __init__(self, address):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "MjLegoSortEnv", env)
    monkeypatch.setattr(module, "OracleRig", rig)
    monkeypatch.setattr(module, "rollout", rollout)
    monkeypatch.setattr(module.policy_client, "ServerPolicy", Server)
    assert module.main(argv) == 0
    return seen


def test_off_the_client_builds_the_env_the_rig_and_the_rollout_as_before(monkeypatch, capsys):
    seen = _wiring(monkeypatch, ARGV)
    assert seen["env"]["export_dir"] is None and seen["env"]["export_all"] is False
    assert seen["env"]["shard_name"] == "shard"  # the env's own default
    assert seen["rig"]["perceivers"] is None and seen["rollout"]["takeover"] is None
    assert "shard" not in capsys.readouterr().out


def test_dagger_dir_records_the_failure_takeovers_with_the_loops_perception(
    tmp_path, monkeypatch, capsys
):
    seen = _wiring(monkeypatch, ARGV + ["--dagger_dir", str(tmp_path), "--shard_name", "proc_3"])
    assert seen["env"]["export_dir"] == str(tmp_path) and seen["env"]["export_all"] is True
    assert seen["env"]["shard_name"] == "proc_3"
    perceivers = seen["rig"]["perceivers"]
    assert len(perceivers) == 1 and isinstance(perceivers[0], perception.Perceiver)
    assert seen["rollout"]["takeover"] == FAILURE
    assert str(tmp_path / "proc_3.hdf5") in capsys.readouterr().out


def test_dagger_dir_refuses_the_oracle_driver_with_one_fail_token(capsys):
    module = load_script(EVAL_HEAP)
    with pytest.raises(SystemExit) as exit_info:
        module.main(["--heap_k", "3", "--oracle_driver", "--dagger_dir", "/tmp/never"])
    assert exit_info.value.code == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("EVAL_FAIL ") and "needs the server" in out[0]

"""CPU test for the ClampRecorder term: it must read the action term's `clamped` flag and
stay silent when the term has none."""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_RECORDERS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/maker_arm_lab/tasks/manager_based/lego_sort/mdp/recorders.py"
)


class RecorderTerm:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def _load():
    names = [
        "isaaclab",
        "isaaclab.envs",
        "isaaclab.envs.mdp",
        "isaaclab.envs.mdp.recorders",
        "isaaclab.envs.mdp.recorders.recorders_cfg",
        "isaaclab.managers",
        "isaaclab.managers.recorder_manager",
        "isaaclab.utils",
    ]
    mods = {n: types.ModuleType(n) for n in names}
    cfg_mod = mods["isaaclab.envs.mdp.recorders.recorders_cfg"]
    for cls in (
        "InitialStateRecorderCfg",
        "PreStepActionsRecorderCfg",
        "PreStepFlatPolicyObservationsRecorderCfg",
    ):
        setattr(cfg_mod, cls, type(cls, (), {}))
    rm = mods["isaaclab.managers.recorder_manager"]
    rm.RecorderManagerBaseCfg = type("RecorderManagerBaseCfg", (), {})
    rm.RecorderTerm = RecorderTerm
    rm.RecorderTermCfg = type("RecorderTermCfg", (), {})
    mods["isaaclab.utils"].configclass = lambda cls: cls
    sys.modules.update(mods)
    try:
        spec = importlib.util.spec_from_file_location("_recorders_standalone", _RECORDERS_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for n in mods:
            sys.modules.pop(n, None)


_rec = _load()


def _env_with_term(term):
    manager = types.SimpleNamespace(get_term=lambda name: term)
    return types.SimpleNamespace(action_manager=manager)


def test_clamp_recorder_reads_the_flag():
    term = types.SimpleNamespace(clamped=torch.tensor([True, False]))
    rec = _rec.ClampRecorder(cfg=None, env=_env_with_term(term))
    key, value = rec.record_post_step()
    assert key == "obs/clamped"
    assert torch.equal(value, torch.tensor([True, False]))
    assert value is not term.clamped  # a copy, not an alias of the live buffer


def test_clamp_recorder_is_silent_without_the_flag():
    rec = _rec.ClampRecorder(cfg=None, env=_env_with_term(types.SimpleNamespace()))
    assert rec.record_post_step() == (None, None)


def test_datagen_manager_cfg_installs_the_clamp_term():
    assert isinstance(
        _rec.DatagenRecorderManagerCfg.record_post_step_clamped, _rec.ClampRecorderCfg
    )

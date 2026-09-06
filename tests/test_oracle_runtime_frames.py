"""The one piece of oracle_runtime that is pure math: composing the oracle's canonical tool
frame (x = jaw axis, z = tool axis) with the arm's body frame. Loaded by file path with an
isaaclab stand-in, like test_mit_mode_actuator.py."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from maker_arm_lab.robots import frames

_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/maker_arm_lab/tasks/manager_based/lego_sort/oracle_runtime.py"
)


def _load():
    names = ["isaaclab", "isaaclab.utils", "isaaclab.utils.math"]
    mods = {n: types.ModuleType(n) for n in names}
    pkg = types.ModuleType("_lego_sort_stub")
    pkg.__path__ = [str(_PATH.parent)]
    stubs = {**mods, "_lego_sort_stub": pkg}
    # oracle_runtime imports `.mdp.recorders` and `.variants` relatively; give it light stand-ins.
    rec = types.ModuleType("_lego_sort_stub.mdp.recorders")
    rec.PHASE_TO_INT = {}
    mdp = types.ModuleType("_lego_sort_stub.mdp")
    mdp.__path__ = []
    var = types.ModuleType("_lego_sort_stub.variants")
    var.IK_SCALE = (0.002, 0.002, 0.002, 0.01, 0.01, 0.01)
    stubs.update(
        {
            "_lego_sort_stub.mdp": mdp,
            "_lego_sort_stub.mdp.recorders": rec,
            "_lego_sort_stub.variants": var,
        }
    )
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_lego_sort_stub.oracle_runtime", _PATH)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "_lego_sort_stub"
        spec.loader.exec_module(module)
        return module
    finally:
        for n in stubs:
            sys.modules.pop(n, None)
        sys.modules.pop("_lego_sort_stub.oracle_runtime", None)


_rt = _load()


def test_canonical_to_body_identity_when_frames_coincide():
    rot = torch.eye(3)
    assert torch.allclose(_rt.canonical_to_body(rot, torch.eye(3)), rot)


def test_canonical_to_body_maps_maker_arm_axes():
    canon = torch.tensor(frames.CANON_IN_BODY)
    # Canonical frame with tool axis straight down and jaw axis along world +x.
    rot_canon = torch.stack(
        [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, -1.0, 0.0]),
            torch.tensor([0.0, 0.0, -1.0]),
        ],
        dim=1,
    )
    rot_body = _rt.canonical_to_body(rot_canon, canon)
    # Body y (the jaw axis) must land on world +x; body -z (the tool axis) on world -z.
    assert torch.allclose(
        rot_body @ torch.tensor(frames.JAW_AXIS_BODY), torch.tensor([1.0, 0.0, 0.0]), atol=1e-6
    )
    assert torch.allclose(
        rot_body @ torch.tensor(frames.TOOL_AXIS_BODY), torch.tensor([0.0, 0.0, -1.0]), atol=1e-6
    )
    assert torch.det(rot_body) == pytest.approx(1.0)

"""CPU-only test for maker_arm_lab.tasks...lego_sort.mdp.terminations.

See test_lego_sort_observations.py's module docstring for why this loads
terminations.py directly by file path rather than
`from maker_arm_lab.tasks...mdp.terminations import ...`: importing through the
real package tree executes tasks/manager_based/lego_sort/mdp/__init__.py,
which does `from isaaclab.envs.mdp import *` -- isaaclab is not installed in
this CPU test venv.

terminations.py itself needs only torch, maker_arm_lab.core (also
isaaclab-free), and a `from .observations import brick_color_ids` *relative*
import -- so, unlike observations.py, it cannot simply be exec'd standalone:
a relative import needs a real parent package to resolve against. A
lightweight stand-in package is registered in sys.modules (not the real
mdp/__init__.py) whose __path__ points at the real mdp/ directory, letting
Python's ordinary import machinery find and load the real observations.py
there, without ever executing the isaaclab-dependent mdp/__init__.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

from maker_arm_lab.core import layout

_MDP_DIR = (
    Path(__file__).resolve().parents[1] / "source/maker_arm_lab/tasks/manager_based/lego_sort/mdp"
)

_STUB_PKG_NAME = "_lego_sort_mdp_stub"
_stub_pkg = types.ModuleType(_STUB_PKG_NAME)
_stub_pkg.__path__ = [str(_MDP_DIR)]
sys.modules[_STUB_PKG_NAME] = _stub_pkg

_spec = importlib.util.spec_from_file_location(
    f"{_STUB_PKG_NAME}.terminations", _MDP_DIR / "terminations.py"
)
_terminations = importlib.util.module_from_spec(_spec)
_terminations.__package__ = _STUB_PKG_NAME
sys.modules[_spec.name] = _terminations
_spec.loader.exec_module(_terminations)

per_brick_stowed = _terminations.per_brick_stowed
all_bricks_stowed = _terminations.all_bricks_stowed

# layout.bin_interior_centers()[0] (red bin), duplicated here rather than
# imported so this test does not depend on terminations.py's own use of the
# constant it is trying to verify.
_BIN0_CENTER = torch.tensor([0.44, -0.21, 0.175])


class _StubAsset:
    def __init__(self, root_pos_w: torch.Tensor, root_lin_vel_w: torch.Tensor):
        self.data = types.SimpleNamespace(root_pos_w=root_pos_w, root_lin_vel_w=root_lin_vel_w)


class _StubScene:
    def __init__(self, env_origins: torch.Tensor, bricks: dict[str, _StubAsset]):
        self.env_origins = env_origins
        self._bricks = bricks

    def __getitem__(self, name: str) -> _StubAsset:
        return self._bricks[name]


class _StubEnv:
    """Minimal stand-in for ManagerBasedRLEnv: only the attributes
    terminations.py actually reads (`.cfg.num_bricks`, `.device`,
    `.scene.env_origins`, `.scene[name].data.root_pos_w` / `.root_lin_vel_w`).
    """

    def __init__(self, num_bricks: int, env_origins: torch.Tensor, bricks: dict[str, _StubAsset]):
        self.cfg = types.SimpleNamespace(num_bricks=num_bricks)
        self.device = "cpu"
        self.scene = _StubScene(env_origins, bricks)


def test_single_env_correct_bin_is_stowed():
    env_origins = torch.zeros(1, 3)
    env = _StubEnv(
        num_bricks=1,
        env_origins=env_origins,
        bricks={"brick_0": _StubAsset(_BIN0_CENTER.unsqueeze(0), torch.zeros(1, 3))},
    )
    assert per_brick_stowed(env).tolist() == [[True]]
    assert all_bricks_stowed(env).tolist() == [True]


def test_single_env_wrong_position_is_not_stowed():
    env_origins = torch.zeros(1, 3)
    displaced = (_BIN0_CENTER + torch.tensor([0.5, 0.0, 0.0])).unsqueeze(0)
    env = _StubEnv(
        num_bricks=1,
        env_origins=env_origins,
        bricks={"brick_0": _StubAsset(displaced, torch.zeros(1, 3))},
    )
    assert not all_bricks_stowed(env).item()


def test_multi_env_grades_each_env_against_its_own_origin():
    """Two envs with different, nonzero origins: brick_0 is placed correctly
    relative to env 0's own origin, and displaced relative to env 1's own
    origin. Env 0's origin is deliberately nonzero too (not just env 1's), so
    that if ``_brick_states`` graded raw world position instead of
    subtracting each env's own ``env.scene.env_origins`` -- i.e. graded env 1
    against env 0's bin locations, or every env against the origin -- env 0's
    *correctly placed* brick would also read as far outside its bin, flipping
    the expected [True, False] verdict to [False, False].
    """
    env_origins = torch.tensor([[3.0, -2.0, 0.0], [7.0, 5.0, 0.0]])
    root_pos_w = torch.stack(
        [
            env_origins[0] + _BIN0_CENTER,  # env 0: correct relative to its own origin
            env_origins[1] + _BIN0_CENTER + torch.tensor([0.5, 0.0, 0.0]),  # env 1: displaced
        ]
    )
    env = _StubEnv(
        num_bricks=1,
        env_origins=env_origins,
        bricks={"brick_0": _StubAsset(root_pos_w, torch.zeros(2, 3))},
    )
    assert all_bricks_stowed(env).tolist() == [True, False]


def test_success_follows_the_env_cfg_bin_layout():
    # Red's bin moved to the rightmost slot (y=+0.21): a red brick resting there
    # must count as stowed, and one at the old red slot (y=-0.21) must not.
    bl = layout.BinLayout(bin_x=0.44, bin_ys=(-0.21, -0.07, 0.07, 0.21), slot_colors=(3, 1, 2, 0))
    red_center_new = torch.tensor([0.44, 0.21, 0.175])
    red_center_old = torch.tensor([0.44, -0.21, 0.175])

    def make_env(pos):
        env = _StubEnv(
            num_bricks=1,
            env_origins=torch.zeros(1, 3),
            bricks={"brick_0": _StubAsset(pos.unsqueeze(0), torch.zeros(1, 3))},
        )
        env.cfg.bin_layout = bl
        return env

    assert bool(all_bricks_stowed(make_env(red_center_new))[0])
    assert not bool(all_bricks_stowed(make_env(red_center_old))[0])


def test_success_defaults_to_nominal_layout_without_cfg_field():
    env = _StubEnv(
        num_bricks=1,
        env_origins=torch.zeros(1, 3),
        bricks={"brick_0": _StubAsset(_BIN0_CENTER.unsqueeze(0), torch.zeros(1, 3))},
    )
    assert not hasattr(env.cfg, "bin_layout")
    assert bool(all_bricks_stowed(env)[0])

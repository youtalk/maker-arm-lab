"""CPU-only test for maker_arm_lab.tasks...lego_sort.mdp.observations.brick_color_ids.

brick_color_ids is pure torch with zero isaaclab dependency (its sibling
brick_poses_w's ManagerBasedRLEnv annotation is deferred via
`from __future__ import annotations` + a `TYPE_CHECKING` guard specifically so
this holds). But the module still lives inside the `tasks/` package tree, and
Python always executes a package's __init__.py chain when importing one of its
submodules: `tasks/manager_based/lego_sort/__init__.py` imports gymnasium
(gym.register), and `.../mdp/__init__.py` does `from isaaclab.envs.mdp import
*` -- neither gymnasium nor isaaclab is installed in this CPU test venv, so a
plain `from maker_arm_lab.tasks...mdp.observations import brick_color_ids` would
still fail here even though the function itself needs neither. Load
observations.py directly by file path instead, bypassing the package
hierarchy, so this test does not require gymnasium or isaaclab.
"""

from helpers import ROOT, load_script

_OBSERVATIONS_PATH = ROOT / "source/maker_arm_lab/tasks/manager_based/lego_sort/mdp/observations.py"
_observations = load_script(_OBSERVATIONS_PATH)
brick_color_ids = _observations.brick_color_ids


def test_brick_color_ids_cycles_through_four_colors():
    assert brick_color_ids(8).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_brick_color_ids_four_bricks():
    assert brick_color_ids(4).tolist() == [0, 1, 2, 3]

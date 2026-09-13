"""``core.stowed`` on a stub env: one layout for all envs, or one layout per env."""

import types

import torch

from maker_arm_lab.core import layout, stowed


class _Scene:
    def __init__(self, brick_pos):
        n, b, _ = brick_pos.shape
        self.env_origins = torch.zeros(n, 3)
        self._items = {
            f"brick_{i}": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=brick_pos[:, i], root_lin_vel_w=torch.zeros(n, 3)
                )
            )
            for i in range(b)
        }

    def __getitem__(self, key):
        return self._items[key]


def _env(brick_pos, bin_layouts=None):
    env = types.SimpleNamespace(
        device="cpu",
        cfg=types.SimpleNamespace(
            num_bricks=brick_pos.shape[1], bin_layout=layout.DEFAULT_BIN_LAYOUT
        ),
        scene=_Scene(brick_pos),
    )
    if bin_layouts is not None:
        env.bin_layouts = bin_layouts
    return env


def test_one_layout_gives_shared_centers():
    env = _env(torch.zeros(2, 2, 3))
    assert stowed.env_bin_centers(env, "cpu").shape == (4, 3)


def test_per_env_layouts_grade_each_env_against_its_own_bins():
    far = layout.BinLayout(bin_x=layout.DEFAULT_BIN_LAYOUT.bin_x + 0.10)
    layouts = [layout.DEFAULT_BIN_LAYOUT, far]
    centers = stowed.env_bin_centers(_env(torch.zeros(2, 2, 3), layouts), "cpu")
    assert centers.shape == (2, 4, 3)
    assert torch.allclose(centers[1, :, 0], torch.full((4,), far.bin_x))
    # Brick 0 (red) of each env sits in ITS OWN env's red bin.
    pos = torch.zeros(2, 2, 3)
    pos[0, 0] = layout.bin_interior_centers(layouts[0])[0]
    pos[1, 0] = layout.bin_interior_centers(layouts[1])[0]
    assert stowed.per_brick_stowed(_env(pos, layouts)).tolist() == [[True, False], [True, False]]
    # Env 1's brick 0 at env 0's red bin is NOT stowed.
    pos[1, 0] = layout.bin_interior_centers(layouts[0])[0]
    assert stowed.per_brick_stowed(_env(pos, layouts)).tolist() == [[True, False], [False, False]]

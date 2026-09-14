"""The target-brick rule the privileged channel and the DAgger resume entry share."""

import torch

from maker_arm_lab.core.target import target_brick


def _bricks():
    return torch.tensor([[0.20, 0.05, 0.01], [0.15, -0.02, 0.01], [0.40, 0.00, 0.01]])


def test_a_held_brick_wins():
    stowed = torch.zeros(3, dtype=torch.bool)
    assert target_brick(2, _bricks(), torch.tensor([0.20, 0.05]), stowed, 0.30) == 2


def test_nearest_unstowed_brick_within_the_x_limit():
    stowed = torch.tensor([False, False, False])
    tool = torch.tensor([0.14, -0.02])
    assert target_brick(-1, _bricks(), tool, stowed, 0.30) == 1
    stowed = torch.tensor([False, True, False])
    assert target_brick(-1, _bricks(), tool, stowed, 0.30) == 0
    # Brick 2 sits past the x limit and is never a candidate, however near.
    assert target_brick(-1, _bricks(), torch.tensor([0.40, 0.0]), stowed, 0.30) == 0


def test_no_candidate_is_minus_one():
    stowed = torch.tensor([True, True, False])
    assert target_brick(-1, _bricks(), torch.tensor([0.2, 0.0]), stowed, 0.30) == -1


def test_brick_exactly_at_x_limit_is_included():
    brick_pos = torch.tensor([[0.30, 0.05, 0.01], [0.20, 0.10, 0.01]])
    stowed = torch.tensor([False, False])
    tool = torch.tensor([0.25, 0.0])
    assert target_brick(-1, brick_pos, tool, stowed, 0.30) == 0


def test_desk_plane_distance_not_3d():
    brick_pos = torch.tensor([[0.2, 0.0, 0.0], [0.1, 0.0, 0.3]])
    stowed = torch.tensor([False, False])
    tool = torch.tensor([0.0, 0.0])
    assert target_brick(-1, brick_pos, tool, stowed, 0.5) == 1

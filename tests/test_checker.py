"""Tests for maker_arm_lab.core.checker (CPU-only)."""

import pytest
import torch

from maker_arm_lab.core import checker, layout
from maker_arm_lab.core.checker import (
    all_bricks_stowed,
    bricks_at_rest_in_bins,
    containing_bin,
    held_brick_from_forces,
)

BIN_CENTERS = torch.tensor(
    [[0.44, -0.21, 0.175], [0.44, -0.07, 0.175], [0.44, 0.07, 0.175], [0.44, 0.21, 0.175]]
)
HALF = torch.tensor([0.045, 0.045, 0.02])


def _still(pos):
    return torch.zeros_like(pos)


def test_brick_in_correct_bin():
    pos = BIN_CENTERS[torch.tensor([0, 1])].unsqueeze(0)  # (1,2,3)
    ok = bricks_at_rest_in_bins(pos, _still(pos), BIN_CENTERS, HALF, torch.tensor([0, 1]))
    assert ok.shape == (1, 2)
    assert ok.all()


def test_brick_in_wrong_bin_fails():
    pos = BIN_CENTERS[torch.tensor([1])].unsqueeze(0)  # sitting in bin 1
    ok = bricks_at_rest_in_bins(pos, _still(pos), BIN_CENTERS, HALF, torch.tensor([0]))
    assert not ok.any()


def test_moving_brick_not_stowed():
    pos = BIN_CENTERS[torch.tensor([2])].unsqueeze(0)
    vel = torch.full_like(pos, 0.05)  # |v| ~ 0.087 > default max_speed 0.02
    ok = bricks_at_rest_in_bins(pos, vel, BIN_CENTERS, HALF, torch.tensor([2]))
    assert not ok.any()


def test_brick_above_rim_not_stowed():
    pos = BIN_CENTERS[torch.tensor([3])].unsqueeze(0).clone()
    pos[..., 2] += 0.05  # hovering above the bin interior AABB
    ok = bricks_at_rest_in_bins(pos, _still(pos), BIN_CENTERS, HALF, torch.tensor([3]))
    assert not ok.any()


def test_episode_success_requires_all():
    pos = BIN_CENTERS[torch.tensor([0, 1])].unsqueeze(0).clone()
    pos[0, 1, 0] += 0.2  # second brick far away
    done = all_bricks_stowed(pos, _still(pos), BIN_CENTERS, HALF, torch.tensor([0, 1]))
    assert done.shape == (1,)
    assert not done.item()


def test_per_env_bin_centers_are_rejected():
    """(N,4,3) used to be silently accepted by an unreachable branch.

    Indexing it with ``brick_color_ids`` selects *envs* by colour id, so every
    brick would be graded against the wrong env's bins with nothing raising.
    """
    pos = BIN_CENTERS[torch.tensor([0])].unsqueeze(0).repeat(3, 1, 1)
    per_env_centers = BIN_CENTERS.unsqueeze(0).repeat(3, 1, 1)
    with pytest.raises(ValueError, match=r"bin_centers_w must be \(4,3\)"):
        bricks_at_rest_in_bins(pos, _still(pos), per_env_centers, HALF, torch.tensor([0]))


def test_containing_bin_rejects_per_env_bin_centers():
    """``containing_bin`` shares ``bricks_at_rest_in_bins``'s (4,3)-only invariant.

    Without this guard, a (N,4,3) centers tensor broadcasts against (N,B,3)
    positions into a silently-wrong (1,N,B) result instead of raising -- and
    Task 6 indexes this per env every step, so a rank-3 return would corrupt
    wrong-bin bookkeeping instead of crashing.
    """
    pos = torch.zeros((4, 4, 3))
    vel = torch.zeros((4, 4, 3))
    per_env_centers = BIN_CENTERS.unsqueeze(0).repeat(4, 1, 1)
    with pytest.raises(ValueError, match=r"bin_centers_w must be \(4,3\)"):
        containing_bin(pos, vel, per_env_centers, HALF)


def test_batched_envs():
    pos = BIN_CENTERS[torch.tensor([0])].unsqueeze(0).repeat(3, 1, 1)
    pos[2, 0, 1] += 1.0
    done = all_bricks_stowed(pos, _still(pos), BIN_CENTERS, HALF, torch.tensor([0]))
    assert done.tolist() == [True, True, False]


def test_containing_bin_reports_any_bin_not_just_the_target():
    bins = torch.tensor([[0.5, y, 0.16] for y in (-0.3, -0.1, 0.1, 0.3)])
    half = torch.tensor([0.05, 0.05, 0.04])
    pos = torch.zeros((1, 2, 3))
    vel = torch.zeros((1, 2, 3))
    pos[0, 0] = torch.tensor([0.5, -0.1, 0.16])  # brick 0 resting in bin 1
    pos[0, 1] = torch.tensor([0.0, 0.0, 0.02])  # brick 1 on the desk
    out = containing_bin(pos, vel, bins, half)
    assert out.tolist() == [[1, -1]]
    assert out.dtype == torch.int64


def test_containing_bin_requires_rest():
    bins = torch.tensor([[0.5, y, 0.16] for y in (-0.3, -0.1, 0.1, 0.3)])
    half = torch.tensor([0.05, 0.05, 0.04])
    pos = torch.zeros((1, 1, 3))
    vel = torch.zeros((1, 1, 3))
    pos[0, 0] = torch.tensor([0.5, -0.1, 0.16])
    vel[0, 0] = torch.tensor([0.1, 0.0, 0.0])  # moving -> not "resting in" any bin
    assert containing_bin(pos, vel, bins, half).tolist() == [[-1]]


def test_held_brick_from_forces_picks_pinched_brick_or_none():
    left = torch.tensor([[0.0, 3.0, 0.2]])
    right = torch.tensor([[0.1, 2.5, 4.0]])
    held = held_brick_from_forces(left, right)
    assert held.tolist() == [1]  # min(l,r) max at brick 1
    assert held.dtype == torch.int64
    assert held_brick_from_forces(left * 0, right).tolist() == [-1]


def test_held_brick_from_forces_below_min_force_is_not_held():
    # pinch = min(l,r) = 0.9 for both bricks, below the default min_force=1.0 --
    # unlike the all-zero case above, this pins down the threshold VALUE, not
    # just a >0 vs. ==0 distinction.
    left = torch.tensor([[0.9, 0.0]])
    right = torch.tensor([[0.9, 0.0]])
    assert held_brick_from_forces(left, right).tolist() == [-1]


class TestPinchMinStall:
    """The pinch stall floor is derived from the brick, not fitted to a trial (D3-12)."""

    def test_it_is_half_the_bricks_smallest_dimension(self):
        """gap = 2 * stall must be at least the thinnest cross-section the brick has."""
        assert checker.PINCH_MIN_STALL == pytest.approx(min(layout.BRICK_SIZE) / 2.0)
        assert checker.PINCH_MIN_STALL == pytest.approx(0.0048)

    def test_a_gap_below_the_floor_cannot_span_the_brick_in_any_orientation(self):
        """That is the whole argument: below it the plates are inside the brick's own size."""
        gap_at_floor = 2.0 * checker.PINCH_MIN_STALL
        assert gap_at_floor == pytest.approx(min(layout.BRICK_SIZE))
        assert all(gap_at_floor <= dimension + 1e-12 for dimension in layout.BRICK_SIZE)

    def test_it_is_stricter_than_the_inherited_openyam_value(self):
        """Re-deriving it tightened the gate; it did not loosen it to admit a near miss."""
        assert checker.PINCH_MIN_STALL > 0.003

    def test_it_admits_an_ideal_grip_on_the_grasped_width(self):
        """The oracle squares the jaws to BRICK_SIZE[1]; that grip must pass comfortably."""
        ideal_stall = layout.BRICK_SIZE[1] / 2.0
        assert ideal_stall > checker.PINCH_MIN_STALL
        # Contact compliance closes the jaws further than geometry alone; the Task D3
        # part-10 carries stalled 5.797-7.487 mm against this 7.9 mm ideal, so the floor
        # needs to sit below the worst of those.
        assert checker.PINCH_MIN_STALL < 0.005797

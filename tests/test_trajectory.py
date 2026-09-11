"""The walk law behind ``OracleRig``'s plan walker (``core/trajectory.py``): a minimum-jerk
time law along a waypoint polyline, and the commanded-motion statistics the smooth-oracle
probe rules read (design ``2026-09-10-ms3-mujoco-design.md`` section 5 item 1)."""

import argparse
from itertools import pairwise

import pytest
import torch

from maker_arm_lab.core import batch_loop
from maker_arm_lab.core import trajectory as tj


def test_min_jerk_is_a_smooth_bell_from_rest_to_rest():
    assert tj.min_jerk(0.0) == 0.0 and tj.min_jerk(1.0) == 1.0
    s = [tj.min_jerk(k / 100) for k in range(101)]
    v = [b - a for a, b in pairwise(s)]
    assert min(v) >= 0.0
    assert v[0] < 1e-3 and v[-1] < 1e-3
    assert max(v) == pytest.approx(tj.MIN_JERK_PEAK_VELOCITY / 100, rel=1e-2)
    a = [b - c for c, b in pairwise(v)]
    jerk = [abs(b - c) for c, b in pairwise(a)]
    # The law's jerk peaks at 60 / T^3, i.e. 6e-5 per step^3 at 100 steps; a staircase-like
    # jump in acceleration would show as a third difference of order the acceleration
    # itself, 5.8e-4. The bound sits between the two.
    assert max(jerk) < 1e-4


def test_walk_seconds_is_speed_bound_on_long_legs_and_acceleration_bound_on_short_ones():
    assert tj.walk_seconds(0.12, mean_speed=0.045, accel_per_peak_speed=4.0) == pytest.approx(
        0.12 / 0.045
    )
    short = tj.walk_seconds(0.02, mean_speed=0.045, accel_per_peak_speed=4.0)
    assert short > 0.02 / 0.045
    peak_speed = tj.MIN_JERK_PEAK_VELOCITY * 0.045
    assert tj.MIN_JERK_PEAK_ACCELERATION * 0.02 / short**2 == pytest.approx(4.0 * peak_speed)
    assert tj.walk_seconds(0.0, mean_speed=0.045, accel_per_peak_speed=4.0) == 0.0


def test_polyline_point_passes_through_every_waypoint_in_order():
    w = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 2.0])]
    assert torch.equal(tj.polyline_point(w, 0.0), w[0])
    assert torch.equal(tj.polyline_point(w, 0.5), w[1])
    assert torch.equal(tj.polyline_point(w, 1.0), w[2])
    assert torch.allclose(tj.polyline_point(w, 0.75), torch.tensor([1.0, 1.0]))
    assert torch.equal(tj.polyline_point(w[:1], 0.3), w[0])


def test_motion_stats_count_static_frames_and_accelerations():
    m = tj.MotionStats(num_envs=1, control_hz=30.0)
    for v in (
        torch.zeros(6),
        torch.zeros(6),
        torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        m.observe(0, v)
    s = m.summary()
    assert s["frames"] == 3 and s["near_static_fraction"] == pytest.approx(1 / 3)
    # second differences of the commanded target, x 30^2: 0.01 and 0.01 rad per step^2
    assert s["accel_max_rad_s2"] == pytest.approx(9.0)
    assert s["accel_p50_rad_s2"] == pytest.approx(9.0)
    m.reset(0)
    m.observe(0, torch.ones(6))  # an episode boundary: no frame is counted across it
    assert m.summary()["frames"] == 3


def test_transport_speed_is_a_batch_argument_recorded_in_run_meta():
    parser = argparse.ArgumentParser()
    batch_loop.add_batch_arguments(parser)
    assert parser.parse_args(["--out_dir", "x"]).transport_speed == tj.TRANSPORT_SPEED == 4.0
    args = parser.parse_args(["--out_dir", "x", "--transport_speed", "3"])
    assert args.transport_speed == 3.0
    draw, _, noise = batch_loop.build_run(args)
    meta = batch_loop.run_meta(
        args, draw, noise, None, num_bricks=6, export_mode="x", backend="t", task="t"
    )
    assert meta["transport_speed"] == 3.0

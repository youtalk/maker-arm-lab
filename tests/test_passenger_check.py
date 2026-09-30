"""MS7e teacher lever T2 (ledger R12), ``passenger_check``: the rig's LIFT gate calls a lift a
miss when a part other than the carried one rose with the jaws, so REOPEN drops both back and
the retry picks the part again. Off, the gate is MS7d's."""

import random

import pytest
import torch

from maker_arm_lab.core import layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_heap_oracle import _place
from test_oracle_runtime_frames import _heap_oracle, _stub_rig

BRICK = parts.make_part("brick", 2, 4)
RIDER = parts.make_part("brick", 1, 2)
BOXES = [(0.12, 0.19, 0.0), (0.12, -0.19, 0.0), (-0.6, 0.3, 0.0), (-0.6, 0.6, 0.0)]


def _staged_lift(rig, rider_rise: float, check: bool = True):
    """Env 0 at LIFT's end: part 0, the target, 0.10 m up in the jaws; part 1 lay on it and
    rose ``rider_rise``; part 2 lies 0.2 m away, where it lay."""
    rig.passenger_check = check
    orc = _heap_oracle(rig)
    rig.oracles[0] = orc
    orc.target_brick = 0
    orc.phase = oracle.Phase.LIFT
    x, y = (float(v) for v in rig.pick_point(0)[:2])
    start = torch.tensor([[x, y, 0.0048], [x + 0.01, y, 0.0144], [x + 0.2, y, 0.0048]])
    rig.parts_start[0] = start.clone() if check else None
    rig.brick_start[0] = start[0].clone()
    now = start + torch.tensor([[0.0, 0.0, 0.10], [0.0, 0.0, rider_rise], [0.0, 0.0, 0.0]])
    for b in range(3):
        rig.env.scene[f"brick_{b}"].data.root_pos_w[0] = now[b]
    return orc


def test_a_part_that_rode_up_with_the_target_makes_the_lift_a_miss():
    _env, rig = _stub_rig(num_envs=1, num_bricks=3)
    _staged_lift(rig, rider_rise=0.10)
    assert rig._lift_result(0, "converged").lifted is False
    assert rig.passenger_log == [[[1]]]


@pytest.mark.parametrize("rise", [0.0, 0.005])
def test_a_neighbour_left_lying_or_nudged_is_no_passenger(rise):
    _env, rig = _stub_rig(num_envs=1, num_bricks=3)
    _staged_lift(rig, rider_rise=rise)
    assert rig._lift_result(0, "converged").lifted is True
    assert rig.passenger_log == [[]]


def test_with_the_switch_off_the_lift_gate_is_ms7ds():
    _env, rig = _stub_rig(num_envs=1, num_bricks=3)
    assert rig.passenger_check is False
    _staged_lift(rig, rider_rise=0.10, check=False)
    assert rig._lift_result(0, "converged").lifted is True
    assert rig.passenger_log == [[]]


def _lift(check: bool, rider: bool) -> dict:
    """A real two-part lift: a 1x2 lies along a 2x4's top, and the rig takes the 2x4 over as
    the heap loop's takeover does, until LIFT reports. Records every commanded joint target."""
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1, draw, parts=[BRICK, RIDER], cameras=False, desk_boxes=4, gripper_mode="continuous"
    )
    env.reset(seed=11)
    env.place_desk_boxes(0, BOXES)
    env.destination_bins[0] = torch.tensor([0, 1])
    x, y = layout.TRAY_CENTER
    rider_at = (
        (x, y, BRICK.size[2] + RIDER.size[2] / 2 + 0.001, 0.0)
        if rider
        else (x, y + 0.2, RIDER.size[2] / 2 + 0.001, 0.0)
    )
    _place(env, [(x, y, BRICK.size[2] / 2 + 0.001, 0.0), rider_at])
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            boxes=env.desk_box_poses[i],
        )

    rig = OracleRig(
        env,
        env.cfg.bin_layout,
        oracle.OracleNoise(),
        rng,
        oracle_factory=factory,
        passenger_check=check,
    )
    rig.rearm([0], idle=True)
    assert rig.resume(0, target=0)
    commands, after = [], None
    for _ in range(3000):
        phase = rig.goals[0].phase
        rig.step()
        commands.append(rig.q_cmd[0].tolist())
        if phase == oracle.Phase.LIFT and rig.goals[0].phase != phase:
            after = rig.goals[0].phase
            break
    rider_z = float(rig._brick_pos(0, 1)[2])
    env.close()
    return {"after": after, "log": rig.passenger_log[0], "commands": commands, "rider_z": rider_z}


def test_a_real_rider_on_the_target_is_refused_and_a_lone_lift_is_untouched():
    """Under the switch the rider trips the gate and REOPEN follows; off, EXTEND carries both.
    A lift with the rider lying clear plans and commands the same with the switch on or off."""
    on, off = _lift(True, True), _lift(False, True)
    assert on["after"] == oracle.Phase.REOPEN and on["log"] == [[1]], on["log"]
    assert off["after"] == oracle.Phase.EXTEND and off["log"] == []
    assert off["rider_z"] > 0.05  # off, the rider went up with the jaws
    lone_on, lone_off = _lift(True, False), _lift(False, False)
    assert lone_on["after"] == lone_off["after"] == oracle.Phase.EXTEND
    assert lone_on["log"] == [] and lone_on["commands"] == lone_off["commands"]

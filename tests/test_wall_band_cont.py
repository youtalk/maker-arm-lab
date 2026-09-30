"""MS7e ledger R13 serving lever S3, ``--wall_band_cont``: under the continuous gripper the grasp
model's wall test puts the fingers where the continuous jaws land, ``grasp.CONT_HOLD_BAND``
past the set point, not the binary gripper's JAW_HOLD_BAND. The teacher and the heap loop's
selectors rank through ``HeapOracle.hold_band``; with the switch off every plan is MS7d's."""

import argparse
import math
import pathlib
import random

import pytest
import torch

from maker_arm_lab.core import batch_loop, grasp, layout, oracle, parts
from maker_arm_lab.core.heap_oracle import HeapOracle
from maker_arm_lab.mj.env import MjLegoSortEnv
from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig
from test_mj_heap_oracle import _place

PLATE = parts.by_name(["plate_1x2"])[0]
OFF_WALL = 0.013
"""m: the plate's centre off the left wall of the 30 degree tray, at 30 degrees of yaw."""


def _checked(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper", default="binary")
    batch_loop.add_ms7e_arguments(parser)
    args = parser.parse_args(argv)
    batch_loop.check_ms7e(parser, args)
    return args


def test_the_switch_needs_the_continuous_gripper_and_every_client_passes_it():
    with pytest.raises(SystemExit):
        _checked(["--wall_band_cont"])
    args = _checked(["--gripper", "continuous", "--wall_band_cont"])
    assert batch_loop.ms7e_switches(args)["wall_band_cont"] is True
    assert batch_loop.ms7e_switches(argparse.Namespace())["wall_band_cont"] is False
    root = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    for name in ("mj_eval_heap.py", "mj_generate_batch.py"):
        assert (root / name).read_text().count("wall_band_cont=args.wall_band_cont") == 1


def test_the_selector_ranks_at_the_binary_band_unless_the_switch_is_on():
    def selector(**kw):
        return HeapOracle(layout.DEFAULT_BIN_LAYOUT, [PLATE], 0, oracle.nominal_draw(), **kw)

    assert selector().hold_band is grasp.JAW_HOLD_BAND
    assert selector(wall_band_cont=False).hold_band is grasp.JAW_HOLD_BAND
    assert selector(wall_band_cont=True).hold_band is grasp.CONT_HOLD_BAND


def _drive(**kw) -> dict:
    """The continuous teacher on the lone plate by the wall, to its delivery or 3000 steps:
    every commanded action, the rakes, the delivery, and the steps a finger touched a wall."""
    draw = layout.sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    env = MjLegoSortEnv(
        1,
        draw,
        parts=[PLATE],
        cameras=False,
        gripper_mode="continuous",
        tray=True,
        tray_lean_deg=30.0,
    )
    env.reset(seed=11)
    (_, _), (y0, _) = layout.tray_bounds()
    _place(env, [(layout.TRAY_CENTER[0], y0 + OFF_WALL, 0.0016, math.radians(30.0))])
    rng = random.Random(1)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            int(env.destination_bins[i]),
            episode_draw,
            rng=rng,
            tray=True,
            tray_lean_deg=30.0,
            **kw,
        )

    rig = OracleRig(env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory)
    m, d = env.models[0], env.datas[0]
    walls = {m.geom(f"tray_wall_{w}").id for w in ("near", "far", "left", "right")}
    fingers = set(env.handles.finger_geoms[0]) | set(env.handles.finger_geoms[1])
    actions, touched = [], 0
    for _ in range(3000):
        action = rig.actions()
        _, _, terminated, truncated, _ = env.step(action)
        out = rig.observe((terminated | truncated).cpu())
        actions.append(action.clone())
        pairs = ({int(c.geom1), int(c.geom2)} for c in d.contact[: d.ncon])
        touched += any(p & walls and p & fingers for p in pairs)
        if bool(out["dones"][0]) or bool(out["abandoned"][0]):
            break
    stats = rig.episode_stats(0)
    env.close()
    return {"actions": actions, "stats": stats, "touched": touched}


@pytest.fixture(scope="module")
def drives():
    return {name: _drive(**kw) for name, kw in (("main", {}), ("off", {"wall_band_cont": False}))}


def test_with_the_switch_off_every_plan_is_ms7ds(drives):
    main, off = drives["main"]["actions"], drives["off"]["actions"]
    assert len(main) == len(off) and all(torch.equal(a, b) for a, b in zip(main, off))
    assert drives["off"]["stats"]["rakes"] >= 1  # no grasp ranks by the wall: the teacher rakes


def test_under_the_switch_the_teacher_picks_the_plate_off_the_wall_without_a_rake(drives):
    on = _drive(wall_band_cont=True)
    assert on["stats"]["rakes"] == 0 and on["stats"]["bricks_delivered"] == 1, on["stats"]
    assert on["stats"]["cycles"] == 1 and on["touched"] == 0
    assert len(on["actions"]) < len(drives["off"]["actions"])

"""The tracking bench's plant and rungs (tracking design section 8)."""

import maker_arm_rs
import numpy as np

from maker_arm_lab.core import dynamics_model as dm
from maker_arm_lab.core import profile as prof
from maker_arm_lab.mj import bench

DESK_POSE = [0.0, -2.03, 1.71, -1.17, 0.0, 0.0]


def _hold(rung, duration, seed=0):
    plant = bench.build_plant()
    exact = dm.dynamics(dm.urdf_bodies())
    tracking = None
    if rung in ("F", "A"):
        params = {"contact": rung == "A"}
        tracking = maker_arm_rs.Tracking(exact, params=params, profile=prof.sim_profile())
    row = [*DESK_POSE, bench.grip_value()]
    trace = bench.run(
        plant,
        rung,
        DESK_POSE,
        [(0.0, 0.0, 1 / 30, [row])],
        duration,
        tracking=tracking,
        exact=exact,
        seed=seed,
    )
    return plant, trace


def test_the_plant_is_the_exact_model_with_the_jaws_held():
    plant = bench.build_plant()
    assert plant.model.nq == 6
    total = sum(b[0] for b in dm.urdf_bodies())
    assert abs(plant.model.body_subtreemass[plant.model.body("link_002").id] - total) < 1e-9


def test_a_short_hold_is_deterministic():
    _, a = _hold("A", 0.5)
    _, b = _hold("A", 0.5)
    for field in ("t", "q", "dq", "tip", "force", "r", "offset", "q_r", "gate"):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field
    assert len(a.t) == 100 and abs(a.t[-1] - 0.5) < 1e-12


def test_b0_sags_under_gravity_and_b1_does_not():
    plant, b0 = _hold("B0", 2.5)
    _, b1 = _hold("B1", 2.5)
    k = round(2.0 / bench.TICK_S) - 1
    cmd = bench.tip_of(plant, DESK_POSE)
    assert np.linalg.norm(b0.tip[k] - cmd) > 0.005
    assert np.linalg.norm(b1.tip[k] - cmd) < 0.002


def test_a_wall_across_the_sag_reports_contact_force():
    # B0 sags the desk-pose tip about 10 mm toward -x; the wall stands 3 mm behind it.
    tip = bench.tip_of(bench.build_plant(), DESK_POSE)
    center = tip - [bench.TIP_RADIUS + 0.003 + bench.WALL_HALF[0], 0.0, 0.0]
    plant = bench.build_plant(wall=(center, [1.0, 0.0, 0.0]))
    trace = bench.run(plant, "B0", DESK_POSE, [], 0.5)
    assert trace.force[0] == 0.0 and trace.force.max() > 0.0

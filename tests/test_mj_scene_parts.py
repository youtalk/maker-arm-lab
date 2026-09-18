"""Parts in the MuJoCo scene (design sections 5 and 6). There is no dish since 2026-09-18."""

import random

import mujoco
import numpy as np
import pytest

from maker_arm_lab.core import parts
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import scene


def _draw():
    draw = sample_run_draw(random.Random(5))
    draw["slot_colors"] = (0, 1, 2, 3)
    return draw


PARTS = [
    parts.make_part("brick", 2, 4),
    parts.make_part("slope", 1, 2),
    parts.make_part("round", 2, 2),
]


@pytest.fixture(scope="module")
def built():
    return scene.build_model(_draw(), parts=PARTS, cameras_on=False)


def test_one_collider_per_part_and_visual_studs(built):
    model, h = built
    assert len(h.brick_bodies) == 3 and len(h.brick_geoms) == 3
    assert model.geom_type[h.brick_geoms[0]] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_type[h.brick_geoms[1]] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_type[h.brick_geoms[2]] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom_size[h.brick_geoms[2]][:2] == pytest.approx([0.0079, 0.0048])
    studs = [g for g in range(model.ngeom) if model.geom(g).name.startswith("brick_0_stud_")]
    assert len(studs) == 8
    assert all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in studs)
    assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER for g in studs)
    assert not [g for g in range(model.ngeom) if model.geom(g).name.startswith("brick_1_stud_")]
    assert h.body_geoms[0][0] == h.brick_geoms[0] and len(h.body_geoms[0]) == 9
    assert len(h.body_geoms[1]) == 1 and len(h.body_geoms[2]) == 5


def test_part_masses_and_sizes_reach_the_model(built):
    model, h = built
    for b, part in enumerate(PARTS):
        assert model.body_mass[h.brick_bodies[b]] == pytest.approx(part.mass, rel=1e-6)
    assert model.geom_size[h.brick_geoms[0]] == pytest.approx([s / 2 for s in PARTS[0].size])
    assert model.geom_contype[h.brick_geoms[1]] == 3 and model.geom_condim[h.brick_geoms[1]] == 4


def test_parts_settle_flat_on_the_desk(built):
    model, h = built
    data = mujoco.MjData(model)
    for b, adr in enumerate(h.brick_qpos):
        data.qpos[adr : adr + 3] = (0.16 + 0.04 * b, 0.0, 0.03)
        data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=480)
    for b, part in enumerate(PARTS):
        z = data.qpos[h.brick_qpos[b] + 2]
        assert z == pytest.approx(part.size[2] / 2, abs=0.002), (part.name, z)
    assert np.linalg.norm(data.qvel[h.brick_dof[0] : h.brick_dof[0] + 3]) < 0.02


def test_the_legacy_scene_is_unchanged_without_parts():
    _model, h = scene.build_model(_draw(), num_bricks=2, cameras_on=False)
    assert len(h.brick_bodies) == 2 and all(len(g) == 1 for g in h.body_geoms)


def test_the_desk_is_a_hard_surface_under_a_pressed_part():
    """A 3 g brick pressed onto the desk with 20 N sinks under a millimetre.

    Before the desk carried its own contact parameters the same press sank the brick 13 mm,
    more than its own height, because the time-constant form scales the stiffness with the
    lighter body (measured 2026-09-18). The desk's priority keeps the part-part and jaw-part
    pairs on their own tuning.
    """
    model, h = scene.build_model(_draw(), parts=[PARTS[0]], cameras_on=False)
    desk = h.desk_geom
    assert model.geom_priority[desk] == 1
    assert model.geom_solimp[desk][:2] == pytest.approx(scene.DESK_SOLIMP[:2])
    assert model.geom_priority[h.brick_geoms[0]] == 0
    data = mujoco.MjData(model)
    adr = h.brick_qpos[0]
    data.qpos[adr : adr + 3] = (0.19, 0.0, PARTS[0].size[2] / 2 + 0.001)
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_step(model, data, nstep=480)
    data.xfrc_applied[h.brick_bodies[0]] = [0.0, 0.0, -20.0, 0.0, 0.0, 0.0]
    mujoco.mj_step(model, data, nstep=480)
    sunk = PARTS[0].size[2] / 2 - data.qpos[adr + 2]
    assert 0.0 <= sunk < 0.001, sunk

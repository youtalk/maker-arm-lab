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
    lighter body (measured 2026-09-18). The desk's solmix weight keeps the part-part and
    jaw-part pairs on their own tuning, and the part-desk pair keeps the part's condim and
    friction: a geom priority handed the pair condim 3 and friction 1.0 instead, in the
    six-brick task too.
    """
    model, h = scene.build_model(_draw(), parts=[PARTS[0]], cameras_on=False)
    desk = h.desk_geom
    assert model.geom_priority[desk] == 0 and model.geom_solmix[desk] == scene.DESK_SOLMIX
    assert model.geom_solimp[desk][:2] == pytest.approx(scene.DESK_SOLIMP[:2])
    model.geom_friction[h.brick_geoms[0], 0] = 1.2  # the top of mj/env.py's friction draw
    data = mujoco.MjData(model)
    adr = h.brick_qpos[0]
    data.qpos[adr : adr + 3] = (0.19, 0.0, PARTS[0].size[2] / 2 + 0.001)
    data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_step(model, data, nstep=480)
    pair = next(c for c in data.contact[: data.ncon] if desk in (c.geom1, c.geom2))
    assert pair.dim == scene.BRICK_CONDIM and pair.friction[0] == pytest.approx(1.2)
    assert pair.solimp[:2] == pytest.approx(scene.DESK_SOLIMP[:2], abs=1e-4)
    data.xfrc_applied[h.brick_bodies[0]] = [0.0, 0.0, -20.0, 0.0, 0.0, 0.0]
    mujoco.mj_step(model, data, nstep=480)
    sunk = PARTS[0].size[2] / 2 - data.qpos[adr + 2]
    assert 0.0 <= sunk < 0.001, sunk


def test_round_physics_changes_the_round_parts_only():
    """MS7e design section 6.4: collision studs and condim 6 on round parts, under the switch
    only. Every other part's geoms are the same bytes either way."""
    mixed = parts._replayable()
    off, h = scene.build_model(_draw(), parts=mixed, cameras_on=False)
    on, _ = scene.build_model(_draw(), parts=mixed, cameras_on=False, round_physics=True)
    assert on.ngeom == off.ngeom
    rolls = {g for b, p in enumerate(mixed) if p.family == "round" for g in h.body_geoms[b]}
    rest = [g for g in range(off.ngeom) if g not in rolls]
    for name in [n for n in dir(off) if n.startswith("geom_")]:
        a, b = getattr(off, name), getattr(on, name)
        if isinstance(a, np.ndarray) and a.shape and a.shape[0] == off.ngeom:
            assert np.array_equal(a[rest], b[rest]), name
    for g in rolls:
        assert on.geom_contype[g] == 3 and on.geom_condim[g] == scene.ROUND_CONDIM
        assert on.geom_friction[g] == pytest.approx(scene.ROUND_FRICTION)
    for b, p in enumerate(mixed):
        if p.family == "round":
            collider, *studs = h.body_geoms[b]
            assert off.geom_condim[collider] == scene.BRICK_CONDIM
            assert off.geom_friction[collider] == pytest.approx(scene.BRICK_FRICTION)
            assert studs and all(off.geom_contype[g] == 0 for g in studs)


@pytest.mark.parametrize("round_physics", [False, True])
def test_a_round_part_on_its_side_stops_rolling_under_the_switch(round_physics):
    """A jaw closes at 0.1 m/s. A round 2x2 on its side kicked that fast across its axis
    rolled 46 mm at the median without the switch and 2.7 mm with it (rolling probe,
    2026-09-29); the studs alone did not stop it."""
    (part,) = parts.by_name(["round_2x2"])
    model, h = scene.build_model(
        _draw(), parts=[part], cameras_on=False, round_physics=round_physics
    )
    data = mujoco.MjData(model)
    adr, dof = h.brick_qpos[0], h.brick_dof[0]
    data.qpos[adr : adr + 3] = (0.19, 0.0, part.size[0] / 2 + 0.001)
    data.qpos[adr + 3 : adr + 7] = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)  # axis along y
    mujoco.mj_step(model, data, nstep=240)
    start = data.qpos[adr].copy()
    data.qvel[dof] = 0.1
    mujoco.mj_step(model, data, nstep=720)
    travel = abs(data.qpos[adr] - start)
    assert (travel < 0.008) if round_physics else (travel > 0.03), travel


LDRAW = [parts.ldraw_part(i, f) for f, ids in parts.LDRAW_FAMILY_IDS.items() for i in ids]
LDRAW += [p for p in parts.by_name(parts.CLASSIFIER_EXTRA) if p.hulls]


@pytest.fixture(scope="module")
def ldraw_built():
    return scene.build_model(_draw(), parts=LDRAW, cameras_on=False)


def test_each_ldraw_hull_is_a_collider_with_the_brick_contacts(ldraw_built):
    """MS7e design section 6.1: BRICK_FRICTION and condim 4 on every hull; the hulls share the
    H2 mass and the studs are visual."""
    model, h = ldraw_built
    for b, part in enumerate(LDRAW):
        colliders = [g for g in h.body_geoms[b] if model.geom_contype[g]]
        assert colliders[0] == h.brick_geoms[b] and len(colliders) == len(part.hulls)
        for g in colliders:
            assert model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            assert model.geom_contype[g] == 3 and model.geom_condim[g] == scene.BRICK_CONDIM
            assert model.geom_friction[g] == pytest.approx(scene.BRICK_FRICTION)
        studs = [g for g in h.body_geoms[b] if not model.geom_contype[g]]
        assert len(studs) == len(part.stud_frames)
        assert model.body_mass[h.brick_bodies[b]] == pytest.approx(part.mass, rel=1e-6)


def test_every_ldraw_part_rests_on_the_desk(ldraw_built):
    """Dropped upright, as LDraw builds it, each part comes to rest with its lowest hull vertex
    on the desk top. The MS7e parts rest level; 3901 (hair) settles back onto its long back
    edge, and a pin may roll a few degrees on its collar. At rest is a pose that holds for
    another second."""
    model, h = ldraw_built
    data = mujoco.MjData(model)
    for b, adr in enumerate(h.brick_qpos):
        data.qpos[adr : adr + 3] = (0.10 + 0.07 * (b % 4), -0.42 + 0.05 * (b // 4), 0.03)
        data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=720)
    settled = data.qpos.copy()
    mujoco.mj_step(model, data, nstep=240)
    for b, part in enumerate(LDRAW):
        adr = h.brick_qpos[b]
        z = data.qpos[adr + 2]
        if part.family == "ldraw":
            assert z == pytest.approx(part.size[2] / 2, abs=0.001), (part.name, z)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, data.qpos[adr + 3 : adr + 7])
        verts = np.concatenate([np.array(v) for _, v in part.hulls])
        lowest = (verts @ rot.reshape(3, 3).T)[:, 2].min() + z
        assert lowest == pytest.approx(0.0, abs=0.001), (part.name, lowest)
        moved = np.abs(data.qpos[adr : adr + 7] - settled[adr : adr + 7]).max()
        assert moved < 1e-4, (part.name, moved)

"""Applying a run draw to a compiled model matches building the model from that draw."""

import random

import numpy as np
import pytest

from maker_arm_lab.core import layout
from maker_arm_lab.core.layout import sample_run_draw
from maker_arm_lab.mj import scene


def _draw(seed, sun=True):
    d = sample_run_draw(random.Random(seed))
    d["slot_colors"] = (0, 1, 2, 3)
    if not sun:
        d["sun_intensity"] = 0.0
    return d


@pytest.mark.parametrize("sun_b", [True, False])
def test_apply_draw_matches_a_model_built_from_it(sun_b):
    a, b = _draw(1), _draw(2, sun=sun_b)
    model, _ = scene.build_model(a, num_bricks=2, cameras_on=True)
    target, _ = scene.build_model(b, num_bricks=2, cameras_on=True)
    bl = layout.BinLayout(bin_x=b["bin_x"], bin_ys=tuple(b["bin_ys"]), slot_colors=b["slot_colors"])
    scene.apply_bins(model, bl)
    scene.apply_lighting(model, b)
    scene.apply_cameras(model, b)
    for name, _center, _size in layout.bin_static_boxes(bl):
        g = model.geom(name).id
        assert np.allclose(model.geom_pos[g], target.geom_pos[g])
    for light in ("dome", "sun"):
        i = model.light(light).id
        assert np.allclose(model.light_diffuse[i], target.light_diffuse[i])
        assert np.allclose(model.light_ambient[i], target.light_ambient[i])
        assert np.allclose(model.light_specular[i], target.light_specular[i])
        assert np.allclose(model.light_dir[i], target.light_dir[i])
    f = model.geom("floor").id
    assert np.allclose(model.geom_rgba[f], target.geom_rgba[f])
    for cam in ("overhead", "wrist"):
        c = model.camera(cam).id
        assert np.allclose(model.cam_pos[c], target.cam_pos[c])
        assert np.allclose(model.cam_quat[c], target.cam_quat[c])


def test_the_sun_exists_at_zero_strength_when_the_draw_has_none():
    model, _ = scene.build_model(_draw(4, sun=False), num_bricks=1, cameras_on=False)
    s = model.light("sun").id
    assert np.allclose(model.light_diffuse[s], 0.0)
    assert model.ncam == 0
    scene.apply_cameras(model, _draw(5))  # a no-op without cameras, not an error

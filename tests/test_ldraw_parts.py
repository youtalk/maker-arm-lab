"""The LDraw family's parts (MS7e design sections 6.5 and 10.2): every JSON part loads with the
size and mass rules, every hull is convex under the cap, and the generator keeps its token
contract. ``test_mj_scene_parts.py`` builds them in the scene."""

import importlib.util
import json
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from maker_arm_lab.core import parts

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "ldraw_parts.py"
LDRAW = [parts.ldraw_part(i) for i in parts.LDRAW_IDS]


def _generator():
    spec = importlib.util.spec_from_file_location("ldraw_parts", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib at import; coacd only inside a run
    return module


def test_every_json_part_loads_with_its_rules():
    gen = _generator()
    assert tuple(gen.PARTS) == parts.LDRAW_IDS
    assert {p.name for p in parts.catalog(("ldraw",))[29:]} == {
        f"ldraw_{i}" for i in parts.LDRAW_IDS
    }
    for part in LDRAW:
        rec = json.loads((parts.LDRAW_DIR / f"{part.name[6:]}.json").read_text())
        assert part.family == "ldraw" and parts.part_class(part) == "special"
        assert part.size == tuple(rec["size"]) and part.size[0] >= part.size[1]
        assert min(part.size[:2]) <= parts.MAX_GRASP_WIDTH  # at most 45 mm wide
        assert part.size[2] >= parts.BRICK_HEIGHT - 1e-9  # at least 9.6 mm tall
        assert part.mass == parts.part_mass(part.size)  # H2, floored
        assert part.closing_axes == tuple(gen.PARTS[part.name[6:]][0]) and part.closing_axes
        assert 1 <= len(part.hulls) <= gen.MAX_HULLS
        assert sum(share for share, _ in part.hulls) == pytest.approx(1.0)
        half = np.array(part.size) / 2 + 1e-6
        for _, verts in part.hulls:
            assert (np.abs(np.array(verts)) <= half).all(), part.name
        for base, axis in part.stud_frames:
            k = int(np.argmax(np.abs(axis)))
            assert abs(axis[k]) == 1.0 and base[k] == pytest.approx(axis[k] * part.size[k] / 2)
    assert parts.by_name([p.name for p in LDRAW]) == LDRAW
    assert (parts.LDRAW_DIR / "NOTICE").read_text().startswith("The part geometry")


def test_every_hull_is_convex_and_keeps_its_volume():
    """MuJoCo's own hull of each vertex list keeps every vertex, and its volume is the one the
    generator measured, which splits the part's mass."""
    for part in LDRAW:
        rec = json.loads((parts.LDRAW_DIR / f"{part.name[6:]}.json").read_text())
        spec = mujoco.MjSpec()
        for k, (_, verts) in enumerate(part.hulls):
            spec.add_mesh(name=f"h{k}", uservert=[c for v in verts for c in v])
            body = spec.worldbody.add_body(name=f"b{k}")
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"h{k}").density = 1.0
        model = spec.compile()
        for k, (_, verts) in enumerate(part.hulls):
            assert model.mesh_graph[model.mesh_graphadr[k]] == len(verts), (part.name, k)
            volume = model.body_mass[k + 1]  # density 1
            assert volume == pytest.approx(rec["hull_volumes"][k], rel=1e-3), (part.name, k)


def test_the_generator_prints_its_fail_token_on_a_usage_error():
    text = GENERATOR.read_text()
    assert "LDRAW_OK " in text and "LDRAW_FAIL " in text and "os._exit(main())" in text
    for argv in (["--no_such_flag"], []):
        cmd = [sys.executable, str(GENERATOR), *argv]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        assert out.stdout.strip().splitlines()[-1].startswith("LDRAW_FAIL "), argv

"""scripts/mj_render_set.py (MS7f design section 6.4): the token contract, and on a small heap
driven by the teacher, labels that equal the truth of the scene at each view."""

import importlib.util
import pathlib
import random
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from maker_arm_lab.core import parts

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mj_render_set.py"
FAMILIES = ("round", "ldraw", "flat")


def _load():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _can_render() -> bool:
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8).close()
    except Exception:  # noqa: BLE001 -- any GL failure means "skip the render test"
        return False
    return True


def test_usage_errors_print_the_fail_token():
    for argv in ((), ("--out_dir", "/tmp/x", "--heap_k", "0"), ("--out_dir", "/tmp/x", "--bad")):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
        )
        assert out.returncode == 2, argv
        lines = out.stdout.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("RENDER_FAIL "), argv


def test_the_ms7f_scene_switches_parse_and_the_serving_ones_refuse():
    base = ["--out_dir", "/tmp/x", "--heap_k", "150", "--tray", "--desk_boxes", "8"]
    base += ["--box_draw", "8", "--release_lean"]
    on = ["--categories", "--deep_tray", "--box_empty", "--abandon_cycle"]
    args = _load().parse_args(base + on)
    assert args.categories and args.deep_tray and args.box_empty and args.abandon_cycle
    for extra in (["--classifier", "m.pt"], ["--passenger_drop"], ["--book_miss"]):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), *base, *extra],
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 2, extra
        assert out.stdout.strip().splitlines()[-1].startswith("RENDER_FAIL "), extra


def test_mix_extra_replaces_the_share_it_is_given():
    mix = _load().mix_extra
    parts, extra = list(range(200)), ["x", "y"]
    assert mix(parts, extra, 0.0, random.Random(1)) == parts
    assert set(mix(parts, extra, 1.0, random.Random(1))) == {"x", "y"}
    half = mix(parts, extra, 0.5, random.Random(1))
    assert half == mix(parts, extra, 0.5, random.Random(1))
    assert 60 < sum(p in extra for p in half) < 140
    assert all(p == k for k, p in enumerate(half) if p not in extra)


def test_tilt_reads_the_part_axis_from_world_up():
    tilt = _load()._tilt
    c = np.cos(np.pi / 4)
    assert tilt((1.0, 0.0, 0.0, 0.0)) == 0.0
    assert tilt((0.0, 0.0, 0.0, 1.0)) == 0.0  # a turn about z leaves it upright
    assert tilt((c, c, 0.0, 0.0)) == pytest.approx(90.0)
    assert tilt((0.0, 1.0, 0.0, 0.0)) == pytest.approx(180.0)


@pytest.mark.skipif(not _can_render(), reason="no MuJoCo GL context on this host")
def test_the_render_set_labels_equal_the_truth(tmp_path, monkeypatch):
    """Every row names a part of the heap with its own category and split side, each overhead
    view aims at its part, and each wrist view's part is the one up in the jaws."""
    render = _load()
    args = render.parse_args(
        [
            "--out_dir",
            str(tmp_path),
            "--num_envs",
            "1",
            "--heap_k",
            "4",
            "--cycle_budget",
            "5",
            "--parts_split",
            "all",
            "--families",
            ",".join(FAMILIES),
            "--run_seed",
            "3",
            "--layout_seed",
            "3",
            "--gripper",
            "continuous",
            "--episode_draws",
            "--frame_jitter",
            "--tray",
            "--tray_lean",
            "30",
            "--desk_boxes",
            "4",
            "--passenger_check",
            "--extra_overhead",
            "1",
        ]
    )
    aims, lifted = [], []
    narrow = render.MjLegoSortEnv.narrow_overhead
    held = render._held

    def aim_spy(env, i, point):
        pos = render._positions(env, i)[0]
        aims.append(int(np.linalg.norm(pos.numpy() - np.asarray(point), axis=1).argmin()))
        return narrow(env, i, point)

    def held_spy(env, i):
        b = held(env, i)
        lifted.append((b, float(render._positions(env, i)[0][b, 2]) if b >= 0 else 0.0))
        return b

    monkeypatch.setattr(render.MjLegoSortEnv, "narrow_overhead", aim_spy)
    monkeypatch.setattr(render, "_held", held_spy)
    summary = render.run(args)
    z = np.load(summary["out"])
    _, heldout = parts.split(parts.catalog(FAMILIES), 20260915)
    by_name = {p.name: p for p in parts.catalog(FAMILIES)}
    views = z["view"].tolist()
    assert summary["by_view"]["overhead"] > 0 and summary["by_view"]["lift"] > 0
    assert z["frames"].shape == (len(views), 224, 224, 3) and z["frames"].dtype == np.uint8
    assert z["wrist_crop"].tolist() == [448, 152, 376, 116, 340]
    for n, name in enumerate(z["part"].tolist()):
        if name:
            assert z["category"][n] == parts.part_category(by_name[name])
            assert bool(z["heldout"][n]) == (name in {p.name for p in heldout})
    over = [n for n, v in enumerate(views) if v == "overhead"]
    assert aims == [int(z["held"][n]) for n in over]
    assert all(z["passengers"][n] == z["riders"][n] == -1 for n in over)
    assert all(0.0 <= z["part_seen"][n] <= 1.0 for n in over)
    assert max(z["part_seen"][n] for n in over) == 1.0
    wrist = [n for n, v in enumerate(views) if v != "overhead"]
    assert [b for b, _ in lifted] == [int(z["held"][n]) for n in wrist]
    assert all(0 <= z["riders"][n] <= z["passengers"][n] for n in wrist)
    assert all((z["rider_seen"][n] >= 0.0) == (z["riders"][n] > 0) for n in wrist)
    assert all(z["rider_seen"][n] == -1.0 for n in over)
    has_part = z["held"] >= 0
    assert ((0.0 <= z["tilt"]) & (z["tilt"] <= 180.0) == has_part).all()
    assert (z["tilt"][~has_part] == -1.0).all()
    for n, (b, height) in zip(wrist, lifted):
        if b >= 0 and views[n] == "lift":
            assert height > 0.04, (n, b, height)  # the part rose with the jaws

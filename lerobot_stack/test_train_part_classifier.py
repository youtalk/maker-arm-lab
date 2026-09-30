"""The MS7f part classifier's training script: the token contract, the operating point, and a
tiny train, export and read on synthetic render set files, CPU only and no weight download."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import train_part_classifier as tpc

SCRIPT = Path(__file__).resolve().parent / "train_part_classifier.py"


def _render_file(path: Path, seed: int, n: int = 12) -> Path:
    """A render set file as ``scripts/mj_render_set.py`` writes it: the category shows as the
    frame's colour, and a passenger as a bright band."""
    rng = np.random.default_rng(seed)
    cats = [tpc.CATEGORIES[k % 4] for k in range(n)]
    views = [tpc.VIEWS[k % 3] for k in range(n)]
    riders = [
        1 if (v != "overhead" and k % 4 == 1) else (0 if v != "overhead" else -1)
        for k, v in enumerate(views)
    ]
    frames = rng.integers(0, 30, (n, 224, 224, 3), dtype=np.uint8)
    for k, c in enumerate(cats):
        frames[k, :, :, tpc.CATEGORIES.index(c) % 3] += 200
        if riders[k] > 0:
            frames[k, :40] = 255
    np.savez_compressed(
        path,
        frames=frames,
        view=np.array(views),
        category=np.array(cats),
        part=np.array([f"{c}_kind" for c in cats]),
        heldout=np.array([k % 5 == 0 for k in range(n)]),
        passengers=np.array(riders),
        riders=np.array(riders),
        rider_seen=np.array([0.5 if r > 0 else -1.0 for r in riders]),
        held=np.arange(n),
        part_seen=np.array([1.0 if v == "overhead" else -1.0 for v in views]),
        env=np.zeros(n, dtype=int),
        step=np.arange(n),
        args=np.array("{}"),
    )
    return path


def test_usage_errors_print_the_fail_token(tmp_path):
    for argv in ((), ("--out", str(tmp_path)), ("--out", str(tmp_path), "--model", "m.pt")):
        out = subprocess.run(
            [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, check=False
        )
        assert out.returncode == 2, argv
        lines = out.stdout.strip().splitlines()
        assert len(lines) == 1 and lines[0].startswith("CLS_FAIL "), argv


def test_the_operating_point_keeps_the_false_alarms_under_the_cap():
    rng = np.random.default_rng(0)
    riders = np.concatenate([rng.normal(0, 1, 1000), rng.normal(4, 1, 50)])
    truth = np.arange(1050) >= 1000
    t = tpc.operating_point(riders, truth)
    assert ((riders >= t) & ~truth).sum() <= tpc.PASSENGER_FALSE_ALARMS * 1000
    assert ((riders >= np.nextafter(t, -np.inf)) & ~truth).sum() > 0 or t == riders.min()


def test_a_tiny_run_trains_exports_and_reads_as_the_loop_loads_it(tmp_path):
    train = [_render_file(tmp_path / f"render_{s}.npz", s) for s in (1, 2)]
    read = [_render_file(tmp_path / "render_9.npz", 9)]
    args = tpc.parse_args(
        ["--train", *map(str, train), "--read", *map(str, read), "--out", str(tmp_path / "m")]
        + ["--epochs", "1", "--batch", "8", "--weights", "none", "--device", "cpu"]
    )
    summary = tpc.run(args)
    served = torch.jit.load(summary["model"], map_location="cpu")
    assert list(served.categories) == list(tpc.CATEGORIES)
    assert isinstance(served.passenger_threshold, float)
    for wrist in (False, True):
        logits, rider = served(
            torch.zeros(2, 224, 224, 3, dtype=torch.uint8), wrist, torch.zeros(2, 3)
        )
        assert logits.shape == (2, 6) and rider.shape == (2,)
    assert list(served.wrist_roi) == list(tpc.WRIST_ROI)
    assert list(served.overhead_roi) == list(tpc.OVERHEAD_ROI)
    c1 = json.loads(Path(summary["c1"]).read_text())
    assert c1["frames"] == 12 and c1["lift"]["total"] == 4
    assert set(c1["confusion_lift"]) == set(c1["categories"]) == set(tpc.CATEGORIES[:4])
    assert set(c1["bars"]) == {
        "lift_accuracy",
        "category_accuracy",
        "passenger_recall",
        "passenger_false_alarms",
    }
    again = tpc.run(
        tpc.parse_args(
            ["--model", summary["model"], "--read", str(read[0]), "--out", str(tmp_path / "r")]
        )
    )
    assert json.loads(Path(again["c1"]).read_text())["lift"] == c1["lift"]


def test_a_file_without_the_rider_label_reads_unknown(tmp_path):
    new = _render_file(tmp_path / "render_1.npz", 1)
    old = dict(np.load(new))
    del old["riders"], old["rider_seen"]
    np.savez_compressed(tmp_path / "render_2.npz", **old)
    assert (tpc.load([tmp_path / "render_2.npz"])["riders"] == -1).all()
    data = tpc.load([new, tmp_path / "render_2.npz"])
    assert (data["riders"][12:] == -1).all() and (data["riders"][:12] >= -1).all()
    assert (data["riders"][:12] == data["passengers"][:12]).all()


def test_cycles_pair_each_lift_with_its_selection_and_carry():
    # env 0: select part 3 (and 5) at step 0, lift 3 at 10, carry at 25; a lift of 5 at 40 with
    # no selection of 5 since the one at step 30 (part 4) and no carry. env 1: its own selection.
    view = ["overhead", "overhead", "lift", "carry", "overhead", "lift", "overhead", "lift"]
    data = {
        "view": np.array(view),
        "held": np.array([3, 5, 3, 3, 4, 5, 3, 3]),
        "step": np.array([0, 0, 10, 25, 30, 40, 0, 12]),
        "env": np.array([0, 0, 0, 0, 0, 0, 1, 1]),
        "file": np.zeros(8, dtype=int),
    }
    assert tpc.cycles(data).tolist() == [[2, 0, 3], [5, -1, -1], [7, 6, -1]]
    assert tpc.targets(data).tolist() == [3, 3, 3, 3, 4, 4, 3, 3]


def test_the_loop_vote_takes_the_third_view_only_on_a_split():
    over, lift, carry = (
        np.array(v) for v in (["a", "a", "a", "a"], ["a", "b", "b", "b"], ["c", "a", "b", "c"])
    )
    assert tpc.loop_vote(over, lift, carry).tolist() == ["a", "a", "b", "c"]


def test_load_reads_the_size_beside_each_file(tmp_path):
    a = _render_file(tmp_path / "render_1.npz", 1, n=4)
    b = _render_file(tmp_path / "render_2.npz", 2, n=3)
    np.save(tmp_path / "render_1.size.npy", np.full((4, 3), 0.01, np.float32))
    size = tpc.load([a, b])["size"]
    assert size.shape == (7, 3) and (size[:4] == 0.01).all() and (size[4:] == -1.0).all()

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
        logits, rider = served(torch.zeros(2, 224, 224, 3, dtype=torch.uint8), wrist)
        assert logits.shape == (2, 6) and rider.shape == (2,)
    assert list(served.wrist_roi) == list(tpc.WRIST_ROI)
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

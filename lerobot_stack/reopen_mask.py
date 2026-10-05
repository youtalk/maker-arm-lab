"""The MS7g corpus mask: the REOPEN frames that follow a risen lift (design 3.6 and 8, ledger G10).

When the teacher's LIFT gate gives up, it opens its jaws where it stands (REOPEN: 20 frames,
gripper command to full open, the arm still) and retries the part from HOVER. After a lift in
which the pick point rose at least ``RISE_MIN_M``, those frames teach the policy to open over the
tray with a part in its jaws. They leave training. A REOPEN at grasp height (a regrasp) stays.

A mask is an ``.npz`` with ``lengths`` (the episode lengths of the dataset it belongs to, in
episode order, so it can never be applied to another dataset) and ``drop`` (sorted frame rows of
that dataset). ``train_masked.py --frame_mask`` reads it. Subcommands, each ending in one
``MASK_OK {json}`` / ``MASK_FAIL ...`` line:

  truth    raw shards -> the mask of the root convert.sh made from them. The phase arrays name
           the REOPEN after a LIFT, and FK of the LIFT frames says whether the lift rose.
           Segments come from the converter's own ``cycle_segments``, in convert.sh's lane
           order, and must reproduce the root's episode lengths exactly.
  proxy    a converted root whose shards are gone -> its mask from state and action alone: a
           run of full-open gripper commands with the arm still, at least ``RISE_MIN_M`` above
           the lowest pick point of the hold before it. The hold began where the gripper
           command took the value it keeps up to the open; in every LIFT-to-REOPEN of the raw
           shards that is the first CLOSE frame.
  compare  the agreement of two masks of one root (the proxy against the truth).
  compose  the masks of a merge's roots, in merge order -> the merged root's mask.

This venv has no maker_arm_rs, so the pick points come from ``scripts/pick_points.py`` in the
lab venv (``lab_pick_points``).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from convert_to_lerobot import GRIPPER_TRAVEL, context_keys, cycle_segments

REPO_ID = "youtalk/maker-arm-legosort"
LAB = Path(__file__).resolve().parents[1]
# PHASE_TO_INT pins (core/oracle.py), checked by tests/test_lerobot_schema_pin.py.
LIFT_INT, REOPEN_INT = 4, 5
RISE_MIN_M = 0.05  # design 3.6: a risen lift is a LIFT in which the pick point rose >= 5 cm
FULL_OPEN_M = 2 * GRIPPER_TRAVEL  # the REOPEN command, 0.1048 m (continuous gripper)
OPEN_TOL_M = 1e-4
# Measured 2026-10-03 on the kept frames of f/w3 and f/w1 Machine A proc_0 (444 full-open runs):
# a risen REOPEN moves at most 0.3 mm in its 20 frames, a RETREAT (the only other full-open
# phase) at least 9.3 mm.
PROXY_STILL_M = 0.002
REOPEN_FRAMES = 20  # oracle.py: REOPEN holds 20 steps


def lab_pick_points(q: np.ndarray) -> np.ndarray:
    """(N, 6) URDF joint rows -> (N, 3) pick points, computed by the lab venv's FK."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, "q.npy"), Path(tmp, "pick.npy")
        np.save(src, np.asarray(q, dtype=np.float64))
        script = [str(LAB / ".venv/bin/python"), str(LAB / "scripts/pick_points.py")]
        out = subprocess.run(
            [*script, str(src), str(dst)],
            capture_output=True,
            text=True,
            check=False,  # the token decides: its absence raises below with the output
            env={**os.environ, "PYTHONPATH": str(LAB / "source")},
        )
        lines = out.stdout.strip().splitlines()
        if not lines or not lines[-1].startswith("PICK_OK"):
            raise RuntimeError(f"pick_points.py: {out.stdout[-500:]} {out.stderr[-2000:]}")
        return np.load(dst)


def _runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each run of equal values."""
    cuts = np.flatnonzero(np.diff(values) != 0) + 1
    return np.r_[0, cuts], np.r_[cuts, len(values)]


def truth_labels(phase: np.ndarray, lift_z: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame labels of one demo. ``lift_z`` is the pick point height on the LIFT frames
    (``phase == LIFT_INT``), in frame order.

    ``drop``: the REOPEN run right after a LIFT whose pick point rose at least RISE_MIN_M
    (highest point of the LIFT minus its first frame). ``reopen``: a REOPEN run right after any
    LIFT. ``risen`` / ``risen_reopen``: the first frame of a risen LIFT / of one that ends in
    REOPEN (the counts of design 3.6)."""
    z = np.full(len(phase), np.nan)
    z[phase == LIFT_INT] = lift_z
    labels = {k: np.zeros(len(phase), bool) for k in ("drop", "reopen", "risen", "risen_reopen")}
    starts, ends = _runs(phase)
    for k, (s, e) in enumerate(zip(starts, ends)):
        if phase[s] != LIFT_INT:
            continue
        risen = z[s:e].max() - z[s] >= RISE_MIN_M
        labels["risen"][s] = risen
        if k + 1 < len(starts) and phase[starts[k + 1]] == REOPEN_INT:
            reopen = slice(starts[k + 1], ends[k + 1])
            labels["reopen"][reopen] = True
            labels["drop"][reopen] = risen
            labels["risen_reopen"][s] = risen
    return labels


def demo_steps(grp, context: str | None) -> int:
    """The frames convert_to_lerobot.convert reads from a demo (the shortest of its datasets)."""
    keys = ["actions", "joint_targets", "obs/joint_pos_abs"]
    keys += ["obs/images/wrist_rgb", "obs/images/overhead_rgb", *context_keys(context)]
    return min(len(grp[k]) for k in keys)


def lane_order(shards: list[Path], lanes: int) -> list[Path]:
    """convert.sh's episode order: shard i goes to lane i % lanes, lanes merge in order."""
    return [s for lane in range(lanes) for s in shards[lane::lanes]]


def truth_mask(shards: list[Path], pick_points=lab_pick_points, context="camera_box_target"):
    """The mask and the design 3.6 counts of the root converted (--split_cycles --dagger) from
    ``shards`` in this order."""
    demos, lift_q = [], []
    for shard in shards:
        with h5py.File(shard, "r") as f:
            for name in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1])):
                grp = f["data"][name]
                steps = demo_steps(grp, context)
                phase = grp["obs/cycle_phase"][:steps]
                lift = phase == LIFT_INT
                lift_q.append(grp["obs/joint_pos_abs"][:steps][lift, :6])
                demos.append((phase, grp["obs/cycle_brick"][:steps], grp["obs/actor"][:steps]))
    lift_z = pick_points(np.concatenate(lift_q))[:, 2] if lift_q else np.zeros(0)
    split = np.cumsum([len(q) for q in lift_q])[:-1]
    lengths, drop, offset = [], [], 0
    counts = dict.fromkeys(("reopen", "risen", "risen_reopen", "episodes_reopen"), 0)
    for (phase, brick, actor), z in zip(demos, np.split(lift_z, split)):
        labels = truth_labels(phase, z)
        for seg in cycle_segments(phase, brick, 0, actor):
            lengths.append(len(seg))
            drop.extend(offset + np.flatnonzero(labels["drop"][seg]))
            offset += len(seg)
            for key in ("reopen", "risen", "risen_reopen"):
                counts[key] += int(labels[key][seg].sum())
            counts["episodes_reopen"] += bool(labels["reopen"][seg].any())
    return np.asarray(lengths, np.int64), np.asarray(drop, np.int64), counts


def episode_lengths(root: Path) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return np.asarray(LeRobotDatasetMetadata(REPO_ID, root=root).episodes["length"], np.int64)


def read_frames(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``episode_index``, ``observation.state`` (N, 7) and ``action`` (N, 7) in frame order."""
    t = pq.read_table(
        root / "data", columns=["index", "episode_index", "observation.state", "action"]
    )
    t = t.sort_by("index")
    if not np.array_equal(t["index"].to_numpy(), np.arange(t.num_rows)):
        raise RuntimeError(f"{root}: frame index is not 0..{t.num_rows - 1}")

    def rows(name):
        return pc.list_flatten(t[name]).to_numpy().reshape(t.num_rows, -1)

    return t["episode_index"].to_numpy(), rows("observation.state"), rows("action")


def proxy_drop(gripper: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """Frame rows (within one episode) of the full-open runs that look like a risen REOPEN."""
    full = np.abs(gripper - FULL_OPEN_M) < OPEN_TOL_M
    held = _runs(gripper)[0]  # where each constant command began
    drop = []
    for s, e in zip(*_runs(full.astype(np.int8))):
        if not full[s] or s == 0:
            continue
        h = held[np.searchsorted(held, s - 1, side="right") - 1]
        rise = pick[s, 2] - pick[h:s, 2].min()
        still = np.abs(pick[s : min(e, s + REOPEN_FRAMES)] - pick[s]).max()
        if rise >= RISE_MIN_M and still < PROXY_STILL_M:
            drop.extend(range(s, e))
    return np.asarray(drop, np.int64)


def proxy_mask(
    episode: np.ndarray, state: np.ndarray, action: np.ndarray, pick_points=lab_pick_points
):
    pick = pick_points(state[:, :6])
    starts, ends = _runs(episode)
    drop = [s + proxy_drop(action[s:e, 6], pick[s:e]) for s, e in zip(starts, ends)]
    return (ends - starts).astype(np.int64), np.concatenate([np.zeros(0, np.int64), *drop])


def compare(truth: np.ndarray, proxy: np.ndarray) -> dict:
    t, p = set(truth.tolist()), set(proxy.tolist())
    both = len(t & p)
    return {
        "truth": len(t),
        "proxy": len(p),
        "both": both,
        "truth_only": len(t - p),
        "proxy_only": len(p - t),
        "recall": both / max(len(t), 1),
        "precision": both / max(len(p), 1),
    }


def compose(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The mask of the root that aggregate_datasets merges from the parts' roots, in order."""
    offsets = np.cumsum([0] + [int(lengths.sum()) for lengths, _ in parts])
    lengths = np.concatenate([lengths for lengths, _ in parts])
    return lengths, np.concatenate([o + drop for o, (_, drop) in zip(offsets, parts)])


def save(path: Path, lengths: np.ndarray, drop: np.ndarray, **info) -> dict:
    summary = {
        "out": str(path),
        "episodes": len(lengths),
        "frames": int(lengths.sum()),
        "dropped": len(drop),
        "share": len(drop) / max(int(lengths.sum()), 1),
        **info,
    }
    np.savez(path, lengths=lengths, drop=drop, summary=json.dumps(summary))
    return summary


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as m:
        return m["lengths"], m["drop"]


def check_lengths(lengths: np.ndarray, root: Path) -> None:
    actual = episode_lengths(root)
    if not np.array_equal(lengths, actual):
        bad = next((i for i, (a, b) in enumerate(zip(lengths, actual)) if a != b), None)
        raise RuntimeError(
            f"{root}: the mask has {len(lengths)} episodes / {lengths.sum()} frames, the root "
            f"{len(actual)} / {actual.sum()}; first differing episode {bad}"
        )


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"MASK_FAIL usage: {message}", flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    truth = sub.add_parser("truth", help="raw shards -> the mask of the root made from them")
    truth.add_argument("--shards_dir", type=Path, required=True)
    truth.add_argument("--first", type=int, required=True, help="convert.sh FIRST")
    truth.add_argument("--last", type=int, required=True, help="convert.sh LAST")
    truth.add_argument("--lanes", type=int, required=True, help="convert.sh LANES")
    truth.add_argument("--root", type=Path, required=True, help="the converted root, to check")
    truth.add_argument("--out", type=Path, required=True)
    proxy = sub.add_parser("proxy", help="a converted root -> its mask from state and action")
    proxy.add_argument("--root", type=Path, required=True)
    proxy.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare", help="agreement of two masks of one root")
    cmp.add_argument("truth", type=Path)
    cmp.add_argument("proxy", type=Path)
    compose = sub.add_parser("compose", help="the masks of a merge's roots, in merge order")
    compose.add_argument("--dataset", type=Path, required=True, help="the merged root")
    compose.add_argument("--masks", type=Path, nargs="+", required=True)
    compose.add_argument("--out", type=Path, required=True)
    return parser


def run(args) -> dict:
    if args.cmd == "truth":
        # convert.sh: proc_p for p in FIRST..LAST, a shard without its meta.json is skipped.
        shards = [args.shards_dir / f"proc_{p}.hdf5" for p in range(args.first, args.last + 1)]
        shards = [s for s in shards if s.with_suffix(".meta.json").exists()]
        lengths, drop, counts = truth_mask(lane_order(shards, args.lanes))
        check_lengths(lengths, args.root)
        return save(
            args.out, lengths, drop, rule="truth", root=str(args.root), shards=len(shards), **counts
        )
    if args.cmd == "proxy":
        lengths, drop = proxy_mask(*read_frames(args.root))
        check_lengths(lengths, args.root)
        return save(args.out, lengths, drop, rule="proxy", root=str(args.root))
    if args.cmd == "compare":
        (lt, dt), (lp, dp) = load(args.truth), load(args.proxy)
        if not np.array_equal(lt, lp):
            raise RuntimeError("the two masks belong to different roots")
        return compare(dt, dp)
    lengths, drop = compose([load(m) for m in args.masks])
    check_lengths(lengths, args.dataset)
    return save(
        args.out, lengths, drop, rule="compose", root=str(args.dataset), parts=len(args.masks)
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:  # noqa: BLE001 -- any failure must become MASK_FAIL, not a crash.
        print(f"MASK_FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    print(f"MASK_OK {json.dumps(summary)}", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())

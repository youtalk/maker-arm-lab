"""Run A's prompt codes of a teacher root (MS7i design rev3 5.3, narrowed by the round 1 audit).

One int8 a frame, by its index: ``3 * subtask + tag``. The tag is ``train_pi05.POSITIVE`` on every
frame of a clean episode (``teacher_tags``) and untagged (0) on the rest. The subtask ids come from
``scripts/subtask_ids.py`` in the lab venv (``lab_subtask_ids``), because this venv has no
maker_arm_rs and cannot import core. The root is a V root (derive_vla_dataset.py or
dagger_vla.py): its frame index runs from 0 to N - 1 in file order, and one episode is one cycle.
The codes go to ``--out``, by default ``<root>/meta/prompt_codes.npy`` (train_pi05.CODES), which
``train_pi05.py --prompt_codes`` reads.

Run in the lerobot venv:
    lerobot_stack/.venv/bin/python lerobot_stack/prompt_codes.py \\
        --root ~/data/maker-arm-ms7/h/vla/dataset \\
        --out ~/data/maker-arm-ms7/i/vla/corpus_prompt_codes.npy
-> one CODES_OK {json} / CODES_FAIL line; the json gives each subtask's share of the frames.
2026-10-13, the corpus (9,676,460 frames, 23,957 episodes) on Machine A CPU, under the
subtask rule of the corpus audit (core.hold, the width the command answered): 23.4 s, shares
reach 0.3431, grasp 0.2380, carry 0.2428, release 0.1761; 83.3 % of the episodes read a carry.
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from derive_vla_dataset import columns, read_tables
from train_pi05 import CODES, PHRASES, POSITIVE

GRASP, CARRY, RELEASE = 1, 2, 3  # subtask ids: reach, grasp, carry, release (PHRASES)

LAB = Path(__file__).resolve().parents[1]


def lab_subtask_ids(frames: dict) -> np.ndarray:
    """The (N,) subtask ids of ``frames`` (scripts/subtask_ids.py's arrays), from the lab venv."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp, "frames.npz"), Path(tmp, "ids.npy")
        np.savez(src, **frames)
        script = [str(LAB / ".venv/bin/python"), str(LAB / "scripts/subtask_ids.py")]
        out = subprocess.run(
            [*script, str(src), str(dst)],
            capture_output=True,
            text=True,
            check=False,  # the token decides: its absence raises below with the output
            env={**os.environ, "PYTHONPATH": str(LAB / "source")},
        )
        lines = out.stdout.strip().splitlines()
        if not lines or not lines[-1].startswith("SUBTASK_OK"):
            raise RuntimeError(f"subtask_ids.py: {out.stdout[-500:]} {out.stderr[-2000:]}")
        return np.load(dst)


def read_frames(root: Path) -> dict:
    """scripts/subtask_ids.py's arrays of every frame of ``root``, in index order."""
    features = json.loads((root / "meta/info.json").read_text())["features"]
    state_names = features["observation.state"]["names"]
    table = read_tables(root / "data", ["index", "episode_index", "observation.state", "action"])
    index = table["index"].to_numpy()
    if not np.array_equal(index, np.arange(len(index))):
        raise ValueError(f"{root}: the frame index is not 0 to N - 1 in file order")
    state, action = columns(table, "observation.state"), columns(table, "action")
    return {
        "q": state[:, :6],
        "opening": state[:, state_names.index("gripper_open")],
        "cmd": action[:, features["action"]["names"].index("gripper_gap_m")],
        "width": state[:, state_names.index("target_width")],
        "episode": table["episode_index"].to_numpy(),
    }


def teacher_tags(ids: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """POSITIVE on each episode that reaches a carry and then a release with no close after it,
    untagged (0) on the rest. The round 1 audit (ledger IR6) judged 6 of 20 corpus episodes as
    failed picks: no carry (16.7 % of the episodes), or a regrasp after the release (12.9 %)."""
    tags = np.zeros(len(ids), np.int8)
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(ids)]):
        q = ids[a:b]
        carry = np.flatnonzero(q == CARRY)
        if not len(carry):
            continue
        release = np.flatnonzero(q[carry[0] :] == RELEASE)
        if len(release) and not np.isin(q[carry[0] + release[0] :], (GRASP, CARRY)).any():
            tags[a:b] = POSITIVE
    return tags


def write_codes(root: Path, out: Path, subtask_ids=lab_subtask_ids) -> dict:
    frames = read_frames(root)
    ids = np.asarray(subtask_ids(frames))
    if ids.shape != frames["episode"].shape:
        raise ValueError(f"{ids.shape} subtask ids for {len(frames['episode'])} frames")
    tags = teacher_tags(ids, frames["episode"])
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, (3 * ids + tags).astype(np.int8))
    os.replace(tmp, out)
    shares = np.bincount(ids, minlength=len(PHRASES)) / max(len(ids), 1)
    return {
        "root": str(root),
        "out": str(out),
        "frames": len(ids),
        "episodes": len(np.unique(frames["episode"])),
        "shares": dict(zip(PHRASES, np.round(shares, 4).tolist())),
        "positive": round(float((tags == POSITIVE).mean()), 4),
    }


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"CODES_FAIL usage: {message}", flush=True)
        os._exit(2)


def main() -> int:
    parser = TokenArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="a V root")
    parser.add_argument("--out", type=Path, help=f"default <root>/{CODES}")
    try:
        args = parser.parse_args()
        started = time.monotonic()
        summary = write_codes(args.root, args.out or args.root / CODES)
        summary["seconds"] = round(time.monotonic() - started, 1)
    except Exception as exc:  # noqa: BLE001 -- any failure must become CODES_FAIL
        traceback.print_exc()
        print(f"CODES_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("CODES_OK " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())

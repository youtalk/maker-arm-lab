"""Run A's prompt codes of a teacher root (MS7i design rev3 5.3: a teacher frame is positive).

One int8 a frame, by its index: ``3 * subtask + train_pi05.POSITIVE``. The subtask ids come from
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
2026-10-13, the corpus (9,676,460 frames, 23,957 episodes) on Machine A CPU: 22.7 s, shares
reach 0.3865, grasp 0.2754, carry 0.1958, release 0.1423.
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


def write_codes(root: Path, out: Path, subtask_ids=lab_subtask_ids) -> dict:
    frames = read_frames(root)
    ids = np.asarray(subtask_ids(frames))
    if ids.shape != frames["episode"].shape:
        raise ValueError(f"{ids.shape} subtask ids for {len(frames['episode'])} frames")
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, (3 * ids + POSITIVE).astype(np.int8))
    os.replace(tmp, out)
    shares = np.bincount(ids, minlength=len(PHRASES)) / max(len(ids), 1)
    return {
        "root": str(root),
        "out": str(out),
        "frames": len(ids),
        "episodes": len(np.unique(frames["episode"])),
        "shares": dict(zip(PHRASES, np.round(shares, 4).tolist())),
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

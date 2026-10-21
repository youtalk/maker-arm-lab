"""Parity of pi0.5 serving across LeRobot versions (MS7j plan, Task 3).

    python pi05_parity.py obs --dataset ROOT --frames 16 --seed 7 --out OBS.npz
    python pi05_parity.py chunk --checkpoint DIR --obs OBS.npz --noise_seed 7 --out CHUNK.npy
    python pi05_parity.py compare A.npy B.npy

``obs`` takes frames of the V corpus, one from each of ``--frames`` episodes picked by the seed.
``chunk`` serves them through ``pi05_server.load_pi05`` with a fixed initial noise, one forward a
frame, and saves the postprocessed actions (frames, 50, 7). Run it once in each venv, then
``compare`` the two files: the p99 of the absolute difference per action column must stay under
0.01 rad for the joints and 0.0005 m for the jaw. It imports only pi05_server and LeRobot's
public API, so it runs under 0.5.1 and 0.6.1. Each command prints one PARITY_OK or PARITY_FAIL
line and exits through os._exit.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

BARS = np.array([0.01] * 6 + [0.0005])  # joints in rad, jaw in m
MAX_ACTION_DIM = 32  # pi05's padded action width


def obs_command(dataset: str, frames: int, seed: int, out: str) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data = LeRobotDataset("local/parity", root=dataset)
    episode = np.asarray(data.hf_dataset["episode_index"])
    rng = np.random.default_rng(seed)
    picks = rng.choice(np.unique(episode), size=frames, replace=False)
    rows = [data[int(rng.choice(np.flatnonzero(episode == e)))] for e in picks]

    def image(row, key):  # CHW float in [0, 1] -> HWC uint8
        return (row[key].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)

    np.savez(
        out,
        state=np.stack([r["observation.state"].numpy() for r in rows]).astype(np.float32),
        wrist=np.stack([image(r, "observation.images.wrist") for r in rows]),
        overhead=np.stack([image(r, "observation.images.overhead") for r in rows]),
        task=np.array([r["task"] for r in rows]),
    )
    return {"frames": frames, "episodes": picks.tolist(), "out": out}


def chunk_command(checkpoint: str, obs: str, noise_seed: int, out: str) -> dict:
    import torch

    from pi05_server import load_pi05

    predict, _, inference, _, _ = load_pi05(checkpoint, "cuda")
    data = np.load(obs)
    frames = len(data["state"])
    shape = (frames, inference["chunk_size"], MAX_ACTION_DIM)
    noise = np.random.default_rng(noise_seed).standard_normal(shape).astype(np.float32)
    chunks = []
    for i in range(frames):
        row = {k: data[k][i : i + 1] for k in ("state", "wrist", "overhead")}
        chunks.append(predict(row, noise=torch.from_numpy(noise[i : i + 1]).to("cuda")))
    np.save(out, np.concatenate(chunks))
    return {"chunks": frames, "out": out}


def compare_command(a: str, b: str) -> tuple[bool, dict]:
    x, y = np.load(a), np.load(b)
    if x.shape != y.shape or x.shape[-1] != len(BARS):
        return False, {"error": f"shapes {x.shape} and {y.shape}, want equal (..., 7)"}
    p99 = np.percentile(np.abs(x - y).reshape(-1, len(BARS)), 99, axis=0)
    return bool((p99 <= BARS).all()), {"p99": p99.tolist(), "bars": BARS.tolist()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("obs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("chunk")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--obs", required=True)
    p.add_argument("--noise_seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    try:
        args = parser.parse_args(argv)
        if args.command == "obs":
            ok, info = True, obs_command(args.dataset, args.frames, args.seed, args.out)
        elif args.command == "chunk":
            ok = True
            info = chunk_command(args.checkpoint, args.obs, args.noise_seed, args.out)
        else:
            ok, info = compare_command(args.a, args.b)
    except SystemExit as exc:  # argparse error or --help
        if exc.code:
            print("PARITY_FAIL " + json.dumps({"error": "usage"}), flush=True)
        return int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 -- any failure must become the fail token
        traceback.print_exc()
        print("PARITY_FAIL " + json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))
        return 1
    print(("PARITY_OK " if ok else "PARITY_FAIL ") + json.dumps(info), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.flush()
    os._exit(main(sys.argv[1:]))

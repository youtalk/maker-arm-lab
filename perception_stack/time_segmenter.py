"""MS5 design P1: the perception worker's two calls, timed on this host with SAM 2.1.

- A perceive call: encode the ``--tray_px`` square tray crop at full resolution, then decode
  ``--parts`` point prompts in one batch. Bar: p95 at most 500 ms.
- A wrist look: encode the ``--wrist_px`` wrist frame, then decode one prompt. Bar: p95 at
  most 150 ms.

Each model in ``--models`` (Hugging Face ids) runs ``--warmup`` untimed rounds, then
``--calls`` timed ones of each call, on random images with prompts on a grid: the encoder
and decoder cost does not depend on the content. bfloat16 autocast, as the worker will run.
Prints ``SEGTIME_OK`` when at least one model meets both bars, else ``SEGTIME_FAIL``, with
the per-model table, on every exit path, and exits through ``os._exit``.

    perception_stack/.venv/bin/python perception_stack/time_segmenter.py \\
        --models facebook/sam2.1-hiera-tiny,facebook/sam2.1-hiera-small --calls 100
"""

import argparse
import json
import math
import os
import time
import traceback

import numpy as np

PERCEIVE_BAR_MS = 500.0
WRIST_BAR_MS = 150.0


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("SEGTIME_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="facebook/sam2.1-hiera-tiny,facebook/sam2.1-hiera-small"
    )
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tray_px", type=int, default=1024)
    parser.add_argument("--wrist_px", type=int, default=640)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def grid_prompts(n: int, size: int) -> np.ndarray:
    """(n, 1, 2) float32 pixel points, one per prompt, on the smallest square grid holding n,
    each at the centre of its cell."""
    side = math.ceil(math.sqrt(n))
    cell = size / side
    pts = [((k % side + 0.5) * cell, (k // side + 0.5) * cell) for k in range(n)]
    return np.asarray(pts, dtype=np.float32).reshape(n, 1, 2)


def _stats(values) -> dict:
    xs = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(xs, 50)), 1),
        "p95": round(float(np.percentile(xs, 95)), 1),
        "max": round(float(xs.max()), 1),
    }


def verdict(perceive_ms, wrist_ms) -> dict:
    perceive, wrist = _stats(perceive_ms), _stats(wrist_ms)
    return {
        "perceive_ms": perceive,
        "wrist_ms": wrist,
        "pass": perceive["p95"] <= PERCEIVE_BAR_MS and wrist["p95"] <= WRIST_BAR_MS,
    }


def time_model(name: str, args) -> dict:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(name, device="cuda")
    rng = np.random.default_rng(args.seed)

    def one(size: int, n: int) -> float:
        image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        labels = np.ones((n, 1), dtype=np.int32)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(image)
            predictor.predict(
                point_coords=grid_prompts(n, size), point_labels=labels, multimask_output=False
            )
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - started)

    for _ in range(args.warmup):
        one(args.tray_px, args.parts)
        one(args.wrist_px, 1)
    perceive = [one(args.tray_px, args.parts) for _ in range(args.calls)]
    wrist = [one(args.wrist_px, 1) for _ in range(args.calls)]
    return verdict(perceive, wrist)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        models = [m for m in args.models.split(",") if m]
        if not models or args.calls < 1:
            raise ValueError("need at least one model and --calls >= 1")
        table = {name: time_model(name, args) for name in models}
        ok = any(row["pass"] for row in table.values())
        summary = {
            "pass": ok,
            "bars_ms": {"perceive_p95": PERCEIVE_BAR_MS, "wrist_p95": WRIST_BAR_MS},
            "tray_px": args.tray_px,
            "wrist_px": args.wrist_px,
            "parts": args.parts,
            "calls": args.calls,
            "models": table,
        }
        print(("SEGTIME_OK " if ok else "SEGTIME_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("SEGTIME_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())

"""MS5 design P1: the perception worker's calls, timed on this host with SAM 2.1.

- A perceive call (the verdict row): encode the ``--tray_px`` square tray crop at full
  resolution, then decode ``--parts`` point prompts in one batch. Bar: p95 at most 500 ms.
  This row assumes the worker already holds one point prompt per part from an upstream
  proposal step, for example the depth blobs above the tray floor. It does not cover a worker
  that has to find the parts with the segmenter alone.
- An automatic perceive call (informational): ``SAM2AutomaticMaskGenerator`` on the same tray
  crop with ``--auto_points_per_side`` (32, the generator's default: 1024 prompts in batches
  of 64, three masks each, then the stability filter and NMS). That is the perceive call of a
  worker without upstream prompts. It runs on a synthetic tray (``--parts`` flat-colored
  rectangles on a grey floor), because its filter and NMS cost depends on what the image
  holds. Its p95 is reported against the same 500 ms bar but does not enter the token: spec
  MS5 design P1 does not fix how the worker proposes parts, and the worker is designed in
  wave 3. ``--auto_calls 0`` skips it.
- A wrist look (the verdict row): encode the ``--wrist_px`` wrist frame, then decode one
  prompt. Bar: p95 at most 150 ms.

Each model in ``--models`` (Hugging Face ids) runs ``--warmup`` untimed rounds, then
``--calls`` timed prompt perceive calls and wrist looks (random images with prompts on a grid:
the encoder and decoder cost does not depend on the content) and ``--auto_calls`` (default
``--calls``) timed automatic calls. bfloat16 autocast, as the worker will run. Prints
``SEGTIME_OK`` when at least one model's prompt perceive row and wrist row both meet their
bars, else ``SEGTIME_FAIL``, with the per-model table (the automatic row included), on every
exit path (a Ctrl-C included, reason ``interrupted``), and exits through ``os._exit``.

Run it with ``LD_LIBRARY_PATH`` unset: a JetPack profile's ``/usr/local/cuda/lib64`` shadows
the venv's pip cuBLAS and every CUDA op then fails.

    env -u LD_LIBRARY_PATH perception_stack/.venv/bin/python perception_stack/time_segmenter.py \\
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
# SAM2AutomaticMaskGenerator's own defaults.
AUTO_POINTS_PER_SIDE = 32
AUTO_POINTS_PER_BATCH = 64


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
    parser.add_argument("--auto_calls", type=int, default=None)
    parser.add_argument("--auto_points_per_side", type=int, default=AUTO_POINTS_PER_SIDE)
    return parser


def validate(args) -> None:
    """Raises ValueError naming every out-of-range argument."""
    errors = []
    if not [m for m in args.models.split(",") if m]:
        errors.append("--models names no model")
    if args.calls < 1:
        errors.append("--calls must be >= 1")
    if args.warmup < 0:
        errors.append("--warmup must be >= 0")
    if args.parts < 1:
        errors.append("--parts must be >= 1")
    if args.tray_px < 1:
        errors.append("--tray_px must be >= 1")
    if args.wrist_px < 1:
        errors.append("--wrist_px must be >= 1")
    if args.auto_calls is not None and args.auto_calls < 0:
        errors.append("--auto_calls must be >= 0")
    if args.auto_points_per_side < 1:
        errors.append("--auto_points_per_side must be >= 1")
    if errors:
        raise ValueError("; ".join(errors))


def grid_prompts(n: int, size: int) -> np.ndarray:
    """(n, 1, 2) float32 pixel points, one per prompt, on the smallest square grid holding n,
    each at the centre of its cell."""
    side = math.ceil(math.sqrt(n))
    cell = size / side
    pts = [((k % side + 0.5) * cell, (k // side + 0.5) * cell) for k in range(n)]
    return np.asarray(pts, dtype=np.float32).reshape(n, 1, 2)


def synthetic_tray(size: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    """(size, size, 3) uint8: a grey floor with ``parts`` flat-colored rectangles, each side
    between 1/24 and 1/10 of the image, placed at random (they may touch or overlap, as a
    heap does)."""
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    lo, hi = max(1, size // 24), max(2, size // 10)
    for _ in range(parts):
        h, w = rng.integers(lo, hi + 1, 2)
        y, x = rng.integers(0, max(1, size - h + 1)), rng.integers(0, max(1, size - w + 1))
        image[y : y + h, x : x + w] = rng.integers(0, 256, 3, dtype=np.uint8)
    return image


def _stats(values) -> dict:
    xs = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(xs, 50)), 1),
        "p95": round(float(np.percentile(xs, 95)), 1),
        "max": round(float(xs.max()), 1),
    }


def verdict(perceive_ms, wrist_ms, auto_ms=None, auto_masks=None) -> dict:
    """One model's row. ``pass`` reads the prompt perceive and the wrist p95 only; the
    automatic row, when given, carries its own ``under_bar`` against the perceive bar."""
    perceive, wrist = _stats(perceive_ms), _stats(wrist_ms)
    row = {
        "perceive_ms": perceive,
        "wrist_ms": wrist,
        "pass": perceive["p95"] <= PERCEIVE_BAR_MS and wrist["p95"] <= WRIST_BAR_MS,
    }
    if auto_ms:
        auto = _stats(auto_ms)
        auto["under_bar"] = auto["p95"] <= PERCEIVE_BAR_MS
        if auto_masks:
            auto["masks_p50"] = float(np.percentile(np.asarray(auto_masks, dtype=float), 50))
        row["auto_ms"] = auto
    return row


def time_model(name: str, args) -> dict:
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(name, device="cuda")
    generator = SAM2AutomaticMaskGenerator(
        predictor.model,
        points_per_side=args.auto_points_per_side,
        points_per_batch=AUTO_POINTS_PER_BATCH,
    )
    rng = np.random.default_rng(args.seed)
    auto_calls = args.calls if args.auto_calls is None else args.auto_calls

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

    def auto() -> tuple[float, int]:
        image = synthetic_tray(args.tray_px, args.parts, rng)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = generator.generate(image)
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - started), len(masks)

    for _ in range(args.warmup):
        one(args.tray_px, args.parts)
        one(args.wrist_px, 1)
        if auto_calls:
            auto()
    perceive = [one(args.tray_px, args.parts) for _ in range(args.calls)]
    wrist = [one(args.wrist_px, 1) for _ in range(args.calls)]
    auto_runs = [auto() for _ in range(auto_calls)]
    return verdict(perceive, wrist, [ms for ms, _ in auto_runs], [n for _, n in auto_runs])


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate(args)
        models = [m for m in args.models.split(",") if m]
        table = {name: time_model(name, args) for name in models}
        ok = any(row["pass"] for row in table.values())
        auto_calls = args.calls if args.auto_calls is None else args.auto_calls
        summary = {
            "pass": ok,
            "verdict_rows": ["perceive_ms", "wrist_ms"],
            "bars_ms": {
                "perceive_p95": PERCEIVE_BAR_MS,
                "wrist_p95": WRIST_BAR_MS,
                "auto_p95": PERCEIVE_BAR_MS,
            },
            "prompts": "one upstream point prompt per part (e.g. depth blobs)",
            "auto": {
                "informational": True,
                "calls": auto_calls,
                "points_per_side": args.auto_points_per_side,
                "points_per_batch": AUTO_POINTS_PER_BATCH,
                "image": "synthetic tray",
                "any_under_bar": any(
                    row.get("auto_ms", {}).get("under_bar", False) for row in table.values()
                ),
            },
            "tray_px": args.tray_px,
            "wrist_px": args.wrist_px,
            "parts": args.parts,
            "calls": args.calls,
            "models": table,
        }
        print(("SEGTIME_OK " if ok else "SEGTIME_FAIL ") + json.dumps(summary), flush=True)
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("SEGTIME_FAIL " + json.dumps({"reason": "interrupted"}), flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("SEGTIME_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())

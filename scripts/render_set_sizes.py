"""The perceived size beside each render set frame (MS7f v3 classifier input): what the heap loop's
perceived state reports for the part a view classifies, recomputed from the frame labels.

The loop's ``perception.Perceiver`` reports a detection's size as the catalog extents along the
part's body axes plus ``SIZE_SIGMA_M`` of Gaussian noise per axis, twice that on the axis nearest
world z (the depth axis), clipped at 0. This script draws the same noise on each overhead row's
part (``parts.by_name``), with the doubled axis taken as the part's smallest extent: a part at
rest lies on its largest face (the render set does not keep the orientation). A wrist row (lift,
carry, wrist) gets the size of its env's latest selection target, the first overhead row of that
selection's step, as the loop reads the cycle target's detection at selection; -1 before any.
Sizes are sorted, smallest first, in metres.

    PYTHONPATH=$PWD/source .venv/bin/python scripts/render_set_sizes.py rs/render_*.npz

Writes ``render_<seed>.size.npy`` (N, 3) float32 beside each file, generator seeded by the file's
run seed. Prints exactly one ``SIZES_OK`` or ``SIZES_FAIL`` line on every exit path, argparse
errors included, and exits through ``os._exit``.
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np

from maker_arm_lab.core import parts
from maker_arm_lab.core.perception import SIZE_SIGMA_M


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("SIZES_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def perceived(size, rng: random.Random) -> list[float]:
    """One detection's size: ``size`` plus the perception noise, sorted."""
    depth = min(range(3), key=lambda k: size[k])  # ponytail: resting-face guess, no pose kept
    noise = [SIZE_SIGMA_M * rng.gauss(0.0, 1.0) * (2.0 if k == depth else 1.0) for k in range(3)]
    return sorted(max(0.0, e + n) for e, n in zip(size, noise))


def sizes(z, rng: random.Random) -> tuple[np.ndarray, int]:
    """(N, 3) sizes for one render file's labels, and the wrist rows whose target is not the
    part in the jaws."""
    view, name, held = z["view"], z["part"], z["held"]
    env, step = z["env"], z["step"]
    table = {p.name: p.size for p in parts.by_name(sorted(set(name[view == "overhead"])))}
    out = np.full((len(view), 3), -1.0, dtype=np.float32)
    target: dict[int, tuple[int, int]] = {}  # env -> (selection step, its first overhead row)
    other = 0
    for n in range(len(view)):
        e = int(env[n])
        if view[n] == "overhead":
            out[n] = perceived(table[name[n]], rng)
            if target.get(e, (None,))[0] != int(step[n]):
                target[e] = (int(step[n]), n)
        elif e in target:
            first = target[e][1]
            out[n] = out[first]
            other += int(held[n] >= 0 and held[first] != held[n])
    return out, other


def main(argv=None) -> int:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="render_<seed>.npz files")
    args = parser.parse_args(argv)
    try:
        rows = other = 0
        for path in args.files:
            z = np.load(path)
            seed = json.loads(str(z["args"]))["run_seed"]
            out, n = sizes(z, random.Random(f"size:{seed}"))
            np.save(path.with_name(path.name.removesuffix(".npz") + ".size.npy"), out)
            rows, other = rows + len(out), other + n
        summary = {"files": len(args.files), "rows": rows, "other": other}
        print("SIZES_OK " + json.dumps(summary), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("SIZES_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())

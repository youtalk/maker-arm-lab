"""Truth layouts of parts in the tray and their iPad templates (MS5 sensing).

    uv run python scripts/real_layouts.py --rig rig.json --parts parts.json --seed 1 \\
        --sparse 3 --cluster 3 --heap 2 --out_dir ~/data/maker-arm-ms5/layouts

``parts.json`` is ``[{"name": "<catalog name>", "count": n}]``. An entry with ``"name": null``
and a ``"label"`` is a real part with no catalog counterpart: it stays out of the draws and is
listed under ``unmatched``. Writes ``<name>.json`` for every layout, ``<name>.png`` for every
sparse and cluster layout, and ``empty.png`` (the tray outline and the scale bar) and ``board.png`` (the AprilTag board of
``real.tags``, for the overhead calibration). Prints
``LAYOUTS_OK`` with the paths, or ``LAYOUTS_FAIL``, on every exit path and exits through
``os._exit``. A heap layout uses every counted part once; a sparse or cluster layout draws
``SPARSE_COUNT`` or ``CLUSTER_COUNT`` of them.
"""

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

SPARSE_COUNT = 4
CLUSTER_COUNT = 5


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("LAYOUTS_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True)
    parser.add_argument("--parts", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sparse", type=int, default=3)
    parser.add_argument("--cluster", type=int, default=3)
    parser.add_argument("--heap", type=int, default=2)
    parser.add_argument("--out_dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from maker_arm_lab.core import parts as parts_mod
        from maker_arm_lab.real.layouts import IPAD_AIR_11, Layout, draw_layout, template_png
        from maker_arm_lab.real.rig import RigConfig

        rig = RigConfig.from_json(args.rig)
        entries = json.loads(Path(args.parts).read_text())
        names, unmatched = [], []
        for e in entries:
            if e.get("name") is None:
                unmatched.append(e["label"])
            else:
                names += [e["name"]] * int(e["count"])
        parts_mod.by_name(names)  # raises KeyError for a name outside the catalog
        out = Path(args.out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"ms5-layouts:{args.seed}")
        paths = []
        for kind, n, take in (
            ("sparse", args.sparse, SPARSE_COUNT),
            ("cluster", args.cluster, CLUSTER_COUNT),
            ("heap", args.heap, len(names)),
        ):
            for i in range(n):
                pick = rng.sample(names, min(take, len(names)))
                lay: Layout = draw_layout(rng, kind, pick, rig, f"{kind}{i}")
                (out / f"{lay.name}.json").write_text(lay.to_json())
                paths.append(str(out / f"{lay.name}.json"))
                if kind != "heap":
                    template_png(lay, rig, IPAD_AIR_11).save(out / f"{lay.name}.png")
                    paths.append(str(out / f"{lay.name}.png"))
        template_png(Layout("empty", "sparse", [], []), rig, IPAD_AIR_11).save(out / "empty.png")
        paths.append(str(out / "empty.png"))
        import cv2

        from maker_arm_lab.real.tags import IPAD_BOARD, board_png

        cv2.imwrite(str(out / "board.png"), board_png(IPAD_BOARD)[..., ::-1])
        paths.append(str(out / "board.png"))
        print("LAYOUTS_OK " + json.dumps({"paths": paths, "unmatched": unmatched}), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("LAYOUTS_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())

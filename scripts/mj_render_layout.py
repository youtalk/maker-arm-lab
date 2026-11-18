"""Render a truth layout in the MuJoCo scene beside the real overhead view.

    MUJOCO_GL=egl uv run python scripts/mj_render_layout.py --layout lay.json --rig rig.json \\
        --real view.png --out side_by_side.png

Builds a one-env heap scene like ``mj_eval_heap.py`` (``--tray --tray_lean 30 --desk_boxes 4``,
no episode draws, no frame jitter) with the layout's parts. A nonzero ``rig.overhead_back_m``
means the B2 camera, so the scene sets ``release_lean``. The simulator's own tray and boxes
are used as they are: the real ones differ in size, and the difference is recorded, not
corrected. Writes the simulator overhead (left) and the real view (right), the same height,
into ``--out``. A heap layout has no positions and fails. Prints ``RENDER_OK`` or
``RENDER_FAIL`` on every exit path and exits through ``os._exit``.
"""

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("RENDER_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    for name in ("layout", "rig", "real", "out"):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from PIL import Image

        from maker_arm_lab.core import parts as parts_mod
        from maker_arm_lab.core.layout import sample_run_draw
        from maker_arm_lab.mj.env import MjLegoSortEnv
        from maker_arm_lab.real.layouts import Layout
        from maker_arm_lab.real.rig import RigConfig

        rig = RigConfig.from_json(args.rig)
        lay = Layout.from_json(args.layout)
        if any(math.isnan(p.x) for p in lay.parts):
            raise ValueError(f"layout {lay.name} has no positions (a heap): nothing to render")
        parts = parts_mod.by_name([p.name for p in lay.parts])
        draw = sample_run_draw(random.Random(0))
        draw["slot_colors"] = (0, 1, 2, 3)
        env = MjLegoSortEnv(
            1,
            draw,
            num_bricks=len(parts),
            parts=parts,
            cameras=True,
            tray=True,
            tray_lean_deg=30.0,
            desk_boxes=4,
            class_names=parts_mod.PART_CLASSES,  # a layout draws from all eight classes
            release_lean=rig.overhead_back_m != 0.0,
        )
        env.reset(seed=0)
        env.place_desk_boxes(0, [(b.x, b.y, b.yaw) for b in lay.boxes])
        env.set_part_poses(0, [(p.x, p.y, p.z, p.yaw) for p in lay.parts])
        sim = Image.fromarray(env.obs_buf["images"]["overhead_rgb"][0].numpy())
        real = Image.open(args.real).convert("RGB")
        sim = sim.resize((round(sim.width * real.height / sim.height), real.height))
        canvas = Image.new("RGB", (sim.width + real.width, real.height))
        canvas.paste(sim, (0, 0))
        canvas.paste(real, (sim.width, 0))
        canvas.save(args.out)
        print("RENDER_OK " + json.dumps({"out": args.out, "parts": len(parts)}), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("RENDER_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())

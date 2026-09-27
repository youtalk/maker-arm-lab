"""The heap gate client on the real arm (MS5 design sections 4.2, 7 and 8): one process runs
one real episode.

The MS7d served configuration on ``RealEnv``: the unchanged heap loop under ``perceived`` with
every switch of the MS7d official read (``SERVED_SWITCHES``) and the knob scales at ``all=0``,
because the noise is the rig's own. The policy server (lerobot_stack) and the perception
worker run on this host; the arm runs ``Tracking`` in the SDK's loop.

    source /opt/ros/jazzy/setup.zsh
    uv run python scripts/real_eval_heap.py --rig ~/data/maker-arm-ms5/rig.json \\
        --sheet ~/data/maker-arm-ms5/sheets/e000.json --heap_k 10 --dilation 0.5 \\
        --run_seed 20265100 --out ~/data/maker-arm-ms5/reads/e000.json

It prints exactly one ``EVAL_OK`` or ``EVAL_FAIL`` line on every exit path and exits through
``os._exit``. A failure (a watchdog, Ctrl-C, an exception) holds the arm first. With the token
printed, the motors are disabled only on the operator's typed ``RELEASE`` or end of input
(``--release prompt``, the default) or at once (``--release now``, for the SDK simulator and
tests): a release makes the arm go limp. After a finished episode the ``Tracking`` plan holds
the last pose until the release. The payload's truth metrics are blind (``RealEnv``): the
operator's box count grades the episode, and the payload says ``real.graded: false``.
"""

import argparse
import json
import os
import random
import signal
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "lerobot_stack"))

from maker_arm_lab.core.eval_loop import EvalFailure

SLOTS_PER_PART = 2
"""The loop's detection slots per part poured: room for a segmenter that splits a part."""
POLICY_TIMEOUT_MS = 1000
"""A policy step with no reply within this holds the arm (MS5 design section 8)."""
SERVED_SWITCHES = {
    "select_at_release": True,
    "replan_on_switch": True,
    "squeeze_floor": True,
    "square_wrist": True,
    "scripted_rake": True,
    "part_failures": True,
    "pick_snap": True,
    "dwell_close": True,
    "adopt_held": True,
    "park_done": True,
    "rake_turn": True,
    "box_filter_xy": True,
    "release_gap": True,
    "rearm_hold": True,
    "box_walls": True,
    "marker_bias": True,
    "wrist_fix": True,
    "class_views": True,
    "hold_force": True,
    "force_cap": True,
}
"""The MS7d official read's switches (docs/ms7d-perception-levers.md, rule R3 and the nine
levers), served unchanged on the rig."""


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("EVAL_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--rig", default="", help="rig JSON (real.rig.RigConfig); default nominal")
    parser.add_argument("--sheet", required=True, help="the episode sheet (real_episode_sheet.py)")
    parser.add_argument("--heap_k", type=int, required=True, help="parts poured into the tray")
    parser.add_argument(
        "--dilation", type=float, required=True, help="0.5 for first episodes, 1.0 for reads"
    )
    parser.add_argument("--run_seed", type=int, default=0)
    parser.add_argument("--layout_seed", type=int, default=0)
    parser.add_argument(
        "--arm", default="", help="'sim' for the SDK simulator; default the rig's CAN interface"
    )
    parser.add_argument("--server", default="tcp://127.0.0.1:5555")
    parser.add_argument("--worker", default="tcp://127.0.0.1:5600")
    parser.add_argument("--namespace", default="/conduit")
    parser.add_argument("--max_wall_minutes", type=float, default=None)
    parser.add_argument("--release", choices=("prompt", "now"), default="prompt")
    parser.add_argument("--out", default="")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.heap_k <= 0:
        parser.error(f"--heap_k must be positive, got {args.heap_k}")
    if not 0.0 < args.dilation <= 1.0:
        parser.error(f"--dilation must be in (0, 1], got {args.dilation}")
    return args


def open_sources(rig, namespace: str, closers: list):
    from maker_arm_lab.core.cameras import WRIST_FOV_DEG
    from maker_arm_lab.real.camera_model import Pinhole, Pose, quat_matrix, wrist_warp
    from maker_arm_lab.real.ros import ConduitSubscriber
    from maker_arm_lab.real.sources import ConduitOverhead, UvcWrist

    camera = None if rig.overhead_k is None else Pinhole.from_k(*rig.overhead_size, rig.overhead_k)
    pose = Pose.from_pos_quat(rig.overhead_pos, rig.overhead_quat)
    overhead = ConduitOverhead(pose, camera, rig.desk_z)
    subscriber = ConduitSubscriber(
        namespace,
        on_jpeg=overhead.on_jpeg,
        on_info=overhead.on_info,
        on_depth=overhead.on_depth,
        on_depth_info=overhead.on_depth_info,
        on_error=overhead.on_error,
    ).start()
    closers.append(subscriber.close)
    if rig.wrist_k is None:
        wrist_camera, warp = Pinhole.sim(WRIST_FOV_DEG), None
    else:
        wrist_camera = Pinhole.from_k(*rig.wrist_size, rig.wrist_k)
        warp = wrist_warp(wrist_camera, quat_matrix(rig.wrist_rot).T)
    wrist = UvcWrist(rig.wrist_device, wrist_camera, warp).start()
    closers.append(wrist.close)
    return overhead, wrist


def build_world(args, world: dict) -> None:
    """Fill ``world`` step by step, so that a failure halfway still finds the arm to hold."""
    import policy_client
    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.core import oracle
    from maker_arm_lab.core.heap_oracle import HeapOracle
    from maker_arm_lab.core.layout import sample_run_draw
    from maker_arm_lab.core.loop import cycle_budget, episode_length_s
    from maker_arm_lab.real.arm import SdkArm
    from maker_arm_lab.real.env import RealEnv
    from maker_arm_lab.real.pacing import TimedPolicy
    from maker_arm_lab.real.perceive import RealPerception, WorkerClient
    from maker_arm_lab.real.rig import RigConfig
    from maker_arm_lab.tasks.manager_based.lego_sort.oracle_runtime import OracleRig

    config = RigConfig.from_json(Path(args.rig).expanduser()) if args.rig else RigConfig()
    sheet = json.loads(Path(args.sheet).expanduser().read_text())
    budget = cycle_budget(args.heap_k)
    world.update(config=config, sheet=sheet, budget=budget, closers=[])
    interface = args.arm or config.can_interface
    arm = SdkArm.connect(
        interface,
        dm.dynamics(dm.urdf_bodies()),
        params=config.tracking_params,
        rt=None if interface == "sim" else config.rt,
    )
    world["arm"] = arm
    arm.start()
    overhead, wrist = open_sources(config, args.namespace, world["closers"])
    world["overhead"] = overhead
    env = RealEnv(
        arm,
        overhead,
        wrist,
        config,
        num_slots=SLOTS_PER_PART * args.heap_k,
        class_to_box=sheet["class_to_box"],
        episode_length_s=episode_length_s(budget),
        dilation=args.dilation,
        draw=sample_run_draw(random.Random(args.layout_seed)),
    )
    world["env"] = env
    client = WorkerClient(args.worker)
    world["closers"].append(client.close)
    world["worker_id"] = client.worker_id
    env.perception_backend = RealPerception(env, client, env.cfg.num_bricks)
    rng = random.Random(args.run_seed)

    def factory(i, bl, episode_draw):
        return HeapOracle(
            bl,
            env.parts,
            env.destination_bins[i],
            episode_draw,
            rng=rng,
            cycle_budget=budget,
            boxes=env.desk_box_poses[i],
            tray=True,
            tray_lean_deg=env.tray_lean_deg,
        )

    world["rig"] = OracleRig(
        env, env.cfg.bin_layout, oracle.OracleNoise(), rng, oracle_factory=factory
    )
    server = policy_client.ServerPolicy(args.server, recv_timeout_ms=POLICY_TIMEOUT_MS)
    world["closers"].append(server.close)
    world["policy"] = TimedPolicy(server, env.ticks, env.trip)


def default_wall_minutes(budget: int, dilation: float) -> float:
    """The simulated protocol's cap (``mj_eval_heap.py``), stretched by the dilation."""
    from maker_arm_lab.core.loop import CYCLE_STEP_BUDGET

    return budget * CYCLE_STEP_BUDGET * 0.15 / 60.0 / dilation


def real_block(args, world: dict) -> dict:
    from maker_arm_lab.real.pacing import tick_summary

    env, arm = world["env"], world["arm"]
    return {
        "graded": False,
        "dilation": args.dilation,
        "heap_k": args.heap_k,
        "slots": env.cfg.num_bricks,
        "sheet": world["sheet"],
        "rig": asdict(world["config"]),
        "worker_id": world.get("worker_id"),
        "ticks": tick_summary(env.ticks.ticks),
        "telemetry": arm.telemetry(),
        "overhead_coverage": env.overhead_coverage,
        "k_drift": getattr(world.get("overhead"), "k_drift", None),
    }


def hold(world: dict) -> None:
    """Hold the arm unless a watchdog already did."""
    env, arm = world.get("env"), world.get("arm")
    if arm is None or (env is not None and env.holding):
        return
    try:
        arm.hold()
    except Exception:  # noqa: BLE001 -- nothing more can be done from here
        traceback.print_exc()


def release(arm, mode: str, stdin=None) -> None:
    """Disable the motors: at once, or on the typed word RELEASE or the end of input. A Ctrl-C
    at the prompt keeps the arm held and asks again."""
    if arm is None:
        return
    if mode == "prompt":
        stdin = stdin if stdin is not None else sys.stdin
        print(
            "The arm holds. Support it, then type RELEASE to disable the motors.",
            file=sys.stderr,
            flush=True,
        )
        while True:
            try:
                line = stdin.readline()
            except KeyboardInterrupt:
                print("Ctrl-C: the arm still holds. Type RELEASE.", file=sys.stderr, flush=True)
                continue
            if not line or line.strip() == "RELEASE":
                break
    try:
        arm.stop()
    except Exception:  # noqa: BLE001 -- the token is already out
        traceback.print_exc()


def main(argv=None) -> int:
    """Run one episode. Ctrl-C ends the episode (``interrupted``); from then on, through the
    closers, the token and the release prompt, a Ctrl-C only says the arm still holds: an
    escaping ``KeyboardInterrupt`` would skip the token and ``os._exit``, and dropping the SDK
    arm at interpreter exit disables the motors without the typed ``RELEASE``."""
    args = parse_args(argv)
    world: dict = {}
    episode = [True]

    def on_sigint(signum, frame):
        if episode[0]:
            episode[0] = False
            raise KeyboardInterrupt
        note = "the arm still holds; type RELEASE to disable the motors"
        # os.write, not print: a SIGINT inside a sys.stderr write (a traceback in a closer,
        # the prompt) would re-enter its buffer and raise RuntimeError past os._exit.
        text = f"Ctrl-C: {note if world.get('arm') else 'the run is closing'}.\n"
        os.write(2, text.encode())

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        return _run(args, world, episode)
    finally:
        signal.signal(signal.SIGINT, previous)


def _run(args, world: dict, episode: list) -> int:
    code, token = 1, None
    try:
        try:
            build_world(args, world)
            from maker_arm_lab.core import loop, perception

            wall = args.max_wall_minutes or default_wall_minutes(world["budget"], args.dilation)
            payload = loop.rollout(
                world["env"],
                world["rig"],
                world["policy"],
                run_seed=args.run_seed,
                layout_seed=args.layout_seed,
                permutation=(0, 1, 2, 3),
                cycle_budget=world["budget"],
                max_wall_minutes=wall,
                perceived=True,
                noise_scales=perception.parse_scales("all=0"),
                **SERVED_SWITCHES,
            )
            payload["real"] = real_block(args, world)
            token, code = "EVAL_OK " + json.dumps(payload), 0
        finally:
            episode[0] = False
    # Each branch sets code = 1: a Ctrl-C can land after the EVAL_OK assignment, before the
    # inner finally, and the exit code must still agree with the token.
    except EvalFailure as err:
        token, code = "EVAL_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail}), 1
    except KeyboardInterrupt:
        token = "EVAL_FAIL " + json.dumps({"reason": "interrupted", "detail": "Ctrl-C"})
        code = 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "EVAL_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    if code != 0:
        hold(world)
    for close in reversed(world.get("closers", [])):
        try:
            close()
        except Exception:  # noqa: BLE001 -- closing must not cost the token
            traceback.print_exc()
    if code == 0 and args.out:
        try:
            out = Path(args.out).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(json.loads(token[len("EVAL_OK ") :]), indent=2))
        except Exception as err:  # noqa: BLE001 -- the episode ran; still one token
            token, code = "EVAL_FAIL " + json.dumps({"reason": "write_out", "detail": str(err)}), 1
    print(token, flush=True)
    release(world.get("arm"), args.release)
    return code


if __name__ == "__main__":
    os._exit(main())

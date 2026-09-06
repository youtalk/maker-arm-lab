"""Sweep the arm's FK for a home pose (spec section 3.3): tool axis within 30 deg of world
-z, grasp_center 0.15-0.25 m above the desk, x in [0.15, 0.30] over the scatter band,
|y| < 0.05, and every arm joint >= 0.30 rad from its stops. Picks the candidate with the
largest minimum joint margin. Also prints where the tool points at all-zero joints so the
mount yaw (urdf_prep.MOUNT_RPY) can be corrected if the arm faces away from +x.

--z_range's default is (0.15, 0.25), not the spec's literal "about 0.25-0.30 m above the
desk" (world z is height above the desk: DESK_POS z -0.02 plus half of DESK_SIZE 0.04 puts
the desk top at world z = 0.0 exactly). A from-scratch numpy FK of the vendored URDF found
the brief's original (0.20, 0.30) box geometrically unreachable while holding the two
functional constraints fixed: with max_tool_deg=30 and margin=0.30, the achievable ceiling is
z = 0.188 m, not 0.20 m -- and neither escalation rung (a finer step, a wider x_range) can
raise that ceiling, since it is the arm's reachable envelope that binds, not the sampling
density. The height target itself is a comfort number, not a functional one: at z around
0.19 the tool still clears the shelf (top at 0.15) and sits about 0.18 m above the bricks. The
two constraints that ARE functional are left untouched: max_tool_deg=30 is what makes every
downstream oracle motion start from a sane attitude (this script's own reset check enforces
it independently), and margin=0.30 is what keeps home away from the stops the oracle watchdog
trips on. Relaxing either to hit a cosmetic height would trade a real invariant for one.

Run: $ISAACLAB_PATH/isaaclab.sh -p scripts/find_home_pose.py --headless
"""

import argparse
import itertools
import json
import math
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--step_deg", type=float, default=10.0)
parser.add_argument("--max_tool_deg", type=float, default=30.0)
parser.add_argument("--z_range", type=float, nargs=2, default=(0.15, 0.25))
parser.add_argument("--x_range", type=float, nargs=2, default=(0.15, 0.30))
parser.add_argument("--y_abs_max", type=float, default=0.05)
parser.add_argument("--margin", type=float, default=0.30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg

from maker_arm_lab.assets import urdf_prep
from maker_arm_lab.robots import frames
from maker_arm_lab.robots.maker_arm import MAKER_ARM_USD_PATH


def fk(robot, sim, q: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Set the joint state, advance one physics step with gravity off, read grasp_center."""
    pos = torch.tensor([q + [0.0, 0.0]], device=robot.device)
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, vel)
    robot.set_joint_position_target(pos)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())
    tool = robot.body_names.index(frames.TOOL_BODY)
    return robot.data.body_pos_w[0, tool], robot.data.body_quat_w[0, tool]


def run() -> bool:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, device=args.device, gravity=(0.0, 0.0, 0.0))
    )
    cfg = ArticulationCfg(
        prim_path="/World/MakerArm",
        spawn=sim_utils.UsdFileCfg(usd_path=str(MAKER_ARM_USD_PATH)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1e4, damping=1e2)},
    )
    robot = Articulation(cfg)
    sim.reset()
    joint_order = [robot.joint_names.index(n) for n in urdf_prep.ARM_JOINT_NAMES]
    if joint_order != list(range(6)):
        print(f"JOINT_ORDER {robot.joint_names}", flush=True)
        raise RuntimeError("arm joints are not the first six in articulation order")
    tool_axis = torch.tensor(frames.TOOL_AXIS_BODY, device=robot.device)

    p0, _ = fk(robot, sim, [0.0] * 6)
    print(f"FORWARD_XY {p0[0]:.4f} {p0[1]:.4f}", flush=True)

    limits = urdf_prep.arm_joint_limits()
    step = math.radians(args.step_deg)

    def grid(i):
        lo, hi = limits[i]
        n = int((hi - lo - 2 * args.margin) / step) + 1
        return [lo + args.margin + k * step for k in range(max(n, 1))]

    best, best_margin = None, -1.0
    for q2, q3, q4, q5 in itertools.product(grid(1), grid(2), grid(3), grid(4)):
        q = [0.0, q2, q3, q4, q5, 0.0]
        pos, quat = fk(robot, sim, q)
        axis = math_utils.quat_apply(quat, tool_axis)
        tool_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(-axis[2])))))
        x, y, z = (float(v) for v in pos)
        if tool_deg > args.max_tool_deg or not (args.z_range[0] <= z <= args.z_range[1]):
            continue
        if not (args.x_range[0] <= x <= args.x_range[1]) or abs(y) > args.y_abs_max:
            continue
        margin = min(min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, limits))
        cand = {"q": q, "xyz": [x, y, z], "tool_deg": tool_deg, "margin": margin}
        print("CANDIDATE " + json.dumps(cand), flush=True)
        if margin > best_margin:
            best, best_margin = cand, margin
    if best is None:
        print("HOME_POSE_NONE", flush=True)
        return False
    home = dict(zip(urdf_prep.ARM_JOINT_NAMES, best["q"]))
    print("HOME_POSE " + json.dumps({"joints": home, **best}), flush=True)
    return True


try:
    ok = run()
except Exception:  # noqa: BLE001
    traceback.print_exc()
    ok = False
os._exit(0 if ok else 1)

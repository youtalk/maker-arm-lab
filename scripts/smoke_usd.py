"""Load the converted Maker Arm USD, simulate N steps, sanity-check names, counts, and the Z-up base.

Run with:  $ISAACLAB_PATH/isaaclab.sh -p scripts/smoke_usd.py
Exits 0 on success, 1 on failure.

Intended to run as its own subprocess: it ends with os._exit(), which kills
the host interpreter outright and would be a surprise if this module were
ever imported or exec'd in-process instead.
"""

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=Path, default=Path("assets/usd/maker_arm.usd"))
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--max_drift", type=float, default=float("inf"))
parser.add_argument("--stiffness", type=float, default=100.0)
parser.add_argument("--damping", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# headless is controlled by the standard --headless flag; GUI mode (no flag)
# is used for the Task 7 visual inspection.

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab.sim (and friends) are only importable once AppLauncher has launched
# Kit; the deferral is deliberate, as in convert_urdf_to_usd.py.
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def run_smoke_test() -> bool:
    """Load the asset, step the sim, and report whether the state is sane."""
    # device: AppLauncher's --device picks the Kit app's GPU but never reaches
    # SimulationCfg, which defaults to "cuda:0" regardless. Pass it through, as
    # the four env entry points do, so the flag means one thing everywhere.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))

    cfg = ArticulationCfg(
        prim_path="/World/MakerArm",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(args.usd.resolve()),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=args.stiffness, damping=args.damping
            )
        },
    )
    robot = Articulation(cfg)

    sim.reset()
    initial_qpos = robot.data.joint_pos.clone()
    max_drift = 0.0
    for _ in range(args.steps):
        sim.step()
        robot.update(sim.get_physics_dt())
        max_drift = max(max_drift, (robot.data.joint_pos - initial_qpos).abs().max().item())

    qpos = robot.data.joint_pos
    names = robot.data.joint_names
    print("JOINTS:", list(names), flush=True)
    print(f"MAX_DRIFT: {max_drift:.6f}", flush=True)

    from maker_arm_lab.assets import urdf_prep

    expected_joints = set(urdf_prep.ARM_JOINT_NAMES) | set(urdf_prep.GRIPPER_JOINT_NAMES)
    names_ok = set(names) == expected_joints
    bodies = list(robot.data.body_names)
    print("BODIES:", bodies, flush=True)
    expected_bodies = {
        "mount_link",
        "base_link",
        "link_002",
        "link_003",
        "link_004",
        "link_005",
        "link_006",
        "link_007",
        "gripper_left",
        "gripper_right",
        "grasp_center",
    }
    bodies_ok = set(bodies) == expected_bodies
    # Z-up check: base_link's +y must now point along world +z (MOUNT_RPY = +90 deg about x).
    import isaaclab.utils.math as math_utils

    base = bodies.index("base_link")
    base_y = math_utils.quat_apply(
        robot.data.body_quat_w[0, base], torch.tensor([0.0, 1.0, 0.0], device=robot.device)
    )
    zup_ok = float(base_y[2]) > 0.999
    print(f"BASE_Y_IN_WORLD: {base_y.tolist()}", flush=True)
    tool = bodies.index("grasp_center")
    print("GRASP_CENTER_W:", robot.data.body_pos_w[0, tool].tolist(), flush=True)
    return (
        names_ok
        and bodies_ok
        and zup_ok
        and bool(torch.isfinite(qpos).all())
        and max_drift <= args.max_drift
    )


# A spawn error (e.g. a missing USD file) raises before reaching the ok/fail
# print below. An uncaught exception would normally exit the interpreter
# with status 1, but Kit's native teardown during interpreter shutdown is
# not guaranteed to preserve that code (see the os._exit() note below).
# Catching everything ourselves keeps the failure entirely inside our own
# os._exit() path, which sets the exit code directly and does not depend on
# interpreter-shutdown behavior.
try:
    ok = run_smoke_test()
except Exception:  # noqa: BLE001 -- any failure here must become SMOKE_FAIL, not a crash.
    traceback.print_exc()  # stderr is line-buffered, so no explicit flush is needed here.
    ok = False

print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)

# simulation_app.close() is not used here: it tears down Kit through a
# native call, and earlier work in this milestone reported that call being
# unreliable as the last thing a script does -- either hanging indefinitely
# on native teardown, or letting Kit's shutdown override the process exit
# code to 0 regardless of `ok`. That report was not independently
# reproduced in this script; os._exit() is used defensively regardless,
# since it sets the exit code directly and correctly no matter the root
# cause, without waiting on native teardown.
os._exit(0 if ok else 1)

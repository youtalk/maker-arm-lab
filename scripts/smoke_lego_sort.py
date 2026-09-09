"""Headless LegoSort env smoke: reset + random steps, finite-state check.

Run:  $ISAACLAB_PATH/isaaclab.sh -p scripts/smoke_lego_sort.py --headless --task MakerArm-LegoSort-v0

Intended to run as its own subprocess: it ends with os._exit(), which kills
the host interpreter outright (see smoke_usd.py for the rationale).
"""

import argparse
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="MakerArm-LegoSort-v0")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--num_bricks",
    type=int,
    default=None,
    help="Override LegoSortEnvCfg.num_bricks (default: the registered cfg's own default, 6). "
    "Rebuilds the cfg through its constructor -- num_bricks only takes effect via "
    "__post_init__, which runs once, at construction -- rather than mutating the cfg "
    "gym.make returned from parse_env_cfg() after the fact, which would silently desync "
    "it from the already-built scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab (and friends) are only importable once AppLauncher has launched
# Kit; the deferral is deliberate, as in smoke_usd.py.
import gymnasium as gym
import torch
from isaaclab.sensors import Camera
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import maker_arm_lab.tasks  # noqa: F401  -- registers MakerArm-* envs

# The contract the -Vision- ids promise (design spec section 4.2). Asserted
# rather than printed: a check that only prints what it found cannot fail, and
# the failure mode this script exists to catch is a camera that silently stops
# producing pixels.
EXPECTED_IMAGE_TERMS = ("wrist_rgb", "overhead_rgb")
EXPECTED_IMAGE_SHAPE = (224, 224, 3)
# Per-env standard deviation floor, on the 0-255 scale. An all-black or
# otherwise constant frame has std exactly 0.0; real frames from this scene
# measure two orders of magnitude above this floor, so 1.0 is a wide margin
# around "the renderer produced nothing".
MIN_IMAGE_STD = 1.0


def check_observations(obs: dict, num_envs: int, camera_names: list[str]) -> bool:
    """Validate every observation group. Returns True iff all checks pass.

    Finiteness alone is close to worthless on images: `torch.isfinite` returns
    all-True for a `uint8` tensor -- both raw and cast, it does not raise on
    integer dtypes -- so every possible uint8 image passes it, including an
    all-black one. The load-bearing checks here are therefore the structural
    ones (the group and both terms exist, with the right shape and dtype) plus
    the per-env variance floor.
    """
    ok = True
    for group, terms in obs.items():
        if not isinstance(terms, dict):
            # concatenate_terms=True groups arrive as a single tensor.
            terms = {group: terms}
        print(
            f"OBS_GROUP {group}:",
            [(name, tuple(t.shape), str(t.dtype)) for name, t in terms.items()],
            flush=True,
        )
        for name, tensor in terms.items():
            if not bool(torch.isfinite(tensor.float()).all()):
                print(f"CHECK_FAIL nonfinite: {group}/{name}", flush=True)
                ok = False

    if "policy" not in obs:
        print("CHECK_FAIL missing group: policy", flush=True)
        ok = False

    # Cameras in the scene must reach the observations as live pixels.
    if not camera_names:
        if "images" in obs:
            print("CHECK_FAIL unexpected 'images' group with no camera sensors", flush=True)
            ok = False
        return ok

    print(f"CAMERAS: {camera_names}", flush=True)
    images = obs.get("images")
    if not isinstance(images, dict):
        print("CHECK_FAIL missing group: images (scene has camera sensors)", flush=True)
        return False
    for term in EXPECTED_IMAGE_TERMS:
        if term not in images:
            print(f"CHECK_FAIL missing term: images/{term}", flush=True)
            ok = False
            continue
        tensor = images[term]
        expected = (num_envs, *EXPECTED_IMAGE_SHAPE)
        if tuple(tensor.shape) != expected:
            print(
                f"CHECK_FAIL shape: images/{term} is {tuple(tensor.shape)}, expected {expected}",
                flush=True,
            )
            ok = False
        if tensor.dtype is not torch.uint8:
            print(
                f"CHECK_FAIL dtype: images/{term} is {tensor.dtype}, expected torch.uint8",
                flush=True,
            )
            ok = False
        # Per env, not pooled: a single live camera would otherwise mask a
        # black one in another env.
        stds = tensor.reshape(tensor.shape[0], -1).float().std(dim=1)
        print(f"IMAGE_STD {term}: {[round(float(s), 3) for s in stds]}", flush=True)
        if not bool((stds > MIN_IMAGE_STD).all()):
            print(
                f"CHECK_FAIL blank frame: images/{term} std {[float(s) for s in stds]} "
                f"is not all > {MIN_IMAGE_STD}",
                flush=True,
            )
            ok = False
    return ok


def run() -> bool:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.num_bricks is not None:
        # Rebuild through the constructor (see --num_bricks help above) instead of
        # `env_cfg.num_bricks = args.num_bricks`, which parse_env_cfg's returned cfg
        # would accept silently but never apply -- __post_init__ already ran.
        # The rebuild discards what parse_env_cfg applied, so --device and
        # --num_envs are re-applied here; hard-coding "cuda:0" here (as this
        # branch used to) made --device behave differently with and without
        # --num_bricks.
        env_cfg = type(env_cfg)(num_bricks=args.num_bricks)
        env_cfg.sim.device = args.device
        env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    import isaaclab.utils.math as math_utils

    from maker_arm_lab.robots import frames

    robot = env.scene["robot"]
    tool = robot.body_names.index(frames.TOOL_BODY)
    tool_quat = robot.data.body_quat_w[:, tool]
    # quat_apply's final `.view(vec.shape)` uses vec's shape as given, not its broadcast
    # result -- an unexpanded (3,) vec against a batched (num_envs, 4) quat therefore raises
    # a view-size mismatch once num_envs > 1 rather than silently broadcasting. Expand first.
    tool_axis = torch.tensor(frames.TOOL_AXIS_BODY, device=env.device).expand(
        tool_quat.shape[0], -1
    )
    axis = math_utils.quat_apply(tool_quat, tool_axis)
    tool_deg = torch.rad2deg(torch.acos((-axis[:, 2]).clamp(-1.0, 1.0)))
    print(f"TOOL_AXIS_DEG {tool_deg.tolist()}", flush=True)
    if bool((tool_deg > 30.0).any()):
        print("CHECK_FAIL tool axis more than 30 deg off world -z at home", flush=True)
        return False
    clamp_term = env.action_manager.get_term("arm_action")
    print(f"CLAMPED_AT_RESET {clamp_term.clamped.tolist()}", flush=True)

    action_dim = env.action_manager.total_action_dim
    for _ in range(args.steps):
        actions = 0.2 * (2.0 * torch.rand(args.num_envs, action_dim, device=env.device) - 1.0)
        obs, _, _, _, _ = env.step(actions)
    # What the scene actually carries decides what the observations must
    # contain -- keying off the task id string would not survive a rename.
    camera_names = sorted(n for n, s in env.scene.sensors.items() if isinstance(s, Camera))
    ok = check_observations(obs, args.num_envs, camera_names)
    env.close()
    return ok


try:
    ok = run()
except Exception:  # noqa: BLE001 -- any failure must become SMOKE_FAIL, not a crash.
    traceback.print_exc()
    ok = False

print("SMOKE_OK" if ok else "SMOKE_FAIL", flush=True)
os._exit(0 if ok else 1)

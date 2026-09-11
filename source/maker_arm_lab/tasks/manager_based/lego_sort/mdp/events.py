"""Reset event terms for LegoSort.

``randomize_brick_poses`` adapts ``randomize_object_pose`` /
``sample_object_poses`` from isaaclab_tasks'
``manager_based/manipulation/stack/mdp/franka_stack_events.py`` (Isaac Lab
v2.3.2, BSD-3-Clause): same rejection-sampling structure -- sample x/y/z/yaw
per asset from ``pose_range``, resample while any pair is closer than
``min_separation``, then ``write_root_pose_to_sim`` / ``write_root_velocity_
to_sim`` with zero velocity, offset by ``env.scene.env_origins`` -- but keyed
by a flat list of scene-entity names (``asset_names``) instead of
``SceneEntityCfg`` objects, since LegoSort's brick count is config-dependent
(``num_bricks``) rather than a fixed set of named cubes.

``validate_brick_count`` is a ``mode="startup"`` guard: it fires once, right
after ``ManagerBasedRLEnv`` builds the ``InteractiveScene`` from the cfg, and
catches the case where ``cfg.num_bricks`` was mutated *after*
``LegoSortEnvCfg.__post_init__`` already built the scene from the old value
(``__post_init__`` runs exactly once, at construction) -- e.g. ``cfg =
parse_env_cfg(...); cfg.num_bricks = 8``, which silently desyncs ``cfg`` from
the live scene with no error otherwise.
"""

import random

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv

from maker_arm_lab.core import layout

# The rejection sampler lives in core.layout (isaaclab-free) so the MuJoCo backend scatters
# bricks with the same code; kept under its old name here for the call below.
_sample_brick_poses = layout.sample_brick_poses


def randomize_brick_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    min_separation: float,
    asset_names: list[str],
    max_sample_tries: int = 5000,
    max_radius: float | None = None,
) -> None:
    """Scatter bricks on reset: per-env rejection sampling, min_separation apart, zero velocity."""
    if env_ids is None:
        return

    # Randomize poses in each environment independently.
    for cur_env in env_ids.tolist():
        pose_list = _sample_brick_poses(
            len(asset_names), min_separation, pose_range, max_sample_tries, max_radius
        )

        for name, pose in zip(asset_names, pose_list):
            asset = env.scene[name]

            # Write pose to simulation.
            pose_tensor = torch.tensor([pose], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            env_id_tensor = torch.tensor([cur_env], device=env.device)
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=env_id_tensor
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=env_id_tensor
            )


def validate_brick_count(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Startup-mode guard: ``cfg.num_bricks`` must match the ``brick_*`` entities
    actually built into the scene. See the module docstring for why this can drift.

    Raises rather than letting ``brick_color_ids(env.cfg.num_bricks)`` and
    ``brick_poses_w(env)`` silently disagree on brick count downstream (Task 8's
    grading path pairs the two).
    """
    # env_ids is always None here: "startup" mode is applied once, un-indexed,
    # right after the scene is built (see EventManager.apply's startup branch).
    del env_ids

    expected = env.cfg.num_bricks
    brick_names = [name for name in env.scene.keys() if name.startswith("brick_")]  # noqa: SIM118
    actual = len(brick_names)
    if actual != expected:
        raise ValueError(
            f"LegoSortEnvCfg.num_bricks={expected} but the scene has {actual} brick_* "
            "entities. num_bricks only takes effect through the LegoSortEnvCfg("
            "num_bricks=...) constructor -- __post_init__ builds the brick/bin scene "
            "entities once, at construction time. Assigning cfg.num_bricks after "
            "construction (e.g. after parse_env_cfg(...)) does not rebuild the scene."
        )


def randomize_desk_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    color_range: dict[str, tuple[float, float]],
) -> None:
    """Per-env desk color: rewrite the resetting envs' PreviewSurface diffuseColor.

    Deliberately not isaaclab's `randomize_visual_color`: its replicator graph
    recolors every env at once ("This triggers the nodes for all the
    environments", events.py:1682), which would change the desk mid-episode for
    the other parallel envs. A direct per-prim USD attribute write touches only
    the envs being reset. Requires replicate_physics=False (already the case)
    so each env has its own Desk prim and material.
    """
    if env_ids is None:
        return
    from pxr import Gf, UsdShade

    stage = env.scene.stage
    for cur_env in env_ids.tolist():
        # The material is bound on the mesh prim `sim_utils.spawn_from_...`
        # creates under the Xform (`{prim_path}/geometry/mesh`, per
        # isaaclab.sim.spawners.shapes.shapes.py), not on the Desk Xform root
        # -- `MaterialBindingAPI.ComputeBoundMaterial` only walks up ancestors
        # for an inherited binding, never down into children, so pointing it
        # at the Xform finds nothing. Found on the real stage (Task 7 GPU
        # smoke): the Xform lookup raised "no material bound" every time.
        mesh_path = f"/World/envs/env_{cur_env}/Desk/geometry/mesh"
        prim = stage.GetPrimAtPath(mesh_path)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise RuntimeError(f"no material bound on {mesh_path}")
        # ComputeSurfaceSource resolves the material's surface output to its
        # driving shader; indexing the material prim's first child (the
        # original approach here) happened to match PreviewSurfaceCfg's own
        # single-shader layout, but is not how USD material graphs are
        # supposed to be walked. Deviation recorded per Task 4 Step 2's note.
        # Returns a (shader, sourceName, sourceType) tuple -- indexed, not
        # unpacked, per that note.
        shader = material.ComputeSurfaceSource()[0]
        if not shader:
            raise RuntimeError(f"no surface shader source on the material bound to {mesh_path}")
        color = Gf.Vec3f(
            random.uniform(*color_range["r"]),
            random.uniform(*color_range["g"]),
            random.uniform(*color_range["b"]),
        )
        shader.GetInput("diffuseColor").Set(color)

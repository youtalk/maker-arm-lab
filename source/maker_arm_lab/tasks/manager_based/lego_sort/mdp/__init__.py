"""MDP terms for the LegoSort environment."""

from isaaclab.envs.mdp import *  # noqa: F403 -- standard isaaclab_tasks mdp re-export idiom

from . import recorders  # noqa: F401
from .events import randomize_brick_poses, randomize_desk_color, validate_brick_count  # noqa: F401
from .observations import brick_color_ids, brick_poses_w  # noqa: F401
from .terminations import all_bricks_stowed, per_brick_stowed  # noqa: F401

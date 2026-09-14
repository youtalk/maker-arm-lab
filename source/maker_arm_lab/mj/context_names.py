"""Names and widths of the two context channels, shared by the env and the recorder."""

# Context channels (MS4 levers 1 and 3). Both are recorded in every shard; `context` picks
# which one the env SERVES in obs_buf["context"], which is what the eval loop puts on the
# wire. "camera" is the overhead camera's pose in the base frame, which the real rig will
# supply from its calibration. "privileged" (MS4 c2 design 5.2, replacing the c1 quaternion
# channel) is, per brick, the position and the first two columns of its rotation matrix, then
# a one-hot target brick under core.target's rule and a stowed mask. It exists only for the
# privileged-state ceiling, never for a deployable policy.
CONTEXT_CAMERA_NAMES = tuple(f"overhead_cam_{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw"))
CONTEXT_MAX_BRICKS = 6
BRICK_POSE_NAMES = ("x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21")
CONTEXT_PRIVILEGED_NAMES = tuple(
    [f"brick_{b}_{k}" for b in range(CONTEXT_MAX_BRICKS) for k in BRICK_POSE_NAMES]
    + [f"target_{b}" for b in range(CONTEXT_MAX_BRICKS)]
    + [f"stowed_{b}" for b in range(CONTEXT_MAX_BRICKS)]
)
CONTEXT_DIMS = {
    "none": 0,
    "camera": len(CONTEXT_CAMERA_NAMES),
    "privileged": len(CONTEXT_PRIVILEGED_NAMES),
}

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

# MS6 (design section 9): the destination box pose in the base frame, recorded beside the
# camera channel in every heap shard. "camera_box" serves both as one 9-wide vector.
CONTEXT_BOX_NAMES = ("box_x", "box_y", "box_yaw")
CONTEXT_DIMS["camera_box"] = len(CONTEXT_CAMERA_NAMES) + len(CONTEXT_BOX_NAMES)

# MS6 heap policy lever (2026-09-21): the part the loop aims at, as a perception module would
# report it. Its centre in the base frame when the cycle opened, the closing-axis heading of
# the grasp the teacher ranks on it as (cos, sin), the part size along that axis and half its
# vertical extent. All zeros between cycles. The rig stashes it on the envs it drives and the
# heap loop writes it at selection (core.heap_oracle.target_context builds the row on both
# sides). "camera_box_target" serves camera, box and target as one 16-wide vector.
CONTEXT_TARGET_NAMES = (
    "target_x",
    "target_y",
    "target_z",
    "target_cos",
    "target_sin",
    "target_width",
    "target_half_height",
)
CONTEXT_DIMS["camera_box_target"] = CONTEXT_DIMS["camera_box"] + len(CONTEXT_TARGET_NAMES)

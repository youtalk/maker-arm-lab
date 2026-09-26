"""Read the vendored Maker Arm URDF and derive the Z-up copy the simulator loads.

The vendored ``robot.urdf`` keeps the CAD's Y-up base frame (upstream README) and is never
edited. Isaac Lab's IK action term and the ported ``AttitudeHold`` work in the articulation
ROOT frame, so the root must be Z-up at identity: this module writes a derived copy with one
extra fixed link (``mount_link`` -> ``base_link``, rotated ``MOUNT_RPY``) into the build
directory. Joint coordinates are unchanged, so the joint-coordinate contract with
maker-arm-rs (spec section 4.1) holds on the derived file too.
"""

import copy
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
VENDORED_URDF = _REPO_ROOT / "assets" / "vendor" / "maker_arm_description" / "robot.urdf"

ARM_JOINT_NAMES: tuple[str, ...] = (
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
    "link_006_joint",
    "link_007_joint",
)
GRIPPER_JOINT_NAMES: tuple[str, str] = ("gripper_left_joint", "gripper_right_joint")

# Rotation taking the CAD's Y-up base to world Z-up: +90 deg about x maps base +y to world +z.
# The yaw term (third element) is a workspace-facing correction: scripts/find_home_pose.py
# found the arm at all-zero joints facing -x (FORWARD_XY -0.2281 -0.0000, matching Task B6's
# GRASP_CENTER_W x < 0 on the unrotated USD), while the scatter band and bins sit at +x. A
# further +180 deg about the new z turns the arm to face +x.
MOUNT_RPY: tuple[float, float, float] = (math.pi / 2, 0.0, math.pi)


def _joints(root: ET.Element) -> dict[str, ET.Element]:
    return {j.get("name"): j for j in root.findall("joint")}


def arm_joint_limits(urdf: Path = VENDORED_URDF) -> list[tuple[float, float]]:
    """(lower, upper) in radians for the six arm joints, kinematic order."""
    joints = _joints(ET.parse(urdf).getroot())
    out = []
    for name in ARM_JOINT_NAMES:
        limit = joints[name].find("limit")
        out.append((float(limit.get("lower")), float(limit.get("upper"))))
    return out


def gripper_travel(urdf: Path = VENDORED_URDF) -> float:
    """Per-jaw prismatic travel in metres (upper limit of the driver joint)."""
    joints = _joints(ET.parse(urdf).getroot())
    return float(joints[GRIPPER_JOINT_NAMES[0]].find("limit").get("upper"))


def gripper_velocity_limit(urdf: Path = VENDORED_URDF) -> float:
    """Per-jaw prismatic velocity limit in m/s (the driver joint's ``<limit velocity>``)."""
    joints = _joints(ET.parse(urdf).getroot())
    return float(joints[GRIPPER_JOINT_NAMES[0]].find("limit").get("velocity"))


def write_zup_urdf(src: Path, dst: Path, mount_rpy: tuple[float, float, float] = MOUNT_RPY) -> Path:
    """Write ``dst``: ``src`` plus a fixed ``mount_link`` root, with absolute mesh paths."""
    tree = ET.parse(src)
    root = tree.getroot()
    if any(link.get("name") == "mount_link" for link in root.findall("link")):
        raise ValueError(f"{src} already has a mount_link; refusing to stack two")

    mesh_dir = src.resolve().parent
    for mesh in root.iter("mesh"):
        mesh.set("filename", str((mesh_dir / mesh.get("filename")).resolve()))

    mount_link = ET.Element("link", {"name": "mount_link"})
    inertial = ET.SubElement(mount_link, "inertial")
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-7", "ixy": "0", "ixz": "0", "iyy": "1e-7", "iyz": "0", "izz": "1e-7"},
    )
    mount_joint = ET.Element("joint", {"name": "mount_joint", "type": "fixed"})
    ET.SubElement(mount_joint, "parent", {"link": "mount_link"})
    ET.SubElement(mount_joint, "child", {"link": "base_link"})
    ET.SubElement(
        mount_joint, "origin", {"xyz": "0 0 0", "rpy": " ".join(repr(v) for v in mount_rpy)}
    )
    root.insert(0, copy.deepcopy(mount_joint))
    root.insert(0, mount_link)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="unicode", xml_declaration=True)
    return dst


def chain_links(
    names: Sequence[str], urdf: Path = VENDORED_URDF
) -> list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...] | None]]:
    """The named joints as ``(xyz, rpy, axis)``, ``axis`` None for a fixed joint, in order.

    The chain format ``maker_arm_rs.Kinematics`` and ``maker_arm_rs.Dynamics`` take.
    """
    joints = _joints(ET.parse(urdf).getroot())
    out = []
    for name in names:
        joint = joints[name]
        origin, axis = joint.find("origin"), joint.find("axis")
        out.append(
            (
                tuple(float(v) for v in origin.get("xyz").split()),
                tuple(float(v) for v in origin.get("rpy").split()),
                tuple(float(v) for v in axis.get("xyz").split()) if axis is not None else None,
            )
        )
    return out

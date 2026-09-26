"""The arm's rigid-body model for ``maker_arm_rs.Dynamics`` (tracking design, 2026-09-25,
section 5).

``Dynamics`` takes one body per revolute joint, in the frame after that joint: the mass, the
center of mass, and the inertia tensor about the center of mass. This module reads them from
the vendored URDF (the sim's own model) or from the housing mass model asset, and lumps the
two jaws into link_007 at a fixed opening, because the loop does not model the gripper.

stdlib and ``maker_arm_rs`` only, so it runs unchanged on the Jetson Thor. MA2's fitted
parameter file replaces the URDF here later.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import maker_arm_rs

from maker_arm_lab.assets.urdf_prep import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    MOUNT_RPY,
    VENDORED_URDF,
    chain_links,
    gripper_travel,
)

ARM_LINKS: tuple[str, ...] = (
    "link_002",
    "link_003",
    "link_004",
    "link_005",
    "link_006",
    "link_007",
)
JAW_LINKS: tuple[str, str] = ("gripper_left", "gripper_right")
JAW_FRACTION = 0.5
"""The lumped jaws sit at half their travel (tracking design section 5)."""
GRAVITY = (0.0, 0.0, -9.81)
HOUSING_ASSET = Path(__file__).resolve().parents[3] / "assets" / "housing_mass_model.json"

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
Body = tuple[float, Vec3, Mat3]


def _floats(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(v) for v in (text or default).split()]


def _rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _mv(m, v) -> list[float]:
    return [sum(m[r][k] * v[k] for k in range(3)) for r in range(3)]


def _rotate(r, inertia) -> list[list[float]]:
    """R I R^T."""
    ri = [[sum(r[a][k] * inertia[k][b] for k in range(3)) for b in range(3)] for a in range(3)]
    return [[sum(ri[a][k] * r[b][k] for k in range(3)) for b in range(3)] for a in range(3)]


def _freeze(mass: float, com, inertia) -> Body:
    return (
        float(mass),
        tuple(float(v) for v in com),
        tuple(tuple(float(v) for v in row) for row in inertia),
    )


def lump(parts: Sequence[Body]) -> Body:
    """One body from several in the same frame: total mass, joint center of mass, and the
    inertia about it by the parallel-axis theorem."""
    mass = sum(p[0] for p in parts)
    com = [sum(p[0] * p[1][k] for p in parts) / mass for k in range(3)]
    inertia = [[0.0] * 3 for _ in range(3)]
    for m, c, i in parts:
        d = [c[k] - com[k] for k in range(3)]
        dd = sum(v * v for v in d)
        for a in range(3):
            for b in range(3):
                inertia[a][b] += i[a][b] + m * ((dd if a == b else 0.0) - d[a] * d[b])
    return _freeze(mass, com, inertia)


def urdf_inertials(urdf: Path = VENDORED_URDF) -> dict[str, Body]:
    """Every URDF link with an ``<inertial>``, in its own link frame."""
    out = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        xyz = _floats(origin.get("xyz") if origin is not None else None)
        rpy = _floats(origin.get("rpy") if origin is not None else None)
        a = inertial.find("inertia").attrib
        i0 = [
            [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
            [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
            [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
        ]
        mass = float(inertial.find("mass").get("value"))
        out[link.get("name")] = _freeze(mass, xyz, _rotate(_rpy(*rpy), i0))
    return out


def _jaw_in_link_007(body: Body, joint: ET.Element, opening: float) -> Body:
    """A jaw body moved into link_007's frame: joint origin, then ``opening`` along the axis."""
    origin = joint.find("origin")
    p = _floats(origin.get("xyz"))
    r = _rpy(*_floats(origin.get("rpy")))
    axis = _floats(joint.find("axis").get("xyz"))
    local = [body[1][k] + opening * axis[k] for k in range(3)]
    com = [v + p[k] for k, v in enumerate(_mv(r, local))]
    return _freeze(body[0], com, _rotate(r, body[2]))


def arm_bodies(
    inertials: dict[str, Body], urdf: Path = VENDORED_URDF, jaw_fraction: float = JAW_FRACTION
) -> list[Body]:
    """The six bodies in joint order, with the jaws lumped into link_007 at ``jaw_fraction``
    of their travel. ``inertials`` maps URDF link names to bodies in their own link frames."""
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    opening = jaw_fraction * gripper_travel(urdf)
    jaws = [
        _jaw_in_link_007(inertials[link], joints[name], opening)
        for name, link in zip(GRIPPER_JOINT_NAMES, JAW_LINKS)
    ]
    bodies = [inertials[link] for link in ARM_LINKS[:-1]]
    bodies.append(lump([inertials[ARM_LINKS[-1]], *jaws]))
    return bodies


def urdf_bodies(urdf: Path = VENDORED_URDF, jaw_fraction: float = JAW_FRACTION) -> list[Body]:
    return arm_bodies(urdf_inertials(urdf), urdf, jaw_fraction)


def housing_bodies(
    path: Path = HOUSING_ASSET, urdf: Path = VENDORED_URDF, jaw_fraction: float = JAW_FRACTION
) -> list[Body]:
    """The housing study's mass model (tracking design section 3), jaws lumped as for the URDF."""
    data = json.loads(path.read_text())
    inertials = {
        name: _freeze(b["mass"], b["com"], b["inertia"]) for name, b in data["bodies"].items()
    }
    return arm_bodies(inertials, urdf, jaw_fraction)


def scaled(bodies: Sequence[Body], factor: float) -> list[Body]:
    """Every mass and inertia times ``factor``; the centers of mass stay (the +-30 % sweep)."""
    return [_freeze(m * factor, c, [[v * factor for v in row] for row in i]) for m, c, i in bodies]


def dynamics(
    bodies: Sequence[Body],
    friction: Sequence[tuple[float, float, float]] | None = None,
    urdf: Path = VENDORED_URDF,
) -> maker_arm_rs.Dynamics:
    return maker_arm_rs.Dynamics(
        tuple(MOUNT_RPY),
        chain_links(ARM_JOINT_NAMES, urdf),
        list(bodies),
        None if friction is None else list(friction),
        GRAVITY,
    )

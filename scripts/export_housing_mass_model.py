"""Export the housing study's mass model as per-link inertials (tracking design section 3).

One-time, not part of any run, and not under test: it needs trimesh, which the lab does not
depend on. Run it from the repo root:

    uv run --no-sync --with trimesh python scripts/export_housing_mass_model.py

The rules are the housing study's (~/data/maker-arm-housing/massmodel.py and torques2.py,
2026-09-25). Prints EXPORT_OK or EXPORT_FAIL.
"""

import argparse
import datetime
import json
import os
import re
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from maker_arm_lab.assets.urdf_prep import VENDORED_URDF
from maker_arm_lab.core import dynamics_model as dm

MOTOR = {
    "part_001": "RS00",
    "part_018": "RS00",
    "part_052": "RS00",
    "part_050": "RS00",
    "part_022": "RS00",
    "part_002": "RS02",
    "part_015": "RS02",
}
# RS00 0.310 kg matches torques2.py's default. RS02 is 0.380 kg, not torques2.py's 0.405 kg
# default: torques.py marks 0.405 "placeholder until datasheet check", and the study's later
# scripts (effects.py, tilt.py) both correct it to 0.380 kg once the datasheet was checked.
# effects.py's "baseline PLA 75%" desk pose is the source of the 5.36 / 3.36 N m figures this
# asset is checked against (gen/effects.json: desk_j2 5.3648300169583285, desk_j3
# 3.359615627433939), and it uses 0.380 kg -- 0.405 kg overshoots desk_j2 to 5.385 N m.
MOTOR_KG = {"RS00": 0.310, "RS02": 0.380}
HARDWARE = {"part_027"}  # 6 mm steel guide rod
STEEL = 7850.0  # kg/m^3
PRINTED = 1240.0 * 0.75  # kg/m^3: PLA at 75 % fill
JAW_LOOPS = (("part_047", "gripper_left"), ("part_046", "gripper_right"))
LINKS = (*dm.ARM_LINKS, *dm.JAW_LINKS)


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("EXPORT_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def _rpy(r, p, y):
    return np.array(dm._rpy(r, p, y))


def _part(mesh_path: Path, name: str, rot, pos):
    import trimesh

    mesh = trimesh.load(mesh_path)
    mesh.apply_scale(1e-3)
    volume = abs(mesh.volume)
    kind = "motor" if name in MOTOR else "hardware" if name in HARDWARE else "printed"
    mass = (
        MOTOR_KG[MOTOR[name]]
        if kind == "motor"
        else volume * (STEEL if kind == "hardware" else PRINTED)
    )
    if mesh.is_volume:
        mesh.density = mass / mesh.volume
        com, inertia = mesh.center_mass, mesh.moment_inertia
    else:
        com, inertia = mesh.centroid, np.zeros((3, 3))
    return (mass, tuple(rot @ com + pos), tuple(map(tuple, rot @ inertia @ rot.T))), kind


def export(out: Path) -> dict:
    root = ET.parse(VENDORED_URDF).getroot()
    meshes = VENDORED_URDF.parent / "meshes"
    parts: dict[str, list] = {link: [] for link in LINKS}
    placement = {}
    for link in root.findall("link"):
        name = link.get("name")
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            filename = visual.find("geometry/mesh").get("filename")
            match = re.search(r"(part_\d+)_solid", filename)
            if match is None:
                continue
            rot = _rpy(*[float(v) for v in origin.get("rpy").split()])
            pos = np.array([float(v) for v in origin.get("xyz").split()])
            placement[match.group(1)] = (rot, pos)
            if name in parts:
                path = meshes / Path(filename).name
                parts[name].append(_part(path, match.group(1), rot, pos)[0])
    for loop, link in JAW_LOOPS:
        rot, pos = placement["part_045"]
        parts[link].append(_part(meshes / f"{loop}_solid_{loop[5:]}.stl", loop, rot, pos)[0])
    bodies = {}
    for link, items in parts.items():
        if not items:
            raise RuntimeError(f"no parts found for {link}")
        mass, com, inertia = dm.lump(items)
        bodies[link] = {"mass": mass, "com": list(com), "inertia": [list(r) for r in inertia]}
    pin = re.search(
        r'PIN_SHA="?([0-9a-f]+)', (ROOT / "scripts/vendor_maker_arm_description.sh").read_text()
    )
    data = {
        "provenance": {
            "source": "housing study, ~/data/maker-arm-housing (massmodel.py, effects.py), 2026-09-25",
            "rules": {
                "motor_kg": MOTOR_KG,
                "motor_parts": MOTOR,
                "steel_parts": sorted(HARDWARE),
                "steel_kg_m3": STEEL,
                "printed_kg_m3": PRINTED,
                "jaw_loops": [list(x) for x in JAW_LOOPS],
                "frames": "each body in its own URDF link frame; inertia about the center of mass",
            },
            "urdf_pin": pin.group(1) if pin else "unknown",
            "script": "scripts/export_housing_mass_model.py",
            "generated": datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        },
        "bodies": bodies,
    }
    out.write_text(json.dumps(data, indent=1) + "\n")
    return {"out": str(out), "masses": {k: round(v["mass"], 4) for k, v in bodies.items()}}


def main() -> int:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=dm.HOUSING_ASSET)
    args = parser.parse_args()
    try:
        summary = export(args.out)
    except Exception as exc:  # noqa: BLE001 -- every failure must become a token
        traceback.print_exc()
        print("EXPORT_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1
    print("EXPORT_OK " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())

"""The LDraw family's parts as JSON collision hulls (MS7e design section 6.5, P2).

A development tool. It runs once on Machine A and its output under ``assets/parts/`` is
committed; the runtime reads those files with the standard library (``core.parts``). It needs
coacd, scipy and trimesh (the ``ldraw`` uv group), which the shared venv does not carry, so run
it from a throwaway venv:

    uv venv -p 3.12 ~/data/ldraw/venv
    VIRTUAL_ENV=~/data/ldraw/venv uv pip install coacd numpy scipy trimesh
    ~/data/ldraw/venv/bin/python scripts/ldraw_parts.py --ldraw ~/data/ldraw/ldraw

``--ldraw`` is the unzipped official complete library, kept outside the repo; ``curl -R`` keeps
the server's date on the zip, which the notice records with its sha256:

    cd ~/data/ldraw && curl -RO https://library.ldraw.org/library/updates/complete.zip
    unzip complete.zip  # -> ~/data/ldraw/ldraw

For each part the tool:

* expands the ``.dat`` file recursively into triangles: line types 3 and 4, with the type-1
  sub-file transforms. Winding and BFC statements are ignored, because CoACD remeshes the
  surface. Line types 2 and 5 (edges) are ignored.
* leaves every stud primitive (a sub-file named ``stud*`` or ``stug*``, and all it holds) out
  of the collision mesh. Studs stay visual, as for every other part: the JSON lists the studs
  that stand out of the collision box (base, axis, height, radius), and the scene draws them.
  A disk closes the wall under each such stud, which an open stud leaves as a hole.
* converts LDraw units to meters (1 LDU = 0.4 mm) and turns LDraw's -y (up) into +z. The long
  side of the collision box goes on x; a tie keeps LDraw's x axis. The part is centred on its
  collision box, so ``size`` excludes the studs, as for every other part.
* fills the collision mesh: it voxelizes the surface at ``VOXEL_M`` and fills, slice by
  horizontal slice, every region the walls enclose. The underside of a brick fills; an opening
  that reaches the outside sideways (an arch, a pin hole, the space under a slope) stays open.
  A part after the MS7e ones fills the slices across x and y as well (``HORIZONTAL_FILL``), so
  the bore of a pin that lies along x fills too.
  The LDraw walls are 1.6 mm thick and a gripped jaw sinks 2 to 4 mm into a part
  (``core.grasp.CLOSE_SQUEEZE_M``), so a hollow part would let the jaw through its wall. The
  solid matches every other part of the catalog, which is a solid box.
* splits the filled mesh with CoACD into at most ``--max_hulls`` convex hulls, then scales the
  hulls per axis so that together they span the collision box exactly (the voxel surface lies
  up to half a voxel outside the true one). A solid that fills ``CONVEX_FILL`` of its convex
  hull is convex up to the voxel steps and keeps one hull (``CONVEX_FILL``).

Prints exactly one ``LDRAW_OK {json}`` or ``LDRAW_FAIL {json}`` line on every exit path,
argparse errors included, and exits through ``os._exit``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import traceback

LDU_M = 0.0004
LICENSE_URLS = {
    "CC BY 2.0": "https://creativecommons.org/licenses/by/2.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
}
MAX_HULLS = 4
"""The hull cap. The design's 8 (section 6.5) dropped to 4 by its steps-per-second rule (D8).
On the LDraw heap read (Machine B, run seeds 20288000 + i, 4 heaps of 50 on the "all" split, 25
LDraw parts each), the steps per second of each lane over the same heap without LDraw read
0.78, 0.70, 0.48 and 0.68 at cap 8 (heap 2 again alone: 0.47), and 0.88, 0.94, 0.64 and 0.86
at cap 4. At cap 4 the lone-part table stows 94 of 96 and the heap read stows 87 of 100 LDraw
parts and 96 of 100 others in the correct box (2026-09-29)."""
HULL_VERTICES = 32  # CoACD's per-hull vertex cap; MuJoCo's support function walks them
VOXEL_M = 0.0002  # 0.5 LDU: a 1.6 mm wall is 8 voxels, a 4.8 mm pin hole 24
CONVEX_FILL = 0.97
"""A solid that fills this share of its convex hull gets one hull. CoACD split 3660, a convex
shape (it fills 0.9937 of its hull, the rest is voxel steps on the slope), into a top slab and
a 4.1 mm wedge under it; the wedge passed through a 5 mm box floor at release on 4 of 8 lifts
(lone-part table, 2026-09-29). The other parts fill 0.72 to 0.88 (arches, pin holes) or 1.0."""
DECIMALS = 7  # 0.1 micrometer
FACE_TOL_M = 1e-4  # a stud that passes the collision box by more stands out of it
# LDraw (x, y, z) with -y up -> (x, z, -y): a proper rotation (det +1) that puts up on +z.
LDRAW_TO_BODY = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))

PARTS = {
    "3700": ((0, 1, 2), "box-shaped; the pin hole runs along y and leaves flat faces around it"),
    "3701": ((0, 1, 2), "box-shaped like 3700; x (32 mm) is inside the grasp limit"),
    "3894": ((0, 1, 2), "box-shaped like 3700; x (48 mm) is over the grasp limit, which drops it"),
    "4490": ((0, 1), "z puts one jaw in the arch opening under the thin bridge"),
    "3659": ((0, 1), "z puts one jaw in the arch opening under the thin bridge"),
    "87087": ((0, 2), "y runs through the side stud, which a real jaw meets 1.6 mm early"),
    "11211": ((0, 2), "y runs through the two side studs, which a real jaw meets 1.6 mm early"),
    "3660": ((0, 2), "y puts a jaw on the slope; z closes top on bottom strip, as a slope's z"),
    # MS7f joint family (MS7f design section 7). LDraw lays a pin along x, its rest on the desk.
    "2780": ((0, 1, 2), "x on the flat pin ends; y and z across the collar when they lie flat"),
    "6558": ((0, 1, 2), "as 2780: x on the flat ends, y and z across the collar"),
    "43093": ((0, 1, 2), "as 2780: x on the flat ends, y and z across the collar"),
    "3937": ((0, 1, 2), "box-shaped: a 1x2 brick whose top is the two hinge lugs"),
    "3938": ((0, 2), "a 1x2 plate over the knuckle; y meets only the 1.8 mm rib under it"),
    "14417": ((0, 2), "a 1x2 plate; y puts one jaw on the side ball, which squeezes it out"),
    "14418": ((0, 1, 2), "a 1x2 plate; x puts one jaw on the socket's round end"),
    "4081b": ((1, 2), "a 1x1 plate; x puts one jaw on the side clip's ring, as 14417's y"),
    # MS7f minifig family: each part as LDraw builds it upright, one rigid body.
    "3626b": ((0, 1, 2), "a head: x and y across its round side, z from the top to the neck"),
    "973": ((0, 1, 2), "a torso: y on the flat front and back, x on the tapered sides"),
    "3815c01": ((0, 1, 2), "the hips and legs as one body, a box to within a millimeter"),
    "3901": ((0, 1, 2), "hair: x and y across its round side, z from the crown to the rim"),
    "3624": ((0, 1, 2), "a police hat: z from its flat top to the rim, x and y across it"),
    "3629": ((0, 1, 2), "a cowboy hat: x and y on the brim's edge, z from crown to brim"),
    "3838": ((0, 1, 2), "air tanks: x across the two tanks, y front to back, z end to end"),
    "4349": ((0, 1, 2), "a loudhailer: x end to end, y across the cone, z cone to handle"),
}
"""Part id -> (closing axes, reason). The axes are body axes after this tool's orientation.

The joint table first offered 14417 its y axis and 4081b its x axis (15.2 mm each, one jaw on
the ball or on the clip's ring): the part squeezed out at LIFT on 10 of 11 such grasps, with
no jaw force left at the gate, and the table stowed 4 and 10 of 12 (MS7f lone-part table,
seeds 20303000 + catalog index, 2026-09-30). The other axes of both parts stowed every lift.

3938's underside opens sideways, so its fill leaves a 1 mm top slab, the two knuckle lugs at
the ends and a 1.8 mm rib between them: across y the jaws close on the rib alone (0.5 to 0.7 N
against 1.2 to 1.8 N across x on the table, or nothing). On the joint heap read (run 20303500
+ i, layout 20303550 + i, 2026-09-30) the loop took 3938 as its target in 56 cycles for 18
parts: 113877 of the joint read's 168738 steps, a median of 10 close edges a cycle."""
HORIZONTAL_FILL = ("3700", "3701", "3894", "4490", "3659", "87087", "11211", "3660")
"""The MS7e parts fill their horizontal slices only, so 3700's pin hole stays open. Every later
part fills its slices across all three axes: a pin lies along x, and only its x slices close
its bore (MS7f design section 13)."""


class _TokenEmittingParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        print("LDRAW_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def parse_args(argv=None) -> argparse.Namespace:
    parser = _TokenEmittingParser(description=__doc__)
    parser.add_argument("--ldraw", type=pathlib.Path, required=True, help="the library root")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("assets/parts"))
    parser.add_argument("--parts", default=",".join(PARTS), help="comma-separated part ids")
    parser.add_argument("--max_hulls", type=int, default=MAX_HULLS)
    parser.add_argument("--seed", type=int, default=0, help="CoACD's seed")
    return parser.parse_args(argv)


class Library:
    """Case-insensitive lookup of LDraw files: ``parts/`` first, then ``p/``."""

    def __init__(self, root: pathlib.Path):
        self.files = {}
        for sub in ("parts", "p"):
            for path in (root / sub).rglob("*.dat"):
                key = path.relative_to(root / sub).as_posix().lower()
                self.files.setdefault(key, path)
        self.cache = {}

    def read(self, name: str) -> tuple[pathlib.Path, list[str]]:
        path = self.files.get(name.replace("\\", "/").lower())
        if path is None:
            raise FileNotFoundError(f"LDraw file {name!r} is not in the library")
        if path not in self.cache:
            self.cache[path] = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return path, self.cache[path]


def _header(lines: list[str]) -> dict:
    head = {"title": lines[0][1:].strip() if lines else ""}
    tags = {
        "name": "0 Name:",
        "author": "0 Author:",
        "license": "0 !LICENSE",
        "update": "0 !LDRAW_ORG",
    }
    for line in lines[:20]:
        for key, tag in tags.items():
            if line.startswith(tag):
                head[key] = line[len(tag) :].strip()
    return head


def expand(lib: Library, name: str, rot, off, owner, out: dict) -> None:
    """Append ``name``'s triangles, in the root file's frame, to ``out``.

    ``owner`` is None outside any stud file, "group" inside a ``stug*`` file and the index of
    the stud in ``out["studs"]`` inside a ``stud*`` file. Every triangle goes to
    ``out["visual"]``; a triangle outside any stud also goes to ``out["collision"]`` and one
    inside a stud to that stud's list. ``out["files"]`` gets the header of every file read.
    """
    import numpy as np

    path, lines = lib.read(name)
    out["files"].setdefault(path.name, _header(lines))
    for line in lines:
        tok = line.split()
        if tok and tok[0] == "1":
            v = [float(t) for t in tok[2:14]]
            sub_rot = rot @ np.array(v[3:12]).reshape(3, 3)
            sub_off = rot @ np.array(v[0:3]) + off
            sub = " ".join(tok[14:])
            base = sub.replace("\\", "/").rsplit("/", 1)[-1].lower()
            sub_owner = owner
            if base.startswith("stud") and not isinstance(owner, int):
                sub_owner = len(out["studs"])
                out["studs"].append([])
            elif base.startswith("stug") and owner is None:
                sub_owner = "group"
            expand(lib, sub, sub_rot, sub_off, sub_owner, out)
        elif tok and tok[0] in ("3", "4"):
            n = int(tok[0])
            pts = [
                rot @ np.array([float(t) for t in tok[2 + 3 * k : 5 + 3 * k]]) + off
                for k in range(n)
            ]
            tris = [pts[:3]] if n == 3 else [pts[:3], [pts[0], pts[2], pts[3]]]
            out["visual"].extend(tris)
            if owner is None:
                out["collision"].extend(tris)
            elif isinstance(owner, int):
                out["studs"][owner].extend(tris)


def outside_studs(studs, size) -> list[dict]:
    """The studs that stand out of the collision box, as base, axis, height and radius (m).

    ``studs`` holds each stud's triangles in the body frame. A stud stands on the face it
    passes furthest; one that stays inside the box (an underside tube) is hidden and dropped.
    """
    import numpy as np

    found = []
    for tris in studs:
        pts = np.array(tris).reshape(-1, 3)
        past = np.concatenate([pts - 0.5 * size, -pts - 0.5 * size], axis=1).max(axis=0)
        j = int(np.argmax(past))
        if past[j] < FACE_TOL_M:
            continue
        k, s = j % 3, 1.0 if j < 3 else -1.0
        plane = [a for a in range(3) if a != k]
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        center[k] = s * 0.5 * size[k]
        radius = np.linalg.norm(pts[:, plane] - center[plane], axis=1).max()
        axis = np.zeros(3)
        axis[k] = s
        found.append({"base": center, "axis": axis, "height": past[j], "radius": radius})
    return found


def disk(center, axis, radius, n: int = 16):
    """(n, 3, 3) triangles of a flat disk: the cap that closes the wall under a stud."""
    import numpy as np

    k = int(np.argmax(np.abs(axis)))
    u, v = [a for a in range(3) if a != k]
    ring = []
    for a in np.linspace(0.0, 2.0 * np.pi, n, endpoint=False):
        p = np.array(center, dtype=float)
        p[u] += radius * np.cos(a)
        p[v] += radius * np.sin(a)
        ring.append(p)
    return np.array([[center, ring[i], ring[(i + 1) % n]] for i in range(n)])


def fill_voxels(tris, lo, hi, axes=(2,)):
    """(solid, origin): the voxels of the triangle soup ``tris`` (N, 3, 3), m, with every region
    its walls enclose in a slice across one of ``axes`` filled (module docstring). The default
    slices are the horizontal ones.

    The grid starts half a voxel below ``lo``, so a face on the LDU grid lies mid-voxel.
    """
    import numpy as np
    from scipy.ndimage import binary_fill_holes

    origin = lo - 0.5 * VOXEL_M
    shape = np.ceil((hi - origin) / VOXEL_M).astype(int) + 1
    surface = np.zeros(shape, dtype=bool)
    for tri in tris:  # barycentric samples a third of a voxel apart
        edge = max(np.linalg.norm(tri[a] - tri[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        n = int(np.ceil(edge / (VOXEL_M / 3))) + 1
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep][:, None] / n, j[keep][:, None] / n
        pts = tri[0] + a * (tri[1] - tri[0]) + b * (tri[2] - tri[0])
        idx = np.floor((pts - origin) / VOXEL_M).astype(int)
        surface[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    solid = np.zeros_like(surface)
    for axis in axes:
        slices = [binary_fill_holes(s) for s in np.moveaxis(surface, axis, 0)]
        solid |= np.moveaxis(np.stack(slices), 0, axis)
    return solid, origin


def voxel_faces(solid, origin):
    """(vertices, faces) of every voxel face between a solid voxel and an empty one."""
    import numpy as np

    pad = np.pad(solid, 1)
    verts, faces = [], []
    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for sign in (0, 1):  # the voxel's low or high face along ``axis``
            neighbor = np.roll(pad, 1 if sign == 0 else -1, axis=axis)
            cells = np.argwhere(pad & ~neighbor) - 1  # back to unpadded indices
            corner = cells.astype(float)
            corner[:, axis] += sign
            quad = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = corner.copy()
                c[:, u] += du
                c[:, v] += dv
                quad.append(origin + c * VOXEL_M)
            if (sign == 1) != (axis != 1):  # the corners run about e_u x e_v; face outward
                quad.reverse()
            n = len(cells)
            k = np.arange(n) + sum(len(q) for q in verts)
            verts.append(np.concatenate(quad))
            faces.append(np.stack([k, k + n, k + 2 * n], axis=1))
            faces.append(np.stack([k, k + 2 * n, k + 3 * n], axis=1))
    return np.concatenate(verts), np.concatenate(faces)


def build_part(lib: Library, part_id: str, max_hulls: int, seed: int) -> dict:
    """The JSON record of one part (module docstring)."""
    import coacd
    import numpy as np
    import trimesh

    out = {"collision": [], "visual": [], "studs": [], "files": {}}
    expand(lib, f"{part_id}.dat", np.eye(3), np.zeros(3), None, out)
    to_body = np.array(LDRAW_TO_BODY) * LDU_M
    coll = np.array(out["collision"]) @ to_body.T
    lo, hi = coll.reshape(-1, 3).min(axis=0), coll.reshape(-1, 3).max(axis=0)
    if hi[1] - lo[1] > hi[0] - lo[0] + 1e-9:  # long side on x: a quarter turn about z
        turn = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        to_body = turn @ to_body
        coll = coll @ turn.T
        lo, hi = coll.reshape(-1, 3).min(axis=0), coll.reshape(-1, 3).max(axis=0)
    center = 0.5 * (lo + hi)
    size = hi - lo
    coll -= center
    studs = outside_studs(
        [np.array(tris) @ to_body.T - center for tris in out["studs"] if tris], size
    )
    # Close the wall under each stud: an open stud leaves a hole into the part otherwise.
    caps = [disk(s["base"], s["axis"], s["radius"]) for s in studs]
    coll = np.concatenate([coll, *caps]) if caps else coll

    axes = (2,) if part_id in HORIZONTAL_FILL else (0, 1, 2)
    solid, origin = fill_voxels(coll, -0.5 * size, 0.5 * size, axes)
    mesh = trimesh.Trimesh(*voxel_faces(solid, origin), process=True)
    convex = mesh.volume >= CONVEX_FILL * mesh.convex_hull.volume
    pieces = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        max_convex_hull=1 if convex else max_hulls,
        decimate=True,
        max_ch_vertex=HULL_VERTICES,
        seed=seed,
    )
    union = np.concatenate([v for v, _ in pieces])
    span_lo, span_hi = union.min(axis=0), union.max(axis=0)
    scale = size / (span_hi - span_lo)
    shift = -0.5 * (span_lo + span_hi)
    hulls = [(v + shift) * scale for v, _ in pieces]
    volumes = [float(trimesh.convex.convex_hull(h).volume) for h in hulls]

    root = out["files"][f"{part_id}.dat"]
    # 3660.dat is a "~Moved to" alias: the title comes from the first file with a real one.
    title = next(h["title"] for h in out["files"].values() if not h["title"].startswith("~Moved"))
    axes, reason = PARTS[part_id]
    return {
        "id": part_id,
        "title": " ".join(title.lstrip("~").split()),
        "size": np.round(size, DECIMALS).tolist(),
        "closing_axes": list(axes),
        "closing_axes_reason": reason,
        "hulls": [np.round(h, DECIMALS).tolist() for h in hulls],
        "hull_volumes": [round(v, 12) for v in volumes],
        "studs": [
            {key: np.round(value, DECIMALS).tolist() for key, value in stud.items()}
            for stud in studs
        ],
        "source": {
            "file": f"{part_id}.dat",
            "author": root.get("author", ""),
            "license": root.get("license", ""),
            "update": root.get("update", ""),
            "files": len(out["files"]),
            "triangles": {"collision": len(out["collision"]), "visual": len(out["visual"])},
        },
        "_files": out["files"],
    }


def notice(library: pathlib.Path, records: list[dict]) -> str:
    """The attribution the library's licence asks for, with every file each part drew on."""
    zip_path = library.parent / "complete.zip"  # its mtime is the server's Last-Modified
    stamp = datetime.datetime.fromtimestamp(zip_path.stat().st_mtime, datetime.UTC)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    licenses = sorted({h.get("license", "") for r in records for h in r["_files"].values()})
    authors = sorted({h.get("author", "") for r in records for h in r["_files"].values()} - {""})
    return "\n".join(
        [
            "The part geometry in this directory is derived from the LDraw Parts Library",
            "(https://library.ldraw.org), by the LDraw.org contributors, used under the",
            "licences the files state:",
            *[f"  {lic}" for lic in licenses],
            *[f"{name}: {url}" for name, url in LICENSE_URLS.items() if name in str(licenses)],
            "",
            f"Library: the official complete.zip of {stamp:%Y-%m-%d}, sha256",
            f"{digest}.",
            "Changes: the triangles were expanded, converted to meters and reoriented, the studs",
            "left out of the collision mesh, the cavities each horizontal slice encloses filled",
            "(each slice along any axis for the parts after 3660), and the result split into",
            "convex hulls by CoACD (scripts/ldraw_parts.py).",
            "LDraw is a trademark of the Estate of James Jessiman.",
            "LEGO is a trademark of the LEGO Group, which does not sponsor or endorse this work.",
            "",
            "Parts: " + ", ".join(f"{r['id']} ({r['title']})" for r in records),
            "Authors of the files used: " + "; ".join(authors),
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict:
    lib = Library(args.ldraw)
    ids = [p for p in args.parts.split(",") if p]
    unknown = [p for p in ids if p not in PARTS]
    if unknown:
        raise ValueError(f"no closing axes registered for {unknown}")
    args.out.mkdir(parents=True, exist_ok=True)
    records = [build_part(lib, p, args.max_hulls, args.seed) for p in ids]
    for r in records:
        body = {k: v for k, v in r.items() if k != "_files"}
        (args.out / f"{r['id']}.json").write_text(json.dumps(body, indent=1) + "\n")
    (args.out / "NOTICE").write_text(notice(args.ldraw, records))
    return {
        r["id"]: {
            "size_mm": [round(1000 * s, 2) for s in r["size"]],
            "hulls": len(r["hulls"]),
            "studs": len(r["studs"]),
            "axes": r["closing_axes"],
        }
        for r in records
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        print("LDRAW_OK " + json.dumps(result), flush=True)
        return 0
    except Exception as err:  # noqa: BLE001 -- any failure must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        print("LDRAW_FAIL " + json.dumps({"reason": "exception", "detail": detail}), flush=True)
        return 1


if __name__ == "__main__":
    os._exit(main())

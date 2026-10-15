"""DAgger shards -> the pi0.5 (V) DAgger dataset of one MS7h DAgger round (HD59).

The chain: the shards' takeover cycles go through convert_to_lerobot.py --split_cycles --dagger
(MS7f wave 3's conversion: --context camera_box_target --gripper continuous) in LANES parallel
lanes; lerobot's aggregate_datasets merges the lanes, as merge_roots.sh does; then
derive_vla_dataset.derive builds the V format (23-wide state, one prompt). The output's
meta/stats.json is then replaced by the corpus's file (--corpus, h/vla/dataset): the fine-tune
keeps the corpus normalizer, and colab_train_pi05.sh checks this file's md5 against the corpus's.
The run stops unless the features, fps, prompt and gripper marker equal the corpus's and every
video decodes (ffmpeg, as check_videos.sh). Last, the root goes into one tar, split into
--part_mb parts for colab upload (one base64 request a file), with its size and sha256.

--drop_reopen_lead TOL_M passes convert_to_lerobot.py's flag (off by default; the rule of
dag_mask.py): each takeover loses the leading frames in which the teacher opens jaws the policy
left closed. The token counts them.

The host link is slow (2.8 MB/s), so another host converts its own shards with --act_only into
an ACT root, and the final run takes that root with --act. Each root carries
meta/dagger_vla.json, and roots converted with another --drop_reopen_lead are refused.

Run in the lerobot venv, on a host with ffmpeg:
    lerobot_stack/.venv/bin/python lerobot_stack/dagger_vla.py --shards DIR... [--act ROOT...]
        --corpus ~/data/maker-arm-ms7/h/vla/dataset --out OUT [--lanes 8] [--drop_reopen_lead M]
-> one DAGGER_VLA_OK / DAGGER_ACT_OK / DAGGER_VLA_FAIL line. After a failure, delete OUT (not
OUT.work) and run again: a rerun on the same shards reuses each lane whose log holds CONVERT_OK.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = "youtalk/maker-arm-legosort"
CONVERT = ["--split_cycles", "--dagger", "--gripper", "continuous"]
CONVERT += ["--context", "camera_box_target"]
MARKER = "meta/dagger_vla.json"
COUNTS = ("episodes", "reopen_frames", "reopen_takeovers", "reopen_emptied")


def shard_files(paths: list[Path]) -> list[Path]:
    """Each directory's *.hdf5, sorted, and each file as given."""
    files = [f for p in paths for f in (sorted(p.glob("*.hdf5")) if p.is_dir() else [p])]
    if not files:
        raise ValueError(f"no shards in {paths}")
    return files


def lane_counts(log: str) -> dict | None:
    """The counts of a lane's CONVERT_OK line, or None."""
    line = next((s for s in log.splitlines() if s.startswith("CONVERT_OK")), None)
    if line is None:
        return None
    found = dict(re.findall(r"(\w+)=(\S+)", line))
    return {k: int(found.get(k, 0)) for k in COUNTS}


def convert_lanes(shards: list[Path], work: Path, lanes: int, tol: float | None) -> list[Path]:
    """Convert ``shards`` in ``lanes`` processes; return the lane roots (each with its marker)."""
    plan = [[str(s) for s in shards[k::lanes]] for k in range(lanes)]
    plan = [p for p in plan if p]
    work.mkdir(parents=True, exist_ok=True)
    plan_file = work / "lanes.json"
    old = json.loads(plan_file.read_text()) if plan_file.exists() else None
    if old != {"lanes": plan, "reopen_tol": tol}:
        shutil.rmtree(work)
        work.mkdir(parents=True)
        plan_file.write_text(json.dumps({"lanes": plan, "reopen_tol": tol}))
    extra = [] if tol is None else ["--drop_reopen_lead", str(tol)]
    jobs = {}
    for k, mine in enumerate(plan):
        log = work / f"convert_lane_{k}.log"
        if log.exists() and lane_counts(log.read_text()) is not None:
            continue
        shutil.rmtree(work / f"conv_{k}", ignore_errors=True)
        argv = [sys.executable, str(HERE / "convert_to_lerobot.py"), "--shards", *mine]
        argv += ["--root", str(work / f"conv_{k}" / "dataset"), *CONVERT, *extra]
        jobs[k] = subprocess.Popen(argv, stdout=log.open("w"), stderr=subprocess.STDOUT)
    for job in jobs.values():
        job.wait()
    roots = []
    for k in range(len(plan)):
        counts = lane_counts((work / f"convert_lane_{k}.log").read_text())
        if counts is None:
            raise RuntimeError(f"lane {k} has no CONVERT_OK, see {work}/convert_lane_{k}.log")
        root = work / f"conv_{k}" / "dataset"
        if counts["episodes"]:
            write_marker(root, tol, counts)
            roots.append(root)
    return roots


def write_marker(root: Path, tol: float | None, counts: dict) -> None:
    (root / MARKER).write_text(json.dumps({"reopen_tol": tol, **counts}))


def read_marker(root: Path, tol: float | None) -> dict:
    path = root / MARKER
    if not path.is_file():
        raise ValueError(f"{root} has no {MARKER}: convert it with dagger_vla.py --act_only")
    marker = json.loads(path.read_text())
    if marker["reopen_tol"] != tol:
        raise ValueError(f"{root} has reopen_tol {marker['reopen_tol']}, this run {tol}")
    return marker


def merge(roots: list[Path], out: Path) -> None:
    """aggregate_datasets as merge_roots.sh calls it, and the one gripper marker of the roots."""
    from lerobot.datasets.aggregate import aggregate_datasets

    if not roots:
        raise ValueError("no takeover episodes to merge")
    markers = {(r / "meta/action_space.json").read_text() for r in roots}
    if len(markers) != 1:
        raise ValueError(f"the roots carry different action_space.json markers: {markers}")
    aggregate_datasets(
        [REPO] * len(roots), REPO, roots=roots, aggr_root=out, video_files_size_in_mb=1
    )
    (out / "meta/action_space.json").write_text(markers.pop())


def mismatches(out: Path, corpus: Path) -> list[str]:
    """What of ``out``'s format differs from the corpus's."""
    info, want = (json.loads((r / "meta/info.json").read_text()) for r in (out, corpus))
    bad = [k for k in ("features", "fps", "codebase_version", "robot_type") if info[k] != want[k]]
    for name in ("meta/action_space.json", "meta/stats.json"):
        if (out / name).read_bytes() != (corpus / name).read_bytes():
            bad.append(name)
    tasks = [list(pd.read_parquet(r / "meta/tasks.parquet").index) for r in (out, corpus)]
    if tasks[0] != tasks[1]:
        bad.append(f"tasks {tasks[0]} != {tasks[1]}")
    return bad


def bad_videos(root: Path) -> tuple[int, list[str]]:
    """(videos, the ones ffmpeg reports an error in)."""
    files = sorted(str(f) for f in root.rglob("*.mp4"))

    def errors(f: str) -> bool:
        cmd = ["ffmpeg", "-v", "error", "-i", f, "-f", "null", "-"]
        run = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return bool(run.stderr.strip()) or run.returncode != 0

    with ThreadPoolExecutor(10) as pool:
        flags = list(pool.map(errors, files))
    return len(files), [f for f, b in zip(files, flags) if b]


def pack(out: Path, part_mb: int) -> dict:
    """tar ``out`` into OUT.tar parts of ``part_mb`` MB; the size and sha256 of the whole tar."""
    tar = out.with_name(out.name + ".tar")
    for old in out.parent.glob(f"{tar.name}*"):
        old.unlink()
    subprocess.run(["tar", "-cf", str(tar), "-C", str(out.parent), out.name], check=True)
    digest = hashlib.sha256()
    with tar.open("rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    size, sha = tar.stat().st_size, digest.hexdigest()
    split = ["split", "-b", f"{part_mb}M", "-d", "-a", "3", str(tar), f"{tar}.part"]
    subprocess.run(split, check=True)
    tar.unlink()
    parts = sorted(out.parent.glob(f"{tar.name}.part*"))
    tar.with_name(tar.name + ".sha256").write_text(f"{sha}  {size}  {tar.name}\n")
    return {"tar": f"{tar}.part*", "bytes": size, "sha256": sha, "parts": len(parts)}


def run(args) -> str:
    from derive_vla_dataset import derive

    out, tol = args.out.expanduser(), args.drop_reopen_lead
    if out.exists():
        raise FileExistsError(f"{out} exists")
    roots = []
    if args.shards:
        work = out.with_name(out.name + ".work")
        roots += convert_lanes(shard_files(args.shards), work / "lanes", args.lanes, tol)
    markers = [json.loads((r / MARKER).read_text()) for r in roots]
    markers += [read_marker(r.expanduser(), tol) for r in args.act]
    roots += [r.expanduser() for r in args.act]
    total = {k: sum(m[k] for m in markers) for k in COUNTS}
    counts = " ".join(f"{k}={v}" for k, v in total.items())
    counts += f" reopen_tol={'off' if tol is None else tol} roots={len(roots)}"
    if args.act_only:
        merge(roots, out)
        write_marker(out, tol, total)
        return f"DAGGER_ACT_OK {counts} out={out}"
    corpus = args.corpus.expanduser()
    act = out.with_name(out.name + ".work") / "act"
    shutil.rmtree(act, ignore_errors=True)
    merge(roots, act)
    result = derive(act, out)
    shutil.copyfile(corpus / "meta/stats.json", out / "meta/stats.json")
    bad = mismatches(out, corpus)
    if bad:
        raise ValueError(f"{out} differs from the corpus {corpus}: {bad}")
    videos, broken = bad_videos(out)
    if broken:
        raise ValueError(f"{len(broken)} of {videos} videos do not decode: {broken[:5]}")
    packed = " ".join(f"{k}={v}" for k, v in pack(out, args.part_mb).items())
    return f"DAGGER_VLA_OK {counts} frames={result['frames']} videos={videos} out={out} {packed}"


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"DAGGER_VLA_FAIL usage: {message}", flush=True)
        os._exit(2)


def main() -> int:
    parser = TokenArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", nargs="*", type=Path, default=[], help="dirs or .hdf5 files")
    parser.add_argument("--act", nargs="*", type=Path, default=[], help="--act_only roots")
    parser.add_argument("--out", required=True, type=Path, help="the new root; must not exist")
    parser.add_argument("--corpus", type=Path, help="the V corpus root (h/vla/dataset)")
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--drop_reopen_lead", type=float, default=None, metavar="TOL_M")
    parser.add_argument("--act_only", action="store_true", help="stop at the merged ACT root")
    parser.add_argument("--part_mb", type=int, default=256, help="tar part size for colab upload")
    try:
        args = parser.parse_args()
        if not args.shards and not args.act:
            raise ValueError("give --shards or --act")
        if not args.act_only and args.corpus is None:
            raise ValueError("--corpus is needed unless --act_only")
        line = run(args)
    except Exception as exc:  # noqa: BLE001 -- any failure must become DAGGER_VLA_FAIL
        traceback.print_exc()
        print(f"DAGGER_VLA_FAIL {type(exc).__name__}: {str(exc)[:500]}", flush=True)
        return 1
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())

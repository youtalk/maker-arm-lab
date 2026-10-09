"""Fleet control for the MS7a campaign: one host table, five subcommands, one token each.

Stdlib-only, like run_gate_eval.py, so it runs under the system Python on every host. The host
table is <campaign>/hosts.json. Per host: "ssh" (the ssh target, null for the local host),
"clone" (the main clone), "worktree" (the campaign worktree), "setup" (a shell line run before
every job), "lanes" (the read lane cap) and "device" (the policy server's, cuda or cpu). The
rules are the MS7a design's hosts and data flow:

  sync --sha SHA  SYNC_OK / SYNC_FAIL. Every host fetches origin, checks SHA out detached in
                  its campaign worktree (added on first use, venvs symlinked from the main
                  clone) and runs FAST_TESTS. A host with "fetch": "bundle" cannot reach
                  GitHub (Machine C's resolver times out), so it fetches SHA from a
                  git bundle this host builds from origin and ships to <campaign>/sync.bundle.
  ship SRC DST    SHIP_OK / SHIP_FAIL. rsync --partial; either end may be HOST:PATH. The file
                  lists (names and sizes) of both ends must match after the copy. A
                  pretrained_model gets its action_space.json marker.
  run HOST NAME -- CMD
                  RUN_OK / RUN_FAIL. CMD in the host's worktree, logged to
                  <campaign>/jobs/NAME.log, whose first line is the SHA. A remote job detaches;
                  a local one runs in the foreground, inside the caller's one tracked call.
  read ...        GATE_OK / GATE_FAIL, READ_FAIL on an orchestration error. One gate read split
                  across hosts by lane caps, then the supervisor's aggregate pass over the
                  payloads pulled into one fresh TAG directory. With --pi05 (MS7h design
                  7.3) a host runs one supervisor whose lanes share one pi05_server.py, so
                  each host holds one pi0.5 copy and no image crosses a host link.
  status          STATUS_OK / STATUS_FAIL. The last token line of every job log per host.

A job's log ends with the wrapper's own "JOB_EXIT rc=N" line after the job's token, so a job
that dies without a token still reads as finished; a log with neither reads as RUNNING.

Every path is the same absolute path on every host (every host uses the same home directory). A payload
records its policy as act:<checkpoint path>, and the aggregate pass reuses a payload only if
that id matches the checkpoint it is given.
"""

import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SSH = ["ssh", "-A", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
RSYNC = ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"]
# Under a minute on Machine C; core/, the Rust SDK import, MuJoCo, the supervisor.
FAST_TESTS = (
    "tests/test_layout.py",
    "tests/test_stowed.py",
    "tests/test_grasp.py",
    "tests/test_heap_oracle.py",
    "tests/test_parts.py",
    "tests/test_gate.py",
    "tests/test_run_gate_eval.py",
    "tests/test_maker_arm_rs_available.py",
    "tests/test_mj_scene_parts.py",
    "tests/test_fleet.py",
)
# A token line, e.g. "GATE_OK {...}" or "GATE_ERROR: ..."; Python re and GNU grep -E agree.
TOKEN_RE = r"^[A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9](\s|:|$)"
MARKER = {"action_space": "absolute", "gripper": "continuous"}
FAIL = {
    "sync": "SYNC_FAIL",
    "ship": "SHIP_FAIL",
    "run": "RUN_FAIL",
    "read": "READ_FAIL",
    "status": "STATUS_FAIL",
}
LOCAL = {"ssh": None}
BUNDLE_REF = "refs/fleet/sync"
q = shlex.quote


class FleetError(RuntimeError):
    """An orchestration step failed; main prints the subcommand's fail token."""


def sh(host: dict, script: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a bash script on the host: locally, or over ssh -A."""
    cmd = ["bash", "-c", script]
    if host["ssh"]:
        cmd = SSH + [host["ssh"], shlex.join(cmd)]
    return subprocess.run(
        cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout, check=False
    )


def check(result: subprocess.CompletedProcess, what: str) -> str:
    if result.returncode != 0:
        raise FleetError(
            f"{what} exited {result.returncode}:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    return result.stdout


def parallel(fn, items) -> list:
    items = list(items)
    with ThreadPoolExecutor(max(1, len(items))) as pool:
        return list(pool.map(fn, items))


# --- sync ---------------------------------------------------------------------------------


def sync_script(host: dict, sha: str, campaign: Path) -> str:
    wt, clone = q(host["worktree"]), q(host["clone"])
    source = q(str(campaign / "sync.bundle")) + " " + BUNDLE_REF
    fetch = source if host.get("fetch") == "bundle" else "origin"
    return f"""set -e
{host["setup"]}
git -C {clone} fetch -q {fetch}
[ -e {wt} ] || git -C {clone} worktree add -q --detach {wt} {sha}
[ -e {wt}/.venv ] || ln -s {clone}/.venv {wt}/.venv
[ -e {wt}/lerobot_stack/.venv ] || ln -s {clone}/lerobot_stack/.venv {wt}/lerobot_stack/.venv
cd {wt}
git checkout -q --detach {sha}
test "$(git rev-parse HEAD)" = {sha}
mkdir -p {q(str(campaign / "jobs"))}
tests=$(ls -d {" ".join(FAST_TESTS)} 2>/dev/null || true)  # an older SHA lacks some
PYTHONPATH={wt}/source .venv/bin/python -m pytest -q -p no:cacheprovider $tests
"""


def ship_bundle(host: dict, sha: str, campaign: Path) -> None:
    """Build a bundle holding SHA from this host's origin and copy it to the host."""
    repo = ["git", "-C", str(Path(__file__).resolve().parents[1])]
    path = campaign / "sync.bundle"
    check(sh(host, f"mkdir -p {q(str(campaign))}"), "mkdir")
    for cmd in (
        repo + ["fetch", "-q", "origin"],
        repo + ["update-ref", BUNDLE_REF, sha],
        repo + ["bundle", "create", "-q", str(path), BUNDLE_REF],
        RSYNC + [str(path), rsync_url(host, str(path))],
    ):
        check(subprocess.run(cmd, capture_output=True, text=True, check=False), "bundle")


def cmd_sync(args, hosts):
    names = args.hosts.split(",") if args.hosts else list(hosts)
    # A job reads its worktree for hours; moving the checkout under it kills it (campaign n).
    busy = {}
    for n in [] if args.force else names:
        jobs = [j for j, tok in job_tokens(hosts[n], args.campaign).items() if tok == "RUNNING"]
        if jobs:
            busy[n] = jobs
    if busy:
        raise FleetError(f"jobs still RUNNING, the worktree stays put (--force moves it): {busy}")

    def one(name):
        t0 = time.monotonic()
        if hosts[name].get("fetch") == "bundle":
            ship_bundle(hosts[name], args.sha, args.campaign)
        result = sh(hosts[name], sync_script(hosts[name], args.sha, args.campaign))
        return name, result, time.monotonic() - t0

    summary = {}
    for name, result, wall in parallel(one, names):
        out = (result.stdout + result.stderr).strip().splitlines()
        last = result.stdout.strip().splitlines()[-1:] if result.returncode == 0 else out[-1:]
        print(f"[sync] {name}: rc={result.returncode} wall={wall:.0f}s {' '.join(last)}")
        if result.returncode:
            print("\n".join(out[-30:]))
        summary[name] = {"rc": result.returncode, "wall_s": round(wall, 1)}
    ok = all(v["rc"] == 0 for v in summary.values())
    return f"{'SYNC_OK' if ok else 'SYNC_FAIL'} " + json.dumps({"sha": args.sha, **summary}), (
        0 if ok else 1
    )


# --- ship ---------------------------------------------------------------------------------


def parse_end(spec: str, hosts: dict) -> tuple[dict, str]:
    """HOST:PATH with a host of the table, else a local path."""
    name, sep, path = spec.partition(":")
    host, path = (hosts[name], path) if sep and name in hosts else (LOCAL, spec)
    return host, path.rstrip("/") or "/"


def rsync_url(host: dict, path: str) -> str:
    return f"{host['ssh']}:{path}" if host["ssh"] else path


def listing(host: dict, path: str) -> list[str]:
    """Sorted "size relpath" lines of every file under path. A single file lists as one
    "size " line (find's %P of a starting point is empty)."""
    out = check(sh(host, f"find {q(path)} -type f -printf '%s %P\\n'"), f"find {path}")
    return sorted(out.splitlines())


def ensure_marker(host: dict, ckpt: str) -> None:
    """The policy server reads the nearest action_space.json of the checkpoint dir and three
    parents. A mid-training checkpoint has none, so its parent gets the campaign's contract."""
    walk = " ".join(q(ckpt + "/.." * i + "/action_space.json") for i in range(4))
    marker = q(os.path.dirname(ckpt) + "/action_space.json")
    script = (
        f'for m in {walk}; do [ -f "$m" ] && exit 0; done; '
        f"printf '%s\\n' {q(json.dumps(MARKER))} > {marker}"
    )
    check(sh(host, script), f"marker for {ckpt}")


def ship(src: dict, src_path: str, dst: dict, dst_path: str, streams: int = 1) -> dict:
    """rsync src to dst over `streams` disjoint file lists, then compare both ends' lists."""
    if src["ssh"] and dst["ssh"]:
        raise FleetError("ship copies local to remote or remote to local, not remote to remote")
    files = listing(src, src_path)
    is_file = len(files) == 1 and files[0].endswith(" ")
    check(sh(dst, f"mkdir -p {q(os.path.dirname(dst_path) if is_file else dst_path)}"), "mkdir")
    if is_file:
        jobs = [(RSYNC + [rsync_url(src, src_path), rsync_url(dst, dst_path)], "")]
    else:
        # Largest first, dealt round robin: the streams carry about the same bytes.
        names = [f.split(" ", 1)[1] for f in sorted(files, key=lambda f: -int(f.split()[0]))]
        urls = [rsync_url(src, src_path) + "/", rsync_url(dst, dst_path) + "/"]
        jobs = [
            (RSYNC + ["--files-from=-", *urls], "\n".join(names[i::streams]))
            for i in range(min(streams, len(names)))
        ]
    for result in parallel(
        lambda job: subprocess.run(
            job[0], input=job[1], capture_output=True, text=True, check=False
        ),
        jobs,
    ):
        check(result, "rsync")
    if listing(dst, dst_path) != files:
        raise FleetError(f"{dst_path}: the file list differs from the source's after the copy")
    if os.path.basename(dst_path) == "pretrained_model":
        ensure_marker(dst, dst_path)
    return {"files": len(files), "bytes": sum(int(f.split()[0]) for f in files)}


def cmd_ship(args, hosts):
    src, src_path = parse_end(args.src, hosts)
    dst, dst_path = parse_end(args.dst, hosts)
    t0 = time.monotonic()
    stats = ship(src, src_path, dst, dst_path, args.streams)
    stats["wall_s"] = round(time.monotonic() - t0, 1)
    return "SHIP_OK " + json.dumps({"src": args.src, "dst": args.dst, **stats}), 0


# --- run and status -----------------------------------------------------------------------


def job_script(host: dict, cmd: str) -> str:
    wt = q(host["worktree"])
    return f"""cd {wt} || {{ echo "JOB_EXIT rc=97"; exit 97; }}
echo "sha $(git rev-parse HEAD 2>&1)"
{host["setup"]}
export PYTHONPATH={wt}/source
(
{cmd}
)
rc=$?
echo "JOB_EXIT rc=$rc"
exit $rc
"""


def detach(log: str, script: str) -> str:
    """A remote launch line. All three fds of the child are redirected, so ssh returns at once."""
    # Two lines, not "mkdir && setsid ... &": that would background the whole list in a
    # subshell that keeps the ssh channel open until the job ends.
    return (
        f"mkdir -p {q(os.path.dirname(log))}\n"
        f"setsid nohup bash -c {q(script)} < /dev/null > {q(log)} 2>&1 &"
    )


def start_local(log: str, script: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as out:
        return subprocess.Popen(
            ["bash", "-c", script], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT
        )


def cmd_run(args, hosts):
    host = hosts[args.host]
    log = str(args.campaign / "jobs" / f"{args.name}.log")
    script = job_script(host, " ".join(args.cmd))
    if host["ssh"]:
        check(sh(host, detach(log, script), timeout=120), f"launch on {args.host}")
        return "RUN_OK " + json.dumps({"host": args.host, "log": log, "detached": True}), 0
    rc = start_local(log, script).wait()
    token = "RUN_OK" if rc == 0 else "RUN_FAIL"
    return f"{token} " + json.dumps({"host": args.host, "log": log, "rc": rc}), 0 if rc == 0 else 1


def status_script(campaign: Path) -> str:
    return f"""shopt -s nullglob
cd {q(str(campaign / "jobs"))} 2>/dev/null || exit 0
for f in *.log; do echo "== ${{f%.log}}"; grep -aE {q(TOKEN_RE)} "$f" | tail -n 2; done
"""


def parse_jobs(text: str) -> dict[str, str]:
    """{job name: its last token line} from status_script's output."""
    jobs: dict[str, list[str]] = {}
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("== "):
            lines = jobs.setdefault(line[3:], [])
        else:
            lines.append(line)
    return {name: last_token(lines) for name, lines in jobs.items()}


def last_token(lines: list[str]) -> str:
    """The job's own last token line; the wrapper's JOB_EXIT only when the job printed none;
    RUNNING when there is neither."""
    tokens = [line for line in lines if re.match(TOKEN_RE, line)]
    own = [t for t in tokens if not t.startswith("JOB_EXIT")]
    return (own or tokens or ["RUNNING"])[-1]


def job_tokens(host: dict, campaign: Path) -> dict[str, str]:
    return parse_jobs(check(sh(host, status_script(campaign), timeout=300), "status"))


def cmd_status(args, hosts):
    def one(name):
        try:
            return name, job_tokens(hosts[name], args.campaign)
        except (FleetError, subprocess.TimeoutExpired) as exc:
            return name, exc

    ok = True
    for name, jobs in parallel(one, hosts):
        if isinstance(jobs, Exception):
            ok = False
            print(f"[status] {name}: UNREACHABLE {str(jobs).splitlines()[0]}")
            continue
        for job, token in sorted(jobs.items()):
            print(f"[status] {name} {job}: {token}")
    return ("STATUS_OK" if ok else "STATUS_FAIL"), 0 if ok else 1


# --- read ---------------------------------------------------------------------------------


def parse_draws(spec: str) -> list[int]:
    """'0-11' or '0,2,5-7' -> the listed draws, in order, each once."""
    draws = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        draws += range(int(lo), int(hi or lo) + 1)
    if not draws or len(set(draws)) != len(draws):
        raise ValueError(f"no draws or a repeated draw in {spec!r}")
    return draws


def parse_hosts(spec: str) -> list[tuple[str, int | None]]:
    """'c,a:8,b:4' -> [(name, lane cap or None for the table's)]."""
    out = []
    for part in spec.split(","):
        name, _, lanes = part.partition(":")
        out.append((name, int(lanes) if lanes else None))
    return out


def split_draws(draws: list[int], caps: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Contiguous blocks of draws per host, sized in proportion to the lane caps by largest
    remainder (ties to the earlier host), so every draw runs exactly once."""
    total = sum(cap for _, cap in caps)
    quotas = [len(draws) * cap / total for _, cap in caps]
    counts = [int(x) for x in quotas]
    by_remainder = sorted(range(len(caps)), key=lambda i: counts[i] - quotas[i])
    for i in by_remainder[: len(draws) - sum(counts)]:
        counts[i] += 1
    out, start = {}, 0
    for (name, _), n in zip(caps, counts):
        out[name], start = draws[start : start + n], start + n
    return out


def lanes_of(draws: list[int], cap: int) -> list[list[int]]:
    k = min(cap, len(draws))
    return [draws[i::k] for i in range(k)]


def fresh_payloads(tag_dir: Path, draws: list[int], start: float) -> dict[int, Path]:
    """Every draw's payload in tag_dir, each written after the read started. A payload older
    than the read is a supervisor cache hit, and a cached payload is not a read."""
    payloads = {int(p.name[5:8]): p for p in tag_dir.glob("eval_[0-9][0-9][0-9].json")}
    missing = sorted(set(draws) - set(payloads))
    stale = sorted(p.name for p in payloads.values() if p.stat().st_mtime < start)
    if missing or stale:
        raise FleetError(f"missing payloads for draws {missing}; stale payloads {stale}")
    return payloads


# The MS7a read protocol (MS7a design, read protocol): K = 50 held-out parts, the 16-wide
# camera_box_target context, the continuous gripper, the heap gate's cycles rule.
PROTOCOL = (
    "--backend mujoco --max_cycles_per_stowed 1.3 --num_envs 5 --heap_k 50 "
    "--parts_seed 20260915 --episode_draws --frame_jitter --context camera_box_target "
    "--gripper continuous --eval_timeout_s 14400"
)


def supervisor_args(args, ledger: str) -> list[str]:
    """The supervisor flags every lane and the aggregate pass share, before the extra flags
    after --, which win on a repeat because argparse keeps the last value."""
    sampling = []  # the pi05 server's sampling switches, only when a read sets them
    for name in (
        "pi05_num_steps",
        "pi05_noise_scale",
        "pi05_jaw_noise_scale",
        "pi05_jaw_noise_shared",
        "pi05_jaw_noise_abs",
        "pi05_still_noise_scale",
        "pi05_still_rad",
        "pi05_still_open",
        "pi05_jaw_latch",
        "pi05_retry_noise_scale",
    ):
        value = getattr(args, name, None)
        if isinstance(value, bool):  # a store_true switch
            sampling += [f"--{name}"] * value
        elif value is not None:
            sampling += [f"--{name}", str(value)]
    return shlex.split(PROTOCOL) + [
        "--checkpoint",
        args.ckpt,
        *(["--pi05"] if getattr(args, "pi05", False) else []),
        *sampling,
        "--ledger",
        ledger,
        "--threshold",
        str(args.threshold),
        "--seed",
        str(args.seed),
        "--processes",
        str(max(args.draws) + 1),
        "--parts_split",
        args.split,
        "--n_action_steps",
        str(args.n_steps),
        "--client_flags",
        args.client_flags,
    ]


def host_jobs(args, base, host, name, draws, cap, port, out) -> list[tuple[str, list[str]]]:
    """(job name, supervisor argv) of every supervisor run of one host, on ports from `port`.
    An ACT read runs one supervisor, and so one server, a lane. A pi0.5 read runs one supervisor
    whose --lanes clients share one pi05_server.py (MS7h design 7.3)."""
    pi05 = getattr(args, "pi05", False)
    groups = [draws] if pi05 and draws else lanes_of(draws, cap)
    jobs = []
    for i, group in enumerate(groups):
        lane = [*base, "--device", host["device"], *args.extra, "--out_dir", out]
        lane += ["--port", str(port + i), "--only_processes", ",".join(map(str, group))]
        if pi05:
            lane += ["--lanes", str(min(cap, len(draws)))]
        jobs.append((f"read-{args.tag}-{name}-{i}", lane))
    return jobs


def cmd_read(args, hosts):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_gate_eval as ge

    start = time.time()
    tag_dir = args.campaign / "reads" / args.tag
    ledger = str(args.ledger or args.campaign / "ledger.json")
    base = supervisor_args(args, ledger)
    only = ",".join(map(str, args.draws))
    aggregate = base + args.extra + ["--out_dir", str(tag_dir), "--only_processes", only]
    with contextlib.redirect_stdout(sys.stderr):  # its usage error prints GATE_ERROR
        cfg = ge._build_cfg(ge._parse_args(aggregate))
    ge.gate_draws(cfg)  # the seed rules, before any host starts
    args.pi05 = cfg.pi05  # --pi05 after -- (read_h.sh ADD) too: never one pi0.5 server a lane
    if tag_dir.exists():
        raise FleetError(f"{tag_dir} exists: a TAG is read once")
    caps = [(name, lanes or hosts[name]["lanes"]) for name, lanes in args.hosts]
    split = split_draws(args.draws, caps)
    used = [name for name, _ in caps if split[name]]

    def prepare(name):
        host = hosts[name]
        if args.pi05:  # before the long checkpoint copy
            venv = f"{host['worktree']}/lerobot_stack/.venv-pi/bin/python"
            check(sh(host, f"test -x {q(venv)}"), f"{name}: no pi0.5 venv {venv}")
        if host["ssh"]:
            ship(LOCAL, args.ckpt, host, args.ckpt)
            ship(LOCAL, ledger, host, ledger)
        ensure_marker(host, args.ckpt)
        fresh = f"test ! -e {q(f'{tag_dir}/hosts/{name}')}"  # else its supervisor reuses payloads
        script = f"{fresh} && git -C {q(host['worktree'])} rev-parse HEAD"
        return check(sh(host, script), f"{name}: an old lane dir, or no worktree").strip()

    shas = dict(zip(used, parallel(prepare, used)))
    if len(set(shas.values())) != 1:
        raise FleetError(f"the hosts are not on one SHA: {shas}")
    tag_dir.mkdir(parents=True)

    port, lanes, local = args.port_base, {}, []
    for name, cap in caps:
        host, out = hosts[name], f"{tag_dir}/hosts/{name}"
        launches = ["set -e", f"mkdir -p {q(str(tag_dir / 'hosts'))}", f"mkdir {q(out)}"]
        for job, lane in host_jobs(args, base, host, name, split[name], cap, port, out):
            log = str(args.campaign / "jobs" / f"{job}.log")
            script = job_script(
                host,
                "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "
                + shlex.join(["python3", "scripts/run_gate_eval.py", *lane]),
            )
            lanes[job], port = name, port + 1
            if host["ssh"]:
                launches.append(detach(log, script))
            else:
                os.makedirs(out, exist_ok=True)
                local.append(start_local(log, script))
        if host["ssh"] and split[name]:
            check(sh(host, "\n".join(launches), timeout=120), f"launch on {name}")
        print(
            f"[read] {name}: draws {split[name]} on {min(cap, len(split[name]))} lanes", flush=True
        )

    done: dict[str, str] = {}
    while len(done) < len(lanes):
        time.sleep(args.poll_s)
        for name in used:
            try:
                tokens = job_tokens(hosts[name], args.campaign)
            except (FleetError, subprocess.TimeoutExpired) as exc:
                print(f"[read] {name}: poll failed, retrying: {str(exc).splitlines()[0]}")
                continue
            for job, host_name in lanes.items():
                token = tokens.get(job, "RUNNING")
                if host_name == name and job not in done and token != "RUNNING":
                    done[job] = token
                    print(f"[read] {job}: {token[:120]}", flush=True)
    for proc in local:
        proc.wait()
    bad = {job: t for job, t in done.items() if not t.startswith(("GATE_OK", "GATE_FAIL"))}
    if bad:
        raise FleetError(f"lanes ended without a gate token: {bad}")

    for name in used:
        src = rsync_url(hosts[name], f"{tag_dir}/hosts/{name}/")
        pull = RSYNC + ["--include=eval_[0-9][0-9][0-9].json", "--exclude=*", src, f"{tag_dir}/"]
        check(subprocess.run(pull, capture_output=True, text=True, check=False), f"pull {name}")
    payloads = fresh_payloads(tag_dir, args.draws, start)
    for name in used:
        eps = [e for d in split[name] for e in json.loads(payloads[d].read_text())["episodes"]]
        stowed, total = sum(e["bricks_stowed"] for e in eps), sum(e["num_bricks"] for e in eps)
        print(
            f"[read] {name}: draws {split[name]}, device {hosts[name]['device']}, "
            f"sha {shas[name][:12]}, stowed {stowed}/{total}",
            flush=True,
        )
    record = {"sha": shas[used[0]], "start": start, "split": split, "supervisor": aggregate}
    (tag_dir / "fleet.json").write_text(json.dumps(record, indent=2))

    def refuse(*_):
        raise FleetError("the aggregate pass would re-run a draw: a payload fails the cache rules")

    results = ge.run_gate(cfg, run_eval=refuse, start_server=refuse)  # prints the GATE_ token
    return None, 0 if results["gate"]["passed"] else 1


# --- main ---------------------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise FleetError(f"{self.prog}: error: {message}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Everything after the first -- is run's command or read's extra supervisor flags."""
    argv, rest = (
        (argv[: argv.index("--")], argv[argv.index("--") + 1 :]) if "--" in argv else (argv, [])
    )
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--sha", required=True, type=lambda s: re.fullmatch("[0-9a-f]{40}", s)[0])
    p_sync.add_argument("--hosts", help="name,name (default: every host in hosts.json)")
    p_sync.add_argument("--force", action="store_true", help="move a worktree under RUNNING jobs")
    p_ship = sub.add_parser("ship")
    p_ship.add_argument("src", help="PATH or HOST:PATH")
    p_ship.add_argument("dst", help="PATH or HOST:PATH")
    p_ship.add_argument("--streams", type=int, default=1, help="parallel rsyncs, disjoint files")
    p_run = sub.add_parser("run")
    p_run.add_argument("host")
    p_run.add_argument("name", type=lambda s: re.fullmatch(r"[\w.-]+", s)[0])
    sub.add_parser("status")
    p_read = sub.add_parser("read")
    p_read.add_argument("--ckpt", required=True, help="pretrained_model dir, absolute")
    p_read.add_argument(
        "--pi05",
        action="store_true",
        help="MS7h design 7.3: --ckpt is a pi0.5 checkpoint, one shared server a host",
    )
    p_read.add_argument("--pi05_num_steps", type=int, help="with --pi05: flow steps a forward")
    p_read.add_argument("--pi05_noise_scale", type=float, help="with --pi05: 0 = zero noise")
    p_read.add_argument("--pi05_jaw_noise_scale", type=float, help="with --pi05: the jaw's")
    p_read.add_argument("--pi05_jaw_noise_shared", action="store_true", help="with --pi05")
    p_read.add_argument("--pi05_jaw_noise_abs", type=float, help="with --pi05: the jaw's std")
    p_read.add_argument("--pi05_still_noise_scale", type=float, help="with --pi05: still rule")
    p_read.add_argument("--pi05_still_rad", type=float, help="with --pi05_still_noise_scale")
    p_read.add_argument("--pi05_still_open", type=float, help="with --pi05_still_noise_scale")
    p_read.add_argument(
        "--pi05_jaw_latch", type=int, help="with --pi05: actions held after a close"
    )
    p_read.add_argument("--pi05_retry_noise_scale", type=float, help="with --pi05: retry arm noise")
    p_read.add_argument("--tag", required=True, type=lambda s: re.fullmatch(r"[\w.-]+", s)[0])
    p_read.add_argument("--seed", type=int, required=True)
    p_read.add_argument("--draws", type=parse_draws, required=True, help="e.g. 0-11")
    p_read.add_argument("--hosts", type=parse_hosts, required=True, help="name[:lanes],...")
    p_read.add_argument("--threshold", type=float, default=0.85)
    p_read.add_argument("--n_steps", type=int, default=50)
    p_read.add_argument("--split", choices=("train", "heldout", "all"), default="heldout")
    p_read.add_argument(
        "--client_flags",
        default="--select_at_release --replan_on_switch --squeeze_floor --square_wrist",
    )
    p_read.add_argument("--ledger", type=Path, default=None, help="default <campaign>/ledger.json")
    p_read.add_argument("--port_base", type=int, default=7100)
    p_read.add_argument("--poll_s", type=float, default=60.0)
    for p in (p_sync, p_ship, p_run, p_read, sub.choices["status"]):
        p.add_argument("--campaign", type=Path, default=Path.home() / "data/maker-arm-ms7")
    args = parser.parse_args(argv)
    args.cmd = args.extra = rest
    if args.command == "run" and not rest:
        parser.error("run needs a command after --")
    if args.command == "read":
        args.ckpt = args.ckpt.rstrip("/")
        if not os.path.isabs(args.ckpt):
            parser.error("--ckpt must be absolute: a payload records the path as its policy id")
    return args


COMMANDS = {
    "sync": cmd_sync,
    "ship": cmd_ship,
    "run": cmd_run,
    "read": cmd_read,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fail = FAIL.get(next((a for a in argv if a in FAIL), None), "FLEET_FAIL")
    try:
        args = _parse_args(argv)
        hosts = json.loads((args.campaign / "hosts.json").read_text())
        token, code = COMMANDS[args.command](args, hosts)
    except SystemExit as exc:  # --help, or the gate supervisor's own usage error
        token, code = fail, exc.code if isinstance(exc.code, int) else 2
    except FleetError as exc:
        print(f"[fleet] {exc}", file=sys.stderr, flush=True)
        token, code = fail, 2
    except BaseException:  # noqa: BLE001 -- the token contract holds on a crash too
        traceback.print_exc()
        token, code = fail, 2
    if token:
        print(token, flush=True)
    return code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

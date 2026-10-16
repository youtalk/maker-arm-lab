"""One powered session of the real arm (MS5 control plan task 6): open the arm under the rig's
reduced caps, run a plan of holds, trajectories and operator prompts, and log every 200 Hz tick
to HDF5 (``real.telemetry``).

    source ~/opt/env.sh  # the Thor's venv
    PATH=$HOME/.cargo/bin:$PATH .venv/bin/python scripts/real_session.py \\
        --rig ~/data/maker-arm-ms5/rig.json --plan ~/data/maker-arm-ms5/plans/hold60.json \\
        --log ~/data/maker-arm-ms5/logs/s2_hold60.h5

``--interface sim`` runs the SDK's simulator; the default is the rig's ``can_interface``. The
rig's ``caps`` and ``calibration`` make the profile (``real.rig.reduced_profile``), its
``tracking_params`` the controller's parameters (``--params PARAMS.json`` overrides keys), its
``rt`` the loop's SCHED_FIFO priority and core (not on the simulator). ``--dynamics urdf`` is
the controller's model. The plan format is ``real.session``'s docstring. Every check that can
fail (the rig, the profile, the plan, each trajectory's ``check_traj``, the log path) runs
before the arm is enabled; the SDK checks the profile again when it connects. The script never
writes the rig JSON and never overwrites a log.

It prints exactly one ``SESSION_OK {segments, ticks, dropped, faults, ...}`` or
``SESSION_FAIL {reason, detail}`` line on every exit path and exits through ``os._exit``. A
failure (a fault, a lost controller, Ctrl-C, an exception) holds the arm first and drains the
last ticks into the log. With the token printed, the motors are disabled only on the
operator's typed ``RELEASE`` or end of input (``--release prompt``, the default) or at once
(``--release now``, for tests): a release makes the arm go limp. After a finished plan the
controller holds the last target until the release. A person stands at the e-stop for every
powered session.
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from maker_arm_lab.real import session as real_session

DYNAMICS = ("urdf",)


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("SESSION_FAIL " + json.dumps({"reason": "usage", "detail": message}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--rig", required=True, help="rig JSON (real.rig.RigConfig)")
    parser.add_argument("--plan", required=True, help="session plan JSON (real.session)")
    parser.add_argument("--log", required=True, help="the HDF5 session log to create")
    parser.add_argument(
        "--interface", default="", help="'sim' for the SDK simulator; default the rig's CAN"
    )
    parser.add_argument("--dynamics", choices=DYNAMICS, default="urdf")
    parser.add_argument("--params", default="", help="JSON of tracking parameters to override")
    parser.add_argument("--release", choices=("prompt", "now"), default="prompt")
    return parser


def lab_revision() -> str:
    """HEAD of this checkout, with ``-dirty`` for uncommitted changes to tracked files."""
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def prepare(args) -> dict:
    """Everything the session needs short of the arm, checked. Raises SessionFailure."""
    import maker_arm_rs

    from maker_arm_lab.core import dynamics_model as dm
    from maker_arm_lab.real.rig import RigConfig, reduced_profile
    from maker_arm_lab.real.telemetry import sdk_revision

    fail = real_session.SessionFailure
    rig_path = Path(args.rig).expanduser()
    try:
        rig = RigConfig.from_json(rig_path)
    except (OSError, ValueError, TypeError) as err:
        raise fail("rig", f"{rig_path}: {err}") from err
    try:
        profile = reduced_profile(rig)
    except (TypeError, ValueError) as err:
        raise fail("profile", str(err)) from err
    plan_path = Path(args.plan).expanduser()
    try:
        segments = real_session.load_plan(plan_path, profile)
    except (OSError, ValueError) as err:
        raise fail("plan", f"{plan_path}: {err}") from err
    params = dict(rig.tracking_params)
    if args.params:
        try:
            override = json.loads(Path(args.params).expanduser().read_text())
        except (OSError, ValueError) as err:
            raise fail("params", f"{args.params}: {err}") from err
        if not isinstance(override, dict):
            raise fail("params", f"{args.params}: the file must hold an object")
        params.update(override)
    dynamics = dm.dynamics(dm.urdf_bodies())
    try:
        # A throwaway controller: the SDK checks the parameters when it builds one.
        maker_arm_rs.Tracking(dynamics, params=params, profile=profile)
    except (TypeError, ValueError) as err:
        raise fail("params", str(err)) from err
    log_path = Path(args.log).expanduser()
    if log_path.exists():
        raise fail("log_exists", f"{log_path} exists; a session log is never overwritten")
    interface = args.interface or rig.can_interface
    rt = None if interface == "sim" or rig.rt is None else (int(rig.rt[0]), int(rig.rt[1]))
    attrs = {
        "rig": asdict(rig),
        "rig_path": str(rig_path),
        "plan": json.loads(plan_path.read_text()),
        "plan_path": str(plan_path),
        "profile": profile,
        "tracking_params": params,
        "dynamics": args.dynamics,
        "interface": interface,
        "rt": rt,
        "control_hz": real_session.CONTROL_HZ,
        "lab_revision": lab_revision(),
        "sdk_revision": sdk_revision(),
        "started": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    return {
        "profile": profile,
        "segments": segments,
        "params": params,
        "dynamics": dynamics,
        "interface": interface,
        "rt": rt,
        "log_path": log_path,
        "attrs": attrs,
    }


def connect_arm(interface: str, dynamics, profile: dict, params: dict, rt):
    """The SDK arm, connected and not enabled. The SDK refuses a loosened profile here with
    ValueError, and a bad parameter set when it builds the controller."""
    from maker_arm_lab.real.arm import SdkArm

    return SdkArm.connect(interface, dynamics, profile=profile, params=params, rt=rt)


def hold(world: dict) -> None:
    """Hold an enabled arm at its current pose, then drain its last ticks into the log."""
    sdk = world.get("arm")
    if sdk is None or not world.get("enabled"):
        return
    try:
        sdk.hold()
    except Exception:  # noqa: BLE001 -- no loop (a failed start): start the SDK's hold instead
        traceback.print_exc()
        try:
            sdk.arm.start_hold()
        except Exception:  # noqa: BLE001 -- nothing more can be done from here
            traceback.print_exc()
    log = world.get("log")
    if log is not None:
        try:
            real_session.drain_into(sdk, log)
        except Exception:  # noqa: BLE001 -- the log keeps every earlier drain
            traceback.print_exc()


def release(world: dict, mode: str, lines) -> None:
    """Disable the motors: at once, or on the word RELEASE typed after this prompt (anything
    typed before it is dropped) or the end of input. A Ctrl-C
    at the prompt keeps the arm held and asks again. An arm never enabled is let go at once."""
    sdk = world.get("arm")
    if sdk is None:
        return
    if mode == "prompt" and world.get("enabled"):
        lines.clear()  # a RELEASE typed during the session must not release at its end
        print(
            "The arm holds. Support it, then type RELEASE to disable the motors.",
            file=sys.stderr,
            flush=True,
        )
        while True:
            try:
                line = lines.get(0.5)
            except EOFError:
                break
            except KeyboardInterrupt:
                print("Ctrl-C: the arm still holds. Type RELEASE.", file=sys.stderr, flush=True)
                continue
            if line is not None and line.strip() == "RELEASE":
                break
    try:
        sdk.stop()
    except Exception:  # noqa: BLE001 -- the token is already out
        traceback.print_exc()


def main(argv=None) -> int:
    """Run one session. Ctrl-C ends the session (``interrupted``): during the setup by an
    exception, once the log exists by a flag the run polls, so no log write is cut short. From
    then on, through the
    hold, the token and the release prompt, a Ctrl-C only says the arm still holds: an
    escaping ``KeyboardInterrupt`` would skip the token and ``os._exit``, and dropping the SDK
    arm at interpreter exit disables the motors without the typed ``RELEASE``."""
    args = build_parser().parse_args(argv)
    world: dict = {}
    phase = ["setup"]

    def on_sigint(signum, frame):
        if phase[0] == "setup":
            phase[0] = "closing"
            raise KeyboardInterrupt
        if phase[0] == "session":
            # A flag, not an exception: an exception could cut a log append between two
            # columns. The run polls the flag every 20 ms and fails with "interrupted".
            phase[0] = "closing"
            world["interrupted"] = True
            return
        note = "the arm still holds; type RELEASE to disable the motors"
        # os.write, not print: a SIGINT inside a sys.stderr write would re-enter its buffer.
        text = f"Ctrl-C: {note if world.get('enabled') else 'the session is closing'}.\n"
        os.write(2, text.encode())

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        return _run(args, world, phase)
    finally:
        signal.signal(signal.SIGINT, previous)


def _run(args, world: dict, phase: list) -> int:
    lines = None
    code, token = 1, None
    try:
        try:
            from maker_arm_lab.real.telemetry import TelemetryLog

            lines = real_session.Lines().start()
            ready = prepare(args)
            try:
                world["arm"] = connect_arm(
                    ready["interface"], ready["dynamics"], ready["profile"], ready["params"],
                    ready["rt"],
                )  # fmt: skip
            except ValueError as err:
                raise real_session.SessionFailure("profile", str(err)) from err
            # From here a Ctrl-C only sets a flag: it must not cut the log's creation or an
            # append. The run's first poll, right after the start, fails with "interrupted".
            phase[0] = "session"
            world["log"] = TelemetryLog(ready["log_path"], ready["attrs"])
            world["enabled"] = True
            world["arm"].start()
            summary = real_session.run_session(
                world["arm"],
                ready["segments"],
                world["log"],
                lines,
                interrupted=lambda: world.get("interrupted", False),
            )
            summary["log"] = str(ready["log_path"])
            token, code = "SESSION_OK " + json.dumps(summary), 0
        finally:
            phase[0] = "closing"
    # Each branch sets code = 1: a Ctrl-C can land after the SESSION_OK assignment, before the
    # inner finally, and the exit code must still agree with the token.
    except real_session.SessionFailure as err:
        token = "SESSION_FAIL " + json.dumps({"reason": err.reason, "detail": err.detail})
        code = 1
    except KeyboardInterrupt:
        token = "SESSION_FAIL " + json.dumps({"reason": "interrupted", "detail": "Ctrl-C"})
        code = 1
    except Exception as err:  # noqa: BLE001 -- any uncaught error must still print a token
        detail = f"{err}\n{traceback.format_exc()}"
        token, code = "SESSION_FAIL " + json.dumps({"reason": "exception", "detail": detail}), 1
    if code != 0:
        hold(world)
    log = world.get("log")
    if log is not None:
        try:
            log.close()
        except Exception as err:  # noqa: BLE001 -- the session ran; still one token
            if code == 0:
                token = "SESSION_FAIL " + json.dumps({"reason": "write_log", "detail": str(err)})
                code = 1
            traceback.print_exc()
    print(token, flush=True)
    release(world, args.release, lines if lines is not None else real_session.Lines())
    return code


if __name__ == "__main__":
    os._exit(main())

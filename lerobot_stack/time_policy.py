"""MS5 design P1: one policy forward on this host, timed through the policy server.

Connects to a running ``policy_server.py`` and sends ``--calls`` steps of one env with random
images, state and context, each with the replan flag set, so every call runs a full forward
(the server's ``ActionQueues`` refills a flagged env; the temporal-ensembling path runs one
every step anyway). The server reports its own time in the policy call (``policy_ms``) and the
client the round trip. The bar is the forward's p99. MS5 design P1 pre-registers 50 ms for 50
actions, two 224x224 images and the 16-wide context, measured with the loop and the arm loop
running on the same host.

numpy and pyzmq only, like ``policy_client``. Prints ``TIMING_OK`` or ``TIMING_FAIL`` with a
JSON summary on every exit path and exits through ``os._exit``.

    lerobot_stack/.venv/bin/python lerobot_stack/time_policy.py \\
        --server tcp://127.0.0.1:5555 --calls 500 --bar_ms 50
"""

import argparse
import json
import os
import traceback

import numpy as np


class TokenArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("TIMING_FAIL " + json.dumps({"reason": f"usage: {message}"}), flush=True)
        os._exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser(description=__doc__)
    parser.add_argument("--server", default="tcp://127.0.0.1:5555")
    parser.add_argument("--calls", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--bar_ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _stats(values) -> dict:
    xs = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(xs, 50)), 2),
        "p99": round(float(np.percentile(xs, 99)), 2),
        "max": round(float(xs.max()), 2),
    }


def summarize(policy_ms, rtt_ms, bar_ms: float, warmup: int) -> dict:
    """The bar read over the calls after ``warmup``: passes when the forward's p99 is at most
    ``bar_ms``."""
    policy, rtt = list(policy_ms)[warmup:], list(rtt_ms)[warmup:]
    if not policy:
        raise ValueError("no timed calls after the warm-up")
    forward = _stats(policy)
    return {
        "pass": forward["p99"] <= bar_ms,
        "bar_ms": bar_ms,
        "calls": len(policy),
        "warmup": warmup,
        "forward_ms": forward,
        "rtt_ms": _stats(rtt),
    }


def run(server: str, calls: int, warmup: int, seed: int):
    """``calls + warmup`` forced forwards; returns the server times, the round trips and the
    closed client (its handshake fields stay readable)."""
    from policy_client import ServerPolicy

    client = ServerPolicy(server, recv_timeout_ms=30_000)
    rng = np.random.default_rng(seed)
    policy_ms, rtt_ms = [], []
    try:
        for _ in range(calls + warmup):
            state = rng.uniform(-1.0, 1.0, (1, 7)).astype(np.float32)
            state[:, 6] = rng.uniform(0.0, 1.0)
            wrist = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
            overhead = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
            width = client.context_dim
            context = rng.normal(size=(1, width)).astype(np.float32) if width else None
            client.step(state, wrist, overhead, context=context, replan=np.ones(1, dtype=bool))
            policy_ms.append(float(client.last_policy_ms))
            rtt_ms.append(float(client.last_rtt_ms))
    finally:
        client.close()
    return policy_ms, rtt_ms, client


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.calls < 1 or args.warmup < 0:
            raise ValueError("--calls must be >= 1 and --warmup >= 0")
        policy_ms, rtt_ms, client = run(args.server, args.calls, args.warmup, args.seed)
        summary = summarize(policy_ms, rtt_ms, args.bar_ms, args.warmup)
        summary.update(
            policy_id=client.policy_id,
            n_action_steps=client.n_action_steps,
            inference=client.inference,
            context_dim=client.context_dim,
        )
        token = "TIMING_OK " if summary["pass"] else "TIMING_FAIL "
        print(token + json.dumps(summary), flush=True)
        return 0 if summary["pass"] else 1
    except Exception as exc:  # noqa: BLE001 -- every failure must still print a token
        print("TIMING_FAIL " + json.dumps({"reason": f"{type(exc).__name__}: {exc}"}), flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    os._exit(main())

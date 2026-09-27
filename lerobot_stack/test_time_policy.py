"""MS5 design P1: the forward timing bench against a live PolicyServer."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

import time_policy
from policy_protocol import ACTION_SHAPE, recv_message, send_message
from policy_server import PolicyServer

SCRIPT = Path(__file__).resolve().parent / "time_policy.py"


def test_summarize_passes_at_the_bar_and_drops_the_warmup():
    policy = [100.0] * 5 + [10.0] * 100
    out = time_policy.summarize(policy, [v + 1.0 for v in policy], bar_ms=50.0, warmup=5)
    assert out["pass"] and out["calls"] == 100 and out["forward_ms"]["p99"] == 10.0
    assert time_policy.summarize(policy, policy, bar_ms=50.0, warmup=0)["pass"] is False


def test_summarize_refuses_an_empty_read():
    with pytest.raises(ValueError):
        time_policy.summarize([1.0], [1.0], bar_ms=50.0, warmup=1)


def test_every_call_is_a_forward_against_a_live_server():
    calls = {"replan": 0}

    def fake_policy(state, wrist, overhead, context=None, replan=None):
        calls["replan"] += int(replan is not None and bool(replan.all()))
        time.sleep(0.002)
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    srv = PolicyServer(fake_policy, lambda: None, "fake:bench", 0, 50, context_dim=16)
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    endpoint = f"tcp://127.0.0.1:{port}"
    try:
        policy_ms, _rtt_ms, client = time_policy.run(endpoint, 12, 2, seed=1)
    finally:
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 5000)
        req.connect(endpoint)
        send_message(req, {"type": "shutdown"})
        recv_message(req)
        req.close(0)
        thread.join(timeout=5)
    assert calls["replan"] == 14 and len(policy_ms) == 14
    assert min(policy_ms) >= 2.0 and client.context_dim == 16


def test_a_usage_error_prints_the_fail_token():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--calls", "many"],
        capture_output=True,
        text=True,
        check=False,
    )
    last = out.stdout.strip().splitlines()[-1]
    assert out.returncode == 2 and last.startswith("TIMING_FAIL ") and "usage" in last

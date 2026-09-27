"""MS5 design P1: the forward timing bench against a live PolicyServer."""

import json
import signal
import socket
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


def _live_server(context_dim: int, n_action_steps: int):
    def fake_policy(state, wrist, overhead, context=None, replan=None):
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    srv = PolicyServer(
        fake_policy, lambda: None, "fake:shape", 0, n_action_steps, context_dim=context_dim
    )
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    endpoint = f"tcp://127.0.0.1:{port}"

    def shutdown():
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 5000)
        req.connect(endpoint)
        send_message(req, {"type": "shutdown"})
        recv_message(req)
        req.close(0)
        thread.join(timeout=5)

    return endpoint, shutdown


def test_shape_errors_name_each_miss_and_require_nothing_by_default():
    assert time_policy.shape_errors(0, 1) == []
    assert time_policy.shape_errors(16, 50, 16, 50) == []
    errors = time_policy.shape_errors(0, 1, 16, 50)
    assert errors == ["context_dim 0 != 16", "n_action_steps 1 != 50"]
    assert time_policy.shape_errors(16, 1, 16, None) == []


def test_a_server_off_the_required_shape_reads_timing_fail(capsys):
    endpoint, shutdown = _live_server(context_dim=0, n_action_steps=50)
    try:
        code = time_policy.main(
            ["--server", endpoint, "--calls", "2", "--warmup", "0", "--require_context", "16"]
        )
    finally:
        shutdown()
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 1 and last.startswith("TIMING_FAIL ")
    assert "context_dim 0 != 16" in json.loads(last.split(" ", 1)[1])["reason"]


def test_a_server_on_the_required_shape_is_timed(capsys):
    endpoint, shutdown = _live_server(context_dim=16, n_action_steps=50)
    try:
        code = time_policy.main(
            [
                "--server",
                endpoint,
                "--calls",
                "3",
                "--warmup",
                "1",
                "--require_context",
                "16",
                "--require_n_action_steps",
                "50",
            ]
        )
    finally:
        shutdown()
    out = capsys.readouterr().out.strip().splitlines()
    tokens = [line for line in out if line.startswith("TIMING_")]
    assert code == 0 and len(tokens) == 1 and tokens[0].startswith("TIMING_OK ")
    summary = json.loads(tokens[0].split(" ", 1)[1])
    assert summary["context_dim"] == 16 and summary["n_action_steps"] == 50


def test_a_keyboard_interrupt_prints_one_fail_token(monkeypatch, capsys):
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(time_policy, "run", interrupted)
    code = time_policy.main(["--calls", "1"])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 130 and out == ['TIMING_FAIL {"reason": "interrupted"}']


def test_a_sigint_while_waiting_on_the_server_prints_one_fail_token():
    with socket.socket() as sock:  # a free port that nothing serves
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--server", f"tcp://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2.0)  # past the imports, blocked in the handshake's receive
    proc.send_signal(signal.SIGINT)
    stdout, _ = proc.communicate(timeout=20)
    tokens = [line for line in stdout.splitlines() if line.startswith("TIMING_")]
    assert tokens == ['TIMING_FAIL {"reason": "interrupted"}'] and proc.returncode == 130


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

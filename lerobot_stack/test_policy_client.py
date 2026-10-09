"""The eval client against the real PolicyServer, over a loopback socket."""

import threading
import time

import numpy as np
import pytest
import zmq

from policy_client import ServerPolicy
from policy_protocol import ACTION_SHAPE, OBS_SCHEMA, recv_message, send_message
from policy_server import PolicyServer


def _obs(batch: int):
    state = np.zeros((batch, *OBS_SCHEMA["state"][0]), dtype="float32")
    state[:, 0] = np.arange(batch, dtype="float32")
    state[:, 6] = 0.5
    wrist = np.zeros((batch, *OBS_SCHEMA["wrist"][0]), dtype="uint8")
    overhead = np.zeros((batch, *OBS_SCHEMA["overhead"][0]), dtype="uint8")
    return state, wrist, overhead


def _serve(**kwargs):
    """Bind and start a PolicyServer thread; return (endpoint, shutdown) for a fixture to yield
    from and tear down with. Factored out so every server-backed fixture in this file shares
    one construction and one shutdown block instead of copying both.
    """
    srv = PolicyServer(port=0, n_action_steps=1, **kwargs)
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    def shutdown():
        req = zmq.Context.instance().socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, 10_000)
        req.connect(f"tcp://127.0.0.1:{port}")
        send_message(req, {"type": "shutdown"})
        recv_message(req)
        thread.join(timeout=5)
        req.close(0)

    return f"tcp://127.0.0.1:{port}", shutdown


@pytest.fixture()
def running_server(request):
    """Serves a fake policy; parametrize indirectly with a context width to advertise one."""
    context_dim = getattr(request, "param", 0)
    calls = {"reset": 0, "steps": 0, "context_shape": None}

    def fake_policy(state, wrist, overhead, context=None):
        calls["steps"] += 1
        calls["context_shape"] = None if context is None else context.shape
        action = np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")
        action[:, 0] = state[:, 0] + 1.0
        return action

    endpoint, shutdown = _serve(
        policy_fn=fake_policy,
        reset_fn=lambda: calls.__setitem__("reset", calls["reset"] + 1),
        policy_id="fake:client-test",
        inference={"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "chunk_size": 100},
        context_dim=context_dim,
    )
    yield endpoint, calls
    shutdown()


@pytest.fixture()
def delta_prev_server():
    def fake_policy(state, wrist, overhead, context=None):
        action = np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")
        action[:, 0] = 0.25
        return action

    endpoint, shutdown = _serve(
        policy_fn=fake_policy,
        reset_fn=lambda: None,
        policy_id="fake:dprev",
        action_space="delta_prev",
    )
    yield endpoint
    shutdown()


@pytest.fixture()
def continuous_server():
    def fake_policy(state, wrist, overhead, context=None):
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    endpoint, shutdown = _serve(
        policy_fn=fake_policy,
        reset_fn=lambda: None,
        policy_id="fake:gap",
        gripper="continuous",
    )
    yield endpoint
    shutdown()


def test_client_reads_the_gripper_contract(continuous_server, running_server):
    client = ServerPolicy(continuous_server)
    assert client.gripper == "continuous"
    client.close()
    endpoint, _ = running_server
    legacy = ServerPolicy(endpoint)
    assert legacy.gripper == "binary"
    legacy.close()


def test_client_handshakes_resets_and_records_the_served_inference(running_server):
    endpoint, calls = running_server
    client = ServerPolicy(endpoint)
    assert client.policy_id == "fake:client-test"
    assert client.n_action_steps == 1
    assert client.inference == {
        "n_action_steps": 1,
        "temporal_ensemble_coeff": 0.01,
        "chunk_size": 100,
    }
    assert calls["reset"] == 1  # the constructor resets once
    client.close()


def test_client_step_round_trips_a_batch(running_server):
    endpoint, calls = running_server
    client = ServerPolicy(endpoint)
    action = client.step(*_obs(3))
    assert action.shape == (3, 7) and action.dtype == np.float32
    assert action[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert calls["steps"] == 1
    client.close()


@pytest.mark.parametrize("running_server", [6], indirect=True)
def test_client_sends_a_context_when_given_one(running_server):
    endpoint, calls = running_server
    client = ServerPolicy(endpoint)
    assert client.context_dim == 6
    action = client.step(*_obs(2), context=np.full((2, 6), 0.5, dtype="float32"))
    assert calls["context_shape"] == (2, 6)
    assert action.shape == (2, 7)
    client.close()


def test_client_reports_a_server_side_validation_error(running_server):
    endpoint, _ = running_server
    client = ServerPolicy(endpoint)
    state, wrist, overhead = _obs(2)
    state[:, 6] = 2.0  # not an opening fraction
    with pytest.raises(RuntimeError, match="server error"):
        client.step(state, wrist, overhead)
    client.close()


def test_client_reads_the_action_space_and_sends_an_anchor(delta_prev_server):
    client = ServerPolicy(delta_prev_server)
    assert client.action_space == "delta_prev"
    anchor = np.full((2, 6), 2.0, dtype="float32")
    action = client.step(*_obs(2), anchor=anchor)
    assert action[:, 0].tolist() == [2.25, 2.25]
    with pytest.raises(RuntimeError, match="server error"):
        client.step(*_obs(2))
    client.close()


def test_client_sends_the_replan_flag_only_when_given():
    seen = []

    def fake_policy(state, wrist, overhead, context=None, replan=None):
        seen.append(None if replan is None else replan.tolist())
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    endpoint, shutdown = _serve(policy_fn=fake_policy, reset_fn=lambda: None, policy_id="fake:r")
    client = ServerPolicy(endpoint)
    client.step(*_obs(2))
    client.step(*_obs(2), replan=np.array([True, False]))
    client.close()
    shutdown()
    assert seen == [None, [True, False]]


def test_a_step_records_the_servers_policy_time_and_the_round_trip():
    def slow_policy(state, wrist, overhead, context=None):
        time.sleep(0.01)
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    endpoint, shutdown = _serve(policy_fn=slow_policy, reset_fn=lambda: None, policy_id="fake:slow")
    try:
        client = ServerPolicy(endpoint, recv_timeout_ms=10_000)
        assert client.last_policy_ms is None and client.last_rtt_ms is None
        client.step(*_obs(1))
        assert client.last_policy_ms >= 10.0
        assert client.last_rtt_ms >= client.last_policy_ms
        client.close()
    finally:
        shutdown()


def test_an_error_reply_clears_the_last_policy_time():
    def policy(state, wrist, overhead, context=None):
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    endpoint, shutdown = _serve(policy_fn=policy, reset_fn=lambda: None, policy_id="fake:err")
    try:
        client = ServerPolicy(endpoint, recv_timeout_ms=10_000)
        client.step(*_obs(1))
        assert client.last_policy_ms is not None
        state, wrist, overhead = _obs(1)
        state[:, 6] = 2.0  # not an opening fraction: the server replies with an error
        with pytest.raises(RuntimeError, match="server error"):
            client.step(state, wrist, overhead)
        assert client.last_policy_ms is None and client.last_rtt_ms is not None
        client.close()
    finally:
        shutdown()


def test_client_sends_the_retry_flag_only_when_given(monkeypatch):
    """The retry flag rides on the wire when given, and the ACT server ignores it."""
    import policy_client
    import policy_protocol

    sent, real = [], policy_protocol.send_message

    def send(socket, header, arrays=None):
        if header["type"] == "step":
            retry = (arrays or {}).get(policy_protocol.RETRY_KEY)
            sent.append(None if retry is None else retry.tolist())
        real(socket, header, arrays)

    monkeypatch.setattr(policy_client.policy_protocol, "send_message", send)
    seen = []

    def fake_policy(state, wrist, overhead, context=None, replan=None):
        seen.append(None if replan is None else replan.tolist())
        return np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")

    endpoint, shutdown = _serve(policy_fn=fake_policy, reset_fn=lambda: None, policy_id="fake:r")
    client = ServerPolicy(endpoint)
    client.step(*_obs(2))
    client.step(*_obs(2), retry=np.array([False, True]))
    client.step(*_obs(2), replan=np.array([True, False]), retry=np.array([True, True]))
    client.close()
    shutdown()
    assert sent == [None, [False, True], [True, True]]
    assert seen == [None, None, [True, False]]

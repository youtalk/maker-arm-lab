"""The eval client against the real PolicyServer, over a loopback socket."""

import threading

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

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


@pytest.fixture()
def running_server():
    calls = {"reset": 0, "steps": 0}

    def fake_policy(state, wrist, overhead):
        calls["steps"] += 1
        action = np.zeros((state.shape[0], *ACTION_SHAPE), dtype="float32")
        action[:, 0] = state[:, 0] + 1.0
        return action

    srv = PolicyServer(
        policy_fn=fake_policy,
        reset_fn=lambda: calls.__setitem__("reset", calls["reset"] + 1),
        policy_id="fake:client-test",
        port=0,
        n_action_steps=1,
        inference={"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "chunk_size": 100},
    )
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"tcp://127.0.0.1:{port}", calls
    req = zmq.Context.instance().socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 10_000)
    req.connect(f"tcp://127.0.0.1:{port}")
    send_message(req, {"type": "shutdown"})
    recv_message(req)
    thread.join(timeout=5)
    req.close(0)


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


def test_client_reports_a_server_side_validation_error(running_server):
    endpoint, _ = running_server
    client = ServerPolicy(endpoint)
    state, wrist, overhead = _obs(2)
    state[:, 6] = 2.0  # not an opening fraction
    with pytest.raises(RuntimeError, match="server error"):
        client.step(state, wrist, overhead)
    client.close()

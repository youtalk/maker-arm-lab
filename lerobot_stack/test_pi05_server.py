"""pi05_server.py (the shared pi0.5 server, MS7h design 7.3) with a fake model, CPU only."""

import threading

import numpy as np
import pytest
import zmq

import pi05_server as ps
from derive_vla_dataset import derived_state
from policy_client import ServerPolicy

N = 50
RNG = np.random.default_rng(0)


class FakeModel:
    """chunk[r, t] = (state[r, 0], t, 0, ...): the lane marker and the time index."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, obs):
        self.calls.append(obs)
        if self.fail:
            raise RuntimeError("forward failed")
        chunk = np.zeros((len(obs["state"]), N, 7), np.float32)
        chunk[:, :, 0] = obs["state"][:, :1]
        chunk[:, :, 1] = np.arange(N)
        return chunk


def server(model=None, warp=1.0):
    inference = {"n_action_steps": N, "chunk_size": N, "num_inference_steps": 10}
    return ps.SharedServer(model or FakeModel(), "pi05:/ckpt", N, inference, "continuous", warp)


def step(marker, batch=1, context=True, replan=None, rng=RNG):
    state = rng.uniform(0, 1, (batch, 7)).astype(np.float32)
    state[:, 0] = marker
    arrays = {
        "state": state,
        "wrist": rng.integers(0, 255, (batch, 224, 224, 3), np.uint8),
        "overhead": rng.integers(0, 255, (batch, 224, 224, 3), np.uint8),
    }
    if context:
        ctx = rng.normal(size=(batch, 16)).astype(np.float32)
        ctx[:, 3] = np.pi - 0.01  # the overhead camera's roll sits near +-pi
        arrays["context"] = ctx
    if replan is not None:
        arrays["replan"] = np.asarray(replan, dtype=bool)
    return {"type": "step", "batch": batch}, arrays


def test_the_state_is_23_wide_with_the_roll_wrapped():
    model = FakeModel()
    header, arrays = step(0.5)
    replies, _ = server(model).process([(b"a", header, arrays)])
    assert replies[0][1]["type"] == "action"
    sent = model.calls[0]
    np.testing.assert_array_equal(sent["state"], derived_state(arrays["state"], arrays["context"]))
    assert sent["state"].shape == (1, 23)
    assert abs(sent["state"][0, 10] - (-0.01)) < 1e-6  # wrap(roll - pi)
    np.testing.assert_array_equal(sent["overhead"], arrays["overhead"])


def test_lanes_with_empty_queues_share_one_forward_and_full_ones_wait_for_none():
    model, s = FakeModel(), server()
    s._predict = model
    replies, _ = s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2]
    assert [r[2]["action"][0, 0] for r in replies] == pytest.approx([0.1, 0.2])
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2]  # both queues still hold actions


def test_each_lane_plays_its_warped_chunk_then_refills():
    s = server(warp=1.5)
    times = [s.process([(b"a", *step(0.3))])[0][0][2]["action"][0, 1] for _ in range(34)]
    assert times[:3] == pytest.approx([0.0, 1.5, 3.0])
    assert len(s.forward_sizes) == 2 and times[33] == 0.0  # 50 actions at 1.5 play 33 steps


def test_replan_refills_only_the_flagged_row():
    s = server()
    s.process([(b"a", *step(0.4, batch=2))])
    reply = s.process([(b"a", *step(0.4, batch=2, replan=[False, True]))])[0][0]
    assert s.forward_sizes == [2, 1]
    assert reply[2]["action"][:, 1].tolist() == [1.0, 0.0]


def test_reset_clears_only_its_lane():
    s = server()
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    replies, _ = s.process([(b"a", {"type": "reset"}, {})])
    assert replies[0][1]["type"] == "reset_ack"
    s.process([(b"a", *step(0.1)), (b"b", *step(0.2))])
    assert s.forward_sizes == [2, 1]


def test_a_bad_request_fails_alone_and_a_failed_forward_reaches_its_lanes():
    model = FakeModel()
    s = server(model)
    replies, _ = s.process([(b"a", *step(0.1, context=False)), (b"b", *step(0.2))])
    assert replies[0][1]["error"] == "bad_obs" and "context width 0" in replies[0][1]["message"]
    assert replies[1][1]["type"] == "action"
    model.fail = True
    replies, keep = s.process([(b"c", *step(0.3))])
    assert replies[0][1]["error"] == "exception" and keep


def test_hello_ack_and_shutdown():
    replies, keep = server(warp=1.5).process([(b"a", {"type": "hello", "protocol": 4}, {})])
    ack = replies[0][1]
    assert ack["policy_id"] == "pi05:/ckpt" and ack["context_dim"] == 16
    assert ack["inference"]["action_warp"] == 1.5 and ack["gripper"] == "continuous"
    replies, keep = server().process([(b"a", {"type": "shutdown"}, {})])
    assert replies[0][1]["type"] == "shutdown_ack" and not keep


def test_unchanged_clients_share_the_server_over_a_router_socket():
    s = server(warp=1.5)
    socket = zmq.Context.instance().socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    thread = threading.Thread(target=s.serve, args=(socket,), daemon=True)
    thread.start()
    results = {}

    def lane(marker):
        client = ServerPolicy(f"tcp://127.0.0.1:{port}", recv_timeout_ms=10_000)
        assert client.context_dim == 16 and client.policy_id == "pi05:/ckpt"
        actions = []
        for _ in range(40):
            _, arrays = step(marker)
            actions.append(
                client.step(
                    arrays["state"], arrays["wrist"], arrays["overhead"], context=arrays["context"]
                )
            )
        results[marker] = np.concatenate(actions)

    lanes = [threading.Thread(target=lane, args=(m,)) for m in (0.1, 0.2, 0.3)]
    for t in lanes:
        t.start()
    for t in lanes:
        t.join(timeout=30)
    for marker, actions in results.items():
        assert actions.shape == (40, 7)
        np.testing.assert_allclose(actions[:, 0], marker, rtol=1e-6)  # never another lane's
        assert actions[:3, 1].tolist() == [0.0, 1.5, 3.0]
    assert len(results) == 3 and sum(s.forward_sizes) == 6  # 40 steps = 2 forwards a lane
    stop = zmq.Context.instance().socket(zmq.REQ)
    stop.connect(f"tcp://127.0.0.1:{port}")
    stop.send_multipart(ps.encode_frames({"type": "shutdown"}))
    stop.recv_multipart()
    thread.join(timeout=10)
    assert not thread.is_alive()

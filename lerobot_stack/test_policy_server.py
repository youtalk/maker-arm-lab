import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import zmq

from policy_protocol import PROTOCOL_VERSION, hello_reply, recv_message, send_message
from policy_server import (
    PolicyServer,
    blank_streams,
    find_markers,
    inference_policy_id,
    resolve_inference_overrides,
)


def _obs(batch=2):
    return {
        "state": np.zeros((batch, 7), dtype=np.float32),
        "wrist": np.zeros((batch, 224, 224, 3), dtype=np.uint8),
        "overhead": np.zeros((batch, 224, 224, 3), dtype=np.uint8),
    }


def _start_server(
    policy_fn,
    calls,
    context_dim=0,
    policy_id="fake:test",
    action_space="absolute",
    gripper="binary",
):
    srv = PolicyServer(
        policy_fn=policy_fn,
        reset_fn=lambda: calls.__setitem__("reset", calls["reset"] + 1),
        policy_id=policy_id,
        port=0,
        n_action_steps=100,
        context_dim=context_dim,
        action_space=action_space,
        gripper=gripper,
    )
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 10_000)
    req.connect(f"tcp://127.0.0.1:{port}")
    return req, thread


def _stop_server(req, thread):
    send_message(req, {"type": "shutdown"})
    header, _ = recv_message(req)
    assert header["type"] == "shutdown_ack"
    thread.join(timeout=5)
    assert not thread.is_alive()
    req.close(0)


@pytest.fixture()
def server():
    calls = {"reset": 0}

    def fake_policy(state, wrist, overhead, context=None):
        calls["context"] = context
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, 0] = state[:, 0] + 1.0  # provably passed through
        return action

    req, thread = _start_server(fake_policy, calls, context_dim=0)
    yield req, calls
    _stop_server(req, thread)


@pytest.fixture()
def server_ctx():
    calls = {"reset": 0}

    def fake_policy(state, wrist, overhead, context=None):
        calls["context"] = context
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, 0] = state[:, 0] + 1.0
        return action

    req, thread = _start_server(fake_policy, calls, context_dim=6)
    yield req, calls
    _stop_server(req, thread)


@pytest.fixture()
def server_delta_prev():
    calls = {"reset": 0}

    def fake_policy(state, wrist, overhead, context=None):
        action = np.zeros((state.shape[0], 7), dtype=np.float32)
        action[:, :6] = 0.1
        action[:, 6] = -1.0
        return action

    req, thread = _start_server(
        fake_policy, calls, policy_id="fake:test+dprev", action_space="delta_prev"
    )
    yield req
    _stop_server(req, thread)


def _rpc(req, header, arrays=None):
    send_message(req, header, arrays)
    return recv_message(req)


def test_handshake_reports_schema_and_identity(server):
    req, _ = server
    header, _ = _rpc(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    assert header["type"] == "hello_ack"
    assert header["protocol"] == PROTOCOL_VERSION
    assert header["policy_id"] == "fake:test"
    assert header["n_action_steps"] == 100
    assert header["obs_schema"]["state"] == [[7], "float32"]
    assert header["action"] == [[7], "float32"]


def test_protocol_mismatch_is_an_error_not_a_crash(server):
    req, _ = server
    header, _ = _rpc(req, {"type": "hello", "protocol": 999})
    assert header["type"] == "error" and header["error"] == "protocol_mismatch"
    # server must still answer afterwards
    header, _ = _rpc(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    assert header["type"] == "hello_ack"


def test_step_runs_the_policy(server):
    req, _ = server
    obs = _obs(batch=3)
    obs["state"][:, 0] = 41.0
    header, arrays = _rpc(req, {"type": "step", "batch": 3}, obs)
    assert header["type"] == "action"
    assert arrays["action"].shape == (3, 7) and arrays["action"].dtype == np.float32
    np.testing.assert_allclose(arrays["action"][:, 0], 42.0)


def test_bad_obs_is_reported_with_the_key(server):
    req, _ = server
    obs = _obs()
    obs["wrist"] = obs["wrist"][:, :100]
    header, _ = _rpc(req, {"type": "step", "batch": 2}, obs)
    assert header["type"] == "error" and header["error"] == "bad_obs"
    assert "wrist" in header["message"]


def test_a_context_step_reaches_the_policy_and_a_width_mismatch_is_refused(server_ctx):
    req, calls = server_ctx
    _rpc(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    header, _ = _rpc(
        req, {"type": "step", "batch": 2}, {**_obs(), "context": np.full((2, 6), 0.5, np.float32)}
    )
    assert header["type"] == "action" and calls["context"].shape == (2, 6)
    header, _ = _rpc(
        req, {"type": "step", "batch": 2}, {**_obs(), "context": np.zeros((2, 5), np.float32)}
    )
    assert header["type"] == "error" and header["error"] == "bad_obs"
    header, _ = _rpc(req, {"type": "step", "batch": 2}, _obs())
    assert header["type"] == "error" and header["error"] == "bad_obs"


def test_hello_ack_reports_the_context_width(server_ctx):
    req, _ = server_ctx
    header, _ = _rpc(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    assert header["context_dim"] == 6


def test_a_delta_prev_server_needs_an_anchor_and_adds_it(server_delta_prev):
    req = server_delta_prev
    send_message(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    header, _ = recv_message(req)
    assert header["action_space"] == "delta_prev"
    send_message(req, {"type": "step", "batch": 2}, _obs())
    header, _ = recv_message(req)
    assert (
        header["type"] == "error" and header["error"] == "bad_obs" and "anchor" in header["message"]
    )
    anchor = np.full((2, 6), 1.0, dtype=np.float32)
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "anchor": anchor})
    header, out = recv_message(req)
    assert header["type"] == "action"
    assert np.allclose(out["action"][:, :6], 1.1) and out["action"][:, 6].tolist() == [-1.0, -1.0]


def test_an_absolute_server_ignores_an_anchor(server):
    req, _ = server
    send_message(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
    header, _ = recv_message(req)
    assert header["action_space"] == "absolute"
    anchor = np.full((2, 6), 5.0, dtype=np.float32)
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "anchor": anchor})
    header, out = recv_message(req)
    assert header["type"] == "action" and float(np.abs(out["action"]).max()) < 5.0


def test_nan_action_is_flagged():
    srv = PolicyServer(
        policy_fn=lambda s, w, o, context=None: np.full((s.shape[0], 7), np.nan, dtype=np.float32),
        reset_fn=lambda: None,
        policy_id="nan:test",
        port=0,
        n_action_steps=1,
    )
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 10_000)
    req.connect(f"tcp://127.0.0.1:{port}")
    header, _ = _rpc(req, {"type": "step", "batch": 1}, _obs(batch=1))
    assert header["type"] == "error" and header["error"] == "nan_action"
    header, _ = _rpc(req, {"type": "shutdown"})
    assert header["type"] == "shutdown_ack"
    thread.join(timeout=5)
    assert not thread.is_alive()
    req.close(0)


def test_reset_calls_reset_fn(server):
    req, calls = server
    header, _ = _rpc(req, {"type": "reset"})
    assert header["type"] == "reset_ack" and calls["reset"] == 1


def test_the_action_reply_carries_the_servers_policy_time(server):
    req, _ = server
    header, arrays = _rpc(req, {"type": "step", "batch": 1}, _obs(1))
    assert header["type"] == "action" and arrays["action"].shape == (1, 7)
    assert isinstance(header["policy_ms"], float) and header["policy_ms"] >= 0.0


def test_no_overrides_is_a_noop():
    assert resolve_inference_overrides(None, None, chunk_size=100) == {}


def test_n_action_steps_override_alone():
    assert resolve_inference_overrides(25, None, chunk_size=100) == {"n_action_steps": 25}


def test_temporal_ensembling_forces_single_step():
    # lerobot 0.5.1 requires n_action_steps == 1 with temporal ensembling
    # (ACTConfig.__post_init__ raises otherwise); the helper encodes that
    # contract so the CLI cannot express an invalid pair.
    assert resolve_inference_overrides(None, 0.01, chunk_size=100) == {
        "temporal_ensemble_coeff": 0.01,
        "n_action_steps": 1,
    }
    assert resolve_inference_overrides(1, 0.01, chunk_size=100) == {
        "temporal_ensemble_coeff": 0.01,
        "n_action_steps": 1,
    }


def test_conflicting_ensembling_and_chunked_steps_is_rejected():
    with pytest.raises(ValueError, match="n_action_steps"):
        resolve_inference_overrides(25, 0.01, chunk_size=100)


def test_out_of_range_n_action_steps_is_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        resolve_inference_overrides(101, None, chunk_size=100)
    with pytest.raises(ValueError, match="positive"):
        resolve_inference_overrides(0, None, chunk_size=100)


def test_policy_id_reflects_the_inference_override():
    assert inference_policy_id("/ckpt", {}) == "act:/ckpt"
    assert inference_policy_id("/ckpt", {"n_action_steps": 25}) == "act:/ckpt+nas25"
    assert (
        inference_policy_id("/ckpt", {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1})
        == "act:/ckpt+te0.01"
    )
    assert inference_policy_id("/ckpt", {}, action_warp=1.0) == "act:/ckpt"
    assert inference_policy_id("/ckpt", {"n_action_steps": 50}, action_warp=2.0) == (
        "act:/ckpt+nas50+warp2.0"
    )


def test_blank_streams_without_an_ablation_passes_both_through():
    wrist = np.full((2, 224, 224, 3), 7, dtype=np.uint8)
    overhead = np.full((2, 224, 224, 3), 9, dtype=np.uint8)

    got_wrist, got_overhead = blank_streams(wrist, overhead, None)

    np.testing.assert_array_equal(got_wrist, wrist)
    np.testing.assert_array_equal(got_overhead, overhead)


def test_blank_streams_zeros_only_the_named_stream():
    wrist = np.full((2, 224, 224, 3), 7, dtype=np.uint8)
    overhead = np.full((2, 224, 224, 3), 9, dtype=np.uint8)

    got_wrist, got_overhead = blank_streams(wrist, overhead, "wrist")
    assert not got_wrist.any()
    np.testing.assert_array_equal(got_overhead, overhead)

    got_wrist, got_overhead = blank_streams(wrist, overhead, "overhead")
    np.testing.assert_array_equal(got_wrist, wrist)
    assert not got_overhead.any()


def test_blank_streams_does_not_mutate_the_callers_arrays():
    wrist = np.full((1, 224, 224, 3), 7, dtype=np.uint8)
    overhead = np.full((1, 224, 224, 3), 9, dtype=np.uint8)

    blank_streams(wrist, overhead, "wrist")

    assert wrist.all()  # the caller's buffer is still 7s, not zeroed in place


def test_blank_streams_rejects_an_unknown_stream_name():
    wrist = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="elbow"):
        blank_streams(wrist, wrist, "elbow")


def test_policy_id_records_the_ablation_alongside_the_inference_override():
    assert inference_policy_id("/ckpt", {}, blank_camera="wrist") == "act:/ckpt+blankwrist"
    assert (
        inference_policy_id("/ckpt", {"temporal_ensemble_coeff": 0.01}, blank_camera="overhead")
        == "act:/ckpt+te0.01+blankoverhead"
    )
    # unchanged when no ablation is active
    assert inference_policy_id("/ckpt", {"n_action_steps": 25}) == "act:/ckpt+nas25"


SCRIPT = Path(__file__).resolve().parent / "policy_server.py"


def test_bind_announces_readiness_with_the_port_protocol_and_identity(capsys):
    # The supervisor blocks on this line before it connects, so its shape is a contract:
    # the token, then one JSON object carrying the port it actually bound.
    srv = PolicyServer(
        policy_fn=lambda s, w, o, context=None: np.zeros((s.shape[0], 7), dtype=np.float32),
        reset_fn=lambda: None,
        policy_id="ready:test",
        port=0,
        n_action_steps=37,  # distinct from the port, the protocol, and every schema width
    )
    try:
        port = srv.bind()
        line = capsys.readouterr().out.strip()
        assert line.startswith("SERVER_READY "), line
        ready = json.loads(line[len("SERVER_READY ") :])
        assert ready == {
            "port": port,
            "policy_id": "ready:test",
            "n_action_steps": 37,
            "protocol": PROTOCOL_VERSION,
            "inference": {},  # nothing was loaded, so nothing is served
            "context_dim": 0,
            "action_space": "absolute",
            "gripper": "binary",
        }
    finally:
        # PolicyServer has no public teardown (the serve loop owns its socket for the
        # process's lifetime), so the test closes the bound socket directly rather than
        # leaking a bound port into the rest of the session.
        srv._socket.close(0)


def test_a_usage_error_still_prints_a_token():
    # argparse's own error() exits 2 having written only to stderr, which is
    # indistinguishable from a server still loading to whoever waits for SERVER_READY.
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--port", "5555"],  # --checkpoint is required
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout.startswith("SERVER_FAIL usage: "), result.stdout
    assert "--checkpoint" in result.stdout


def test_served_inference_reads_the_resolved_config():
    from types import SimpleNamespace

    from policy_server import served_inference

    plain = SimpleNamespace(n_action_steps=100, temporal_ensemble_coeff=None, chunk_size=100)
    assert served_inference(plain) == {
        "n_action_steps": 100,
        "temporal_ensemble_coeff": None,
        "chunk_size": 100,
    }
    ensembled = SimpleNamespace(n_action_steps=1, temporal_ensemble_coeff=0.01, chunk_size=100)
    assert served_inference(ensembled) == {
        "n_action_steps": 1,
        "temporal_ensemble_coeff": 0.01,
        "chunk_size": 100,
    }


def test_handshake_carries_the_served_inference():
    import threading

    srv = PolicyServer(
        policy_fn=lambda s, w, o, context=None: np.zeros((s.shape[0], 7), dtype=np.float32),
        reset_fn=lambda: None,
        policy_id="inf:test",
        port=0,
        n_action_steps=1,
        inference={"n_action_steps": 1, "temporal_ensemble_coeff": 0.01, "chunk_size": 100},
    )
    port = srv.bind()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    req = zmq.Context.instance().socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 10_000)
    req.connect(f"tcp://127.0.0.1:{port}")
    try:
        send_message(req, {"type": "hello", "protocol": PROTOCOL_VERSION})
        header, _ = recv_message(req)
        assert header["inference"] == {
            "n_action_steps": 1,
            "temporal_ensemble_coeff": 0.01,
            "chunk_size": 100,
        }
    finally:
        send_message(req, {"type": "shutdown"})
        recv_message(req)
        thread.join(timeout=5)
        req.close(0)


def test_action_space_marker_is_found_up_the_checkpoint_tree(tmp_path):
    ckpt = tmp_path / "checkpoints" / "575000" / "pretrained_model"
    ckpt.mkdir(parents=True)
    assert find_markers(str(ckpt)) == ("absolute", "binary", None)
    (tmp_path / "action_space.json").write_text('{"action_space": "delta_prev"}')
    assert find_markers(str(ckpt)) == ("delta_prev", "binary", None)


def test_integrate_delta_adds_the_anchor_and_keeps_the_gripper():
    from policy_server import integrate_delta

    action = np.zeros((2, 7), np.float32)
    action[:, :6] = 0.1
    action[:, 6] = -1.0
    anchor = np.full((2, 6), 1.0, np.float32)
    out = integrate_delta(action, anchor)
    assert np.allclose(out[:, :6], 1.1) and out[:, 6].tolist() == [-1.0, -1.0]
    assert np.allclose(action[:, :6], 0.1)  # the caller's array is untouched


def test_policy_id_marks_delta_prev_checkpoints():
    assert inference_policy_id("ck", {}, action_space="delta_prev").endswith("+dprev")
    assert "+d" not in inference_policy_id("ck", {})


def test_policy_id_marks_a_continuous_gripper_checkpoint():
    assert inference_policy_id("ck", {}, gripper="continuous") == "act:ck+gap"
    assert inference_policy_id("ck", {}) == "act:ck"
    # The suffix comes last, after every other override.
    assert (
        inference_policy_id("ck", {"n_action_steps": 25}, context_dim=9, gripper="continuous")
        == "act:ck+nas25+ctx9+gap"
    )


def test_the_handshake_carries_the_gripper_contract(tmp_path):
    marker = tmp_path / "action_space.json"
    marker.write_text(json.dumps({"action_space": "absolute", "gripper": "continuous"}))
    ckpt = tmp_path / "checkpoints" / "000100" / "pretrained_model"
    ckpt.mkdir(parents=True)
    assert find_markers(str(ckpt))[1] == "continuous"
    marker.write_text(json.dumps({"action_space": "absolute"}))
    assert find_markers(str(tmp_path))[1] == "binary"
    srv = PolicyServer(
        policy_fn=lambda *a: None,
        reset_fn=lambda: None,
        policy_id="id",
        port=0,
        n_action_steps=1,
        gripper="continuous",
    )
    assert hello_reply({"protocol": PROTOCOL_VERSION}, srv.ready())["gripper"] == "continuous"


def _chunk_of(state, n):
    """A per-sample chunk: row b, step t is state[b] + t in every column."""
    return np.stack([state + t for t in range(n)], axis=1).astype(np.float32)


def test_action_queues_drain_in_lockstep_without_flags_and_replan_per_env():
    from policy_server import ActionQueues

    n = 3
    seen = []

    def predict(obs):
        seen.append(obs["state"][:, 0].tolist())
        return _chunk_of(obs["state"], n)

    def obs(step):
        state = np.zeros((2, 7), np.float32)
        state[:, 0] = [10 * step, 100 + 10 * step]  # the two envs differ and drift
        return {"state": state}

    queues = ActionQueues(predict, n)
    # Steps 0 to 2: one forward over both envs at step 0, then the queue heads, which is what
    # ACTPolicy.select_action's single queue gives.
    got = [queues.step(obs(s)) for s in range(3)]
    assert seen == [[0.0, 100.0]]
    for t in range(3):
        np.testing.assert_array_equal(got[t], _chunk_of(obs(0)["state"], n)[:, t])
    # Step 3: both queues are empty, one refill from step 3's observation.
    queues.step(obs(3))
    assert seen[-1] == [30.0, 130.0]
    # Step 4: env 1 replans from its own row alone; env 0 keeps draining its queue.
    out = queues.step(obs(4), np.array([False, True]))
    assert seen[-1] == [140.0]
    np.testing.assert_array_equal(out[0], _chunk_of(obs(3)["state"], n)[0, 1])
    np.testing.assert_array_equal(out[1], _chunk_of(obs(4)["state"], n)[1, 0])
    out = queues.step(obs(5))
    assert seen[-1] == [140.0]
    np.testing.assert_array_equal(out[0], _chunk_of(obs(3)["state"], n)[0, 2])
    np.testing.assert_array_equal(out[1], _chunk_of(obs(4)["state"], n)[1, 1])
    queues.reset()
    queues.step(obs(6))
    assert seen[-1] == [60.0, 160.0]


def test_action_warp_resamples_each_chunk_and_refuses_a_slowdown():
    from policy_server import ActionQueues, warp_chunk

    chunk = np.random.default_rng(0).normal(size=(2, 50, 7)).astype(np.float32)
    assert warp_chunk(chunk, 1.0) is chunk  # byte-equal: every unwarped read stays identical
    np.testing.assert_array_equal(warp_chunk(chunk, 2.0), chunk[:, ::2])
    square = np.stack([np.arange(5.0) ** 2] * 7, axis=-1)[None].astype(np.float32)
    # t = 0, 1.5, 3: 0, halfway between 1 and 4, then 9; t = 4.5 is past n - 1 = 4.
    np.testing.assert_allclose(warp_chunk(square, 1.5)[0, :, 3], [0.0, 2.5, 9.0])
    forwards = []

    def predict(obs):
        forwards.append(len(obs["state"]))
        return _chunk_of(obs["state"], 4)

    queues = ActionQueues(predict, 4, action_warp=2.0)
    got = [queues.step({"state": np.zeros((2, 7), np.float32)})[0, 0] for _ in range(4)]
    assert got == [0.0, 2.0, 0.0, 2.0] and forwards == [2, 2]  # a half-length queue
    with pytest.raises(ValueError, match="action_warp"):
        ActionQueues(predict, 4, action_warp=0.5)


def test_the_step_hands_the_replan_flag_to_the_policy_only_when_sent():
    calls = {"reset": 0, "replan": []}

    def fake_policy(state, wrist, overhead, context=None, replan=None):
        calls["replan"].append(None if replan is None else replan.tolist())
        return np.zeros((state.shape[0], 7), dtype=np.float32)

    req, thread = _start_server(fake_policy, calls)
    send_message(req, {"type": "step", "batch": 2}, _obs())
    assert recv_message(req)[0]["type"] == "action"
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "replan": np.array([True, False])})
    assert recv_message(req)[0]["type"] == "action"
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "replan": np.zeros(3, dtype=bool)})
    header, _ = recv_message(req)
    assert header["type"] == "error" and header["error"] == "bad_obs"
    assert "replan" in header["message"]
    _stop_server(req, thread)
    assert calls["replan"] == [None, [True, False]]


def test_the_step_validates_the_retry_flag_and_never_hands_it_to_the_policy():
    calls = {"reset": 0, "replan": []}

    def fake_policy(state, wrist, overhead, context=None, replan=None):
        calls["replan"].append(None if replan is None else replan.tolist())
        return np.zeros((state.shape[0], 7), dtype=np.float32)

    req, thread = _start_server(fake_policy, calls)
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "retry": np.array([True, False])})
    assert recv_message(req)[0]["type"] == "action"
    retry = {"replan": np.array([False, True]), "retry": np.ones(2, dtype=bool)}
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), **retry})
    assert recv_message(req)[0]["type"] == "action"
    send_message(req, {"type": "step", "batch": 2}, {**_obs(), "retry": np.zeros(3, dtype=bool)})
    header, _ = recv_message(req)
    assert header["type"] == "error" and header["error"] == "bad_obs"
    assert "retry" in header["message"]
    _stop_server(req, thread)
    assert calls["replan"] == [None, [False, True]]


CHECKPOINT = Path.home() / "data/maker-arm-ms6/m/train/checkpoints/400000/pretrained_model"


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="the m 400k checkpoint is not on this host")
def test_the_per_env_queue_equals_the_batched_forward_on_the_checkpoint():
    """ACT in eval mode is per-sample: an env replanned alone must get what it gets inside
    the batch, step for step, through the real checkpoint's chunk path."""
    from policy_server import load_act

    n = 3
    policy_fn, reset_fn, _id, _n, _inf, context_dim, _space, _grip = load_act(
        str(CHECKPOINT), "cpu", n_action_steps=n
    )
    rng = np.random.default_rng(0)
    state = rng.uniform(-1.0, 1.0, (2, 7)).astype(np.float32)
    state[:, 6] = 0.5
    wrist = rng.integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    overhead = rng.integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    context = rng.uniform(-0.2, 0.2, (2, context_dim)).astype(np.float32)
    reset_fn()
    batched = [policy_fn(state, wrist, overhead, context) for _ in range(n)]
    assert batched[0].shape == (2, 7) and not np.array_equal(batched[0], batched[1])
    for b in range(2):
        reset_fn()
        rows = (state[b : b + 1], wrist[b : b + 1], overhead[b : b + 1], context[b : b + 1])
        alone = [policy_fn(*rows) for _ in range(n)]
        for t in range(n):
            np.testing.assert_allclose(alone[t][0], batched[t][b], atol=1e-5)
    # A replan mid-chunk cuts env 1's queue and leaves env 0's alone.
    reset_fn()
    first = policy_fn(state, wrist, overhead, context)
    cut = policy_fn(state, wrist, overhead, context, replan=np.array([False, True]))
    np.testing.assert_array_equal(cut[0], batched[1][0])
    np.testing.assert_allclose(cut[1], first[1], atol=1e-5)


# MS7g D12: the wrapped, normalized context, gated by the checkpoint's marker.


def test_the_roll_wraps_to_a_small_signed_angle_on_numpy_and_torch():
    import torch

    from policy_server import CONTEXT_ROLL, wrap_context_roll

    context = np.zeros((3, 16), np.float32)
    context[:, CONTEXT_ROLL] = [3.13, -3.135, np.pi]
    context[:, 0] = 0.16
    for array in (context, torch.from_numpy(context)):
        out = wrap_context_roll(array)
        np.testing.assert_allclose(
            np.asarray(out[:, CONTEXT_ROLL]), [3.13 - np.pi, -3.135 + np.pi, 0.0], atol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(out[:, 0]), context[:, 0])
        assert out.dtype == array.dtype
    assert context[0, CONTEXT_ROLL] == np.float32(3.13)  # the caller's array is untouched


def test_the_context_marker_gates_the_transform(tmp_path):
    from policy_server import CONTEXT_NORM

    ckpt = tmp_path / "checkpoints" / "050000" / "pretrained_model"
    ckpt.mkdir(parents=True)
    assert find_markers(str(ckpt))[2] is None
    marker = tmp_path / "action_space.json"
    marker.write_text('{"action_space": "absolute", "gripper": "continuous"}')
    assert find_markers(str(ckpt))[2] is None  # every checkpoint before D12
    marker.write_text(json.dumps({"action_space": "absolute", "context": CONTEXT_NORM}))
    assert find_markers(str(ckpt))[2] == CONTEXT_NORM
    marker.write_text('{"action_space": "absolute", "context": "zscore"}')
    with pytest.raises(ValueError, match="unknown context transform"):
        find_markers(str(ckpt))


def test_policy_id_marks_a_normalized_context_only():
    assert inference_policy_id("ck", {}, context_dim=16, gripper="continuous") == "act:ck+ctx16+gap"
    assert (
        inference_policy_id("ck", {}, context_dim=16, gripper="continuous", context_norm="x")
        == "act:ck+ctx16+ctxnorm+gap"
    )


F5 = Path.home() / "data/maker-arm-ms7/f/train_F5/checkpoints/200000/pretrained_model"


@pytest.mark.skipif(not F5.is_dir(), reason="the F5 200k checkpoint is not on this host")
def test_an_old_checkpoint_serves_its_context_untouched(monkeypatch):
    import policy_server

    def refuse(context):
        raise AssertionError("an old checkpoint must not wrap its context")

    monkeypatch.setattr(policy_server, "wrap_context_roll", refuse)
    policy_fn, _reset, policy_id, *_ = policy_server.load_act(str(F5), "cpu", n_action_steps=1)
    rng = np.random.default_rng(0)
    state = rng.uniform(-1.0, 1.0, (1, 7)).astype(np.float32)
    image = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    context = np.full((1, 16), 0.1, np.float32)
    assert policy_fn(state, image, image, context).shape == (1, 7)
    assert "ctxnorm" not in policy_id


# MS7g D20: the serving-side policy ensemble (--ensemble_with).


def _stub_member(path, scale, *, context_norm=None, gripper="continuous"):
    """A member whose chunk is scale * (state + t) and whose temporal-ensemble step is scale *
    state, with the contract and id fields _load_member returns."""
    from types import SimpleNamespace

    resets = []
    inference = {"n_action_steps": 3, "temporal_ensemble_coeff": None, "chunk_size": 100}
    return SimpleNamespace(
        policy=SimpleNamespace(reset=lambda: resets.append(path)),
        resets=resets,
        predict_chunk=lambda obs: scale * _chunk_of(obs["state"], 3),
        select=lambda state, wrist, overhead, context: scale * state,
        policy_id=f"act:{path}+nas3+ctx16" + ("+ctxnorm" if context_norm else "") + "+gap",
        inference=inference,
        n_action_steps=3,
        context_dim=16,
        context_norm=context_norm,
        temporal_ensemble=False,
        contract={
            "action_space": "absolute",
            "gripper": gripper,
            "action_shape": [7],
            "context_dim": 16,
            **inference,
        },
    )


def _stub_members(monkeypatch, **members):
    import policy_server

    loaded = []

    def load(path, *_args):
        loaded.append(members[path])
        return members[path]

    monkeypatch.setattr(policy_server, "_load_member", load)
    return loaded


def test_without_ensemble_with_the_server_serves_one_member_unchanged(monkeypatch):
    from policy_server import load_act

    member = _stub_member("a", 1.0)
    _stub_members(monkeypatch, a=member)
    policy_fn, _reset, policy_id, n, inference, context_dim, space, gripper = load_act("a", "cpu")
    obs = {**_obs(), "context": np.zeros((2, 16), np.float32)}
    obs["state"][:] = np.arange(14, dtype=np.float32).reshape(2, 7)
    out = policy_fn(**obs)
    np.testing.assert_array_equal(out, _chunk_of(obs["state"], 3)[:, 0])
    assert policy_id == member.policy_id and "+ens" not in policy_id
    assert inference == {**member.inference, "action_warp": 1.0}  # no ensemble_with key
    assert (n, context_dim, space, gripper) == (3, 16, "absolute", "continuous")


def test_the_ensemble_serves_the_mean_of_its_members_and_names_both(monkeypatch):
    from policy_server import load_act, mean_action, warp_chunk

    one = np.ones((2, 7), np.float32)
    assert mean_action([one]) is one  # a single member's own array, byte for byte
    loaded = _stub_members(
        monkeypatch, a=_stub_member("a", 1.0), b=_stub_member("b", 3.0, context_norm="x")
    )
    policy_fn, reset_fn, policy_id, _n, inference, *_ = load_act(
        "a", "cpu", n_action_steps=3, action_warp=1.5, ensemble_with="b"
    )
    obs = {**_obs(), "context": np.zeros((2, 16), np.float32)}
    obs["state"][:] = np.arange(14, dtype=np.float32).reshape(2, 7)
    # The mean of the 1x and 3x chunks, then the warp, as on one checkpoint.
    want = warp_chunk(2.0 * _chunk_of(obs["state"], 3), 1.5)
    for t in range(want.shape[1]):
        row = policy_fn(**obs)
        np.testing.assert_array_equal(row, want[:, t])
        assert row.dtype == np.float32
    assert policy_id == "act:a+nas3+ctx16+gap+ens:b+ctxnorm"
    assert policy_id.startswith("act:a")  # the supervisor's cache prefix
    assert inference["ensemble_with"] == "b" and inference["action_warp"] == 1.5
    reset_fn()
    assert [m.resets for m in loaded] == [["a"], ["b"]]


def test_the_ensemble_averages_the_temporal_ensemble_steps(monkeypatch):
    from policy_server import load_act

    a, b = _stub_member("a", 1.0), _stub_member("b", 2.0)
    a.temporal_ensemble = b.temporal_ensemble = True
    _stub_members(monkeypatch, a=a, b=b)
    policy_fn, *_ = load_act("a", "cpu", ensemble_with="b")
    state = np.ones((1, 7), np.float32)
    np.testing.assert_array_equal(policy_fn(state, None, None, None), 1.5 * state)


def test_the_ensemble_refuses_members_that_differ(monkeypatch):
    from policy_server import load_act

    _stub_members(monkeypatch, a=_stub_member("a", 1.0), c=_stub_member("c", 1.0, gripper="binary"))
    with pytest.raises(ValueError, match="ensemble_with refused.*gripper"):
        load_act("a", "cpu", ensemble_with="c")


F9N = Path.home() / "data/maker-arm-ms7/g/train_F9n/checkpoints/800000/pretrained_model"


@pytest.mark.skipif(
    not (F5.is_dir() and F9N.is_dir()), reason="the F5 200k and F9n 800k checkpoints are not here"
)
def test_each_member_runs_its_own_preprocessing_and_only_f9n_wraps_the_roll(monkeypatch):
    import policy_server

    wraps = []
    wrap = policy_server.wrap_context_roll

    def counted(context):
        wraps.append(context.shape)
        return wrap(context)

    monkeypatch.setattr(policy_server, "wrap_context_roll", counted)
    load = policy_server._load_member
    members = []
    monkeypatch.setattr(
        policy_server, "_load_member", lambda *a: members.append(load(*a)) or members[-1]
    )
    policy_fn, _reset, policy_id, *_ = policy_server.load_act(
        str(F5), "cpu", n_action_steps=1, ensemble_with=str(F9N)
    )
    assert policy_id.startswith(f"act:{F5}+nas1+ctx16+gap+ens:{F9N}") and policy_id.endswith(
        "+ctxnorm"
    )
    rng = np.random.default_rng(0)
    obs = {
        "state": rng.uniform(-1.0, 1.0, (1, 7)).astype(np.float32),
        "wrist": rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8),
        "overhead": rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8),
        "context": np.full((1, 16), 0.1, np.float32),
    }
    obs["context"][:, policy_server.CONTEXT_ROLL] = 3.1
    f5 = members[0].predict_chunk(obs)
    assert wraps == []  # F5 carries no context marker
    f9n = members[1].predict_chunk(obs)
    assert len(wraps) == 1  # F9n's marker wraps its own copy of the context
    assert not np.allclose(f5, f9n, atol=1e-3)
    served = policy_fn(obs["state"], obs["wrist"], obs["overhead"], obs["context"])
    assert len(wraps) == 2  # one forward of each member, one wrap
    np.testing.assert_allclose(served, (f5[:, 0] + f9n[:, 0]) / 2, atol=1e-5)
    assert obs["context"][0, policy_server.CONTEXT_ROLL] == np.float32(3.1)

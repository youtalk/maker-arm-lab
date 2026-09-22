import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import zmq

from policy_protocol import PROTOCOL_VERSION, recv_message, send_message
from policy_server import (
    PolicyServer,
    blank_streams,
    find_gripper_bce_marker,
    find_gripper_marker,
    inference_policy_id,
    resolve_inference_overrides,
    squash_gripper_logit,
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


# --- BCE gripper head tests. The code under test is PORTED BUT INACTIVE in this repo:
# nothing here writes a gripper_bce.json, so these exercise a path no live checkpoint takes.
# See the section marker in policy_server.py. ---
#
# Gripper-dim action stats standing in for a checkpoint's own: raw = logit * RAW_STD +
# RAW_MEAN. These two values are an OPENYAM measurement (its v1 dataset's action mean/std
# at dim 6), carried over unchanged and UNMEASURED on the Maker Arm. Nothing under test
# depends on their values -- squash_gripper_logit inverts whatever (mean, std) it is
# handed -- so they are fixture constants here, not a claim about this robot's data.
RAW_MEAN = 0.1294
RAW_STD = 0.9916


def _action_with_gripper_raw(raw_values):
    action = np.zeros((len(raw_values), 7), dtype=np.float32)
    action[:, 6] = raw_values
    return action


def test_squash_maps_the_unnormalized_logit_back_to_a_unit_action():
    # a saturated open logit (+10) lands near +1, a saturated closed logit
    # (-10) near -1, and the p=0.5 boundary (logit 0) at exactly raw 0 --
    # so the client's raw >= 0 open threshold now coincides with the
    # boundary the BCE head was trained against
    raw = [10.0 * RAW_STD + RAW_MEAN, -10.0 * RAW_STD + RAW_MEAN, RAW_MEAN]
    out = squash_gripper_logit(_action_with_gripper_raw(raw), RAW_MEAN, RAW_STD)
    assert out[0, 6] == pytest.approx(1.0, abs=1e-3)
    assert out[1, 6] == pytest.approx(-1.0, abs=1e-3)
    assert out[2, 6] == pytest.approx(0.0, abs=1e-6)
    assert np.abs(out[:, 6]).max() < 1.0 + 1e-6


def test_squash_touches_only_the_gripper_dim_and_not_in_place():
    action = np.full((2, 7), 0.5, dtype=np.float32)
    action[:, 6] = 8.26
    out = squash_gripper_logit(action, RAW_MEAN, RAW_STD)
    np.testing.assert_array_equal(out[:, :6], action[:, :6])
    assert out.dtype == np.float32
    assert action[0, 6] == pytest.approx(8.26)  # caller's buffer untouched


def test_marker_is_found_in_the_checkpoint_dir_itself(tmp_path):
    marker = {"gripper_bce": True, "gripper_raw_mean": RAW_MEAN, "gripper_raw_std": RAW_STD}
    (tmp_path / "gripper_bce.json").write_text(json.dumps(marker))
    assert find_gripper_bce_marker(str(tmp_path)) == marker


def test_marker_is_found_by_walking_up_to_the_training_output_dir(tmp_path):
    # openyam-lab's train_act_gripper wrote the marker to output_dir while the server is
    # handed output_dir/checkpoints/<step>/pretrained_model. That trainer is not ported, so
    # this layout is the one the lookup is designed for, not one produced in this repo.
    marker = {"gripper_bce": True, "gripper_raw_mean": RAW_MEAN, "gripper_raw_std": RAW_STD}
    (tmp_path / "gripper_bce.json").write_text(json.dumps(marker))
    ckpt = tmp_path / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    assert find_gripper_bce_marker(str(ckpt)) == marker


def test_no_marker_means_no_squash(tmp_path):
    assert find_gripper_bce_marker(str(tmp_path)) is None


def test_policy_id_records_the_gripper_squash():
    assert inference_policy_id("/ckpt", {}, gripper_squash=True) == "act:/ckpt+gbce"
    assert (
        inference_policy_id("/ckpt", {"temporal_ensemble_coeff": 0.01}, gripper_squash=True)
        == "act:/ckpt+te0.01+gbce"
    )
    # default stays byte-identical for non-BCE checkpoints
    assert inference_policy_id("/ckpt", {}) == "act:/ckpt"


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
    from policy_server import find_action_space_marker

    ckpt = tmp_path / "checkpoints" / "575000" / "pretrained_model"
    ckpt.mkdir(parents=True)
    assert find_action_space_marker(str(ckpt)) == "absolute"
    (tmp_path / "action_space.json").write_text('{"action_space": "delta_prev"}')
    assert find_action_space_marker(str(ckpt)) == "delta_prev"


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
    assert find_gripper_marker(str(ckpt)) == "continuous"
    marker.write_text(json.dumps({"action_space": "absolute"}))
    assert find_gripper_marker(str(tmp_path)) == "binary"
    srv = PolicyServer(
        policy_fn=lambda *a: None,
        reset_fn=lambda: None,
        policy_id="id",
        port=0,
        n_action_steps=1,
        gripper="continuous",
    )
    assert srv._hello_ack()["gripper"] == "continuous"

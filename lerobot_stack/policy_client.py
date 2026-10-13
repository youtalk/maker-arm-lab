"""REQ-side client of the policy server's wire contract (numpy + zmq only, no Kit).

Split out of scripts/eval_policy.py so the handshake, the schema check and the step
round-trip have CPU tests against the real PolicyServer (lerobot_stack/test_policy_client.py);
the eval script only adds the Isaac loop around it.
"""

import time

import numpy as np
import zmq

import policy_protocol


class ServerPolicy:
    """Connects, validates the handshake against this client's schema, and resets."""

    def __init__(self, endpoint: str, recv_timeout_ms: int = 60_000):
        self._socket = zmq.Context.instance().socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        policy_protocol.send_message(
            self._socket, {"type": "hello", "protocol": policy_protocol.PROTOCOL_VERSION}
        )
        header, _ = policy_protocol.recv_message(self._socket)
        if header.get("type") != "hello_ack":
            raise RuntimeError(f"handshake refused: {header}")
        want = {k: [list(s), d] for k, (s, d) in policy_protocol.OBS_SCHEMA.items()}
        want_action = [list(policy_protocol.ACTION_SHAPE), policy_protocol.ACTION_DTYPE]
        if header["obs_schema"] != want or header["action"] != want_action:
            raise RuntimeError(
                f"schema mismatch: server={header['obs_schema']}/{header['action']}, "
                f"client={want}/{want_action}"
            )
        self.policy_id: str = header["policy_id"]
        self.n_action_steps: int = header["n_action_steps"]
        # What the server actually serves with (n_action_steps, temporal_ensemble_coeff,
        # chunk_size), echoed straight off the handshake so the eval payload records the
        # served config rather than the requested one. Empty from a server that predates it.
        self.inference: dict = dict(header.get("inference", {}))
        self.context_dim: int = int(header.get("context_dim", 0))
        self.action_space: str = str(header.get("action_space", "absolute"))
        self.gripper: str = str(header.get("gripper", "binary"))
        # MS5 design P1: the last step's server time in the policy call and round trip, ms.
        self.last_policy_ms: float | None = None
        self.last_rtt_ms: float | None = None
        self.reset()

    def reset(self) -> None:
        policy_protocol.send_message(self._socket, {"type": "reset"})
        header, _ = policy_protocol.recv_message(self._socket)
        if header.get("type") != "reset_ack":
            raise RuntimeError(f"reset refused: {header}")

    def step(
        self,
        state: np.ndarray,
        wrist: np.ndarray,
        overhead: np.ndarray,
        context: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
        replan: np.ndarray | None = None,
        retry: np.ndarray | None = None,
        prompt: np.ndarray | None = None,
    ) -> np.ndarray:
        """``replan`` is the (batch,) bool per-env replan flag of ``--replan_on_switch``,
        ``retry`` the per-env retry flag of ``--retry_noise_at`` and ``prompt`` the per-env
        subtask id of ``--pi05_prompt``; None leaves each off the wire, which an older server
        needs."""
        arrays = {"state": state, "wrist": wrist, "overhead": overhead}
        if context is not None:
            arrays[policy_protocol.CONTEXT_KEY] = context
        if anchor is not None:
            arrays[policy_protocol.ANCHOR_KEY] = anchor
        if replan is not None:
            arrays[policy_protocol.REPLAN_KEY] = np.asarray(replan, dtype=bool)
        if retry is not None:
            arrays[policy_protocol.RETRY_KEY] = np.asarray(retry, dtype=bool)
        if prompt is not None:
            arrays[policy_protocol.PROMPT_KEY] = np.asarray(prompt, dtype=np.int16)
        started = time.perf_counter()
        policy_protocol.send_message(
            self._socket, {"type": "step", "batch": int(state.shape[0])}, arrays
        )
        header, out = policy_protocol.recv_message(self._socket)
        self.last_rtt_ms = 1e3 * (time.perf_counter() - started)
        # Cleared before the type check, so an error reply never leaves the previous step's
        # time behind for a reader that catches the exception.
        self.last_policy_ms = None
        if header.get("type") != "action":
            raise RuntimeError(f"server error: {header}")
        self.last_policy_ms = header.get("policy_ms")
        action = out["action"]
        if (
            action.shape != (state.shape[0], *policy_protocol.ACTION_SHAPE)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError(f"bad action from server: shape={action.shape}")
        return action

    def close(self) -> None:
        self._socket.close(0)

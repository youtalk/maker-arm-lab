"""ACT policy server: a ZMQ REP loop running in the LeRobot venv.

The serve loop is dependency-injected (policy_fn/reset_fn) so the protocol
handling has CPU tests against a fake policy; load_act() is the only code
that touches lerobot, and only the CLI path calls it. The server holds the
checkpoint's own pre/post processor pipelines, so normalization always
matches training exactly -- the client sends raw uint8 images and raw joint
positions and receives raw joint-position targets, mirroring what a real
robot's driver would exchange (the same boundary later reused for Jetson
Thor and the real arm).

REP discipline: every request gets exactly one reply, including failures --
an unanswered REQ would deadlock the client, which is worse than any error.

Token contract: every exit path prints SERVER_READY or SERVER_FAIL on stdout. A
supervisor waits for SERVER_READY before connecting, so a start-up that dies
silently -- a bad flag, a missing checkpoint, a bound port -- reads to it as a hang
rather than a failure, which is the worse of the two outcomes.
"""

import argparse
import json
import time
import traceback
from collections import deque
from types import SimpleNamespace

import numpy as np
import zmq

from policy_protocol import (
    ANCHOR_KEY,
    CONTEXT_KEY,
    PROTOCOL_VERSION,
    REPLAN_KEY,
    hello_reply,
    recv_message,
    send_message,
    validate_action,
    validate_obs,
)


class PolicyServer:
    def __init__(
        self,
        policy_fn,
        reset_fn,
        policy_id: str,
        port: int,
        n_action_steps: int,
        inference: dict | None = None,
        context_dim: int = 0,
        action_space: str = "absolute",
        gripper: str = "binary",
    ):
        self._policy_fn = policy_fn
        self._reset_fn = reset_fn
        self._policy_id = policy_id
        self._port = port
        self._n_action_steps = n_action_steps
        self._action_space = action_space
        # The checkpoint's gripper contract (Task 12's action_space.json "gripper" key):
        # "binary" is a raw sign (>= 0 opens), "continuous" is a jaw gap in meters. Echoed in
        # SERVER_READY and hello_ack so a client (and core.loop's mismatch check) knows which.
        self._gripper = gripper
        # The inference config the loaded policy actually runs with (n_action_steps,
        # temporal_ensemble_coeff, chunk_size, action_warp), read off the policy after the
        # overrides were applied and echoed in SERVER_READY and hello_ack so an eval payload
        # records what was served rather than what the supervisor asked for.
        self._inference = dict(inference or {})
        # Width of the optional context array this policy takes (0 = none). Echoed in the
        # handshake so a client knows whether to send one, and a step's context width is
        # checked against it -- a mismatch, including a missing context when this is > 0,
        # is a bad_obs error rather than a shape exception surfacing inside the policy.
        self._context_dim = context_dim
        self._socket = None

    def bind(self) -> int:
        ctx = zmq.Context.instance()
        self._socket = ctx.socket(zmq.REP)
        if self._port == 0:
            self._port = self._socket.bind_to_random_port("tcp://127.0.0.1")
        else:
            self._socket.bind(f"tcp://127.0.0.1:{self._port}")
        print("SERVER_READY " + json.dumps(self.ready()), flush=True)
        return self._port

    def ready(self) -> dict:
        return {
            "port": self._port,
            "policy_id": self._policy_id,
            "n_action_steps": self._n_action_steps,
            "protocol": PROTOCOL_VERSION,
            "inference": self._inference,
            "context_dim": self._context_dim,
            "action_space": self._action_space,
            "gripper": self._gripper,
        }

    def handle_one(self) -> bool:
        header, arrays = recv_message(self._socket)
        kind = header.get("type")
        try:
            if kind == "hello":
                send_message(self._socket, hello_reply(header, self.ready()))
            elif kind == "reset":
                self._reset_fn()
                send_message(self._socket, {"type": "reset_ack"})
            elif kind == "step":
                batch = int(header["batch"])
                try:
                    validate_obs(arrays, batch)
                except ValueError as err:
                    send_message(
                        self._socket, {"type": "error", "error": "bad_obs", "message": str(err)}
                    )
                    return True
                context = arrays.get(CONTEXT_KEY)
                width = 0 if context is None else int(context.shape[1])
                if width != self._context_dim:
                    send_message(
                        self._socket,
                        {
                            "type": "error",
                            "error": "bad_obs",
                            "message": f"context width {width}, this policy takes {self._context_dim}",
                        },
                    )
                    return True
                anchor = arrays.get(ANCHOR_KEY)
                if self._action_space == "delta_prev" and anchor is None:
                    send_message(
                        self._socket,
                        {
                            "type": "error",
                            "error": "bad_obs",
                            "message": "a delta_prev checkpoint needs the anchor array "
                            "(the previous clamped arm command)",
                        },
                    )
                    return True
                # The replan flag is passed only when the client sent one, so a policy_fn
                # that predates it (every test fake) keeps its four-argument signature.
                replan = arrays.get(REPLAN_KEY)
                extra = {} if replan is None else {"replan": replan}
                started = time.perf_counter()
                action = np.asarray(
                    self._policy_fn(
                        arrays["state"], arrays["wrist"], arrays["overhead"], context, **extra
                    ),
                    dtype=np.float32,
                )
                policy_ms = round(1e3 * (time.perf_counter() - started), 3)
                validate_action(action, batch)
                if self._action_space == "delta_prev":
                    action = integrate_delta(action, anchor)
                if not np.isfinite(action).all():
                    send_message(
                        self._socket,
                        {
                            "type": "error",
                            "error": "nan_action",
                            "message": "policy produced non-finite actions",
                        },
                    )
                    return True
                # MS5 design P1: the server's time in the policy call. A queue pop is
                # microseconds; a step that refilled a queue ran a forward.
                send_message(
                    self._socket, {"type": "action", "policy_ms": policy_ms}, {"action": action}
                )
            elif kind == "shutdown":
                send_message(self._socket, {"type": "shutdown_ack"})
                return False
            else:
                send_message(
                    self._socket,
                    {
                        "type": "error",
                        "error": "exception",
                        "message": f"unknown message type {kind!r}",
                    },
                )
        except Exception:  # noqa: BLE001 REP discipline requires always replying
            send_message(
                self._socket,
                {"type": "error", "error": "exception", "message": traceback.format_exc()},
            )
        return True

    def serve_forever(self) -> None:
        while self.handle_one():
            pass


class ActionQueues:
    """One action queue per env, the per-env twin of ``ACTPolicy.select_action``'s queue.

    LeRobot keeps ONE queue for the whole batch, refilled every ``n_action_steps`` from one
    forward, so every env replans in lockstep and a target row written mid-chunk reaches the
    policy 0 to n_action_steps - 1 steps late. Here ``step`` refills the queues of the envs
    whose queue is empty or whose ``replan`` flag is set, from one forward over that sub-batch
    only, and every env pops the head of its own queue. ACT in eval mode is per-sample, so the
    sub-batch forward is the batched forward restricted to those rows. With no flags every
    queue empties on the same step and the refill is the lockstep one.

    ``predict_chunk(obs) -> (b, n_action_steps, action_dim)`` takes the observation arrays of
    the sub-batch, keyed like the wire (state, wrist, overhead, optional context).

    ``action_warp`` X plays each chunk X times faster (``warp_chunk``) before it fills the
    queues; 1.0 queues the chunk unchanged. A shorter queue only means more forwards.
    """

    def __init__(self, predict_chunk, n_action_steps: int, action_warp: float = 1.0):
        if not ACTION_WARP_MIN <= action_warp <= ACTION_WARP_MAX:
            raise ValueError(
                f"action_warp must be in [{ACTION_WARP_MIN}, {ACTION_WARP_MAX}], got {action_warp}"
            )
        self._predict = predict_chunk
        self._n = n_action_steps
        self._warp = action_warp
        self._queues: list[deque] = []

    def reset(self) -> None:
        self._queues = []

    def step(self, obs: dict[str, np.ndarray], replan: np.ndarray | None = None) -> np.ndarray:
        batch = int(obs["state"].shape[0])
        if len(self._queues) != batch:
            self._queues = [deque() for _ in range(batch)]
        flags = np.zeros(batch, dtype=bool) if replan is None else np.asarray(replan, dtype=bool)
        need = [i for i in range(batch) if flags[i] or not self._queues[i]]
        if need:
            chunk = np.asarray(self._predict({k: v[need] for k, v in obs.items()}))
            if chunk.shape[:2] != (len(need), self._n):
                raise ValueError(
                    f"predict_chunk returned {chunk.shape}, want ({len(need)}, {self._n}, ...)"
                )
            chunk = warp_chunk(chunk, self._warp)
            for row, i in enumerate(need):
                self._queues[i] = deque(chunk[row])
        return np.stack([q.popleft() for q in self._queues])


# MS7f speed track: the policy copies the teacher corpus's pace (about 15.4 s a cycle, 158
# correct stows an hour against the 300 bar), and playing its chunks faster needs no retrain.
ACTION_WARP_MIN = 1.0
ACTION_WARP_MAX = 3.0


def warp_chunk(chunk: np.ndarray, action_warp: float) -> np.ndarray:
    """Resample a (b, n, action_dim) chunk at times 0, X, 2X, ... while t <= n - 1, every
    channel linear between chunk[floor(t)] and chunk[floor(t) + 1]. X = 1.0 returns the chunk
    itself, so an unwarped read is byte-identical to one served before the switch."""
    if action_warp == 1.0:
        return chunk
    n = chunk.shape[1]
    t = np.arange(0.0, n - 1 + 1e-9, action_warp)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (t - lo)[None, :, None]
    return (chunk[:, lo] + w * (chunk[:, hi] - chunk[:, lo])).astype(chunk.dtype)


def resolve_inference_overrides(
    n_action_steps: int | None, temporal_ensemble_coeff: float | None, chunk_size: int
) -> dict:
    """Translate the CLI's inference flags into ACTConfig field replacements.

    Encodes lerobot 0.5.1's own contract up front (clearer errors than the
    dataclass __post_init__ that re-validates after dataclasses.replace):
    temporal ensembling queries the model every step, so it requires
    n_action_steps == 1; a chunked queue must fit inside the chunk.
    """
    if temporal_ensemble_coeff is not None:
        if n_action_steps is not None and n_action_steps != 1:
            raise ValueError(
                "temporal ensembling runs the model every step: n_action_steps "
                f"must be 1 (or omitted), got {n_action_steps}"
            )
        return {"temporal_ensemble_coeff": temporal_ensemble_coeff, "n_action_steps": 1}
    if n_action_steps is not None:
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
        if n_action_steps > chunk_size:
            raise ValueError(
                f"n_action_steps ({n_action_steps}) cannot exceed chunk_size ({chunk_size})"
            )
        return {"n_action_steps": n_action_steps}
    return {}


ACTION_SPACE_MARKER = "action_space.json"


def _read_action_space_marker(checkpoint_dir: str) -> dict | None:
    """The parsed action_space.json nearest checkpoint_dir, walking up to output_dir.

    train_act.sh writes it to the run's output_dir, while the server is handed
    output_dir/checkpoints/<step>/pretrained_model -- so look in the checkpoint dir itself,
    then walk up the three levels to output_dir.
    """
    from pathlib import Path

    directory = Path(checkpoint_dir)
    for candidate in [directory, *directory.parents[:3]]:
        marker = candidate / ACTION_SPACE_MARKER
        if marker.is_file():
            return json.loads(marker.read_text())
    return None


# MS7g D12 (2026-10-03): the overhead camera looks down, so its roll sits near pi and
# core.quat.rpy_from_quat's atan2 wraps it to either sign (f/w12345: 52.8 % in +3.107..+pi,
# 47.2 % in -pi..-3.107, a +-2.0 degree wobble), and LeRobot 0.5.1 ACT maps no ENV feature,
# so its context entered raw. A checkpoint whose action_space.json says "context":
# "roll_wrapped_norm" was trained on wrap(roll - pi) with the context MEAN_STD normalized by its
# own preprocessor (train_masked.py --context_norm); without the key the context passes as is.
CONTEXT_NORM = "roll_wrapped_norm"
CONTEXT_ROLL = 3  # overhead_cam_roll's index (tests/test_lerobot_schema_pin.py)


def find_markers(checkpoint_dir: str) -> tuple[str, str, str | None]:
    """The action space, the gripper contract (MS6 phase B design 5.4) and the context
    transform recorded next to the checkpoints. Without a marker they are "absolute", "binary"
    and None; without a gripper key "binary"; without a context key None (every checkpoint
    before MS7g D12). An unknown context transform is refused."""
    marker = _read_action_space_marker(checkpoint_dir)
    if marker is None:
        return "absolute", "binary", None
    context = marker.get("context")
    if context not in (None, CONTEXT_NORM):
        raise ValueError(f"unknown context transform {context!r} in {ACTION_SPACE_MARKER}")
    return marker["action_space"], marker.get("gripper", "binary"), context


def wrap_context_roll(context):
    """``context`` with the camera roll replaced by wrap(roll - pi), a small signed angle in
    [-pi, pi). A copy, for a numpy array or a torch tensor (..., K)."""
    context = context * 1
    context[..., CONTEXT_ROLL] = context[..., CONTEXT_ROLL] % (2 * np.pi) - np.pi
    return context


def integrate_delta(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """A delta_prev checkpoint's joint part plus the client's anchor, the previous clamped
    arm command: an absolute target again. The gripper column is a command, not a position,
    and is left alone.
    """
    out = action.copy()
    out[:, :6] += state[:, :6]
    return out


BLANKABLE_STREAMS = ("wrist", "overhead")


def blank_streams(wrist: np.ndarray, overhead: np.ndarray, blank_camera: str | None):
    """Zero one camera stream before inference; ``None`` passes both through.

    The occlusion ablation for "does the policy act on what it can see?". It exists
    because a two-camera policy can fail with every object still in frame, leaving
    which stream it actually uses an open question; blanking one at a time answers
    it. (openyam-lab reached that question from its own preview videos, where the arm
    retreated to a pose the wrist camera could not see the bricks from while the
    overhead camera still framed them all -- an OPENYAM observation, carried over as
    the reason the knob exists and unmeasured on this arm.)

    Zeros, not noise or a frozen frame: zeros are the standard no-information
    baseline, and the checkpoint's own normalizer maps them to a constant field.
    The caller's arrays are never modified in place -- the eval client owns
    those buffers and reuses them.
    """
    if blank_camera is None:
        return wrist, overhead
    if blank_camera not in BLANKABLE_STREAMS:
        raise ValueError(f"blank_camera must be one of {BLANKABLE_STREAMS}, got {blank_camera!r}")
    if blank_camera == "wrist":
        return np.zeros_like(wrist), overhead
    return wrist, np.zeros_like(overhead)


def inference_policy_id(
    checkpoint_dir: str,
    overrides: dict,
    blank_camera: str | None = None,
    context_dim: int = 0,
    action_space: str = "absolute",
    gripper: str = "binary",
    action_warp: float = 1.0,
    context_norm: str | None = None,
) -> str:
    """Suffix the policy id with the active overrides so eval payloads are self-describing."""
    policy_id = f"act:{checkpoint_dir}"
    if "temporal_ensemble_coeff" in overrides:
        policy_id += f"+te{overrides['temporal_ensemble_coeff']}"
    elif "n_action_steps" in overrides:
        policy_id += f"+nas{overrides['n_action_steps']}"
    if action_warp != 1.0:
        policy_id += f"+warp{action_warp}"
    if blank_camera is not None:
        policy_id += f"+blank{blank_camera}"
    if context_dim:
        policy_id += f"+ctx{context_dim}"
    if context_norm:
        policy_id += "+ctxnorm"
    if action_space == "delta_prev":
        policy_id += "+dprev"
    if gripper == "continuous":
        policy_id += "+gap"
    return policy_id


def load_act(
    checkpoint_dir: str,
    device: str,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
    blank_camera: str | None = None,
    action_warp: float = 1.0,
    ensemble_with: str | None = None,
):
    """Build (policy_fn, reset_fn, policy_id, n_action_steps, inference, context_dim, action_space,
    gripper).

    checkpoint_dir is a `pretrained_model` directory (e.g.
    <output_dir>/checkpoints/last/pretrained_model), loaded by ``_load_member``.
    ``action_warp`` plays each queued chunk faster (``ActionQueues``); the ensembler queues no
    chunk to warp.

    ``ensemble_with`` (MS7g D20) is a second `pretrained_model` directory, loaded the same way
    with its own pre- and postprocessor (so its own context marker), and every forward serves
    the elementwise mean of the two members' unnormalized actions. The queues, the warp, the
    replan flags and temporal ensembling run on the mean as they run on one checkpoint. The two
    must agree on each member's ``contract``, or the load is refused. The policy id keeps
    ``act:<checkpoint_dir>`` first and ends ``+ens:<ensemble_with>`` (plus the member's
    ``+ctxnorm``); the inference block gains ``ensemble_with``. None loads one checkpoint
    exactly as before.
    """
    if action_warp != 1.0 and temporal_ensemble_coeff is not None:
        raise ValueError("action_warp plays the queued chunk: temporal ensembling queues none")
    members = [
        _load_member(
            path, device, n_action_steps, temporal_ensemble_coeff, blank_camera, action_warp
        )
        for path in [checkpoint_dir, *([ensemble_with] if ensemble_with else [])]
    ]
    first = members[0]
    for member in members[1:]:
        diff = {
            key: (value, member.contract[key])
            for key, value in first.contract.items()
            if member.contract[key] != value
        }
        if diff:
            raise ValueError(f"--ensemble_with refused, the members differ: {diff}")

    def predict_chunk(obs: dict) -> np.ndarray:
        return mean_action([m.predict_chunk(obs) for m in members])

    queues = ActionQueues(predict_chunk, first.n_action_steps, action_warp)

    def policy_fn(
        state: np.ndarray,
        wrist: np.ndarray,
        overhead: np.ndarray,
        context: np.ndarray | None,
        replan: np.ndarray | None = None,
    ) -> np.ndarray:
        if first.temporal_ensemble:
            # The ensembler runs the model every step, so there is no queue to cut short.
            return mean_action([m.select(state, wrist, overhead, context) for m in members])
        obs = {"state": state, "wrist": wrist, "overhead": overhead}
        if first.context_dim:
            obs["context"] = context
        return queues.step(obs, replan)

    def reset_fn() -> None:
        for member in members:
            member.policy.reset()
        queues.reset()

    policy_id = first.policy_id
    inference = {**first.inference, "action_warp": action_warp}
    if ensemble_with:
        policy_id += f"+ens:{ensemble_with}" + ("+ctxnorm" if members[1].context_norm else "")
        inference["ensemble_with"] = ensemble_with
    return (
        policy_fn,
        reset_fn,
        policy_id,
        first.n_action_steps,
        inference,
        first.context_dim,
        first.contract["action_space"],
        first.contract["gripper"],
    )


def mean_action(actions: list[np.ndarray]) -> np.ndarray:
    """The elementwise mean of the members' unnormalized actions; one member's own array as is,
    so a read without --ensemble_with serves the same bytes as before the switch."""
    return actions[0] if len(actions) == 1 else np.mean(actions, axis=0, dtype=np.float32)


def _load_member(
    checkpoint_dir: str,
    device: str,
    n_action_steps: int | None,
    temporal_ensemble_coeff: float | None,
    blank_camera: str | None,
    action_warp: float,
):
    """One checkpoint with its own pre/post processor pipelines, as a namespace: ``policy``,
    ``predict_chunk(obs)``, ``select(state, wrist, overhead, context)`` (one temporal-ensemble
    step), ``policy_id``, ``inference``, ``n_action_steps``, ``context_dim``,
    ``context_norm``, ``temporal_ensemble`` and ``contract`` (what two ensemble members must
    share).

    The checkpoint's own
    saved pre/post processor pipelines carry normalization + device transfer,
    the same way lerobot_eval loads them (verified against lerobot 0.5.1:
    `preprocessor_overrides={"device_processor": {"device": ...}}`).
    Image conversion (uint8 HWC -> float32 CHW / 255) mirrors
    lerobot.envs.utils.preprocess_observation; images are sent to the policy only
    when the checkpoint has image features. `context_dim` is the checkpoint's
    `env_state_feature.shape[0]`, or 0 for a checkpoint that takes no context.
    Inference-config overrides (n_action_steps, temporal_ensemble_coeff) are applied
    at checkpoint load and reflected in the returned policy_id suffix (``action_warp`` only
    names itself there).
    """
    import dataclasses

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    overrides = resolve_inference_overrides(
        n_action_steps, temporal_ensemble_coeff, config.chunk_size
    )
    if overrides:
        # dataclasses.replace re-runs ACTConfig.__post_init__, so lerobot's own
        # validation (temporal ensembling => n_action_steps == 1, steps <= chunk)
        # double-checks the resolved overrides.
        config = dataclasses.replace(config, **overrides)
    policy = ACTPolicy.from_pretrained(checkpoint_dir, config=config)
    policy.to(device)
    policy.eval()
    env_feature = policy.config.env_state_feature
    context_dim = int(env_feature.shape[0]) if env_feature is not None else 0
    uses_images = bool(policy.config.image_features)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    action_space, gripper, context_norm = find_markers(checkpoint_dir)
    if context_norm and "ENV" not in policy.config.normalization_mapping:
        raise ValueError(f"the marker says {context_norm} but the checkpoint maps no ENV feature")
    if action_space == "delta":
        raise ValueError(
            "this checkpoint was trained on the c1 delta definition (target minus the measured "
            "joints of the same frame), which MS4 c2 removed; only absolute and delta_prev serve"
        )

    def _images(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(torch.float32) / 255.0

    def _batch(state, wrist, overhead, context) -> dict:
        batch = {"observation.state": torch.from_numpy(state)}
        if uses_images:
            wrist, overhead = blank_streams(wrist, overhead, blank_camera)
            batch["observation.images.wrist"] = _images(wrist)
            batch["observation.images.overhead"] = _images(overhead)
        if context_dim:
            if context_norm:
                context = wrap_context_roll(context)
            batch["observation.environment_state"] = torch.from_numpy(context)
        return pre(batch)

    def _finish(action) -> np.ndarray:
        return np.asarray(post(action).detach().to("cpu"), dtype=np.float32)

    def predict_chunk(obs: dict) -> np.ndarray:
        """(b, n_action_steps, 7): the chunk head ``select_action`` would queue, unnormalized
        in one call (the unnormalizer is elementwise, so this equals per-step ``post``)."""
        batch = _batch(obs["state"], obs["wrist"], obs["overhead"], obs.get("context"))
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)[:, : policy.config.n_action_steps]
        return _finish(chunk)

    def select(state, wrist, overhead, context) -> np.ndarray:
        with torch.inference_mode():
            return _finish(policy.select_action(_batch(state, wrist, overhead, context)))

    return SimpleNamespace(
        policy=policy,
        predict_chunk=predict_chunk,
        select=select,
        policy_id=inference_policy_id(
            checkpoint_dir,
            overrides,
            blank_camera,
            context_dim=context_dim,
            action_space=action_space,
            gripper=gripper,
            action_warp=action_warp,
            context_norm=context_norm,
        ),
        inference=served_inference(policy.config),
        n_action_steps=int(policy.config.n_action_steps),
        context_dim=context_dim,
        context_norm=context_norm,
        temporal_ensemble=policy.config.temporal_ensemble_coeff is not None,
        contract={
            "action_space": action_space,
            "gripper": gripper,
            "action_shape": list(policy.config.action_feature.shape),
            "context_dim": context_dim,
            **served_inference(policy.config),
        },
    )


def served_inference(config) -> dict:
    """The inference settings a loaded ACT config will actually serve with.

    Read off the config AFTER `resolve_inference_overrides` was applied, not off the CLI:
    the gate record has to say what ran. `temporal_ensemble_coeff` is None for plain
    chunked serving, which is lerobot's own encoding of "ensembling off".
    """
    return {
        "n_action_steps": int(config.n_action_steps),
        "temporal_ensemble_coeff": (
            None
            if config.temporal_ensemble_coeff is None
            else float(config.temporal_ensemble_coeff)
        ),
        "chunk_size": int(config.chunk_size),
    }


class TokenArgumentParser(argparse.ArgumentParser):
    """An ArgumentParser whose usage errors still print a token.

    argparse's own ``error()`` writes usage to stderr and exits 2 with nothing on stdout, so
    a missing --checkpoint or a bad --blank_camera would look identical to a server still
    starting up to the supervisor that waits on this script's stdout. ``--help`` is
    deliberately NOT covered: it exits through ``exit()`` rather than ``error()``, and a
    machine-readable verdict printed after a help text would be nonsense.
    """

    def error(self, message: str) -> None:
        print(f"SERVER_FAIL usage: {message}", flush=True)
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = TokenArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="a pretrained_model directory")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=None)
    parser.add_argument(
        "--action_warp",
        type=float,
        default=1.0,
        help=f"play each chunk this many times faster, {ACTION_WARP_MIN} (off) to {ACTION_WARP_MAX}",
    )
    parser.add_argument(
        "--blank_camera",
        choices=BLANKABLE_STREAMS,
        default=None,
        help="occlusion ablation: zero this camera stream before inference (diagnostic only)",
    )
    parser.add_argument(
        "--ensemble_with",
        default=None,
        help="MS7g D20: a second pretrained_model directory; serve the mean of both "
        "checkpoints' unnormalized actions, each through its own pre/post processor",
    )
    return parser


def main() -> int:
    # Everything that can fail on user input is inside the token contract: argparse failures
    # go through TokenArgumentParser.error, and everything else -- an unloadable checkpoint,
    # a bound port, a bad override pair -- becomes SERVER_FAIL below. Nothing that can fail
    # runs at module scope, where no token could be printed. serve_forever() is inside the
    # try as well: by then SERVER_READY has been printed, so the token is no longer what the
    # supervisor is waiting on, but a named failure still beats a bare traceback.
    parser = build_parser()
    try:
        args = parser.parse_args()
        (
            policy_fn,
            reset_fn,
            policy_id,
            n_action_steps,
            inference,
            context_dim,
            action_space,
            gripper,
        ) = load_act(
            args.checkpoint,
            args.device,
            n_action_steps=args.n_action_steps,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
            blank_camera=args.blank_camera,
            action_warp=args.action_warp,
            ensemble_with=args.ensemble_with,
        )
        server = PolicyServer(
            policy_fn,
            reset_fn,
            policy_id,
            args.port,
            n_action_steps,
            inference=inference,
            context_dim=context_dim,
            action_space=action_space,
            gripper=gripper,
        )
        server.bind()
        server.serve_forever()
    except Exception as exc:  # noqa: BLE001 -- any failure must become SERVER_FAIL, not a crash.
        print(f"SERVER_FAIL {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

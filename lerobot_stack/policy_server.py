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
import traceback

import numpy as np
import zmq

from policy_protocol import (
    ACTION_DTYPE,
    ACTION_SHAPE,
    ANCHOR_KEY,
    CONTEXT_KEY,
    OBS_SCHEMA,
    PROTOCOL_VERSION,
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
    ):
        self._policy_fn = policy_fn
        self._reset_fn = reset_fn
        self._policy_id = policy_id
        self._port = port
        self._n_action_steps = n_action_steps
        self._action_space = action_space
        # The inference config the loaded policy actually runs with (n_action_steps,
        # temporal_ensemble_coeff, chunk_size), read off the policy after the overrides were
        # applied and echoed in SERVER_READY and hello_ack so an eval payload records what
        # was served rather than what the supervisor asked for.
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
        ready = {
            "port": self._port,
            "policy_id": self._policy_id,
            "n_action_steps": self._n_action_steps,
            "protocol": PROTOCOL_VERSION,
            "inference": self._inference,
            "context_dim": self._context_dim,
            "action_space": self._action_space,
        }
        print("SERVER_READY " + json.dumps(ready), flush=True)
        return self._port

    def _hello_ack(self) -> dict:
        return {
            "type": "hello_ack",
            "protocol": PROTOCOL_VERSION,
            "policy_id": self._policy_id,
            "n_action_steps": self._n_action_steps,
            "inference": self._inference,
            "context_dim": self._context_dim,
            "action_space": self._action_space,
            "obs_schema": {k: [list(shape), dtype] for k, (shape, dtype) in OBS_SCHEMA.items()},
            "action": [list(ACTION_SHAPE), ACTION_DTYPE],
        }

    def handle_one(self) -> bool:
        header, arrays = recv_message(self._socket)
        kind = header.get("type")
        try:
            if kind == "hello":
                if header.get("protocol") != PROTOCOL_VERSION:
                    send_message(
                        self._socket,
                        {
                            "type": "error",
                            "error": "protocol_mismatch",
                            "message": f"server speaks {PROTOCOL_VERSION}, client sent {header.get('protocol')}",
                        },
                    )
                else:
                    send_message(self._socket, self._hello_ack())
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
                action = np.asarray(
                    self._policy_fn(arrays["state"], arrays["wrist"], arrays["overhead"], context),
                    dtype=np.float32,
                )
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
                send_message(self._socket, {"type": "action"}, {"action": action})
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


# --------------------------------------------------------------------------------------
# BCE gripper head -- PORTED BUT INACTIVE IN THIS REPO.
#
# openyam-lab trained a variant whose gripper dim is a BCE logit rather than an L1 output,
# and its trainer (train_act_gripper.py, NOT ported) dropped a gripper_bce.json marker next
# to the checkpoint. NOTHING IN THIS REPO WRITES THAT FILE: find_gripper_bce_marker()
# returns None for every checkpoint this milestone trains, squash_gripper_logit() is never
# reached on a live path, and the "+gbce" policy-id suffix never appears.
#
# Kept, not deleted: it is tested, self-contained, and a ported BCE trainer would need no
# server-side change. Marked, not silent: do not go looking for the producer of the marker
# file when debugging a gripper problem here -- there is no producer, and the None path is
# the only one that runs.
# --------------------------------------------------------------------------------------

GRIPPER_DIM = 6  # the ACTION's gripper column (policy_protocol.STATE_GRIPPER_DIM is the state's)


def find_gripper_bce_marker(checkpoint_dir: str) -> dict | None:
    """Locate the gripper_bce.json a BCE-head training run left behind.

    openyam-lab's train_act_gripper.py wrote it to the run's output_dir, while the
    server is handed output_dir/checkpoints/<step>/pretrained_model -- so look in
    the checkpoint dir itself, then walk up the three levels to output_dir.
    Returns the parsed marker, or None for stock-L1 checkpoints. No trainer in this
    repo emits one, so every checkpoint produced here takes the None path today; the
    lookup is kept so a ported BCE trainer needs no change on the server side.
    """
    from pathlib import Path

    directory = Path(checkpoint_dir)
    for candidate in [directory, *directory.parents[:3]]:
        marker = candidate / "gripper_bce.json"
        if marker.is_file():
            return json.loads(marker.read_text())
    return None


ACTION_SPACE_MARKER = "action_space.json"


def find_action_space_marker(checkpoint_dir: str) -> str:
    """The action space train_act.sh recorded next to the checkpoints, "absolute" if none.

    Same walk as find_gripper_bce_marker: the checkpoint dir, then up to output_dir.
    """
    from pathlib import Path

    directory = Path(checkpoint_dir)
    for candidate in [directory, *directory.parents[:3]]:
        marker = candidate / ACTION_SPACE_MARKER
        if marker.is_file():
            return json.loads(marker.read_text())["action_space"]
    return "absolute"


def integrate_delta(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """A delta_prev checkpoint's joint part plus the client's anchor, the previous clamped
    arm command: an absolute target again. The gripper column is a command, not a position,
    and is left alone.
    """
    out = action.copy()
    out[:, :6] += state[:, :6]
    return out


def squash_gripper_logit(action: np.ndarray, raw_mean: float, raw_std: float) -> np.ndarray:
    """Turn the unnormalized BCE logit back into a unit gripper action. INACTIVE here.

    Unreachable in this repo -- see the section marker above; nothing produces the marker
    file that switches it on.

    A BCE-head checkpoint emits dim 6 as a logit, which the checkpoint's own
    unnormalizer then treats as a normalized action: raw = logit * std + mean,
    landing well outside the +-1 the protocol promises. (openyam-lab measured
    near +-8 on its own action stats -- an OPENYAM measurement, unmeasured on
    this arm; the inversion below does not depend on the magnitude, only on the
    (mean, std) it is handed.) Invert that and squash with 2*sigmoid(logit) - 1
    (== tanh(logit/2)), so the trained p=0.5 boundary (logit 0) lands exactly at
    raw 0 -- the gripper convention's open threshold, since a raw action >= 0
    opens and < 0 closes. The caller's array is not modified.
    """
    action = action.copy()
    logit = (action[:, GRIPPER_DIM] - raw_mean) / raw_std
    action[:, GRIPPER_DIM] = np.tanh(logit / 2.0)
    return action


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
    gripper_squash: bool = False,
    context_dim: int = 0,
    action_space: str = "absolute",
) -> str:
    """Suffix the policy id with the active overrides so eval payloads are self-describing."""
    policy_id = f"act:{checkpoint_dir}"
    if "temporal_ensemble_coeff" in overrides:
        policy_id += f"+te{overrides['temporal_ensemble_coeff']}"
    elif "n_action_steps" in overrides:
        policy_id += f"+nas{overrides['n_action_steps']}"
    if gripper_squash:
        policy_id += "+gbce"
    if blank_camera is not None:
        policy_id += f"+blank{blank_camera}"
    if context_dim:
        policy_id += f"+ctx{context_dim}"
    if action_space == "delta_prev":
        policy_id += "+dprev"
    return policy_id


def load_act(
    checkpoint_dir: str,
    device: str,
    n_action_steps: int | None = None,
    temporal_ensemble_coeff: float | None = None,
    blank_camera: str | None = None,
):
    """Build (policy_fn, reset_fn, policy_id, n_action_steps, inference, context_dim, action_space).

    checkpoint_dir is a `pretrained_model` directory (e.g.
    <output_dir>/checkpoints/last/pretrained_model). The checkpoint's own
    saved pre/post processor pipelines carry normalization + device transfer,
    the same way lerobot_eval loads them (verified against lerobot 0.5.1:
    `preprocessor_overrides={"device_processor": {"device": ...}}`).
    Image conversion (uint8 HWC -> float32 CHW / 255) mirrors
    lerobot.envs.utils.preprocess_observation; images are sent to the policy only
    when the checkpoint has image features. `context_dim` is the checkpoint's
    `env_state_feature.shape[0]`, or 0 for a checkpoint that takes no context.
    Inference-config overrides (n_action_steps, temporal_ensemble_coeff) are applied
    at checkpoint load and reflected in the returned policy_id suffix.
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
    marker = find_gripper_bce_marker(checkpoint_dir)
    action_space = find_action_space_marker(checkpoint_dir)
    if action_space == "delta":
        raise ValueError(
            "this checkpoint was trained on the c1 delta definition (target minus the measured "
            "joints of the same frame), which MS4 c2 removed; only absolute and delta_prev serve"
        )
    if marker is not None and ("gripper_raw_mean" not in marker or "gripper_raw_std" not in marker):
        raise ValueError(
            "gripper_bce.json next to this checkpoint predates the logit-scale "
            "fix and lacks gripper_raw_mean/gripper_raw_std; backfill them from "
            "the training dataset's meta/stats.json action mean/std at dim 6"
        )

    def _images(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(torch.float32) / 255.0

    def policy_fn(
        state: np.ndarray, wrist: np.ndarray, overhead: np.ndarray, context: np.ndarray | None
    ) -> np.ndarray:
        batch = {"observation.state": torch.from_numpy(state)}
        if uses_images:
            wrist, overhead = blank_streams(wrist, overhead, blank_camera)
            batch["observation.images.wrist"] = _images(wrist)
            batch["observation.images.overhead"] = _images(overhead)
        if context_dim:
            batch["observation.environment_state"] = torch.from_numpy(context)
        batch = pre(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        action = np.asarray(action.detach().to("cpu"), dtype=np.float32)
        if marker is not None:
            action = squash_gripper_logit(
                action, marker["gripper_raw_mean"], marker["gripper_raw_std"]
            )
        return action

    return (
        policy_fn,
        policy.reset,
        inference_policy_id(
            checkpoint_dir,
            overrides,
            blank_camera,
            gripper_squash=marker is not None,
            context_dim=context_dim,
            action_space=action_space,
        ),
        policy.config.n_action_steps,
        served_inference(policy.config),
        context_dim,
        action_space,
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
        "--blank_camera",
        choices=BLANKABLE_STREAMS,
        default=None,
        help="occlusion ablation: zero this camera stream before inference (diagnostic only)",
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
        policy_fn, reset_fn, policy_id, n_action_steps, inference, context_dim, action_space = (
            load_act(
                args.checkpoint,
                args.device,
                n_action_steps=args.n_action_steps,
                temporal_ensemble_coeff=args.temporal_ensemble_coeff,
                blank_camera=args.blank_camera,
            )
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

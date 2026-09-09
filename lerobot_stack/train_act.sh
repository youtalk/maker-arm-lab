#!/usr/bin/env bash
# ACT training wrapper (design spec section 5, acceptance gate): LeRobot 0.5.1
# directly, ACT preset hyperparameters, no validation split (the gate is an
# in-sim rollout evaluation; held-out data would buy nothing the eval doesn't
# measure). MS1 only puts the STEPS=100 smoke through this script -- training
# proper is the next milestone -- but the defaults below are the real ones, so
# the smoke exercises the path a real run would take.
# push_to_hub=false: this host has no hub login and the policy is local-only.
# env=null (the default): lerobot-train skips env eval when cfg.env is None,
# so no --eval_freq juggling is needed.
# wandb: enabled by default in `online` mode, which requires a one-time
# `.venv/bin/wandb login` on this host -- without it wandb.init prompts for an
# API key on a TTY and fails outright without one, so an unattended run dies at
# startup rather than training silently un-logged. Set WANDB_MODE=offline to
# log to "$OUT"/wandb/ with no account (sync later with `wandb sync`), or
# WANDB_ENABLE=false to turn logging off entirely.
# CAVEAT carried over from openyam-lab's M3, and NOT re-established on this
# arm: in-distribution loss and the rollout gate rate were decoupled there (a
# checkpoint that halved serving-path action error scored worse on the gate),
# so these curves are a liveness/throughput/ETA instrument, not a
# checkpoint-selection one. MS1 runs no rollout gate at all, so nothing in this
# milestone re-tests it; the milestone that trains for real is what would.
# disable_artifact=true: log_policy() would upload the full checkpoint at every
# --save_freq -- 10 of them at the default 100k steps, and MS1's own train smoke
# measured this policy at 51,599,239 learnable parameters for a 197 MB
# model.safetensors, so that is ~2 GB of uploads per run. They are already on
# local disk under "$OUT", which is where any later gate eval reads them from.
# tolerance_s=0.01: carried over unchanged from openyam-lab, where a v3.0
# dataset's video files concatenated many episodes into single files up to
# ~5278 s long, and float32's rounding error on timestamps that large (ULP
# ~4.9e-4 s) exceeded lerobot's 1e-4 default -- so its DataLoader workers hit a
# deterministic FrameTimestampError partway through every run. That length has
# not been re-measured on the Maker Arm dataset, which is much smaller. 0.01 s
# is kept because it is harmless at any size: it is still well under half a
# frame period (16.7 ms at the 30 fps convert_to_lerobot.py writes), so it
# cannot select the wrong adjacent frame. The MS1 train smoke is what
# re-validates that it does not break this dataset.
set -euo pipefail

# Token contract: TRAIN_OK on a run that completed, TRAIN_FAIL on every other
# exit. An EXIT trap is the mechanism, rather than an echo at the end of the
# script, because `set -e` and the WANDB_ENTITY guard below both leave through
# paths a trailing echo would never reach -- the same "not every exit path is
# tokenised" defect found in scripted_grasp_check.py and policy_server.py.
# lerobot-train is therefore deliberately NOT exec'd: a process replaced by
# exec never runs its traps. The cost is that this shell stays as the child's
# parent, so a SIGTERM addressed to this PID alone would orphan lerobot-train;
# a signal to the process group (Ctrl-C, or the group kill a sweep uses)
# reaches both, which is how long runs on this host are actually stopped.
emit_train_token() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        echo "TRAIN_OK out=${OUT:-unset} steps=${STEPS:-unset}"
    else
        echo "TRAIN_FAIL status=$status out=${OUT:-unset} steps=${STEPS:-unset}"
    fi
}
trap emit_train_token EXIT

cd "$(dirname "$0")"

STEPS="${STEPS:-100000}"
OUT="${OUT:-$HOME/data/maker-arm-ms1/train/act_smoke}"
JOB_NAME="${JOB_NAME:-act_smoke}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/maker-arm-ms1/dataset}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-maker-arm-lab}"
WANDB_MODE="${WANDB_MODE:-online}"
# youtalk-openyam-lab is deliberately kept: it is the owner's personal wandb
# entity, named after the project it was created for. It is the entity all of
# this lab's runs live under, so renaming it here would split the history
# rather than tidy it.
# `-` and not `:-`, so that an explicitly empty WANDB_ENTITY= reaches the guard
# below instead of being silently replaced by this default. openyam-lab used
# `:-` here, which made that guard unreachable while its comment claimed the
# opposite; verified on this host by running the script with WANDB_ENTITY= and
# watching lerobot-train start under the default entity anyway.
WANDB_ENTITY="${WANDB_ENTITY-youtalk-openyam-lab}"

# The entity is named explicitly because lerobot's WandBConfig.entity defaults
# to None and this wandb account has no default entity -- an online run
# without one dies several frames deep inside wandb with "entityName required
# for model query (user has no default)". WANDB_ENTITY= (empty) still reaches
# that path, so fail here instead, where the message says what to set.
WANDB_FLAGS=(
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
    --wandb.mode="$WANDB_MODE"
    --wandb.disable_artifact=true
)
if [ -n "$WANDB_ENTITY" ]; then
    WANDB_FLAGS+=(--wandb.entity="$WANDB_ENTITY")
elif [ "$WANDB_ENABLE" = true ] && [ "$WANDB_MODE" = online ]; then
    echo "train_act.sh: set WANDB_ENTITY=<your wandb entity> for an online run" >&2
    echo "  (or WANDB_MODE=offline / WANDB_ENABLE=false to train without it)" >&2
    exit 1
fi

.venv/bin/lerobot-train \
    --dataset.repo_id=youtalk/maker-arm-legosort \
    --dataset.root="$DATASET_ROOT" \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir="$OUT" \
    --job_name="$JOB_NAME" \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=8 \
    --save_freq=10000 \
    --log_freq=100 \
    --tolerance_s=0.01 \
    "${WANDB_FLAGS[@]}"

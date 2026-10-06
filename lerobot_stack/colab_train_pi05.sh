#!/usr/bin/env bash
# pi0.5 fine-tune on Google Colab (MS7h design 7.2): stock lerobot-train 0.5.1 on the dataset
# derive_vla_dataset.py builds, from lerobot/pi05_base (revision pinned below), a full fine-tune
# in bfloat16 with gradient checkpointing on an A100 80GB. The notebook stays outside the
# repository. Its cells, in order:
#   1. from google.colab import drive; drive.mount("/content/drive")
#   2. import os; from google.colab import userdata
#      os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
#      (a Colab secret named HF_TOKEN with notebook access on. A subprocess cannot read Colab
#      secrets itself, so without this export the script sees no token.)
#   3. !DIR=<Drive folder> bash colab_train_pi05.sh --smoke   (then without --smoke)
#
# Drive layout under /content/drive/MyDrive/$DIR:
#   dataset/              the derived dataset (DATASET=dataset_third for the one-third copy)
#   checkpoints/<step>/   the last complete checkpoint, with its training state
#   checkpoints/last.txt  its step directory name
#   action_space.json     the dataset's gripper contract, found by policy_server's marker walk
# The dataset is copied to the local disk once a session (Drive is too slow for random frame
# reads). lerobot-train checkpoints to the local disk every SAVE_FREQ steps; a loop copies each
# new complete checkpoint (the one checkpoints/last points to) to Drive and then deletes the older
# ones on Drive and on the local disk, so Drive holds one checkpoint, two while one is copied. A
# session that ends (24 hours at most) resumes from the Drive copy: run the same cell again.
#
# The pi05 tokenizer google/paligemma-3b-pt-224 is a gated repo: HF_TOKEN must belong to an
# account that was granted access (the owner's was, 2026-10-05). The script checks that first,
# before it copies anything. transformers 5.3's AutoTokenizer reads the repo's config.json as
# well as the tokenizer files, so an offline host needs config.json in its HF cache too.
#
# usage: DIR=<Drive folder> bash colab_train_pi05.sh [--smoke] [--expert_only]
#   --smoke        200 steps into a fresh local dir with one checkpoint at the end and no Drive
#                  writes; prints seconds a step and the checkpoint size (MS7h design 5.4 step 5)
#   --expert_only  the A100 40GB fallback, train_expert_only=true (the report must say so)
# env: DATASET (dataset), STEPS (30000), BATCH (32), SAVE_FREQ (2000), NUM_WORKERS (8),
#      OUT_ROOT (/content): where the local checkpoints go, e.g. the runtime's local-scratch disk
#      (the owner's A100 runtime had 70 GB free on /content after the copy and 368 GB there).
#      Keep it the same across sessions: a resumed train_config.json names the old output dir
# -> one COLAB_TRAIN_OK / COLAB_TRAIN_FAIL line.
set -euo pipefail

emit_token() {
    local status=$?
    if [ "$status" -eq 0 ]; then
        echo "COLAB_TRAIN_OK ${RESULT:-}"
    else
        echo "COLAB_TRAIN_FAIL status=$status ${REASON:-}"
    fi
}
trap emit_token EXIT
fail() {
    REASON=$1
    echo "colab_train_pi05.sh: $1" >&2
    exit 1
}

SMOKE=0
EXPERT_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --expert_only) EXPERT_ONLY=true ;;
        *) fail "unknown argument $arg" ;;
    esac
done
[ -n "${DIR:-}" ] || fail "set DIR=<folder under /content/drive/MyDrive>"
DRIVE=/content/drive/MyDrive/$DIR
DATA=${DATASET:-dataset}
STEPS=${STEPS:-30000} # INITIAL: MS7h design 7.2, about a tenth of the frames at batch 32
BATCH=${BATCH:-32}
SAVE_FREQ=${SAVE_FREQ:-2000}
NUM_WORKERS=${NUM_WORKERS:-8}
LOCAL_DATA=/content/$DATA
OUT=${OUT_ROOT:-/content}/pi05_out
LOG_FREQ=100
if [ "$SMOKE" -eq 1 ]; then
    STEPS=200 SAVE_FREQ=200 LOG_FREQ=10 OUT=/content/pi05_smoke
fi

[ -f "$DRIVE/$DATA/meta/info.json" ] || fail "no $DRIVE/$DATA/meta/info.json (Drive mounted?)"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' || fail "lerobot 0.5.1: Python 3.12"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$EXPERT_ONLY" = false ] && [ "$memory" -lt 70000 ]; then
    fail "the full fine-tune needs an A100 80GB, this GPU has $memory MiB (--expert_only?)"
fi

python3 -c 'import lerobot, sys; sys.exit(lerobot.__version__ != "0.5.1")' 2>/dev/null \
    || pip install -q "lerobot[pi]==0.5.1"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not set: export it from the Colab secret (cell 2)"
# A secret pasted with a newline sends a bad header, which the hub answers 401 "Please log in".
HF_TOKEN=$(printf %s "$HF_TOKEN" | tr -d '[:space:]')
export HF_TOKEN
# The owner's first smoke (2026-10-05) failed here on a 401 with HF_TOKEN set: tell a bad token
# from an account or token without access to the gated repo before the tokenizer reads it.
python3 -c 'import os, sys
from huggingface_hub import auth_check, whoami
try:
    name = whoami(token=os.environ["HF_TOKEN"])["name"]
except Exception as err:
    sys.exit(f"HF_TOKEN is not a valid token ({type(err).__name__}): copy it again into the secret")
try:
    auth_check("google/paligemma-3b-pt-224", token=os.environ["HF_TOKEN"])
except Exception as err:
    sys.exit(f"account {name}: no access to google/paligemma-3b-pt-224 ({type(err).__name__}); "
             "a fine-grained token needs read access to public gated repos")
print(f"HF_TOKEN: account {name} has access to google/paligemma-3b-pt-224")' \
    || fail "HF_TOKEN cannot read google/paligemma-3b-pt-224 (see the line above)"
python3 -c 'from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")' \
    || fail "no google/paligemma-3b-pt-224: HF_TOKEN of an account that accepted its licence"
# lerobot/pi05_base before its 2026-06-03 commit "Add relative action processor steps", whose
# preprocessor names relative_actions_processor, a step lerobot 0.5.1 registers as
# delta_actions_processor, so it fails to load. Same model.safetensors and config.json.
PI05_REV=a538eb273274eb30f126a118f39dbc0ee212c883
PI05_BASE=$(python3 -c 'import sys; from huggingface_hub import snapshot_download
print(snapshot_download("lerobot/pi05_base", revision=sys.argv[1]))' "$PI05_REV") \
    || fail "cannot download lerobot/pi05_base"

if [ ! -e "$LOCAL_DATA/.copied" ]; then
    rm -rf "$LOCAL_DATA"
    echo "copying $DRIVE/$DATA to $LOCAL_DATA"
    cp -r "$DRIVE/$DATA" "$LOCAL_DATA"
    touch "$LOCAL_DATA/.copied"
fi
df -h /content | tail -1
# 2026-10-05 on the RTX 5090 under a 19.44 GiB cap, train_expert_only at batch 32 with gradient
# checkpointing: out of memory after step 1 with the default allocator, 18.45 GiB allocated and
# 19.41 reserved at 3.0 s a step with expandable segments
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TRAIN=(
    --dataset.repo_id=youtalk/maker-arm-legosort
    --dataset.root="$LOCAL_DATA"
    --tolerance_s=0.01 # as train_act.sh: float32 timestamps in long v3.0 video files
    --policy.type=pi05
    --policy.pretrained_path="$PI05_BASE"
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    --policy.train_expert_only="$EXPERT_ONLY"
    --policy.chunk_size=50
    --policy.n_action_steps=50
    --policy.device=cuda
    --policy.push_to_hub=false
    --output_dir="$OUT"
    --job_name=pi05_ms7h
    --steps="$STEPS"
    --batch_size="$BATCH"
    --num_workers="$NUM_WORKERS"
    --save_checkpoint=true
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --wandb.enable=false
)

if [ "$SMOKE" -eq 1 ]; then
    rm -rf "$OUT"
    lerobot-train "${TRAIN[@]}" 2>&1 | tee /content/pi05_smoke.log
    # the mean of updt_s + data_s over the logged steps past step 50 (warmup)
    sec=$(awk '/ step:/ {
        for (i = 1; i <= NF; i++) {
            split($i, kv, ":")
            if (kv[1] == "step") s = kv[2] + 0
            if (kv[1] == "updt_s") u = kv[2]
            if (kv[1] == "data_s") d = kv[2]
        }
        if (s > 50) { n++; t += u + d }
    } END { if (n) printf "%.3f", t / n }' /content/pi05_smoke.log)
    [ -n "$sec" ] || fail "no step lines in /content/pi05_smoke.log"
    gib=$(du -sb "$OUT/checkpoints/last/" | awk '{printf "%.2f", $1 / 2^30}')
    RESULT="smoke sec_per_step=$sec checkpoint_gib=$gib batch=$BATCH expert_only=$EXPERT_ONLY"
    exit 0
fi

mkdir -p "$DRIVE/checkpoints"
cp "$LOCAL_DATA/meta/action_space.json" "$DRIVE/action_space.json"
sync_last() { # copy a new complete local checkpoint to Drive, then drop the older ones
    local last synced old
    last=$(readlink "$OUT/checkpoints/last" 2>/dev/null) || return 0
    synced=$(cat "$DRIVE/checkpoints/last.txt" 2>/dev/null || true)
    [ "$last" != "$synced" ] || return 0
    # chained: set -e is off inside a function called from ||, and last.txt must never name
    # a checkpoint that is not complete on Drive
    rm -rf "$DRIVE/checkpoints/.tmp_$last" \
        && cp -r "$OUT/checkpoints/$last" "$DRIVE/checkpoints/.tmp_$last" \
        && mv "$DRIVE/checkpoints/.tmp_$last" "$DRIVE/checkpoints/$last" \
        && echo "$last" > "$DRIVE/checkpoints/last.txt" || return 1
    for old in "$DRIVE"/checkpoints/[0-9]* "$OUT"/checkpoints/[0-9]*; do
        [ -e "$old" ] || continue
        if [[ "$(basename "$old")" < "$last" ]]; then rm -rf "$old"; fi
    done
    echo "synced checkpoint $last to Drive"
}

sync_last # a complete local checkpoint of an earlier run in this session reaches Drive first
if [ -f "$DRIVE/checkpoints/last.txt" ]; then
    last=$(cat "$DRIVE/checkpoints/last.txt")
    if [ ! -d "$OUT/checkpoints/$last" ]; then
        rm -rf "$OUT"
        mkdir -p "$OUT/checkpoints"
        cp -r "$DRIVE/checkpoints/$last" "$OUT/checkpoints/$last"
    fi
    ln -sfn "$last" "$OUT/checkpoints/last"
    echo "resuming from checkpoint $last"
    config=$OUT/checkpoints/last/pretrained_model/train_config.json
    CMD=(lerobot-train --config_path="$config" --resume=true)
else
    rm -rf "$OUT" # no complete checkpoint anywhere, so nothing to keep
    CMD=(lerobot-train "${TRAIN[@]}")
fi

("${CMD[@]}" 2>&1 | tee -a /content/pi05_train.log) &
train_pid=$!
while kill -0 "$train_pid" 2>/dev/null; do
    sync_last || echo "colab_train_pi05.sh: the Drive copy failed, next try in 60 s" >&2
    sleep 60
done
wait "$train_pid" || { sync_last; fail "lerobot-train failed, see /content/pi05_train.log"; }
sync_last
RESULT="steps=$STEPS last=$(cat "$DRIVE/checkpoints/last.txt") drive=$DRIVE/checkpoints"

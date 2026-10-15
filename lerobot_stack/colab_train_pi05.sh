#!/usr/bin/env bash
# pi0.5 fine-tune on Google Colab (MS7h design 7.2, recipe 5.6 and 9): train_pi05.py (the loss
# masked on padded actions) under pi05_preflight.py, a warm start from START with a fresh
# optimizer, on an A100 80GB. The notebook stays outside the repository. Its cells, in order:
#   1. from google.colab import drive; drive.mount("/content/drive")
#   2. import os; from google.colab import userdata
#      os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
#      (a Colab secret named HF_TOKEN with notebook access on. A subprocess cannot read Colab
#      secrets itself, so without this export the script sees no token.)
#   3. !DIR=<folder> START=<weights> RUN_NAME=<run> \
#          bash /content/drive/MyDrive/<folder>/colab_train_pi05.sh --smoke
#      then the same cell without --smoke. After a session ends, run the same cell again. On a
#      Shared drive, set DRIVE_ROOT=/content/drive/Shareddrives/<drive name> and run the script
#      from there. The fresh run from pi05_base (2026-10-09), 60000 steps over three sessions,
#      kept for a read at every 10000 steps:
#      !DRIVE_ROOT=/content/drive/Shareddrives/maker-arm-lab DIR=maker-arm-ms7h-pi05 \
#          START=lerobot/pi05_base RUN_NAME=checkpoints_fresh STEPS=60000 SPAN=1 \
#          KEEP_EVERY=10000 DRIVE_MIN_GB=0 OUT_ROOT=/mnt/local-scratch \
#          bash /content/drive/Shareddrives/maker-arm-lab/maker-arm-ms7h-pi05/colab_train_pi05.sh \
#          --smoke
#
# Drive layout under $DRIVE_ROOT/$DIR (DRIVE_ROOT /content/drive/MyDrive by default):
#   colab_train_pi05.sh, pi05_preflight.py, train_pi05.py   this script and its two helpers
#   dataset/              the derived dataset (DATASET=dataset_third for the one-third copy)
#   $START                the pretrained_model directory the run starts from, e.g.
#                         checkpoints/030000/pretrained_model (run 1) or checkpoints_ab/<step>/...
#                         START=lerobot/pi05_base is downloaded instead and is not on Drive
#   $RUN_NAME/<step>/     the run's last complete checkpoint; one folder a run, e.g.
#                         checkpoints_ab, checkpoints_c1, checkpoints_c2
#   $RUN_NAME/last.txt    its step directory name
#   $RUN_NAME/keep/<step>/pretrained_model  with KEEP_EVERY, the weights (no training_state,
#                         9.35 GB) of each checkpoint whose step is a multiple of KEEP_EVERY, for
#                         the reads; the sync never deletes them
#   action_space.json     the dataset's gripper contract, found by policy_server's marker walk
#
# A fresh run copies START to the local disk, sets its saved tokenizer step to MAX_LENGTH tokens
# (a warm start reads the saved policy_preprocessor.json and no config.json), and passes every
# recipe flag again; lerobot builds a new optimizer and a new schedule. START=lerobot/pi05_base
# downloads run 1's start instead (da1f27a), at the revision pinned below; its normalizer takes
# the dataset's statistics as a warm start's does (run 1's 030000 holds the corpus stats). STEPS
# is the decay length. Its default is N, the steps that fit this session: min(STEPS_MAX, (24 h -
# uptime - 0.4 h) / SEC_PER_STEP), since the 24 hours count from the creation of the runtime
# (F11). A fresh run stops if STEPS > N, unless SPAN=1: then a session trains to its last
# checkpoint that fits (the largest multiple of the save freq <= its start step + N), stops the
# trainer once that checkpoint is on Drive and prints COLAB_TRAIN_OK with session_end=1; the next
# session resumes toward the same STEPS on the same schedule (lerobot restores the optimizer and
# the scheduler). With KEEP_EVERY, each synced checkpoint whose step is a multiple of it also
# copies its pretrained_model to $RUN_NAME/keep/<step>.
# The dataset is copied to the local disk once a session (Drive is too slow for random frame
# reads). lerobot checkpoints to the local disk every SAVE_FREQ steps; a loop copies each new
# complete checkpoint (the one checkpoints/last points to) to Drive and then deletes the older
# ones. When Drive lacks the room for a second copy, the older one goes first, so Drive never
# holds two (F7). A session that ends resumes from the Drive copy. A run that reached its last
# step does not resume: it deletes its training_state on Drive and prints COLAB_TRAIN_DONE.
#
# The run stops on (F3, recipe 5.6):
#   - Drive under DRIVE_MIN_GB free at the start, plus KEEP_GB for each copy that KEEP_EVERY
#     keeps from here to STEPS (files deleted through the Colab mount may wait in the Drive trash
#     and still count, so look there if this fires). The free space is what df reads on the Drive
#     mount, the account's own quota; on a Shared drive that is most likely not the shared drive's
#     room, so DRIVE_MIN_GB=0 skips these checks there
#   - a corpus meta/stats.json whose md5 is not STATS_MD5
#   - a preflight miss before the first update (pi05_preflight.py: the full weight load, the
#     normalizer statistics against the corpus file, MAX_LENGTH tokens with the longest prompt
#     under it, action_is_pad at the masked forward), or a trainer that logs a step with no
#     PREFLIGHT_OK line before it
#   - a loss outside [LOSS_LO, LOSS_HI] at step 200 (another default band from pi05_base)
#   - SPAN=1 with no checkpoint past the start step that fits this session, or a KEEP_EVERY that
#     is not a multiple of the save freq
#   - a final Drive sync that still fails after SYNC_RETRY_S
#
# The mix mode (MS7h HD59 with one root, MS7i Run A and Run B with two), with MIX="ROOT:SHARE ...":
# the run trains on the corpus and each ROOT in one mix (train_pi05.py --mix=ROOT:SHARE, a draw is
# a frame of ROOT with probability SHARE, a corpus frame with 1 minus their sum), a warm start from
# START at its own recipe: warmup 200 to a peak lr of 1e-5, then cosine to 2.5e-6 over STEPS (8000
# by default, not N), every checkpoint kept (KEEP_EVERY defaults to the save freq), and its own
# step-200 loss band. A ROOT under $DRIVE (a relative path) is copied to $OUT_ROOT/mix_<path> once
# a session, as the corpus is, so a browser runtime gets every input from the Shared drive; an
# absolute ROOT is used where it is. Every ROOT must carry the corpus's meta/stats.json
# (STATS_MD5): the corpus normalizer trains them all. PROMPT_CODES=<a file under $DRIVE> (MS7i
# Run A) is the corpus's meta/prompt_codes.npy: it is copied into the local corpus copy each
# session, every ROOT must hold its own, and train_pi05.py gets --prompt_codes. A run keeps
# "MIX|PROMPT_CODES" in $RUN_NAME/mix.txt, and a resume needs the same two again.
# DAGGER=<root> [DAGGER_SHARE=0.3] stays as the one-root MIX that colab_dagger.sh sets on a CLI
# runtime.
#
# The pi05 tokenizer google/paligemma-3b-pt-224 is a gated repo: HF_TOKEN must belong to an
# account that was granted access (the owner's was, 2026-10-05). The script checks that first,
# before it copies anything. transformers 5.3's AutoTokenizer reads the repo's config.json as
# well as the tokenizer files, so an offline host needs config.json in its HF cache too.
#
# usage: DIR=<Drive folder> START=<dir | lerobot/pi05_base> RUN_NAME=<name> bash
#            colab_train_pi05.sh [--smoke | --prepare] [--expert_only]
#   --smoke        200 steps from START into a local dir, with the preflight and the loss band,
#                  one checkpoint at the end and no Drive writes; prints seconds a step and the
#                  checkpoint size. RUN_NAME is not needed.
#   --prepare      the smoke's checks, the lerobot install, the HF_TOKEN checks and the dataset
#                  copy, then stop before the trainer: COLAB_TRAIN_OK prepared
#   --expert_only  train_expert_only=true, the A100 40GB fallback (the report must say so)
# env: STEPS (N), BATCH (32), SAVE_FREQ (2000), NUM_WORKERS (8), SEED (1000, run 1's and M2k's:
#      the same seed draws the same sample order), DATASET (dataset),
#      OUT_ROOT (/content): where the local checkpoints and the start copy go, e.g. the
#      runtime's local-scratch disk (the owner's A100 runtime had 70 GB free on /content after
#      the dataset copy and 368 GB there), LOSS_LO, LOSS_HI, DRIVE_MIN_GB (60), STATS_MD5,
#      SPAN (0), KEEP_EVERY (none), MAX_LENGTH (128), DRIVE_ROOT (/content/drive/MyDrive),
#      LOCAL_DATA ($CONTENT/$DATASET: where the dataset copy goes), MIX (none), PROMPT_CODES
#      (none), DAGGER (none), DAGGER_SHARE (0.3).
#      CONTENT (/content), UPTIME_FILE (/proc/uptime), POLL_S (60) and SYNC_RETRY_S (1800) are
#      for the dry-run test (lerobot_stack/test_colab_train_pi05.py).
# -> one COLAB_TRAIN_OK / COLAB_TRAIN_DONE / COLAB_TRAIN_FAIL line.
set -euo pipefail

emit_token() {
    local status=$? pid
    for pid in ${train_pid:-} ${tail_pid:-}; do kill "$pid" 2>/dev/null || true; done
    if [ "$status" -eq 0 ]; then
        echo "${TOKEN:-COLAB_TRAIN_OK} ${RESULT:-}"
    else
        echo "COLAB_TRAIN_FAIL status=$status ${REASON:-}"
    fi
}
trap emit_token EXIT
trap 'exit 130' INT TERM
fail() {
    REASON=$1
    echo "colab_train_pi05.sh: $1" >&2
    exit 1
}

SMOKE=0
PREPARE=0
EXPERT_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --prepare) SMOKE=1 PREPARE=1 ;;
        --expert_only) EXPERT_ONLY=true ;;
        *) fail "unknown argument $arg" ;;
    esac
done
[ -n "${DIR:-}" ] || fail "set DIR=<folder under DRIVE_ROOT, /content/drive/MyDrive by default>"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for f in pi05_preflight.py train_pi05.py; do
    [ -f "$HERE/$f" ] || fail "no $HERE/$f: put it next to this script"
done
CONTENT=${CONTENT:-/content}
DRIVE_ROOT=${DRIVE_ROOT:-$CONTENT/drive/MyDrive}
DRIVE=$DRIVE_ROOT/$DIR
DATA=${DATASET:-dataset}
LOCAL=${OUT_ROOT:-$CONTENT}
LOCAL_DATA=${LOCAL_DATA:-$CONTENT/$DATA}
BATCH=${BATCH:-32}
SAVE_FREQ=${SAVE_FREQ:-2000}
NUM_WORKERS=${NUM_WORKERS:-8}
SEED=${SEED:-1000}
POLL_S=${POLL_S:-60}
SYNC_RETRY_S=${SYNC_RETRY_S:-1800}
DRIVE_MIN_GB=${DRIVE_MIN_GB:-60}
KEEP_GB=10 # a kept pretrained_model: run 1's model.safetensors is 9,354,050,752 bytes
# md5 of h/vla/dataset/meta/stats.json on Machine A, the corpus uploaded to Drive (2026-10-08)
STATS_MD5=${STATS_MD5:-c10e2a394888ad72532a83779febdc3c}
# 128 by default: no prompt of the corpus reaches it, the longest is 114 tokens with BOS (review);
# MS7i Run A and Run B set 200 (MS7i design rev3 6.1), and the preflight checks the longest prompt
MAX_LENGTH=${MAX_LENGTH:-128}
[[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]] || fail "MAX_LENGTH=$MAX_LENGTH is not a token count"
BASE=lerobot/pi05_base
# its revision before the 2026-06-03 commit "Add relative action processor steps", whose
# preprocessor names relative_actions_processor, a step lerobot 0.5.1 registers as
# delta_actions_processor, so it fails to load (run 1's pin, da1f27a)
PI05_REV=a538eb273274eb30f126a118f39dbc0ee212c883
PEAK_LR=2.5e-5 MIX_WARMUP=
MIX=${MIX:-${DAGGER:+$DAGGER:${DAGGER_SHARE:-0.3}}}
read -ra MIX_ENTRIES <<< "$MIX"
MIX_ARGS=() MIX_SRC=() MIX_LOCAL=()
for entry in "${MIX_ENTRIES[@]}"; do
    root=${entry%:*}
    [[ "$entry" == *:* && -n "$root" && "${entry##*:}" =~ ^[0-9]*\.?[0-9]+$ ]] \
        || fail "MIX entry $entry: want ROOT:SHARE"
    case "$root" in
        /*) MIX_SRC+=("$root") MIX_LOCAL+=("$root") ;;
        *) MIX_SRC+=("$DRIVE/$root") MIX_LOCAL+=("$LOCAL/mix_${root//\//_}") ;;
    esac
    [ -f "${MIX_SRC[-1]}/meta/info.json" ] \
        || fail "no ${MIX_SRC[-1]}/meta/info.json: MIX root $root"
    MIX_ARGS+=(--mix="${MIX_LOCAL[-1]}:${entry##*:}")
done
if [ -n "${PROMPT_CODES:-}" ]; then
    [ -n "$MIX" ] || fail "PROMPT_CODES needs MIX"
    case "$PROMPT_CODES" in /*) CODES_SRC=$PROMPT_CODES ;; *) CODES_SRC=$DRIVE/$PROMPT_CODES ;; esac
    [ -f "$CODES_SRC" ] || fail "no PROMPT_CODES file $CODES_SRC"
    for src in "${MIX_SRC[@]}"; do
        [ -f "$src/meta/prompt_codes.npy" ] || fail "no $src/meta/prompt_codes.npy: PROMPT_CODES"
    done
    MIX_ARGS+=(--prompt_codes)
fi
if [ -n "$MIX" ]; then
    PEAK_LR=1e-5 MIX_WARMUP=200
    STEPS=${STEPS:-8000}
    KEEP_EVERY=${KEEP_EVERY:-$SAVE_FREQ}
    # INITIAL: a warm start from a fresh save on the mix, which the DAgger smoke's step 200
    # measures again. The fresh run read loss 0.033-0.035 at 20k (MS7h ledger 2026-10-10 20:38) on
    # the corpus; the DAgger frames are states the corpus never showed, at worst as far off as
    # the untrained base's 0.171 (the 2026-10-05 smoke), so 0.7 x 0.034 + 0.3 x 0.171 = 0.075 is
    # the worst mix of a sound start, and LOSS_HI 0.10 sits above it. It is a coarse stop (the
    # base itself reads 0.092 after 200 steps): the preflight's full-load check guards the
    # weights. LOSS_LO stays the warm band's.
    LOSS_LO=${LOSS_LO:-0.015}
    LOSS_HI=${LOSS_HI:-0.10}
fi
if [ "${START:-}" = "$BASE" ]; then
    # INITIAL: the band of a start from pi05_base, which this run's step 200 measures again.
    # The Colab smoke from it (2026-10-05: full fine-tune, batch 32, the loss not masked, 200
    # tokens, warmup 6) read loss 0.171 -> 0.092 over its 200 steps; the 5090 dry run from it
    # (expert only, batch 32, h/tables/vla/dry_run_real_b32gc.log) read 0.144 to 0.217 over its
    # first 10 steps. LOSS_LO is half of 0.092, as the warm band's is half of M2k's smallest;
    # LOSS_HI is twice 0.171, above every read of the untrained base. A run's step 200 sits in its
    # warmup (2000 steps for 60000), so its loss can stay near the untrained reads.
    LOSS_LO=${LOSS_LO:-0.046}
    LOSS_HI=${LOSS_HI:-0.34}
fi
# INITIAL: for a full fine-tune, which the A/B's step-200 loss measures again; set from the 5090
# M2k run (the 30k weights + 2000 masked expert-only steps, h/tables/vla/mask_ft.train.log):
# loss 0.033, 0.032, 0.031 at steps 150, 200, 250, and 0.029 to 0.033 over all its logged steps.
# LOSS_HI is about the midpoint of its largest and the 0.092 that the full fine-tune from
# pi05_base read at step 200 (the 2026-10-05 smoke): a start that lost its weights stops, and a
# fresh AdamW on all 4B weights has 1.8 times M2k's loss of room. LOSS_LO is half of M2k's
# smallest: a mask that drops most of the loss stops.
LOSS_LO=${LOSS_LO:-0.015}
LOSS_HI=${LOSS_HI:-0.06}
BAND_STEP=200
STEPS_MAX=20000
SEC_PER_STEP=3.85 # the A100 80GB full fine-tune of run 1, batch 32 (MS7h ledger 2026-10-07)
LOG=$CONTENT/pi05_${RUN_NAME:-smoke}_$(date +%Y%m%d_%H%M%S).log

[ -f "$DRIVE/$DATA/meta/info.json" ] || fail "no $DRIVE/$DATA/meta/info.json (Drive mounted?)"

drive_free_kb() { df -Pk "$DRIVE" | awk 'NR == 2 {print $4}'; }
run_cfg() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$@"; }
drop_older() { # delete the checkpoint directories older than $1 in each directory that follows
    local keep=$1 dir old
    shift
    for dir in "$@"; do
        for old in "$dir"/[0-9]*; do
            if [ -e "$old" ] && [[ "$(basename "$old")" < "$keep" ]]; then rm -rf "$old"; fi
        done
    done
}
sync_last() { # copy a new complete local checkpoint to Drive, then drop the older ones
    local last synced need keep= kb
    last=$(readlink "$OUT/checkpoints/last" 2>/dev/null) || return 0
    synced=$(cat "$RUN_DIR/last.txt" 2>/dev/null || true)
    [ "$last" != "$synced" ] || return 0
    # chained: set -e is off inside a function called from ||, and last.txt must never name
    # a checkpoint that is not complete on Drive
    mkdir -p "$RUN_DIR" && rm -rf "$RUN_DIR"/.tmp_* "$RUN_DIR"/keep/.tmp_* || return 1
    need=$(du -sk "$OUT/checkpoints/$last" | cut -f1) || return 1
    if [ -n "${KEEP_EVERY:-}" ] && [ $((10#$last % KEEP_EVERY)) -eq 0 ] \
        && [ ! -d "$RUN_DIR/keep/$last" ]; then
        keep=$RUN_DIR/keep/$last
        kb=$(du -sk "$OUT/checkpoints/$last/pretrained_model" | cut -f1) || return 1
        need=$((need + kb))
    fi
    if [ "$(drive_free_kb)" -lt $((need + 1000000)) ]; then
        echo "Drive is short of room for a second checkpoint: the older goes before $last"
        rm -f "$RUN_DIR/last.txt" && drop_older "$last" "$RUN_DIR" || return 1
    fi
    # the read copy goes before last.txt names $last, so a failed copy is tried again
    if [ -n "$keep" ]; then
        mkdir -p "$RUN_DIR/keep/.tmp_$last" \
            && cp -r "$OUT/checkpoints/$last/pretrained_model" "$RUN_DIR/keep/.tmp_$last/" \
            && mv "$RUN_DIR/keep/.tmp_$last" "$keep" || return 1
        echo "kept the pretrained_model of checkpoint $last in $keep"
    fi
    cp -r "$OUT/checkpoints/$last" "$RUN_DIR/.tmp_$last" \
        && mv "$RUN_DIR/.tmp_$last" "$RUN_DIR/$last" \
        && echo "$last" > "$RUN_DIR/last.txt" || return 1
    drop_older "$last" "$RUN_DIR" "$OUT/checkpoints"
    echo "synced checkpoint $last to Drive"
}

if [ "$SMOKE" -eq 0 ]; then
    [[ "${RUN_NAME:-}" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "set RUN_NAME=<the run's Drive folder>"
    RUN_DIR=$DRIVE/$RUN_NAME
    OUT=$LOCAL/pi05_$RUN_NAME
    # a complete local checkpoint of an earlier session in this runtime reaches Drive first
    sync_last || fail "cannot copy the local checkpoint to $RUN_DIR"
    if [ -f "$RUN_DIR/last.txt" ]; then
        last=$(cat "$RUN_DIR/last.txt")
        target=$(run_cfg "$RUN_DIR/$last/pretrained_model/train_config.json" steps) \
            || fail "no steps in $RUN_DIR/$last/pretrained_model/train_config.json"
        if [ "$((10#$last))" -eq "$target" ]; then
            rm -rf "$RUN_DIR/$last/training_state"
            TOKEN=COLAB_TRAIN_DONE
            RESULT="run=$RUN_NAME last=$last: the run reached its last step and does not resume"
            exit 0
        fi
    fi
fi

free_gb=$(($(drive_free_kb) * 1024 / 1000000000))
[ "$free_gb" -ge "$DRIVE_MIN_GB" ] || fail "Drive has $free_gb GB free, a run needs $DRIVE_MIN_GB"
# the sync's room check reads the same df: under a checkpoint + 1 GB, the older one goes first
echo "Drive: $free_gb GB free, as df reads $DRIVE"
stats_md5=$(md5sum < "$DRIVE/$DATA/meta/stats.json" | cut -d' ' -f1)
[ "$stats_md5" = "$STATS_MD5" ] \
    || fail "$DRIVE/$DATA/meta/stats.json md5 $stats_md5 is not the corpus's $STATS_MD5"
for src in "${MIX_SRC[@]}"; do
    stats_md5=$(md5sum < "$src/meta/stats.json" | cut -d' ' -f1)
    [ "$stats_md5" = "$STATS_MD5" ] \
        || fail "$src/meta/stats.json md5 $stats_md5 is not the corpus's $STATS_MD5"
done

uptime_s=$(cut -d' ' -f1 "${UPTIME_FILE:-/proc/uptime}")
FIT=$(awk -v up="$uptime_s" -v sps="$SEC_PER_STEP" -v max="$STEPS_MAX" 'BEGIN {
    n = int((86400 - up - 1440) / sps); if (n > max) n = max; if (n < 0) n = 0; print n }')
hours=$(awk -v up="$uptime_s" 'BEGIN { printf "%.2f", up / 3600 }')
echo "N = min($STEPS_MAX, (24 h - uptime $hours h - 0.4 h) / $SEC_PER_STEP s) = $FIT steps"
if [ "$SMOKE" -eq 1 ]; then
    STEPS=$BAND_STEP SAVE_FREQ=$BAND_STEP LOG_FREQ=10 OUT=$LOCAL/pi05_smoke
else
    STEPS=${STEPS:-$FIT} LOG_FREQ=100
fi

# the plan of this session, checked before anything is copied
RESUME=0
if [ "$SMOKE" -eq 0 ] && [ -f "$RUN_DIR/last.txt" ]; then
    RESUME=1
    recorded=$(cat "$RUN_DIR/mix.txt" 2>/dev/null || echo "|") # no file: a run without a mix
    [ "$recorded" = "$MIX|${PROMPT_CODES:-}" ] \
        || fail "$RUN_NAME trains on MIX|PROMPT_CODES '$recorded': resume it with the same two"
else
    [ "${SPAN:-0}" = 1 ] || [ "$STEPS" -le "$FIT" ] \
        || fail "STEPS=$STEPS does not fit this session: N=$FIT"
    [ "$STEPS" -ge "$BAND_STEP" ] \
        || fail "STEPS=$STEPS: a run needs step $BAND_STEP of the loss band"
    [ -n "${START:-}" ] || fail "set START=<the pretrained_model directory to start from>"
    case "$START" in "$BASE") src= ;; /*) src=$START ;; *) src=$DRIVE/$START ;; esac
    [ -z "$src" ] || [ -f "$src/model.safetensors" ] || fail "no model.safetensors in START $src"
    if [ "$SMOKE" -eq 0 ]; then
        for f in "$RUN_DIR"/[0-9]* "$RUN_DIR"/.tmp_* "$RUN_DIR"/keep; do
            if [ -e "$f" ]; then
                fail "$RUN_DIR has $(basename "$f") but no last.txt: delete it to start over"
            fi
        done
    fi
fi
STOP=
if [ "$SMOKE" -eq 0 ] && { [ "${SPAN:-0}" = 1 ] || [ -n "${KEEP_EVERY:-}" ]; }; then
    from=0 total=$STEPS
    if [ "$RESUME" -eq 1 ]; then # a resume trains toward its own steps at its own save freq
        from=$((10#$last)) total=$target
        SAVE_FREQ=$(run_cfg "$RUN_DIR/$last/pretrained_model/train_config.json" save_freq) \
            || fail "no save_freq in $RUN_DIR/$last/pretrained_model/train_config.json"
    fi
    if [ "${SPAN:-0}" = 1 ]; then
        STOP=$(((from + FIT) / SAVE_FREQ * SAVE_FREQ))
        [ $((from + FIT)) -lt "$total" ] || STOP=$total
        [ "$STOP" -gt "$from" ] || fail "SPAN=1: no checkpoint past $from fits this session: N=$FIT"
        [ "$STOP" -ge "$total" ] || echo "SPAN=1: this session stops at checkpoint $STOP of $total"
    fi
    if [ -n "${KEEP_EVERY:-}" ]; then
        [ "$KEEP_EVERY" -gt 0 ] && [ $((KEEP_EVERY % SAVE_FREQ)) -eq 0 ] \
            || fail "KEEP_EVERY=$KEEP_EVERY is not a multiple of the save freq $SAVE_FREQ"
        kept=$((total / KEEP_EVERY - from / KEEP_EVERY))
        need=$((DRIVE_MIN_GB + kept * KEEP_GB))
        [ "$DRIVE_MIN_GB" -eq 0 ] || [ "$free_gb" -ge "$need" ] \
            || fail "Drive has $free_gb GB free, a run that keeps $kept more copies needs $need"
    fi
fi

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

if [ ! -e "$LOCAL_DATA/.copied" ]; then
    rm -rf "$LOCAL_DATA"
    mkdir -p "$(dirname "$LOCAL_DATA")"
    echo "copying $DRIVE/$DATA to $LOCAL_DATA"
    cp -r "$DRIVE/$DATA" "$LOCAL_DATA"
    touch "$LOCAL_DATA/.copied"
fi
for i in "${!MIX_SRC[@]}"; do
    dst=${MIX_LOCAL[$i]}
    if [ "$dst" != "${MIX_SRC[$i]}" ] && [ ! -e "$dst/.copied" ]; then
        rm -rf "$dst"
        mkdir -p "$(dirname "$dst")"
        echo "copying ${MIX_SRC[$i]} to $dst"
        cp -r "${MIX_SRC[$i]}" "$dst"
        touch "$dst/.copied"
    fi
done
[ -z "${PROMPT_CODES:-}" ] || cp "$CODES_SRC" "$LOCAL_DATA/meta/prompt_codes.npy"
df -h "$CONTENT" | tail -1
if [ "$PREPARE" -eq 1 ]; then
    RESULT="prepared data=$LOCAL_DATA"
    exit 0
fi
# 2026-10-05 on the RTX 5090 under a 19.44 GiB cap, train_expert_only at batch 32 with gradient
# checkpointing: out of memory after step 1 with the default allocator, 18.45 GiB allocated and
# 19.41 reserved at 3.0 s a step with expandable segments
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PREFLIGHT_STATS=$DRIVE/$DATA/meta/stats.json PREFLIGHT_MAX_LENGTH=$MAX_LENGTH
export PYTHONUNBUFFERED=1
TRAIN=(python3 "$HERE/pi05_preflight.py" train)

if [ "$RESUME" -eq 1 ]; then
    last=$(cat "$RUN_DIR/last.txt")
    if [ ! -d "$OUT/checkpoints/$last" ]; then
        rm -rf "$OUT"
        mkdir -p "$OUT/checkpoints"
        cp -r "$RUN_DIR/$last" "$OUT/checkpoints/$last"
    fi
    ln -sfn "$last" "$OUT/checkpoints/last"
    config=$OUT/checkpoints/last/pretrained_model/train_config.json
    STEPS=$(run_cfg "$config" steps)
    START_STEP=$((10#$last))
    echo "resuming $RUN_NAME from checkpoint $last toward step $STEPS ($FIT steps fit)"
    # --output_dir: a resumed train_config.json names the first session's output dir;
    # --dataset.root: this session's local copy of the dataset, as a fresh run passes it
    CMD=("${TRAIN[@]}" --config_path="$config" --resume=true --output_dir="$OUT"
        --dataset.root="$LOCAL_DATA" "${MIX_ARGS[@]}")
else
    if [ "$SMOKE" -eq 0 ]; then
        mkdir -p "$RUN_DIR"
        rm -f "$RUN_DIR/mix.txt"
        [ -z "$MIX" ] || echo "$MIX|${PROMPT_CODES:-}" > "$RUN_DIR/mix.txt"
    fi
    START_COPY=$LOCAL/start_model
    # a fresh run: nothing local to keep, and a smoke checkpoint (22.8 GiB) would crowd /content
    rm -rf "$START_COPY" "$OUT" "$LOCAL/pi05_smoke"
    if [ -z "$src" ]; then
        echo "downloading $BASE at $PI05_REV to $START_COPY"
        python3 -c 'import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' \
            "$BASE" "$PI05_REV" "$START_COPY" || fail "cannot download $BASE"
        [ -f "$START_COPY/model.safetensors" ] || fail "no model.safetensors in $BASE"
    else
        echo "copying $src to $START_COPY"
        cp -r "$src" "$START_COPY"
    fi
    python3 "$HERE/pi05_preflight.py" patch "$START_COPY" "$MAX_LENGTH" \
        || fail "cannot set $MAX_LENGTH tokens in $START_COPY/policy_preprocessor.json"
    START_STEP=0
    # The pi05 preset's schedule (warmup 1000, decay 30000) scaled into STEPS as lerobot scaled
    # it for M2k (2000 steps: warmup 66); equal steps and decay steps skip that rescale.
    WARMUP=${MIX_WARMUP:-$((1000 * STEPS / 30000))}
    CMD=(
        "${TRAIN[@]}"
        --dataset.repo_id=youtalk/maker-arm-legosort
        --dataset.root="$LOCAL_DATA"
        --tolerance_s=0.01 # as train_act.sh: float32 timestamps in long v3.0 video files
        --policy.type=pi05
        --policy.pretrained_path="$START_COPY"
        --policy.dtype=bfloat16
        --policy.gradient_checkpointing=true
        --policy.train_expert_only="$EXPERT_ONLY"
        --policy.chunk_size=50
        --policy.n_action_steps=50
        --policy.tokenizer_max_length="$MAX_LENGTH"
        --policy.optimizer_lr="$PEAK_LR"
        --policy.scheduler_warmup_steps="$WARMUP"
        --policy.scheduler_decay_steps="$STEPS"
        --policy.scheduler_decay_lr=2.5e-6
        --policy.device=cuda
        --policy.push_to_hub=false
        --output_dir="$OUT"
        --job_name="pi05_${RUN_NAME:-smoke}"
        --seed="$SEED"
        --steps="$STEPS"
        --batch_size="$BATCH"
        --num_workers="$NUM_WORKERS"
        --save_checkpoint=true
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --wandb.enable=false
        "${MIX_ARGS[@]}"
    )
fi
if [ "$SMOKE" -eq 0 ]; then cp "$LOCAL_DATA/meta/action_space.json" "$DRIVE/action_space.json"; fi

BANDED=0
LOSS200=
watch_log() { # the preflight before step 1 and the loss band at BAND_STEP
    if grep -q ' step:' "$LOG" && ! grep -q 'PREFLIGHT_OK ' "$LOG"; then
        fail "the trainer logged a step with no PREFLIGHT_OK line before it, see $LOG"
    fi
    [ "$BANDED" -eq 0 ] || return 0
    LOSS200=$(awk -v s="step:$BAND_STEP" '$0 ~ " " s " " { for (i = 1; i <= NF; i++)
        if ($i ~ /^loss:/) { sub(/^loss:/, "", $i); print $i } }' "$LOG" | tail -1)
    [ -n "$LOSS200" ] || return 0
    awk -v x="$LOSS200" -v lo="$LOSS_LO" -v hi="$LOSS_HI" 'BEGIN { exit !(x >= lo && x <= hi) }' \
        || fail "loss band: step $BAND_STEP loss $LOSS200 outside [$LOSS_LO, $LOSS_HI], see $LOG"
    BANDED=1
    echo "loss band: step $BAND_STEP loss $LOSS200 in [$LOSS_LO, $LOSS_HI]"
}
session_end() { # SPAN=1: stop once this session's last checkpoint that fits is on Drive
    local synced
    synced=$(cat "$RUN_DIR/last.txt" 2>/dev/null || echo 0)
    [ -n "$STOP" ] && [ "$STOP" -lt "$STEPS" ] && [ "$((10#$synced))" -ge "$STOP" ] || return 0
    RESULT="run=$RUN_NAME steps=$STEPS last=$synced session_end=1: run the same cell in a new"
    RESULT+=" session to resume"
    exit 0
}

echo "trainer log $LOG"
"${CMD[@]}" > "$LOG" 2>&1 &
train_pid=$!
tail -n +1 -f --pid="$train_pid" "$LOG" &
tail_pid=$!
while kill -0 "$train_pid" 2>/dev/null; do
    watch_log
    if [ "$SMOKE" -eq 0 ]; then
        sync_last || echo "colab_train_pi05.sh: the Drive copy failed, next try in $POLL_S s" >&2
        session_end
    fi
    sleep "$POLL_S"
done
rc=0
wait "$train_pid" || rc=$?
train_pid=
watch_log
if [ "$rc" -ne 0 ]; then
    if [ "$SMOKE" -eq 0 ]; then sync_last || true; fi
    reason=$(grep -m1 -o 'PREFLIGHT_FAIL.*' "$LOG" || true)
    fail "${reason:-the trainer exited $rc}, see $LOG"
fi
[ "$START_STEP" -ge "$BAND_STEP" ] || [ "$BANDED" -eq 1 ] \
    || fail "no step $BAND_STEP loss line in $LOG"

if [ "$SMOKE" -eq 1 ]; then
    # the mean of updt_s + data_s over the logged steps past step 50 (warmup)
    sec=$(awk '/ step:/ {
        for (i = 1; i <= NF; i++) {
            split($i, kv, ":")
            if (kv[1] == "step") s = kv[2] + 0
            if (kv[1] == "updt_s") u = kv[2]
            if (kv[1] == "data_s") d = kv[2]
        }
        if (s > 50) { n++; t += u + d }
    } END { if (n) printf "%.3f", t / n }' "$LOG")
    [ -n "$sec" ] || fail "no step lines in $LOG"
    gib=$(du -sb "$OUT/checkpoints/last/" | awk '{printf "%.2f", $1 / 2^30}')
    RESULT="smoke sec_per_step=$sec checkpoint_gib=$gib loss$BAND_STEP=$LOSS200 batch=$BATCH"
    RESULT+=" expert_only=$EXPERT_ONLY"
    exit 0
fi

deadline=$((SECONDS + SYNC_RETRY_S))
until sync_last; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        fail "the final Drive sync failed for $SYNC_RETRY_S s; the checkpoint stays in $OUT"
    fi
    echo "colab_train_pi05.sh: the final Drive sync failed, next try in $POLL_S s" >&2
    sleep "$POLL_S"
done
last=$(cat "$RUN_DIR/last.txt")
[ "$((10#$last))" -eq "$STEPS" ] || fail "the run ended at checkpoint $last, not at step $STEPS"
rm -rf "$RUN_DIR/$last/training_state"
RESULT="run=$RUN_NAME steps=$STEPS last=$last drive=$RUN_DIR/$last (its training_state deleted)"

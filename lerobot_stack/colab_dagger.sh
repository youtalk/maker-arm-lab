#!/usr/bin/env bash
# The MS7h DAgger fine-tune (HD59) on a CLI runtime (colab new --gpu A100 --high-mem, Drive
# mounted by the owner): short `colab exec` calls drive colab_train_pi05.sh's DAgger mode. The
# long steps run under nohup into $JOBS/<step>.log and return at once; call the step again, or
# status, to poll. A step that is running or done reports so and starts nothing; a failed step
# starts again only with RETRY=1. Put this file, colab_train_pi05.sh, pi05_preflight.py and
# train_pi05.py in one folder on the runtime (/content/dag). The DAgger mode runs lerobot 0.5.1,
# so train_pi05.py is that of a53379c (git show a53379c:lerobot_stack/train_pi05.py).
#   setup                 the GPU (80 GB) and $SCRATCH (SCRATCH_MIN_GB free) checks, then
#                         colab_train_pi05.sh --prepare (lerobot, HF_TOKEN, START on Drive, the
#                         corpus copy to $SCRATCH/dataset)       DAGGER_SETUP_<STATE>
#   receive BYTES SHA256  join the dagger_vla.py tar parts in $PARTS, check the size and the
#                         sha256, unpack the root to $DAGGER, check it against the corpus, then
#                         delete the parts                      DAGGER_RECEIVE_OK | _FAIL
#   smoke                 colab_train_pi05.sh --smoke on the mix (needs setup and receive)
#                                                               DAGGER_SMOKE_<STATE>
#   start                 the run into $RUN_NAME on Drive (needs a passed smoke, or a run on
#                         Drive to resume)                      DAGGER_RUN_<STATE>
#   status                DAGGER_STATUS setup= smoke= run= (none | running | ok | fail) step=
#                         loss= lr= last_save= kept= gpu_mib=
# <STATE> is STARTED, RUNNING, OK (with the job's COLAB_TRAIN line) or FAIL.
# env: HF_TOKEN (setup, smoke, start: colab exec --env, or the file $HERE/.hf_token), START
#      (the pretrained_model under
#      $DRIVE_ROOT/$DIR: setup, smoke, start), DRIVE_ROOT (/content/drive/Shareddrives/
#      maker-arm-lab), DIR (maker-arm-ms7h-pi05), RUN_NAME (checkpoints_dagger), DAGGER_SHARE
#      (0.3), STEPS (8000), SAVE_FREQ (2000), SCRATCH (/mnt/local-scratch), PARTS
#      (/content/dag/parts), SCRATCH_MIN_GB (160: corpus 66 + two checkpoints 46 + the start
#      copy 10 + the DAgger tar and root), RETRY.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CMD=${1:-}
export DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/Shareddrives/maker-arm-lab}
export DIR=${DIR:-maker-arm-ms7h-pi05} RUN_NAME=${RUN_NAME:-checkpoints_dagger}
export DAGGER_SHARE=${DAGGER_SHARE:-0.3} STEPS=${STEPS:-8000} SAVE_FREQ=${SAVE_FREQ:-2000}
SCRATCH=${SCRATCH:-/mnt/local-scratch}
PARTS=${PARTS:-/content/dag/parts}
JOBS=$SCRATCH/dag_jobs
DAGGER=$SCRATCH/dagger_dataset
CORPUS=$DRIVE_ROOT/$DIR/dataset
# the run's layout: checkpoints and the start copy on the scratch disk, Drive's room unchecked
export OUT_ROOT=$SCRATCH LOCAL_DATA=$SCRATCH/dataset DRIVE_MIN_GB=0
TOK=DAGGER_$(echo "${CMD:-USAGE}" | tr 'a-z' 'A-Z')
[ "$CMD" != start ] || TOK=DAGGER_RUN
fail() {
    echo "${TOK}_FAIL $*"
    exit 1
}

state() { # none | running | ok <token> | fail <token or reason>
    local log=$JOBS/$1.log token
    [ -f "$log" ] || { echo none; return; }
    token=$(grep -aE '^COLAB_TRAIN_(OK|DONE|FAIL)' "$log" | tail -1)
    case $token in
        COLAB_TRAIN_FAIL*) echo "fail $token" ;;
        COLAB_TRAIN_*) echo "ok $token" ;;
        *) if kill -0 "$(cat "$JOBS/$1.pid" 2>/dev/null)" 2>/dev/null; then echo running; else
            echo "fail no token and no process: see $log"; fi ;;
    esac
}
last_step() { # the trainer's last "step:" line of a job log, as step= loss= lr=
    grep -a ' step:' "$JOBS/$1.log" 2>/dev/null | tail -1 | awk '{ for (i = 1; i <= NF; i++)
        if ($i ~ /^(step|loss|lr):/) { sub(":", "=", $i); printf "%s ", $i } }'
}
job() { # job <name> <colab_train_pi05.sh args...>: one token line; starts the job if it is new
    local name=$1 s
    shift
    s=$(state "$name")
    case $s in
        running) echo "${TOK}_RUNNING $(last_step "$name")log=$JOBS/$name.log" ;;
        ok*) echo "${TOK}_OK ${s#ok }" ;;
        fail*) if [ "${RETRY:-0}" != 1 ]; then fail "${s#fail } (RETRY=1 starts it again)"; fi ;;&
        *)
            mkdir -p "$JOBS"
            [ ! -f "$JOBS/$name.log" ] || mv "$JOBS/$name.log" "$JOBS/$name.$(date +%s).log"
            # its own session: a kernel interrupt or restart signals the kernel's process group
            nohup setsid bash "$HERE/colab_train_pi05.sh" "$@" > "$JOBS/$name.log" 2>&1 < /dev/null &
            echo $! > "$JOBS/$name.pid"
            echo "${TOK}_STARTED pid=$! log=$JOBS/$name.log"
            ;;
    esac
}
need_token() {
    # The owner can leave the token in $HERE/.hf_token from a terminal, so it never passes
    # through the controller (read -s T; echo "open('/content/dag/.hf_token','w').write('$T')"
    # | colab exec -s ms7h-dag; unset T).
    if [ -z "${HF_TOKEN:-}" ] && [ -s "$HERE/.hf_token" ]; then
        HF_TOKEN=$(< "$HERE/.hf_token")
        export HF_TOKEN
    fi
    [ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not set: colab exec --env, or $HERE/.hf_token"
    [ -n "${START:-}" ] || fail "set START=<pretrained_model under $DRIVE_ROOT/$DIR>"
}
free_gb() { df -Pk "$1" | awk 'NR == 2 { printf "%d", $4 / 1000000 }'; }

case $CMD in
    setup)
        need_token
        [ -f "$CORPUS/meta/info.json" ] || fail "no $CORPUS/meta/info.json (Drive mounted?)"
        mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        [ "${mib:-0}" -ge 70000 ] || fail "the GPU has ${mib:-no} MiB: the full fine-tune needs 80 GB"
        mkdir -p "$SCRATCH" "$PARTS" || fail "cannot make $SCRATCH and $PARTS"
        if [ "$(state setup)" = none ] && [ "$(free_gb "$SCRATCH")" -lt "${SCRATCH_MIN_GB:-160}" ]
        then
            fail "$SCRATCH has $(free_gb "$SCRATCH") GB free, the run needs ${SCRATCH_MIN_GB:-160}"
        fi
        DAGGER= job setup --prepare
        ;;
    receive)
        bytes=${2:-} sha=${3:-}
        [[ "$bytes" =~ ^[0-9]+$ && "$sha" =~ ^[0-9a-f]{64}$ ]] || fail "usage: receive BYTES SHA256"
        parts=("$PARTS"/*.tar.part[0-9]*)
        [ -f "${parts[0]}" ] || fail "no *.tar.part* files in $PARTS"
        [ "$(free_gb "$SCRATCH")" -ge $((2 * bytes / 1000000000 + 1)) ] \
            || fail "$SCRATCH has $(free_gb "$SCRATCH") GB free, the tar needs twice its size"
        tar=$SCRATCH/dagger.tar
        cat "${parts[@]}" > "$tar" || fail "cannot join the parts into $tar"
        got=$(stat -c %s "$tar")
        [ "$got" = "$bytes" ] || { rm -f "$tar"; fail "the tar has $got bytes, want $bytes"; }
        got=$(sha256sum "$tar" | cut -d' ' -f1)
        [ "$got" = "$sha" ] || { rm -f "$tar"; fail "the tar's sha256 is $got, want $sha"; }
        rm -rf "$SCRATCH/dagger_unpack" "$DAGGER"
        mkdir -p "$SCRATCH/dagger_unpack" && tar -xf "$tar" -C "$SCRATCH/dagger_unpack" \
            || fail "cannot unpack $tar"
        roots=("$SCRATCH"/dagger_unpack/*/meta/info.json)
        [ "${#roots[@]}" -eq 1 ] && [ -f "${roots[0]}" ] || fail "the tar holds no single root"
        mv "$(dirname "$(dirname "${roots[0]}")")" "$DAGGER" && rm -rf "$SCRATCH/dagger_unpack"
        cmp -s "$DAGGER/meta/stats.json" "$CORPUS/meta/stats.json" \
            || fail "$DAGGER/meta/stats.json is not the corpus's"
        summary=$(python3 - "$DAGGER" "$CORPUS" <<'PY'
import json, sys
dagger, corpus = (json.load(open(f"{r}/meta/info.json")) for r in sys.argv[1:])
bad = [k for k in ("features", "fps") if dagger.get(k) != corpus.get(k)]
n = (dagger.get("total_episodes"), dagger.get("total_frames"))
print(f"BAD {bad}" if bad else f"episodes={n[0]} frames={n[1]}")
PY
        )
        case $summary in BAD*) fail "$DAGGER differs from the corpus: ${summary#BAD }" ;; esac
        rm -f "$tar" "${parts[@]}"
        echo "${TOK}_OK root=$DAGGER $summary bytes=$bytes sha256=$sha"
        ;;
    smoke | start)
        need_token
        case $(state setup) in ok*) ;; *) fail "setup has not passed: $(state setup)" ;; esac
        [ -f "$DAGGER/meta/info.json" ] || fail "no $DAGGER: receive the DAgger tar first"
        export DAGGER
        if [ "$CMD" = smoke ]; then
            job smoke --smoke
        else
            case $(state smoke) in
                ok*) ;;
                *) [ -f "$DRIVE_ROOT/$DIR/$RUN_NAME/last.txt" ] \
                    || fail "the smoke has not passed: $(state smoke)" ;;
            esac
            job run
        fi
        ;;
    status)
        run_dir=$DRIVE_ROOT/$DIR/$RUN_NAME
        kept=$(ls "$run_dir/keep" 2>/dev/null | tr '\n' ',')
        line="setup=$(state setup | cut -d' ' -f1) smoke=$(state smoke | cut -d' ' -f1)"
        line+=" run=$(state run | cut -d' ' -f1) $(last_step run)"
        line+="last_save=$(cat "$run_dir/last.txt" 2>/dev/null || echo none) kept=${kept%,}"
        line+=" gpu_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
        echo "DAGGER_STATUS $line"
        ;;
    *) fail "usage: colab_dagger.sh setup | receive BYTES SHA256 | smoke | start | status" ;;
esac

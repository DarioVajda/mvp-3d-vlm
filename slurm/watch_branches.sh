#!/usr/bin/env bash
# Watch a wsd trunk and submit one decay branch (slurm/decay_eval.sh, 1 GPU) per branch state as soon as the trunk
# has written it; once the trunk has left the queue and every state is handled, submit mvp.train.wsd_collect to run
# after all branches. Runs as a CPU-only Slurm job (slurm/wsd.sh starts it), outside the container:
#   bash slurm/watch_branches.sh results/train/<trunk> <trunk job id> <step> [<step> ...]
# Without steps (a --fit-epoch trunk, whose branch steps depend on the epoch length), the steps are read from the
# trunk's args.json (branch_steps) once the trunk has written it.
# Idempotent and restartable: a branch counts as submitted once <trunk>/branches/step<N>/submitted holds its job id.
# BRANCH_SBATCH adds sbatch options to the branch jobs (default 1 B200/B300/H100, 8 CPUs, 4 h); DECAY_STEPS and EVAL_ARGS
# are passed on to slurm/decay_eval.sh.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
trunk=$1; job=$2; shift 2; steps=("$@")
name=$(basename "$trunk")
POLL=${POLL:-300}
# a clean environment for sbatch: this job's own SLURM_* variables must not leak into the jobs it submits
submit() { env -i HOME="$HOME" USER="$USER" PATH="$PATH" LOGDIR=train/branches DECAY_STEPS="${DECAY_STEPS:-750}" \
    EVAL_ARGS="${EVAL_ARGS:-}" bash slurm/submit.sh "$@"; }
log() { echo "[$(date '+%F %T')] $*"; }

submit_ready() {
    for n in "${steps[@]}"; do
        d=$trunk/branches/step$n
        [[ -f $trunk/state_step$n.pt && ! -f $d/submitted ]] || continue
        mkdir -p "$d"
        id=$(submit -J "decay_${name}_s$n" -t 04:00:00 -c 8 ${BRANCH_SBATCH:-} -- bash slurm/decay_eval.sh "$trunk" "$n")
        if [[ $id =~ ^[0-9]+$ ]]; then echo "$id" > "$d/submitted"; log "step $n: branch job $id"
        else log "step $n: submission failed ($id), retrying next poll"; fi
    done
}

if (( ! ${#steps[@]} )); then
    log "waiting for the branch steps in $trunk/args.json"
    until [[ -f $trunk/log.jsonl ]] && grep -q '"fit_epoch": true' "$trunk/args.json" 2>/dev/null; do
        state=$(sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
        [[ $state =~ ^(PENDING|RUNNING|REQUEUED|)$ ]] || { log "trunk job $job ended ($state) before its first step"; exit 1; }
        sleep "$POLL"
    done
    read -ra steps <<< "$(python3 -c "import json, sys; print(*json.load(open(sys.argv[1]))['branch_steps'])" "$trunk/args.json")"
fi
log "watching $trunk (job $job), branch steps ${steps[*]}"
while true; do
    submit_ready
    state=$(sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
    case $state in
        COMPLETED|FAILED*|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
            submit_ready  # states written just before the trunk ended
            log "trunk job $job ended: $state"; break ;;
    esac
    sleep "$POLL"
done

ids=()
for n in "${steps[@]}"; do
    f=$trunk/branches/step$n/submitted
    if [[ -f $f ]]; then ids+=("$(cat "$f")"); else log "step $n: no branch state, no branch"; fi
done
if (( ${#ids[@]} )); then
    dep=$(IFS=:; echo "${ids[*]}")
    id=$(env -i HOME="$HOME" USER="$USER" PATH="$PATH" LOGDIR=train/branches bash slurm/submit.sh -J "collect_$name" \
        -t 00:30:00 -c 2 --mem 16G --gres=none --dependency="afterany:$dep" \
        --constraint 'GPU_BRD:L4|GPU_BRD:A100|GPU_BRD:H100|GPU_BRD:B200|GPU_BRD:B300' -- python -m mvp.train.wsd_collect "$trunk")
    log "collector job $id after branches $dep"
fi
log "done"

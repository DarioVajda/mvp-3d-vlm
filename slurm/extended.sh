#!/usr/bin/env bash
# Extended training (2026-11-20): Qwen3.5-2B on the 450K mix (data/mix/ext_450k.jsonl, mvp.train.mix ext), one pass,
# cross-attention (the ablation's base, v10) and LoRA only, seed 0, otherwise the ablation recipe (lr, frame mix 8/16/32
# = 60/30/10, LoRA r64). Questions about different media are packed into one micro-batch, grouped by length
# (--pack-tokens, data.pack_plan). wsd: 750 warmup steps, stable to one epoch minus the 750 decay steps; a single decay
# branch at the end, then the full VSI-Bench eval at 32 frames. Training states at a quarter, half and three quarters of
# the stable phase are kept for a restart (--continue-from) only; they are not decayed.
#
#   slurm/extended.sh run <run>     <run>: 2B_ext_xa3_nu_s0 or 2B_ext_lora_s0, on the 4 visible GPUs
#   slurm/extended.sh smoke <run>   20 steps, decay 10, 5 VSI questions per task; results/train/smoke_<run> (not tracked)
#
#   slurm/submit.sh -J 2B_ext_xa3_nu_s0 -A lkm --constraint GPU_BRD:B300 --gres gpu:4 -c 64 --mem 512G -t 3-00:00:00 \
#       -- bash slurm/extended.sh run 2B_ext_xa3_nu_s0
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

MIX=data/mix/ext_450k.jsonl
NP=4
common=(--mix "$MIX" --pack-tokens 80000 --appearance --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}' --workers 12)
config() {
    case $1 in
        2B_ext_xa3_nu)  echo xattn_aligned --pe 3d --rope3 --local-bias --pe-dmax 10 --bias-nu 0.1 6 ;;
        2B_ext_lora)    echo lora ;;
        *) echo "unknown row $1" >&2; return 1 ;;
    esac
}
job=${SLURM_JOB_ID:-local}
LOGS=logs/train/extended
log() { echo "[$(date '+%F %T')] $*"; }

lane() {  # lane <run> [extra train options]
    local run=$1; shift
    local row=${run%_s*} seed=${run##*_s} trunk=results/train/$run
    row=${row#smoke_}
    local cfg; read -ra cfg <<< "$(config "$row")" || return 1
    local gpus; IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?no GPU}"
    (( ${#gpus[@]} == NP )) || { log "$run: needs $NP GPUs, ${#gpus[@]} visible"; return 1; }
    mkdir -p "$trunk" "$LOGS"
    echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$NP job $job" > "$trunk/gpu.txt"
    log "$run: trunk on GPUs $CUDA_VISIBLE_DEVICES ($(cat "$trunk/gpu.txt"))"
    torchrun --standalone --nproc_per_node "$NP" -m mvp.train.run --model Qwen/Qwen3.5-2B --arm "${cfg[0]}" --out "$trunk" \
        --schedule wsd --warmup-steps 750 --decay-steps "${DECAY_STEPS:-750}" "${common[@]}" "${cfg[@]:1}" \
        --init-seed "$seed" "$@" > "$LOGS/train_${run}_$job.out" 2>&1 || { log "$run: trunk FAILED (exit $?)"; return 1; }
    local step; step=$(python -c "import json; print(json.load(open('$trunk/args.json'))['branch_steps'][-1])")
    log "$run: decay branch from step $step"
    NPROC=$NP DECAY_STEPS=${DECAY_STEPS:-750} EVAL_ARGS=${EVAL_ARGS:-} bash slurm/decay_eval.sh "$trunk" "$step" \
        > "$LOGS/decay_${run}_$job.out" 2>&1 || { log "$run: branch FAILED (exit $?)"; return 1; }
    if [[ " $* " != *" --no-wandb "* ]]; then
        python -m mvp.train.wsd_collect "$trunk" > "$LOGS/collect_${run}_$job.out" 2>&1 || log "$run: collector FAILED"
    fi
    log "$run: done"
}

case ${1:-} in
    run)
        [[ ${2:-} =~ ^2B_ext_(xa3_nu|lora)_s[0-9]$ ]] || { echo "unknown run '${2:-}'" >&2; exit 1; }
        lane "$2" --fit-epoch --n-branches 4 ;;
    smoke)
        DECAY_STEPS=10 EVAL_ARGS="--per-type 5" lane "smoke_$2" --no-wandb --max-steps 20 --stable-steps 10 \
            --warmup-steps 10 --branch-steps 20 --eval-every 10 --dev-every 10 ;;
    *) sed -n '2,15p' "$0"; exit 1 ;;
esac

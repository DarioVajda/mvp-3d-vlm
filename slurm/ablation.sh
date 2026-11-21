#!/usr/bin/env bash
# Final ablation set (2026-11-14): every reported setting on one recipe, three seeds (--init-seed 0, 1, 2; data seed 0).
# Recipe: Qwen3.5-2B, VSI-590K, frame mix 8/16/32 = 60/30/10, lr 5e-5 / fusion lr 5e-6, wsd with 750 warmup + 5,000
# stable steps, one decay branch from step 5,750 (750 steps), then the full VSI-Bench eval at 32 frames. One GPU per
# run, B200 or B300. Seeds 0 and 1 of the first five rows exist already (v9, v10, v11); those rows get seed 2 only.
#
#   slurm/ablation.sh list            the 26 runs, <row>_s<seed>, in the order they were queued
#   slurm/ablation.sh run <run>       one run on the (first) visible GPU: trunk (mvp.train.run), decay branch + VSI
#                                     eval (slurm/decay_eval.sh), collector (mvp.train.wsd_collect)
#   slurm/ablation.sh smoke <row>     short smoke test of <row>: 20 steps, decay 10 steps, 5 VSI questions per task,
#                                     frames 8/16/32 in equal shares; writes results/train/smoke_<row>_s1 (not tracked)
# Runs inside the container from the project root, e.g.
#   slurm/submit.sh -J 2B_abl_grid_s1 -A lkm --constraint 'GPU_BRD:B300|GPU_BRD:B200' -c 12 --mem 128G -t 1-00:00:00 \
#       -- bash slurm/ablation.sh run 2B_abl_grid_s1
# Logs go to logs/train/ablation/; the node and GPU type of each run to results/train/<run>/gpu.txt.
# Scale check (2026-11-20): the same recipe on Qwen3.5-4B, cross-attention and LoRA only, seed 0 (rows 4B_*, options
# as their 2B rows). One B300 node with 2 GPUs per run, NPROC=2 (torchrun; the same steps and batch as on one GPU):
#   NPROC=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/submit.sh -J 4B_v10_xa3_nu_s0 -A lkm \
#       --constraint GPU_BRD:B300 --gres gpu:2 -c 24 --mem 320G -t 2-00:00:00 -- bash slurm/ablation.sh run 4B_v10_xa3_nu_s0
# B300 only: the 4B cross-attention arm ran out of memory on B200 (job 217719).
# The same for Qwen3.5-9B (rows 9B_*; 32 decoder layers like the 4B) on 4 GPUs: NPROC=4 --gres gpu:4 -c 48 --mem 640G.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

common=(--appearance --vsi590k --cap 50000 --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}')
xa3=(--pe 3d --rope3 --local-bias --pe-dmax 10 --bias-nu 0.1 6)  # the base: v10 aligned cross-attention
# config <row>: the arm and its options
config() {
    case $1 in
        2B_v9_fmix_lora)       echo lora ;;
        2B_v9_fmix_additive)   echo additive ;;
        2B_v10_xa3_nu)         echo xattn_aligned "${xa3[@]}" ;;
        2B_v9_fmix_xa3_both)   echo xattn_aligned --pe 3d --rope3 --local-bias --pe-dmax 10 --bias-nu 1.5 ;;
        2B_v11_xa3_nu_text)    echo xattn_aligned "${xa3[@]}" --text-queries ;;
        2B_abl_norope)         echo xattn_aligned "${xa3[@]}" --no-rope3 ;;
        2B_abl_nobias)         echo xattn_aligned "${xa3[@]}" --no-local-bias ;;
        2B_abl_nopos)          echo xattn_aligned "${xa3[@]}" --no-rope3 --no-local-bias ;;
        2B_abl_grid)           echo xattn_aligned --pe grid --rope3 --local-bias --bias-lam 0.1 6 --bias-mu 3.0 ;;
        2B_abl_late)           echo xattn_aligned "${xa3[@]}" --fusion-after 6 10 14 ;;
        2B_abl_earlylate)      echo xattn_aligned "${xa3[@]}" --vggt-layers 11 17 23 11 17 23 --fusion-after 0 1 2 6 10 14 ;;
        2B_abl_flamingo)       echo xattn --resampler-layers 8 --gate none ;;
        *) echo "unknown row $1" >&2; return 1 ;;
    esac
}
# the runs: third seeds of the main comparison, then the new rows by seed; longest runs first within each part
MAIN=(2B_v10_xa3_nu 2B_v9_fmix_xa3_both 2B_v11_xa3_nu_text 2B_v9_fmix_additive 2B_v9_fmix_lora)
NEW=(2B_abl_earlylate 2B_abl_norope 2B_abl_nobias 2B_abl_nopos 2B_abl_grid 2B_abl_late 2B_abl_flamingo)
SCALE=(4B_v10_xa3_nu 4B_v9_fmix_lora 9B_v10_xa3_nu 9B_v9_fmix_lora)  # scale check: one seed each
runs() {
    for r in "${MAIN[@]}"; do echo "${r}_s2"; done
    for s in 0 1 2; do for r in "${NEW[@]}"; do echo "${r}_s$s"; done; done
    for r in "${SCALE[@]}"; do echo "${r}_s0"; done
}

job=${SLURM_JOB_ID:-local}
LOGS=logs/train/ablation
log() { echo "[$(date '+%F %T')] $*"; }

# lane <run> [extra train options]: trunk, decay branch + VSI eval, collector (skipped with --no-wandb). WARMUP, STABLE,
# DECAY_STEPS and EVAL_ARGS override the schedule and the eval (smoke tests).
lane() {
    local run=$1; shift
    local row=${run%_s*} seed=${run##*_s} trunk=results/train/$run rc=0
    row=${row#smoke_}  # smoke tests: smoke_<row>_s<seed>
    local size=${row%%_*}  # 2B, 4B or 9B; a 4B or 9B row has its 2B row's options
    local warm=${WARMUP:-750} stable=${STABLE:-5000} np=${NPROC:-1}
    local step=$((warm + stable))
    local cfg; read -ra cfg <<< "$(config "2B_${row#*_}")" || return 1
    local gpus; IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?no GPU}"
    (( ${#gpus[@]} >= np )) || { log "$run: NPROC=$np but only ${#gpus[@]} GPUs visible"; return 1; }
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpus[*]:0:np}"); export CUDA_VISIBLE_DEVICES  # the first NPROC GPUs
    local py=(python); (( np > 1 )) && py=(torchrun --standalone --nproc_per_node "$np")
    mkdir -p "$trunk" "$LOGS"
    echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$np job $job" > "$trunk/gpu.txt"
    log "$run: trunk on GPU $CUDA_VISIBLE_DEVICES ($(cat "$trunk/gpu.txt"))"
    "${py[@]}" -m mvp.train.run --model "Qwen/Qwen3.5-$size" --arm "${cfg[0]}" --out "$trunk" --schedule wsd \
        --warmup-steps "$warm" --stable-steps "$stable" --branch-steps "$step" "${common[@]}" "${cfg[@]:1}" \
        --init-seed "$seed" "$@" > "$LOGS/train_${run}_$job.out" 2>&1 \
        || { log "$run: trunk FAILED (exit $?)"; return 1; }
    log "$run: branch $step"
    NPROC=$np DECAY_STEPS=${DECAY_STEPS:-750} EVAL_ARGS=${EVAL_ARGS:-} bash slurm/decay_eval.sh "$trunk" "$step" \
        > "$LOGS/decay_${run}_$job.out" 2>&1 || { log "$run: branch FAILED (exit $?)"; return 1; }
    if [[ " $* " != *" --no-wandb "* ]]; then
        python -m mvp.train.wsd_collect "$trunk" > "$LOGS/collect_${run}_$job.out" 2>&1 \
            || { log "$run: collector FAILED (exit $?)"; rc=1; }
    fi
    log "$run: done (rc $rc)"
    return $rc
}

case ${1:-} in
    list) runs ;;
    run)
        runs | grep -qx "${2:-}" || { echo "unknown run '${2:-}' (see: $0 list)" >&2; exit 1; }
        lane "$2" ;;
    smoke)
        config "2B_${2#*_}" > /dev/null || exit 1
        WARMUP=5 STABLE=15 DECAY_STEPS=10 EVAL_ARGS="--per-type 5" lane "smoke_$2_s1" \
            --no-wandb --frame-mix '{"8":34,"16":33,"32":33}' ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
esac

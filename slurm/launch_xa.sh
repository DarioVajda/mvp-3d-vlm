#!/usr/bin/env bash
# Aligned cross-attention arm (2026-10-28): the 2x2 of RoPE3 x locality bias, on the v4 recipe (VLM-3R ScanNet cap 6k
# + all VSI-590K appearance order, 5 epochs, dev + checkpoint every half epoch, LoRA r64 at 1e-4, new modules at 1e-5,
# weight decay 0.01), bias init lam 1.5, mu 3. Best-dev checkpoint on full VSI-Bench at 32 frames, chained.
#   slurm/launch_xa.sh <hw for none> <hw for rope> <hw for bias> <hw for both>    hw: B (1x B300/B200) or H (2x H100)
set -euo pipefail
common=(--epochs 5 --cap 6000 --appearance --evals-per-epoch 2 --lr 1e-4 --fusion-lr 1e-5 --weight-decay 0.01
        --bias-lam 1.5 --bias-mu 3)
export EVAL_SBATCH="--constraint GPU_BRD:B300|GPU_BRD:B200|GPU_BRD:H100"
jobs=results/train/jobs_2B_xa.txt
: > "$jobs"
hw=("$@")
i=0
for cfg in "none --no-rope3 --no-local-bias" "rope --rope3 --no-local-bias" "bias --no-rope3 --local-bias" "both --rope3 --local-bias"; do
    set -- $cfg
    name=2B_xa_$1; shift
    if [[ ${hw[$i]} == H ]]; then
        TRAIN_SBATCH="-t 20:00:00 --gres gpu:2 -c 32 --mem 256G --constraint GPU_BRD:H100" \
        TRAIN_LAUNCH="torchrun --standalone --nproc_per_node 2" \
            slurm/train_eval.sh xattn_aligned "$name" "${common[@]}" "$@" | tee -a "$jobs"
    else
        TRAIN_SBATCH="-t 16:00:00 --constraint GPU_BRD:B300|GPU_BRD:B200" \
            slurm/train_eval.sh xattn_aligned "$name" "${common[@]}" "$@" | tee -a "$jobs"
    fi
    i=$((i + 1))
done

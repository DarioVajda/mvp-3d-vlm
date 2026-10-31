#!/usr/bin/env bash
# Round v6 (2026-10-30, aligned cross-attention spec draft 5 §9.6), all on the v4 recipe of slurm/launch_xa.sh:
#   2B_xa3_{bias,rope,both}  the v5 2x2 of rotary x bias with both signals made 3D from VGGT's point head (--pe 3d: RoPE
#                            on (t~, x, y, z), bias -nu * min(3D distance, 10 cells), nu init 1.5); "none" is v5's 2B_xa_none
#   2B_xa_both_s1, 2B_v4_additive_nm1e-5_s1  second seeds of the two anchors: --init-seed 1 changes init, dropout and
#                            the training order; the data, dev split and selection set stay those of --seed 0
# Best-dev checkpoint of each on full VSI-Bench at 32 frames, chained.
#   slurm/launch_xa3.sh <hw> x 5 (in the order above: bias rope both xa_both_s1 additive_s1)    hw: B (1x B300/B200) or H (2x H100)
set -euo pipefail
common=(--epochs 5 --cap 6000 --appearance --evals-per-epoch 2 --lr 1e-4 --fusion-lr 1e-5 --weight-decay 0.01)
pe3d=(--pe 3d --bias-nu 1.5 --pe-dmax 10)
export EVAL_SBATCH="--constraint GPU_BRD:B300|GPU_BRD:B200|GPU_BRD:H100"
jobs=results/train/jobs_2B_v6.txt
: > "$jobs"
hw=("$@")
[[ ${#hw[@]} == 5 ]] || { echo "need 5 hw letters (B or H)" >&2; exit 1; }
runs=(
    "2B_xa3_bias xattn_aligned ${pe3d[*]} --no-rope3 --local-bias"
    "2B_xa3_rope xattn_aligned ${pe3d[*]} --rope3 --no-local-bias"
    "2B_xa3_both xattn_aligned ${pe3d[*]} --rope3 --local-bias"
    "2B_xa_both_s1 xattn_aligned --rope3 --local-bias --bias-lam 1.5 --bias-mu 3 --init-seed 1"
    "2B_v4_additive_nm1e-5_s1 additive --init-seed 1"
)
for i in "${!runs[@]}"; do
    set -- ${runs[$i]}
    name=$1 arm=$2; shift 2
    if [[ ${hw[$i]} == H ]]; then
        TRAIN_SBATCH="-t 20:00:00 --gres gpu:2 -c 32 --mem 256G --constraint GPU_BRD:H100" \
        TRAIN_LAUNCH="torchrun --standalone --nproc_per_node 2" \
            slurm/train_eval.sh "$arm" "$name" "${common[@]}" "$@" | tee -a "$jobs"
    else
        TRAIN_SBATCH="-t 16:00:00 --constraint GPU_BRD:B300|GPU_BRD:B200" \
            slurm/train_eval.sh "$arm" "$name" "${common[@]}" "$@" | tee -a "$jobs"
    fi
done

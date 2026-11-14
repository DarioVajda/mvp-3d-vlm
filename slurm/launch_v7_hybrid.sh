#!/usr/bin/env bash
# Round v7, hybrid arm (2026-11-02): additive + aligned cross-attention on one merger per injection point (shared
# trunk, a zero-initialised additive output layer and a key / value output layer; mvp.fusion.model, "hybrid"). Tests
# whether giving each visual token its own cell unaveraged recovers the abs-distance gap of xattn_aligned to additive.
# Exactly the v7 recipe of slurm/launch_v7.sh (data, cap, schedule, lrs, branches, 3D signals of xa3_both), two seeds.
# Trunks on 2 H100s (DDP, the same steps as on one GPU: each step's micro-batches are split over the ranks); decay
# branches on one H100 / B200 / B300 (slurm/submit.sh's default), 6 h as H100s are slower than the B200 of v7.
#   slurm/launch_v7_hybrid.sh        -> results/train/jobs_2B_v7_hybrid.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200)
xa3=(--pe 3d --rope3 --local-bias --bias-nu 1.5 --pe-dmax 10)
export TRAIN_SBATCH="--gres gpu:2 -c 16 --mem 256G --constraint GPU_BRD:H100"
export TRAIN_LAUNCH="torchrun --standalone --nproc_per_node 2"
export BRANCH_SBATCH="-t 06:00:00"
jobs=results/train/jobs_2B_v7_hybrid.txt
: > "$jobs"
for seed in 0 1; do
    slurm/wsd.sh hybrid "2B_v7_hybrid_s$seed" "${common[@]}" "${xa3[@]}" --init-seed "$seed" | tee -a "$jobs"
done

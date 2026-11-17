#!/usr/bin/env bash
# Round v7 (2026-11-01): training length under a warmup-stable-decay schedule. Arms: additive, aligned cross-attention
# with the v6 3D signals (3D distance bias + (t~, x, y, z) RoPE), and the LoRA-only control (no geometry path; same
# data, schedule and LoRA lr); two seeds each. Per trunk (slurm/wsd.sh): 750 warmup steps, 10,000 at the peak lr,
# decay branches of 750 steps to lr 0 from steps 5,750 / 6,750 / ... / 10,750 (5k-10k stable steps), each followed by
# dev accuracy and full VSI-Bench at 32 frames, collected into the trunk's W&B run.
# Data: VLM-3R ScanNet + VSI-590K's six VSI-Bench tasks on all video sources (--vsi590k) + all appearance order,
# capped at CAP questions per task; same dev split as v4-v6. lr 0.5x of v6 (LoRA 5e-5, new modules 5e-6), slopes 1e-3.
#   slurm/launch_v7.sh        -> results/train/jobs_2B_v7.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200)
xa3=(--pe 3d --rope3 --local-bias --bias-nu 1.5 --pe-dmax 10)
jobs=results/train/jobs_2B_v7.txt
: > "$jobs"
for seed in 0 1; do
    slurm/wsd.sh lora "2B_v7_lora_s$seed" "${common[@]}" --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh additive "2B_v7_additive_s$seed" "${common[@]}" --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh xattn_aligned "2B_v7_xa3_both_s$seed" "${common[@]}" "${xa3[@]}" --init-seed "$seed" | tee -a "$jobs"
done

#!/usr/bin/env bash
# Round v9 (2026-11-03): more frames in training. The v7 arms (LoRA-only, additive, aligned cross-attention xa3_both)
# and recipe of slurm/launch_v7.sh with v8's schedule (750 warmup + 5,000 stable steps, one decay branch from step
# 5,750, compared with v7's step-5750 branches), except that each step decodes 8, 16 or 32 frames: 60 / 30 / 10 % of
# the steps (--frame-mix), one count for all micro-batches of a step, about 1.6x v7's frames per step. Dev stays at 8
# frames; the full VSI eval after the branch is at 32 frames as before. Two seeds each.
# Trunks and branches on one B300 / B200 only, under the lkm account (much higher fairshare than povejmo on 2026-11-03);
# one slurm/watch_many.sh job watches all six trunks (the per-user job limit).
#   slurm/launch_v9_frames.sh        -> results/train/jobs_2B_v9_frames.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}')
xa3=(--pe 3d --rope3 --local-bias --bias-nu 1.5 --pe-dmax 10)
blackwell="-A lkm --constraint GPU_BRD:B300|GPU_BRD:B200"
export STABLE=5000 BRANCH_STEPS=5750 NO_WATCH=1
export TRAIN_SBATCH="$blackwell" BRANCH_SBATCH="-t 06:00:00 $blackwell"
jobs=results/train/jobs_2B_v9_frames.txt
: > "$jobs"
for seed in 0 1; do
    slurm/wsd.sh lora "2B_v9_fmix_lora_s$seed" "${common[@]}" --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh additive "2B_v9_fmix_additive_s$seed" "${common[@]}" --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh xattn_aligned "2B_v9_fmix_xa3_both_s$seed" "${common[@]}" "${xa3[@]}" --init-seed "$seed" | tee -a "$jobs"
done
pairs=$(awk '{printf "results/train/%s %s ", $1, $3}' "$jobs")
mkdir -p logs/watch
watch=$(sbatch --parsable -p frida -A lkm -c 1 --mem 2G -t 4-00:00:00 -J watch_2B_v9_frames -o "logs/watch/%x_%j.out" \
    --export=ALL,POLL=${POLL:-300} --wrap "bash slurm/watch_many.sh '$BRANCH_STEPS' $pairs" | grep -oE '^[0-9]+' | tail -1)
echo "watch $watch" | tee -a "$jobs"

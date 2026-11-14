#!/usr/bin/env bash
# Longer training for the best XA arm and its control: continue v10 xa3_qknu and v9 additive (both seeds) from their
# step-5750 branch states at the stable lr to step 10,750 (750 warmup + 10,000 stable), in place (run.py
# --continue-from), with decay branches (750 steps, then full VSI-Bench at 32 frames) at 8,250 and 10,750; the existing
# step-5750 branch is the first point of each curve. Blackwell GPUs on the lkm account; seed 1 queued behind seed 0
# (--nice). One watch_many.sh job for the branches and collectors.
#   slurm/launch_v10_continue.sh            -> results/train/jobs_2B_v10_continue.txt
set -euo pipefail
cd "$(dirname "$0")/.."
runs=(2B_v10_xa3_qknu_s0 2B_v9_fmix_additive_s0 2B_v10_xa3_qknu_s1 2B_v9_fmix_additive_s1)
steps="8250 10750"
blackwell="-A lkm --constraint GPU_BRD:B300|GPU_BRD:B200"
jobs=results/train/jobs_2B_v10_continue.txt
pairs=""
for run in "${runs[@]}"; do
    nice=$([[ $run == *_s1 ]] && echo 1000 || echo 0)
    id=$(LOGDIR=train slurm/submit.sh -J "cont_$run" -t 20:00:00 -c 8 $blackwell --nice=$nice -- python -m mvp.train.run \
        --continue-from "results/train/$run" --continue-step 5750 --stable-steps 10000 --branch-steps $steps)
    echo "$run continue $id" | tee -a "$jobs"
    pairs+=" results/train/$run $id"
done
id=$(BRANCH_SBATCH="-t 06:00:00 $blackwell" sbatch --parsable -p frida -A lkm -c 1 --mem 2G -t 4-00:00:00 -J watch_2B_v10_continue \
    -o logs/watch/%x_%j.out --export=ALL --wrap "bash slurm/watch_many.sh '$steps'$pairs" | tail -1)
echo "watch_many watch $id" | tee -a "$jobs"

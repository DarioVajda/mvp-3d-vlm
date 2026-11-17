#!/usr/bin/env bash
# The missing cell of v10's 2x2 (qk lr x nu init): xa3_nu = xa3_both (v9, W_q, W_k at the 5e-6 fusion lr) with nu
# log-spaced over the heads from 0.1 to 6 (--bias-nu 0.1 6), i.e. xa3_qknu without --qk-lr. Separates the effect of
# the spread nu init from that of the higher qk lr. Same recipe as slurm/launch_v10_xa_lr.sh; two seeds, seed 1 queued
# behind seed 0 (--nice).
#   slurm/launch_v10_xa_nu.sh        -> results/train/jobs_2B_v10_xa_nu.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}')
xa3=(--pe 3d --rope3 --local-bias --pe-dmax 10)
blackwell="-A lkm --constraint GPU_BRD:B300|GPU_BRD:B200"
export STABLE=5000 BRANCH_STEPS=5750 NO_WATCH=1
jobs=results/train/jobs_2B_v10_xa_nu.txt
: > "$jobs"
for seed in 0 1; do
    nice=$(( seed ? 1000 : 0 ))
    export TRAIN_SBATCH="$blackwell --nice=$nice" BRANCH_SBATCH="-t 06:00:00 $blackwell --nice=$nice"
    slurm/wsd.sh xattn_aligned "2B_v10_xa3_nu_s$seed" "${common[@]}" "${xa3[@]}" --bias-nu 0.1 6 \
        --init-seed "$seed" | tee -a "$jobs"
done
pairs=$(awk '{printf "results/train/%s %s ", $1, $3}' "$jobs")
mkdir -p logs/watch
watch=$(sbatch --parsable -p frida -A lkm -c 1 --mem 2G -t 4-00:00:00 -J watch_2B_v10_xa_nu -o "logs/watch/%x_%j.out" \
    --export=ALL,POLL=${POLL:-300} --wrap "bash slurm/watch_many.sh '$BRANCH_STEPS' $pairs" | grep -oE '^[0-9]+' | tail -1)
echo "watch $watch" | tee -a "$jobs"

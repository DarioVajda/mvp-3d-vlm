#!/usr/bin/env bash
# Round v10 (2026-11-05): can aligned cross-attention learn where to look? In v9 the AlignedXAttn W_q, W_k barely moved
# from their init at the 5e-6 fusion lr and the per-head 3D-distance slopes nu stayed near their shared init of 1.5, so
# the attention pattern was mostly the fixed 3D bias. The v9 recipe (slurm/launch_v9_frames.sh: frame mix, wsd schedule,
# step-5750 branch, 32-frame full VSI eval), with:
#   xa3_qk    xa3_both with W_q, W_k at their own lr 1e-4 (--qk-lr; everything else as in v9)
#   xa3_qknu  xa3_qk with nu log-spaced over the 16 heads from 0.1 to 6 per cell (--bias-nu 0.1 6)
#   add_mlr   additive with its mergers at 1e-4 (--merger-lr), the matching control for a higher fusion lr
# Two seeds; the seed-1 runs are submitted with a lower priority (--nice) so all three seed-0 trunks start first.
#   slurm/launch_v10_xa_lr.sh        -> results/train/jobs_2B_v10_xa_lr.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}')
xa3=(--pe 3d --rope3 --local-bias --pe-dmax 10)
blackwell="-A lkm --constraint GPU_BRD:B300|GPU_BRD:B200"
export STABLE=5000 BRANCH_STEPS=5750 NO_WATCH=1
jobs=results/train/jobs_2B_v10_xa_lr.txt
: > "$jobs"
for seed in 0 1; do
    nice=$(( seed ? 1000 : 0 ))
    export TRAIN_SBATCH="$blackwell --nice=$nice" BRANCH_SBATCH="-t 06:00:00 $blackwell --nice=$nice"
    slurm/wsd.sh xattn_aligned "2B_v10_xa3_qk_s$seed" "${common[@]}" "${xa3[@]}" --bias-nu 1.5 --qk-lr 1e-4 \
        --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh xattn_aligned "2B_v10_xa3_qknu_s$seed" "${common[@]}" "${xa3[@]}" --bias-nu 0.1 6 --qk-lr 1e-4 \
        --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh additive "2B_v10_add_mlr_s$seed" "${common[@]}" --merger-lr 1e-4 --init-seed "$seed" | tee -a "$jobs"
done
pairs=$(awk '{printf "results/train/%s %s ", $1, $3}' "$jobs")
mkdir -p logs/watch
watch=$(sbatch --parsable -p frida -A lkm -c 1 --mem 2G -t 4-00:00:00 -J watch_2B_v10_xa_lr -o "logs/watch/%x_%j.out" \
    --export=ALL,POLL=${POLL:-300} --wrap "bash slurm/watch_many.sh '$BRANCH_STEPS' $pairs" | grep -oE '^[0-9]+' | tail -1)
echo "watch $watch" | tee -a "$jobs"

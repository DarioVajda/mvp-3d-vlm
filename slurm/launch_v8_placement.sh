#!/usr/bin/env bash
# Round v8 (2026-11-02): where the geometry enters. Aligned cross-attention with the v7 3D signals (xa3_both: 3D
# distance bias + (t~, x, y, z) RoPE) and the v7 recipe of slurm/launch_v7.sh, except the
# schedule: 750 warmup + 5,000 stable steps and a single decay branch from step 5,750 (the comparison point for all
# later rounds; v7's own step-5750 branches are the controls). Two placements, two seeds each, VGGT layers 11/17/23
# reused so that only the injection points change:
#   deep  after decoder layers 0, 1, 2 and 6, 10, 14 (the v7 points + one just before each of the softmax layers
#         7, 11, 15); six merger + cross-attention pairs, twice v7's fusion parameters
#   late  after decoder layers 6, 10, 14 only; three pairs, exactly v7's fusion parameters
# Checked by mvp.fusion.check_placement.
# Trunks and branches on one B300 / B200 / H100 (slurm/submit.sh's default), 6 h for the branch as H100s are slower.
#   slurm/launch_v8_placement.sh        -> results/train/jobs_2B_v8_placement.txt
set -euo pipefail
CAP=${CAP:-50000}
common=(--appearance --vsi590k --cap "$CAP" --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200)
deep=(--fusion-after 0 1 2 6 10 14 --vggt-layers 11 17 23 11 17 23)
late=(--fusion-after 6 10 14 --vggt-layers 11 17 23)
xa3=(--pe 3d --rope3 --local-bias --bias-nu 1.5 --pe-dmax 10)
export STABLE=5000 BRANCH_STEPS=5750
export BRANCH_SBATCH="-t 06:00:00"
jobs=results/train/jobs_2B_v8_placement.txt
: > "$jobs"
for seed in 0 1; do
    slurm/wsd.sh xattn_aligned "2B_v8_xa3_deep_s$seed" "${common[@]}" "${xa3[@]}" "${deep[@]}" --init-seed "$seed" | tee -a "$jobs"
    slurm/wsd.sh xattn_aligned "2B_v8_xa3_late_s$seed" "${common[@]}" "${xa3[@]}" "${late[@]}" --init-seed "$seed" | tee -a "$jobs"
done

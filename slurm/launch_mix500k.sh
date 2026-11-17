#!/usr/bin/env bash
# The 500K-question SFT mix (mvp.train.mix): cross-attention (3D positions, spread nu init) and additive fusion, each
# on Qwen3.5-2B and Qwen3.5-4B, with the recipe of the v9/v10 runs (lrs, frame mix, appearance data). wsd with warmup
# and decay 750, the stable length fitted so that warmup + stable + decay is one epoch of the mix (--fit-epoch), 5
# equally spaced branch states (--n-branches 5), the last at the end of the trunk.
# Trunks: DDP on 4 GPUs (4B) or 2 (2B; the trunk world size is at most mb_per_step = 4). Branches: separate 1-GPU jobs
# (slurm/decay_eval.sh, 32-frame full VSI-Bench), submitted by one slurm/watch_many.sh job as each state appears, then
# one collector per trunk.
#   slurm/launch_mix500k.sh        -> results/train/jobs_mix500k.txt
# SIZES="2B" APPEND=1 resubmits only those trunks, adds them to the jobs file and starts a watcher over every trunk in
# it (cancel the old watcher first). ARMS="xattn" limits the arms.
# WATCH_ONLY=1 submits no trunks, only a new watcher over the trunks in the jobs file (cancel the old one first).
# 4B cross-attention runs on B300 only: on its 32-frame questions the backward of the biased attention allocates about
# 60 GB at once, past a B200's 180 GB (job 217719, OOM at step 821).
set -euo pipefail
common=(--mix data/mix/sft_500k.jsonl --appearance --lr 5e-5 --fusion-lr 5e-6 --weight-decay 0.01 --bias-lr 1e-3
        --eval-every 1000 --dev-every 200 --frame-mix '{"8":60,"16":30,"32":10}' --rope3 --local-bias --pe-dmax 10
        --schedule wsd --warmup-steps 750 --decay-steps 750 --fit-epoch --n-branches 5)
xattn=(--arm xattn_aligned --pe 3d --bias-nu 0.1 6)
additive=(--arm additive --pe grid --bias-nu 1.5)
blackwell=(-A lkm --constraint 'GPU_BRD:B300|GPU_BRD:B200')
jobs=results/train/jobs_mix500k.txt
[[ ${APPEND:-} == 1 || ${WATCH_ONLY:-} == 1 ]] || : > "$jobs"
sed -i '/^watch /d' "$jobs"
[[ ${WATCH_ONLY:-} == 1 ]] && SIZES=" "
for size in ${SIZES:-4B 2B}; do
    ng=$(( ${size%B} == 4 ? 4 : 2 ))
    for arm in ${ARMS:-xattn additive}; do
        name=${size}_mix500k_$arm
        gpus=("${blackwell[@]}")
        [[ $name == 4B_mix500k_xattn ]] && gpus=(-A lkm --constraint GPU_BRD:B300)
        declare -n arm_args=$arm
        id=$(LOGDIR=train PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True slurm/submit.sh "${gpus[@]}" -J "train_$name" -t 3-00:00:00 -c $(( 12 * ng )) \
            --mem $(( 160 * ng ))G --gres gpu:$ng -- torchrun --standalone --nproc_per_node $ng -m mvp.train.run \
            --model "Qwen/Qwen3.5-$size" --out "results/train/$name" "${common[@]}" "${arm_args[@]}")
        unset -n arm_args
        if [[ $name == 4B_mix500k_xattn ]]; then  # its decay branches train too (slurm/watch_many.sh)
            mkdir -p "results/train/$name"
            echo "-t 10:00:00 --mem 128G ${gpus[*]}" > "results/train/$name/branch_sbatch"
        fi
        sed -i "/^$name /d" "$jobs"
        echo "$name train $id" | tee -a "$jobs"
    done
done
pairs=$(awk '{printf "results/train/%s %s ", $1, $3}' "$jobs")
mkdir -p logs/watch
export POLL=300 BRANCH_SBATCH="-t 10:00:00 --mem 128G ${blackwell[*]}"
watch=$(sbatch --parsable -p frida -A lkm -c 1 --mem 2G -t 6-00:00:00 -J watch_mix500k -o "logs/watch/%x_%j.out" \
    --export=ALL \
    --wrap "bash slurm/watch_many.sh '' $pairs" | grep -oE '^[0-9]+' | tail -1)
echo "watch $watch" | tee -a "$jobs"

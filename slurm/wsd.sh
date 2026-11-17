#!/usr/bin/env bash
# A wsd trunk (mvp.train.run --schedule wsd, 1 GPU) plus its CPU-only branch watcher (slurm/watch_branches.sh), which
# submits one decay branch + full VSI eval per branch state and, at the end, the W&B collector.
#   slurm/wsd.sh <arm> <run name> [mvp.train.run args ...]      -> prints "<run> train <job> watch <job>"
# WARMUP (750), STABLE (10000) and BRANCH_STEPS (warmup + 5k, 6k, ..., 10k) set the schedule; TRAIN_SBATCH adds sbatch
# options to the trunk and TRAIN_LAUNCH replaces its "python" (two GPUs: TRAIN_SBATCH="--gres gpu:2 -c 16 --mem 256G"
# TRAIN_LAUNCH="torchrun --standalone --nproc_per_node 2"; the steps are the same on any number of GPUs, and branches
# always run on 1 GPU). BRANCH_SBATCH, DECAY_STEPS, EVAL_ARGS and POLL (seconds) reach the watcher through the
# environment (sbatch --export=ALL). NO_WATCH=1 submits the trunk only (prints "<run> train <job>"), for one
# slurm/watch_many.sh job over several trunks.
set -euo pipefail
arm=$1; name=$2; shift 2
WARMUP=${WARMUP:-750}; STABLE=${STABLE:-10000}
BRANCH_STEPS=${BRANCH_STEPS:-$(for k in 5000 6000 7000 8000 9000 10000; do echo -n "$((WARMUP + k)) "; done)}
trunk=results/train/$name
train=$(LOGDIR=train slurm/submit.sh -J "train_$name" -t 20:00:00 -c 8 ${TRAIN_SBATCH:-} -- \
    ${TRAIN_LAUNCH:-python} -m mvp.train.run --model Qwen/Qwen3.5-2B --arm "$arm" --out "$trunk" --schedule wsd \
    --warmup-steps "$WARMUP" --stable-steps "$STABLE" --branch-steps $BRANCH_STEPS "$@")
if [[ ${NO_WATCH:-} == 1 ]]; then echo "$name train $train"; exit 0; fi
mkdir -p logs/watch
watch=$(sbatch --parsable -p frida -A povejmo -c 1 --mem 2G -t 4-00:00:00 -J "watch_$name" -o "logs/watch/%x_%j.out" \
    --export=ALL,POLL=${POLL:-300} \
    --wrap "bash slurm/watch_branches.sh $trunk $train $BRANCH_STEPS" 2>/dev/null | grep -oE '^[0-9]+' | tail -1)
echo "$name train $train watch $watch"

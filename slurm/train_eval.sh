#!/usr/bin/env bash
# Train one arm and, the moment training completes, evaluate its best-dev checkpoint on the full VSI-Bench
# (slurm/eval_best.sh, queued with --dependency=afterok). Prints "<run> train <job> eval <job>".
#   slurm/train_eval.sh <arm> <run name> [mvp.train.run args...]
#   e.g. slurm/train_eval.sh additive 2B_v3_additive --epochs 6 --cap 6000 --fusion-lr 1e-5 --weight-decay 0.01
# TRAIN_SBATCH adds sbatch options to the training job, e.g. TRAIN_SBATCH="-t 16:00:00 --dependency=afterok:<check job>".
set -euo pipefail
arm=$1; name=$2; shift 2
out=results/train/$name
train=$(LOGDIR=train slurm/submit.sh -J "train_$name" -t 12:00:00 ${TRAIN_SBATCH:-} -- \
    python -m mvp.train.run --model Qwen/Qwen3.5-2B --arm "$arm" --out "$out" "$@")
eval=$(LOGDIR=eval slurm/submit.sh -J "eval_best_$name" -t 03:00:00 --dependency="afterok:$train" -- \
    bash slurm/eval_best.sh "$out")
echo "$name train $train eval $eval"

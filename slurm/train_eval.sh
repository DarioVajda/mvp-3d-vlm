#!/usr/bin/env bash
# Train one arm and, the moment training completes, evaluate its best-dev checkpoint on the full VSI-Bench
# (slurm/eval_best.sh, queued with --dependency=afterok). Prints "<run> train <job> eval <job>".
#   slurm/train_eval.sh <arm> <run name> [mvp.train.run args...]
#   e.g. slurm/train_eval.sh additive 2B_v3_additive --epochs 6 --cap 6000 --fusion-lr 1e-5 --weight-decay 0.01
# TRAIN_SBATCH / EVAL_SBATCH add sbatch options to the training / eval job, e.g. TRAIN_SBATCH="-t 16:00:00".
# TRAIN_LAUNCH replaces "python" for training, e.g. TRAIN_LAUNCH="torchrun --standalone --nproc_per_node 2" with
# TRAIN_SBATCH="--gres gpu:2 -c 32 --mem 256G --constraint GPU_BRD:H100" (same steps as one GPU, see mvp.train.run).
set -euo pipefail
arm=$1; name=$2; shift 2
out=results/train/$name
train=$(LOGDIR=train slurm/submit.sh -J "train_$name" -t 12:00:00 ${TRAIN_SBATCH:-} -- \
    ${TRAIN_LAUNCH:-python} -m mvp.train.run --model Qwen/Qwen3.5-2B --arm "$arm" --out "$out" "$@")
eval=$(LOGDIR=eval slurm/submit.sh -J "eval_best_$name" -t 03:00:00 ${EVAL_SBATCH:-} --dependency="afterok:$train" -- \
    bash slurm/eval_best.sh "$out")
echo "$name train $train eval $eval"

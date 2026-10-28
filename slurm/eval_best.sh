#!/usr/bin/env bash
# Final evaluation of a training run: its checkpoint with the highest dev accuracy (mvp.train.select_best) on the
# full VSI-Bench at 32 frames. Extra arguments go to mvp.eval.vsibench (e.g. --per-type 2 for a smoke test).
#   slurm/submit.sh --dependency=afterok:<train job> -J eval_best_<run> -t 03:00:00 -- bash slurm/eval_best.sh results/train/<run>
set -euo pipefail
run=$1; shift
ckpt=$(python -m mvp.train.select_best "$run")
echo "best checkpoint: $ckpt"
mkdir -p "$run/eval/best"
python -m mvp.eval.vsibench --fusion-ckpt "$ckpt" --nframes 32 --split full --out "$run/eval/best/vsi_f32_full.jsonl" "$@"

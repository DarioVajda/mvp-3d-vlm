#!/usr/bin/env bash
# One decay branch of a wsd trunk, then its full VSI-Bench evaluation at 32 frames. Submitted by slurm/watch_branches.sh
# through slurm/submit.sh (runs in the container, from the project root):
#   bash slurm/decay_eval.sh results/train/<trunk> <step>
# DECAY_STEPS (750) and EVAL_ARGS (extra vsibench options, e.g. --per-type 5 for smoke tests) come from the watcher.
# Writes results/train/<trunk>/branches/step<step>/ (log.jsonl with the post-decay dev accuracy, final.pt, eval/).
set -euo pipefail
trunk=$1; step=$2
out=$trunk/branches/step$step
python -m mvp.train.run --branch-of "$trunk" --branch-step "$step" --decay-steps "${DECAY_STEPS:-750}"
mkdir -p "$out/eval"
python -m mvp.eval.vsibench --fusion-ckpt "$out/final.pt" --nframes 32 --split full --out "$out/eval/vsi_f32_full.jsonl" ${EVAL_ARGS:-}

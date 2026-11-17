#!/usr/bin/env bash
# One decay branch of a wsd trunk, then its full VSI-Bench evaluation at 32 frames. Submitted by slurm/watch_branches.sh
# through slurm/submit.sh (runs in the container, from the project root):
#   bash slurm/decay_eval.sh results/train/<trunk> <step>
# DECAY_STEPS (750) and EVAL_ARGS (extra vsibench options, e.g. --per-type 5 for smoke tests) come from the watcher.
# NPROC (1): GPUs for the branch (torchrun, the visible ones) and for the evaluation (that many video shards in
# parallel, merged into one file; vsibench then only writes the summary, as every question is already answered).
# Writes results/train/<trunk>/branches/step<step>/ (log.jsonl with the post-decay dev accuracy, final.pt, eval/).
set -euo pipefail
trunk=$1; step=$2
out=$trunk/branches/step$step
np=${NPROC:-1}
if (( np > 1 )); then
    torchrun --standalone --nproc_per_node "$np" -m mvp.train.run --branch-of "$trunk" --branch-step "$step" \
        --decay-steps "${DECAY_STEPS:-750}"
else
    python -m mvp.train.run --branch-of "$trunk" --branch-step "$step" --decay-steps "${DECAY_STEPS:-750}"
fi
mkdir -p "$out/eval"
ev=(python -m mvp.eval.vsibench --fusion-ckpt "$out/final.pt" --nframes 32 --split full ${EVAL_ARGS:-})
if (( np > 1 )); then
    IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
    pids=()
    for ((i = 0; i < np; i++)); do
        CUDA_VISIBLE_DEVICES=${gpus[i]} "${ev[@]}" --shard "$i" --num-shards "$np" --out "$out/eval/vsi_f32_full.shard$i.jsonl" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    cat "$out"/eval/vsi_f32_full.shard*.jsonl > "$out/eval/vsi_f32_full.jsonl"
    rm "$out"/eval/vsi_f32_full.shard*
fi
"${ev[@]}" --out "$out/eval/vsi_f32_full.jsonl"

#!/usr/bin/env bash
# MMSI-Bench, MindCube-Tiny, SPAR-Bench-Tiny, VSTI-Bench and Video-MME for several models in one job. The (model,
# benchmark) tasks form one queue, longest benchmarks first; one worker per GPU takes the next task until none is left.
# Tasks with a summary are skipped and unfinished ones resume, so a resubmission picks up where a job stopped.
# A model is a trained arm's final.pt (results go to <its dir>/eval/) or a base model id (through mvp.fusion; results
# go to results/zeroshot/images/<id>/). A base model reasons before it answers MMSI and MindCube, past their short
# answer limits, so it gets BASE_MAX_NEW_TOKENS (default 1024) there. Runs inside the container:
#   slurm/submit.sh -p dev -A lkm --gres gpu:A100_80GB:4 -c 32 --mem 240G -t 12:00:00 -J eval_bench \
#       -- bash slurm/eval_bench.sh results/train/2B_mix500k_xattn/branches/step54691/final.pt ... Qwen/Qwen3.5-2B
# SMOKE=1 first runs 2 questions per task of the first and the last model (to eval/smoke/) and stops if any fails.
# BENCHES, LIMIT (questions per task; results go to eval/smoke/) and TOKEN_BUDGET (default 120000) can be set.
# Exits non-zero if any summary is missing.
set -uo pipefail
benches=(${BENCHES:-videomme vstibench spar mindcube mmsibench})  # longest first
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
export DECORD_EOF_RETRY_MAX=20480  # a few Video-MME files need more retries to reach their last frames
# Frame preprocessing runs on the CPU; uncapped, each worker starts one thread per CPU of the job and they starve
# each other (Video-MME ran ~10x slower).
export OMP_NUM_THREADS=$(( $(nproc) / ${#gpus[@]} > 0 ? $(nproc) / ${#gpus[@]} : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS

outdir() {
    local base
    if [[ $1 == *.pt ]]; then base=$(dirname "$1")/eval; else base=$(basename "$1" | tr '.-' '__'); base=results/zeroshot/images/${base,}; fi
    echo "$base${LIMIT:+/smoke}"
}

run_task() {  # model, benchmark, gpu
    local m=$1 b=$2 dir src
    dir=$(outdir "$m")
    [[ $m == *.pt ]] && src=(--fusion-ckpt "$m") || src=(--model "$m")
    [[ $m != *.pt && $b =~ ^(mmsibench|mindcube)$ ]] && src+=(--max-new-tokens "${BASE_MAX_NEW_TOKENS:-1024}")
    mkdir -p "$dir"
    echo "GPU $3: $b $m"
    if [[ $b == mmsibench ]]; then
        CUDA_VISIBLE_DEVICES=$3 python -m mvp.eval.mmsibench "${src[@]}" $([[ $m == *.pt ]] || echo --images) \
            ${LIMIT:+--per-type $LIMIT} --out "$dir/mmsibench.jsonl"
    else
        CUDA_VISIBLE_DEVICES=$3 python -m mvp.eval.bench --bench "$b" "${src[@]}" ${LIMIT:+--limit $LIMIT} \
            --token-budget "${TOKEN_BUDGET:-120000}" --out "$dir/$b.jsonl"
    fi >> "$dir/eval_bench.log" 2>&1
}

if [[ ${SMOKE:-} == 1 ]]; then
    models=("$1"); [[ $# -gt 1 ]] && models+=("${@: -1}")
    SMOKE= LIMIT=2 bash "$0" "${models[@]}" || { echo "smoke test failed"; exit 1; }
    echo "smoke test passed"
fi

queue=$(mktemp)
for b in "${benches[@]}"; do
    for m in "$@"; do
        [[ -f $(outdir "$m")/${b}_summary.json ]] || echo "$m $b" >> "$queue"
    done
done
echo "$(wc -l < "$queue") tasks on ${#gpus[@]} GPUs"
for g in "${gpus[@]}"; do
    while task=$(flock "$queue" bash -c 'head -1 "$0"; sed -i 1d "$0"' "$queue") && [[ -n $task ]]; do
        run_task $task "$g"
    done &
done
wait
rm -f "$queue"

status=0
for m in "$@"; do  # one line per model
    dir=$(outdir "$m")
    line=""
    for b in "${benches[@]}"; do
        s=$(python -c "import json; print(round(json.load(open('$dir/${b}_summary.json'))['overall'], 1))" 2>/dev/null) \
            || { s=FAILED; status=1; }
        line+="$b $s  "
    done
    echo "$m: $line"
done
exit $status

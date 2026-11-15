#!/usr/bin/env bash
# The whole comparison in one job: the setup (hpc/setup_env.sh, on the first GPU, only if no earlier setup finished),
# then every scene of hpc/scenes.tsv, one scene per GPU at a time, then results/SUMMARY.md. As in
# slurm/eval_bench.sh, the scenes form one queue, most prompts first (SAM 3 time grows with them); one worker per GPU
# takes the next scene until none is left. A scene whose comparison.json exists for every selection in RUNS is skipped,
# so a resubmission picks up where a job stopped. hpc/start.sh submits it; by hand (after creating .venv-ov4d):
#
#   slurm/submit.sh -J ov4d_scenes --gres gpu:5 -c 40 --mem 320G -t 02:00:00 -- bash tasks/openvocab_adapter/hpc/run_packed.sh
#
# 5 GPUs for 10 scenes: two rounds of five. 8 GPUs would also need two rounds (8 + 2) and leave 6 idle in the second.
# Settings as in run_all.sh (RUNS, REID, PY, DENSE_ROOT, DATA), plus SCENES_FILE (default hpc/scenes.tsv) and SETUP=1
# (run the setup again).
# One log per scene in logs/ov4d/scenes_<job id>/; each scene's run.log also goes to results/<scene>/.
# Exits non-zero if any scene failed.
set -uo pipefail
cd "$(dirname "$0")/../../.."
A=tasks/openvocab_adapter
SCENES_FILE=${SCENES_FILE:-$A/hpc/scenes.tsv}
export RUNS=${RUNS:-half}
PY=${PY:-.venv-ov4d/bin/python}
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:?no GPU visible: run this as a job}"
# SAM 3 frame loading and the CPU tests run on the CPU; uncapped, every worker starts one thread per CPU of the job
export OMP_NUM_THREADS=$(( $(nproc) / ${#gpus[@]} > 0 ? $(nproc) / ${#gpus[@]} : 1 ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
LOGS=logs/ov4d/scenes_${SLURM_JOB_ID:-local}
mkdir -p "$LOGS"

if [[ -f .venv-ov4d/.setup_ok && ${SETUP:-0} != 1 ]]; then
    echo "an earlier setup finished, skipping it (SETUP=1 runs it again)"
else
    echo "== setup on GPU ${gpus[0]} (log $LOGS/setup.out) =="
    CUDA_VISIBLE_DEVICES=${gpus[0]} bash $A/hpc/setup_env.sh > "$LOGS/setup.out" 2>&1 \
        || { echo "setup FAILED, no scene was run; see $LOGS/setup.out" >&2; tail -5 "$LOGS/setup.out" >&2; exit 1; }
    tail -3 "$LOGS/setup.out"
fi

done_already() {  # scene: true if every selection in RUNS has its comparison
    local r
    for r in $RUNS; do [[ -f $A/results/${1/\//__}/$r/comparison.json ]] || return 1; done
}

queue=$(mktemp)
while IFS=$'\t' read -r scene prompts; do
    prompts=${prompts%$'\r'}
    [[ -z $scene || $scene == \#* ]] && continue
    if done_already "$scene"; then echo "skip $scene (done)"; continue; fi
    n=$(tr ',' '\n' <<< "$prompts" | grep -c .)
    printf '%s\t%s\t%s\n' "$n" "$scene" "$prompts"
done < "$SCENES_FILE" | sort -t$'\t' -k1,1nr | cut -f2- > "$queue"
echo "$(wc -l < "$queue") scenes on ${#gpus[@]} GPUs ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)), runs: $RUNS"

failed=$(mktemp)
for g in "${gpus[@]}"; do
    while line=$(flock "$queue" bash -c 'head -1 "$0"; sed -i 1d "$0"' "$queue") && [[ -n $line ]]; do
        IFS=$'\t' read -r scene prompts <<< "$line"
        log=$LOGS/${scene/\//__}.out
        echo "[$(date +%H:%M:%S)] GPU $g: $scene (log $log)"
        if CUDA_VISIBLE_DEVICES=$g SCENE=$scene PROMPTS=$prompts bash $A/hpc/run_all.sh > "$log" 2>&1; then
            echo "[$(date +%H:%M:%S)] GPU $g: $scene done"
        else
            echo "[$(date +%H:%M:%S)] GPU $g: $scene FAILED (exit $?, see $log)"
            flock "$failed" bash -c 'echo "$1" >> "$0"' "$failed" "$scene"
        fi
    done &
done
wait
rm -f "$queue"

echo "== summary =="
$PY $A/summarize.py || echo "summarize.py failed"
if [[ -s $failed ]]; then
    echo "failed scenes: $(paste -sd' ' "$failed")"
    rm -f "$failed"
    exit 1
fi
rm -f "$failed"
echo "all scenes finished"

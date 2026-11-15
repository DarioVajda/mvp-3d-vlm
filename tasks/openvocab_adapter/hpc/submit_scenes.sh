#!/usr/bin/env bash
# Submit the comparison for every scene in hpc/scenes.tsv: one GPU job per scene (they run in parallel), then a small
# job that waits for all of them and writes results/SUMMARY.md. Run it on the login node, after hpc/setup_env.sh:
#
#   bash tasks/openvocab_adapter/hpc/submit_scenes.sh
#
# Optional: RUNS (default "half": every other frame), SCENES_FILE (default hpc/scenes.tsv), TIME (default 01:00:00 per scene),
# AFTER=<job id>: start the scenes only once that job has succeeded, and cancel them if it fails (used by hpc/start.sh).
set -euo pipefail
cd "$(dirname "$0")/../../.."
SCENES_FILE=${SCENES_FILE:-tasks/openvocab_adapter/hpc/scenes.tsv}
export RUNS=${RUNS:-half}
TIME=${TIME:-01:00:00}
export LOGDIR=ov4d
after=()
[[ -n ${AFTER:-} ]] && after=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)

ids=()
while IFS=$'\t' read -r scene prompts; do
    prompts=${prompts%$'\r'}  # in case the file was saved with Windows line endings
    [[ -z $scene || $scene == \#* ]] && continue
    name=${scene//\//__}
    id=$(SCENE=$scene PROMPTS=$prompts slurm/submit.sh -J "ov4d_$name" -t "$TIME" ${after[@]+"${after[@]}"} -- \
        bash tasks/openvocab_adapter/hpc/run_all.sh)
    echo "submitted $scene (selections: $RUNS) as job $id"
    ids+=("$id")
done < "$SCENES_FILE"
[[ ${#ids[@]} -gt 0 ]] || { echo "no scenes in $SCENES_FILE" >&2; exit 1; }

deps=$(IFS=:; echo "${ids[*]}")
id=$(slurm/submit.sh -J ov4d_summary -t 00:10:00 --dependency="afterany:$deps" -- \
    .venv-ov4d/bin/python tasks/openvocab_adapter/summarize.py)
echo "submitted the summary as job $id; it runs once all ${#ids[@]} scenes have finished"
echo "follow with: squeue -u \$USER    logs: logs/$LOGDIR/"

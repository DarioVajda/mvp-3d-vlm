#!/usr/bin/env bash
# Everything in one command, on the login node:
#
#   bash tasks/openvocab_adapter/hpc/start.sh
#
# 1. checks the inputs of every scene in hpc/scenes.tsv, before anything is submitted
# 2. creates .venv-ov4d if it is missing (here: the container has no ensurepip)
# 3. submits the setup job, unless an earlier setup finished
# 4. submits one job per scene (they wait for the setup, and are cancelled if it fails) and the summary job
#
# Settings are passed on to submit_scenes.sh (RUNS, TIME, SCENES_FILE). SETUP=1 forces the setup job to run again.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SCENES_FILE=${SCENES_FILE:-tasks/openvocab_adapter/hpc/scenes.tsv}
DENSE_ROOT=${DENSE_ROOT:-data/vggt}
export SCENES_FILE DENSE_ROOT

echo "== 1. inputs =="
missing=()
count=0
while IFS=$'\t' read -r scene prompts; do
    [[ -z $scene || $scene == \#* ]] && continue
    count=$((count + 1))
    for f in depth.npy depth_conf.npy; do
        [[ -f $DENSE_ROOT/$scene/$f ]] || missing+=("$DENSE_ROOT/$scene/$f")
    done
done < "$SCENES_FILE"
if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'missing: %s\n' "${missing[@]}" >&2
    echo "Nothing was submitted. Run this from the project checkout that has data/vggt/, or remove those scenes from $SCENES_FILE" >&2
    exit 1
fi
echo "dense depth found for all $count scenes in $SCENES_FILE"
if [[ -x .venv/bin/python ]]; then  # a hint only: the setup job does the real check
    .venv/bin/python -c "from huggingface_hub import auth_check; auth_check('facebook/sam3')" 2>/dev/null \
        && echo "Hugging Face: access to facebook/sam3 ok" \
        || echo "warning: could not confirm access to facebook/sam3; if the setup job fails on it, accept the license at https://huggingface.co/facebook/sam3 and run 'hf auth login'"
fi

echo "== 2. environment =="
if [[ ! -x .venv-ov4d/bin/python ]]; then
    ${SYSTEM_PYTHON:-/usr/bin/python3} -m venv .venv-ov4d
    echo "created .venv-ov4d"
else
    echo ".venv-ov4d exists"
fi

echo "== 3. setup job =="
mkdir -p logs/ov4d
if [[ -f .venv-ov4d/.setup_ok && ${SETUP:-0} != 1 ]]; then
    echo "an earlier setup finished, skipping it (SETUP=1 runs it again)"
    unset AFTER
else
    AFTER=$(LOGDIR=ov4d slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash tasks/openvocab_adapter/hpc/setup_env.sh)
    export AFTER
    echo "submitted the setup as job $AFTER (log: logs/ov4d/ov4d_setup_$AFTER.out)"
fi

echo "== 4. scene jobs =="
bash tasks/openvocab_adapter/hpc/submit_scenes.sh

cat <<EOF

All submitted. Nothing else to do until the jobs finish:
  squeue -u \$USER            shows what is still queued or running (empty = all done)
When the list is empty, send the results back:
  git add tasks/openvocab_adapter/results
  git commit -m 'OpenVocab-4D comparison results'
  git push
EOF

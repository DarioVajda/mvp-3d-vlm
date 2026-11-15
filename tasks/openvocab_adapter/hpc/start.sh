#!/usr/bin/env bash
# Everything in one command, on the login node, as one Slurm job:
#
#   bash tasks/openvocab_adapter/hpc/start.sh
#
# 1. checks the inputs of every scene in hpc/scenes.tsv, before anything is submitted
# 2. creates .venv-ov4d if it is missing, and clones SAM 3 and VGGT at OpenVocab-4D's pinned commits (both on the
#    login node: the container has no ensurepip, and may have no git)
# 3. submits one job (hpc/run_packed.sh) on GPUS GPUs of one node: the setup if no earlier setup finished, then every
#    scene, one per GPU at a time, then the summary
#
# Settings: GPUS (default 5), TIME (default 02:00:00), RUNS, SCENES_FILE, DENSE_ROOT, DATA; SETUP=1 runs the setup
# again. A short test first, on two scenes:
#   RUNS=smoke5 GPUS=2 TIME=01:00:00 SCENES_FILE=tasks/openvocab_adapter/hpc/scenes_smoke.tsv \
#       bash tasks/openvocab_adapter/hpc/start.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
SCENES_FILE=${SCENES_FILE:-tasks/openvocab_adapter/hpc/scenes.tsv}
DENSE_ROOT=${DENSE_ROOT:-data/vggt}
RUNS=${RUNS:-half}
SETUP=${SETUP:-0}
export SCENES_FILE DENSE_ROOT RUNS SETUP

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

OV=openvocab-4D-main
clone() {  # clone <url> <dir> <commit>: the commits pinned in hpc/setup_env.sh
    [[ -d $2 ]] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
}
clone https://github.com/facebookresearch/vggt $OV/vggt a288dd0f14786c93483e45524328726ab7b1b4ce
clone https://github.com/facebookresearch/sam3 $OV/sam3 5dd401d1c5c1d5c3eedff06d41b77af824517619
echo "SAM 3 and VGGT cloned at the pinned commits"

echo "== 3. one job =="
GPUS=${GPUS:-5}
mkdir -p logs/ov4d
id=$(LOGDIR=ov4d slurm/submit.sh -J ov4d_compare --gres "gpu:$GPUS" -c $((8 * GPUS)) --mem $((64 * GPUS))G \
    -t "${TIME:-02:00:00}" --constraint 'GPU_BRD:B300|GPU_BRD:B200' -- bash tasks/openvocab_adapter/hpc/run_packed.sh)
echo "submitted job $id on $GPUS GPUs (runs: $RUNS); log: logs/ov4d/ov4d_compare_$id.out"

cat <<EOF

All submitted. Nothing else to do until the job finishes:
  squeue -u \$USER            shows whether it is still queued or running (empty = done)
When it is done, send the results back:
  git add tasks/openvocab_adapter/results
  git commit -m 'OpenVocab-4D comparison results'
  git push
EOF

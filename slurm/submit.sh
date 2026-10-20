#!/usr/bin/env bash
# Submit a command to Slurm; it runs inside the project container with .venv first on PATH.
#
#   slurm/submit.sh [sbatch options] -- <command ...>
#
# Defaults: partition frida, 1 GPU (B200/B300),
# 16 CPUs, 128G RAM, 1 hour. Any sbatch option given here overrides the default, e.g.
#   slurm/submit.sh -J t1 -t 02:00:00 --constraint GPU_BRD:B300 -- python src/experiments/pca_analysis/hidden_state.py --out ...
# The command runs from the project root; the log goes to logs/<job name>_<job id>.out.
# Prints the job id.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

opts=()
while [[ $# -gt 0 && $1 != -- ]]; do opts+=("$1"); shift; done
[[ $# -gt 1 ]] || { echo "usage: $0 [sbatch options] -- <command ...>" >&2; exit 1; }
shift
cmd=$(printf '%q ' "$@")
mkdir -p "$ROOT/logs"

sbatch --parsable -p frida -A povejmo -c 16 --mem 128G -t 01:00:00 --gres gpu:1 \
    --constraint 'GPU_BRD:B300|GPU_BRD:B200' \
    -J mvp -o "$ROOT/logs/%x_%j.out" "${opts[@]}" \
    --wrap "srun --export=ALL,MELLANOX_VISIBLE_DEVICES=none --container-image=$CONTAINER --container-mounts=/shared:/shared \
        env HOME=$HOME PYTHONUNBUFFERED=1 bash -c 'cd $ROOT && export PATH=$ROOT/.venv/bin:\$PATH && echo [\$(hostname)] && $cmd'"

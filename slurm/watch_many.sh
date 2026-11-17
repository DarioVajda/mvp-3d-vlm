#!/usr/bin/env bash
# Several slurm/watch_branches.sh watchers in one CPU-only job (one job instead of one per trunk, for the per-user job
# limit), each line of the log prefixed with its trunk's name; ends when every watcher has.
#   bash slurm/watch_many.sh "<step> [<step> ...]" results/train/<trunk> <trunk job id> [<trunk> <job> ...]
# ("" for the steps: each watcher reads its trunk's branch_steps, see slurm/watch_branches.sh)
# Submit with the same environment as slurm/wsd.sh's watcher (POLL, BRANCH_SBATCH, DECAY_STEPS, EVAL_ARGS); a trunk's
# <trunk>/branch_sbatch file, if present, replaces BRANCH_SBATCH for that trunk's branches:
#   sbatch -p frida -A povejmo -c 1 --mem 2G -t 4-00:00:00 -J watch_<round> -o logs/watch/%x_%j.out --export=ALL \
#       --wrap "bash slurm/watch_many.sh '5750' results/train/a 123 results/train/b 124"
set -uo pipefail
cd "$(dirname "$0")/.."
read -ra steps <<< "$1"; shift
(( $# % 2 == 0 )) || { echo "usage: $0 \"<steps>\" <trunk> <job> [<trunk> <job> ...]" >&2; exit 1; }
while (( $# )); do
    trunk=$1 job=$2; shift 2
    name=$(basename "$trunk")
    bs=${BRANCH_SBATCH:-}; [[ -f $trunk/branch_sbatch ]] && bs=$(< "$trunk/branch_sbatch")
    BRANCH_SBATCH=$bs bash slurm/watch_branches.sh "$trunk" "$job" "${steps[@]}" 2>&1 | sed -u "s/^/[$name] /" &
done
wait

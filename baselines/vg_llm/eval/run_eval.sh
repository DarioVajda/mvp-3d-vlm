#!/usr/bin/env bash
# Run VG-LLM's evaluation command from third_party/VG-LLM/scripts/evaluation/eval.sh (and the README): their bundled
# lmms_eval, --model vgllm, pretrained=<checkpoint>,use_flash_attention_2=true,max_num_frames=32,max_length=12800,
# --batch_size 1, from the repo root. Only the number of processes (eval.sh hard-codes 8) follows the GPUs of the job,
# and the checkpoint is given as its pinned local snapshot.
#   run_eval.sh <4b|8b> <task[,task...]> <output dir> [limit]
# The vgllm tasks used are vsibench, cvbench and blink (as in the paper). Two task files are added with --include_path:
#   vsibench_shard  VG-LLM's vsibench restricted to one contiguous slice, chosen with VSI_SHARD=<index>/<number>
#                   (eval/tasks/vsibench_shard; merge_shards.py joins and scores the slices);
#   mmsibench       not one of VG-LLM's tasks; symlinks to SpatialStack's task file, the one used for our SpatialStack runs.
# VG_ENV=blackwell selects the B200/B300 cross-check environment (env/setup_blackwell.sh) instead of the pinned one.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)  # the method directory
MVP=$(cd "$HERE/../.." && pwd)
REPO=$MVP/third_party/VG-LLM
SNAP=$HOME/.cache/huggingface/hub
case $1 in
  4b) CKPT=$SNAP/models--zd11024--vgllm-qa-vggt-4b/snapshots/1cf227a134353753aba6ac6c7509d2f79d40d00c ;;
  8b) CKPT=$SNAP/models--zd11024--vgllm-qa-vggt-8b/snapshots/73d51445fbb307c453c944bc3354aef68ec3cc59 ;;
  *) echo "unknown model $1" >&2; exit 1 ;;
esac
TASKS=$2 LIMIT=${4:-}
mkdir -p "$3"
OUT=$(cd "$3" && pwd)
export LMMS_EVAL_LAUNCHER=accelerate NCCL_NVLS_ENABLE=0
export HF_HOME=$HOME/.cache/huggingface
# datasets 3.6.0 cannot read cache metadata written by newer datasets versions
export HF_DATASETS_CACHE=$HOME/.cache/vg_llm/datasets
NPROC=${SLURM_GPUS_ON_NODE:-1}
case ${VG_ENV:-pinned} in
  pinned) VENV=$HERE/.venv FREEZE=$HERE/env/freeze.txt ;;
  blackwell) VENV=$HERE/.venv-blackwell FREEZE=$HERE/env/freeze_blackwell.txt ;;
  *) echo "unknown VG_ENV $VG_ENV" >&2; exit 1 ;;
esac
{ echo "date: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
  echo "repo: $(git -C "$REPO" rev-parse HEAD)"; echo "checkpoint: $CKPT"; echo "tasks: $TASKS"; echo "limit: $LIMIT"
  echo "processes: $NPROC"; echo "VSI_SHARD: ${VSI_SHARD:-}"; echo "environment: ${VG_ENV:-pinned}"; } > "$OUT/manifest.txt"
cp "$FREEZE" "$OUT/"
cd "$REPO"
"$VENV/bin/accelerate" launch --num_processes=$NPROC --main_process_port $((29500 + RANDOM % 1000)) -m lmms_eval \
    --model vgllm \
    --model_args pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800 \
    --tasks "$TASKS" \
    --include_path "$HERE/eval/tasks" \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUT" \
    ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a "$OUT/run.log"

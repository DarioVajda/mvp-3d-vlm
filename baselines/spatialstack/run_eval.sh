#!/usr/bin/env bash
# Run SpatialStack's own scripts/evaluation/eval.sh unchanged, with only paths set through its environment variables.
#   run_eval.sh <qwen35|qwen25> <benchmarks> <output dir> [limit]
# Model arguments are the README's (Qwen3.5) and the eval.sh defaults (Qwen2.5). The checkpoint is given as its
# pinned local snapshot. Qwen3.5 uses MODEL_IMPL=qwen3_5 as in the authors' Slurm script
# (scripts/evaluation/run_qwen35_vsibench_multinode.sbatch); the README example leaves MODEL_IMPL at its default
# "spatialstack", which only supports Qwen2.5 and rejects the Qwen3.5 arguments. Qwen2.5 runs in its own
# transformers-4.50 environment (setup_qwen25.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$HERE/../../third_party/SpatialStack
SNAP=$HOME/.cache/huggingface/hub
case $1 in
  qwen35) CKPT=$SNAP/models--Journey9ni--SpatialStack-Qwen3.5-4B/snapshots/777b2289252f7c78628e0f8ac63ffddf50fe7f7b
          export MODEL_IMPL=qwen3_5 QWEN35_ENV_ROOT=$HERE/.venv FREEZE=$HERE/freeze.txt
          export HF_DATASETS_CACHE=$HOME/.cache/huggingface/datasets
          export MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800,geometry_encoder_path=facebook/VGGT-1B,disable_thinking=true" ;;
  qwen25) CKPT=$SNAP/models--Journey9ni--SpatialStack-Qwen2.5-4B/snapshots/6e70accdad7cde7dede9a497a88deb94d992fd9e
          export MODEL_IMPL=spatialstack QWEN35_ENV_ROOT=$HERE/.venv-qwen25 FREEZE=$HERE/freeze_qwen25.txt
          # eval.sh puts $CUDA_HOME/lib64 (the container's CUDA 12.1) first on LD_LIBRARY_PATH, whose libnvJitLink is
          # older than the one torch 2.5.1's cuSPARSE needs; load the venv's own copy first.
          # datasets 3.6.0 cannot read cache metadata written by the newer datasets of the Qwen3.5 environment.
          export HF_DATASETS_CACHE=$HOME/.cache/spatialstack/datasets-qwen25
          export LD_PRELOAD=$(ls "$QWEN35_ENV_ROOT"/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12)
          export MODEL_ARGS_BASE="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800" ;;
  *) echo "unknown model $1" >&2; exit 1 ;;
esac
export BENCHMARKS=$2 OUTPUT_PATH=$3 LIMIT=${4:-}
export HF_HOME=$HOME/.cache/huggingface HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub
export CACHE_ROOT=$HOME/.cache/spatialstack
export PROJECT_ROOT=$REPO
mkdir -p "$OUTPUT_PATH"
{ echo "date: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
  echo "repo: $(git -C "$REPO" rev-parse HEAD)"; echo "checkpoint: $CKPT"; echo "MODEL_IMPL=$MODEL_IMPL"
  echo "MODEL_ARGS_BASE=$MODEL_ARGS_BASE"; echo "BENCHMARKS=$BENCHMARKS LIMIT=$LIMIT"; } > "$OUTPUT_PATH/manifest.txt"
cp "$FREEZE" "$OUTPUT_PATH/"
bash "$REPO/scripts/evaluation/eval.sh"

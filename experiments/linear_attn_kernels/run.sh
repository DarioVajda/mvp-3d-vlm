#!/bin/bash
# Qwen3.5-4B with vs without the linear-attention kernels on whatever GPU the job lands on.
#   LOGDIR=env slurm/submit.sh -J la_bench_B300 -t 00:45:00 --constraint GPU_BRD:B300 -- bash experiments/linear_attn_kernels/run.sh
# NAME labels the kernel run (kernels: fla only, kernels_conv: fla + causal-conv1d); the PyTorch reference run is
# skipped if it already exists for this GPU.
set -e
cd "$(dirname "$0")/../.."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cd '[:alnum:]')
OUT=results/_scratch/linear_attn_kernels/$GPU
mkdir -p $OUT
.venv/bin/python experiments/linear_attn_kernels/bench.py --model ${MODEL:-Qwen/Qwen3.5-4B} --out $OUT/${NAME:-kernels}.json
[[ -f $OUT/torch.json ]] || .venv/bin/python experiments/linear_attn_kernels/bench.py --model ${MODEL:-Qwen/Qwen3.5-4B} --no-kernels --out $OUT/torch.json

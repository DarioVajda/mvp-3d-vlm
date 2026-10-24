#!/usr/bin/env bash
# Fast kernels for Qwen3.5's linear-attention (Gated DeltaNet) layers, picked up by transformers automatically:
#   flash-linear-attention  gated delta rule (Triton, so it JIT-compiles for whatever GPU it runs on);
#   causal-conv1d           the short depthwise convolution (CUDA extension).
# causal-conv1d has no prebuilt wheel for torch 2.14 and the container's nvcc is CUDA 12.1, so it is built from
# source against a pip-installed CUDA 13.0 toolkit (.build/cuda13, matching torch's cu130). With nvcc 13 its setup.py
# compiles for sm_75/80/87/90/100/103/110/120/121, i.e. A100, H100, B200 and B300 in one wheel. The wheel is kept in
# .build/wheels, so later reinstalls skip the ~15 min build. No GPU needed:
#   LOGDIR=env slurm/submit.sh -J la_kernels -t 01:00:00 --gres gpu:0 -c 32 -- bash slurm/install_linear_attn_kernels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PIP=".venv/bin/pip install -q"
$PIP flash-linear-attention==0.6.0 ninja packaging

WHEELS=.build/wheels
if ! ls $WHEELS/causal_conv1d-1.7.0-*.whl >/dev/null 2>&1; then
    TK=$PWD/.build/cuda13
    $PIP --target $TK nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
        nvidia-cuda-cccl==13.0.85 nvidia-cuda-runtime==13.0.96
    export CUDA_HOME=$TK/nvidia/cu13 PATH=$TK/nvidia/cu13/bin:$PATH
    ln -sfn lib $CUDA_HOME/lib64                                   # torch.utils.cpp_extension links from lib64
    ln -sf libcudart.so.13 $CUDA_HOME/lib/libcudart.so
    nvcc --version | tail -1
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${SLURM_CPUS_PER_TASK:-16} \
        .venv/bin/pip wheel -v --no-build-isolation --no-deps causal-conv1d==1.7.0 -w $WHEELS 2>&1 | grep -E "nvcc|error|Error|Successfully|warning: " | grep -v -- "-gencode" | tail -40
fi
$PIP --no-deps $WHEELS/causal_conv1d-1.7.0-*.whl
.venv/bin/python -c "
import torch, fla, causal_conv1d, causal_conv1d_cuda
print('torch', torch.__version__, 'fla', fla.__version__, 'causal_conv1d', causal_conv1d.__version__)"

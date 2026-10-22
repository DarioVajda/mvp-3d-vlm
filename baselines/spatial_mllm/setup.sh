#!/usr/bin/env bash
# Spatial-MLLM environment, following third_party/Spatial-MLLM/README.md (Setup, step 2): Python 3.10,
# torch 2.6.0+cu124, transformers 4.51.3, the README's unpinned packages, prebuilt flash_attn 2.7.4.post1.
# uv venv instead of conda. The unpinned packages are resolved as of the v1.1 release (2026-01-05) with
# --exclude-newer. setuptools is listed because conda environments include it and torch imports it at runtime.
# torch 2.6/cu124 has no Blackwell kernels, so this environment runs on H100/A100 only.
# Runs on the login node (downloads only, no compilation).
set -euo pipefail
cd "$(dirname "$0")"
UV=~/.local/bin/uv
[[ -d .venv ]] || $UV venv --python 3.10 .venv
PIP="$UV pip install --python .venv/bin/python"

$PIP torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$PIP --exclude-newer 2026-01-06 transformers==4.51.3 torch==2.6.0 torchvision==0.21.0 \
  setuptools accelerate datasets decord deepspeed einops matplotlib pandas python_Levenshtein qwen_vl_utils ray safetensors tqdm tyro wandb
$PIP https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

$UV pip freeze --python .venv/bin/python > freeze.txt

#!/usr/bin/env bash
# SpatialStack environment, following third_party/SpatialStack/README.md (Setup, steps 1-6 + lmms_eval extras).
# Python 3.12 venv (uv-managed interpreter) instead of conda. One deviation: the README builds flash_attn against
# torch 2.10.0+cu129; the container only has nvcc 12.1, so we use torch 2.10.0+cu130 with the official prebuilt
# flash_attn 2.8.3 and causal-conv1d wheels for torch 2.10 / CUDA 13. All Python package versions match the README.
# Runs on the login node (downloads only, no compilation).
set -euo pipefail
cd "$(dirname "$0")"
UV=~/.local/bin/uv
[[ -d .venv ]] || $UV venv --python 3.12 .venv
PIP="$UV pip install --python .venv/bin/python"

$PIP torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
$PIP psutil ninja wheel setuptools packaging
$PIP https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
$PIP transformers==5.3.0 accelerate==1.13.0 qwen_vl_utils==0.0.14 decord
$PIP https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
FLA_COMMIT=$(git ls-remote https://github.com/fla-org/flash-linear-attention HEAD | cut -f1)
$PIP "git+https://github.com/fla-org/flash-linear-attention@$FLA_COMMIT"
$PIP -e ../../third_party/SpatialStack --no-deps
$PIP datasets pyarrow evaluate pytablewriter pandas \
  loguru jsonlines sqlitedict sacrebleu terminaltables zss tenacity==8.3.0 \
  wandb openai tiktoken scipy openpyxl numexpr sympy nltk sentencepiece ftfy \
  timm opencv-python-headless av tqdm-multiprocess transformers-stream-generator \
  hf_transfer

$UV pip freeze --python .venv/bin/python > freeze.txt
echo "fla commit: $FLA_COMMIT" >> freeze.txt

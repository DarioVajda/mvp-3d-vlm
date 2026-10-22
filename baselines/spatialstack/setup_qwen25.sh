#!/usr/bin/env bash
# Environment for SpatialStack-Qwen2.5-4B. Its model code (src/qwen_vl/model/modeling_qwen2_5_vl.py) is the VG-LLM
# fork written for transformers 4.x, and the checkpoint's config.json records transformers 4.50.0, so the Qwen3.5
# environment (transformers 5.3.0) cannot import it. We use VG-LLM's pins (third_party/VG-LLM/setup.py): Python 3.10,
# torch 2.5.1, transformers 4.50.0, accelerate 1.4.0, flash_attn 2.7.4.post1, datasets 3.6.0, numpy 1.26.4.
# Unpinned packages are resolved as of the checkpoint release (2026-02-27). torch 2.5.1 has no Blackwell kernels,
# so this environment runs on H100/A100 only. Runs on the login node (downloads only, no compilation).
set -euo pipefail
cd "$(dirname "$0")"
UV=~/.local/bin/uv
V=.venv-qwen25
[[ -d $V ]] || $UV venv --python 3.10 $V
PIP="$UV pip install --python $V/bin/python"
PINS="torch==2.5.1 torchvision==0.20.1 transformers==4.50.0 accelerate==1.4.0 datasets==3.6.0 numpy==1.26.4"

$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
$PIP https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
$PIP --exclude-newer 2026-02-28 $PINS qwen_vl_utils decord
$PIP -e ../../third_party/SpatialStack --no-deps
$PIP --exclude-newer 2026-02-28 $PINS pyarrow evaluate pytablewriter pandas \
  loguru jsonlines sqlitedict sacrebleu terminaltables zss tenacity==8.3.0 \
  wandb openai tiktoken scipy openpyxl numexpr sympy nltk sentencepiece ftfy \
  timm opencv-python-headless av tqdm-multiprocess transformers-stream-generator \
  hf_transfer einops

$UV pip freeze --python $V/bin/python > freeze_qwen25.txt

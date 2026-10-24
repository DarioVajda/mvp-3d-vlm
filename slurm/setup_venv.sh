#!/usr/bin/env bash
# Install requirements.txt into .venv inside the container (venv itself created on login node with
# /usr/bin/python3 -m venv .venv, as the container has no ensurepip).
#   slurm/submit.sh -J setup_venv -t 00:30:00 -- bash slurm/setup_venv.sh
set -euo pipefail
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps -e .   # the mvp package (src/mvp), importable from anywhere
bash slurm/install_linear_attn_kernels.sh  # Qwen3.5 kernels; builds causal-conv1d (~45 min) unless .build/wheels has it
.venv/bin/python -c "import torch, transformers, qwen_vl_utils, decord; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

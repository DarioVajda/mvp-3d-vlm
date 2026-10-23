#!/usr/bin/env bash
# Cross-check environment for running VG-LLM on B200/B300 GPUs, which the pinned environment (setup.sh: torch 2.5.1,
# triton 3.1.0, flash_attn 2.7.4.post1) cannot run on. Same transformers/accelerate/datasets/numpy pins and the same
# VG-LLM code; only torch 2.7.1+cu128 (and its triton), torchvision 0.22.1 and flash_attn 2.8.3 differ. torchcodec
# (pinned to a torch-2.5 build, not imported by the evaluation) is left out. Results from this environment are a
# cross-check against the pinned one, not the reproduction itself.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../third_party/VG-LLM && pwd)
UV=~/.local/bin/uv
V=.venv-blackwell
[[ -d $V ]] || $UV venv --python 3.10 $V
PIP="$UV pip install --python $V/bin/python"

mapfile -t REQS < <(python3 - "$REPO/setup.py" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup")
reqs = next(k.value for k in call.keywords if k.arg == "install_requires")
skip = {"pytorch3d", "flash_attn", "torch", "torchvision", "triton", "torchcodec"}
for r in ast.literal_eval(reqs):
    if r.split("=")[0].split(">")[0] not in skip:
        print(r)
PY
)

$PIP torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
$PIP https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
DS_BUILD_OPS=0 $PIP --exclude-newer 2025-09-24 --no-build-isolation-package deepspeed setuptools wheel \
  torch==2.7.1 torchvision==0.22.1 "${REQS[@]}"
$PIP -e "$REPO" --no-deps

$UV pip freeze --python $V/bin/python > freeze_blackwell.txt

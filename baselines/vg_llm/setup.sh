#!/usr/bin/env bash
# VG-LLM environment, following third_party/VG-LLM/README.md (Setup: Python 3.10, `pip install -e .`). uv venv
# instead of conda. The dependency list is read from the repo's setup.py and installed as written, except pytorch3d,
# which has no prebuilt wheel and is only imported by the 3D video detection task (lmms_eval/tasks/threedod), and
# flash_attn, which is installed from its prebuilt wheel. Unpinned packages are resolved as of the release of the
# last checkpoint we evaluate (vgllm-qa-vggt-8b, 2025-09-23) with --exclude-newer.
# torch 2.5.1/cu124 has no Blackwell kernels, so this environment runs on H100/A100 only.
# Runs on the login node (downloads only, no compilation).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../third_party/VG-LLM && pwd)
UV=~/.local/bin/uv
[[ -d .venv ]] || $UV venv --python 3.10 .venv
PIP="$UV pip install --python .venv/bin/python"

mapfile -t REQS < <(python3 - "$REPO/setup.py" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setup")
reqs = next(k.value for k in call.keywords if k.arg == "install_requires")
skip = {"pytorch3d", "flash_attn"}
for r in ast.literal_eval(reqs):
    if r.split("=")[0].split(">")[0] not in skip:
        print(r)
PY
)
echo "installing ${#REQS[@]} requirements from setup.py"

$PIP torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
$PIP https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# deepspeed's setup.py imports torch, so it is built against the venv's torch (no compiled ops: DS_BUILD_OPS=0)
DS_BUILD_OPS=0 $PIP --exclude-newer 2025-09-24 --no-build-isolation-package deepspeed setuptools wheel "${REQS[@]}"
$PIP -e "$REPO" --no-deps

$UV pip freeze --python .venv/bin/python > freeze.txt

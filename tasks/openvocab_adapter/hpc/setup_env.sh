#!/usr/bin/env bash
# One-time setup: an environment for OpenVocab-4D (SAM 3 + VGGT) on the cluster's B200/B300 GPUs, in .venv-ov4d.
#
# OpenVocab-4D's own install.sh installs PyTorch for CUDA 12.6, which has no kernels for Blackwell GPUs. This uses the
# project's PyTorch instead (requirements.txt, CUDA 13.0), with OpenVocab-4D's pinned SAM 3 and VGGT commits and
# numpy<2, which they need. The project's .venv is not touched.
#
# On the login node (the container has no ensurepip), then as a job:
#   /usr/bin/python3 -m venv .venv-ov4d
#   slurm/submit.sh -J ov4d_setup -t 00:45:00 -- bash tasks/openvocab_adapter/hpc/setup_env.sh
#
# SAM 3's weights are gated: accept the license at https://huggingface.co/facebook/sam3 with the account whose token
# is in ~/.cache/huggingface or $HF_HOME. The last step below checks this.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv-ov4d/bin/python
OV=openvocab-4D-main
NUMPY_SPEC=${NUMPY_SPEC:-"numpy>=1.26,<2"}

[[ -x $PY ]] || { echo "no $PY: create it on the login node with /usr/bin/python3 -m venv .venv-ov4d" >&2; exit 1; }
rm -f .venv-ov4d/.setup_ok  # written again at the end, only if every check passes (hpc/start.sh looks for it)
$PY --version

clone() {  # clone <url> <dir> <commit>
    if ! command -v git >/dev/null && [[ -d $2 ]]; then  # cloned on the login node by hpc/start.sh
        echo "no git here: using $2 as cloned"; return
    fi
    if [[ ! -d $2 ]]; then
        command -v git >/dev/null || { echo "no git in this container: run 'git clone $1 $2' on the login node" >&2; exit 1; }
        git clone -q "$1" "$2"
    fi
    git -C "$2" checkout -q "$3"
}
# the commits pinned in openvocab-4D-main/install.sh; milestone 4 imports VGGT from openvocab-4D-main/vggt
clone https://github.com/facebookresearch/vggt $OV/vggt a288dd0f14786c93483e45524328726ab7b1b4ce
clone https://github.com/facebookresearch/sam3 $OV/sam3 5dd401d1c5c1d5c3eedff06d41b77af824517619

$PY -m pip install -q --upgrade pip
$PY -m pip install -q "torch==2.14.1+cu130" torchvision --extra-index-url https://download.pytorch.org/whl/cu130
# --no-deps, so that neither package can swap the CUDA 13.0 PyTorch for another build
$PY -m pip install -q --no-deps -e $OV/vggt -e $OV/sam3
$PY -m pip install -q "$NUMPY_SPEC" Pillow huggingface_hub einops safetensors "rerun-sdk>=0.23" "pyarrow<26" \
    opencv-python-headless "timm>=1.0.17" "transformers==5.20.1" "setuptools<81" \
    ftfy regex iopath pycocotools psutil tqdm typing_extensions
# setuptools<81: SAM 3's model_builder imports pkg_resources, which setuptools 81 removed

echo "== checks =="
$PY - <<'EOF'
import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__)
assert torch.cuda.is_available(), "no CUDA GPU visible: run this as a job (slurm/submit.sh)"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()  # fails here if this PyTorch has no kernels for the GPU
print("GPU ok:", torch.cuda.get_device_name(0))

import cv2, rerun  # noqa: F401
from sam3.model_builder import build_sam3_image_model  # noqa: F401
from transformers import Sam3Model, Sam3Processor  # noqa: F401
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
from vggt.models.vggt import VGGT  # noqa: F401
from vggt.utils.geometry import depth_to_world_coords_points  # noqa: F401
print("imports ok: sam3, transformers, vggt, cv2, rerun")

import os
snapshot = os.path.join(os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
                        "models--facebook--sam3", "snapshots", "3c879f39826c281e95690f02c7821c4de09afae7")
if os.path.isfile(os.path.join(snapshot, "model.safetensors")):
    print("SAM 3: cached transformers snapshot found, no Hugging Face login needed (sam3_infer.py --backend transformers)")
else:
    try:
        from huggingface_hub import auth_check
        auth_check("facebook/sam3")
        print("SAM 3: no cached snapshot, but access to facebook/sam3 ok")
    except Exception as error:
        raise SystemExit(f"SAM 3: no cached snapshot and no access to facebook/sam3 ({type(error).__name__}). Accept the "
                         "license at https://huggingface.co/facebook/sam3 and log in with 'hf auth login'")
EOF
touch .venv-ov4d/.setup_ok
echo "setup done"

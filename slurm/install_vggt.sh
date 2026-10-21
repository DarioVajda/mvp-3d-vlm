#!/bin/bash
# Install the VGGT reference implementation (cloned into third_party/vggt) into .venv; run inside a job.
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m pip install einops
.venv/bin/python -m pip install --no-deps -e third_party/vggt
.venv/bin/python -c "import vggt, einops; print(vggt.__file__)"

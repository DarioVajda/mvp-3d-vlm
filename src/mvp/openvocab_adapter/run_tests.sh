#!/usr/bin/env bash
# Runs the adapter's CPU tests (tests/test_openvocab_adapter.py). Works from any folder, on Windows (Git Bash) and Linux.
#
#   bash src/mvp/openvocab_adapter/run_tests.sh              # every test
#   bash src/mvp/openvocab_adapter/run_tests.sh -k GoldSheet # extra arguments go to unittest
#   PYTHON=.venv-ov4d/bin/python bash src/mvp/openvocab_adapter/run_tests.sh
#
# It uses the repository's .venv unless PYTHON is set. The tests start the pipeline scripts with the same Python, so
# this has to be the one with the packages (on Windows, `py` is usually another one). It first checks the packages:
# it stops if numpy, Pillow or torch is missing, and says which tests will be skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

if [[ -z "${PYTHON:-}" ]]; then
    for candidate in "$REPO/.venv/Scripts/python.exe" "$REPO/.venv/bin/python"; do
        if [[ -x "$candidate" ]]; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "${PYTHON:-}" ]]; then
    echo "[tests] no Python in $REPO/.venv: create it with 'uv venv', or set PYTHON=/path/to/python" >&2
    exit 1
fi

cd "$REPO"

"$PYTHON" - <<'EOF'
import importlib.util
import sys

print(f"[tests] Python {sys.version.split()[0]}: {sys.executable}")
needed = {"numpy": "numpy", "PIL": "Pillow", "torch": "torch"}  # module: package
missing = [package for module, package in needed.items() if importlib.util.find_spec(module) is None]
if missing:
    sys.exit(f"[tests] the tests need packages that are not installed: {', '.join(missing)}. From the repository root:\n"
             f"  uv pip install --python {sys.executable} -r requirements.txt")

import torch

gpu = torch.cuda.is_available()
print(f"[tests] torch {torch.__version__}, " + ("CUDA GPU found: the 2 tests of what happens without one will be skipped"
                                                if gpu else "no CUDA GPU"))
if not (importlib.util.find_spec("spacy") and importlib.util.find_spec("nltk")):
    print("[tests] spaCy or NLTK not installed: the QuestionPrompts tests will be skipped (see requirements.txt)")
if importlib.util.find_spec("spacy") and not importlib.util.find_spec("en_core_web_md"):
    print("[tests] spaCy's en_core_web_md model not installed: the QuestionPrompts tests will fail (see requirements.txt)")
EOF

exec "$PYTHON" -m unittest discover -s src/mvp/openvocab_adapter/tests -v "$@"

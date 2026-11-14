"""Explainability analyses of the fusion arms (saliency, attention, J-lens, object words, attention-swap ablation).

ROOT is the repository root; scripts read results/ and write pages relative to it, so they run from any directory.
JLENS is a clone of github.com/anthropics/jacobian-lens at the pinned commit (scripts/fetch_jacobian_lens.sh); set
JLENS_DIR to use another location.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JLENS = Path(os.environ.get("JLENS_DIR", ROOT / "third_party" / "jacobian-lens"))

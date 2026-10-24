"""Table of bench.py runs: speed-ups of the kernel runs over the PyTorch reference, and how far their logits are apart.

  python experiments/linear_attn_kernels/compare.py results/_scratch/linear_attn_kernels
"""
import json
import sys
from pathlib import Path

import numpy as np

KEYS = ["prefill_bs1_s", "prefill_bs8_s", "generate256_bs1_s", "train_step_bs1_s"]
root = Path(sys.argv[1])
print("| GPU | run | " + " | ".join(KEYS) + " | train mem GB | max abs logit diff | top-1 agree | greedy16 |")
print("|---" * (len(KEYS) + 6) + "|")
for gpu in sorted(p for p in root.iterdir() if p.is_dir()):
    runs = {n: json.loads((gpu / f"{n}.json").read_text()) for n in ["torch", "kernels", "kernels_conv"]
            if (gpu / f"{n}.json").exists()}
    ref = runs.get("torch")
    ref_logits = np.load(gpu / "torch_logits.npy") if ref else None
    for name, r in runs.items():
        cells = []
        for k in KEYS:
            s = f"{r[k]:.2f}"
            if ref and name != "torch":
                s += f" ({ref[k] / r[k]:.1f}x)"
            cells.append(s)
        diff = agree = ""
        if ref and name != "torch":
            lg = np.load(gpu / f"{name}_logits.npy")
            diff = f"{np.abs(lg - ref_logits).max():.3f}"
            agree = f"{(lg.argmax(-1) == ref_logits.argmax(-1)).mean():.3f}"
        print(f"| {gpu.name} | {name} | " + " | ".join(cells) +
              f" | {r['train_peak_mem_gb']:.1f} | {diff} | {agree} | `{r['greedy16']}` |")

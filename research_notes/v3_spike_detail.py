"""Detail on the two anomaly types found: warmup gradient spikes, and single-step injection-ratio jumps (which data?)."""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/shared/workspace/povejmo/math/projects/mvp/src")
R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
RUNS = [f"{s}_v3_{a}" for s in ("2B", "4B") for a in ("lora", "additive", "xattn")]
steps = {r: [json.loads(l) for l in open(f"{R}/{r}/log.jsonl")] for r in RUNS}
steps = {r: [x for x in v if "loss" in x] for r, v in steps.items()}
SPE = 931

print("B1-detail. gradient norm (pre-clip) in the first 15 steps, and spike counts by step range")
print(f"{'run':16}" + "".join(f"{s:>7}" for s in range(1, 16)))
for r in RUNS:
    g = {x["step"]: x["grad_norm"] for x in steps[r]}
    print(f"{r:16}" + "".join(f"{g[s]:7.0f}" for s in range(1, 16)))
for r in RUNS:
    g = [(x["step"], x["grad_norm"]) for x in steps[r]]
    rng = {"1-12": 0, "13-100": 0, "101-559": 0, "560+": 0}
    for s, v in g:
        if v > 50:
            rng["1-12" if s <= 12 else "13-100" if s <= 100 else "101-559" if s <= 559 else "560+"] += 1
    last = max((s for s, v in g if v > 50), default=0)
    med_after = np.median([v for s, v in g if s > 559])
    print(f"{r:16} spikes by range {rng}; last spike at step {last}; median gnorm after warmup {med_after:.2f}")

print("\nB3-detail. log rows at the injection-jump steps (per-point inject, questions, tokens, loss)")
JUMPS = [4164, 4165, 4166, 4167, 4892, 4893, 4894, 5557, 5558, 5559, 5560]
for r in ("2B_v3_additive", "4B_v3_additive", "2B_v3_xattn", "4B_v3_xattn"):
    g = {x["step"]: x for x in steps[r]}
    print(r)
    for s in JUMPS:
        x = g[s]
        print(f"   step {s} (ep {s // SPE + 1}, pos {(s - 1) % SPE}): inject {dict((k, round(v, 3)) for k, v in x['inject'].items())} q {x['questions']} tok {x['tokens']} loss {x['loss']:.3f} gnorm {x['grad_norm']:.1f}")

print("\nB6. micro-batches at those steps (same seed => same order for every arm of a size)")
try:
    from mvp.train.data import epoch_plan, load_vlm3r, rank_steps
    samples, _ = load_vlm3r(6000, 0, 60, 100)
    cache = {}
    for s in (4165, 4166, 4893, 5558, 5559):
        e, i = (s - 1) // SPE, (s - 1) % SPE
        if e not in cache:
            cache[e] = rank_steps(epoch_plan(samples, e, 0, 8), 4, 1, 0)
        mbs, n_q = cache[e][i]
        for mb in mbs:
            qs = Counter(x["task"] for x in mb)
            print(f"   step {s}: video {mb[0]['video']}  {len(mb)} q  tasks {dict(qs)}  example: {mb[0]['question'][:90]!r}")
except Exception as ex:
    print("   could not reconstruct:", type(ex).__name__, ex)

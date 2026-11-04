"""Paired bootstrap of VSI overall (mean of per-task means) between v3 arms on the full benchmark, read-only."""
import json
from collections import defaultdict

import numpy as np

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
rng = np.random.default_rng(0)


def recs(run):
    return {r["id"]: r for r in map(json.loads, open(f"{R}/{run}/eval/best/vsi_f32_full.jsonl"))}


def overall(rs, ids):
    d = defaultdict(list)
    for i in ids:
        d[rs[i]["task"]].append(rs[i]["score"])
    return 100 * np.mean([np.mean(v) for v in d.values()])


def boot(a, b):
    ra, rb = recs(a), recs(b)
    ids = sorted(set(ra) & set(rb))
    by_task = defaultdict(list)
    for i in ids:
        by_task[ra[i]["task"]].append(i)
    obs = overall(rb, ids) - overall(ra, ids)
    bs = []
    for _ in range(1000):
        ii = [i for t in by_task.values() for i in rng.choice(t, len(t))]
        bs.append(overall(rb, ii) - overall(ra, ii))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{obs:+5.1f} [{lo:+.1f}, {hi:+.1f}]"


for size in ["2B", "4B"]:
    L, A, X = (f"{size}_v3_{a}" for a in ("lora", "additive", "xattn"))
    print(size, "additive-lora", boot(L, A), "| xattn-lora", boot(L, X), "| xattn-additive", boot(A, X))
print("4B lora - 2B lora", boot("2B_v3_lora", "4B_v3_lora"), "| 4B additive - 2B additive", boot("2B_v3_additive", "4B_v3_additive"))
# zero-shot comparison on the 1,000-question subset ids is not possible on full; report task counts instead
rs = recs("2B_v3_lora")
print("questions per task:", {t: n for t, n in sorted(__import__("collections").Counter(r["task"] for r in rs.values()).items())})

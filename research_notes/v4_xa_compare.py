"""v4 + aligned cross-attention runs: full per-task table and paired bootstraps against v4 additive (fusion lr 1e-5) and v4 LoRA."""
import json
import os
from collections import defaultdict

import numpy as np

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
RUNS = ["2B_v4_lora", "2B_v4_additive_nm1e-5", "2B_v4_additive_nm3e-5", "2B_v4_xattn2L_nm1e-5", "2B_v4_xattn8L_nm1e-5",
        "2B_xa_none", "2B_xa_rope", "2B_xa_bias", "2B_xa_both"]
TASKS = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance",
         "object_rel_direction", "route_planning", "obj_appearance_order"]
SHORT = ["count", "absdist", "size", "room", "reldist", "reldir", "route", "appear"]


def recs(run):
    p = f"{R}/{run}/eval/best/vsi_f32_full.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else None


def overall(rs, ids):
    d = defaultdict(list)
    for i in ids:
        d[rs[i]["task"]].append(rs[i]["score"])
    return 100 * np.mean([np.mean(v) for v in d.values()])


data = {r: recs(r) for r in RUNS}
print(f"{'run':24}{'overall':>8}" + "".join(f"{s:>9}" for s in SHORT) + f"{'best ep':>9}{'dev':>7}")
for r in RUNS:
    rs = data[r]
    if rs is None:
        print(f"{r:24}   (eval not finished)")
        continue
    per = defaultdict(list)
    for x in rs.values():
        per[x["task"]].append(x["score"])
    b = json.load(open(f"{R}/{r}/best_epoch.json"))
    print(f"{r:24}{overall(rs, list(rs)):8.1f}" + "".join(f"{100 * np.mean(per[t]):9.1f}" for t in TASKS) + f"{b['epoch']:>9}{b['dev_acc']:7.1f}")

rng = np.random.default_rng(0)


def boot(a, b):
    ra, rb = data[a], data[b]
    ids = sorted(set(ra) & set(rb))
    by = defaultdict(list)
    for i in ids:
        by[ra[i]["task"]].append(i)
    obs = overall(rb, ids) - overall(ra, ids)
    bs = []
    for _ in range(1000):
        ii = [i for t in by.values() for i in rng.choice(t, len(t))]
        bs.append(overall(rb, ii) - overall(ra, ii))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{obs:+5.1f} [{lo:+.1f}, {hi:+.1f}]"


print()
for r in RUNS[2:]:
    if data[r] is None:
        continue
    print(f"{r:24} vs additive 1e-5: {boot('2B_v4_additive_nm1e-5', r)}   vs LoRA: {boot('2B_v4_lora', r)}")
print(f"{'2B_v4_additive_nm1e-5':24} vs LoRA: {boot('2B_v4_lora', '2B_v4_additive_nm1e-5')}")
print(f"{'xa_both vs xa_none':24} {boot('2B_xa_none', '2B_xa_both') if data['2B_xa_none'] else 'pending'}")
print(f"{'xa_both vs xa_bias':24} {boot('2B_xa_bias', '2B_xa_both')}")
print(f"{'xa_both vs xa_rope':24} {boot('2B_xa_rope', '2B_xa_both')}")

# learned lambda / mu and attention stats if logged
print()
for r in ("2B_xa_bias", "2B_xa_both", "2B_xa_rope", "2B_xa_none"):
    rows = [json.loads(l) for l in open(f"{R}/{r}/log.jsonl")]
    keys = sorted({k for x in rows for k in x if any(s in k for s in ("lam", "mu", "self_mass", "frame_mass", "entropy", "bias"))})
    dv = [x for x in rows if "dev_acc" in x]
    st = [x for x in rows if "loss" in x]
    print(r, "logged keys:", keys[:30])
    for x in dv[-1:] + st[-1:]:
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items() if k in keys or k in ("step", "inject")})

"""Dump v4 + aligned-xattn run data (dev curves, per-task full VSI, attention stats, bias scalars, injection, loss) as JSON for charts."""
import json
import os
from collections import defaultdict

import numpy as np

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
RUNS = {"lora": "2B_v4_lora", "add1e5": "2B_v4_additive_nm1e-5", "add3e5": "2B_v4_additive_nm3e-5",
        "x2L": "2B_v4_xattn2L_nm1e-5", "x8L": "2B_v4_xattn8L_nm1e-5",
        "xa_none": "2B_xa_none", "xa_rope": "2B_xa_rope", "xa_bias": "2B_xa_bias", "xa_both": "2B_xa_both"}
TASKS = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation", "object_rel_distance",
         "object_rel_direction", "route_planning", "obj_appearance_order"]
out = {}
for key, run in RUNS.items():
    d = f"{R}/{run}"
    rows = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    st = [x for x in rows if "loss" in x]
    spe = 7625 / 5
    dv = [x for x in rows if "dev_acc" in x]
    rec = {"name": run, "steps_per_epoch": spe, "total": st[-1]["step"]}
    rec["dev_acc"] = [[round(x["step"] / spe, 2), round(x["dev_acc"], 2)] for x in dv]
    rec["dev_loss"] = [[round(x["step"] / spe, 3), round(x["dev_loss"], 4)] for x in rows if "dev_loss" in x]
    rec["dev_tasks"] = {k[8:]: [[round(x["step"] / spe, 2), round(x[k], 1)] for x in dv if k in x] for k in dv[-1] if k.startswith("dev_acc/")}
    tl = [x["loss"] for x in st]
    rec["train_loss"] = [[round(st[i]["step"] / spe, 3), round(float(np.mean(tl[max(0, i - 49):i + 1])), 4)] for i in range(0, len(st), 25)]
    if st[-1].get("inject"):
        rec["inject"] = [[round(st[i]["step"] / spe, 3)] + [round(float(np.mean([r["inject"][k] for r in st[max(0, i - 49):i + 1]])), 3) for k in ("0", "1", "2")] for i in range(0, len(st), 25)]
    if st[-1].get("bias"):
        rec["bias"] = [[round(st[i]["step"] / spe, 3)] + [round(st[i]["bias"][k][w], 3) for k in ("0", "1", "2") for w in ("lam", "mu")] for i in range(0, len(st), 25)]
    attn_keys = [k for k in dv[-1] if k.startswith("attn/")]
    if attn_keys:
        rec["attn"] = {k[5:]: [[round(x["step"] / spe, 2), round(x[k], 3)] for x in dv if k in x] for k in attn_keys}
    g = [x["grad_norm"] for x in st]
    rec["grad_spikes"] = int(sum(v > 50 for v in g))
    b = json.load(open(f"{d}/best_epoch.json"))
    rec["best_epoch"] = b["epoch"]; rec["best_dev"] = round(b["dev_acc"], 1)
    p = f"{d}/eval/best/vsi_f32_full.jsonl"
    if os.path.exists(p):
        rs = [json.loads(l) for l in open(p)]
        per = defaultdict(list)
        for x in rs:
            per[x["task"]].append(x["score"])
        rec["vsi"] = {t: round(100 * float(np.mean(per[t])), 1) for t in TASKS}
        rec["vsi"]["overall"] = round(float(np.mean([100 * np.mean(per[t]) for t in TASKS])), 1)
        # per source dataset
        ds = defaultdict(lambda: defaultdict(list))
        for x in rs:
            ds[x["dataset"]][x["task"]].append(x["score"])
        rec["vsi_by_dataset"] = {k: round(100 * float(np.mean([np.mean(v) for v in t.values()])), 1) for k, t in ds.items()}
    out[key] = rec
json.dump(out, open("/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/v4_data.json", "w"))
for k, v in out.items():
    print(k, v["name"], "best", v["best_epoch"], v["best_dev"], "vsi", v.get("vsi", {}).get("overall"), "by ds", v.get("vsi_by_dataset"), "spikes", v["grad_spikes"], "attn keys", list(v.get("attn", {}).keys())[:3])
print("dev task keys:", list(out["lora"]["dev_tasks"].keys()))
print("attn self_mass trajectory xa_bias:", out["xa_bias"]["attn"]["0_self_mass"])
print("lam trajectory xa_both:", out["xa_both"]["bias"][::20])

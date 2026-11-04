"""Read-only status of the v3 runs in the main checkout: dev curves, best epoch, full-VSI result."""
import json
import os

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
for r in ["2B_v3_lora", "2B_v3_additive", "2B_v3_xattn", "4B_v3_lora", "4B_v3_additive", "4B_v3_xattn"]:
    d = f"{R}/{r}"
    print("==", r)
    if os.path.exists(f"{d}/best_epoch.json"):
        b = json.load(open(f"{d}/best_epoch.json"))
        print(" best epoch", b["epoch"], "dev_acc %.1f" % b["dev_acc"])
        print(" dev by epoch:", [(e["epoch"], round(e["dev_acc"], 1), round(e["dev_loss"], 3)) for e in b["epochs"]])
    elif os.path.exists(f"{d}/log.jsonl"):
        rows = [json.loads(l) for l in open(f"{d}/log.jsonl")]
        st = [x for x in rows if "loss" in x]
        dv = [x for x in rows if "dev_acc" in x]
        if st:
            print(" running: step", st[-1]["step"], "loss %.3f" % st[-1]["loss"], "inject", {k: round(v, 2) for k, v in st[-1].get("inject", {}).items()})
        print(" dev so far:", [(x["epoch"] + (1 if x["step"] else 0), round(x["dev_acc"], 1)) for x in dv])
    p = f"{d}/eval/best/vsi_f32_full_summary.json"
    if os.path.exists(p):
        s = json.load(open(p))
        print(" FULL VSI f32: n", s["n"], "overall %.1f" % s["overall"])
        print(" ", {k[:14]: round(v, 1) for k, v in s.items() if isinstance(v, float) and k != "overall"})
    if os.path.exists(f"{d}/log.jsonl"):
        rows = [json.loads(l) for l in open(f"{d}/log.jsonl")]
        st = [x for x in rows if "loss" in x]
        if st and "inject" in st[-1] and st[-1]["inject"]:
            eps = sorted({x["epoch"] for x in st})
            print(" inject by epoch end:", [{k: round(v, 2) for k, v in [x for x in st if x["epoch"] == e][-1]["inject"].items()} for e in eps])
            print(" train loss by epoch:", [round(sum(x["loss"] for x in st if x["epoch"] == e) / len([x for x in st if x["epoch"] == e]), 3) for e in eps])
        elif st:
            eps = sorted({x["epoch"] for x in st})
            print(" train loss by epoch:", [round(sum(x["loss"] for x in st if x["epoch"] == e) / len([x for x in st if x["epoch"] == e]), 3) for e in eps])

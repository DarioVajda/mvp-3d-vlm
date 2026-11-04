"""Dump v3 training curves (train loss, dev loss, dev accuracy, injection ratio, grad norm) as compact JSON for the notebook charts."""
import json

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
out = {}
for size in ["2B", "4B"]:
    for arm in ["lora", "additive", "xattn"]:
        run = f"{size}_v3_{arm}"
        rows = [json.loads(l) for l in open(f"{R}/{run}/log.jsonl")]
        st = [r for r in rows if "loss" in r]
        spe = max(r["step"] for r in st) // 6
        # train loss: 50-step moving average, sampled every 25 steps
        loss, tl = [], [r["loss"] for r in st]
        for i in range(0, len(st), 25):
            w = tl[max(0, i - 49):i + 1]
            loss.append([st[i]["step"], round(sum(w) / len(w), 4)])
        dev_loss = [[r["step"], round(r["dev_loss"], 4)] for r in rows if "dev_loss" in r]
        dev_acc = [[r["step"], round(r["dev_acc"], 2)] for r in rows if "dev_acc" in r]
        dev_task = {}
        for r in rows:
            if "dev_acc" in r:
                for k, v in r.items():
                    if k.startswith("dev_acc/"):
                        dev_task.setdefault(k[8:], []).append([r["step"], round(v, 1)])
        inject = []
        if st[-1].get("inject"):
            for i in range(0, len(st), 25):
                w = st[max(0, i - 49):i + 1]
                inject.append([st[i]["step"]] + [round(sum(r["inject"][k] for r in w) / len(w), 3) for k in ("0", "1", "2")])
        g = [r["grad_norm"] for r in st]
        out[run] = {"steps_per_epoch": spe, "train_loss": loss, "dev_loss": dev_loss, "dev_acc": dev_acc, "dev_task": dev_task,
                    "inject": inject, "grad_spikes": sum(x > 50 for x in g), "grad_max": round(max(g), 1),
                    "epoch_loss": [round(sum(r["loss"] for r in st if r["epoch"] == e) / len([r for r in st if r["epoch"] == e]), 4) for e in range(6)]}
json.dump(out, open("/shared/workspace/povejmo/math/projects/mvp/.claude/worktrees/je-xattn-proposal/research_notes/v3_curves.json", "w"))
for k, v in out.items():
    print(k, "spe", v["steps_per_epoch"], "n_loss", len(v["train_loss"]), "n_devloss", len(v["dev_loss"]), "dev_acc", [a for _, a in v["dev_acc"]], "spikes", v["grad_spikes"], v["grad_max"])
print(json.dumps(out)[:300])

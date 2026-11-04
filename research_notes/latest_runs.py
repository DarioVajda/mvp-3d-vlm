"""Read-only overview of every training run newer than v3: status, dev curve, best checkpoint, full-VSI result."""
import glob
import json
import os
import time

R = "/shared/workspace/povejmo/math/projects/mvp/results/train"
runs = sorted((d for d in glob.glob(f"{R}/*") if os.path.isdir(d)), key=os.path.getmtime)
for d in runs:
    name = os.path.basename(d)
    log = f"{d}/log.jsonl"
    if not os.path.exists(log):
        continue
    mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(log)))
    rows = [json.loads(l) for l in open(log)]
    st = [x for x in rows if "loss" in x]
    dv = [x for x in rows if "dev_acc" in x and x["step"] > 0]
    args = json.load(open(f"{d}/args.json")) if os.path.exists(f"{d}/args.json") else {}
    keys = ("arm", "epochs", "cap", "lr", "fusion_lr", "weight_decay", "vggt_layers", "fusion_after", "seed")
    print(f"== {name}  (log updated {mt})  " + " ".join(f"{k}={args.get(k)}" for k in keys if k in args))
    extra = {k: v for k, v in args.items() if k not in keys and k not in ("model", "out", "wandb_entity", "wandb_project", "wandb_name", "no_wandb", "git_commit", "host", "gpu", "workers", "world_size")}
    if extra:
        print("   other args:", {k: v for k, v in extra.items() if v not in (None, False)})
    if st:
        print(f"   last step {st[-1]['step']}  epoch {st[-1]['epoch']}  loss {st[-1]['loss']:.3f}")
    if dv:
        print("   dev:", [(round(x.get("epoch_frac", x["epoch"] + 1), 2), round(x["dev_acc"], 1)) for x in dv])
    b = f"{d}/best_epoch.json"
    if os.path.exists(b):
        bj = json.load(open(b))
        print(f"   best: epoch {bj['epoch']} dev_acc {bj['dev_acc']:.1f}")
    for p in sorted(glob.glob(f"{d}/eval/best/*summary.json")):
        s = json.load(open(p))
        tasks = {k[:12]: round(v, 1) for k, v in s.items() if isinstance(v, float) and k not in ("overall", "parse_rate")}
        print(f"   {os.path.basename(p)}: n {s.get('n')} overall {s['overall']:.1f}  {tasks}")

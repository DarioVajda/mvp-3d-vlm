"""Gather a wsd trunk's decay branches (mvp.train.run --schedule wsd) into one table and into the trunk's W&B run.

  python -m mvp.train.wsd_collect results/train/<trunk> [--no-wandb]

Per branch <trunk>/branches/step<N>/: dev accuracy after the decay (its log.jsonl), and full VSI-Bench at 32 frames
(eval/vsi_f32_full.jsonl: overall, per task and on the debiased subset). Next to it, the trunk's own dev accuracy at
step N, before the decay. Writes <trunk>/branches.json. Then reopens the trunk's W&B run (wandb_run.json; the trunk
has finished, so nothing else writes to it) and logs one row per branch under decay/, against the x-axis
decay/trunk_step = N: each decay/* metric becomes one curve over the branch points in the trunk's own run.
Branches that have not finished are skipped and listed.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from mvp.benchmarks.vsibench import load_vsibench, summarize


def dev_row(log, step=None, key="dev_acc"):
    rows = [json.loads(l) for l in open(log)] if log.exists() else []
    rows = [r for r in rows if key in r and (step is None or r["step"] == step)]
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trunk")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    trunk = Path(args.trunk)
    steps = json.loads((trunk / "args.json").read_text())["branch_steps"]
    debiased = set(load_vsibench("debiased")["id"])
    table, missing = [], []
    for n in steps:
        b = trunk / "branches" / f"step{n}"
        after, before = dev_row(b / "log.jsonl"), dev_row(trunk / "log.jsonl", n)
        pred = b / "eval" / "vsi_f32_full.jsonl"
        if after is None or not pred.exists() or before is None:
            missing.append(n)
            continue
        df = pd.DataFrame([json.loads(l) for l in open(pred)])
        if len(df) != 5130:
            print(f"step {n}: VSI eval has {len(df)} of 5130 questions (a subset run?)")
        full, deb = summarize(df), summarize(df[df["id"].isin(debiased)])
        table.append({"trunk_step": n, "decayed_to": after["step"], "dev_acc_before": before["dev_acc"],
                      "dev_acc": after["dev_acc"], "dev_acc_all": after.get("dev_acc/all"), "dev_loss": after["dev_loss"],
                      "vsi_n": len(df), "vsi_overall": full["overall"], "vsi_debiased": deb["overall"],
                      **{f"vsi/{t}": v for t, v in full.items() if t != "overall"},
                      **{f"dev/{k[8:]}": v for k, v in after.items() if k.startswith("dev_acc/") and k not in ("dev_acc/mean", "dev_acc/all")}})
    (trunk / "branches.json").write_text(json.dumps({"branches": table, "missing": missing}, indent=1))
    print(f"{len(table)} branches collected, missing {missing}")
    for r in table:
        print(f"  step {r['trunk_step']}: dev {r['dev_acc_before']:.1f} -> {r['dev_acc']:.1f}, VSI {r['vsi_overall']:.1f}, "
              f"debiased {r['vsi_debiased']:.1f}")
    if args.no_wandb or not table:
        return
    import wandb
    w = json.loads((trunk / "wandb_run.json").read_text())
    run = wandb.init(id=w["id"], entity=w["entity"], project=w["project"], resume="must", dir=str(trunk))
    run.define_metric("decay/trunk_step")
    run.define_metric("decay/*", step_metric="decay/trunk_step")
    for r in table:
        run.log({f"decay/{k}": v for k, v in r.items() if v is not None and k != "decayed_to"})
    run.summary["decay/table"] = wandb.Table(dataframe=pd.DataFrame(table))
    run.finish()


if __name__ == "__main__":
    main()

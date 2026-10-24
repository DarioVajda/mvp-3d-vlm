"""Per-task results tables for the main zero-shot VSI-Bench / MMSI-Bench runs in results/zeroshot/ (shards merged).

  python -m mvp.reporting.tables            # prints markdown tables, writes results/zeroshot/summary.json
"""
import json
import re
from collections import defaultdict

import pandas as pd

from mvp.benchmarks.vsibench import TASKS, load_vsibench, summarize
from mvp.paths import ZEROSHOT

# Run folders summarised as "vsi/<run>" and "mmsi/<run>" in summary.json
VSI_RUNS = ["vsibench/qwen_sampling", "vsibench/spatial_mllm_sampling", "vsibench/qwen3_5"]
MMSI_RUNS = ["mmsibench"]
# Qwen3-VL technical report (arXiv 2511.21631), Table 4: VSI-Bench overall, instruct models
REPORTED = {"2B": 53.9, "4B": 59.3}


def to_md(t):
    t = t.round(1).astype(object).where(t.notna(), "")
    lines = ["| " + " | ".join([t.index.name or ""] + [str(c) for c in t.columns]) + " |",
             "|" + "---|" * (len(t.columns) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(t.index, t.values)]
    return "\n".join(lines)


def load_runs(subs):
    runs = defaultdict(list)
    for p in sorted((p for s in subs for p in (ZEROSHOT / s).glob("*.jsonl")), key=lambda p: p.name):
        runs[re.sub(r"_s\d+$", "", p.stem)].append(pd.read_json(p, lines=True))
    return {k: pd.concat(v).drop_duplicates("id") for k, v in runs.items()}


def main():
    out = {}
    n_total = len(load_vsibench("full"))
    debiased = set(load_vsibench("debiased")["id"])
    rows = []
    for name, df in load_runs(VSI_RUNS).items():
        s = summarize(df)
        sd = summarize(df[df["id"].isin(debiased)])
        rows.append({"run": name, "n": len(df), **{t: s.get(t) for t in TASKS}, "overall": s["overall"],
                     "debiased": sd["overall"], "reported": REPORTED.get(name.split("_")[0])})
        out[f"vsi/{name}"] = {"n": len(df), "complete": len(df) == n_total, **s, "debiased_overall": sd["overall"]}
    if rows:
        t = pd.DataFrame(rows).set_index("run")
        print("## VSI-Bench (full, 5130 qs; MCA accuracy / NA MRA, macro overall as in lmms-eval)\n")
        print(to_md(t), "\n")

    rows = []
    for name, df in load_runs(MMSI_RUNS).items():
        per = (100 * df.groupby("question_type")["score"].mean()).to_dict()
        rows.append({"run": name, "n": len(df), "overall": 100 * df["score"].mean(),
                     "parse_rate": 100 * df["choice"].notna().mean(), **per})
        out[f"mmsi/{name}"] = rows[-1]
    if rows:
        print("## MMSI-Bench (1000 qs, accuracy)\n")
        print(to_md(pd.DataFrame(rows).set_index("run").T))
    json.dump(out, open(ZEROSHOT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

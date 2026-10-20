"""Compare several VSI-Bench runs on the questions they share (e.g. 8 / 32 frames vs 2 fps on a subset).

  python src/compare_runs.py results/vsi/4B_f8.jsonl results/vsi/4B_f32.jsonl results/vsi_subsets/4B_fps2_fail50.jsonl
  ... --show         # also print every question with all predictions
"""
import argparse
from pathlib import Path

import pandas as pd

from vsibench import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    runs = {Path(p).stem: pd.read_json(p, lines=True).set_index("id") for p in args.runs}
    ids = sorted(set.intersection(*[set(r.index) for r in runs.values()]))
    base = next(iter(runs.values())).loc[ids]
    score = pd.DataFrame({k: r.loc[ids, "score"] for k, r in runs.items()})
    score["task"] = base["task"]
    print(f"{len(ids)} shared questions; mean score (x100) per task:")
    t = (100 * score.groupby("task").mean()).reindex([t for t in TASKS if t in set(score["task"])])
    t.loc["ALL"] = 100 * score.drop(columns="task").mean()
    t.insert(0, "n", score.groupby("task").size().reindex(t.index).fillna(len(ids)).astype(int))
    print(t.round(1).to_string())
    frames = {k: r.loc[ids, "n_frames"].describe()[["min", "50%", "max"]].tolist() for k, r in runs.items()}
    print("\nframes (min / median / max):", {k: [int(x) for x in v] for k, v in frames.items()})

    if args.show:
        for i in ids:
            d = base.loc[i]
            print(f"\n[{i}] {d['question_type']} {d['dataset']}/{d['scene_name']}  GT={d['ground_truth']}")
            print("   ", d["question"].strip().replace("\n", " ")[:300])
            if isinstance(d["options"], list):
                print("    options:", " | ".join(d["options"]))
            for k, r in runs.items():
                print(f"    {k:22s} {r.loc[i, 'score']:.2f}  {r.loc[i, 'prediction']!r}  ({r.loc[i, 'n_frames']} frames)")


if __name__ == "__main__":
    main()

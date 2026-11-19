"""Compare several VSI-Bench runs on the questions they share (e.g. 8 / 32 frames vs 2 fps on a subset).

  python -m mvp.reporting.compare_runs results/zeroshot/vsibench/qwen_sampling/4B_f8.jsonl \
      results/zeroshot/vsibench/qwen_sampling/4B_f32.jsonl results/zeroshot/vsibench/fail50_subset/4B_fps2_fail50.jsonl
  ... --show         # also print every question with all predictions

Besides the scores per task, it prints the VSI-Bench overall (macro average, as lmms-eval) and the parse rate of
each run, and compares every run with a reference run (--reference, default the first) question by question: how many
questions score higher and lower, with a two-sided sign test and Holm's correction over the comparisons.
"""
import argparse
from math import comb
from pathlib import Path

import pandas as pd

from mvp.benchmarks.vsibench import TASKS, parsed, summarize


def sign_test(better, worse):
    """Exact two-sided sign test: the chance of a split at least this uneven if better and worse were equally likely.
    Ties are left out, as usual."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(better, worse) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def holm(pvalues):
    """Holm's step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(reference, other):
    """(questions where other scores higher, lower, the same) than reference, matched by index."""
    diff = other - reference.reindex(other.index)
    return int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())


def parse_rate(run):
    """Share of answers the extraction rule could read (mvp.benchmarks.vsibench.parsed): the final-answer rule for a
    run made with --final-answer (it has score_first), else lmms-eval's."""
    final = "score_first" in run
    return sum(parsed(row, row["prediction"], final) for _, row in run.iterrows()) / len(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reference", help="run to compare the others with (its path or file stem); default: the first")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    runs = {Path(p).stem: pd.read_json(p, lines=True).set_index("id") for p in args.runs}
    ids = sorted(set.intersection(*[set(r.index) for r in runs.values()]))
    runs = {k: r.loc[ids] for k, r in runs.items()}
    base = next(iter(runs.values()))
    score = pd.DataFrame({k: r["score"] for k, r in runs.items()})
    score["task"] = base["task"]
    print(f"{len(ids)} shared questions; mean score (x100) per task:")
    t = (100 * score.groupby("task").mean()).reindex([t for t in TASKS if t in set(score["task"])])
    t.loc["ALL"] = 100 * score.drop(columns="task").mean()
    t.insert(0, "n", score.groupby("task").size().reindex(t.index).fillna(len(ids)).astype(int))
    print(t.round(1).to_string())
    frames = {k: r["n_frames"].describe()[["min", "50%", "max"]].tolist() for k, r in runs.items()}
    print("\nframes (min / median / max):", {k: [int(x) for x in v] for k, v in frames.items()})

    print("\nVSI-Bench overall (macro over tasks) and parse rate:")
    for k, r in runs.items():
        print(f"  {k:30s} overall {summarize(r.reset_index())['overall']:.1f}  parsed {100 * parse_rate(r.reset_index()):.1f}%")

    ref = Path(args.reference).stem if args.reference else next(iter(runs))
    others = [k for k in runs if k != ref]
    if others:
        counts = [paired(runs[ref]["score"], runs[k]["score"]) for k in others]
        p = [sign_test(b, w) for b, w, _ in counts]
        print(f"\nagainst {ref}, question by question (two-sided sign test; Holm over {len(others)} comparisons):")
        for k, (b, w, s), pk, ph in zip(others, counts, p, holm(p)):
            print(f"  {k:30s} {b} better / {w} worse / {s} same   p = {pk:.3g}   Holm p = {ph:.3g}")

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

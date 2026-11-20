"""The structured-CoT pilot's predictions analysed as the README's Metrics and analysis section specifies; writes
analysis.md next to this file.

  PYTHONPATH=src python experiments/2026-11-19-structured-cot/analysis.py [--root data/scene_sheets_75]

Reads results/zeroshot/vsibench/structured_cot/explore200/ (A_video = A, video_cot = B, gold_csv = C, gold_cot = D,
gold_relations = E) and, for the reasoning check, the gold sheets in --root (the ids and numbers D's prompt carried).
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mvp.benchmarks.scene_sheet import fmt_float, load_index, object_id
from mvp.benchmarks.vsibench import MCA_QUESTION_TYPES, parsed, summarize, task_of
from mvp.paths import ROOT, SCENE_SHEETS
from mvp.reporting.compare_runs import holm, paired, sign_test

HERE = Path(__file__).parent
RESULTS = ROOT / "results/zeroshot/vsibench/structured_cot/explore200"
CONDITIONS = {"A": ("A_video", "video only, direct"), "B": ("video_cot", "video only, structured CoT"),
              "C": ("gold_csv", "gold sheet, direct"), "D": ("gold_cot", "gold sheet, structured CoT"),
              "E": ("gold_relations", "gold sheet + relations, direct")}
COMPARISONS = [("D", "C", "does CoT help with a sheet"), ("B", "A", "does CoT help without a sheet"),
               ("E", "D", "do relations beat computing them in writing")]
COMPUTING = ["object_abs_distance", "object_rel_distance", "object_size_estimation"]
TYPES = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
         "object_rel_distance", "object_rel_direction_easy", "object_rel_direction_medium",
         "object_rel_direction_hard", "route_planning", "obj_appearance_order"]
B, SEED = 10_000, 0


def load():
    runs = {c: pd.read_json(RESULTS / f"{stem}.jsonl", lines=True).set_index("id").sort_index()
            for c, (stem, _) in CONDITIONS.items()}
    ids = list(runs["A"].index)
    assert all(list(r.index) == ids for r in runs.values()), "the conditions answered different questions"
    return runs


def scores(run):
    """VSI-Bench overall (8 tasks), overall without counting, the computing tasks' mean, per task (x100)."""
    s = summarize(run.reset_index())
    tasks = {k: v for k, v in s.items() if k != "overall"}
    return {"overall": s["overall"], "no_count": float(np.mean([v for k, v in tasks.items() if k != "object_counting"])),
            "computing": float(np.mean([tasks[k] for k in COMPUTING])), **tasks}


def bootstrap(runs):
    """Paired, stratified bootstrap: questions resampled within each question type, the same resample for every
    condition. {measure: {condition: array of B values}}."""
    qtypes = runs["A"]["question_type"]
    members = {t: np.flatnonzero(qtypes.to_numpy() == t) for t in sorted(set(qtypes))}
    rng = np.random.default_rng(SEED)
    S = {c: r["score"].to_numpy(float) for c, r in runs.items()}
    out = {m: {c: np.empty(B) for c in runs} for m in ("overall", "no_count", "computing")}
    for b in range(B):
        idx = {t: rng.choice(m, size=len(m)) for t, m in members.items()}
        for c in runs:
            per_type = {t: S[c][i].mean() for t, i in idx.items()}
            task = {}
            for t, v in per_type.items():
                task.setdefault(task_of(t), []).append(v)
            task = {k: 100 * float(np.mean(v)) for k, v in task.items()}
            out["overall"][c][b] = np.mean(list(task.values()))
            out["no_count"][c][b] = np.mean([v for k, v in task.items() if k != "object_counting"])
            out["computing"][c][b] = np.mean([task[k] for k in COMPUTING])
    return out


def interval(values):
    return f"[{np.percentile(values, 2.5):+.1f}, {np.percentile(values, 97.5):+.1f}]"


def sheet_ids_and_numbers(q, root):
    """The ids and the numbers (as the sheet writes them) of the gold sheet D's prompt carried for question q."""
    with open(Path(root) / load_index(root)[q]["gold_sheet"] / "objects.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = {object_id(r) for r in rows}
    numbers = {fmt_float(r[k]) for r in rows for k in ("x", "y", "z", "size_x", "size_y", "size_z")}
    return ids, {n for n in numbers if n not in ("0.000",)}


def reasoning_check(run, root):
    """For each of D's replies: does it name a sheet row, does it copy one of the sheet's numbers."""
    rows = []
    for q, r in run.iterrows():
        ids, numbers = sheet_ids_and_numbers(q, root)
        text = str(r["prediction"])
        if not ids:
            continue
        rows.append({"id": q, "type": r["question_type"], "names_row": any(re.search(rf"\b{re.escape(i)}\b", text)
                                                                           for i in ids),
                     "copies_number": any(n in text for n in numbers), "right": r["score"] == 1.0
                     if r["question_type"] in MCA_QUESTION_TYPES else r["score"] >= 0.5,
                     "score": r["score"], "truncated": bool(r["hit_token_limit"]), "tokens": int(r["n_new_tokens"])})
    return pd.DataFrame(rows).set_index("id")


def excerpt(text, head=700, tail=300):
    text = str(text)
    return text if len(text) <= head + tail + 20 else text[:head] + "\n[... " + f"{len(text) - head - tail:,}" + \
        " characters ...]\n" + text[-tail:]


def report(root):
    runs = load()
    n = len(runs["A"])
    s = {c: scores(r) for c, r in runs.items()}
    boot = bootstrap(runs)
    lines = ["# Structured CoT pilot: analysis", "",
             "<!-- Written by analysis.py in this folder; edit that, not this file. -->", "",
             f"The predictions of `explore_cot.ipynb` (`results/zeroshot/vsibench/structured_cot/explore200/`), "
             f"analysed as the README's Metrics and analysis section specifies: {n} questions (`explore200_ids.txt`, "
             "25 per task), Qwen3.5-2B, 16 frames (`uniform128`), 2,048 new tokens, final answer scored, gold sheets "
             "as csv, on one L4. Bootstrap intervals: questions resampled within each question type, the same "
             f"resample for every condition ({B:,} resamples, seed {SEED}).", "",
             "## Scores", "",
             "| | Condition | Overall | w/o counting | Computing tasks | " + " | ".join(
                 ["Count", "Abs. dist.", "Obj. size", "Room size", "Rel. dist.", "Rel. dir.", "Route", "Order"]) + " |",
             "|---|---|---:|---:|---:|" + "---:|" * 8]
    task_cols = ["object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
                 "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"]
    for c, (stem, what) in CONDITIONS.items():
        lines.append(f"| {c} | {what} | {s[c]['overall']:.1f} | {s[c]['no_count']:.1f} | {s[c]['computing']:.1f} | "
                     + " | ".join(f"{s[c][t]:.1f}" for t in task_cols) + " |")
    lines += ["", "Computing tasks: the mean of absolute distance, relative distance and object size. Per question type "
              "(x100, 25 questions each; relative direction 8, 8 and 9):", "",
              "| Question type | n | " + " | ".join(CONDITIONS) + " |", "|---|---:|" + "---:|" * len(CONDITIONS)]
    for t in TYPES:
        k = runs["A"]["question_type"] == t
        lines.append(f"| {t} | {int(k.sum())} | " + " | ".join(f"{100 * runs[c].loc[k, 'score'].mean():.1f}"
                                                               for c in CONDITIONS) + " |")

    # comparisons
    counts = [paired(runs[ref]["score"], runs[c]["score"]) for c, ref, _ in COMPARISONS]
    p = [sign_test(b, w) for b, w, _ in counts]
    hp = holm(p)
    lines += ["", "## The main comparisons", "",
              "Question by question: better / worse / same, an exact two-sided sign test, Holm over the three; and "
              "the difference in overall, without counting and on the computing tasks, with bootstrap 95% intervals.",
              "", "| Comparison | Question | Better / worse / same | p | Holm p | Overall | w/o counting | Computing |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (c, ref, what), (b, w, same), pc, ph in zip(COMPARISONS, counts, p, hp):
        cells = []
        for m in ("overall", "no_count", "computing"):
            d = boot[m][c] - boot[m][ref]
            cells.append(f"{s[c][m] - s[ref][m]:+.1f} {interval(d)}")
        lines.append(f"| {c} − {ref} | {what} | {b} / {w} / {same} | {pc:.3g} | {ph:.3g} | " + " | ".join(cells) + " |")

    # replies
    lines += ["", "## Replies", "",
              "Parse rate: the share of replies the final-answer rule reads (`parsed(..., final=True)`). Reply length "
              "in new tokens, as the eval records it.", "",
              "| | Condition | Parsed | Median length | 90th percentile | Hit the 2,048-token limit |",
              "|---|---|---:|---:|---:|---:|"]
    for c, (stem, what) in CONDITIONS.items():
        r = runs[c]
        ok = sum(parsed(row, row["prediction"], final=True) for _, row in r.reset_index().iterrows())
        tok = r["n_new_tokens"]
        lines.append(f"| {c} | {what} | {100 * ok / len(r):.1f}% | {tok.median():.0f} | {tok.quantile(0.9):.0f} | "
                     f"{int(r['hit_token_limit'].sum())} ({100 * r['hit_token_limit'].mean():.0f}%) |")

    # reasoning check
    rc = reasoning_check(runs["D"], root)
    comp = rc[rc["type"].isin(COMPUTING)]
    lines += ["", "## Reasoning check (D)", "",
              f"Over D's {len(rc)} replies whose gold sheet has rows: a reply **names a row** if it contains one of the "
              "sheet's ids (such as `chair_2`), and **copies a number** if it contains one of the sheet's positions or "
              "extents exactly as the sheet writes them (three decimals).", "",
              "| Replies | n | Name a row | Copy a number | Right |", "|---|---:|---:|---:|---:|"]
    for name, df in [("all", rc), ("computing tasks", comp), ("computing tasks, not cut off", comp[~comp["truncated"]]),
                     ("cut off at the limit", rc[rc["truncated"]])]:
        lines.append(f"| {name} | {len(df)} | {100 * df['names_row'].mean():.0f}% | {100 * df['copies_number'].mean():.0f}% "
                     f"| {100 * df['right'].mean():.0f}% |")
    right_definition = ("right: the exact option for multiple choice, a score of at least 0.5 (within 25% of the "
                        "answer) for numbers")
    lines += ["", f"({right_definition}.)"]

    # examples
    D = runs["D"]
    use = comp[~comp["truncated"] & comp["copies_number"]]
    picks = [("right", use[use["right"]].index.min()), ("wrong", use[~use["right"]].index.min()),
             ("cut off", rc[rc["truncated"]].index.min())]
    lines += ["", "### Three of D's replies", "",
              "The first by question id of each kind: a right and a wrong computing-task reply that copies the sheet's "
              "numbers, and the first reply cut off at the limit (shortened in the middle)."]
    for kind, q in picks:
        if pd.isna(q):
            continue
        r = D.loc[q]
        lines += ["", f"**{kind}**, question {q} ({r['question_type']}): {r['question'].splitlines()[0]}  ",
                  f"ground truth {r['ground_truth']}; score {r['score']:.2f}; {int(r['n_new_tokens'])} tokens"
                  + ("; hit the limit" if r["hit_token_limit"] else ""), "", "```", excerpt(r["prediction"]), "```"]

    # which case
    d_c = s["D"]["overall"] - s["C"]["overall"]
    gap = s["E"]["overall"] - s["C"]["overall"]
    holm_dc = hp[0]
    if d_c > 0 and holm_dc < 0.05:
        case = ("CoT helps with the sheet" + (" and closes much of the gap to E" if d_c >= gap / 2 else
                                               " but leaves most of the gap to E"))
    else:
        case = "CoT doesn't help even with the sheet"
    lines += ["", "## Which of the README's next steps", "",
              f"D − C is {d_c:+.1f} overall (Holm p = {holm_dc:.2g}; interval "
              f"{interval(boot['overall']['D'] - boot['overall']['C'])}), against a gap from C to E of {gap:+.1f}. "
              f"The README's case is therefore **\"{case}\"**, whose next step is: precomputed relations are the way "
              "forward; stop prompting the 2B to compute from coordinates, and try a larger model only if relations "
              "alone are not enough." if case.startswith("CoT doesn't") else
              f"D − C is {d_c:+.1f} overall (Holm p = {holm_dc:.2g}); the README's case is \"{case}\"."]
    return "\n".join(lines) + "\n", dict(runs=runs, scores=s, rc=rc, p=p, holm=hp, counts=counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCENE_SHEETS), help="the sheets folder, for the gold sheets D carried")
    ap.add_argument("--out", default=str(HERE / "analysis.md"))
    args = ap.parse_args()
    text, _ = report(Path(args.root))
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

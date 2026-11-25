"""The follow-up CoT variants (F1-F3) against the structured-CoT pilot's conditions A, C, D and E on the same questions;
writes analysis.md and errors.csv into the results folder.

  PYTHONPATH=src python experiments/2026-11-25-concise-math-cot/analysis.py --results <out> \
      --sheets-root tasks/openvocab_adapter/out_full [--baselines results/zeroshot/vsibench/structured_cot/explore200]

Scores are VSI-Bench's (score_final, as every condition was scored), x100. The computing score is the mean of the three
computing tasks' scores (summarize). Comparisons are question by question, on the questions every condition answered:
better / worse / same, an exact two-sided sign test, Holm within each family (FAMILIES), and a paired bootstrap 95%
interval of the computing score's difference (questions resampled within each question type, the same resample for
every condition, as 2026-11-19-structured-cot/analysis.py). With fewer than FULL questions per task the report says it
is exploratory.

errors.csv has one row per (condition, question) for the CoT conditions with what can be measured automatically (cut
off, no answer line, the number or letter read, whether it equals the centre-distance or closest-point rule computed
from the gold sheet, a metres-for-centimetres slip, whether the gold sheet's own rule disagrees with the ground truth)
and empty columns for what needs reading the reply (object identification, attribute extraction, operation, arithmetic,
unit, notes). A reply that contains the sheet's numbers or ids is not counted as correct reasoning.
"""
import argparse
import csv
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mvp.benchmarks.vsibench import (MCA_QUESTION_TYPES, final_letter, final_number, parsed, score_final, summarize,
                                     task_of)
from mvp.paths import ROOT
from mvp.reporting.compare_runs import holm, paired, sign_test

COMPUTING = ("object_abs_distance", "object_rel_distance", "object_size_estimation")
BASELINES = ROOT / "results/zeroshot/vsibench/structured_cot/explore200"
BASELINE_FILES = {"A": ("A_video", "video only, direct"), "C": ("gold_csv", "gold sheet, direct"),
                  "D": ("gold_cot", "gold sheet, structured CoT (pilot)"),
                  "E": ("gold_relations", "gold sheet + relations, direct")}
VARIANTS = {"F1": ("cot_f1", "gold sheet, concise CoT"), "F2": ("cot_f2", "gold sheet, + geometric formulas"),
            "F3": ("cot_f3", "gold sheet, + worked examples")}
FAMILIES = {  # (condition, reference, question); Holm within each family
    "primary: each intervention": [("F1", "D", "concise reasoning"), ("F2", "F1", "explicit formulas"),
                                   ("F3", "F2", "worked examples")],
    "against direct answering with the sheet (C)": [(f, "C", "") for f in ("F1", "F2", "F3")],
    "against precomputed relations (E)": [(f, "E", "") for f in ("F1", "F2", "F3")],
    "against video only (A)": [(f, "A", "") for f in ("F1", "F2", "F3")],
}
FULL = 25  # questions per task in explore200
B, SEED = 10_000, 0
CLOSE = 0.05  # a number within 5% of a rule's value counts as that rule's


def load_run(path):
    return pd.read_json(path, lines=True).drop_duplicates("id").set_index("id").sort_index()


def load(results, baselines=BASELINES):
    runs = {}
    for c, (stem, _) in VARIANTS.items():
        path = Path(results) / stem / "predictions.jsonl"
        if path.is_file():
            runs[c] = load_run(path)
    if not runs:
        raise SystemExit(f"{results}: no predictions of {', '.join(s for s, _ in VARIANTS.values())}")
    ids = sorted(set.intersection(*(set(r.index) for r in runs.values())))
    for c, (stem, _) in BASELINE_FILES.items():
        base = load_run(Path(baselines) / f"{stem}.jsonl")
        missing = set(ids) - set(base.index)
        if missing:
            raise SystemExit(f"baseline {c} has no answer for {sorted(missing)[:5]}")
        runs[c] = base
    return {c: r.loc[ids] for c, r in runs.items()}, ids


def computing_score(run):
    s = summarize(run.reset_index())
    return float(np.mean([s[t] for t in COMPUTING if t in s]))


def measures(run, timing=None):
    s = summarize(run.reset_index())
    out = {"computing": computing_score(run), **{t: s.get(t) for t in COMPUTING},
           "parse rate": 100 * np.mean([parsed(row, row["prediction"], True) for _, row in run.reset_index().iterrows()])}
    if "n_new_tokens" in run:
        out["reply tokens median"] = float(np.median(run["n_new_tokens"]))
        out["reply tokens p90"] = float(np.percentile(run["n_new_tokens"], 90))
    if "hit_token_limit" in run:
        out["cut off"] = 100 * float(run["hit_token_limit"].fillna(False).astype(bool).mean())
    if timing:
        out["seconds per question"] = timing
    return out


def bootstrap_diff(runs, a, b, seed=SEED, n=B):
    """95% interval of computing(a) - computing(b), questions resampled within each question type."""
    qtypes = runs[a]["question_type"].to_numpy()
    groups = {t: np.flatnonzero(qtypes == t) for t in sorted(set(qtypes))}
    sa, sb = runs[a]["score"].to_numpy(float), runs[b]["score"].to_numpy(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = {t: rng.choice(m, size=len(m)) for t, m in groups.items()}
        diffs[i] = 100 * (np.mean([sa[m].mean() for m in idx.values()]) - np.mean([sb[m].mean() for m in idx.values()]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def comparisons(runs):
    out = []
    for family, pairs in FAMILIES.items():
        rows = []
        for c, ref, what in pairs:
            if c not in runs or ref not in runs:
                continue
            better, worse, same = paired(runs[ref]["score"], runs[c]["score"])
            lo, hi = bootstrap_diff(runs, c, ref)
            rows.append({"family": family, "comparison": f"{c} - {ref}", "what": what,
                         "diff": computing_score(runs[c]) - computing_score(runs[ref]),
                         **{f"diff {t}": summarize(runs[c].reset_index()).get(t, np.nan)
                            - summarize(runs[ref].reset_index()).get(t, np.nan) for t in COMPUTING},
                         "better": better, "worse": worse, "same": same, "p": sign_test(better, worse),
                         "ci": (lo, hi)})
        for r, h in zip(rows, holm([r["p"] for r in rows])):
            r["holm p"] = h
        out += rows
    return out


# ---- error analysis ----

@functools.lru_cache(maxsize=None)
def gold_index(sheets_root):
    with open(Path(sheets_root) / "index.csv", newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def gold_rows(sheets_root, question_id):
    row = gold_index(str(sheets_root))[question_id]
    with open(Path(sheets_root) / row["gold_sheet"] / "objects.csv", newline="") as f:
        return list(csv.DictReader(f)), row


def box(r):
    return np.array([float(r[k]) for k in "xyz"]), np.array([float(r[f"size_{k}"]) for k in "xyz"])


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.abs(a[0] - b[0]) - (a[1] + b[1]) / 2)))


def centre(a, b):
    return float(np.linalg.norm(a[0] - b[0]))


def rules(question, rows, prompts):
    """What the gold sheet gives under each rule, or {} if a named object has no row."""
    by = {}
    for r in rows:
        by.setdefault(r["concept"], []).append(box(r))
    names = [p.strip() for p in prompts.split(",") if p.strip()]
    t = question["question_type"]
    if t == "object_size_estimation":
        if names[0] not in by:
            return {}
        sides = sorted({float(s) for _, e in by[names[0]] for s in e})
        return {"longest_cm": 100 * max(sides), "longest_m": max(sides), "other_sides_cm": [100 * s for s in sides[:-1]]}
    if t == "object_abs_distance":
        a, b = names[:2]
        if a not in by or b not in by:
            return {}
        return {"closest": min(gap(x, y) for x in by[a] for y in by[b]),
                "centre": min(centre(x, y) for x in by[a] for y in by[b])}
    *cands, target = names
    if target not in by or not all(c in by for c in cands):
        return {}
    d = {k: [min(f(x, y) for x in by[c] for y in by[target]) for c in cands] for k, f in (("closest", gap), ("centre", centre))}
    return {"closest_option": "ABCD"[int(np.argmin(d["closest"]))], "centre_option": "ABCD"[int(np.argmin(d["centre"]))]}


def near(value, target):
    return value is not None and target is not None and abs(value - target) <= CLOSE * max(abs(target), 1e-9)


def error_rows(runs, sheets_root, conditions=("D", "F1", "F2", "F3")):
    out = []
    for c in conditions:
        if c not in runs:
            continue
        for q, row in runs[c].iterrows():
            doc = row.to_dict()
            text = str(doc["prediction"])
            rows, idx = gold_rows(sheets_root, q)
            r = rules(doc, rows, idx["prompts"])
            mca = doc["question_type"] in MCA_QUESTION_TYPES
            read = final_letter(text) if mca else final_number(text)
            flags = {"cut_off": bool(doc.get("hit_token_limit")), "no_answer_line": "answer:" not in text.lower(),
                     "unread": not parsed(doc, text, True)}
            rule_score = None
            if doc["question_type"] == "object_abs_distance" and r:
                flags["equals_centre_rule"] = near(read, r["centre"]) and not near(read, r["closest"])
                flags["equals_closest_rule"] = near(read, r["closest"])
                rule_score = score_final(doc, str(r["closest"]))
            elif doc["question_type"] == "object_rel_distance" and r:
                flags["picks_centre_rule"] = read == r["centre_option"] != r["closest_option"]
                flags["picks_closest_rule"] = read == r["closest_option"]
                rule_score = score_final(doc, r["closest_option"])
            elif doc["question_type"] == "object_size_estimation" and r:
                flags["metres_not_cm"] = near(read, r["longest_m"])
                flags["another_side"] = any(near(read, s) for s in r["other_sides_cm"]) and not near(read, r["longest_cm"])
                rule_score = score_final(doc, str(r["longest_cm"]))
            flags["sheet_rule_disagrees_with_truth"] = rule_score is not None and rule_score < 0.5
            flags["named_objects_missing_from_sheet"] = not r
            out.append({"condition": c, "id": q, "question_type": doc["question_type"], "ground_truth": doc["ground_truth"],
                        "read": read, "score": doc["score"], "reply_tokens": doc.get("n_new_tokens"), **flags,
                        # to be filled by reading the reply
                        "manual_objects": "", "manual_attributes": "", "manual_operation": "", "manual_arithmetic": "",
                        "manual_unit": "", "manual_notes": ""})
    return pd.DataFrame(out)


def timing_per_question(results, stem):
    path = Path(results) / stem / "timing.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    new = sum(r["new"] for r in rows)
    return sum(r["seconds"] for r in rows) / new if new else None


def report(results, sheets_root, baselines=BASELINES):
    runs, ids = load(results, baselines)
    per_task = runs[next(iter(VARIANTS.keys() & runs.keys()))]["question_type"].value_counts()
    n_task = int(per_task.min())
    names = {**{c: w for c, (_, w) in BASELINE_FILES.items()}, **{c: w for c, (_, w) in VARIANTS.items()}}
    timing = {c: timing_per_question(results, stem) for c, (stem, _) in VARIANTS.items()}
    lines = ["# Concise and mathematical CoT: results", "",
             "<!-- Written by analysis.py; edit that, not this file. -->", ""]
    if n_task < FULL:
        lines += [f"**Exploratory: {len(ids)} questions ({n_task} per task).** At this size the intervals are wide "
                  "and nothing here is an established difference; it is for checking the runs and the prompts.", ""]
    lines += [f"Questions: {len(ids)} ({', '.join(f'{t} {n}' for t, n in per_task.items())}), the same for every "
              "condition. A, C, D, E: the structured-CoT pilot's predictions (2026-11-19-structured-cot), not run "
              "again. Scores x100 (score_final).", "",
              "| Condition | What | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | "
              "Cut off | s / question |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in ("A", "C", "D", "E", "F1", "F2", "F3"):
        if c not in runs:
            continue
        m = measures(runs[c], timing.get(c))
        tokens = f"{m['reply tokens median']:.0f} / {m['reply tokens p90']:.0f}" if "reply tokens median" in m else "-"
        secs = f"{m['seconds per question']:.1f}" if m.get("seconds per question") else "-"
        lines.append(f"| {c} | {names[c]} | {m['computing']:.1f} | {m['object_abs_distance']:.1f} | "
                     f"{m['object_rel_distance']:.1f} | {m['object_size_estimation']:.1f} | {m['parse rate']:.1f}% | "
                     f"{tokens} | {m.get('cut off', 0):.1f}% | {secs} |")
    comps = comparisons(runs)
    lines += ["", "Comparisons, question by question (better / worse: questions where the first condition scores "
              "higher / lower; exact two-sided sign test; Holm within each family; paired bootstrap 95% interval of "
              f"the computing score's difference, {B:,} resamples within question types):", ""]
    for family in FAMILIES:
        rows = [r for r in comps if r["family"] == family]
        if not rows:
            continue
        lines += [f"**{family}**", "", "| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | "
                  "Better / worse / same | p | Holm p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['comparison']}{' (' + r['what'] + ')' if r['what'] else ''} | {r['diff']:+.1f} | "
                         f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | {r['diff object_abs_distance']:+.1f} | "
                         f"{r['diff object_rel_distance']:+.1f} | {r['diff object_size_estimation']:+.1f} | "
                         f"{r['better']} / {r['worse']} / {r['same']} | {r['p']:.3g} | {r['holm p']:.3g} |")
        lines.append("")
    errors = error_rows(runs, sheets_root)
    errors.to_csv(Path(results) / "errors.csv", index=False)
    flag_cols = [c for c in errors.columns if errors[c].dtype == bool]
    lines += ["Automatic error flags (share of each CoT condition's replies; errors.csv has them per reply, with empty "
              "columns for what needs reading: object identification, attribute extraction, operation, arithmetic, "
              "unit). A flag such as 'equals the centre rule' says what number the reply ended on, not how it got "
              "there.", "", "| Flag | " + " | ".join(sorted(set(errors["condition"]))) + " |",
              "|---|" + "---:|" * errors["condition"].nunique()]
    for f in flag_cols:
        vals = errors.groupby("condition")[f].mean()
        lines.append(f"| {f} | " + " | ".join(f"{100 * vals.get(c, np.nan):.0f}%" for c in sorted(set(errors['condition']))) + " |")
    text = "\n".join(lines) + "\n"
    (Path(results) / "analysis.md").write_text(text)
    return text, runs, comps, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="followup.py's --out: <variant>/predictions.jsonl")
    ap.add_argument("--sheets-root", required=True, help="the gold sheets the runs used")
    ap.add_argument("--baselines", default=str(BASELINES))
    args = ap.parse_args()
    print(report(args.results, args.sheets_root, args.baselines)[0])


if __name__ == "__main__":
    main()

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
unit, notes). A reply that contains the sheet's numbers or ids is not counted as correct reasoning. Two flags check
whether a reply took F3's worked example for its own: its answer is the example's (1.059 m, 183 cm), or it quotes two or
more of the example's numbers that the question's own sheet does not have; F1 and F2, which never see the example, are
the control.

--model-size (exploratory, README "Model size"): Qwen3.5-9B against Qwen3.5-2B on the same questions
(model_size_ids.txt) and prompts: a table per model and condition, and one Holm family of contrasts (9B F1 - 9B C,
2B F1 - 2B C, their difference, 9B C - 2B C, 9B E - 2B E), each a per-question contrast of scores with its sign test and
paired bootstrap interval. Writes model_size.md into the results folder.

  PYTHONPATH=src python experiments/2026-11-25-concise-math-cot/analysis.py --results <out> --sheets-root <out_full> \
      --model-size
"""
import argparse
import csv
import functools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mvp.benchmarks.cot_variants import EXAMPLES
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
# the model-size comparison (README, "Model size"): the 9B's folder, its questions, the conditions per model
LARGE, LARGE_DIR = "Qwen/Qwen3.5-9B", "qwen3.5-9b"  # followup.model_dir's name for it
SIZE_IDS = Path(__file__).resolve().parent / "model_size_ids.txt"
SIZE_CONDITIONS = {"2B": ("A", "C", "D", "E", "F1"), "9B": ("A", "C", "E", "F1", "F2")}
OPTIONAL = {("9B", "F2")}  # run last if time allows; left out unless it answered every question the others did
SIZE_CONTRASTS = [  # (comparison, question, {(model, condition): weight}); one Holm family
    ("9B F1 - 9B C", "does CoT help the 9B?", {("9B", "F1"): 1, ("9B", "C"): -1}),
    ("2B F1 - 2B C", "does CoT help the 2B?", {("2B", "F1"): 1, ("2B", "C"): -1}),
    ("(9B F1 - 9B C) - (2B F1 - 2B C)", "does model size change the effect of CoT?",
     {("9B", "F1"): 1, ("9B", "C"): -1, ("2B", "F1"): -1, ("2B", "C"): 1}),
    ("9B C - 2B C", "the sheet, direct", {("9B", "C"): 1, ("2B", "C"): -1}),
    ("9B E - 2B E", "the sheet and relations, direct", {("9B", "E"): 1, ("2B", "E"): -1}),
]
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


def example_answer(task):
    """The answer F3's worked example for this task ends on."""
    return EXAMPLES[task].rsplit("Answer: ", 1)[1].strip()


def example_numbers(task, rows):
    """The numbers written with three decimals in F3's example for this task that the question's own sheet rows do
    not have: a reply that quotes them took them from the example."""
    own = {f"{float(r[k]):.3f}" for r in rows for k in ("x", "y", "z", "size_x", "size_y", "size_z")}
    return set(re.findall(r"-?\d+\.\d{3}", EXAMPLES[task])) - own


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
            quoted = sorted(n for n in example_numbers(doc["question_type"], rows) if n in text)
            flags["copies_example_answer"] = (not mca and read is not None
                                              and near(read, float(example_answer(doc["question_type"]))))
            flags["quotes_example_numbers"] = len(quoted) >= 2
            flags["sheet_rule_disagrees_with_truth"] = rule_score is not None and rule_score < 0.5
            flags["named_objects_missing_from_sheet"] = not r
            out.append({"condition": c, "id": q, "question_type": doc["question_type"], "ground_truth": doc["ground_truth"],
                        "read": read, "score": doc["score"], "reply_tokens": doc.get("n_new_tokens"),
                        "example_numbers_quoted": " ".join(quoted), **flags,
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


# ---- model size (exploratory) ----

STEMS = {**{c: s for c, (s, _) in BASELINE_FILES.items()}, **{c: s for c, (s, _) in VARIANTS.items()}}


def load_sizes(results, baselines=BASELINES, ids_file=SIZE_IDS):
    """{(model, condition): run} on the questions of ids_file that every run answered, and those ids. The 2B's A, C,
    D and E are the pilot's, its F1 is <results>/cot_f1; the 9B's are <results>/qwen3.5-9b/<stem>. A missing run
    stops it, except an optional one, which is left out unless it answered all the others' questions."""
    ids = [int(x) for x in Path(ids_file).read_text().split()]
    runs = {}
    for model, conditions in SIZE_CONDITIONS.items():
        for c in conditions:
            if model == "2B" and c in BASELINE_FILES:
                path = Path(baselines) / f"{STEMS[c]}.jsonl"
            else:
                path = Path(results) / (LARGE_DIR if model == "9B" else "") / STEMS[c] / "predictions.jsonl"
            if path.is_file():
                runs[(model, c)] = load_run(path)
            elif (model, c) not in OPTIONAL:
                raise SystemExit(f"{model} {c}: no predictions at {path}")
    common = set(ids)
    for key, r in runs.items():
        if key not in OPTIONAL:
            common &= set(r.index)
    common = sorted(common)
    if not common:
        raise SystemExit("no question answered by every condition of both models")
    runs = {k: r.loc[common] for k, r in runs.items() if set(common) <= set(r.index)}
    return runs, common


def contrast(runs, weights, seed=SEED, n=B):
    """A per-question contrast of scores, sum of weight x score: its computing-score value (the mean of the task
    means), per task, better / worse / same (contrast > 0 / < 0 / = 0), the sign test and a paired bootstrap 95%
    interval (questions resampled within question type)."""
    first = runs[next(iter(weights))]
    qtypes = first["question_type"].to_numpy()
    v = sum(w * runs[k]["score"].to_numpy(float) for k, w in weights.items())
    groups = {t: np.flatnonzero(qtypes == t) for t in sorted(set(qtypes))}
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean([v[rng.choice(m, size=len(m))].mean() for m in groups.values()]) for _ in range(n)])
    better, worse = int((v > 1e-12).sum()), int((v < -1e-12).sum())
    return {"diff": 100 * float(np.mean([v[m].mean() for m in groups.values()])),
            **{f"diff {t}": 100 * float(v[m].mean()) for t, m in groups.items()},
            "better": better, "worse": worse, "same": len(v) - better - worse, "p": sign_test(better, worse),
            "ci": (100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5)))}


def size_report(results, baselines=BASELINES, ids_file=SIZE_IDS):
    runs, ids = load_sizes(results, baselines, ids_file)
    first = next(iter(runs.values()))
    per_task = first["question_type"].value_counts()
    names = {**{c: w for c, (_, w) in BASELINE_FILES.items()}, **{c: w for c, (_, w) in VARIANTS.items()}}
    lines = ["# Model size: Qwen3.5-9B against Qwen3.5-2B", "", "<!-- Written by analysis.py --model-size. -->", "",
             f"**Exploratory: {len(ids)} questions ({int(per_task.min())} per task), development questions "
             "(explore200), on which the prompts were designed.** Nothing here is an established difference.", "",
             f"Questions: {len(ids)} ({', '.join(f'{t} {n}' for t, n in per_task.items())}), the same for every row. "
             "The same prompts, frames, decoding, token limit, batches and answer rule for both models; the 2B's A, "
             "C, D and E are the structured-CoT pilot's predictions. Scores x100 (score_final).", "",
             "| Model | Condition | What | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | "
             "Reply tokens, median / p90 | Cut off | s / question |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (model, c), run in runs.items():
        folder = Path(results) / (LARGE_DIR if model == "9B" else "")
        pilot = model == "2B" and c in BASELINE_FILES  # the pilot's runs have no timing.jsonl
        m = measures(run, None if pilot else timing_per_question(folder, STEMS[c]))
        tokens = f"{m['reply tokens median']:.0f} / {m['reply tokens p90']:.0f}" if "reply tokens median" in m else "-"
        secs = f"{m['seconds per question']:.1f}" if m.get("seconds per question") else "-"
        lines.append(f"| {model} | {c} | {names[c]} | {m['computing']:.1f} | {m['object_abs_distance']:.1f} | "
                     f"{m['object_rel_distance']:.1f} | {m['object_size_estimation']:.1f} | {m['parse rate']:.1f}% | "
                     f"{tokens} | {m.get('cut off', 0):.1f}% | {secs} |")
    rows = [{"comparison": name, "what": what, **contrast(runs, w)} for name, what, w in SIZE_CONTRASTS
            if all(k in runs for k in w)]
    for r, h in zip(rows, holm([r["p"] for r in rows])):
        r["holm p"] = h
    lines += ["", "s / question includes loading the model once per call. Contrasts, question by question (better / "
              "worse: questions where the contrast is above / below 0; exact two-sided sign test; Holm over the "
              f"family; paired bootstrap 95% interval, {B:,} resamples within question types):", "",
              "| Comparison | Question | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | "
              "Better / worse / same | p | Holm p |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['comparison']} | {r['what']} | {r['diff']:+.1f} | [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | "
                     f"{r['diff object_abs_distance']:+.1f} | {r['diff object_rel_distance']:+.1f} | "
                     f"{r['diff object_size_estimation']:+.1f} | {r['better']} / {r['worse']} / {r['same']} | "
                     f"{r['p']:.3g} | {r['holm p']:.3g} |")
    text = "\n".join(lines) + "\n"
    (Path(results) / "model_size.md").write_text(text)
    return text, runs, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="followup.py's --out: <variant>/predictions.jsonl")
    ap.add_argument("--sheets-root", required=True, help="the gold sheets the runs used")
    ap.add_argument("--baselines", default=str(BASELINES))
    ap.add_argument("--model-size", action="store_true", help="the 9B against the 2B (model_size.md)")
    args = ap.parse_args()
    if args.model_size:
        print(size_report(args.results, args.baselines)[0])
    else:
        print(report(args.results, args.sheets_root, args.baselines)[0])


if __name__ == "__main__":
    main()

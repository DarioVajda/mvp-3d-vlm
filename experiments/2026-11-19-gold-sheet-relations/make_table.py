"""The experiment's results once runs A–C (and D, E if present) have finished: the README's Results block and
results.csv, in this folder.

  PYTHONPATH=src python experiments/2026-11-19-gold-sheet-relations/make_table.py \
      [--results results/zeroshot/vsibench/gold_sheet_relations] [--ids .../question_ids.txt] \
      [--sheets-root tasks/openvocab_adapter/out] [--readme .../README.md]

Checks that every run answered exactly the questions of --ids, then per condition: VSI-Bench overall (macro over the
tasks present, as mvp.benchmarks.vsibench.summarize), the overall without counting, per task and question type, the
parse rate (the final-answer rule, compare_runs.parse_rate) and the mean and median input tokens. Comparisons, question by
question with an exact two-sided sign test: B − A and C − B, Holm-corrected as one family; C − A and B − A without the
counting questions, descriptive; with D and E (no video), B − D, E − C and D − A as a second Holm family. C − B is also broken down by whether the question gets relations (scene_sheet.relations) and,
among those, by absolute and relative distance (where the relations state the answer) against the other types.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import TASKS, summarize
from mvp.paths import SCENE_SHEETS
from mvp.reporting.compare_runs import holm, paired, parse_rate, sign_test

HERE = Path(__file__).parent
RESULTS = Path("results/zeroshot/vsibench/gold_sheet_relations")
# (condition, name, prediction file stem)
CONDITIONS = [("A", "video only", "qwen3_5_2B_A_video"),
              ("B", "+ gold sheet", "qwen3_5_2B_B_gold_csv"),
              ("C", "+ gold sheet + relations", "qwen3_5_2B_C_gold_relations_csv")]
# without the video (--no-video); included when their files are there
NO_VIDEO = [("D", "gold sheet only", "qwen3_5_2B_D_gold_csv_no_video"),
            ("E", "gold sheet + relations only", "qwen3_5_2B_E_gold_relations_csv_no_video")]
TASK_NAMES = {"object_counting": "Count", "object_abs_distance": "Abs. dist.", "object_size_estimation": "Obj. size",
              "room_size_estimation": "Room size", "object_rel_distance": "Rel. dist.",
              "object_rel_direction": "Rel. dir.", "route_planning": "Route", "obj_appearance_order": "Order"}
STATES_ANSWER = {"object_abs_distance", "object_rel_distance"}  # the relations state the answer to most of these
START, END = "<!-- make_table:start -->", "<!-- make_table:end -->"


def load_runs(results, ids, conditions=CONDITIONS):
    """{condition: predictions indexed by id}, each checked to answer exactly `ids`, once each."""
    runs, problems = {}, []
    for cond, _, stem in conditions:
        path = Path(results) / f"{stem}.jsonl"
        if not path.exists():
            problems.append(f"{cond}: {path} is missing")
            continue
        run = pd.read_json(path, lines=True)
        dup = run["id"][run["id"].duplicated()].tolist()
        missing, extra = set(ids) - set(run["id"]), set(run["id"]) - set(ids)
        if dup or missing or extra:
            problems.append(f"{cond}: {len(missing)} questions missing, {len(extra)} not in the list, "
                            f"{len(dup)} answered twice ({path})")
        runs[cond] = run.drop_duplicates("id").set_index("id").sort_index()
    if problems:
        raise SystemExit("prediction files don't cover the same questions:\n  " + "\n  ".join(problems))
    return runs


def overall(run, without_counting=False):
    run = run.reset_index()
    if without_counting:
        run = run[run["task"] != "object_counting"]
    return summarize(run)["overall"]


def compare(runs, ref, other, mask=None):
    """other − ref: overall and overall without counting (both macro), and the question-by-question sign test on the
    questions in mask (all by default)."""
    a, b = runs[ref], runs[other]
    if mask is not None:
        a, b = a[mask.reindex(a.index)], b[mask.reindex(b.index)]
    better, worse, same = paired(a["score"], b["score"])
    return {"comparison": f"{other} − {ref}", "overall_diff": overall(b) - overall(a),
            "no_count_diff": overall(b, True) - overall(a, True) if (a["task"] != "object_counting").any() else None,
            "better": better, "worse": worse, "same": same, "p": sign_test(better, worse)}


def table(runs, conditions=CONDITIONS):
    rows = []
    for cond, name, stem in conditions:
        run = runs[cond].reset_index()
        summary = summarize(run)
        per_type = (100 * run.groupby("question_type")["score"].mean()).add_prefix("type:")
        rows.append({"condition": cond, "name": name, "file": f"{stem}.jsonl", "n": len(run),
                     "overall": summary["overall"], "no_counting": overall(run, True),
                     **{t: summary.get(t) for t in TASKS}, **per_type.to_dict(),
                     "parse_rate": 100 * parse_rate(run), "input_tokens": run["n_input_tokens"].mean(),
                     "input_tokens_median": run["n_input_tokens"].median()})
    return pd.DataFrame(rows).set_index("condition")


def groups(runs, has_rel):
    """C − B on groups of questions; scores are means over the group's questions (x100)."""
    task = runs["A"]["task"]
    masks = [("get relations", has_rel), ("get no relations", ~has_rel),
             ("get relations: abs. and rel. distance", has_rel & task.isin(STATES_ANSWER)),
             ("get relations: the other types", has_rel & ~task.isin(STATES_ANSWER))]
    out = []
    for name, mask in masks:
        if not mask.any():
            continue
        b, c = runs["B"]["score"][mask], runs["C"]["score"][mask]
        better, worse, same = paired(b, c)
        out.append({"group": name, "n": int(mask.sum()), "A": 100 * runs["A"]["score"][mask].mean(),
                    "B": 100 * b.mean(), "C": 100 * c.mean(), "better": better, "worse": worse, "same": same,
                    "p": sign_test(better, worse)})
    return out


def fmt(v, spec=".1f"):
    return "–" if v is None or pd.isna(v) else format(v, spec)


def markdown(df, comparisons, by_group, runs, has_rel, ids):
    tasks = [t for t in TASKS if t in df.columns and df[t].notna().any()]
    absent = [TASK_NAMES[t] for t in TASKS if t not in tasks]
    lines = [f"Written by `make_table.py` ({len(ids):,} questions per condition; scores ×100; overall: the mean over "
             f"the {len(tasks)} tasks present" + (f", without {', '.join(absent)}, which these questions don't have"
                                                  if absent else "") + ").", "",
             "| ID | Condition | Overall | Without counting | " + " | ".join(TASK_NAMES[t] for t in tasks)
             + " | Parsed | Input tokens (mean / median) |",
             "|---|---|" + "---:|" * (len(tasks) + 4)]
    for cond, r in df.iterrows():
        lines.append(f"| {cond} | {r['name']} | {fmt(r.overall)} | {fmt(r.no_counting)} | "
                     + " | ".join(fmt(r[t]) for t in tasks) + f" | {r.parse_rate:.1f}% | {r.input_tokens:.0f} / {r.input_tokens_median:.0f} |")
    lines += ["", "Comparisons, question by question (better / worse: questions where the second condition scores "
              "higher / lower; exact two-sided sign test; Holm over B − A and C − B, and with D and E over B − D, "
              "E − C and D − A):", "",
              "| Comparison | Overall | Without counting | Better / worse / same | p | Holm p |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in comparisons:
        lines.append(f"| {c['comparison']}{c.get('note', '')} | {fmt(c['overall_diff'], '+.1f')} | "
                     f"{fmt(c['no_count_diff'], '+.1f')} | {c['better']} / {c['worse']} / {c['same']} | "
                     f"{c['p']:.3g} | {fmt(c.get('holm_p'), '.3g')} |")
    lines += ["", f"C − B by group ({int(has_rel.sum())} of {len(has_rel)} questions get relations; scores: the mean "
              "over the group's questions, not over tasks):", "",
              "| Questions that | n | A | B | C | C − B | Better / worse / same | p |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in by_group:
        lines.append(f"| {g['group']} | {g['n']} | {g['A']:.1f} | {g['B']:.1f} | {g['C']:.1f} | {g['C'] - g['B']:+.1f} | "
                     f"{g['better']} / {g['worse']} / {g['same']} | {g['p']:.3g} |")
    types = sorted(c[5:] for c in df.columns if c.startswith("type:"))
    n_type = runs["A"].groupby("question_type").size()
    rel_type = has_rel.groupby(runs["A"]["question_type"]).sum()
    lines += ["", "Per question type:", "", "| Question type | n | With relations | A | B | C | C − B |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for t in types:
        a, b, c = (df.loc[x, f"type:{t}"] for x in "ABC")
        lines.append(f"| {t} | {n_type[t]} | {int(rel_type[t])} | {a:.1f} | {b:.1f} | {c:.1f} | {c - b:+.1f} |")
    datasets = runs["A"]["dataset"].unique()
    if len(datasets) > 1:
        lines += ["", "Per dataset (mean over questions):", "", "| Dataset | n | A | B | C |", "|---|---:|---:|---:|---:|"]
        for d in sorted(datasets):
            m = runs["A"]["dataset"] == d
            lines.append(f"| {d} | {int(m.sum())} | " + " | ".join(f"{100 * runs[x]['score'][m].mean():.1f}"
                                                                     for x in "ABC") + " |")
    return "\n".join(lines)


def write_readme(readme, block):
    text = Path(readme).read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme}: no {START} ... {END} markers in Results")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
    Path(readme).write_text(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--ids", default=str(HERE / "question_ids.txt"))
    ap.add_argument("--sheets-root", default=str(SCENE_SHEETS))
    ap.add_argument("--readme", default=str(HERE / "README.md"))
    ap.add_argument("--csv", default=str(HERE / "results.csv"))
    args = ap.parse_args()
    ids = [int(line) for line in open(args.ids) if line.strip()]
    no_video = [c for c in NO_VIDEO if (Path(args.results) / f"{c[2]}.jsonl").exists()]
    if len(no_video) == 1:
        raise SystemExit(f"only {no_video[0][0]} of D and E is in {args.results}: both or neither")
    conditions = CONDITIONS + no_video
    runs = load_runs(args.results, ids, conditions)
    has_rel = pd.Series({q: ss.relations(q, Path(args.sheets_root)) != "" for q in runs["A"].index})
    df = table(runs, conditions)
    family = [compare(runs, "A", "B"), compare(runs, "B", "C")]
    second = [compare(runs, "D", "B"), compare(runs, "C", "E"), compare(runs, "A", "D")] if no_video else []
    for fam in (family, second):
        for c, h in zip(fam, holm([c["p"] for c in fam])):
            c["holm_p"] = h
    no_count = runs["A"]["task"] != "object_counting"
    descriptive = [compare(runs, "A", "C"), {**compare(runs, "A", "B", no_count), "note": ", counting left out"}]
    by_group = groups(runs, has_rel)
    df.to_csv(args.csv, float_format="%.4g")
    block = markdown(df, family + descriptive + second, by_group, runs, has_rel, ids)
    write_readme(args.readme, block)
    print(block)
    print(f"\nwrote the Results block of {args.readme} and {args.csv}")


if __name__ == "__main__":
    main()

"""The experiment's results once runs A–E have finished: the README's Results block, results.csv and the reasoning
check's sample (reasoning_check.md), in this folder.

  PYTHONPATH=src python experiments/2026-11-19-structured-cot/make_table.py \
      [--results results/zeroshot/vsibench/structured_cot] [--ids .../question_ids.txt] \
      [--sheets-root tasks/openvocab_adapter/out] [--tokenizer Qwen/Qwen3.5-2B] [--readme .../README.md]

Checks that every run answered exactly the questions of --ids, then per condition: VSI-Bench overall (macro over the
tasks present, mvp.benchmarks.vsibench.summarize), the overall without counting, the computing tasks' mean (absolute
distance, relative distance, object size), per task and question type, the parse rate (the final-answer rule), the
mean input tokens, and the reply length in tokens (median, 90th percentile, share at the run's --max-new-tokens).

Comparisons, question by question with an exact two-sided sign test: D − C, B − A and E − D, Holm-corrected as one
family; E − C and (D − C) − (B − A) descriptive. Breakdowns: question type, computing tasks vs the rest, multiple
choice vs numeric, dataset.

Reasoning check (README, Metrics): the share of D's replies that name a sheet row (an id such as chair_2) or copy one
of its numbers (a number with two or more decimals that equals a sheet value rounded to as many); and 20 wrong D answers
on the computing tasks (score below 0.5), drawn with seed 0, written to reasoning_check.md for labelling by hand. The
labels go in reasoning_labels.csv (id,label,note); once it exists, their counts are added to the Results block.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import MCA_QUESTION_TYPES, TASKS, final_letter, final_number, summarize
from mvp.paths import SCENE_SHEETS
from mvp.reporting.compare_runs import holm, paired, parse_rate, sign_test

HERE = Path(__file__).parent
RESULTS = Path("results/zeroshot/vsibench/structured_cot")
# (condition, name, prediction file stem)
CONDITIONS = [("A", "video, direct", "qwen3_5_2B_A_direct_video"),
              ("B", "video, CoT", "qwen3_5_2B_B_cot_video"),
              ("C", "+ gold sheet, direct", "qwen3_5_2B_C_direct_gold"),
              ("D", "+ gold sheet, CoT", "qwen3_5_2B_D_cot_gold"),
              ("E", "+ gold sheet + relations, direct", "qwen3_5_2B_E_direct_gold_relations")]
COMPUTING = ["object_abs_distance", "object_rel_distance", "object_size_estimation"]
TASK_NAMES = {"object_counting": "Count", "object_abs_distance": "Abs. dist.", "object_size_estimation": "Obj. size",
              "room_size_estimation": "Room size", "object_rel_distance": "Rel. dist.",
              "object_rel_direction": "Rel. dir.", "route_planning": "Route", "obj_appearance_order": "Order"}
LABELS = ["wrong row", "coordinates misread", "arithmetic wrong", "sheet ignored", "no answer line", "wrong method"]
LIMIT_MARGIN = 8  # a reply within this many tokens of --max-new-tokens counts as cut off (re-tokenising can differ)
WRONG = 0.5  # a computing-task answer scoring below this is wrong, for the reasoning check
N_SAMPLE = 20
START, END = "<!-- make_table:start -->", "<!-- make_table:end -->"
DECIMAL = re.compile(r"-?\d+\.(\d{2,})")


def load_runs(results, ids):
    """{condition: predictions indexed by id}, each checked to answer exactly `ids`, once each; and each run's
    --max-new-tokens (from its _summary.json, default 2048)."""
    runs, limits, problems = {}, {}, []
    for cond, _, stem in CONDITIONS:
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
        summary = path.with_name(f"{stem}_summary.json")
        limits[cond] = json.loads(summary.read_text()).get("max_new_tokens", 2048) if summary.exists() else 2048
    if problems:
        raise SystemExit("prediction files don't cover the same questions:\n  " + "\n  ".join(problems))
    return runs, limits


def macro(run, tasks=None, drop=()):
    """summarize's overall over the given tasks (all present by default), leaving out `drop`."""
    run = run.reset_index()
    s = summarize(run[~run["task"].isin(drop)])
    keys = [t for t in (tasks or TASKS) if t in s and t not in drop]
    return sum(s[t] for t in keys) / len(keys) if keys else None


def scores(run):
    return {"overall": macro(run), "no_counting": macro(run, drop=["object_counting"]),
            "computing": macro(run, COMPUTING)}


def compare(runs, ref, other):
    a, b = runs[ref], runs[other]
    sa, sb = scores(a), scores(b)
    better, worse, same = paired(a["score"], b["score"])
    comp = a["task"].isin(COMPUTING)
    cb, cw, _ = paired(a["score"][comp], b["score"][comp])
    return {"comparison": f"{other} − {ref}", **{f"{k}_diff": sb[k] - sa[k] for k in sa},
            "better": better, "worse": worse, "same": same, "p": sign_test(better, worse),
            "computing_better": cb, "computing_worse": cw, "computing_p": sign_test(cb, cw)}


def reply_tokens(runs, count):
    return {c: r["prediction"].map(count) for c, r in runs.items()}


def table(runs, limits, lengths):
    rows = []
    for cond, name, stem in CONDITIONS:
        run = runs[cond].reset_index()
        summary = summarize(run)
        per_type = (100 * run.groupby("question_type")["score"].mean()).add_prefix("type:")
        n = lengths[cond]
        rows.append({"condition": cond, "name": name, "file": f"{stem}.jsonl", "n": len(run), **scores(runs[cond]),
                     **{t: summary.get(t) for t in TASKS}, **per_type.to_dict(),
                     "parse_rate": 100 * parse_rate(run), "input_tokens": run["n_input_tokens"].mean(),
                     "reply_median": n.median(), "reply_p90": n.quantile(0.9), "max_new_tokens": limits[cond],
                     "at_limit": 100 * (n >= limits[cond] - LIMIT_MARGIN).mean()})
    return pd.DataFrame(rows).set_index("condition")


# ---- the reasoning check on D ----

def sheet_facts(question_id, root):
    """The gold sheet's object ids and numbers (floats) for this question."""
    objects = ss.load_objects(question_id, "gold", root)
    values = {float(o[k]) for o in objects for k in ss.POSITION + ss.EXTENT}
    return {o["id"].lower() for o in objects}, values


def copies_number(reply, values):
    for m in DECIMAL.finditer(reply):
        d = min(len(m.group(1)), ss.DECIMALS)
        x = float(m.group(0))
        if any(round(v, d) == round(x, d) for v in values):
            return True
    return False


def names_row(reply, ids):
    text = reply.lower()
    return any(re.search(rf"(?<![a-z0-9_]){re.escape(i)}(?![a-z0-9_])", text) for i in ids)


def reasoning_check(run, root):
    rows = []
    for q, r in run.iterrows():
        ids, values = sheet_facts(q, root)
        rows.append({"id": q, "task": r["task"], "names_row": names_row(r["prediction"], ids),
                     "copies_number": copies_number(r["prediction"], values),
                     "answer_line": bool(re.search(r"answer\s*:", r["prediction"], re.I))})
    return pd.DataFrame(rows).set_index("id")


def wrong_sample(run):
    wrong = run[run["task"].isin(COMPUTING) & (run["score"] < WRONG)]
    return wrong.sample(min(N_SAMPLE, len(wrong)), random_state=0).sort_index(), len(wrong)


def write_sample(sample, n_wrong, root, path):
    lines = [f"# Reasoning check: {len(sample)} wrong D answers on the computing tasks", "",
             f"Drawn with seed 0 from the {n_wrong} D answers on absolute distance, relative distance and object size "
             f"that score below {WRONG} (make_table.py). Each is labelled by hand in reasoning_labels.csv with one of: "
             + "; ".join(LABELS) + ".", ""]
    for q, r in sample.iterrows():
        answer = final_letter(r["prediction"]) if r["question_type"] in MCA_QUESTION_TYPES else final_number(r["prediction"])
        options = "\n".join(r["options"]) if isinstance(r["options"], list) else ""
        lines += [f"## {q} ({r['question_type']}, {r['dataset']}/{r['scene_name']})", "",
                  f"**Question:** {r['question'].strip()}", ""] + ([f"```\n{options}\n```", ""] if options else []) + [
                  f"**Ground truth:** {r['ground_truth']}  **Answer read:** {answer}  **Score:** {r['score']:.2f}", "",
                  "**Gold sheet:**", "", "```", ss.render_sheet(q, "gold", "csv", root).split("\n", 1)[1], "```", "",
                  "**Reply:**", "", "```", r["prediction"].strip(), "```", ""]
    Path(path).write_text("\n".join(lines))


# ---- the Results block ----

def fmt(v, spec=".1f"):
    return "–" if v is None or pd.isna(v) else format(v, spec)


def markdown(df, family, descriptive, runs, lengths, limits, check, labels, sample_ids, n_wrong, ids):
    tasks = [t for t in TASKS if t in df.columns and df[t].notna().any()]
    L = [f"Written by `make_table.py` ({len(ids):,} questions per condition; scores ×100; overall: the mean over the "
         f"{len(tasks)} tasks present; computing: the mean of absolute distance, relative distance and object size).",
         "", "| ID | Condition | Overall | Without counting | Computing | " + " | ".join(TASK_NAMES[t] for t in tasks)
         + " | Parsed | Input tokens | Reply tokens, median / p90 | At the limit |",
         "|---|---|" + "---:|" * (len(tasks) + 7)]
    for c, r in df.iterrows():
        L.append(f"| {c} | {r['name']} | {fmt(r.overall)} | {fmt(r.no_counting)} | {fmt(r.computing)} | "
                 + " | ".join(fmt(r[t]) for t in tasks) + f" | {r.parse_rate:.1f}% | {r.input_tokens:.0f} | "
                 f"{r.reply_median:.0f} / {r.reply_p90:.0f} | {r.at_limit:.1f}% |")
    L += ["", "Comparisons, question by question (better / worse: questions where the first condition scores higher / "
          "lower; exact two-sided sign test; Holm over D − C, B − A and E − D; on the computing tasks' questions alone, "
          "the sign test without Holm):", "",
          "| Comparison | Overall | Without counting | Computing | Better / worse / same | p | Holm p | "
          "Computing: better / worse | p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in family + descriptive:
        if "better" not in c:
            L.append(f"| {c['comparison']} | {fmt(c['overall_diff'], '+.1f')} | {fmt(c['no_counting_diff'], '+.1f')} | "
                     f"{fmt(c['computing_diff'], '+.1f')} | – | – | – | – | – |")
            continue
        L.append(f"| {c['comparison']} | {fmt(c['overall_diff'], '+.1f')} | {fmt(c['no_counting_diff'], '+.1f')} | "
                 f"{fmt(c['computing_diff'], '+.1f')} | {c['better']} / {c['worse']} / {c['same']} | {c['p']:.3g} | "
                 f"{fmt(c.get('holm_p'), '.3g')} | {c['computing_better']} / {c['computing_worse']} | "
                 f"{c['computing_p']:.3g} |")
    types = sorted(c[5:] for c in df.columns if c.startswith("type:"))
    n_type = runs["A"].groupby("question_type").size()
    L += ["", "Per question type:", "", "| Question type | n | " + " | ".join(df.index) + " | D − C | E − D |",
          "|---|---:|" + "---:|" * (len(df.index) + 2)]
    for t in types:
        v = {c: df.loc[c, f"type:{t}"] for c in df.index}
        L.append(f"| {t} | {n_type[t]} | " + " | ".join(f"{v[c]:.1f}" for c in df.index)
                 + f" | {v['D'] - v['C']:+.1f} | {v['E'] - v['D']:+.1f} |")
    task, qtype, dataset = runs["A"]["task"], runs["A"]["question_type"], runs["A"]["dataset"]
    groups = [("computing tasks", task.isin(COMPUTING)), ("the other tasks", ~task.isin(COMPUTING)),
              ("multiple choice", qtype.isin(MCA_QUESTION_TYPES)), ("numeric", ~qtype.isin(MCA_QUESTION_TYPES))]
    groups += [(d, dataset == d) for d in sorted(dataset.unique())]
    L += ["", "Groups of questions (mean over the group's questions, not over tasks):", "",
          "| Questions | n | " + " | ".join(df.index) + " |", "|---|---:|" + "---:|" * len(df.index)]
    for name, m in groups:
        L.append(f"| {name} | {int(m.sum())} | " + " | ".join(f"{100 * runs[c]['score'][m].mean():.1f}" for c in df.index)
                 + " |")
    L += ["", "Reply length per task, CoT runs (tokens; at the limit: within "
          f"{LIMIT_MARGIN} tokens of --max-new-tokens):", "",
          "| Task | n | B median | B at the limit | D median | D at the limit |", "|---|---:|---:|---:|---:|---:|"]
    for t in [t for t in TASKS if t in set(task)]:
        m = task == t
        cells = []
        for c in "BD":
            n = lengths[c][m]
            cells += [f"{n.median():.0f}", f"{100 * (n >= limits[c] - LIMIT_MARGIN).mean():.0f}%"]
        L.append(f"| {TASK_NAMES[t]} | {int(m.sum())} | " + " | ".join(cells) + " |")
    comp = check["task"].isin(COMPUTING)
    L += ["", "Reasoning check, D's replies:", "", "| Replies | n | Name a sheet row | Copy a sheet number | "
          "Have an \"Answer:\" line |", "|---|---:|---:|---:|---:|"]
    for name, m in [("all", check.index == check.index), ("computing tasks", comp), ("the other tasks", ~comp)]:
        sub = check[m]
        L.append(f"| {name} | {len(sub)} | {100 * sub['names_row'].mean():.0f}% | {100 * sub['copies_number'].mean():.0f}% "
                 f"| {100 * sub['answer_line'].mean():.0f}% |")
    if labels is not None:
        counts = labels["label"].value_counts()
        L += ["", f"Why D got it wrong: {len(labels)} of its {n_wrong} wrong answers on the computing tasks (score below "
              f"{WRONG}), drawn with seed 0 and labelled by hand (reasoning_check.md, reasoning_labels.csv):", "",
              "| Reason | Answers |", "|---|---:|"]
        L += [f"| {lab} | {counts.get(lab, 0)} |" for lab in LABELS + sorted(set(counts.index) - set(LABELS))]
    else:
        L += ["", f"The {len(sample_ids)} wrong D answers of the reasoning check are in reasoning_check.md; their labels "
              "are not in yet (reasoning_labels.csv)."]
    return "\n".join(L)


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
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--readme", default=str(HERE / "README.md"))
    ap.add_argument("--csv", default=str(HERE / "results.csv"))
    ap.add_argument("--sample", default=str(HERE / "reasoning_check.md"))
    ap.add_argument("--labels", default=str(HERE / "reasoning_labels.csv"))
    args = ap.parse_args()
    ids = [int(line) for line in open(args.ids) if line.strip()]
    root = Path(args.sheets_root)
    runs, limits = load_runs(args.results, ids)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = reply_tokens(runs, lambda text: len(tokenizer(text)["input_ids"]))
    df = table(runs, limits, lengths)

    family = [compare(runs, "C", "D"), compare(runs, "A", "B"), compare(runs, "D", "E")]
    for c, h in zip(family, holm([c["p"] for c in family])):
        c["holm_p"] = h
    dc, ba = family[0], family[1]
    descriptive = [compare(runs, "C", "E"),
                   {"comparison": "(D − C) − (B − A)",
                    **{k: dc[k] - ba[k] for k in ["overall_diff", "no_counting_diff", "computing_diff"]}}]

    check = reasoning_check(runs["D"], root)
    sample, n_wrong = wrong_sample(runs["D"])
    write_sample(sample, n_wrong, root, args.sample)
    labels = None
    if Path(args.labels).exists():
        labels = pd.read_csv(args.labels).set_index("id")
        if set(labels.index) != set(sample.index):
            raise SystemExit(f"{args.labels} labels {sorted(labels.index)}, not the sample {sorted(sample.index)}")
        unknown = set(labels["label"]) - set(LABELS)
        if unknown:
            print(f"note: labels outside {LABELS}: {sorted(unknown)}")

    df.to_csv(args.csv, float_format="%.4g")
    block = markdown(df, family, descriptive, runs, lengths, limits, check, labels, list(sample.index), n_wrong, ids)
    write_readme(args.readme, block)
    print(block)
    print(f"\nwrote the Results block of {args.readme}, {args.csv} and {args.sample}")


if __name__ == "__main__":
    main()

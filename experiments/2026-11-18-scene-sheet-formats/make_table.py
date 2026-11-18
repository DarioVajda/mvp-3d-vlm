"""The experiment's results, once the 11 runs (A–K) have finished: README's Results table, results.csv and the two LaTeX
tables (pipeline, gold), all in this folder.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/make_table.py \
      [--results results/zeroshot/vsibench/scene_sheet_formats] [--sheets-root tasks/openvocab_adapter/out] \
      [--vggt-root data/vggt] [--tokenizer Qwen/Qwen3.5-2B]

Checks that every run answered exactly the questions of question_ids.txt, then per condition: VSI-Bench overall and
per task and question type (mvp.benchmarks.vsibench.summarize), parse rate, mean sheet tokens (the rendered sheet,
preamble included, with --tokenizer), and a two-sided sign test against A, Holm-corrected over the ten conditions.
Pipeline vs gold per format is tested the same way (Holm over the five formats).
"""
import argparse
import re
import statistics
from pathlib import Path

import pandas as pd

from mvp.benchmarks import scene_sheet as ss
from mvp.benchmarks.vsibench import summarize
from mvp.paths import VGGT_DIR
from mvp.reporting.compare_runs import holm, paired, parse_rate, sign_test

HERE = Path(__file__).parent
RESULTS = Path("results/zeroshot/vsibench/scene_sheet_formats")
ORDER = ["csv", "markdown_table", "json", "markdown_kv", "text"]  # B–F and G–K, as in the README
# (condition, sheet source, sheet format, prediction file stem)
CONDITIONS = [("A", None, None, "qwen3_5_2B_A_video")] + [
    (c, "pipeline", f, f"qwen3_5_2B_{c}_{f}") for c, f in zip("BCDEF", ORDER)] + [
    (c, "gold", f, f"qwen3_5_2B_{c}_gold_{f}") for c, f in zip("GHIJK", ORDER)]
FORMAT_NAMES = {"csv": "CSV", "markdown_table": "Markdown table", "json": "JSON", "markdown_kv": "Markdown key-value",
                "text": "Text"}
COLUMNS = {"overall": "Overall", "object_counting": "Count", "object_abs_distance": "Abs. dist.",
           "object_size_estimation": "Obj. size", "room_size_estimation": "Room size",
           "object_rel_distance": "Rel. dist.", "object_rel_direction": "Rel. dir.", "route_planning": "Route",
           "obj_appearance_order": "Order"}
START, END = "<!-- make_table:start -->", "<!-- make_table:end -->"


def load_runs(results, ids):
    """The 11 prediction files, each checked to answer exactly `ids`, once each."""
    runs, problems = {}, []
    for cond, _, _, stem in CONDITIONS:
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


def sheet_tokens(count, sheets_root, vggt_root, ids):
    """{condition: mean tokens of the rendered sheet over ids}; count: text -> number of tokens."""
    return {cond: statistics.mean(count(ss.render_sheet(q, source, fmt, sheets_root, vggt_root=vggt_root)) for q in ids)
            for cond, source, fmt, _ in CONDITIONS if source}


def table(runs, tokens):
    """One row per condition: scores (x100), parse rate, tokens, and the sign test against A."""
    rows = []
    for cond, source, fmt, stem in CONDITIONS:
        run = runs[cond].reset_index()
        summary = summarize(run)
        per_type = (100 * run.groupby("question_type")["score"].mean()).add_prefix("type:")
        rows.append({"condition": cond, "source": source or "none", "format": fmt or "none", "file": f"{stem}.jsonl",
                     "n": len(run), **{k: summary.get(k) for k in COLUMNS}, **per_type.to_dict(),
                     "parse_rate": 100 * parse_rate(run), "sheet_tokens": tokens.get(cond)})
    df = pd.DataFrame(rows).set_index("condition")
    others = [c for c in df.index if c != "A"]
    counts = [paired(runs["A"]["score"], runs[c]["score"]) for c in others]
    p = [sign_test(b, w) for b, w, _ in counts]
    for c, (b, w, s), pc, ph in zip(others, counts, p, holm(p)):
        df.loc[c, ["better_than_A", "worse_than_A", "same_as_A", "p_vs_A", "holm_p_vs_A"]] = [b, w, s, pc, ph]
    return df


def pipeline_vs_gold(runs):
    """Per format: pipeline − gold, question by question (better = pipeline higher), Holm over the formats."""
    pairs = [(fmt, p, g) for (p, _, fmt, _), (g, _, _, _) in zip(CONDITIONS[1:6], CONDITIONS[6:11])]
    counts = [paired(runs[g]["score"], runs[p]["score"]) for _, p, g in pairs]
    pv = [sign_test(b, w) for b, w, _ in counts]
    return [dict(format=fmt, pipeline=p, gold=g, overall_diff=summarize(runs[p].reset_index())["overall"]
                 - summarize(runs[g].reset_index())["overall"], better=b, worse=w, same=s, p=pc, holm_p=ph)
            for (fmt, p, g), (b, w, s), pc, ph in zip(pairs, counts, pv, holm(pv))]


def markdown(df, versus):
    names = list(COLUMNS.values())
    lines = [f"| ID | Sheet | Format | {' | '.join(names)} | Parsed | Sheet tokens | vs A: better / worse | p | Holm p |",
             "|---|---|---|" + "---:|" * (len(names) + 5)]
    for cond, r in df.iterrows():
        scores = " | ".join(f"{r[k]:.1f}" for k in COLUMNS)
        tokens = f"{r.sheet_tokens:.0f}" if pd.notna(r.sheet_tokens) else "–"
        test = (f"{r.better_than_A:.0f} / {r.worse_than_A:.0f} | {r.p_vs_A:.3g} | {r.holm_p_vs_A:.3g}"
                if cond != "A" else "– | – | –")
        lines.append(f"| {cond} | {r.source} | {r.format} | {scores} | {r.parse_rate:.1f}% | {tokens} | {test} |")
    lines += ["", "Pipeline − gold, per format (better / worse: questions where the pipeline sheet scores higher / lower; "
              "Holm over the five formats):", "",
              "| Format | Conditions | Overall difference | Better / worse / same | p | Holm p |",
              "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {v['format']} | {v['pipeline']} − {v['gold']} | {v['overall_diff']:+.1f} | "
              f"{v['better']} / {v['worse']} / {v['same']} | {v['p']:.3g} | {v['holm_p']:.3g} |" for v in versus]
    return "\n".join(lines)


def latex(df, source):
    """booktabs table: A and the five formats of one source; the best value per column in bold (fewest tokens)."""
    rows = df.loc[["A"] + [c for c, s, _, _ in CONDITIONS if s == source]]
    cols = list(COLUMNS) + ["sheet_tokens"]
    text = {c: [f"{v:.1f}" for v in rows[c]] for c in COLUMNS}
    text["sheet_tokens"] = ["--"] + [f"{v:.0f}" for v in rows["sheet_tokens"].iloc[1:]]
    for c in cols:
        values = [float(t) for t in text[c] if t != "--"]
        best = f"{(min if c == 'sheet_tokens' else max)(values):.{0 if c == 'sheet_tokens' else 1}f}"
        text[c] = [rf"\textbf{{{t}}}" if t == best else t for t in text[c]]
    n = f"{int(df.loc['A', 'n']):,}".replace(",", "{,}")
    label = "tab:sheet-format" if source == "pipeline" else "tab:sheet-format-gold"
    sheet = "pipeline (VGGT + SAM~3)" if source == "pipeline" else "gold (VSI-Bench labels)"
    lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{l" + "r" * len(cols) + "}",
             r"\toprule", "Format & " + " & ".join(list(COLUMNS.values()) + ["Tokens"]) + r" \\", r"\midrule"]
    for i, (cond, r) in enumerate(rows.iterrows()):
        name = "Video only" if cond == "A" else FORMAT_NAMES[r.format]
        lines.append(f"{name} & " + " & ".join(text[c][i] for c in cols) + r" \\")
        if cond == "A":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Qwen3.5-2B on the {n} VSI-Bench questions of the 75 "
              rf"exported scenes, 16 frames, video plus the {sheet} scene sheet in each format. Scores $\times 100$; "
              rf"Tokens: mean tokens of the sheet. Best per column in bold.}}",
              rf"\label{{{label}}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def write_readme(readme, block):
    text = Path(readme).read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{readme}: no {START} ... {END} markers in Results")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
    Path(readme).write_text(new)


def run(results, ids, tokens_of, out_dir=HERE, readme=HERE / "README.md"):
    """Everything after argument parsing; tokens_of(ids) -> {condition: mean sheet tokens}."""
    runs = load_runs(results, ids)
    df = table(runs, tokens_of(ids))
    versus = pipeline_vs_gold(runs)
    out_dir = Path(out_dir)
    df.to_csv(out_dir / "results.csv", float_format="%.4g")
    for source in ss.SOURCES:
        (out_dir / f"results_{source}.tex").write_text(latex(df, source))
    write_readme(readme, f"Written by `make_table.py` from `{results}` ({len(ids):,} questions per condition; scores "
                         f"×100).\n\n" + markdown(df, versus))
    return df, versus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--ids", default=str(HERE / "question_ids.txt"))
    ap.add_argument("--sheets-root", default="tasks/openvocab_adapter/out")
    ap.add_argument("--vggt-root", default=str(VGGT_DIR))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()
    ids = [int(line) for line in open(args.ids) if line.strip()]

    def tokens_of(ids):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        return sheet_tokens(lambda t: len(tokenizer(t)["input_ids"]), Path(args.sheets_root), Path(args.vggt_root), ids)

    df, _ = run(args.results, ids, tokens_of)
    print(df[list(COLUMNS) + ["parse_rate", "sheet_tokens", "p_vs_A", "holm_p_vs_A"]].round(3).to_string())
    print(f"wrote README.md (Results), results.csv, results_pipeline.tex, results_gold.tex in {HERE}")


if __name__ == "__main__":
    main()

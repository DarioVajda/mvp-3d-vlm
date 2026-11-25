"""The report's two scene-sheet tables, in the style of its VSI-Bench table (booktabs, \\small, one decimal; relative
direction as VSI-Bench's mean of its easy, medium and hard types, mvp.benchmarks.vsibench.summarize):

- figures/table_explore200_gold.tex: explore200's 200 questions, video only (structured-cot's A_video) and the gold
  sheet without and with relations (the gold preview of 2026-11-19-gold-sheet-relations, --preview-dir).
- figures/table_batch2A_v4.tex: batch2_A's 798 questions, final2's video only and video + pipeline_v4 sheet, the full
  method assembled from them as prereg_batch2_A.md defines it (pipeline_v4's answer for ROUTED, video's for the rest),
  and pipeline_v4's sheet without the video (v4_sheet_only, exploratory).

Within each table the best value per column is bold (ties all bold, after rounding).

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/make_report_tables.py --preview-dir <gold_preview_results>
"""
import argparse
from pathlib import Path

import pandas as pd

from mvp.benchmarks.vsibench import TASKS, summarize

HERE = Path(__file__).parent
REPO = HERE.parents[1]
HEADERS = {"object_counting": "Obj.\\\\Count", "object_abs_distance": "Abs.\\\\Dist.",
           "object_size_estimation": "Obj.\\\\Size", "room_size_estimation": "Room\\\\Size",
           "object_rel_distance": "Rel.\\\\Dist.", "object_rel_direction": "Rel.\\\\Dir.",
           "route_planning": "Route\\\\Plan", "obj_appearance_order": "Appr.\\\\Order"}
ROUTED = {"object_abs_distance", "object_rel_distance", "object_rel_direction", "route_planning"}  # prereg_batch2_A.md


def load(path, ids):
    run = pd.read_json(path, lines=True).drop_duplicates("id")
    if sorted(run["id"]) != sorted(ids):
        raise SystemExit(f"{path}: does not answer exactly the {len(ids)} questions")
    return run


def full_method(video, sheet):
    """prereg_batch2_A.md's full method: the sheet condition's answer for the ROUTED tasks, video's for the others."""
    out = video.set_index("id").copy()
    routed = out["task"].isin(ROUTED)
    out.loc[routed, "score"] = sheet.set_index("id").loc[out.index[routed], "score"]
    return out.reset_index()


def table(rows, caption, label, marked=None):
    """rows: [(name, summary)]; marked: {row name: tasks whose cells get a dagger}."""
    cols = ["overall"] + list(TASKS)
    rounded = {name: {c: round(s[c], 1) for c in cols} for name, s in rows}
    best = {c: max(r[c] for r in rounded.values()) for c in cols}
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Condition & Overall & " + " & ".join(f"\\shortstack[r]{{{HEADERS[t]}}}" for t in TASKS) + " \\\\",
             "\\midrule"]
    for name, _ in rows:
        cells = []
        for c in cols:
            v = f"{rounded[name][c]:.1f}"
            v = f"\\textbf{{{v}}}" if rounded[name][c] == best[c] else v
            if marked and c in marked.get(name, ()):
                v += "$^\\dagger$"
            cells.append(v)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview-dir", required=True, help="the gold preview's results (B_gold_csv, C_gold_relations_csv)")
    ap.add_argument("--out", default=str(HERE / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    common = ("Qwen3.5-2B zero-shot, thinking off, 16 video frames (uniform among VGGT's 128), greedy decoding, the "
              "final answer scored; VSI-Bench scores $\\times 100$, relative direction as the mean of its easy, medium "
              "and hard types. Best per column in bold. These scores are not comparable with the trained models' "
              "table, which uses other frames and other questions.")

    ex = [int(x) for x in (HERE / "explore200_ids.txt").read_text().split()]
    cot = REPO / "results/zeroshot/vsibench/structured_cot/explore200"
    a = [("Video only", summarize(load(cot / "A_video.jsonl", ex))),
         ("Video + gold sheet", summarize(load(Path(args.preview_dir) / "B_gold_csv.jsonl", ex))),
         ("Video + gold sheet + relations", summarize(load(Path(args.preview_dir) / "C_gold_relations_csv.jsonl", ex)))]
    (out / "table_explore200_gold.tex").write_text(table(a, (
        "Preliminary: the gold (VSI-Bench-labelled) scene sheet on 200 VSI-Bench questions (explore200: 25 per task "
        "from 25 scenes). " + common + " Relations: closest-point distances in metres between the named objects' "
        "boxes, which state the answer to most distance questions. These results will be replaced by the run on the "
        "full benchmark (5,130 questions)."), "tab:gold-sheet-explore200"))

    b2 = [int(x) for x in (HERE / "batch2_A_ids.txt").read_text().split()]
    video, sheet = load(HERE / "results/final2/A_video.jsonl", b2), load(HERE / "results/final2/pipeline_v4.jsonl", b2)
    no_video = load(HERE / "results/v4_sheet_only/pipeline_v4_no_video.jsonl", b2)
    if not (no_video["n_frames"] == 0).all():
        raise SystemExit("pipeline_v4_no_video: a question was answered with frames")
    b = [("Video only", summarize(video)), ("Video + v4 sheet", summarize(sheet)),
         ("v4 full method", summarize(full_method(video, sheet))), ("v4 sheet only (no video)", summarize(no_video))]
    (out / "table_batch2A_v4.tex").write_text(table(b, (
        "The pipeline's scene sheet (pipeline\\_v4) on 798 VSI-Bench questions no earlier experiment used (batch2\\_A: "
        "every question of 38 scenes). " + common + " The full method answers the tasks marked $^\\dagger$ "
        "(absolute and relative distance, relative direction, route planning) with video + v4 sheet and the others "
        "with video only, as pre-registered. The sheet-only row was run afterwards and is exploratory."),
        "tab:pipeline-v4-batch2A", marked={"v4 full method": ROUTED}))
    for name, rows in (("table_explore200_gold.tex", a), ("table_batch2A_v4.tex", b)):
        print(f"{name}:")
        for row, s in rows:
            print(f"  {row:32s} " + " ".join(f"{s[c]:5.1f}" for c in ["overall"] + list(TASKS)))


if __name__ == "__main__":
    main()

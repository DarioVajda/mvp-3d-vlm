"""EXPLORATORY: pipeline_v4's sheet without the video on batch2_A (v4_sheet_only.ipynb), beside the two registered runs
of final_test2 (A_video, pipeline_v4), question by question. Not part of prereg_batch2_A.md: run after its results
were known, so it tests nothing registered. Writes v4_sheet_only.md next to this file.

  PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/v4_sheet_only_summary.py
"""
from pathlib import Path

import pandas as pd

from mvp.benchmarks.vsibench import TASKS, summarize
from mvp.reporting.compare_runs import paired, sign_test

HERE = Path(__file__).parent
IDS = [int(x) for x in (HERE / "batch2_A_ids.txt").read_text().split()]
RUNS = {"A_video": HERE / "results/final2/A_video.jsonl", "pipeline_v4": HERE / "results/final2/pipeline_v4.jsonl",
        "pipeline_v4_no_video": HERE / "results/v4_sheet_only/pipeline_v4_no_video.jsonl"}
NAMES = {"A_video": "video only", "pipeline_v4": "video + v4 sheet", "pipeline_v4_no_video": "v4 sheet only"}
TASK_NAMES = {"object_counting": "Count", "object_abs_distance": "Abs. dist.", "object_size_estimation": "Obj. size",
              "room_size_estimation": "Room size", "object_rel_distance": "Rel. dist.",
              "object_rel_direction": "Rel. dir.", "route_planning": "Route", "obj_appearance_order": "Order"}
COMPARISONS = [("pipeline_v4", "pipeline_v4_no_video"), ("A_video", "pipeline_v4_no_video"), ("A_video", "pipeline_v4")]


def load(path):
    run = pd.read_json(path, lines=True).drop_duplicates("id").set_index("id").sort_index()
    if set(run.index) != set(IDS):
        raise SystemExit(f"{path}: {len(set(IDS) - set(run.index))} questions missing, {len(set(run.index) - set(IDS))} extra")
    return run


def main():
    runs = {name: load(path) for name, path in RUNS.items()}
    if not (runs["pipeline_v4_no_video"]["n_frames"] == 0).all():
        raise SystemExit("pipeline_v4_no_video: a question was answered with frames")
    scores = {name: summarize(run.reset_index()) for name, run in runs.items()}
    tasks = [t for t in TASKS if t in scores["A_video"]]
    lines = ["# pipeline_v4's sheet without the video on batch2_A (exploratory)", "",
             "**Exploratory, not pre-registered.** `prereg_batch2_A.md` registered `A_video` and `pipeline_v4` on these "
             "798 questions (`results/final2/`); `pipeline_v4_no_video` was added after their results were known, so "
             "nothing here tests a registered hypothesis, and the p-values are not corrected for multiple comparisons.",
             "", "- **`pipeline_v4_no_video`**: pipeline_v4's sheet, as final2 gave it (`v4_eval.py`, b381205's "
             "renderer and the v4 relations), and the question, without the video (`--no-video`): final2's prompt less "
             "the video and its first line. Same model (Qwen3.5-2B, revision `15852e8`), thinking off, up to 2,048 new "
             "tokens, final answer scored. Code `e60f467` (the commit that fixed `--no-video`'s progress line), Colab, "
             "one Tesla T4; final2's GPU is not recorded. `results/v4_sheet_only/`, `v4_sheet_only.ipynb`.", "",
             "## Scores", "", "VSI-Bench's scores ×100 (`summarize`; relative direction's three types as one task; "
             "overall: the mean of the 8 tasks).", "",
             "| Condition | Overall | " + " | ".join(TASK_NAMES[t] for t in tasks) + " |",
             "|---|---:|" + "---:|" * len(tasks)]
    for name in RUNS:
        lines.append(f"| {NAMES[name]} (`{name}`) | {scores[name]['overall']:.1f} | "
                     + " | ".join(f"{scores[name][t]:.1f}" for t in tasks) + " |")
    lines += ["", "## Question by question", "",
              "Better / worse: questions where the second condition scores higher / lower than the first (numeric "
              "answers by VSI-Bench's partial score); exact two-sided sign test on those, uncorrected.", "",
              "| Comparison | Overall diff | Better / worse / same | p |", "|---|---:|---:|---:|"]
    for ref, other in COMPARISONS:
        better, worse, same = paired(runs[ref]["score"], runs[other]["score"])
        lines.append(f"| {NAMES[other]} − {NAMES[ref]} | {scores[other]['overall'] - scores[ref]['overall']:+.1f} | "
                     f"{better} / {worse} / {same} | {sign_test(better, worse):.3g} |")
    lines += ["", "The last row is shown for context only: the registered analysis of final2 is its own write-up "
              "(`pipeline_tests.md`), not this sign test.", "",
              "Per task, v4 sheet only − video + v4 sheet (what removing the video changes):", "",
              "| Task | n | Video + v4 sheet | v4 sheet only | Diff | Better / worse / same | p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    task = runs["A_video"]["task"]
    for t in tasks:
        m = task == t
        a, b = runs["pipeline_v4"]["score"][m], runs["pipeline_v4_no_video"]["score"][m]
        better, worse, same = paired(a, b)
        lines.append(f"| {TASK_NAMES[t]} | {int(m.sum())} | {scores['pipeline_v4'][t]:.1f} | "
                     f"{scores['pipeline_v4_no_video'][t]:.1f} | {scores['pipeline_v4_no_video'][t] - scores['pipeline_v4'][t]:+.1f} | "
                     f"{better} / {worse} / {same} | {sign_test(better, worse):.3g} |")
    nv = runs["pipeline_v4_no_video"]
    lines += ["", "## Input tokens", "",
              "`pipeline_v4_no_video` records each question's own input length (`n_prompt_tokens`); final2's runs "
              "record only the padded length of each batch of up to 8 questions of one video (`n_input_tokens`), so "
              "their numbers are upper bounds.", "",
              "| Condition | Tokens | Mean | Median | Min–max |", "|---|---|---:|---:|---:|"]
    for name, run in runs.items():
        col = "n_prompt_tokens" if "n_prompt_tokens" in run else "n_input_tokens"
        lines.append(f"| {NAMES[name]} | `{col}` | {run[col].mean():.0f} | {run[col].median():.0f} | "
                     f"{run[col].min()}–{run[col].max()} |")
    lines += ["", f"Replies cut off at 2,048 tokens: {int(nv['hit_token_limit'].sum())} of {len(nv)} "
              "(`pipeline_v4_no_video`)."]
    out = HERE / "v4_sheet_only.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Input tokens per condition in the runs already in the repository (Qwen3.5-2B, thinking off, 16 frames uniform128,
direct answers): video only, video + gold sheet, video + gold sheet + relations, video + pipeline v4 sheet. Each sheet
condition is set beside the video-only run of the same questions, which shows the same frames, so the per-question
difference is what the sheet (and its relations) cost. These runs recorded only n_input_tokens, the padded length of
each question's batch (up to 8 questions of one video), so a question's count is its batch's longest prompt: the
means overstate, the sheet's cost most. Runs that record n_prompt_tokens (each question's own length) use it.
Writes token_cost.md next to this file.

  python experiments/2026-11-19-gold-sheet-relations/token_cost.py
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SSF = REPO / "experiments/2026-11-18-scene-sheet-formats/results"
COT = REPO / "results/zeroshot/vsibench/structured_cot/explore200"
# (question set, what it is, [(condition, file)]); the first file of each set is video only
SETS = [
    ("explore200", "structured-cot's 200 explore200 questions (25 scenes; gold sheets)",
     [("video only", COT / "A_video.jsonl"), ("video + gold sheet", COT / "gold_csv.jsonl"),
      ("video + gold sheet + relations", COT / "gold_relations.jsonl")]),
    ("dev200_v4", "dev200's 50 questions run with pipeline v4",
     [("video only", SSF / "dev200_v4/A_video.jsonl"), ("video + pipeline v4 sheet", SSF / "dev200_v4/pipeline_v4.jsonl")]),
    ("batch2_A (final2)", "the 798 questions of the registered pipeline v4 test",
     [("video only", SSF / "final2/A_video.jsonl"), ("video + pipeline v4 sheet", SSF / "final2/pipeline_v4.jsonl")]),
]


def load(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def tokens(r):
    return r.get("n_prompt_tokens", r["n_input_tokens"])


def stats(values):
    return statistics.mean(values), statistics.median(values), min(values), max(values)


def main():
    lines = ["# Input tokens per condition", "",
             "Written by `token_cost.py` from the predictions in the repository: the whole input (system prompt, video and "
             "text) as the eval sends it. All runs: Qwen3.5-2B, thinking off, 16 frames `uniform128`, direct answers. "
             "**Added by the sheet**: the per-question difference from video only on the same questions and frames. "
             "**These runs record only `n_input_tokens`, the padded length of each batch** (up to 8 questions of one "
             "video), so each count is the longest prompt in its batch: the numbers are upper bounds, and the sheet's "
             "cost is overstated more than the video's (about 2,500 tokens, the same in every prompt). The eval now "
             "also records each question's own length (`n_prompt_tokens`), which the coming runs (A–E of this "
             "experiment, and pipeline v4 without the video) will give. No sheet-only run exists yet.", "",
             "| Questions | n | Condition | Input tokens, mean | Median | Min–max | Added by the sheet, mean | Median |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, _, runs in SETS:
        base = load(runs[0][1])
        for cond, path in runs:
            run = load(path)
            if set(run) != set(base):
                raise SystemExit(f"{path}: not the questions of {runs[0][1]}")
            if any(run[q]["frame_indices"] != base[q]["frame_indices"] for q in run):
                raise SystemExit(f"{path}: other frames than {runs[0][1]}")
            mean, median, lo, hi = stats([tokens(r) for r in run.values()])
            added = [tokens(run[q]) - tokens(base[q]) for q in run]
            extra = (f"{statistics.mean(added):.0f} | {statistics.median(added):.0f}" if path != runs[0][1] else "– | –")
            lines.append(f"| {name} | {len(run)} | {cond} | {mean:.0f} | {median:.0f} | {lo}–{hi} | {extra} |")
    lines += ["", "Question sets:", ""] + [f"- **{name}:** {what}; `{runs[1][1].relative_to(REPO).parent}`"
                                           for name, what, runs in SETS]
    lines += ["", "The format experiment's 2,728-question runs are not in the repository; its `results.csv` gives only "
              "the sheet's mean tokens per format (gold csv 215.4, pipeline csv 469)."]
    out = HERE / "token_cost.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

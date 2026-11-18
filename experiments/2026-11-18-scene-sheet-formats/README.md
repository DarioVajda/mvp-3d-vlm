---
id: 2026-11-18-scene-sheet-formats    # also the folder name: experiments/<id>/
title: Which scene-sheet format helps Qwen3.5-2B most on VSI-Bench, against video alone
owner: Tamara Grujičić, Nicolás Ramos Fernandez
created: 2026-11-18
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, prompting, format]
varies: [scene_sheet.format]  # keys that differ between conditions
depends_on: [scene sheets of the 75 exported scenes (docs/scene_sheets_75.md, src/mvp/openvocab_adapter/hpc/run_sheets.sh)]
---

<!--
How to use this template
- Copy it to experiments/<id>/README.md, where <id> is YYYY-MM-DD-<short-kebab-slug> with the day it was written, as
  for tasks. Code that only this experiment uses (a sheet renderer, a prompt file, an analysis script) goes in the
  same folder.
- Keep every section and every Configuration key, so that all experiments record the same fields. Write n/a where one
  doesn't apply.
- Settle the question, hypothesis, conditions, comparisons and primary metric before the first run. Any later change
  goes in the Log, with the reason.
- Runs on the cluster are tasks for Dario (tasks/README.md): one task per run or group of runs, listed under Runs.
- Comments like this one are guidance. They don't show when the file is rendered, so deleting them is optional.
-->

# Which scene-sheet format helps Qwen3.5-2B most on VSI-Bench

**Outcome:** <!-- one line once the experiment is done: the answer and its headline number -->

## Question

<!-- What we want to learn, in one or two sentences. -->

Which way of writing the scene sheet in the prompt (csv, markdown_table, json, markdown_kv or text) improves
Qwen3.5-2B's VSI-Bench score the most over the video alone? The sheet's content is the same in every format; only
how it is written changes.

## Hypothesis and what would change our mind

- **Expected result:** every format beats video only (A), with the largest gains on the distance, direction and size
  questions (absolute distance, relative distance, relative direction, object size). The differences between formats
  are smaller than the gain from adding a sheet at all.
- **Result that would prove the hypothesis wrong:**
  - no format beats A overall, or the formats help only on question types other than distance, direction and size; or
  - the spread between the best and the worst format is as large as, or larger than, the best format's gain over A
    (format matters as much as the sheet itself).
- **What we do next in each case:**
  - *Supported:* use the best format in the next scene-sheet experiments (e.g. `experiments/scene_sheet_vs_video.md`).
    If several formats are within the smallest difference that matters of the best, take the one with the fewest
    tokens (csv, see the token table).
  - *No format helps:* the problem is more likely the sheet's content than its format. Check the pipeline metrics
    below (empty sheets, split objects) and run the same comparison with the gold sheet, before changing the format.
  - *Format matters as much as the sheet:* keep the best format, and add a second model or seed before building on
    the result, since a format effect that large may not transfer.

## Background

<!-- What is already known: the results this builds on (numbers and paths, e.g. the README's tables), design notes in
docs/, earlier experiments, and papers in papers/ (file, page and verbatim quote). Mark anything that no result or
paper backs as an assumption. -->

- **Baseline model.** Qwen3.5-2B, thinking off, 16 frames: **43.5** overall on the full VSI-Bench (README, "Qwen3.5
  (VSI-Bench, 16 uniform frames)"; `results/zeroshot/vsibench/qwen3_5/qwen3_5_2B_f16.jsonl`). That run used
  `sampling: qwen` (16 frames over the whole video, per its `_summary.json`), not the sheets' frames, so it is not
  reused as condition A.
- **The sheets.** One pipeline sheet (VGGT + SAM 3 + tracking on the 16 `uniform128` frames) for each of the 2,728
  questions of the 75 exported scenes; all were made and passed their checks (`docs/scene_sheets_75.md`). Their
  quality is limited: on the 223 counting questions, the sheet has the right number of rows of the asked class for 38,
  too many for 97, too few for 52 and none for 36, because one object is often split into several rows (same doc).
  208 of the 2,728 pipeline sheets have no objects.
- **Units.** Pipeline positions and sizes are in VGGT's relative units, not metres, with the origin at the first
  frame's camera (x right, y down, z forward); which way is up in the room can't be read from the sheet
  (`docs/scene_sheet_plan.md`, "Units and scale"; checked in `src/mvp/benchmarks/scene_sheet.py`). So the metric
  questions (absolute distance, object size, room size) can't be answered from the sheet's numbers alone.
- **Format choice.** *Assumption:* the five formats were chosen from earlier reading on how LLMs read structured data
  in prompts. TODO, not blocking the runs: add the sources (papers in `papers/`, page and verbatim quote), or keep
  this marked as an assumption.
- **Tokens.** The formats differ a lot in length; see the token table under Metrics.

## Design

### Conditions

<!-- One row per condition, and one column per key in `varies`. Every other setting is in Configuration.
Role: baseline, control, treatment, upper bound (e.g. a perfect sheet) or lower bound (e.g. neither video nor sheet,
which shows what the question text alone gives). Existing run: an earlier .jsonl reused as this condition, only if it
was run with exactly this configuration; compare on the questions both runs share.

Example, from docs/scene_sheet_plan.md (varies: [visual.video, scene_sheet.source]):
| ID | Condition | `visual.video` | `scene_sheet.source` | Role | Existing run |
|---|---|---|---|---|---|
| A | Video only | true | none | baseline | results/zeroshot/vsibench/spatial_mllm_sampling/4B_uniform16.jsonl |
| B | Sheet only | false | pipeline | treatment | |
| C | Video + sheet | true | pipeline | treatment | |
| D | Video + perfect sheet | true | oracle | upper bound | |
-->

A has no sheet (`scene_sheet.source: none`); B–F have the pipeline sheet.

| ID | Condition | `scene_sheet.format` | Role | Existing run |
|---|---|---|---|---|
| A | Video only | none | baseline | none: a new run (the existing Qwen3.5-2B run used `sampling: qwen`) |
| B | Video + sheet, csv | csv | treatment | |
| C | Video + sheet, markdown table | markdown_table | treatment | |
| D | Video + sheet, json | json | treatment | |
| E | Video + sheet, markdown key-value | markdown_kv | treatment | |
| F | Video + sheet, text | text | treatment | |

The same question rendered in every format is in `examples/` (question 706: `706_pipeline_<format>.txt`).

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run, e.g. "C − A: does the
sheet add anything to the video?", "D − C: how much do segmentation and VGGT errors cost?" -->

- **B − A, C − A, D − A, E − A, F − A:** how much each format adds to the video. The best format is the one with the
  largest gain; this answers the question.
- **Tables vs key-value vs text:** {B, C} (csv, markdown_table) vs {D, E} (json, markdown_kv) vs F (text), each
  group's per-question score being the mean over its formats.
- Per question type for all of the above, to test the hypothesis that the gain is largest on distance, direction and
  size questions.

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source, as
in docs/idea.md. -->

| Decision | Choice | Why |
|---|---|---|
| One experiment, six conditions | A plus one condition per format, all on the same questions | the formats are compared with each other and with A question by question; separate experiments would let settings drift |
| Model | Qwen/Qwen3.5-2B, thinking off | the team's evaluation model; thinking is turned off by `QwenVL` for any Qwen3.5 model id (`src/mvp/models/qwen_vl.py`) |
| Frames | 16, `--sampling uniform128` | the sheets are built on exactly these frames (`adapter_report.json`: frames 0, 8, …, 127 of the 128); the eval refuses sheets with other sampling |
| Questions | the 2,728 questions of the 75 exported scenes, empty sheets included | the only questions with sheets; dropping empty sheets would make the formats' question sets depend on the pipeline |
| Sheet source | pipeline only | this experiment is about format; pipeline vs gold is `experiments/scene_sheet_vs_video.md` |
| Same content in every format | one canonical object list, five renderers; checked by round trip on every question | so the format is the only difference (`src/mvp/benchmarks/tests/test_scene_sheet.py`) |
| Fields | id (`chair_1`, spaces as `_`), label, x, y, z, size_x, size_y, size_z, first_seen, num_frames; no volume, no cameras | the sheet's per-object content; `size` (volume) is off, behind a flag (`load_objects(volume=...)`) |
| Rounding | 3 decimals, no negative zero | pipeline extents have a median of 0.245 units, so 2 decimals lost about 2% at the median and turned 471 values into 0.00 |
| first_seen | the timestamp the video input shows for that frame, in seconds | the model sees `<t seconds>` labels, not frame numbers; one label per pair of frames, as Qwen's processor computes it (`frame_timestamps`, checked against transformers 5.20.1 on all 75 scenes) |
| Placement | after "These are frames of a video." and before the question | `build_prompt` in `src/mvp/benchmarks/vsibench.py`; without a sheet the prompt is unchanged |

## Configuration

<!-- The settings every condition shares. A key in `varies` has the value `varies` here, and its values in the
Conditions table. The values filled in are the code's defaults and the scene-sheet baseline (docs/scene_sheet_plan.md,
condition A: Qwen3-VL-4B, 16 frames uniform over the 128 VGGT frames, 51.9 on VSI-Bench): change them to what was
actually run. When an experiment changes something that has no key, add the key here and to experiments/TEMPLATE.md,
so that later experiments record it too. -->

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-18-scene-sheet-formats/question_ids.txt  # the 2,728 ids of index.csv
  scenes: tasks/vggt_data/scenes.csv  # 75 exported scenes
  question_types: all          # all 10
  n_questions: 2728            # empty sheets included; the same questions in every condition

visual:
  video: true
  model: Qwen/Qwen3.5-2B       # TODO, not blocking the runs: record the revision of the cluster's cached snapshot
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:                   # A has none; B–F as below
  source: pipeline             # none in A
  geometry: saved VGGT         # data/vggt, 128 frames
  frames: subset:uniform16
  sam3_prompts: full           # prompt_mode "full" in all 2,728 question.json files
  sam3: {backend: transformers, model: facebook/sam3@3c879f3, min_score: 0.5}
  tracking: {reid: true, conf_percentile: 55, min_points: 300}  # question.json of the sheets
  content: [position, size, first_seen, num_frames]
  units: relative              # VGGT's scale, said in the sheet's preamble
  format: varies
  renderer: src/mvp/benchmarks/scene_sheet.py:render_sheet
  placement: between video and question  # after "These are frames of a video.", before the question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval          # build_prompt in src/mvp/benchmarks/vsibench.py, with the sheet inserted
  reasoning: direct
  thinking: off                # enable_thinking=False
  max_new_tokens: 16
  decoding: greedy
  answer_extraction: lmms-eval

training: none
```

## Metrics and analysis

- **Primary metric:** VSI-Bench overall (lmms-eval's macro average over the 8 tasks: accuracy for multiple choice,
  mean relative accuracy for numeric answers) on the 2,728 questions, for each condition.
- **Secondary metrics:** score per question type; parse rate, the share of answers the extraction rule could read
  (an option letter for multiple choice, a number for numeric questions; `parsed` in `src/mvp/benchmarks/vsibench.py`,
  printed per run by `src/mvp/reporting/compare_runs.py`); input tokens per question (`n_input_tokens` in each
  `.jsonl`); the debiased overall on the **1,249** of the 2,728 questions that are in VSI-Bench's debiased split
  (`test_debiased.parquet`, 2,362 questions; counted 2026-11-18), computed as in `src/mvp/reporting/tables.py`
  (`summarize` on the debiased ids). Per type, the 1,249 are: appearance order 255, absolute distance 224, object
  size 165, relative direction easy 152 / medium 36 / hard 80, relative distance 147, counting 102, room size 54,
  route planning 34. Sheet tokens per format, with the Qwen3.5-2B tokenizer (`python -m mvp.benchmarks.scene_sheet --tokens Qwen/Qwen3.5-2B`, run
  2026-11-18):

  | format | mean | median | max |
  |---|---|---|---|
  | csv | 468 | 390 | 2,202 |
  | markdown_table | 562 | 466 | 2,620 |
  | json | 766 | 607 | 4,075 |
  | markdown_kv | 691 | 547 | 3,637 |
  | text | 707 | 560 | 3,731 |

- **Pipeline metrics:** the sheets are the same in B–F, so these are reported once: 208 questions with an empty
  sheet; objects per sheet (median 5, max 41); the counting check of `docs/scene_sheets_75.md`; `prompt_mode` is
  `full` for all 2,728. SAM 3 prompts against the objects the question names, named objects with at least one row,
  and the answer a fixed rule reads off the sheet alone: TODO, not blocking the runs (not computed yet).
- **Breakdowns:** question type, always. Also: dataset (ScanNet 1,177, ScanNet++ 882, ARKitScenes 669 questions);
  metric tasks (absolute distance, object size, room size) vs the rest, since the units are relative; relative
  direction easy, medium and hard; empty vs non-empty sheets.
- **Statistics:** every condition on the same 2,728 questions, compared question by question with A: how many get
  better and worse, with an exact two-sided sign test (ties left out), as in the README's space-aware vs uniform
  table. The five format − A comparisons are corrected with Holm's method, at α = 0.05. Both are in
  `src/mvp/reporting/compare_runs.py` (`sign_test`, `holm`; it reproduces the README's 4B 16-frame row, 941 better /
  888 worse, p = 0.22):
  `python -m mvp.reporting.compare_runs --reference results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_A_video.jsonl results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_*.jsonl`.
  The table vs key-value vs text comparison is descriptive (no test).
- **Smallest difference that matters:** 2 points on the overall, 5 points on a single question type. A format "helps"
  if it beats A by at least that much and its Holm-corrected p is below 0.05; two formats closer than that are treated
  as equal, and the one with fewer tokens is preferred.

## Threats to validity and checks

<!-- Keep the rows that apply and add this experiment's own. Problems with running (memory, time limits) go in the
task files. -->

| Risk | How we'd notice | What to do |
|---|---|---|
| The 75-scene subset: 2,728 questions, chosen for many question types, so it leans towards larger scenes (`tasks/vggt_data/README.md`); some types are small (route planning 58, room size 75) | wide intervals for the small types | report per-type counts next to scores; read the small types as indicative only |
| One seed and one model: greedy decoding has no sampling noise, but a format effect may not transfer to another model | n/a within this experiment | state it as a caveat; repeat the best and worst format on a second model before building on a small format effect |
| Timestamps in the sheet don't match the video's: the sheet uses the VGGT export's fps (decord's average fps), while the eval reads the video with torchcodec if installed, else decord, else torchvision | the `<t seconds>` labels in a printed prompt differ from the sheet's `first_seen` values | run the fps check under Runs before the full runs |
| Empty sheets (208) dilute the format effect | empty-sheet questions score the same in B–F | report the breakdown by empty vs non-empty sheets |
| Split objects: the sheet often lists one object several times | low counting scores in B–F | read counting separately; the pipeline metrics show how often |
| Format effect is confounded with length: json, markdown_kv and text are about 1.5–1.6× longer than csv on average | gains that track token counts | report tokens with each score |
| Linear-attention kernels: Qwen3.5 gives slightly different predictions with and without them (README: 89.6% identical) | n/a | run all six conditions in the same environment, with the kernels |

## Runs and cost

<!-- One row per run. A run can serve several conditions (one sheet job for B and C, say). Output: the folder with the
predictions (.jsonl), the summaries and the run's freeze.txt or environment.txt. -->

All runs from the repository root on the cluster, at the commit that has `src/mvp/benchmarks/scene_sheet.py`. The
sheets are those of `run_sheets.sh`, in `tasks/openvocab_adapter/out/`. Common arguments:
`--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --ids experiments/2026-11-18-scene-sheet-formats/question_ids.txt`.

| Run | Conditions | Task or command | Output | Commit |
|---|---|---|---|---|
| A | A | `slurm/submit.sh -J ssf_A -t 01:00:00 -- .venv/bin/python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --ids experiments/2026-11-18-scene-sheet-formats/question_ids.txt --out results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_A_video.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_A_video.jsonl | TODO |
| B | B | as A, plus `--sheet-source pipeline --sheet-format csv --sheets-root tasks/openvocab_adapter/out`, `-J ssf_B`, `--out .../qwen3_5_2B_B_csv.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_B_csv.jsonl | TODO |
| C | C | as B with `--sheet-format markdown_table`, `-J ssf_C`, `--out .../qwen3_5_2B_C_markdown_table.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_C_markdown_table.jsonl | TODO |
| D | D | as B with `--sheet-format json`, `-J ssf_D`, `--out .../qwen3_5_2B_D_json.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_D_json.jsonl | TODO |
| E | E | as B with `--sheet-format markdown_kv`, `-J ssf_E`, `--out .../qwen3_5_2B_E_markdown_kv.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_E_markdown_kv.jsonl | TODO |
| F | F | as B with `--sheet-format text`, `-J ssf_F`, `--out .../qwen3_5_2B_F_text.jsonl` | results/zeroshot/vsibench/scene_sheet_formats/qwen3_5_2B_F_text.jsonl | TODO |

Cluster checks, before the full runs:

1. **The VGGT metadata is there:** `ls data/vggt/*/*/meta.json | wc -l` (288 expected, one per VSI-Bench video), and
   each of the 75 scenes in `tasks/vggt_data/scenes.csv` has one with `video_fps`. `render_sheet` reads the fps from
   there and stops with an error if a file is missing.
2. **The timestamps match:** print the full prompts of four questions without loading the weights,

   ```bash
   slurm/submit.sh -J ssf_prompt -t 00:20:00 -- .venv/bin/python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B \
       --nframes 16 --sampling uniform128 --sheet-source pipeline --sheet-format text \
       --sheets-root tasks/openvocab_adapter/out --ids experiments/2026-11-18-scene-sheet-formats/prompt_check_ids.txt \
       --print-prompt 4
   ```

   and check that every `first_seen` in each sheet is one of that prompt's `<t seconds>` labels, and that the printed
   video fps is the one in `data/vggt/<scene>/meta.json`. The four (`prompt_check_ids.txt`) are 706 and 680
   (ARKitScenes, 30 fps; 680's scene has a pair at exactly 139.65 s, where rounding is most fragile), 4853 (ScanNet,
   24 fps) and 1776 (ScanNet++, 30 fps). This is also the first run of `render_prompt` (`src/mvp/models/qwen_vl.py`),
   which could not be run locally.
3. **Smoke test:** each condition on 20 questions (`--ids` with 2 per question type), checking that predictions parse
   and that B–F's `n_input_tokens` exceed A's by about the sheet's token count.

**Cost:** about 1–1.5 GPU hours in total (*estimate*). The README times Qwen3.5-4B, 16 frames, on the full VSI-Bench
(5,130 questions) at about 18 min on one H100 with the kernels; there is no timing for the 2B. Scaled to 2,728
questions that is about 10 min per run as an upper bound for the 2B, plus the sheet's tokens (median 390–607 per
question). Six runs: about 1–1.5 GPU hours. No sheets are built (they exist) and nothing is trained.

## Results


### Examples

-

### What the results show

-

## Conclusion

- **Answer to the question:**
- **Hypothesis:** <!-- supported, rejected or inconclusive, and why -->
- **Caveats:** <!-- what limits the conclusion: subset size, one seed, one model -->
- **Next:** <!-- the decision this leads to, and follow-up experiments by id -->

## Log

<!-- Dated changes to the plan, reruns and bugs found, each with its reason. -->

- 2026-11-18: wrote the renderers (`src/mvp/benchmarks/scene_sheet.py`: csv, json, markdown_kv, markdown_table, text;
  3 decimals, `first_seen` timestamps), their checks (`src/mvp/benchmarks/tests/test_scene_sheet.py`, all passing),
  the eval options (`--sheet-source`, `--sheet-format`, `--sheets-root` in `src/mvp/eval/vsibench.py`), the token
  table and this plan. Added markdown_kv and markdown_table to the template's format values and first_seen to its
  content options. Fixed before the first run: Holm over the five format − A comparisons; smallest difference that
  matters 2 points overall, 5 per question type. Added `--print-prompt` to the eval, the sign test, Holm and parse
  rate to `compare_runs.py` (with tests), and counted the debiased questions (1,249 of 2,728). Open TODOs, none of
  which blocks the runs: (1) the revision of the cluster's cached Qwen/Qwen3.5-2B snapshot (Configuration,
  `visual.model`); (2) sources for the choice of formats (Background); (3) the pipeline metrics not computed yet
  (Metrics); (4) the Commit column under Runs, filled in when the runs are submitted.

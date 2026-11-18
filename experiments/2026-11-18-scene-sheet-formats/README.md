---
id: 2026-11-18-scene-sheet-formats    # also the folder name: experiments/<id>/
title: Which scene-sheet format helps Qwen3.5-2B most on VSI-Bench, against video alone, with pipeline and gold sheets
owner: Tamara Grujičić, Nicolás Ramos Fernandez
created: 2026-11-18
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, prompting, format]
varies: [scene_sheet.source, scene_sheet.format]  # keys that differ between conditions
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
how it is written changes. Each format is run with the pipeline's sheet and with the gold sheet, so that the effect of
the format can be told apart from the quality of the sheet's content.

## Hypothesis and what would change our mind

- **Expected result:** with gold sheets (G–K), every format beats video only (A), with the largest gains on the
  distance, direction and size questions (absolute distance, relative distance, relative direction, object size), and
  the differences between formats are clearer than with pipeline sheets. With pipeline sheets (B–F), the gains over A
  may be small or absent: `sheet_quality.md` finds that only 49% of questions have every named object in the pipeline
  sheet, and that 40% of the objects VSI-Bench treats as unique have several rows. In both sources, the differences
  between formats are smaller than the gain from adding a sheet at all.
- **Result that would prove the hypothesis wrong:**
  - no gold format beats A overall, or the gold formats help only on question types other than distance, direction
    and size; or
  - the spread between the best and the worst format is as large as, or larger than, the best format's gain over A
    (format matters as much as the sheet itself).
- **What we do next in each case:**
  - *Supported:* use the best format in the next scene-sheet experiments (e.g. `experiments/scene_sheet_vs_video.md`).
    If several formats are within the smallest difference that matters of the best, take the one with the fewest
    tokens (csv, see the token table).
  - *No pipeline format helps, but gold formats do:* the problem is the sheet's content, not its format; improve
    the pipeline (re-identification, missed small objects; `sheet_quality.md`) before changing the format.
  - *No format helps even with gold sheets:* the model doesn't use the sheet as written; try reasoning before the
    answer or a different placement before more format variants.
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
  208 of the 2,728 pipeline sheets have no objects. The full comparison with the gold sheets is in
  [`sheet_quality.md`](sheet_quality.md) (made by `sheet_quality.py`): 73% of the objects the questions name have a
  pipeline row (97% in gold), 69% of pipeline rows are seen in only one of the 16 frames, and counts match gold in
  28% of (question, class) pairs.
- **Units.** Pipeline positions and sizes are in VGGT's relative units, not metres, with the origin at the first
  frame's camera (x right, y down, z forward); which way is up in the room can't be read from the sheet
  (`docs/scene_sheet_plan.md`, "Units and scale"; checked in `src/mvp/benchmarks/scene_sheet.py`). So the metric
  questions (absolute distance, object size, room size) can't be answered from the sheet's numbers alone.
- **Format choice.** The five formats come from work on how the way structured data is written in a prompt
  affects LLMs:
  - **Prompt format matters:** LLM accuracy is sensitive to prompt formatting (Sclar et al. 2024, ICLR; He et al.
    2024, arXiv 2411.10541).
  - **Table formats:** CSV is the most token-efficient, while Markdown and HTML are often more accurate (Sui et al.
    2024, WSDM, "Table Meets LLM"; Qiu et al. 2024, TQA-Bench, arXiv 2411.19504). These give csv and markdown_table.
  - **Markdown key-value records:** from the Improving Agents benchmark (blog, 2025). This gives markdown_kv.
  - **Natural-language templates:** from TabLLM (Hegselmann et al. 2023). This gives text.
  - **3D objects as text on VSI-Bench:** MLLMs have been given 3D object information as text on VSI-Bench (Yuan et
    al. 2026, GR3D, arXiv 2603.08592; Zhu et al. 2025, Struct2D).

  None of these tested the formats on a small VLM with 3D scene sheets, which is what this experiment does. TODO,
  not blocking the runs: add the PDFs to `papers/`, with page and verbatim quote for each claim.
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

A has no sheet; B–F have the pipeline sheet, G–K the gold sheet, in the same five formats.

| ID | Condition | `scene_sheet.source` | `scene_sheet.format` | Role | Existing run |
|---|---|---|---|---|---|
| A | Video only | none | none | baseline | none: a new run (the existing Qwen3.5-2B run used `sampling: qwen`) |
| B | Video + pipeline sheet, csv | pipeline | csv | treatment | |
| C | Video + pipeline sheet, markdown table | pipeline | markdown_table | treatment | |
| D | Video + pipeline sheet, json | pipeline | json | treatment | |
| E | Video + pipeline sheet, markdown key-value | pipeline | markdown_kv | treatment | |
| F | Video + pipeline sheet, text | pipeline | text | treatment | |
| G | Video + gold sheet, csv | gold | csv | upper bound for content | |
| H | Video + gold sheet, markdown table | gold | markdown_table | upper bound for content | |
| I | Video + gold sheet, json | gold | json | upper bound for content | |
| J | Video + gold sheet, markdown key-value | gold | markdown_kv | upper bound for content | |
| K | Video + gold sheet, text | gold | text | upper bound for content | |

The same question rendered in every format and source is in `examples/` (question 706:
`706_<source>_<format>.txt`).

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run, e.g. "C − A: does the
sheet add anything to the video?", "D − C: how much do segmentation and VGGT errors cost?" -->

- **B − A, C − A, D − A, E − A, F − A:** how much each format of the pipeline sheet adds to the video. The best
  format is the one with the largest gain; this answers the question for the sheets we can actually build.
- **G − A, H − A, I − A, J − A, K − A:** the same with gold sheets: how much each format can add when the content is
  right, and which format is best without pipeline errors in the way.
- **B − G, C − H, D − I, E − J, F − K:** pipeline − gold per format, the cost of pipeline errors in each format.
- **Tables vs key-value vs text:** {B, C} (csv, markdown_table) vs {D, E} (json, markdown_kv) vs F (text), each
  group's per-question score being the mean over its formats; the same for G–K.
- Per question type for all of the above, to test the hypothesis that the gain is largest on distance, direction and
  size questions.

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source, as
in docs/idea.md. -->

| Decision | Choice | Why |
|---|---|---|
| One experiment, eleven conditions | A plus one condition per format and source, all on the same questions | the formats are compared with each other and with A question by question; separate experiments would let settings drift |
| Model | Qwen/Qwen3.5-2B, thinking off | the team's evaluation model; thinking is turned off by `QwenVL` for any Qwen3.5 model id (`src/mvp/models/qwen_vl.py`) |
| Frames | 16, `--sampling uniform128` | the sheets are built on exactly these frames (`adapter_report.json`: frames 0, 8, …, 127 of the 128); the eval refuses sheets with other sampling |
| Questions | the 2,728 questions of the 75 exported scenes, empty sheets included | the only questions with sheets; dropping empty sheets would make the formats' question sets depend on the pipeline |
| Sheet source | pipeline and gold | `sheet_quality.md` shows the pipeline sheets are noisy enough to hide a format effect; gold sheets (G–K) show the format effect on correct content, and pipeline − gold per format the cost of pipeline errors |
| Empty sheet | "The detector did not find the objects named in the question. This does not mean they are absent from the room." in every format | 208 pipeline sheets are empty, 205 of them where gold has the objects (`sheet_quality.md`); the earlier "No objects were found." read as if the objects weren't there |
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
  model: Qwen/Qwen3.5-2B       # revision: the cluster's cached snapshot, recorded by the smoke test (Runs) in model_revision.txt
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:                   # A has none; B–K as below
  source: varies               # pipeline in B–F, gold in G–K, none in A
  geometry: saved VGGT         # data/vggt, 128 frames
  frames: subset:uniform16
  sam3_prompts: full           # prompt_mode "full" in all 2,728 question.json files
  sam3: {backend: transformers, model: facebook/sam3@3c879f3, min_score: 0.5}
  tracking: {reid: true, conf_percentile: 55, min_points: 300}  # question.json of the sheets
  content: [position, size, first_seen, num_frames]
  units: varies                # pipeline (B–F): VGGT's relative units; gold (G–K): metres. Said in each source's preamble
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

- **Pipeline metrics:** the sheets are the same in B–F, so these are reported once, in `sheet_quality.md`: named
  objects found per question type, the 208 empty sheets, counts and relative sizes against gold. Also: objects per
  sheet (median 5, max 41); `prompt_mode` is `full` for all 2,728. The answer a fixed rule reads off the sheet alone:
  TODO, not blocking the runs.
- **Breakdowns:** question type, always. Also: dataset (ScanNet 1,177, ScanNet++ 882, ARKitScenes 669 questions);
  metric tasks (absolute distance, object size, room size) vs the rest, since the units are relative; relative
  direction easy, medium and hard; empty vs non-empty sheets.
- **Statistics:** every condition on the same 2,728 questions, compared question by question with A: how many get
  better and worse, with an exact two-sided sign test (ties left out), as in the README's space-aware vs uniform
  table. The ten condition − A comparisons are corrected together with Holm's method, at α = 0.05; the five
  pipeline − gold comparisons are a second family, also Holm-corrected. The tests are `sign_test` and `holm` in
  `src/mvp/reporting/compare_runs.py` (they reproduce the README's 4B 16-frame row, 941 better / 888 worse,
  p = 0.22); `make_table.py` runs them all and writes the Results table. The table vs key-value vs text comparison is
  descriptive (no test).
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
| Pipeline vs gold differs in units as well as content: gold sheets are in metres (z up), pipeline sheets in VGGT's relative units (camera axes) | gold's gain over pipeline is largest on the metric questions (absolute distance, object size, room size) | read pipeline − gold on the metric questions as content plus units; the other question types compare content only |
| Noisy pipeline content hides the format effect: 51% of questions miss a named object, 40% of unique objects have several rows (`sheet_quality.md`) | B–F all close to each other and to A | compare formats on G–K too |
| Split objects: the sheet often lists one object several times | low counting scores in B–F | read counting separately; the pipeline metrics show how often |
| Format effect is confounded with length: json, markdown_kv and text are about 1.5–1.6× longer than csv on average | gains that track token counts | report tokens with each score |
| Linear-attention kernels: Qwen3.5 gives slightly different predictions with and without them (README: 89.6% identical) | n/a | run all eleven conditions in the same environment, with the kernels |

## Runs and cost

<!-- One row per run. A run can serve several conditions (one sheet job for B and C, say). Output: the folder with the
predictions (.jsonl), the summaries and the run's freeze.txt or environment.txt. -->

All commands run from the repository root on the cluster, at a commit with this folder, in the project's environment
(`.venv`, with the linear-attention kernels). The sheets of `run_sheets.sh` are in `tasks/openvocab_adapter/out/`.

**Checklist, before the runs:**

1. **meta.json exists.** Every scene needs `data/vggt/<dataset>/<scene>/meta.json` with `video_fps`, because
   `render_sheet` reads it for `first_seen`. This prints the number of scenes and any without `video_fps`, and stops
   with an error if a file is missing:

   ```bash
   python -c "import csv, json; s = {r['scene'] for r in csv.DictReader(open('tasks/openvocab_adapter/out/index.csv'))}; print(len(s), 'scenes; without video_fps:', [x for x in s if 'video_fps' not in json.load(open(f'data/vggt/{x}/meta.json'))])"
   ```

2. **Timestamp check with `--print-prompt`.** This prints the full prompts of `prompt_check_ids.txt` without loading
   the weights: 706 and 680 (ARKitScenes, 30 fps; 680's scene has a pair at exactly 139.65 s, where rounding is most
   fragile), 4853 (ScanNet, 24 fps) and 1776 (ScanNet++). The check then confirms three things: the video's
   `<t seconds>` labels are the sheet's timestamps, every `first_seen` is one of them, and each prompt is exactly
   `build_sheet_prompt`'s. This is also the first run of `QwenVL.render_prompt`.

   ```bash
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --sheet-source pipeline \
       --sheet-format csv --sheets-root tasks/openvocab_adapter/out \
       --ids experiments/2026-11-18-scene-sheet-formats/prompt_check_ids.txt --print-prompt 4 > prompts.txt
   python experiments/2026-11-18-scene-sheet-formats/cluster_checks.py prompts prompts.txt \
       --sheets-root tasks/openvocab_adapter/out
   ```

3. **20-question smoke test.** This runs A and B on `smoke_ids.txt` (2 per question type). It passes if both answer
   all 20, at least 90% of answers parse, and B's prompts are longer than A's. It then records the model's cached
   revision next to the outputs.

   ```bash
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 \
       --ids experiments/2026-11-18-scene-sheet-formats/smoke_ids.txt --out results/_scratch/ssf_smoke/A.jsonl
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 \
       --ids experiments/2026-11-18-scene-sheet-formats/smoke_ids.txt --out results/_scratch/ssf_smoke/B.jsonl \
       --sheet-source pipeline --sheet-format csv --sheets-root tasks/openvocab_adapter/out
   python experiments/2026-11-18-scene-sheet-formats/cluster_checks.py smoke results/_scratch/ssf_smoke/A.jsonl \
       results/_scratch/ssf_smoke/B.jsonl --n 20
   python experiments/2026-11-18-scene-sheet-formats/cluster_checks.py revision Qwen/Qwen3.5-2B \
       results/zeroshot/vsibench/scene_sheet_formats/model_revision.txt
   ```

**The 11 runs**, one command per condition. They are independent, so they can run in parallel, and each resumes
from its `--out` file if stopped:

```bash
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --ids experiments/2026-11-18-scene-sheet-formats/question_ids.txt"
OUT=results/zeroshot/vsibench/scene_sheet_formats
SHEETS=tasks/openvocab_adapter/out
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_A_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source pipeline --sheet-format csv            --sheets-root $SHEETS --out $OUT/qwen3_5_2B_B_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source pipeline --sheet-format markdown_table --sheets-root $SHEETS --out $OUT/qwen3_5_2B_C_markdown_table.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source pipeline --sheet-format json           --sheets-root $SHEETS --out $OUT/qwen3_5_2B_D_json.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source pipeline --sheet-format markdown_kv    --sheets-root $SHEETS --out $OUT/qwen3_5_2B_E_markdown_kv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source pipeline --sheet-format text           --sheets-root $SHEETS --out $OUT/qwen3_5_2B_F_text.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold     --sheet-format csv            --sheets-root $SHEETS --out $OUT/qwen3_5_2B_G_gold_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold     --sheet-format markdown_table --sheets-root $SHEETS --out $OUT/qwen3_5_2B_H_gold_markdown_table.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold     --sheet-format json           --sheets-root $SHEETS --out $OUT/qwen3_5_2B_I_gold_json.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold     --sheet-format markdown_kv    --sheets-root $SHEETS --out $OUT/qwen3_5_2B_J_gold_markdown_kv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold     --sheet-format text           --sheets-root $SHEETS --out $OUT/qwen3_5_2B_K_gold_text.jsonl
```

Record the commit each run was made at (`git rev-parse HEAD`) in the Log.

**Afterwards**, `PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/make_table.py` does the rest. It
checks that the 11 files answer the same 2,728 questions, then writes the Results table below, `results.csv`, and
the LaTeX tables `results_pipeline.tex` (`tab:sheet-format`) and `results_gold.tex` (`tab:sheet-format-gold`).

**To see a prompt** without the cluster, use the exact text input the eval sends for one question, with each frame
pair's video tokens as one `<|video_pad|>`:
`python -m mvp.benchmarks.scene_sheet --question 706 --source pipeline --format csv --prompt` (`--source gold`, or
`none` for A; `build_sheet_prompt` in `src/mvp/benchmarks/scene_sheet.py`).

**Cost:** about 2–2.5 GPU hours in total, for the eleven runs (*estimate*).
- **Basis:** the repository README times Qwen3.5-4B, 16 frames, on the full VSI-Bench (5,130 questions) at about 18
  min on one H100 with the kernels. There is no timing for the 2B.
- **Per run:** scaled to 2,728 questions, that is about 10 min per run as an upper bound for the 2B, before the
  sheet's tokens.
- **Sheet tokens:** the median is 390–607 per question for pipeline sheets, and gold sheets are about half as long.

The checks add a few minutes. No sheets are built (they exist) and nothing is trained.

## Results

<!-- make_table:start -->
Not run yet: `make_table.py` writes the table here.
<!-- make_table:end -->

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

- 2026-11-18: the pipeline sheets were compared with the gold sheets (`sheet_quality.md`, written by
  `sheet_quality.py`). 73.4% of the objects the questions name have a pipeline row, against 97.4% in gold, and only
  48.8% of questions have every one. 69% of pipeline rows are seen in a single frame, so large objects are split into
  several rows, and counts match gold in 28% of (question, class) pairs. 205 of the 208 empty pipeline sheets have
  gold objects, and 122 of them are object-size questions. Relative sizes are roughly right: the same largest object
  in 68% of questions, against 38% by chance. Gold misses only prompts matched to no class (route-planning
  landmarks, plus the `ceiling` + `light` and `shoe` extraction bugs). These changes follow, all before the first run:
  - Gold-sheet conditions G–K were added (upper bound for content), with `varies` now
    `[scene_sheet.source, scene_sheet.format]`. Noise at this level could hide a format effect in B–F; the same formats
    with gold content show the format effect on correct content, and pipeline − gold per format gives the cost of
    pipeline errors. Holm is applied to the ten comparisons with A as one family and to the five pipeline − gold
    comparisons as another.
  - The hypothesis now expects clearer format differences with gold sheets and possibly small gains with pipeline
    sheets, given the coverage and split objects above.
  - Threats now cover the units (gold in metres, pipeline in relative units), which make pipeline − gold a difference
    in units as well as content on the metric questions, and the noisy pipeline content.
  - The empty-sheet line changed from "No objects were found." to "The detector did not find the objects named in the
    question. This does not mean they are absent from the room.". In 205 of the 208 empty sheets the objects are in
    the room, and the old line read as if they weren't.
  - `make_table.py` (Results table, `results.csv`, LaTeX tables) and `cluster_checks.py` (prompt and timestamp check,
    smoke-test check, model revision) were added, both tested in `test_make_table.py`, so the analysis and the checks
    before the runs are scripted rather than done by hand.
  - `build_sheet_prompt(question_id, source, fmt)` was added to `src/mvp/benchmarks/scene_sheet.py`, with the CLI
    `python -m mvp.benchmarks.scene_sheet --question <id> --source pipeline|gold|none --format <fmt> --prompt`. It
    returns the exact text input the eval sends: the chat template with thinking off, the 8 `<t seconds>` labels
    with a `<|video_pad|>` per frame pair, the sheet and the question. Any condition's prompt can then be read without
    a GPU. It is tested against the Qwen3.5-2B tokenizer's chat template and against VSI-Bench's `test.jsonl` for all
    2,728 questions, and the `--print-prompt` check compares each whole prompt with it.
  - Runs lists a checklist and one command per condition, without job-system wrappers, so that each condition's
    settings are visible in the plan and the runs can be scheduled with whatever the cluster uses.

  TODO (1), the model revision, is now recorded by the smoke-test step (`model_revision.txt`), and TODO (4), the
  commit of each run, goes in this Log when the runs are made. TODO (3) is mostly resolved, since the pipeline
  metrics are in `sheet_quality.md`, but the answer a fixed rule reads off the sheet alone isn't computed yet. That
  part and TODO (2), the sources for the choice of formats, are still open, and neither blocks the runs.
- 2026-11-18: three fixes to the plan:
  - The cost estimate now covers the eleven runs only (about 2–2.5 GPU hours). The six-run figure from before G–K
    was removed.
  - `scene_sheet.units` is now `varies`, because pipeline sheets are in VGGT's relative units and gold sheets in
    metres.
  - The format choice now cites its sources instead of being marked an assumption. TODO (2) is reduced to adding the
    PDFs to `papers/` with page and verbatim quote.

---
id: YYYY-MM-DD-short-slug    # also the folder name: experiments/<id>/
title: One-line description of the experiment
owner: Name
created: YYYY-MM-DD
status: draft                # draft | planned | running | analysing | done | abandoned
tags: []                     # free labels for grouping, e.g. [scene-sheet, prompting]
varies: []                   # keys that differ between conditions, e.g. [visual.video, scene_sheet.source]
depends_on: []               # experiments or tasks that must be done first
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

# Title

**Outcome:** <!-- one line once the experiment is done: the answer and its headline number -->

## Question

<!-- What we want to learn, in one or two sentences. -->

## Hypothesis and what would change our mind

- Expected result:
- Result that would prove the hypothesis wrong:
- What we do next in each case:

## Background

<!-- What is already known: the results this builds on (numbers and paths, e.g. the README's tables), design notes in
docs/, earlier experiments, and papers in papers/ (file, page and verbatim quote). Mark anything that no result or
paper backs as an assumption. -->

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

| ID | Condition | `<key>` | `<key>` | Role | Existing run |
|---|---|---|---|---|---|
| A | | | | baseline | |
| B | | | | treatment | |

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run, e.g. "C − A: does the
sheet add anything to the video?", "D − C: how much do segmentation and VGGT errors cost?" -->

- B − A:

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source, as
in docs/idea.md. -->

| Decision | Choice | Why |
|---|---|---|
| | | |

## Configuration

<!-- The settings every condition shares. A key in `varies` has the value `varies` here, and its values in the
Conditions table. The values filled in are the code's defaults and the scene-sheet baseline (docs/scene_sheet_plan.md,
condition A: Qwen3-VL-4B, 16 frames uniform over the 128 VGGT frames, 51.9 on VSI-Bench): change them to what was
actually run. When an experiment changes something that has no key, add the key here and to experiments/TEMPLATE.md,
so that later experiments record it too. -->

```yaml
data:
  benchmark: vsibench          # vsibench | mmsibench
  split: full                  # full (5,130 questions) | debiased (2,362) | pruned (2,768)
  subset: none                 # none | data/subsets/vsi_per_type100_seed0.txt (100 per question type) | <ids file>
  scenes: all                  # all (288 videos) | tasks/vggt_data/scenes.csv (75 exported, 2,728 questions) | <file>
  question_types: all          # all 10 | a list, e.g. [object_rel_direction_hard, route_planning]
  n_questions:                 # after every filter; the same questions in every condition

visual:                        # what the VLM sees of the video
  video: true                  # false: no frames, and the prompt must not say "These are frames of a video."
  model: Qwen/Qwen3-VL-4B-Instruct  # Hugging Face id and revision, or a trained run's checkpoint
  protocol: video              # video: one Qwen video, frames merged in pairs | images: one image per frame (--images)
  nframes: 16
  sampling: uniform128         # qwen: whole video | uniform128 / sa: uniform / space-aware among the 128 VGGT frames
  layout: sit                  # sit: video, then question (lmms-eval) | sti | stit (README, "Prompt echoing")
  resolution: default          # default: as in the README's "Evaluation protocol"

scene_sheet:                   # "scene_sheet: none" instead of this block if no condition has a sheet
  source: none                 # none | pipeline | oracle
  geometry: saved VGGT         # saved VGGT (data/vggt, 128 frames) | OpenVocab-4D's own VGGT | ground truth (source)
  frames: subset:uniform16     # the VLM's frames (subset:uniform16 = 16 uniform128 frames) | all | half | uniform:K
  sam3_prompts: full           # full | syntactic (prompt_object_extractor.py) | scene (hpc/scenes.tsv) | ground truth
  sam3: {backend: transformers, model: facebook/sam3@3c879f3, min_score: 0.5}
  tracking: {reid: true, conf_percentile: 55, min_points: 300}  # question_scene_sheet.py defaults
  content: [position, size, num_frames, frame_flags]  # frame_flags or frame_positions; + cameras, pairwise distances
  units: relative              # relative: VGGT's scale, said in the sheet | metres: rescaled (how) | ground truth
  format: csv                  # csv | json | text
  renderer:                    # the code that turns the sheet into prompt text (file:function)
  placement: between video and question  # before the video | between video and question | after the question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval          # lmms-eval: build_prompt in src/mvp/benchmarks/vsibench.py | <prompt file>
  reasoning: direct            # direct | structured: fixed steps (objects -> coordinates -> compute -> answer) | free
  thinking: n/a                # Qwen3.5 only: off (enable_thinking=False) | on
  max_new_tokens: 16           # enough for direct answers; reasoning needs far more (Qwen3.5 on MMSI-Bench: 2048)
  decoding: greedy
  answer_extraction: lmms-eval # first word (multiple choice), first number (numeric) | <rule> for reasoning outputs

training:                      # "training: none" instead of this block if nothing is trained
  arm: lora                    # none | lora | additive | xattn | xattn_aligned (python -m mvp.train.run --arm)
  base_model: Qwen/Qwen3.5-2B
  data: VLM-3R ScanNet QA, cap 3000 per task  # --cap; add VSI-590K appearance order with --appearance
  sheet_in_training: false     # true: training prompts have sheets, so the training videos need sheets too
  test_overlap: none           # training scenes that are also test scenes, and how that was checked
  lora: {r: 64, alpha: 128, dropout: 0.05}
  schedule: {epochs: 10, lr: 0.0001, warmup: 0.1, min_lr: 0.1, weight_decay: 0.0}
  batch: {mb_per_step: 4, max_q: 8}  # 32 questions per step
  train_frames: 8
  seeds: [0]                   # --init-seed per rerun; --seed fixes the training subsample and dev split
  checkpoint: best dev         # python -m mvp.train.select_best: highest dev accuracy (60 held-out videos)
  args: results/train/<run>/args.json  # every argument of the run
```

## Metrics and analysis

- **Primary metric:** <!-- one, fixed before the first run, e.g. VSI-Bench overall on the 1,000-question subset -->
- **Secondary metrics:** score per task, debiased overall, parse rate and input tokens. For any trained condition,
  MMSI-Bench overall too, as a regression check: VSI fine-tuning can cost general spatial reasoning (docs/idea.md).
- **Pipeline metrics** (when the sheet changes): SAM 3 prompts against the objects the question names (precision and
  recall); how often the full extractor fell back to syntactic mode (`prompt_mode` in the sheet's `question.json`);
  questions with an empty sheet; named objects with at least one row; objects and tokens per sheet; and the answer a
  fixed rule reads off the sheet alone (rows of the class for counting, the closest object for relative distance),
  which tells a wrong sheet apart from a model that can't use a right one.
- **Breakdowns:** question type, always. As needed: dataset (ScanNet, ScanNet++, ARKitScenes); multiple choice vs
  numeric; the metric tasks (absolute distance, object size, room size) vs the rest when the sheet's units are
  relative; relative direction easy, medium and hard; complete vs incomplete sheets.
- **Statistics:** every condition on the same questions, compared question by question: how many get better and
  worse, with a two-sided sign test (as in the README's space-aware vs uniform table), or a paired bootstrap interval.
  On the 1,000-question subset the standard error is about 1.5 points overall and 5 per task (docs/idea.md).
- **Smallest difference that matters:** <!-- the one that would change a decision, e.g. 2 points overall, 5 on rel. direction -->

## Threats to validity and checks

<!-- Keep the rows that apply and add this experiment's own. Problems with running (memory, time limits) go in the
task files. -->

| Risk | How we'd notice | What to do |
|---|---|---|
| | | |

## Runs and cost

<!-- One row per run. A run can serve several conditions (one sheet job for B and C, say). Output: the folder with the
predictions (.jsonl), the summaries and the run's freeze.txt or environment.txt. -->

| Run | Conditions | Task or command | Output | Commit |
|---|---|---|---|---|
| | | | | |

**Cost:** <!-- GPU hours for the sheets, the evaluation and any training, and the total, with the timings the
estimate is based on (README, earlier task-results/) -->

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

- YYYY-MM-DD:

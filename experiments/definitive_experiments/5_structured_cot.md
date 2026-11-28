---
id: definitive-5-structured-cot
title: Does structured or concise chain-of-thought help Qwen3.5-2B use a gold scene sheet on VSI-Bench, and does it recover what precomputed relations give
owner: Tamara Grujičić
created: 2026-11-19
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, prompting, chain-of-thought, definitive, exploratory]
varies: [prompt.reasoning, scene_sheet.source]
depends_on: [2026-11-19-structured-cot (plan and pilot), 2026-11-25-concise-math-cot (the follow-up), explore200's gold sheets]
---

# Structured and concise chain-of-thought

**Outcome:** on explore200's 200 questions, structured CoT does not help Qwen3.5-2B use the gold sheet (D − C −3.3,
Holm p 0.27) and hurts without it (B − A −13.3, Holm p 0.004); precomputed relations stay ahead (E − D +13.9, Holm
p 0.0002). With the sheet the model computes centre distances, where VSI-Bench measures closest points, and 12–28% of
the CoT replies loop to the token limit. In the follow-up, on 30 of its computing questions, concise CoT (F1) is no
better than structured CoT (F1 − D −2.0), the formulas (F2) and the worked example (F3) do not help, and no variant
comes near the given relations.

This file describes the structured-CoT pilot as it was run (2026-11-20; `experiments/2026-11-19-structured-cot/`,
write-up `explore_cot.md`). **Exploratory, not pre-registered:** the questions are explore200's, the format
experiment's exploration set, and this pilot's replies are what the follow-up's prompts (below) and experiment 6's
were developed from. The hypothesis and comparisons below were written in the structured-cot README on 2026-11-19,
before the pilot; the README's planned run on 2,728 questions was not made. The settings are the ones the pilot used, not this folder's
`README.md` section 2.

## Question

Does step-by-step reasoning help Qwen3.5-2B use a correct scene sheet, and does it recover what precomputed relations
give it?

## Hypothesis and what would change our mind

As written before the pilot (structured-cot README):

- **Expected:**
  - D − C > 0: the sheet supplies the numbers, and the model can copy two objects' coordinates and compute their
    distance in writing.
  - C < D < E: CoT recovers part of the gap to the relations, not all, because a 2B model computes less reliably than
    the relations state.
  - B − A ≈ 0 or below: without a sheet, reasoning adds little, as VSI-Bench's authors report for linguistic prompting.
- **Would prove it wrong:**
  - D − C ≤ 0, overall and on the computing tasks (absolute distance, relative distance, object size);
  - D ≥ E: computing from the sheet is as good as being given the relations;
  - B − A clearly positive (Holm p < 0.05 and above the smallest difference that matters), which would point to the
    prompt rather than the sheet.
- **Next:**
  - CoT helps and closes much of the gap to E: try it with the pipeline sheet;
  - it helps but leaves most of the gap: precomputed relations are the better way to give spatial facts;
  - it doesn't help: precomputed relations are the way forward; try a larger model only if relations alone are not
    enough.

## Background

- **The prompt:** `--reasoning cot` (`vsibench.cot_prompt`, text in the structured-cot folder's `cot_prompt.md`,
  generated at `6642c48`). It asks the model to say which objects the question is about, write down the facts it
  needs, work out the answer one step at a time showing the arithmetic, check the result against the video, and end
  on "Answer: X". Each task adds one hint (`vsibench.COT_HINTS`).
- **Prior work:** `experiments/2026-11-25-concise-math-cot/README.md`, Background.
- **The direct conditions on these questions:** C and E equal, byte for byte, B and C of the gold-sheet experiment's
  explore200 preview on the rebuilt gold sheets (`2026-11-19-gold-sheet-relations`, Log, 2026-11-24).

## Design

### Conditions

| ID | Condition | `prompt.reasoning` | `scene_sheet.source` | Role | Run |
|---|---|---|---|---|---|
| A | Video only, direct | direct | none | baseline | `A_video.jsonl` |
| B | Video only, CoT | cot | none | treatment | `video_cot.jsonl` |
| C | Video + gold sheet, direct | direct | gold | control | `gold_csv.jsonl` |
| D | Video + gold sheet, CoT | cot | gold | treatment | `gold_cot.jsonl` |
| E | Video + gold sheet + relations, direct | direct | gold_relations | upper reference | `gold_relations.jsonl` |

All in `results/zeroshot/vsibench/structured_cot/explore200/`, run at one commit; none reused.

### Comparisons

- **One Holm family:** D − C (primary), B − A and E − D.
- **Descriptive:** E − C, and (D − C) − (B − A).
- Each also without counting, and on the computing tasks alone (sign test, no Holm).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | `explore200_ids.txt` (format experiment): 200 questions, 25 per task (relative direction 8 easy / 8 medium / 9 hard), 25 scenes | A pilot before the planned 2,728-question run, on the exploration set |
| CoT prompt | `--reasoning cot`, unchanged | It has the four steps and an "Answer: X" line that `score_final` reads |
| Sheet | gold, csv | Isolates reasoning from detection errors |
| Relations | closest-point distances (`gold_relations`) | VSI-Bench measures between closest points |
| Token limit | 2,048 new tokens in every condition | The eval's CoT default; the direct runs use the same limit with `--final-answer` |
| Thinking mode | off, no condition for it | dropped on 2026-11-20; the reasoning comes from the prompt |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt
  scenes: explore200's 25
  question_types: all          # all 10
  n_questions: 200

visual:
  video: true
  model: Qwen/Qwen3.5-2B
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:
  source: varies               # none (A, B) | gold (C, D) | gold_relations (E)
  format: csv
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval
  reasoning: varies            # direct (A, C, E) | cot (B, D)
  thinking: off
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final
  batches: up to 8 questions of one video

training: none
```

## Metrics and analysis

`make_table.py --explore` (structured-cot folder): VSI-Bench overall, without counting, the computing score (the mean
of absolute distance, relative distance and object size), per task and question type, parse rate, input tokens, reply
length and the share at the limit; cut-off replies that loop (10 or more mm:ss times, or a last 2,000 characters that
compress below 25%); what fixed rules on the gold sheet answer; the reasoning check (D's replies that name a sheet row
or copy a sheet number). 20 of D's 30 wrong computing answers (score below 0.5), drawn with seed 0, were labelled by
reading each reply (`explore_cot_reasoning_labels.csv`; one labeller, Claude, no second). The notebook's paired
bootstrap intervals are reported beside the sign tests.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| 200 questions, 25 per task | intervals of 30 points or more per type | read per-task differences as descriptive |
| CoT replies loop until the limit and lose for that alone | the share cut off and looping | report it; the loops are the parse failures |
| The gold sheet contains the counting answer, and its largest side is the object-size answer | counting and object size near 100 | read without counting, and object size apart |
| The relations state the answer to most distance questions | E near its ceiling there | E is an upper reference for using given relations, not a test of reasoning |
| explore200 is the exploration set | — | these results shaped later prompts (the follow-up below, experiment 6); they are development results |

## Runs and cost

`explore_cot.ipynb` on Colab, one NVIDIA L4, 2026-11-20, at commit `7d39234` (`abde60f` plus `metric_scale=None` in
the notebook's `namespace()`). The five conditions ran in one session; `reused.json` is empty.

**Cost:** 53 ms per decoding step; the CoT runs took 63,773 (B) and 51,641 (D) decoding steps, about 1.7 hours
together on the L4.

## Results

From `explore_cot.md` (written by `make_table.py --explore`; its per-type, group, length, loop, rule and reasoning
tables are there):

| ID | Condition | Overall | Without counting | Computing | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Parsed | Reply tokens, median / p90 | At the limit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video, direct | 48.6 | 48.4 | 61.5 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 | 100.0% | 1 / 3 | 0.0% |
| B | video, CoT | 35.2 | 36.7 | 36.9 | 25.2 | 23.6 | 39.2 | 41.6 | 48.0 | 40.3 | 20.0 | 44.0 | 88.5% | 444 / 2048 | 28.0% |
| C | + gold sheet, direct | 52.1 | 45.3 | 48.5 | 100.0 | 26.4 | 55.2 | 47.2 | 64.0 | 56.0 | 32.0 | 36.0 | 100.0% | 1 / 4 | 0.0% |
| D | + gold sheet, CoT | 48.8 | 44.9 | 58.0 | 76.0 | 30.0 | 76.0 | 36.4 | 68.0 | 39.8 | 28.0 | 36.0 | 93.5% | 521 / 2048 | 11.5% |
| E | + gold sheet + relations, direct | 62.7 | 57.4 | 82.0 | 100.0 | 94.8 | 55.2 | 48.0 | 96.0 | 47.7 | 28.0 | 32.0 | 100.0% | 1 / 5 | 0.0% |

| Comparison | Overall | Without counting | Computing | Better / worse / same | Holm p | 95% interval (overall) | Computing: better / worse, p |
|---|---:|---:|---:|---:|---:|---:|---:|
| D − C | −3.3 | −0.4 | +9.5 | 43 / 55 / 102 | 0.266 | [−10.7, +4.1] | 27 / 19, 0.30 |
| B − A | −13.3 | −11.7 | −24.5 | 40 / 74 / 86 | 0.0037 | [−20.5, −5.8] | 14 / 35, 0.0038 |
| E − D | +13.9 | +12.5 | +24.0 | 71 / 30 / 99 | 0.00017 | [+7.1, +20.8] | 36 / 11, 0.00035 |
| E − C (descriptive) | +10.6 | +12.1 | +33.5 | 40 / 11 / 149 | – | – | 33 / 1, 4e-9 |

### Examples

In `explore_cot.md`, Examples: D computing the centre distance and calling it the closest-point distance (question
837: 3.358 m against a truth of 1.9); C giving the table's height and D its longest side (question 266); D listing the
right chair rows, then "verifying with the video" in invented mm:ss frames until the limit (question 4410).

### What the results show

- **CoT without a sheet hurts:** B − A −13.3; 28% of B's replies stop at the limit, 51 of those 56 narrating frames or
  repeating.
- **With the sheet, CoT moves tasks in opposite directions:** D − C +20.8 on object size, −24.0 on counting, −16.2 on
  relative direction, −10.8 on room size; +9.5 on the computing tasks' mean, almost all from object size.
- **The model computes from the sheet, but the wrong distance:** on the computing tasks 99% of D's replies copy a sheet
  number and 89% name a row; 84% of D's absolute-distance answers are the centre distance and none the closest-point
  one. Of 20 labelled wrong answers: 11 wrong method, 4 wrong rows, 3 cut off, 2 misread, none wrong arithmetic.

## Conclusion

- **Answer to the question:** on these 200 questions, structured CoT does not improve how Qwen3.5-2B uses the gold
  sheet overall, and it hurts without one. The model computes from the coordinates with correct arithmetic, but
  computes centre distances, and the prompt's check against the video sends replies into loops.
- **Hypothesis:** D − C > 0 not supported overall (−3.3, Holm p 0.27), only on the computing mean (+9.5, p 0.30, from
  object size); D < E holds, C < D does not; B − A ≤ 0 supported, clearly below (−13.3).
- **Caveats:** exploratory: 200 development questions, one model, one seed, greedy decoding, on an L4; one labeller
  for the reasoning check; the gold sheet contains the counting and object-size answers.
- **Next:** the failure analysis (wrong distance, loops) led to the concise and mathematical variants (the follow-up
  below) and the model-size check (experiment 6).

## Follow-up: concise, mathematical and few-shot CoT (exploratory)

**Outcome:** exploratory, on 30 development questions with Qwen3.5-2B: concise CoT (F1) scores 64.7 on the computing
tasks against structured CoT's 66.7 (F1 − D −2.0, 95% interval −15.7 to +11.7), and adding the formulas (F2) gives
57.0 (F2 − F1 −7.7: +16.0 on absolute distance, −40.0 on relative distance). The worked example (F3), run only on the
smoke stage's 3 questions, copied the example's numbers in 2 of 3 replies. No variant comes near the given relations
(E, 85.3).

This section describes what was run of `experiments/2026-11-25-concise-math-cot/` (its README has the full design, the
prompts' sources and the conventions' checks). **Exploratory, not pre-registered:** every question is from explore200,
where these prompts were developed from the structured-CoT pilot's replies (above). The hypothesis and
comparisons below were written before any run (README, 2026-11-25); the README's later stages (75 questions) and its
held-out evaluation were not run. A, C, D and E are the pilot's predictions on the same questions, not rerun; their
settings are the pilot's. The 9B runs of the same prompts are experiment 6.

### Question

Can a more constrained, mathematically explicit CoT, with or without one worked example per task, make Qwen3.5-2B
answer the computing tasks better from the gold sheet than structured CoT (D), than direct answering (C) and than the
video alone (A)? How far does it stay from the given relations (E)?

### Hypothesis and what would change our mind

As written before any run (concise-math-cot README):

- **Expected:**
  - F1 − D ≥ 0, with shorter replies and fewer cut off: the pilot's loops came from its check against the video.
  - F2 − F1 > 0, almost all of it on absolute distance: D computed centre distances (84% of its answers), and the
    closest-point formula is what VSI-Bench measures. Little on object size (D already took the longest side in 76%)
    and on relative distance in the development set, where the two rules never disagree.
  - F3 − F2 ≥ 0, small: a worked example shows the formula once more.
  - E − F2 > 0: E is given the distances that F2 has to compute.
- **Would prove it wrong:** F2 ≤ F1 on absolute distance (the model does not apply a formula it is given), F1 below D
  (brevity costs answers), or a rise in unread answers.
- **Next:** if F2 or F3 improves on D and C on the development set, freeze it and evaluate it on held-out questions,
  then on the pipeline's sheets. If none does, the model does not use a stated procedure, and precomputed relations
  (E) remain the way to give it distances.

### Background

- **The pilot's computing questions** (the pilot above; 75 questions): A 61.5, C 48.5, D 58.0, E 82.0. 84% of D's
  absolute-distance answers were the centre distance; 11 of 20 labelled wrong answers used the wrong method, none the
  wrong arithmetic; 11.5% of D's replies looped to the limit.
- **The rules, applied to the gold sheet on the full benchmark** (README, Design): absolute distance, closest-point
  gap 91.4, centre-to-centre 34.3; relative distance, nearest by closest points 98.3, by centres 83.7 (688 questions);
  object size, largest side × 100, 100.0. On explore200's relative-distance questions the two rules pick the same
  answer.
- **Prior work:** `experiments/2026-11-25-concise-math-cot/README.md`, Background.

### Design

#### Conditions

| ID | Condition | `prompt.reasoning` | Role | Run |
|---|---|---|---|---|
| A | Video only, direct | direct | baseline | the pilot's `A_video.jsonl` |
| C | Video + gold sheet, direct | direct | baseline | the pilot's `gold_csv.jsonl` |
| D | Video + gold sheet, structured CoT | cot | baseline | the pilot's `gold_cot.jsonl` |
| E | Video + gold sheet + relations, direct | direct | upper reference | the pilot's `gold_relations.jsonl` |
| F1 | Video + gold sheet, concise CoT | `cot_f1` | treatment | `dev_v1/cot_f1` |
| F2 | F1 + geometric formulas | `cot_f2` | treatment | `dev_v1/cot_f2` |
| F3 | F2 + one worked example | `cot_f3` | treatment | `dev_v1/cot_f3` (smoke stage only) |

**The prompts** (`src/mvp/benchmarks/cot_variants.py`, `PROMPT_VERSION` `2026-11-25.1`): each replaces the "answer
directly" line, as D does; the video, the sheet and the question are as in C, D and E.

- F1: `CONCISE` (reason briefly, use the sheet, the video only to tell which object is meant), the task's hint
  (`vsibench.COT_HINTS`, unchanged from D), the `FORMAT` lines (`Objects:`, `Attributes:`, `Calculation:`) and
  `vsibench.COT_ANSWER`.
- F2: F1 plus `MATH[task]`: the closest-point gap per axis, `max(0, |A_k − B_k| − (A_size_k + B_size_k) / 2)`, and
  its straight-line length (absolute and relative distance, `scene_sheet.box_gap`); the largest side × 100 (object
  size).
- F3: F2 plus one synthetic worked example per task, marked as another scene: absolute distance 1.059 m, relative
  distance where the nearest centre and the nearest box differ, object size 1.830 m → 183 cm.

They are defined only for these three tasks, with the gold sheet.

#### Comparisons

As written before any run; all on the computing score (the mean of the three tasks' scores) and per task:

- **Primary, one Holm family:** F1 − D, F2 − F1 and F3 − F2.
- **Secondary**, each its own Holm family: every F against C; every F against E; every F against A.

#### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | explore200's computing questions, nested per task (`followup.nested_subset`, seed 0): the smoke stage's 3 (1 per task), then the first 10 per task, `model_size_ids.txt` (30, 14 scenes) | The development set; the 30 are the README's `extended` stage, shared with experiment 6 |
| Baselines | the pilot's predictions, not rerun | The same model, frames, sheet text, decoding, limit and answer rule; `test_followup.Preservation` pins the files' hashes |
| F3 | the smoke stage only | its replies copied the worked example (Results); it was not run on the 30 |
| Prompt text | `PROMPT_VERSION` `2026-11-25.1` for every F reply | each folder's `config.json` holds the prompt fingerprint, and a run under another stops |

### Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: explore200's computing questions; smoke 3, then experiments/2026-11-25-concise-math-cot/model_size_ids.txt (30)
  scenes: explore200's (14 for the 30)
  question_types: [object_abs_distance, object_rel_distance, object_size_estimation]
  n_questions: 30              # F3: 3

visual:
  video: true
  model: Qwen/Qwen3.5-2B
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:
  source: gold                 # tasks/openvocab_adapter/out_full, index 22e0ee1e…; gold_relations in E
  format: csv
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  reasoning: varies            # direct (A, C, E) | cot (D) | cot_f1 | cot_f2 | cot_f3
  thinking: off
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final
  batches: up to 8 questions of one video (--batch-size 8 --token-budget 120000)

training: none
```

### Metrics and analysis

`analysis.py` (concise-math-cot folder; it marks any table below 25 questions per task "Exploratory"): the computing
score and each task's, parse rate, reply length (median, 90th percentile), the share cut off at 2,048 tokens, seconds
per question; per comparison the difference, better / worse / same, an exact sign test, Holm within the family and a
paired bootstrap 95% interval (10,000 resamples within question type, seed 0); and `errors.csv`, the automatic flags
per reply (cut off; no "Answer:" line; unread; equal to the centre or the closest-point rule; copies the example's
answer or numbers; and others), with empty columns for reading by hand.

### Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| 3 and 30 questions | the analysis prints "Exploratory" | read as checks of the prompts, not results |
| explore200 shaped these prompts | — | development results; a claim needs held-out questions |
| On explore200's relative-distance questions the centre and closest-point rules pick the same answer | — | F2 cannot gain there on these questions |
| The gold sheet gives the object-size answer (largest side) | object size near 100 | read object size apart |
| The model copies the worked example (2507.13362, IV-E) | F3 answers equal to 1.059, 183 or the example's letter | found in the smoke stage (Results) |
| The baselines ran in the pilot's batches, F1 and F2 in the subset's | — | stated; greedy decoding in a batch can differ in the last digits |
| F1 changes the instructions and the output form together | — | stated; both target the length |

### Runs and cost

1. **Smoke stage** (`cot_followup.ipynb`, Colab, 2026-11-25; code `64e566e`, a snapshot of the uncommitted
   follow-up): F1, F2 and F3 on 3 questions (one per task), 9 replies; 122–159 s per question including the model's
   load.
2. **The 30 questions** (`cot_9b.ipynb`, main stage, Colab, one NVIDIA L4; code `e2cd67a`): F1 and then F2 on the 27
   questions the smoke stage did not answer, the smoke stage's 3 reused (same prompt fingerprint). 246.5 s (F1) and
   532.6 s (F2) for the 27, 24.1 and 32.8 s per question.

The predictions are in `results/zeroshot/vsibench/concise_math_cot/dev_v1/` (the run folder `dev_v1`, from the notebook's
`cot_9b_results_main.zip`, SHA-256 `5db6d4d61a28374f0729477fae705d53e6b7ffc95c96123fadaf52e774dfb4c3`; its top-level
`analysis.md` and `errors.csv` are the smoke stage's).

### Results

**Smoke stage, 3 questions** (`analysis.py`):

| Condition | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | Cut off |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 76.7 | 100.0 | 100.0 | 30.0 | 100.0% | 3 / 3 | 0.0% |
| C | 83.3 | 90.0 | 100.0 | 60.0 | 100.0% | 4 / 4 | 0.0% |
| D | 6.7 | 20.0 | 0.0 | 0.0 | 100.0% | 1196 / 1878 | 33.3% |
| E | 86.7 | 100.0 | 100.0 | 60.0 | 100.0% | 4 / 5 | 0.0% |
| F1 | 26.7 | 20.0 | 0.0 | 60.0 | 100.0% | 183 / 825 | 0.0% |
| F2 | 33.3 | 0.0 | 0.0 | 100.0 | 100.0% | 404 / 1719 | 33.3% |
| F3 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0% | 235 / 299 | 0.0% |

**30 questions, 10 per task** (`analysis.py` on F1 and F2):

| Condition | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | Cut off |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 57.0 | 43.0 | 70.0 | 58.0 | 100.0% | 3 / 3 | 0.0% |
| C | 52.0 | 27.0 | 70.0 | 59.0 | 100.0% | 4 / 5 | 0.0% |
| D | 66.7 | 40.0 | 80.0 | 80.0 | 100.0% | 536 / 1548 | 3.3% |
| E | 85.3 | 97.0 | 100.0 | 59.0 | 100.0% | 4 / 5 | 0.0% |
| F1 | 64.7 | 45.0 | 70.0 | 79.0 | 100.0% | 143 / 463 | 0.0% |
| F2 | 57.0 | 61.0 | 30.0 | 80.0 | 96.7% | 358 / 1263 | 6.7% |

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 − D | −2.0 | [−15.7, +11.7] | +5.0 | −10.0 | −1.0 | 5 / 6 / 19 | 1 |
| F2 − F1 | −7.7 | [−27.7, +12.0] | +16.0 | −40.0 | +1.0 | 9 / 10 / 11 | 1 |
| F1 − C | +12.7 | [−5.3, +31.0] | +18.0 | +0.0 | +20.0 | 10 / 6 / 14 | 0.85 |
| F2 − C | +5.0 | [−18.0, +28.0] | +34.0 | −40.0 | +21.0 | 15 / 10 / 5 | 0.85 |
| F1 − E | −20.7 | [−34.0, −7.7] | −52.0 | −30.0 | +20.0 | 3 / 12 / 15 | 0.035 |
| F2 − E | −28.3 | [−44.3, −12.3] | −36.0 | −70.0 | +21.0 | 5 / 19 / 6 | 0.013 |
| F1 − A | +7.7 | [−13.0, +28.3] | +2.0 | +0.0 | +21.0 | 14 / 7 / 9 | 0.38 |
| F2 − A | +0.0 | [−21.0, +22.3] | +18.0 | −40.0 | +22.0 | 15 / 9 / 6 | 0.38 |

F3 − F2 has only the smoke stage's 3 questions (−33.3; 0 better, 1 worse).

#### Examples

- **F3 copies the worked example** (smoke stage): on question 838 (absolute distance, truth 2.2 m) the reply is the
  absolute-distance example itself, with the example's `sofa_1` and `lamp_1` rows and its answer 1.059; on question
  2227 (object size, truth 73 cm) it lists the example's sides (0.452, 1.830, 0.600) and answers 183. On question 2984
  (relative distance) it uses the scene's own rows but writes zero gaps for three of the four options.
- **F2 on relative distance:** 3 of 10 right, against 7 for F1 and C and 10 for E.

#### What the results show

- **Concise reasoning does what it was meant to for length, not for the score:** F1's replies are a quarter of D's
  length (median 143 against 536 tokens) and none is cut off, but F1 − D is −2.0, within the interval.
- **The formulas help absolute distance and hurt relative distance:** F2 − F1 is +16.0 on absolute distance and −40.0
  on relative distance; two F2 replies are cut off and one is unread.
- **The worked example is copied:** F3 gave the example's own answers in 2 of 3 smoke replies, so it was not run on
  the 30.
- **The relations stay ahead:** F1 − E −20.7 and F2 − E −28.3, the only comparisons with Holm p below 0.05.

### Conclusion

- **Answer to the question:** on these 30 development questions, neither concise CoT nor the formulas make Qwen3.5-2B
  answer the computing tasks better than structured CoT, and both stay well below being given the relations. The
  worked example is copied rather than followed.
- **Hypothesis:**
  - F1 − D ≥ 0: not supported (−2.0), though replies are shorter and none is cut off;
  - F2 − F1 > 0 on absolute distance: holds (+16.0), but the computing score falls (−7.7) through relative distance;
  - F3 − F2 ≥ 0: not tested beyond 3 questions; F3 copied its example;
  - E − F2 > 0: holds (+28.3).
  - Of the disproofs: F2 ≤ F1 on absolute distance is not met; F1 is below D by 2.0 points (within the interval);
    unread answers rise from 0 to one F2 reply.
- **Caveats:** exploratory: 3 and 30 questions from the development set, one model, one run per condition; the
  baselines are the pilot's predictions, in other batches.
- **Next** (the README's branch for "none improves on D and C"): precomputed relations (E) remain the way to give the
  model distances. The 9B check of the same prompts is experiment 6.

## Planned: full-benchmark rerun (not run)

The rerun planned in this folder's `README.md` (sections 3 and 4: its hypotheses, comparisons and shared runs),
as written there on 2026-11-28. Nothing in this section has been run; the results above are the exploratory runs'.

### Conditions

| ID | Condition | `prompt.reasoning` | `scene_sheet.source` | Role | Run |
|---|---|---|---|---|---|
| A | Video only, direct | direct | none | baseline | R1 |
| B | Video only, CoT | CoT | none | treatment | R2 |
| C | Video + gold sheet, direct | direct | gold | control | R3 |
| D | Video + gold sheet, CoT | CoT | gold | treatment | R4 |
| E | Video + gold sheet + relations, direct | direct | gold_relations | upper reference | R5 |
| F1 | Video + gold sheet, concise CoT | `cot_f1` | gold | treatment | R17 |

F1 runs on the 2,497 computing questions only (`computing_ids.txt`); its comparisons read A, C, D and E on the same
questions.

### Runs

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
DEF=experiments/definitive_experiments         # computing_ids.txt, computing_smoke_ids.txt
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv"
COT="--reasoning cot"                          # as the code stands (cot_prompt.md)
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** As experiment 2, check 1. `computing_ids.txt` must have SHA-256
   `ed4b146f98e064965e2783fedcae120637e96c4bfc1afbfa2503d0e6050a0a5c`.
2. **Tests.** All must pass:

   ```bash
   PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet
   PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_cot_variants
   PYTHONPATH=src python -m unittest discover -s experiments/2026-11-25-concise-math-cot -p "test_*.py"
   ```

3. **Prompt check** for D, as `experiments/2026-11-19-gold-sheet-relations/README.md`'s Runs step 2, with
   `--sheet-source gold $COT --model Qwen/Qwen3.5-2B`. Compare the CoT text with `cot_prompt.md`.
4. **Smoke test** of A and D, then B, then the model revision. For B and D, also report reply length, the share cut off
   and the parse rate.

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R4.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_2B_R1.jsonl $S/qwen3_5_2B_R4.jsonl --n 20
   python -m mvp.eval.vsibench $COMMON $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R2.jsonl
   python $C/cluster_checks.py revision Qwen/Qwen3.5-2B $OUT/model_revision_qwen3_5_2B.txt
   ```

5. **Smoke test** of F1 on the 6 computing questions of the smoke set (`computing_smoke_ids.txt`, SHA-256
   `7f0e27eb891134bba048a7255fe01487ae21ee2482390626c223aa727da98e49`). Print every prompt and reply. The run must
   answer all 6, and every reply must have an "Answer:" line.

   ```bash
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold --reasoning cot_f1 --ids $DEF/computing_smoke_ids.txt \
       --out $S/qwen3_5_2B_R17.jsonl
   ```

**Runs:** R1 to R5 each report 5,130 questions, F1 (R17) 2,497:

```bash
python -m mvp.eval.vsibench $COMMON                                   --out $OUT/qwen3_5_2B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $COT                              --out $OUT/qwen3_5_2B_R2.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold         --out $OUT/qwen3_5_2B_R3.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold $COT    --out $OUT/qwen3_5_2B_R4.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations --out $OUT/qwen3_5_2B_R5.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold --reasoning cot_f1 --ids $DEF/computing_ids.txt \
    --out $OUT/qwen3_5_2B_R17.jsonl
```

**Analysis:** `<make_table_5.py>` (`README.md` section 5, item 5) writes the Results table and the CoT measures. For
F1 it builds on `experiments/2026-11-25-concise-math-cot/analysis.py` and also writes `errors.csv`.

## Log

- 2026-11-19: plan, hypothesis and comparisons written (structured-cot README), for a run on 2,728 questions.
- 2026-11-20: thinking-mode condition dropped; the pilot ran A–E on explore200 at `7d39234` (structured-cot README,
  Log).
- 2026-11-25: follow-up plan, prompts (`PROMPT_VERSION` 2026-11-25.1), runs and analysis written (concise-math-cot
  README).
- 2026-11-25: smoke stage of F1–F3 (3 questions) at `64e566e`; F3 copied its example in 2 of 3 replies.
- 2026-11-25: F1 and F2 on the 30 questions of `model_size_ids.txt` at `e2cd67a`, in the 9B notebook's main stage.
- 2026-11-28: this file rewritten to describe the pilot as run, with the 2B; the 4B plan and its open questions
  removed.
- 2026-11-28: the follow-up's file (`6_concise_math_cot.md`) rewritten to describe what was run, with the 2B; the 4B
  plan and its open questions removed.

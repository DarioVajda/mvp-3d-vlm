---
id: definitive-6-concise-math-cot
title: Concise, mathematical and few-shot structured CoT for Qwen3.5-2B on the computing tasks with the gold sheet
owner: Tamara Grujičić
created: 2026-11-25
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [prompting, chain-of-thought, gold-sheet, definitive, exploratory]
varies: [prompt.reasoning]
depends_on: [definitive-5-structured-cot (A, C, D, E and their predictions), 2026-11-25-concise-math-cot]
---

# Concise, mathematical and few-shot structured CoT

**Outcome:** exploratory, on 30 development questions with Qwen3.5-2B: concise CoT (F1) scores 64.7 on the computing
tasks against structured CoT's 66.7 (F1 − D −2.0, 95% interval −15.7 to +11.7), and adding the formulas (F2) gives
57.0 (F2 − F1 −7.7: +16.0 on absolute distance, −40.0 on relative distance). The worked example (F3), run only on the
smoke stage's 3 questions, copied the example's numbers in 2 of 3 replies. No variant comes near the given relations
(E, 85.3).

This file describes what was run of `experiments/2026-11-25-concise-math-cot/` (its README has the full design, the
prompts' sources and the conventions' checks). **Exploratory, not pre-registered:** every question is from explore200,
where these prompts were developed from the structured-CoT pilot's replies (experiment 5). The hypothesis and
comparisons below were written before any run (README, 2026-11-25); the README's later stages (75 questions) and its
held-out evaluation were not run. A, C, D and E are experiment 5's predictions on the same questions, not rerun; their
settings are experiment 5's. The 9B runs of the same prompts are experiment 7.

## Question

Can a more constrained, mathematically explicit CoT, with or without one worked example per task, make Qwen3.5-2B
answer the computing tasks better from the gold sheet than structured CoT (D), than direct answering (C) and than the
video alone (A)? How far does it stay from the given relations (E)?

## Hypothesis and what would change our mind

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

## Background

- **The pilot's computing questions** (experiment 5; 75 questions): A 61.5, C 48.5, D 58.0, E 82.0. 84% of D's
  absolute-distance answers were the centre distance; 11 of 20 labelled wrong answers used the wrong method, none the
  wrong arithmetic; 11.5% of D's replies looped to the limit.
- **The rules, applied to the gold sheet on the full benchmark** (README, Design): absolute distance, closest-point
  gap 91.4, centre-to-centre 34.3; relative distance, nearest by closest points 98.3, by centres 83.7 (688 questions);
  object size, largest side × 100, 100.0. On explore200's relative-distance questions the two rules pick the same
  answer.
- **Prior work:** `experiments/2026-11-25-concise-math-cot/README.md`, Background.

## Design

### Conditions

| ID | Condition | `prompt.reasoning` | Role | Run |
|---|---|---|---|---|
| A | Video only, direct | direct | baseline | experiment 5's `A_video.jsonl` |
| C | Video + gold sheet, direct | direct | baseline | experiment 5's `gold_csv.jsonl` |
| D | Video + gold sheet, structured CoT | cot | baseline | experiment 5's `gold_cot.jsonl` |
| E | Video + gold sheet + relations, direct | direct | upper reference | experiment 5's `gold_relations.jsonl` |
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

### Comparisons

As written before any run; all on the computing score (the mean of the three tasks' scores) and per task:

- **Primary, one Holm family:** F1 − D, F2 − F1 and F3 − F2.
- **Secondary**, each its own Holm family: every F against C; every F against E; every F against A.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | explore200's computing questions, nested per task (`followup.nested_subset`, seed 0): the smoke stage's 3 (1 per task), then the first 10 per task, `model_size_ids.txt` (30, 14 scenes) | The development set; the 30 are the README's `extended` stage, shared with experiment 7 |
| Baselines | experiment 5's predictions, not rerun | The same model, frames, sheet text, decoding, limit and answer rule; `test_followup.Preservation` pins the files' hashes |
| F3 | the smoke stage only | its replies copied the worked example (Results); it was not run on the 30 |
| Prompt text | `PROMPT_VERSION` `2026-11-25.1` for every F reply | each folder's `config.json` holds the prompt fingerprint, and a run under another stops |

## Configuration

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

## Metrics and analysis

`analysis.py` (concise-math-cot folder; it marks any table below 25 questions per task "Exploratory"): the computing
score and each task's, parse rate, reply length (median, 90th percentile), the share cut off at 2,048 tokens, seconds
per question; per comparison the difference, better / worse / same, an exact sign test, Holm within the family and a
paired bootstrap 95% interval (10,000 resamples within question type, seed 0); and `errors.csv`, the automatic flags
per reply (cut off; no "Answer:" line; unread; equal to the centre or the closest-point rule; copies the example's
answer or numbers; and others), with empty columns for reading by hand.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| 3 and 30 questions | the analysis prints "Exploratory" | read as checks of the prompts, not results |
| explore200 shaped these prompts | — | development results; a claim needs held-out questions |
| On explore200's relative-distance questions the centre and closest-point rules pick the same answer | — | F2 cannot gain there on these questions |
| The gold sheet gives the object-size answer (largest side) | object size near 100 | read object size apart |
| The model copies the worked example (2507.13362, IV-E) | F3 answers equal to 1.059, 183 or the example's letter | found in the smoke stage (Results) |
| The baselines ran in the pilot's batches, F1 and F2 in the subset's | — | stated; greedy decoding in a batch can differ in the last digits |
| F1 changes the instructions and the output form together | — | stated; both target the length |

## Runs and cost

1. **Smoke stage** (`cot_followup.ipynb`, Colab, 2026-11-25; code `64e566e`, a snapshot of the uncommitted
   follow-up): F1, F2 and F3 on 3 questions (one per task), 9 replies; 122–159 s per question including the model's
   load.
2. **The 30 questions** (`cot_9b.ipynb`, main stage, Colab, one NVIDIA L4; code `e2cd67a`): F1 and then F2 on the 27
   questions the smoke stage did not answer, the smoke stage's 3 reused (same prompt fingerprint). 246.5 s (F1) and
   532.6 s (F2) for the 27, 24.1 and 32.8 s per question.

The predictions are in `results/zeroshot/vsibench/concise_math_cot/dev_v1/` (the run folder `dev_v1`, from the notebook's
`cot_9b_results_main.zip`, SHA-256 `5db6d4d61a28374f0729477fae705d53e6b7ffc95c96123fadaf52e774dfb4c3`; its top-level
`analysis.md` and `errors.csv` are the smoke stage's).

## Results

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

### Examples

- **F3 copies the worked example** (smoke stage): on question 838 (absolute distance, truth 2.2 m) the reply is the
  absolute-distance example itself, with the example's `sofa_1` and `lamp_1` rows and its answer 1.059; on question
  2227 (object size, truth 73 cm) it lists the example's sides (0.452, 1.830, 0.600) and answers 183. On question 2984
  (relative distance) it uses the scene's own rows but writes zero gaps for three of the four options.
- **F2 on relative distance:** 3 of 10 right, against 7 for F1 and C and 10 for E.

### What the results show

- **Concise reasoning does what it was meant to for length, not for the score:** F1's replies are a quarter of D's
  length (median 143 against 536 tokens) and none is cut off, but F1 − D is −2.0, within the interval.
- **The formulas help absolute distance and hurt relative distance:** F2 − F1 is +16.0 on absolute distance and −40.0
  on relative distance; two F2 replies are cut off and one is unread.
- **The worked example is copied:** F3 gave the example's own answers in 2 of 3 smoke replies, so it was not run on
  the 30.
- **The relations stay ahead:** F1 − E −20.7 and F2 − E −28.3, the only comparisons with Holm p below 0.05.

## Conclusion

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
  model distances. The 9B check of the same prompts is experiment 7.

## Log

- 2026-11-25: plan, prompts (`PROMPT_VERSION` 2026-11-25.1), runs and analysis written (concise-math-cot README).
- 2026-11-25: smoke stage of F1–F3 (3 questions) at `64e566e`; F3 copied its example in 2 of 3 replies.
- 2026-11-25: F1 and F2 on the 30 questions of `model_size_ids.txt` at `e2cd67a`, in the 9B notebook's main stage.
- 2026-11-28: this file rewritten to describe what was run, with the 2B; the 4B plan and its open questions removed.

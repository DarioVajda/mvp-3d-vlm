---
id: definitive-7-model-size
title: Does Qwen3.5-9B use the gold scene sheet better than the 2B on the computing tasks, answering directly and with concise CoT
owner: Tamara Grujičić
created: 2026-11-25
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [gold-sheet, chain-of-thought, model-size, definitive, exploratory]
varies: [visual.model, prompt.reasoning]
depends_on: [definitive-5-structured-cot (the 2B's A, C, D, E), definitive-6-concise-math-cot (the prompts, the 2B's F1 and F2)]
---

# Model size

**Outcome:** exploratory, on 30 development questions: concise CoT does not help the 9B (9B F1 − 9B C −3.7, 95%
interval −17.0 to +9.0), where it gave the 2B +12.7, so the interaction is −16.3 (interval −35.0 to +1.3). Answering
directly with the gold sheet, the 9B is 28.0 points above the 2B (80.0 against 52.0, interval +13.0 to +42.7, Holm
p 0.096).

This file describes the model-size check as it was run (`experiments/2026-11-25-concise-math-cot/README.md`, "Model
size", and `model_size.md` from `analysis.py --model-size`). **Exploratory, not pre-registered:** the 30 questions
(`model_size_ids.txt`) are explore200's, where the prompts were developed; the plan was written on 2026-11-25 before
any 9B answer existed. The prompts and the 2B's F1 and F2 are experiment 6's; the 2B's A, C, D and E are experiment
5's predictions.

## Question

Does CoT over the gold scene sheet fail because the 2B model is too small? With the same questions and prompts, does
concise CoT (F1) beat direct answering with the sheet (C) for Qwen3.5-9B, and is its gain larger than the 2B's?

## Hypothesis and what would change our mind

As planned before the 9B runs (concise-math-cot README):

- **Size is a reason:** 9B F1 − 9B C > 0 and (9B F1 − 9B C) − (2B F1 − 2B C) > 0, with the replies showing the 9B
  applying a procedure the 2B does not.
- **Size is not the reason, at this scale:** 9B F1 − 9B C ≤ 0. Then the next suspects are the prompt and the sheet's
  form, not the model's size.
- **Read alongside:** 9B C − 2B C (does the 9B read the sheet better when answering directly?) and 9B E − 2B E (with
  the distances given).
- **Next:** the result decides whether a larger 9B run (25 per task, then held-out questions) is worth it. With 10
  questions per task, only differences of about 15 points or more can clear the intervals.

## Background

- **Prior work:** `experiments/2026-11-25-concise-math-cot/README.md`, Background.
- **The 2B on these questions:** experiment 6's Results (A 57.0, C 52.0, D 66.7, E 85.3, F1 64.7, F2 57.0).

## Design

### Conditions

| Model | Condition | `prompt.reasoning` | Run |
|---|---|---|---|
| 2B | A, C, D, E | direct; cot (D) | experiment 5's predictions |
| 2B | F1, F2 | `cot_f1`, `cot_f2` | experiment 6's `dev_v1/cot_f1`, `dev_v1/cot_f2` |
| 9B | A, C, E | direct | `dev_v1/qwen3.5-9b/A_video`, `gold_csv`, `gold_relations` |
| 9B | F1 | `cot_f1` | `dev_v1/qwen3.5-9b/cot_f1` |
| 9B | F2 | `cot_f2` | `dev_v1/qwen3.5-9b/cot_f2` (run last) |

A: video only; C: video + gold sheet; E: video + gold sheet + relations; F1, F2: experiment 6's prompts with the gold
sheet. The 9B's A, C and E use the pilot's direct flags (`followup.DIRECT`).

### Comparisons

As planned, one Holm family of five per-question contrasts, on the computing score (the mean of the three tasks) and
per task:

| Comparison | Question |
|---|---|
| 9B F1 − 9B C | does CoT help the 9B? |
| 2B F1 − 2B C | does CoT help the 2B? |
| (9B F1 − 9B C) − (2B F1 − 2B C) | does model size change the effect of CoT? |
| 9B C − 2B C | the sheet, direct |
| 9B E − 2B E | the sheet and relations, direct |

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | `model_size_ids.txt`: 30, the first 10 of each computing task in `followup.nested_subset`'s seed-0 order over explore200's computing questions; 14 scenes | They include the smoke stage's 3, and every one has the pilot's A, C, D and E (`test_followup.ModelSize.test_question_ids`) |
| The same for both models | thinking off, 16 frames `uniform128`, the gold sheet as csv, greedy decoding, 2,048 new tokens, the final-answer rule, the prompts (`PROMPT_VERSION` 2026-11-25.1) | Each 9B `config.json` differs from the 2B's only in the model, the device map and the batch size (`test_followup.ModelSize`) |
| The 9B's loading | bf16, `--device-map auto`, no quantization; `QwenVL` stops if any weight would go to the CPU or disk | The same model family at full precision |
| Batches | the 9B one question at a time (`--batch-size 1`); the 2B the pilot's batches of up to 8 | The 9B fits a 24 GB L4, and its answers do not depend on which questions share a batch |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-25-concise-math-cot/model_size_ids.txt
  scenes: 14 of explore200's
  question_types: [object_abs_distance, object_rel_distance, object_size_estimation]
  n_questions: 30

visual:
  video: true
  model: varies                # Qwen/Qwen3.5-2B | Qwen/Qwen3.5-9B (device_map auto, batch size 1)
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:
  source: gold                 # tasks/openvocab_adapter/out_full, index 22e0ee1e…; gold_relations in E; none in A
  format: csv
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  reasoning: varies            # direct (A, C, E) | cot_f1 | cot_f2 (cot (D) for the 2B only)
  thinking: off
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final

training: none
```

## Metrics and analysis

`analysis.py --model-size` (`model_size.md`): per model and condition the scores, parse rate, the share cut off at
2,048 tokens, reply length and seconds per question; each contrast's difference, better / worse / same, an exact sign
test, Holm over the five, and a paired bootstrap 95% interval (10,000 resamples within question type, seed 0). The
error flags are experiment 6's (`errors.csv`).

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| 30 questions, 10 per task | intervals of 15 points or more | read as a decision about a larger run, not an answer |
| explore200 shaped the prompts, on the 2B | — | development results |
| Batches differ: the 2B's baselines in the pilot's batches, the 9B one question at a time, possibly on another GPU | — | stated; batching and the GPU can change greedy decoding in the last digits |
| The 2B's F1 and F2 come from two sessions (the smoke stage's 3, the other 27) | — | stated; the same prompt fingerprint |
| Larger models write longer replies, and the limit cuts more of them | the share cut off | reported per condition |

## Runs and cost

`cot_9b.ipynb` on Colab, one NVIDIA L4 (23.7 GB), 2026-11-25, code `e2cd67a`. The 9B loaded on `cuda:0` alone (18.8 GB of
weights, 14.6 s), peaking at 19.5 GB.

1. **Smoke stage:** the four 9B conditions on 3 questions (12 replies), 29–59 s per condition with a load each.
2. **Main stage**, the 27 other questions: 9B C 92.5 s, 9B F1 773.4 s, 9B E 97.5 s, 2B F1 246.5 s, 9B A 89.9 s; then
   all 30 of 9B F2, 1,408.5 s; then 2B F2, 532.6 s.

The predictions are in `results/zeroshot/vsibench/concise_math_cot/dev_v1/` (the run folder `dev_v1`, from the notebook's
`cot_9b_results_main.zip`, SHA-256 `5db6d4d61a28374f0729477fae705d53e6b7ffc95c96123fadaf52e774dfb4c3`; its top-level
`analysis.md` and `errors.csv` are the smoke stage's).

## Results

From `model_size.md` (30 questions, 10 per task; written before the 2B's F2 ran, which is experiment 6's 57.0):

| Model | Condition | Computing | Abs. dist. | Rel. dist. | Obj. size | Parsed | Reply tokens, median / p90 | Cut off | s / question |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B | A | 57.0 | 43.0 | 70.0 | 58.0 | 100.0% | 3 / 3 | 0.0% | - |
| 2B | C | 52.0 | 27.0 | 70.0 | 59.0 | 100.0% | 4 / 5 | 0.0% | - |
| 2B | D | 66.7 | 40.0 | 80.0 | 80.0 | 100.0% | 536 / 1548 | 3.3% | - |
| 2B | E | 85.3 | 97.0 | 100.0 | 59.0 | 100.0% | 4 / 5 | 0.0% | - |
| 2B | F1 | 64.7 | 45.0 | 70.0 | 79.0 | 100.0% | 143 / 463 | 0.0% | 24.1 |
| 9B | A | 53.3 | 54.0 | 60.0 | 46.0 | 100.0% | 2 / 3 | 0.0% | 4.0 |
| 9B | C | 80.0 | 71.0 | 70.0 | 99.0 | 100.0% | 4 / 4 | 0.0% | 4.2 |
| 9B | E | 98.7 | 97.0 | 100.0 | 99.0 | 100.0% | 4 / 5 | 0.0% | 4.2 |
| 9B | F1 | 76.3 | 49.0 | 80.0 | 100.0 | 100.0% | 174 / 952 | 0.0% | 27.7 |
| 9B | F2 | 79.3 | 98.0 | 40.0 | 100.0 | 83.3% | 406 / 2048 | 16.7% | 46.9 |

| Comparison | Computing | 95% interval | Abs. dist. | Rel. dist. | Obj. size | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 9B F1 − 9B C | −3.7 | [−17.0, +9.0] | −22.0 | +10.0 | +1.0 | 4 / 9 / 17 | 0.267 | 0.534 |
| 2B F1 − 2B C | +12.7 | [−5.3, +31.0] | +18.0 | +0.0 | +20.0 | 10 / 6 / 14 | 0.454 | 0.534 |
| (9B F1 − 9B C) − (2B F1 − 2B C) | −16.3 | [−35.0, +1.3] | −40.0 | +10.0 | −19.0 | 6 / 13 / 11 | 0.167 | 0.501 |
| 9B C − 2B C | +28.0 | [+13.0, +42.7] | +44.0 | +0.0 | +40.0 | 15 / 4 / 11 | 0.0192 | 0.0961 |
| 9B E − 2B E | +13.3 | [+5.7, +21.0] | +0.0 | +0.0 | +40.0 | 6 / 1 / 23 | 0.125 | 0.5 |

### Examples

- **The formulas on the 9B** (F2): absolute distance 98.0 against F1's 49.0, relative distance 40.0 against 80.0; 5 of
  30 replies run to 2,048 tokens and are not read.

### What the results show

- **CoT does not help the 9B:** 9B F1 − 9B C is −3.7, mostly absolute distance (−22.0); the 2B gains +12.7 from the
  same prompt.
- **The 9B reads the sheet better when answering directly:** 9B C 80.0 against the 2B's 52.0, from absolute distance
  (+44.0) and object size (+40.0); the 9B takes the box's longest side for object size (99.0).
- **With the formulas, the 9B applies the closest-point rule** (absolute distance 98.0) but loses relative distance and
  a sixth of its replies to the limit.
- **No contrast clears Holm at 0.05.**

## Conclusion

- **Answer to the question:** on these 30 development questions, model size is not why concise CoT fails to beat
  direct answering with the gold sheet: the 9B gains nothing from F1 over its own direct answers, which are already
  much better than the 2B's.
- **Hypothesis:** "size is not the reason, at this scale" (9B F1 − 9B C ≤ 0); the interaction is negative (−16.3), its
  interval reaching +1.3.
- **Caveats:** exploratory: 30 development questions, one run per condition, batches that differ between the models;
  intervals that only differences of about 15 points can clear.
- **Next** (the plan's branch for "size is not the reason"): the prompt and the sheet's form are the next suspects;
  the 9B's direct answers with the sheet (C) are the stronger baseline.

## Planned: full-benchmark rerun (not run)

The rerun planned in this folder's `README.md` (sections 3 and 4: its hypotheses, comparisons and shared runs),
as written there on 2026-11-28. Nothing in this section has been run; the results above are the exploratory runs'.

### Conditions

All six have the video and the gold sheet (csv).

| ID | `visual.model` | `prompt.reasoning` | Run |
|---|---|---|---|
| S2D | Qwen/Qwen3.5-2B | direct | R3 |
| S2C | Qwen/Qwen3.5-2B | CoT | R4 |
| S4D | Qwen/Qwen3.5-4B | direct | R20 |
| S4C | Qwen/Qwen3.5-4B | CoT | R21 |
| S9D | Qwen/Qwen3.5-9B | direct | R22 |
| S9C | Qwen/Qwen3.5-9B | CoT | R23 |

### Runs

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
BASE="--nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv --sheet-source gold"
COT="--reasoning <TODO C4>"
M4="--model Qwen/Qwen3.5-4B"
M9="--model Qwen/Qwen3.5-9B --device-map auto <TODO C5: --batch-size>"
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** As experiment 2, check 1.
2. **Smoke test** per model, direct and CoT, then each model's revision. Report each model's peak GPU memory and
   seconds per question.

   ```bash
   for m in 4 9; do
     M=M$m
     python -m mvp.eval.vsibench ${!M} $BASE $GOLD      --ids $C/smoke_ids.txt --out $S/qwen3_5_${m}B_direct.jsonl
     python -m mvp.eval.vsibench ${!M} $BASE $GOLD $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_${m}B_cot.jsonl
   done
   python $C/cluster_checks.py revision Qwen/Qwen3.5-4B $OUT/model_revision_qwen3_5_4B.txt
   python $C/cluster_checks.py revision Qwen/Qwen3.5-9B $OUT/model_revision_qwen3_5_9B.txt
   ```

**Runs:** R3 and R4 come from experiment 5. The other four each report 5,130 questions:

```bash
python -m mvp.eval.vsibench $M4 $BASE $GOLD      --out $OUT/qwen3_5_4B_R20.jsonl
python -m mvp.eval.vsibench $M4 $BASE $GOLD $COT --out $OUT/qwen3_5_4B_R21.jsonl
python -m mvp.eval.vsibench $M9 $BASE $GOLD      --out $OUT/qwen3_5_9B_R22.jsonl
python -m mvp.eval.vsibench $M9 $BASE $GOLD $COT --out $OUT/qwen3_5_9B_R23.jsonl
```

**Analysis:** `<make_table_7.py>` (`README.md` section 5, item 5) writes the Results table.

### Open questions

- **C1:** the structured-CoT prompt (experiment 5).
- **C4:** the CoT prompt here. Either C1's prompt for every question type, or a concise variant (experiment 6) on the
  three computing tasks with C1's prompt elsewhere.
- **C5:** the 9B's batch size. Either the 2B's and 4B's 8, if memory allows, or one smaller size, for the 9B alone or
  for all three models, stated as a deviation.

## Log

- 2026-11-25: the model-size check planned (concise-math-cot README, "Model size") before any 9B answer existed.
- 2026-11-25: smoke and main stages run in `cot_9b.ipynb` on an L4 at `e2cd67a`, then 9B F2 and the 2B's F2.
- 2026-11-28: this file rewritten to describe the check as run (2B and 9B, 30 questions); the three-model plan and its
  open questions removed.

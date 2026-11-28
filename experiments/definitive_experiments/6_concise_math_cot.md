---
id: definitive-6-concise-math-cot
title: Concise, mathematical and few-shot structured CoT for Qwen3.5-2B on the computing tasks with the gold sheet
owner: Tamara Grujičić
created: 2026-11-27
status: draft                # draft | planned | running | analysing | done | abandoned
tags: [prompting, chain-of-thought, gold-sheet, definitive]
varies: [prompt.reasoning]
depends_on: [definitive-5-structured-cot (runs R1, R3, R4, R5), TODOs C1–C3]
---

# Concise, mathematical and few-shot structured CoT

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder).

## Question

Can a more constrained, mathematically explicit CoT, with or without one worked example per task, make Qwen3.5-2B
answer the computing tasks better from the gold sheet than structured CoT (D), than direct answering (C) and than the
video alone (A)? How far does it stay from the given relations (E)?

## Hypothesis and what would change our mind

- **Expected:**
  - F1 − D ≥ 0, with shorter replies and fewer cut off: the pilot's loops came from its check against the video.
  - F2 − F1 > 0, mostly on absolute distance, where the pilot computed centre distances. Also on relative distance,
    where the centre and closest-point rules pick different answers.
  - F3 − F2 ≥ 0, small.
  - E − F2 > 0: E is given the distances that F2 has to compute.
- **Would prove it wrong:** F2 ≤ F1 on absolute distance (the model does not apply a formula it is given), F1 below D,
  or more answers that can't be read.
- **Next:**
  - F2 or F3 beats D and C: use it with the pipeline's sheets, which needs its own wording of the units;
  - none does: precomputed relations (E) remain the way to give the model distances.

## Background

- **The pilot's computing questions** (`experiments/2026-11-19-structured-cot/explore_cot.md`; Qwen3.5-2B, 75
  questions): A 61.5, C 48.5, D 58.0, E 82.0.
  - 84% of D's absolute-distance answers were the centre-to-centre distance.
  - 11.5% of D's replies looped to the limit.
- **The rules, applied to the gold sheet on the full benchmark**
  (`experiments/2026-11-25-concise-math-cot/README.md`):
  - absolute distance: closest-point gap 91.4, centre-to-centre 34.3;
  - relative distance: nearest by closest points 98.3, by centres 83.7;
  - object size: largest side × 100, 100.0.
- **The prompts:** `src/mvp/benchmarks/cot_variants.py`, `PROMPT_VERSION` `2026-11-25.1`. Each variant adds one piece
  to the last:
  - F1: `CONCISE`, the task's hint and the `FORMAT` lines;
  - F2: F1 plus `MATH`, the box gap or the size rule;
  - F3: F2 plus one synthetic worked example.

  They are defined only for these three tasks, with the gold sheet.

## Design

### Conditions

| ID | Condition | `prompt.reasoning` | Role | Run |
|---|---|---|---|---|
| A | Video only, direct | direct | baseline | R1, computing questions |
| C | Video + gold sheet, direct | direct | baseline | R3, computing questions |
| D | Video + gold sheet, structured CoT | CoT (TODO C1) | baseline | R4, computing questions |
| E | Video + gold sheet + relations, direct | direct | upper reference | R5, computing questions |
| F1 | Video + gold sheet, concise CoT | `cot_f1` | treatment | R17 |
| F2 | F1 + geometric formulas | `cot_f2` | treatment | R18 |
| F3 | F2 + one worked example | `cot_f3` | treatment | R19 |

### Comparisons

All on the computing score (the mean of the three tasks' scores) and per task.

- **Primary, one Holm family:** F1 − D, F2 − F1 and F3 − F2.
- **Secondary**, each its own Holm family: every F against C; every F against E; every F against A.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | all 2,497 computing questions (`computing_ids.txt`, SHA-256 `ed4b146f…0a5c`): 834 absolute distance, 710 relative distance, 953 object size | all scenes; the variants are defined for these tasks only |
| Baselines | the shared runs on all 5,130 questions, read on the computing ones | one run per configuration (`README.md` section 4); batches: TODO C3 |
| Prompt text | `PROMPT_VERSION` `2026-11-25.1`, unless TODO C2 changes it | never run on a GPU yet |

## Configuration

```yaml
# README.md section 2, except:
data:
  subset: experiments/definitive_experiments/computing_ids.txt
  question_types: [object_abs_distance, object_rel_distance, object_size_estimation]
  n_questions: 2497
scene_sheet:
  source: gold                 # gold_relations in E
prompt:
  reasoning: varies            # direct (A, C, E) | CoT (D, TODO C1) | cot_f1 | cot_f2 | cot_f3
```

## Metrics and analysis

As `README.md` 2.7, with its CoT measures, and `errors.csv` as
`experiments/2026-11-25-concise-math-cot/README.md` defines it:

- automatic flags per reply: cut off; no "Answer:" line; unread; the answer equal to the centre rule or to the
  closest-point rule; an option picked by centres where closest points pick another; metres for centimetres; another
  side of the box; the gold rule disagreeing with the truth; named objects missing from the sheet;
- empty columns for what needs reading: identification, attributes, operation, arithmetic, unit.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet gives the object-size answer (its largest side) | object size near 100 | read object size apart; the gains that matter are on distances |
| F3 copies the worked example | answers equal to 1.059, 183 or the example's letter | the example says it is for another scene; check the flags and replies |
| Closest points between axis-aligned boxes | absolute-distance errors on large or rotated objects | stated; the same approximation as E |
| F1 changes the instructions and the output form together | — | stated; both target the length, and they are not separated |

## Runs and cost

```bash
D=experiments/definitive_experiments
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv --sheet-source gold"
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** As experiment 2, check 1. `computing_ids.txt` must have SHA-256
   `ed4b146f98e064965e2783fedcae120637e96c4bfc1afbfa2503d0e6050a0a5c`.
2. **Tests.** Both must pass:

   ```bash
   PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_cot_variants
   PYTHONPATH=src python -m unittest discover -s experiments/2026-11-25-concise-math-cot -p "test_*.py"
   ```

3. **Smoke test** of F1–F3 on the 6 computing questions of the smoke set (`computing_smoke_ids.txt`, SHA-256
   `7f0e27eb891134bba048a7255fe01487ae21ee2482390626c223aa727da98e49`). Print every prompt and reply. Each run must
   answer all 6, every reply must have an "Answer:" line, and no F3 answer may be the example's (1.059, 183).

   ```bash
   for v in 1 2 3; do
     python -m mvp.eval.vsibench $COMMON $GOLD --reasoning cot_f$v --ids $D/computing_smoke_ids.txt \
         --out $S/qwen3_5_2B_R$((16 + v)).jsonl
   done
   ```

**Runs:** R1, R3, R4 and R5 come from experiment 5. They are on all 5,130 questions and are read here on the 2,497.
F1–F3 each report 2,497 questions:

```bash
python -m mvp.eval.vsibench $COMMON $GOLD --reasoning cot_f1 --ids $D/computing_ids.txt --out $OUT/qwen3_5_2B_R17.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --reasoning cot_f2 --ids $D/computing_ids.txt --out $OUT/qwen3_5_2B_R18.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --reasoning cot_f3 --ids $D/computing_ids.txt --out $OUT/qwen3_5_2B_R19.jsonl
```

**Analysis:** `<make_table_6.py>` (`README.md` section 5, item 5), built on
`experiments/2026-11-25-concise-math-cot/analysis.py`, writes the Results table and `errors.csv`.

## TODO

- **C1:** the structured-CoT prompt, D's (experiment 5). If it changes, F1 − D no longer compares F1's pieces with the
  pilot's prompt.
- **C2:** freeze `PROMPT_VERSION` `2026-11-25.1` as it is, or run its development stages on explore200 first.
- **C3:** A, C, D and E run in batches that mix question types, and F1–F3 on the computing questions alone. Accept it
  as a stated deviation, or rerun A, C, D and E on `computing_ids.txt`?

## Results

<!-- make_table:start -->
Not run yet.
<!-- make_table:end -->

### Examples

### What the results show

## Conclusion

- **Answer to the question:**
- **Hypothesis:**
- **Caveats:**
- **Next:**

## Log

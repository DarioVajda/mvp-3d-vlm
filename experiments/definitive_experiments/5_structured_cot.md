---
id: definitive-5-structured-cot
title: Does structured chain-of-thought help Qwen3.5-4B use a gold scene sheet on VSI-Bench, and does it recover what precomputed relations give
owner: Tamara Grujičić
created: 2026-11-27
status: draft                # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, prompting, chain-of-thought, definitive]
varies: [prompt.reasoning, scene_sheet.source]
depends_on: [gold sheets in tasks/openvocab_adapter/out_full (README.md 2.4), TODO C1]
---

# Structured chain-of-thought

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once
TODO C1 is settled.

## Question

Does step-by-step reasoning help Qwen3.5-4B use a correct scene sheet, and does it recover what precomputed relations
give it?

## Hypothesis and what would change our mind

- **Expected:**
  - D − C > 0: the sheet supplies the numbers, and the model can compute from them in writing.
  - C < D < E: CoT recovers part of the gap to the relations, not all of it.
  - B − A ≈ 0 or below: without a sheet, reasoning adds little, as VSI-Bench's authors found for linguistic prompting.
- **Would prove it wrong:**
  - D − C ≤ 0, overall and on the computing tasks;
  - D ≥ E, so precomputing relations adds nothing;
  - B − A clearly positive, which would point to the prompt rather than the sheet.
- **Next:**
  - CoT helps and closes much of the gap to E: try it with the pipeline's sheet;
  - it helps but leaves most of the gap: relations are the better way to give spatial facts;
  - it doesn't help: relations, and experiment 7 for whether size matters.

## Background

- **The 200-question pilot** (`experiments/2026-11-19-structured-cot/explore_cot.md`): Qwen3.5-2B on explore200.
  - D − C −3.3 (Holm p = 0.27), B − A −13.3 (Holm p = 0.004), E − D +13.9 (Holm p = 0.0002).
  - With the sheet, D computed centre-to-centre distances with correct arithmetic. VSI-Bench measures between closest
    points.
  - 12% of D's replies and 28% of B's looped until the 2,048-token limit.
- **The prompt:** `--reasoning cot` (`cot_prompt`; its text in that folder's `cot_prompt.md`). It asks the model to:
  - say which objects the question is about;
  - write down the facts it needs;
  - work out the answer one step at a time, showing the arithmetic;
  - check the result against the video;
  - end on "Answer: X".

  Each task adds one hint.
- **The literature:**
  - CoT's gains appear mainly in large models (Wei et al. 2022);
  - linguistic prompting did not improve VSI-Bench (Yang et al. 2025, "Thinking in Space").

## Design

### Conditions

| ID | Condition | `prompt.reasoning` | `scene_sheet.source` | Role | Run |
|---|---|---|---|---|---|
| A | Video only, direct | direct | none | baseline | R1 |
| B | Video only, CoT | CoT | none | treatment | R2 |
| C | Video + gold sheet, direct | direct | gold | control | R3 |
| D | Video + gold sheet, CoT | CoT | gold | treatment | R4 |
| E | Video + gold sheet + relations, direct | direct | gold_relations | upper reference | R5 |

### Comparisons

- **One Holm family:** D − C (primary), B − A and E − D.
- **Descriptive:** E − C, and (D − C) − (B − A).
- Each comparison also without counting, and on the computing tasks (absolute distance, relative distance, object size)
  alone.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| CoT prompt | `<TODO C1>` | open since 2026-11-20 |
| Token limit | 2,048, with no rule to raise it | the pilot's cut-off replies were loops, which a higher limit would not end |
| Thinking mode | off, no condition for it | dropped on 2026-11-20; the reasoning comes from the prompt |

## Configuration

```yaml
# README.md section 2, except:
scene_sheet:
  source: varies               # none (A, B) | gold (C, D) | gold_relations (E)
prompt:
  reasoning: varies            # direct (A, C, E) | CoT, <TODO C1> (B, D)
```

## Metrics and analysis

As `README.md` 2.7, with its CoT measures. In addition, 20 wrong D answers from the computing tasks, drawn with seed
0, labelled by hand: wrong row, coordinates misread, arithmetic wrong, sheet ignored, or no answer line.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| CoT replies loop until the limit and lose for that alone | the share cut off and looping, per condition | report the score with cut-off replies counted wrong too |
| The gold sheet contains the counting answer, and its largest side is the object-size answer | counting and object size near 100 in C–E | read every comparison without counting; read object size apart |
| The relations state the answer to most distance questions | E near its ceiling there | E is an upper reference for using given relations, not a test of reasoning |
| A 4B model may still be too small to gain from CoT | D − C and B − A both near 0 | read as a result about this size; experiment 7 varies it |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
COMMON="--model Qwen/Qwen3.5-4B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv"
COT="--reasoning <TODO C1>"                    # cot as the code stands
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** As experiment 2, check 1.
2. **Tests.** `PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet` must pass.
3. **Prompt check** for D, as `experiments/2026-11-19-gold-sheet-relations/README.md`'s Runs step 2, with
   `--sheet-source gold $COT --model Qwen/Qwen3.5-4B`. Compare the CoT text with `cot_prompt.md` (or its replacement,
   TODO C1).
4. **Smoke test** of A and D, then B, then the model revision. For B and D, also report reply length, the share cut off
   and the parse rate.

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R4.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_4B_R1.jsonl $S/qwen3_5_4B_R4.jsonl --n 20
   python -m mvp.eval.vsibench $COMMON $COT --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R2.jsonl
   python $C/cluster_checks.py revision Qwen/Qwen3.5-4B $OUT/model_revision_qwen3_5_4B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON                                   --out $OUT/qwen3_5_4B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $COT                              --out $OUT/qwen3_5_4B_R2.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold         --out $OUT/qwen3_5_4B_R3.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold $COT    --out $OUT/qwen3_5_4B_R4.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations --out $OUT/qwen3_5_4B_R5.jsonl
```

**Analysis:** `<make_table_5.py>` (`README.md` section 5, item 6) writes the Results table and the CoT measures.

## TODO

- **C1** (`README.md` section 5): the CoT prompt. Either `--reasoning cot` as it stands, whose check against the video
  made 12% (gold) and 28% (no sheet) of the pilot's replies loop, or a revised version for all question types, such as
  one without that step.

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

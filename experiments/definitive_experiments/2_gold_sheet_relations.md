---
id: definitive-2-gold-sheet-relations
title: Do correct scene sheets, and precomputed relations, help Qwen3.5-4B on VSI-Bench, with and without the video
owner: Tamara Grujičić
created: 2026-11-27
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, relations, definitive]
varies: [scene_sheet.source, visual.video]
depends_on: [gold sheets in tasks/openvocab_adapter/out_full (README.md 2.4)]
---

# Gold sheet and relations

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder).

## Question

Do correct scene sheets, and precomputed relations, help Qwen3.5-4B on VSI-Bench? How much of it does the sheet carry
without the video, and at what cost in input tokens (frames only, sheet only, both)?

## Hypothesis and what would change our mind

- **Expected:**
  - B − A > 0, almost all of it on counting; without counting the sheet adds little.
  - C − B > 0, almost all of it on absolute and relative distance, where the relations state the answer and the model
    copies it. On relative direction, appearance order and route planning, which also get relations, C − B is near 0
    or negative.
  - Without the video, D and E keep what the sheet states: counting, and with relations the distances. They fall on
    appearance order, and their overall is below B's and C's.
- **Would prove it wrong:**
  - C − B ≤ 0, or within the smallest difference that matters.
  - C − B helps (at least 5 points, Holm p < 0.05) on a type with relations other than absolute and relative distance.
  - B − A clearly positive without counting.
  - D ≥ B overall, within the smallest difference that matters.
- **Next:**
  - relations help beyond the distance questions: give the pipeline's sheet the same relations (experiment 3);
  - they help only where they state the answer: precompute what each question type needs, or have the model reason
    (experiments 5–7);
  - they don't help: the sheet's value for this model is counting.

## Background

- **Qwen3.5-2B on all 5,130 questions**, the same five conditions at `0f7c5f2`
  (`experiments/2026-11-19-gold-sheet-relations/`, `results/zeroshot/vsibench/gold_sheet_relations_full/`). Background
  only, since the model differs:

  | Condition | Overall | Notable tasks |
  |---|---:|---|
  | A | 43.3 | |
  | B | 48.9 | counting 40.1 → 98.7; appearance order 45.0 → 34.1 |
  | C | 61.3 | absolute distance 34.6 → 91.2; relative distance 46.3 → 96.3 |
  | D | 39.9 | |
  | E | 55.0 | |

- **The model copies given relations,** even wrong ones
  (`experiments/2026-11-18-scene-sheet-formats/pipeline_tests.md`): 80 of 80 distances and 25 of 25 closest objects.

## Design

### Conditions

| ID | Condition | `scene_sheet.source` | `visual.video` | Role | Run |
|---|---|---|---|---|---|
| A | Video only | none | true | baseline | R1 |
| B | Video + gold sheet | gold | true | treatment | R3 |
| C | Video + gold sheet + relations | gold_relations | true | treatment | R5 |
| D | Gold sheet only | gold | false | sheet only | R6 |
| E | Gold sheet + relations only | gold_relations | false | sheet only | R7 |

### Comparisons

- **First family (Holm):** B − A and C − B (primary). Both are also read without counting. C − B is also read on the
  distance types and on the other types with relations, apart.
- **Second family (Holm):** B − D, E − C and D − A.
- **Descriptive:** C − A, and the input tokens of A (frames only), D and E (sheet only), and B and C (both).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Which questions get relations | as `gold_relations` writes them (`README.md` 2.4): a distance table for 3,210 of the 5,130, an answer line for absolute and relative distance | the 2B runs' and the pilots' condition, unchanged |
| Sheet only | `--no-video`: the prompt of B or C without the video and its sentence | the proposal asks for frames only, sheet only and both |

## Configuration

```yaml
# README.md section 2, except:
visual:
  video: varies                # true (A, B, C) | false (D, E: --no-video)
scene_sheet:
  source: varies               # none (A) | gold (B, D) | gold_relations (C, E)
```

## Metrics and analysis

As `README.md` 2.7. In addition, the share of questions that get relations, and the scores with relations and
without.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet contains the counting answer | counting near 100 in B–E | read every comparison without counting too |
| The relations state the answer to most distance questions, and the model copies them | C near its ceiling on absolute and relative distance | read C − B on the other types with relations apart |
| Closest points between axis-aligned boxes can be off for objects at an angle | C's absolute-distance errors on large or diagonal objects | stated |
| Relations make some prompts very long (up to 48k characters, in ScanNet++ rooms with many chairs) | C's input tokens | report tokens; break down by the number of relation lines if needed |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
COMMON="--model Qwen/Qwen3.5-4B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheets-root tasks/openvocab_adapter/out_full --sheet-format csv"
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** Build and check as `experiments/2026-11-19-gold-sheet-relations/README.md`, Runs step
   1. The build must end with `sheets failed: 0` and its two SHA-256, and `check_sheets.py` with `OK`.
2. **Tests.** `PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet` must pass, `NoVideo` included.
3. **Prompt check** for C, as that README's Runs step 2, with `--model Qwen/Qwen3.5-4B`.
4. **Smoke test** of A, C and E, then the model revision. E's 20 lines must each have `"n_frames": 0`.

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R5.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_4B_R1.jsonl $S/qwen3_5_4B_R5.jsonl --n 20
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations --no-video --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R7.jsonl
   python $C/cluster_checks.py revision Qwen/Qwen3.5-4B $OUT/model_revision_qwen3_5_4B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_4B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold                      --out $OUT/qwen3_5_4B_R3.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations            --out $OUT/qwen3_5_4B_R5.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold           --no-video --out $OUT/qwen3_5_4B_R6.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-source gold_relations --no-video --out $OUT/qwen3_5_4B_R7.jsonl
```

**Analysis:** `<make_table_2.py>` (`README.md` section 5, item 6) writes the Results table.

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

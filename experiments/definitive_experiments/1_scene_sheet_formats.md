---
id: definitive-1-scene-sheet-formats
title: Which format of the gold scene sheet helps Qwen3.5-2B most on VSI-Bench, against video alone
owner: Tamara Grujičić, Nicolás Ramos Fernandez
created: 2026-11-27
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, format, gold-sheet, definitive]
varies: [scene_sheet.source, scene_sheet.format]
depends_on: [gold sheets in tasks/openvocab_adapter/out_full (README.md 2.4)]
---

# Scene-sheet formats

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once the
code of `README.md` section 5 exists.

## Question

Which way of writing the gold sheet (csv, markdown table, json or text) helps Qwen3.5-2B most over the video alone?
The sheet's content is the same in every format.

## Hypothesis and what would change our mind

- **Expected:** every format beats A overall, almost all of it on counting, where the gold sheet holds the answer. The
  differences between formats are smaller than the gain from adding the sheet.
- **Would prove it wrong:** the spread between the best and the worst format is at least the best format's gain over
  A, so the format matters as much as the sheet itself.
- **Next:**
  - formats equal within the smallest difference that matters: keep csv, the shortest, in every later sheet;
  - one format clearly best: use it in later experiments, and repeat the best and the worst on a second model first.

## Background

- **The 2026-11-19 gold runs** (`experiments/2026-11-18-scene-sheet-formats/README.md`, Results, G–K): Qwen3.5-2B on
  2,728 questions, with gold sheets built before `980cf00` (boxes not yet fitted to their objects). Video only scored
  43.2.

  | Format | Overall | Mean sheet tokens |
  |---|---:|---:|
  | csv | 49.4 | 215 |
  | markdown table | 49.4 | 269 |
  | json | 49.4 | 341 |
  | markdown key-value | 48.4 | 305 |
  | text | 48.7 | 281 |

- In every format, counting rose from 39.6 to about 98, and without counting each format was below video only.
- **Experiment 2** (`2_gold_sheet_relations.md`): the gold sheet in csv on all 5,130 questions, 48.9 against video
  only's 43.3, all of it from counting.

## Design

### Conditions

| ID | Condition | `scene_sheet.source` | `scene_sheet.format` | Role | Run |
|---|---|---|---|---|---|
| A | Video only | none | none | baseline | R1 |
| B | Video + gold sheet | gold | csv | treatment | R3 |
| C | Video + gold sheet | gold | markdown_table | treatment | R6 |
| D | Video + gold sheet | gold | json | treatment | R7 |
| E | Video + gold sheet | gold | text | treatment | R8 |

### Comparisons

- **Primary**, one Holm family: B − A, C − A, D − A and E − A.
- **Secondary**, one Holm family: the six pairs of formats.
- All also without counting.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Sheet | gold only | correct content, so that the pipeline's errors don't hide a format effect |
| Formats | csv, markdown_table, json, text | markdown_kv was the worst in the 2026-11-19 runs (gold 48.4, pipeline 25.5) |
| Relations, cameras | none | so that the format is the only difference |

## Configuration

```yaml
# README.md section 2, except:
scene_sheet:
  source: varies               # none (A) | gold (B–E)
  format: varies               # csv | markdown_table | json | text
```

## Metrics and analysis

As `README.md` 2.7. In addition, sheet tokens per format (mean, median, max) with the Qwen3.5-2B tokenizer.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet contains the counting answer, so every format gains there alike | counting near 100 in B–E | read every comparison without counting too |
| The formats differ in length (json about 1.6× csv) | gains that track token counts | report tokens next to every score |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
GOLD="--sheet-source gold --sheets-root tasks/openvocab_adapter/out_full"
```

**Checks** (`README.md` 2.8):

1. **Inputs and gold sheets.** Build and check as `experiments/2026-11-19-gold-sheet-relations/README.md`, Runs step
   1. The build must end with `sheets failed: 0` and its two SHA-256, and `check_sheets.py` with `OK`.
2. **Tests and render check.** `PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet` and
   `<README.md section 5, item 4>`: all must pass, with none skipped.
3. **Prompt check**, every format: as that README's Runs step 2, with `--model Qwen/Qwen3.5-2B --sheet-source gold`
   and each `--sheet-format`.
4. **Smoke test** of A and B, then the model revision:

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $GOLD --sheet-format csv --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R3.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_2B_R1.jsonl $S/qwen3_5_2B_R3.jsonl --n 20
   python $C/cluster_checks.py revision Qwen/Qwen3.5-2B $OUT/model_revision_qwen3_5_2B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-format csv            --out $OUT/qwen3_5_2B_R3.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-format markdown_table --out $OUT/qwen3_5_2B_R6.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-format json           --out $OUT/qwen3_5_2B_R7.jsonl
python -m mvp.eval.vsibench $COMMON $GOLD --sheet-format text           --out $OUT/qwen3_5_2B_R8.jsonl
```

**Analysis:** `<make_table_1.py>` (`README.md` section 5, item 5) writes the Results table.

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

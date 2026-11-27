---
id: definitive-1-scene-sheet-formats
title: Which format of pipeline_v4.1's scene sheet helps Qwen3.5-4B most on VSI-Bench, against video alone
owner: Tamara Grujičić, Nicolás Ramos Fernandez
created: 2026-11-27
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, format, pipeline, definitive]
varies: [scene_sheet.source, scene_sheet.format]
depends_on: [README.md section 5, items 1–6 (pipeline_v4.1's sheets and the eval's support for them)]
---

# Scene-sheet formats

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once the
code of `README.md` section 5 exists.

## Question

Which way of writing pipeline_v4.1's sheet (csv, markdown table, json or text) helps Qwen3.5-4B most over the video
alone? The sheet's content is the same in every format.

## Hypothesis and what would change our mind

- **Expected:** the sheet adds little over A, or nothing. The differences between formats are smaller than the
  difference between having the sheet and not. The tables (csv, markdown table) are at least as good as json and text.
- **Would prove it wrong:** the spread between the best and the worst format is at least the best format's gain over
  A, so the format matters as much as the sheet itself.
- **Next:**
  - formats equal within the smallest difference that matters: keep csv, the shortest, in every later sheet;
  - one format clearly best: use it in later pipeline experiments, and repeat the best and the worst on a second model
    first.

## Background

- `experiments/2026-11-18-scene-sheet-formats/README.md`, Results. That run used Qwen3.5-2B on 2,728 questions, with
  sheets in relative units and first-camera axes. Overall scores:

  | Condition | Overall |
  |---|---:|
  | Video only | 43.2 |
  | csv | 34.9 |
  | markdown table | 35.1 |
  | json | 31.3 |
  | markdown key-value | 25.5 |
  | text | 29.0 |

- The model read the relative units as metres, so the metric tasks fell 26–31 points in every format.
- The five gold formats were within 1 point of each other.

## Design

### Conditions

| ID | Condition | `scene_sheet.source` | `scene_sheet.format` | Role | Run |
|---|---|---|---|---|---|
| A | Video only | none | none | baseline | R1 |
| B | Video + pipeline sheet | pipeline | csv | treatment | R8 |
| C | Video + pipeline sheet | pipeline | markdown_table | treatment | R9 |
| D | Video + pipeline sheet | pipeline | json | treatment | R10 |
| E | Video + pipeline sheet | pipeline | text | treatment | R11 |

### Comparisons

- **Primary**, one Holm family: B − A, C − A, D − A and E − A.
- **Secondary**, one Holm family: the six pairs of formats.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Formats | csv, markdown_table, json, text | markdown_kv was the worst in the 2026-11-19 run (25.5, against 34.9–35.1 for the tables) |
| Gold arms | none | the gold formats were within 1 point of each other; experiment 2 has gold csv |
| Relations, cameras | none | so that the format is the only difference |

## Configuration

```yaml
# README.md section 2, except:
scene_sheet:
  source: varies               # none (A) | pipeline: pipeline_v4.1 (B–E)
  format: varies               # csv | markdown_table | json | text
```

## Metrics and analysis

As `README.md` 2.7. In addition, sheet tokens per format (mean, median, max) with the Qwen3.5-4B tokenizer, and the
scores on empty and non-empty sheets.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The formats differ in length (json and text are about 1.5× csv) | gains that track token counts | report tokens next to every score |
| Empty sheets dilute the format effect | the same scores in B–E on them | report empty and non-empty sheets apart |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
ROOT=tasks/openvocab_adapter/out_v4_1         # the pipeline_v4.1 build (README.md section 5, item 3)
COMMON="--model Qwen/Qwen3.5-4B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
PIPE="--sheet-source pipeline --sheets-root $ROOT --metric-scale <SHEET_SCALE>"
```

**Checks** (`README.md` 2.8):

1. **Inputs.** `test.jsonl`'s SHA-256 (`README.md` 2.3); 288 `data/vggt/*/*/frame_indices.npy`; `$ROOT/index.csv` with
   SHA-256 `<from the build>`.
2. **Tests and render check.** `<README.md section 5, items 4–5>`: all must pass, with none skipped.
3. **Prompt check**, every format:

   ```bash
   for f in csv markdown_table json text; do
     python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format $f --ids $C/prompt_check_ids.txt --print-prompt 4 > prompts_$f.txt
     python $C/cluster_checks.py prompts prompts_$f.txt --sheets-root $ROOT --format $f
   done
   ```

4. **Smoke test** of A and B, then the model revision:

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv --ids $C/smoke_ids.txt --out $S/qwen3_5_4B_R8.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_4B_R1.jsonl $S/qwen3_5_4B_R8.jsonl --n 20
   python $C/cluster_checks.py revision Qwen/Qwen3.5-4B $OUT/model_revision_qwen3_5_4B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_4B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv            --out $OUT/qwen3_5_4B_R8.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format markdown_table --out $OUT/qwen3_5_4B_R9.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format json           --out $OUT/qwen3_5_4B_R10.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format text           --out $OUT/qwen3_5_4B_R11.jsonl
```

**Analysis:** `<make_table_1.py>` (`README.md` section 5, item 6) writes the Results table.

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

---
id: definitive-3-pipeline-sheet-vs-video
title: Does pipeline_v4's scene sheet, and its relations, help Qwen3.5-2B on VSI-Bench, with and without the video
owner: Nicolás Ramos Fernandez
created: 2026-11-27
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, pipeline, relations, definitive]
varies: [scene_sheet.source, visual.video]
depends_on: [README.md section 5, items 1–5, definitive-2-gold-sheet-relations (the gold twins)]
---

# Pipeline sheet vs video

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once the
code of `README.md` section 5 exists.

## Question

Does pipeline_v4's sheet, and its relations, help Qwen3.5-2B over the video alone? How much is left without the
video, and how far is each condition from its gold twin in experiment 2?

## Hypothesis and what would change our mind

- **Expected:**
  - with relations (C), the distance questions score higher than with the video alone (H1), and so does relative
    direction (H2);
  - the full method, which takes C's answers on the four types the sheet is meant for and A's elsewhere, raises the
    overall (H3).

  These are `experiments/2026-11-18-scene-sheet-formats/prereg_batch2_A.md`'s H1–H3, the newest pipeline hypotheses.
- **Would prove it wrong:** C − A ≤ 0 on the distance questions.
- **Next:**
  - supported: pipeline − gold shows where better detection would still add;
  - not supported: read pipeline − gold per question type, which shows whether the sheet's content or its relations
    fall short.

## Background

- **No version of the pipeline sheet has beaten video only on held-out questions**
  (`experiments/2026-11-18-scene-sheet-formats/pipeline_tests.md`). The registered pipeline_v4 on batch2_A
  (Qwen3.5-2B, the same pipeline in first-camera axes):
  - distance questions: 53 helped, 89 hurt (p = 0.0032);
  - relative direction: 20 helped, 14 hurt (p = 0.39);
  - the full method's overall: +0.28 [−2.42, +2.97].
- **The model copies the relations it is given,** even wrong ones (same file). So C on the distance questions mostly
  measures how accurate the relations are.
- **The gold twins** (experiment 2, `2_gold_sheet_relations.md`), all 5,130 questions: A 43.3, B 48.9, C 61.3,
  D 39.9, E 55.0 (`results/zeroshot/vsibench/gold_sheet_relations_full/`, at `0f7c5f2`).

## Design

### Conditions

| ID | Condition | `scene_sheet.source` | `visual.video` | Role | Run | Gold twin |
|---|---|---|---|---|---|---|
| A | Video only | none | true | baseline | R1 | 2A |
| B | Video + pipeline sheet | pipeline | true | treatment | R9 | 2B |
| C | Video + pipeline sheet + relations | pipeline_relations | true | treatment | R11 | 2C |
| D | Pipeline sheet only | pipeline | false | sheet only | R12 | 2D |
| E | Pipeline sheet + relations only | pipeline_relations | false | sheet only | R13 | 2E |

### Comparisons

- **First family (Holm):** B − A and C − B. Also C − A on the distance questions (H1) and on relative direction (H2).
- **Second family (Holm):** B − D, E − C and D − A.
- **Third family (Holm):** pipeline − gold for each of B, C, D and E, against experiment 2's runs (`results/zeroshot/vsibench/gold_sheet_relations_full/`), per question type.
- **The full method (H3):** C's answer for absolute distance, relative distance, relative direction and route planning,
  A's for the rest. Compared with A, with its interval. It needs no extra run.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Format | csv | so that every condition pairs with its gold twin |
| Relations | pipeline_v4's (`README.md` 2.5): only distance and relative direction, only when every named object has a row | pipeline_v4's design; pipeline − gold on C and E therefore mixes content with coverage |
| Gold twins | experiment 2's runs, not rerun | they are the same conditions with the gold sheet, on the same questions, with the same model and settings |

## Configuration

```yaml
# README.md section 2, except:
visual:
  video: varies                # true (A, B, C) | false (D, E: --no-video)
scene_sheet:
  source: varies               # none (A) | pipeline (B, D) | pipeline_relations (C, E): pipeline_v4
```

## Metrics and analysis

As `README.md` 2.7. In addition, per question type, the share of questions with relations in pipeline_v4 and in
gold. For distance questions with a relation, its error against VSI-Bench's answer.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The relations cover fewer questions than gold's | pipeline − gold largest where gold has relations and the pipeline none | break pipeline − gold down by whether each has a relation |
| Without the video, appearance order and route planning have little to go on | D and E near chance there | expected; read B − D per task |
| The gold twins ran at another commit (`0f7c5f2`), possibly in another environment | — | stated; their prompts are the same as at later commits (experiment 2) |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
ROOT=tasks/openvocab_adapter/out_v4         # the pipeline_v4 build (README.md section 5, item 2)
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
PIPE="--sheets-root $ROOT --sheet-format csv --metric-scale <SHEET_SCALE>"
```

**Checks** (`README.md` 2.8):

1. **Inputs.** `test.jsonl`'s SHA-256 (`README.md` 2.3); 288 `data/vggt/*/*/frame_indices.npy`; `$ROOT/index.csv` and
   the relations file with SHA-256 `<from the build>`.
2. **Tests and render check.** `<README.md section 5, items 3–4>`, including a `--no-video` test of
   `pipeline_relations`: all must pass, with none skipped.
3. **Prompt check** for C:

   ```bash
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline_relations --ids $C/prompt_check_ids.txt --print-prompt 4 > prompts_C.txt
   python $C/cluster_checks.py prompts prompts_C.txt --sheets-root $ROOT --format csv
   ```

4. **Smoke test** of A, C and E, then the model revision. E's 20 lines must each have `"n_frames": 0`.

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline_relations --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R11.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_2B_R1.jsonl $S/qwen3_5_2B_R11.jsonl --n 20
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline_relations --no-video --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R13.jsonl
   python $C/cluster_checks.py revision Qwen/Qwen3.5-2B $OUT/model_revision_qwen3_5_2B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline                      --out $OUT/qwen3_5_2B_R9.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline_relations            --out $OUT/qwen3_5_2B_R11.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline           --no-video --out $OUT/qwen3_5_2B_R12.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-source pipeline_relations --no-video --out $OUT/qwen3_5_2B_R13.jsonl
```

**Analysis:** `<make_table_3.py>` (`README.md` section 5, item 5) writes the Results table, with pipeline − gold
against experiment 2's runs.

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

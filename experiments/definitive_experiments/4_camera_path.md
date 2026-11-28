---
id: definitive-4-camera-path
title: Does the camera's path, as a table, as text or in one narrative, add to pipeline_v4's objects for Qwen3.5-2B on VSI-Bench
owner: Nicolás Ramos Fernandez
created: 2026-11-27
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, cameras, narrative, pipeline, definitive]
varies: [scene_sheet.cameras, scene_sheet.format]
depends_on: [README.md section 5, items 1–5 (including cameras in metres on the sheet's scale)]
---

# Camera path

**Outcome:**

Everything not stated here is `README.md` section 2 (this folder). Text in `<angle brackets>` is filled in once the
code of `README.md` section 5 exists.

## Question

Does the camera's path add to pipeline_v4's objects for Qwen3.5-2B? Each form is tested against the objects alone
in the same format: a camera table, a compass path in text, and one narrative of the walk that merges the objects
with the path.

## Hypothesis and what would change our mind

- **Expected:** no camera form beats its objects-only control on direction and route by 5 points or more. The cameras
  push multiple-choice answers to B.
- **Would prove it wrong:** a camera form beats its control on direction and route by at least 5 points, with Holm
  p < 0.05, and the gain holds on the questions whose answer is not B.
- **Next:**
  - supported: drop the camera's path, and give direction relations instead ("standing at the sofa facing the table,
    the tv is front-left");
  - not supported: develop that form, with CoT or a larger model.

## Background

- **The camera pilot** (`experiments/2026-11-20-camera/README.md`, run 2). It used Qwen3.5-2B on explore200's 200
  questions, with sheets in metres by the priors scale and first-camera axes.
  - Against the objects-only controls, the cameras changed the overall by −1.6 to +2.8, and every interval included 0.
  - CoT lowered every camera form by 10.5–11.1 points.
  - The camera forms pushed answers to B.
- **Two pilot findings bias the metric tasks:**
  - the narrative answered 17 of 25 absolute-distance questions with a camera-to-object distance;
  - `obj_csv` answered 10 of 25 room-size questions with a number from the `room` rows.

## Design

### Conditions

| ID | Condition | Objects | `scene_sheet.cameras` | `scene_sheet.format` | Role | Run |
|---|---|---|---|---|---|---|
| A | Video only | none | none | none | baseline | R1 |
| B | `obj_csv` | csv | none | csv | control for C | R9 |
| C | `cam_csv` | csv | csv table | csv | treatment | R14 |
| D | `obj_text` | sentences | none | text | control for E and F | R10 |
| E | `cam_text` | sentences | compass path, a separate block | text | treatment | R15 |
| F | `narrative` | in the walk | in the walk | narrative | treatment | R16 |

### Comparisons

Each comparison is made on the overall and on the direction-and-route score (the mean of relative direction and route
planning), with Holm within each family and score.

- **Primary family:** C − B (camera table), E − D (camera text) and F − D (narrative).
- **Secondary family:** F − E, and every condition − A.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Table format | csv | the csv rule (`README.md` 2.6). The 2026-11-25 redesign had chosen the markdown table (pilot: 44.6 against 38.4 with cameras) |
| Camera format | the objects' format | the code's rule since `080acf4`; one control per format |
| Text options | the defaults: timestamps in seconds, bridged visibility, move-then-turn steps, object directions along the floor | the pilot's variants of them stayed within a few points |
| Relations | none | pipeline_v4's direction relation would state the answer the cameras could help with |
| Reasoning | direct only | the pilot's CoT lowered every camera form; CoT is experiments 5–7 |
| `room` rows, the narrative's camera distances | kept | part of what the pipeline writes; the metric tasks are read apart |

## Configuration

```yaml
# README.md section 2, except:
scene_sheet:
  source: pipeline             # pipeline_v4 (A: none)
  cameras: varies              # none (B, D) | in the objects' format (C, E) | in the walk (F); metres on the sheet's scale, upright axes
  format: varies               # csv (B, C) | text (D, E) | narrative (F)
```

## Metrics and analysis

As `README.md` 2.7. In addition:

- the direction-and-route score, and the metric tasks (absolute distance, object size, room size) as a group;
- on multiple choice, a table per condition: the share of B answers, accuracy where the answer is B and where it is
  not, and the score of always answering one letter;
- the median numeric answer per numeric task, against the true median;
- the numeric answers equal to a camera-to-object distance (F) or to a `room` row's number.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The shift to B passes for a gain | a gain on direction and route that vanishes where the answer is not B | read the letter table with every gain |
| The narrative's distances and the `room` rows are copied as answers | F or the object tables answering metric questions with those numbers | read the metric tasks apart |
| One scale error for both objects and cameras | errors shared by all sheet conditions | stated; the comparisons are within the same scale |

## Runs and cost

```bash
C=experiments/2026-11-18-scene-sheet-formats   # cluster_checks.py and the check ids
OUT=results/zeroshot/vsibench/definitive
S=results/_scratch/definitive_smoke
ROOT=tasks/openvocab_adapter/out_v4         # the pipeline_v4 build (README.md section 5, item 2)
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
PIPE="--sheet-source pipeline --sheets-root $ROOT --metric-scale <SHEET_SCALE>"
```

**Checks** (`README.md` 2.8):

1. **Inputs.** `test.jsonl`'s SHA-256 (`README.md` 2.3); 288 `data/vggt/*/*/frame_indices.npy`; `$ROOT/index.csv` with
   SHA-256 `<from the build>`.
2. **Tests and render check.** `<README.md section 5, items 3–4>`. They must show camera positions and move lengths in
   metres on the sheet's scale. All must pass, with none skipped.
3. **Prompt check** for C, E and F:

   ```bash
   P="--ids $C/prompt_check_ids.txt --print-prompt 4"
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv --cameras  $P > prompts_C.txt
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format text --cameras $P > prompts_E.txt
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format narrative      $P > prompts_F.txt
   python $C/cluster_checks.py prompts prompts_C.txt --sheets-root $ROOT --format csv
   python $C/cluster_checks.py prompts prompts_E.txt --sheets-root $ROOT --format text
   python $C/cluster_checks.py prompts prompts_F.txt --sheets-root $ROOT --format narrative
   ```

4. **Smoke test** of A and C, then the model revision:

   ```bash
   python -m mvp.eval.vsibench $COMMON --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R1.jsonl
   python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv --cameras --ids $C/smoke_ids.txt --out $S/qwen3_5_2B_R14.jsonl
   python $C/cluster_checks.py smoke $S/qwen3_5_2B_R1.jsonl $S/qwen3_5_2B_R14.jsonl --n 20
   python $C/cluster_checks.py revision Qwen/Qwen3.5-2B $OUT/model_revision_qwen3_5_2B.txt
   ```

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_R1.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv                --out $OUT/qwen3_5_2B_R9.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format csv --cameras      --out $OUT/qwen3_5_2B_R14.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format text               --out $OUT/qwen3_5_2B_R10.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format text --cameras     --out $OUT/qwen3_5_2B_R15.jsonl
python -m mvp.eval.vsibench $COMMON $PIPE --sheet-format narrative          --out $OUT/qwen3_5_2B_R16.jsonl
```

**Analysis:** `<make_table_4.py>` (`README.md` section 5, item 5) writes the Results table and the letter table.

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

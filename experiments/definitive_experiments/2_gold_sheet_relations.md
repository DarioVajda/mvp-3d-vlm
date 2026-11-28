---
id: definitive-2-gold-sheet-relations
title: Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench, with and without the video
owner: Tamara Grujičić
created: 2026-11-19
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, relations, definitive]
varies: [scene_sheet.source, visual.video]
depends_on: [gold sheets in tasks/openvocab_adapter/out_full (experiments/2026-11-19-gold-sheet-relations, Runs step 1)]
---

# Gold sheet and relations

**Outcome:** on all 5,130 questions, the relations raise the overall from 48.9 (gold sheet) to 61.3 (C − B +12.4,
Holm p 8e-157), all of it on absolute and relative distance (+55.9), where they state the answer; on the other types
with relations C − B is −4.2. The gold sheet adds +5.6 over video only (43.3), all of it from counting.

This file describes the experiment as it was registered in `experiments/2026-11-19-gold-sheet-relations/README.md`
(the design as it stood at `0f7c5f2`, the commit the runs were made at) and as it was run (results in `c4d74df`).
Where this folder's `README.md` section 2 differs (the model, the statistics, the outputs folder), this file holds:
these are the settings the runs used. Everything added after the runs is in "Post-hoc", below the Conclusion.

## Question

Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench? And how much of it does the sheet
carry without the video, at what cost in input tokens (frames only, sheet only, both)?

## Hypothesis and what would change our mind

As registered (the gold-sheet README, rewritten 2026-11-24 before any run):

- **Expected:**
  - The gold sheet helps over video only (B − A > 0), almost all of it on counting, as in the format experiment;
    without counting it adds little.
  - The relations raise the overall well beyond the gold sheet (C − B > 0), almost all of it on absolute and relative
    distance, where they state the answer and the model copies it. On relative direction, appearance order and route
    planning, which also get relations, C − B is near 0 or negative: the model uses distances it is given but does not
    compute anything else from them.
  - Without the video (D, E), the sheet keeps what it states: counting, and with relations absolute and relative
    distance, about as in B and C. Appearance order needs the video (the gold sheet has no times), so D and E fall
    there, and the overall falls below B and C. Sheet-only prompts are a small fraction of the video's input tokens.
- **Would prove it wrong:**
  - C − B ≤ 0, or within the smallest difference that matters: the pilot's gain from the relations was noise.
  - C − B positive only on absolute and relative distance: the model copies the answer lines (the relations state the
    answer to most distance questions) but gets nothing else from the relations.
  - B − A clearly positive without counting: the gold sheet helps on its own, unlike in the format experiment.
  - D ≥ B overall (within the smallest difference that matters): with a correct sheet the video adds nothing.
- **Next:**
  - the relations help beyond the distance questions: give the pipeline sheet the same relations (with the pipeline's
    content and a usable metric scale) and measure how much survives;
  - they help only where they state the answer: the gain is the relations doing the task, not the model using the
    sheet; precompute the quantity each question type needs, or move to a model that reasons over the sheet
    (2026-11-19-structured-cot);
  - they don't help: the pilot was noise; the sheet's value for this model is counting only.

The expected result and the second disproof describe the same pattern (C − B from the distance questions only). Both
were registered this way; the Conclusion reads the result against both.

## Background

As registered; the full text, with paths, is the gold-sheet README's Background.

- **Format experiment** (2,728 questions of the 75 exported scenes, direct answers): the gold sheet raised the overall
  from 43.2 to 48.4–49.4 across five formats, all of it from counting. Without counting it was below video only, and
  gold coordinates in metres did not help on absolute or relative distance or relative direction.
- **The 40-question pilot** (Qwen3.5-2B, the settings below, `explore_ids.txt`): video only 47.3, video + gold csv 54.7,
  video + gold sheet + precomputed relations (`gold_relations_v2`) 67.3. 40 questions is noisy: one question moves a
  task by 20 points and the overall by about 2.5.
- **The same three conditions on 200 questions** (structured-cot's direct A, C and E on `explore200_ids.txt`): video
  only 48.6, gold sheet 52.1, gold sheet + relations 62.7, mostly on absolute distance (39.6 → 94.8) and relative
  distance (76.0 → 96.0). Descriptive only, and these 200 questions are among the 5,130.
- **Closest points:** VSI-Bench's distance questions measure "from the closest point of each object"; on the pilot's
  5 absolute-distance questions, closest-point distances between gold boxes were within about 0.3 m of the answers and
  centre distances 0.5–1.6 m off.
- **The model copies given relations** (`experiments/2026-11-18-scene-sheet-formats/pipeline_tests.md`): with the
  pipeline's relations it repeated the given distance in 80 of 80 absolute-distance questions and the given closest
  object in 25 of 25 relative-distance ones, even when they were wrong.
- **Assumption:** that the gold sheet's boxes are close enough to the objects' shapes for closest-point distances; the
  pilot's 5 distance questions are the only check.

## Design

### Conditions

| ID | Condition | `scene_sheet.source` | `visual.video` | Role | Run |
|---|---|---|---|---|---|
| A | Video only | none | true | baseline | `qwen3_5_2B_A_video.jsonl` |
| B | Video + gold sheet (csv) | gold | true | treatment | `qwen3_5_2B_B_gold_csv.jsonl` |
| C | Video + gold sheet + relations (csv) | gold_relations | true | treatment, upper bound for "uses given relations" | `qwen3_5_2B_C_gold_relations_csv.jsonl` |
| D | Gold sheet only (csv), no video | gold | false | sheet only | `qwen3_5_2B_D_gold_csv_no_video.jsonl` |
| E | Gold sheet + relations only (csv), no video | gold_relations | false | sheet only, with relations | `qwen3_5_2B_E_gold_relations_csv_no_video.jsonl` |

All five in `results/zeroshot/vsibench/gold_sheet_relations_full/`, run new at one commit; no earlier run reused.

### Comparisons

As registered, before any run:

- **First family (Holm):** B − A (does a correct sheet help?, also read without counting) and C − B (do the relations
  help beyond the sheet?, the primary comparison). C − B is also read on the question types that get relations, with
  absolute and relative distance apart from the rest.
- **Second family (Holm), added 2026-11-24 before any run:** B − D and E − C (what the video adds to a correct sheet,
  without and with relations) and D − A (sheet only against video only).
- **Descriptive:** C − A, and B − A without counting; the input tokens of A (frames only), D and E (sheet only), and B
  and C (both).

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | All 5,130 of VSI-Bench (288 scenes); fallback, not used: the 2,728 of the 75 exported scenes | The whole benchmark; gold sheets need only the labels, so every scene has them |
| Model | Qwen/Qwen3.5-2B, revision `15852e8c16360a2fea060d615a32b45270f8a8fc` | The pilot's and structured-cot's model |
| Format | csv | Among the best and the shortest in the format experiment; the pilot's format |
| Relations | Closest-point distances (`--sheet-source gold_relations`, the same text as the pilot's `gold_relations_v2`) | VSI-Bench's distance questions measure between closest points |
| Which questions get relations | As `gold_relations_v2`, unchanged: the distance table between every two objects of different named classes for every question whose gold sheet has at least two of them (3,210 of 5,130); the closest-point answer line only for absolute- and relative-distance questions | The pilot's and structured-cot's condition; distance types and the other types with relations reported apart |
| Gold sheets | Built for all 288 scenes with `build_gold_sheets.py` into `tasks/openvocab_adapter/out_full` (index.csv `22e0ee1e…`, objects.csv `bc0214c9…`) | One build, from one commit, for every question; boxes fitted to their objects since `980cf00` |
| Prompts | `prompts_full.jsonl`, extracted once with spaCy 3.8 / `en_core_web_md` 3.8.0 | The gold sheets are then built with numpy only, the same on any machine |
| Decoding | 2,048 new tokens, final answer scored (`--final-answer`) | The format experiment's and the pilot's settings |
| Sheet only (D, E) | `--no-video`: the prompt of B or C without the video and its sentence ("These are frames of a video."); same model, decoding and answer rule | The proposal asks for frames only, sheet only and both, with their token cost. A–C's prompts unchanged (checked for all 5,130 questions against `f3fdf2f`) |
| Runs | All five at one commit, none reused | The renderer had changed since the format experiment |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full                  # 5,130 questions
  subset: none
  scenes: all 288 (gold sheets: tasks/openvocab_adapter/out_full, from build_gold_sheets.py)
  question_types: all          # all 10
  n_questions: 5130

visual:
  video: varies                # true (A, B, C) | false (D, E: --no-video, text only)
  model: Qwen/Qwen3.5-2B       # revision 15852e8c16360a2fea060d615a32b45270f8a8fc (model_revision.txt)
  protocol: video              # one Qwen video, frames merged in pairs
  nframes: 16
  sampling: uniform128         # 16 uniform frames among the 128 VGGT frames
  layout: sit                  # video, then question (lmms-eval)
  resolution: default

scene_sheet:
  source: varies               # none (A) | gold (B, D) | gold_relations (C, E)
  geometry: ground truth       # VSI-Bench's labelled boxes
  frames: n/a
  sam3_prompts: ground truth   # the question's prompts matched to the dataset's classes
  sam3: n/a
  tracking: n/a
  content: [position, size]    # C and E add closest-point distances between the named objects
  units: metres
  format: csv
  renderer: src/mvp/benchmarks/scene_sheet.py:render_sheet (relations: scene_sheet.relations)
  placement: between video and question (D, E: before the question, no video)

prompt:
  system: You are a helpful assistant.
  template: lmms-eval          # build_prompt in src/mvp/benchmarks/vsibench.py
  reasoning: direct
  thinking: off                # enable_thinking=False
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final  # --final-answer: the last "Answer: X", else the last lone letter / the last number
  batches: eval defaults       # --batch-size 8 --token-budget 120000

training: none
```

## Metrics and analysis

As registered:

- **Primary metric:** VSI-Bench overall (8 tasks) on the 5,130 questions, for C − B.
- **Secondary metrics:** overall without counting; score per task and question type; parse rate
  (`compare_runs.parse_rate`, the final-answer rule); input tokens, mean and median per condition, from each question's
  own `n_prompt_tokens`; the share of questions that get relations.
- **Breakdowns:** question type; questions with relations vs without; absolute and relative distance vs the other
  types with relations; dataset (ScanNet, ScanNet++, ARKitScenes).
- **Statistics:** every condition on the same questions, compared question by question: how many get better and worse,
  with an exact two-sided sign test (`mvp.reporting.compare_runs.sign_test`), Holm-corrected within each family above.
  No bootstrap intervals were registered.
- **Smallest difference that matters:** 2 points overall, 5 on a task.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet contains the counting answer | Counting near 98% in B and C | Read B − A and C − A without counting too |
| The relations state the answer to most distance questions, and the model copies them | C near 100 on absolute and relative distance | C − B there measures copying a given number; read the other types with relations apart |
| Closest-point distances treat each box as axis-aligned, so they can be off for objects at an angle | C's absolute-distance errors on large or diagonal objects | Stated; the pilot's 5 checks are the only evidence |
| The relations lengthen the prompt (up to 48k characters; 45 questions over 10k, all relative distance in ScanNet++ rooms with many chairs and ceiling lights) | Input tokens in C | Report tokens |
| The pilot's 40 and structured-cot's 200 questions are among these | — | They motivate the plan; they are not baselines for these runs |
| D and E differ from B and C in more than the video: the video's sentence is left out | — | Stated (Decisions) |
| Without the video, appearance order and route planning have little to go on (no times in the gold sheet) | D and E near chance there | Expected; read B − D per task |
| The model's revision differs from the pilot's | The smoke test's revision | Stated in the Log |

## Runs and cost

As run, on the cluster, from the repository root at `0f7c5f2`, with `PYTHONPATH=src` where shown:

```bash
G=experiments/2026-11-19-gold-sheet-relations
OUT=results/zeroshot/vsibench/gold_sheet_relations_full
SHEETS=tasks/openvocab_adapter/out_full
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
```

**Checks:**

1. **Inputs and gold sheets.** `data/raw/VSI-Bench/test.jsonl` SHA-256 `aa3ff8cc…4db9db`; 288
   `data/vggt/*/*/frame_indices.npy`. `build_gold_sheets.py --prompts $G/prompts_full.jsonl --out $SHEETS` ends with
   `sheets failed: 0`, index.csv `22e0ee1e…`, objects.csv `bc0214c9…`; `check_sheets.py` ends with `OK`.
2. **Prompt check** for C on `prompt_check_ids.txt` (`--print-prompt 4`, then `cluster_checks.check_prompts`).
3. **Smoke test** of A, C and E on `smoke_ids.txt` (20 questions), `cluster_checks.py smoke`, the
   `test_scene_sheet.NoVideo` tests, and the model revision into `$OUT/model_revision.txt`.

**Runs**, each reporting 5,130 questions:

```bash
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_A_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_B_gold_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_C_gold_relations_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --no-video --out $OUT/qwen3_5_2B_D_gold_csv_no_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --no-video --out $OUT/qwen3_5_2B_E_gold_relations_csv_no_video.jsonl
```

| Run | Conditions | Output | Commit |
|---|---|---|---|
| A | A | `$OUT/qwen3_5_2B_A_video.jsonl` | `0f7c5f2` |
| B | B | `$OUT/qwen3_5_2B_B_gold_csv.jsonl` | `0f7c5f2` |
| C | C | `$OUT/qwen3_5_2B_C_gold_relations_csv.jsonl` | `0f7c5f2` |
| D | D | `$OUT/qwen3_5_2B_D_gold_csv_no_video.jsonl` | `0f7c5f2` |
| E | E | `$OUT/qwen3_5_2B_E_gold_relations_csv_no_video.jsonl` | `0f7c5f2` |

**Analysis:** `$G/make_table.py --results $OUT --sheets-root $SHEETS`, with `--ids` all 5,130 question ids, wrote the
Results below (and `$G/results.csv`).

**Cost:** estimated at 30–40 GPU minutes on one B200; the measured time is not recorded in the results.

## Results

<!-- make_table:start -->
Written by `make_table.py` (5,130 questions per condition; scores ×100; overall: the mean over the 8 tasks present).

| ID | Condition | Overall | Without counting | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Parsed | Input tokens (mean / median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | video only | 43.3 | 43.7 | 40.1 | 34.6 | 59.6 | 40.7 | 46.3 | 46.5 | 33.5 | 45.0 | 100.0% | 2562 / 2546 |
| B | + gold sheet | 48.9 | 41.8 | 98.7 | 31.4 | 55.4 | 45.1 | 45.2 | 47.7 | 33.5 | 34.1 | 100.0% | 2800 / 2738 |
| C | + gold sheet + relations | 61.3 | 55.9 | 98.7 | 91.2 | 55.0 | 44.6 | 96.3 | 46.1 | 33.0 | 25.2 | 100.0% | 3048 / 2762 |
| D | gold sheet only | 39.9 | 31.5 | 98.5 | 22.4 | 78.0 | 0.6 | 37.3 | 37.0 | 28.4 | 16.8 | 100.0% | 318 / 257 |
| E | gold sheet + relations only | 55.0 | 48.8 | 98.5 | 91.0 | 77.6 | 0.6 | 93.4 | 37.1 | 26.8 | 15.4 | 100.0% | 566 / 279 |

Comparisons, question by question (better / worse: questions where the second condition scores higher / lower; exact two-sided sign test; Holm over B − A and C − B, and with D and E over B − D, E − C and D − A):

| Comparison | Overall | Without counting | Better / worse / same | p | Holm p |
|---|---:|---:|---:|---:|---:|
| B − A | +5.6 | -2.0 | 1464 / 1211 / 2455 | 1.08e-06 | 1.08e-06 |
| C − B | +12.4 | +14.2 | 1265 / 263 / 3602 | 4.05e-157 | 8.1e-157 |
| C − A | +18.0 | +12.2 | 2253 / 928 / 1949 | 1.4e-125 | – |
| B − A, counting left out | -2.0 | -2.0 | 1064 / 1207 / 2294 | 0.00288 | – |
| B − D | +9.0 | +10.3 | 1286 / 919 / 2925 | 5.63e-15 | 1.69e-14 |
| E − C | -6.2 | -7.1 | 602 / 728 / 3800 | 0.000604 | 0.00121 |
| D − A | -3.4 | -12.2 | 1591 / 1548 / 1991 | 0.453 | 0.453 |

C − B by group (3210 of 5130 questions get relations; scores: the mean over the group's questions, not over tasks):

| Questions that | n | A | B | C | C − B | Better / worse / same | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| get relations | 3210 | 42.4 | 39.7 | 64.4 | +24.7 | 1232 / 224 / 1754 | 1.06e-168 |
| get no relations | 1920 | 49.6 | 65.4 | 65.1 | -0.3 | 33 / 39 / 1848 | 0.556 |
| get relations: abs. and rel. distance | 1540 | 40.0 | 37.8 | 93.7 | +55.9 | 1119 / 41 / 380 | 8.51e-274 |
| get relations: the other types | 1670 | 44.6 | 41.6 | 37.4 | -4.2 | 113 / 183 / 1374 | 5.63e-05 |

Per question type:

| Question type | n | With relations | A | B | C | C − B |
|---|---:|---:|---:|---:|---:|---:|
| obj_appearance_order | 618 | 618 | 45.0 | 34.1 | 25.2 | -8.9 |
| object_abs_distance | 834 | 830 | 34.6 | 31.4 | 91.2 | +59.8 |
| object_counting | 565 | 0 | 40.1 | 98.7 | 98.7 | +0.0 |
| object_rel_direction_easy | 217 | 217 | 53.0 | 51.2 | 48.8 | -2.3 |
| object_rel_direction_hard | 373 | 373 | 39.9 | 42.1 | 43.2 | +1.1 |
| object_rel_direction_medium | 378 | 378 | 46.6 | 49.7 | 46.3 | -3.4 |
| object_rel_distance | 710 | 710 | 46.3 | 45.2 | 96.3 | +51.1 |
| object_size_estimation | 953 | 0 | 59.6 | 55.4 | 55.0 | -0.4 |
| room_size_estimation | 288 | 0 | 40.7 | 45.1 | 44.6 | -0.5 |
| route_planning | 194 | 84 | 33.5 | 33.5 | 33.0 | -0.5 |

Per dataset (mean over questions):

| Dataset | n | A | B | C |
|---|---:|---:|---:|---:|
| arkitscenes | 1601 | 49.1 | 52.6 | 68.1 |
| scannet | 2071 | 44.3 | 47.7 | 61.9 |
| scannetpp | 1458 | 41.7 | 48.0 | 64.8 |
<!-- make_table:end -->

All five runs answered the same 5,130 questions, every reply parsed, none reached the token limit, and every D and E
prediction has `"n_frames": 0`.

### Examples

- **Absolute distance** (question 680, the sofa and the stove, truth 2.9 m): A answers 2.4, B 3.376, C and E 3.154,
  the relation's closest-point distance.
- **Appearance order** (question 2725: chair, toilet, blanket, basket; truth B): A and B answer B, C answers A and E
  answers D.
- **Room size without the video:** D and E answer 120 to all 288 room-size questions; the gold sheet has no room
  dimensions.

### What the results show

On the registered comparisons:

- **B − A, +5.6 (Holm p 1.1e-6), from counting only:** counting 40.1 → 98.7; without counting B − A is −2.0 (1,064
  better, 1,207 worse, p 0.003, descriptive), with the largest loss on appearance order (45.0 → 34.1).
- **C − B, +12.4 (Holm p 8e-157), the primary comparison:** +55.9 on the absolute- and relative-distance questions
  that get relations (1,119 better, 41 worse), −4.2 on the other types with relations (113 better, 183 worse,
  p 5.6e-5), −0.3 on the questions without relations. Per type: absolute distance +59.8, relative distance +51.1,
  appearance order −8.9, relative direction −3.4 to +1.1, route planning −0.5.
- **Without the video:** D 39.9, E 55.0. Counting stays at 98.5, and E keeps absolute and relative distance (91.0,
  93.4); appearance order falls to 16.8 and 15.4, room size to 0.6. B − D +9.0 (Holm p 1.7e-14) and E − C −6.2 (Holm
  p 0.0012): the video adds to a correct sheet. D − A −3.4 (Holm p 0.45).
- **Token cost** (mean / median): frames only 2,562 / 2,546; both 2,800 / 2,738 (B) and 3,048 / 2,762 (C); sheet only
  318 / 257 (D) and 566 / 279 (E), 12% and 22% of video only's mean.
- **Datasets:** C − A is +19.0 (ARKitScenes), +17.6 (ScanNet) and +23.1 (ScanNet++).

## Conclusion

- **Answer to the question:** a correct scene sheet helps Qwen3.5-2B only on counting, where it holds the answer.
  Precomputed relations raise the overall by 12.4 points, all of it on absolute and relative distance, where they
  state the answer; on the other types with relations the model does slightly worse. Without the video the sheet keeps
  these gains and loses appearance order and room size, at about a tenth of the input tokens.
- **Hypothesis:** against the registered criteria:
  - the expected result holds on every point (B − A from counting; C − B from the distance questions, near 0 or
    negative elsewhere; D and E keep counting and distances and fall on appearance order; sheet-only prompts short);
  - first disproof (C − B ≤ 0): not met;
  - second disproof (C − B positive only on absolute and relative distance): met. By its registered reading, the model
    copies the answer lines and gets nothing else from the relations;
  - third (B − A clearly positive without counting): not met (−2.0);
  - fourth (D ≥ B): not met (B − D +9.0).

  So the relations help where they state the answer and nowhere else: the registered "Next" for that case applies.
- **Caveats:** one model (the 2B), one run with greedy decoding, direct answers. Gold sheets and relations are an upper
  bound: the model copies given distances, so with a pipeline sheet the distance gain is only as good as the pipeline's
  distances. B − A without counting and C − A are descriptive. The runs were made at `0f7c5f2`, before `b0ed3c1`,
  which changes only a frame check that gold sheets skip; the prompts of A–E for all 5,130 questions are the same at
  both.
- **Next** (the registered branch for "help only where they state the answer"): precompute the quantity each question
  type needs, or move to a model that reasons over the sheet (2026-11-19-structured-cot).

## Post-hoc (added after the runs)

Nothing in this section was registered before the runs; it is exploratory.

- **Analyses computed after the results** (2026-11-27, from the same predictions and `scene_sheet.relations` on the
  same sheets):
  - copying: C's answer equals the stated distance in 830 of 830 absolute-distance questions with an answer line, and
    the stated closest object in 694 of 710 relative-distance ones (E: 830 and 676);
  - the stated relations' accuracy: 798 of 830 distances within 25% of the truth (median error 0.07 m); 684 of 710
    closest objects right;
  - appearance order: C answers B in 53% of the 618 questions, whose truths are spread evenly; A right and C wrong on
    157, the reverse on 35;
  - object size: D 78.0 against B 55.4; where B and D answer differently (74% of size questions), D scores 74.4 and B
    43.7.
- **A replacement criterion written after the results** (this file's earlier draft, 2026-11-27, two days after the
  results were pushed in `c4d74df`): it dropped the second registered disproof and put in its place "C − B helps (at
  least 5 points, Holm p < 0.05) on a type with relations other than absolute and relative distance". It is not a
  registered criterion of this experiment. Read post hoc on these results, no such type improves by 5 points (the
  largest is relative direction hard, +1.1).
- **A rerun with Qwen3.5-4B** (this folder's `README.md`, sections 2 and 3.2) would be a separate experiment on the
  same design, with criteria to be registered before its runs.

## Log

- 2026-11-19 to 2026-11-25, before any run: designed and registered in
  `experiments/2026-11-19-gold-sheet-relations/README.md` (its Log has each change and its reason). The design as run
  is the README at `0f7c5f2`; `d508cf7` later changed only its run requirements (`b0ed3c1` or later for B and C).
- 2026-11-25: the five full-benchmark runs (A–E, 5,130 questions each) were made at `0f7c5f2`, with Qwen/Qwen3.5-2B at
  revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; predictions and summaries in
  `results/zeroshot/vsibench/gold_sheet_relations_full/` (`c4d74df`).
- 2026-11-27: Results written with `make_table.py` into the gold-sheet README (committed 2026-11-28, `a00f631`).
- 2026-11-28: this file rewritten to describe the experiment as registered and run, with the 2B; the criterion and
  model of the 2026-11-27 draft moved to Post-hoc.

---
id: 2026-11-19-gold-sheet-relations    # also the folder name: experiments/<id>/
title: Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench
owner: Tamara Grujičić
created: 2026-11-19
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, relations]
varies: [scene_sheet.source, visual.video]  # keys that differ between conditions
depends_on: [2026-11-18-scene-sheet-formats (gold sheets, the 40-question pilot)]
---

# Gold sheets and precomputed relations

**Outcome:** <!-- one line once the experiment is done: the answer and its headline number -->

## Question

Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench? And how much of it does the sheet
carry without the video, at what cost in input tokens (frames only, sheet only, both)?

## Hypothesis and what would change our mind

- **Expected result:**
  - The gold sheet helps over video only (B − A > 0), almost all of it on counting, as in the format experiment; without
    counting it adds little.
  - The relations raise the overall well beyond the gold sheet (C − B > 0), almost all of it on absolute and relative
    distance, where they state the answer and the model copies it (pilot, structured-cot's 200 questions, the
    pipeline tests). On relative direction, appearance order and route planning, which also get relations, C − B is
    near 0 or negative: the model uses distances it is given but does not compute anything else from them.
  - Without the video (D, E), the sheet keeps what it states: counting, and with relations absolute and relative
    distance, about as in B and C. Appearance order needs the video (the gold sheet has no times), so D and E fall
    there, and the overall falls below B and C. Sheet-only prompts are a small fraction of the video's input tokens.
- **Results that would prove it wrong:**
  - C − B ≤ 0, or within the smallest difference that matters: the pilot's gain from the relations was noise.
  - C − B positive only on absolute and relative distance: the model copies the answer lines (the relations state the
    answer to most distance questions) but gets nothing else from the relations.
  - B − A clearly positive without counting: the gold sheet helps on its own, unlike in the format experiment.
  - D ≥ B overall (within the smallest difference that matters): with a correct sheet the video adds nothing.
- **What we do next in each case:**
  - The relations help beyond the distance questions: give the pipeline sheet the same relations (with the pipeline's
    content and a usable metric scale) and measure how much survives; that is the step towards a usable system.
  - They help only where they state the answer: the gain is the relations doing the task, not the model using the
    sheet; precompute the quantity each question type needs, or move to a model that reasons over the sheet
    (2026-11-19-structured-cot).
  - They don't help: the pilot was noise; the sheet's value for this model is counting only.

## Background

<!-- What is already known: the results this builds on (numbers and paths, e.g. the README's tables), design notes in
docs/, earlier experiments, and papers in papers/ (file, page and verbatim quote). Mark anything that no result or
paper backs as an assumption. -->

- **Format experiment** (`experiments/2026-11-18-scene-sheet-formats/README.md`, Results): on the 2,728 questions of
  the 75 exported scenes, with direct answers, every reply was a bare letter or number. The gold sheet raised the
  overall from 43.2 to 48.4–49.4 across five formats, all of it from counting (39.6 → about 98%, because the gold sheet
  lists every labelled object of the class). Without counting the gold overall was below video only, and gold
  coordinates in metres did not help on absolute or relative distance or relative direction.
- **The 40-question pilot** (`experiments/2026-11-18-scene-sheet-formats/explore.ipynb`, run on Google Colab):
  Qwen3.5-2B, thinking off, 16 frames (`uniform128`), up to 2,048 new tokens, the final answer scored, on the 40
  questions of `explore_ids.txt` (5 per task, from 5 scenes). The conditions compared here were:
  - **video only** (`A_video`): 47.3;
  - **video + gold sheet, csv** (`gold_csv`): 54.7;
  - **video + gold sheet + precomputed relations** (`gold_relations_v2`): the gold sheet, then the distances in metres
    between the closest points of the boxes of every two objects of different named classes, the two objects' distance
    for absolute-distance questions, and each candidate's distance to the target for relative-distance questions: 67.3.

  The model uses relations it is given but does not compute them from the coordinates. **40 questions is noisy**: one
  question moves a task's score by 20 points and the overall by about 2.5. The pilot's predictions are on Google Drive
  (`MyDrive/mvp/ssf_explore_results/`), not in the repository, and it ran the code at `b381205` with the relations
  added by `explore_sheets.py`.
- **Closest points:** all 428 absolute-distance and 379 relative-distance questions of the 2,728 say "Measuring from
  the closest point of each object". On the pilot's 5 absolute-distance questions, closest-point distances between gold
  boxes were within about 0.3 m of the answers and centre distances 0.5–1.6 m off. That is too few questions to rely
  on; it is why the relations use closest points.
- **The source:** `--sheet-source gold_relations` (`src/mvp/benchmarks/scene_sheet.py`, `relations`) writes the
  pilot's `gold_relations_v2` text; the two give identical relations for all 2,728 questions (checked 2026-11-19).
  1,992 of the 2,728 questions get relations: every appearance-order, relative-direction and relative-distance
  question, 424 of the 428 absolute-distance questions and 28 of the 58 route-planning ones. The others name fewer
  than two classes that are in the gold sheet: counting, object size and room size, and the rest of absolute distance
  and route planning.
- **The same three conditions on 200 questions** (`2026-11-19-structured-cot`, its direct conditions A, C and E, on
  `explore200_ids.txt`; Colab L4, code at `7d39234`; `results/zeroshot/vsibench/structured_cot/explore200/`): video
  only 48.6, gold sheet 52.1, gold sheet + relations 62.7. Most of the gain is on the types where the relations state
  the answer:

  | Task | Video only | Gold csv | Gold + relations |
  |---|---:|---:|---:|
  | Counting | 49.6 | 100.0 | 100.0 |
  | Absolute distance | 39.6 | 26.4 | 94.8 |
  | Relative distance | 76.0 | 64.0 | 96.0 |
  | Relative direction | 47.2 | 56.0 | 47.7 |
  | Appearance order | 44.0 | 36.0 | 32.0 |
  | Route planning | 24.0 | 32.0 | 28.0 |

  With 25 questions per task this is descriptive only, and these 200 questions are among the 2,728.
- **The model copies given relations** (`experiments/2026-11-18-scene-sheet-formats/pipeline_tests.md`): with the
  pipeline's relations it repeated the given distance in 80 of 80 absolute-distance questions and the given closest
  object in 25 of 25 relative-distance ones, even when they were wrong. So C on the distance questions measures how
  good the given numbers are, not whether the model reasons with them. The other types with relations (direction,
  appearance order, route planning) are where C − B says whether relations help beyond stating the answer.
- **Assumption:** that the gold sheet's axis-aligned boxes are close enough to the objects' shapes for closest-point
  distances; the pilot's 5 distance questions are the only check.

## Design

### Conditions

<!-- One row per condition, and one column per key in `varies`. Every other setting is in Configuration. -->

| ID | Condition | `scene_sheet.source` | `visual.video` | Role | Existing run |
|---|---|---|---|---|---|
| A | Video only | none | true | baseline | |
| B | Video + gold sheet (csv) | gold | true | treatment | |
| C | Video + gold sheet + relations (csv) | gold_relations | true | treatment, upper bound for "uses given relations" | |
| D | Gold sheet only (csv), no video | gold | false | sheet only | |
| E | Gold sheet + relations only (csv), no video | gold_relations | false | sheet only, with relations | |

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run. -->

- **B − A: does a correct scene sheet help?** Also without counting, since the gold sheet contains the counting answer.
- **C − B: do precomputed relations help beyond the sheet?** The primary comparison. Also on the question types that
  get relations, and with absolute and relative distance (where the relations state the answer) apart from the rest.
- **C − A:** both together, descriptive.
- **Without the video (added before any run; a second family, so the A–C comparisons above are unchanged):**
  - **B − D:** what the video adds to a correct sheet; **E − C** the same with relations.
  - **D − A:** sheet only against video only.
  - **Token cost:** mean and median input tokens per condition: frames only (A), sheet only (D, E), both (B, C).

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source. -->

| Decision | Choice | Why |
|---|---|---|
| Questions | All 5,130 of VSI-Bench (288 scenes); fallback: the 2,728 of the 75 exported scenes (`question_ids.txt`) | The whole benchmark, for the report; gold sheets need only the labels, so every scene has them. The pilot's 40 and structured-cot's 200 are among them (under 5%) |
| Format | csv | Among the best and the shortest in the format experiment; the pilot's format |
| Relations | Closest-point distances (`--sheet-source gold_relations`, the same text as the pilot's `gold_relations_v2`) | VSI-Bench's distance questions measure between closest points (Background) |
| Which questions get relations | As `gold_relations_v2`, unchanged: the table of distances between every two objects of different named classes for every question whose gold sheet has at least two of them (3,210 of 5,130; 1,992 of the 2,728); the closest-point answer line only for absolute- and relative-distance questions | The pilot's and structured-cot's condition; distance types and the other types with relations are reported apart (Metrics) |
| Gold sheets | Built again for all 288 scenes with the current code (`build_gold_sheets.py`, into `tasks/openvocab_adapter/out_full`), the 75 scenes included; not the 75-scene sheets of `run_sheets.sh` | One build, from one commit, for every question keeps the full run and the fallback the same, whichever copy of the 75-scene sheets a machine has. Since `980cf00` a gold box is turned to fit its object; `scene_sheets_75.zip` (Drive, 18 Nov, and its local copies) was built before it, with larger axis-aligned extents (2,643 of its 2,728 `objects.csv` differ in the sizes, none in the positions). Which sheets structured-cot read is not recorded; its answers fit the rebuilt sheets and not that zip (Log, the explore200 preview) |
| Prompts | `prompts_full.jsonl`: each question's SAM 3 prompts, extracted once with spaCy 3.8 / `en_core_web_md` 3.8.0 (`extract_prompts.py`, on Colab, `prompts_full.ipynb`) | The gold sheets are then built with numpy only, the same on any machine; the prompts of the 2,728 questions are identical to the 75-scene sheets' |
| Decoding | 2,048 new tokens, final answer scored (`--final-answer`) | The format experiment's and the pilot's settings: replies were bare answers, and the same rule scores every condition |
| Sheet only (D, E) | `--no-video`: the same prompt as B and C without the video and without its sentence ("These are frames of a video."), so it starts with the sheet; same model, decoding and answer rule | The proposal's success criteria ask for frames only, sheet only and both, with their token cost. Added before any run; A–C's prompts are unchanged (checked for all 5,130 questions against `f3fdf2f`) |
| Runs | All five at one commit, none reused | The renderer has changed since the format experiment (`980cf00`, `080acf4`, `6642c48`) |

## Configuration

<!-- The settings every condition shares. A key in `varies` has the value `varies` here, and its values in the
Conditions table. -->

```yaml
data:
  benchmark: vsibench          # vsibench | mmsibench
  split: full                  # full (5,130 questions) | debiased (2,362) | pruned (2,768)
  subset: none                 # all 5,130 questions (fallback: question_ids.txt, the 2,728 of the 75 exported scenes)
  scenes: all 288 (gold sheets: tasks/openvocab_adapter/out_full, from build_gold_sheets.py)
  question_types: all          # all 10
  n_questions: 5130            # fallback: 2728

visual:                        # what the VLM sees of the video
  video: varies                # true (A, B, C) | false (D, E: --no-video, text only)
  model: Qwen/Qwen3.5-2B       # revision: the cluster's cached snapshot, recorded by the smoke test (model_revision.txt)
  protocol: video              # one Qwen video, frames merged in pairs
  nframes: 16
  sampling: uniform128         # 16 uniform frames among the 128 VGGT frames, as the sheets
  layout: sit                  # video, then question (lmms-eval)
  resolution: default

scene_sheet:
  source: varies               # none (A) | gold (B, D) | gold_relations (C, E)
  geometry: ground truth (source)  # gold sheets: VSI-Bench's labelled boxes
  frames: n/a                  # gold sheets are not built from frames
  sam3_prompts: ground truth   # the question's prompts matched to the dataset's classes
  sam3: n/a
  tracking: n/a
  content: [position, size]    # C adds closest-point distances between the named objects (relations)
  units: metres                # gold
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

training: none
```

## Metrics and analysis

- **Primary metric:** VSI-Bench overall (8 tasks) on the 5,130 questions (fallback: the 2,728), for C − B.
- **Secondary metrics:** overall without counting; score per task and question type; parse rate
  (`compare_runs.parse_rate`, the final-answer rule); input tokens, mean and median per condition (the token cost of
  frames only, sheet only and both), from each question's own `n_prompt_tokens` (`n_input_tokens` is the batch's
  padded length); the share of questions that get relations.
- **Pipeline metrics:** n/a (gold sheets only).
- **Breakdowns:** question type, always; questions with relations vs without; absolute and relative distance (the
  relations state the answer) vs the other types with relations; dataset (ScanNet, ScanNet++, ARKitScenes).
- **Statistics:** every condition on the same questions, compared question by question: how many get better and
  worse, with an exact two-sided sign test (`mvp.reporting.compare_runs.sign_test`), Holm-corrected over B − A and
  C − B; B − D, E − C and D − A are a second family, Holm-corrected among themselves.
- **Smallest difference that matters:** 2 points overall, 5 on a task, as in the format experiment.

## Threats to validity and checks

<!-- Keep the rows that apply and add this experiment's own. -->

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet contains the counting answer | Counting near 98% in B and C | Read B − A and C − A without counting too |
| The relations state the answer to most distance questions | C near 100 on absolute and relative distance | C − B there measures copying a given number; read the other types with relations apart |
| Closest-point distances treat each box as axis-aligned (its fitted sides matched to the world axes), so they can be off for objects at an angle | C's absolute-distance errors concentrated on large or diagonal objects | Stated; the pilot's 5 checks are the only evidence |
| The relations lengthen the prompt (one line per pair of objects: up to 48k characters, 45 questions over 10k, all relative distance in ScanNet++ rooms with many chairs and ceiling lights) | Input tokens in C, and C − B lower where the relations are longest | Report tokens; break down by relation lines if needed |
| The gold sheets differ from the ones the pilot used | — | A fresh run on the rebuilt sheets gives structured-cot's answers on its 200 questions (Log), though which sheets structured-cot read is not recorded; the 40-question pilot's sheets were not checked. Their numbers (67.3, 62.7) motivate the plan; they are not baselines for these runs |
| The pilot's conditions are among these questions | — | 40 of 5,130 (of 2,728 in the fallback); the pilot motivates the plan, it is not a result here |
| structured-cot ran the same conditions on 200 of these questions | — | 200 of 5,130 (4%; 7% of the fallback's 2,728); reported in Background, not reused, and every run here is new |
| The model copies relations, so C on distance questions is set by the relations' accuracy | C near 100 on absolute and relative distance | Read C − B on the other types with relations apart; that is the comparison about using relations |
| D and E differ from B and C in more than the video: the video's sentence is left out | — | Stated (Decisions); it is the only other difference, and its words describe the frames that are not there |
| Without the video, appearance order and route planning have little to go on (the gold sheet has no times) | D and E near chance there | Expected; read B − D per task |
| The model's revision or the sheets differ from the pilot's | The smoke test's revision; the prompt check | Stated in the Log |

## Runs and cost

<!-- One row per run. Output: the folder with the predictions (.jsonl), the summaries and the run's freeze.txt or
environment.txt. -->

All commands run from the repository root on the cluster, on `scene_sheet_formats` at `b0ed3c1` or later (D and E need
`1abeca4`'s `--no-video` fix, and B and C `b0ed3c1`'s frame check; the commits in between stop B and C), in the project's environment, with `PYTHONPATH=src` where shown. Nothing here needs spaCy or a
GPU except the eval.

**1. Inputs (CPU, about a minute).** VSI-Bench's questions must be the version the prompts were taken from, and every
scene needs VGGT's frame indices (`--sampling uniform128`):

```bash
shasum -a 256 data/raw/VSI-Bench/test.jsonl   # aa3ff8cc2010e555f4f865ff96483812ec0b275005b68562f53bb114ad4db9db
ls data/vggt/*/*/frame_indices.npy | wc -l     # 288
G=experiments/2026-11-19-gold-sheet-relations
PYTHONPATH=src python $G/build_gold_sheets.py --questions data/raw/VSI-Bench/test.jsonl \
    --prompts $G/prompts_full.jsonl --out tasks/openvocab_adapter/out_full
PYTHONPATH=src python $G/check_sheets.py --sheets-root tasks/openvocab_adapter/out_full --questions data/raw/VSI-Bench/test.jsonl
```

`build_gold_sheets.py` must end with `sheets failed: 0` and these two SHA-256, which a local build gave (twice, the same):

```
index.csv sha256 22e0ee1ed3400a5ca59bc49c3791f433265115b4301c183b9772e4e014f5573c
objects.csv sha256 bc0214c926be913606031474df123b2d6ee749fba39f799d56f4e8b1b21f41a5
```

`check_sheets.py` renders B and C for all 5,130 questions as the eval does and must end with `OK` (3,210 with
relations). Locally: 288 scenes, 5,130 sheets, none failed; counting rows equal the true count in 553 of 565
counting questions (the others are where VSI-Bench's labels and its answer disagree).

**2. Checks before the runs.**

- **The prompts.** The full prompts of `prompt_check_ids.txt` for C, without loading the weights, checked against
  `build_sheet_prompt` with the format experiment's `cluster_checks.check_prompts`. It reads the video's timestamps
  from the question's pipeline sheet folder, which only the 75-scene sheets have, so it runs on those: it checks how
  the eval assembles the prompt, which is the same for both roots.

  ```bash
  python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --sheet-source gold_relations \
      --sheet-format csv --sheets-root tasks/openvocab_adapter/out \
      --ids experiments/2026-11-19-gold-sheet-relations/prompt_check_ids.txt --print-prompt 4 > prompts_C.txt
  PYTHONPATH=src python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/2026-11-18-scene-sheet-formats'); import cluster_checks as c; p = c.check_prompts(Path('prompts_C.txt').read_text(), Path('tasks/openvocab_adapter/out'), Path('data/vggt'), source='gold_relations'); print(*p, sep='\n') if p else print('OK'); sys.exit(bool(p))"
  ```

- **20-question smoke test** of A, C and E, then the model revision:

  ```bash
  SMOKE="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer --ids experiments/2026-11-19-gold-sheet-relations/smoke_ids.txt"
  python -m mvp.eval.vsibench $SMOKE --out results/_scratch/gsr_smoke/A.jsonl
  python -m mvp.eval.vsibench $SMOKE --sheet-source gold_relations --sheet-format csv \
      --sheets-root tasks/openvocab_adapter/out_full --out results/_scratch/gsr_smoke/C.jsonl
  python experiments/2026-11-18-scene-sheet-formats/cluster_checks.py smoke \
      results/_scratch/gsr_smoke/A.jsonl results/_scratch/gsr_smoke/C.jsonl --n 20
  python -m mvp.eval.vsibench $SMOKE --sheet-source gold_relations --sheet-format csv --no-video \
      --sheets-root tasks/openvocab_adapter/out_full --out results/_scratch/gsr_smoke/E.jsonl
  PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet.NoVideo   # 6 tests, none skipped here
  python experiments/2026-11-18-scene-sheet-formats/cluster_checks.py revision Qwen/Qwen3.5-2B \
      results/zeroshot/vsibench/gold_sheet_relations_full/model_revision.txt
  ```

`E.jsonl` must have 20 lines, each with `"n_frames": 0`, an `n_prompt_tokens` and an input far shorter than C's for
the same question.

**3. The 5 runs on the full benchmark**, one command per condition, no `--ids`. They are independent and each resumes
from its `--out` file if stopped:

```bash
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer"
OUT=results/zeroshot/vsibench/gold_sheet_relations_full
SHEETS=tasks/openvocab_adapter/out_full
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_A_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_B_gold_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_C_gold_relations_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --no-video --out $OUT/qwen3_5_2B_D_gold_csv_no_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --no-video --out $OUT/qwen3_5_2B_E_gold_relations_csv_no_video.jsonl
```

Each must report 5,130 questions (the eval prints how many have a sheet; all do). Record the commit each run was
made at (`git rev-parse HEAD`) in the Log.

| Run | Conditions | Task or command | Output | Commit |
|---|---|---|---|---|
| A | A | the first command above | `results/zeroshot/vsibench/gold_sheet_relations_full/qwen3_5_2B_A_video.jsonl` | |
| B | B | the second | `.../qwen3_5_2B_B_gold_csv.jsonl` | |
| C | C | the third | `.../qwen3_5_2B_C_gold_relations_csv.jsonl` | |
| D | D | the fourth | `.../qwen3_5_2B_D_gold_csv_no_video.jsonl` | |
| E | E | the fifth | `.../qwen3_5_2B_E_gold_relations_csv_no_video.jsonl` | |

**Cost:** about 20–30 GPU minutes on one B200 for A–C (15,390 answers) and the checks, estimated from the format
experiment, whose checks and eleven runs of the same model and frames (30,008 answers) took 31 minutes there. D and E
(10,260 answers) carry no video, so their inputs are a few hundred tokens instead of about 3,000: an estimated 5–10
minutes more for both, about 30–40 in all.
C's longest prompts (Threats) make a few of its batches slower. Building and checking the gold sheets took under two
minutes on a laptop CPU.

### Fallback: the 75 exported scenes (2,728 questions)

If the full runs are too much, the same steps with `--ids experiments/2026-11-19-gold-sheet-relations/question_ids.txt`
added to `COMMON` and `OUT=results/zeroshot/vsibench/gold_sheet_relations`. Keep `SHEETS=tasks/openvocab_adapter/out_full`
(step 1 still runs): its sheets of these scenes are the current code's, unlike `tasks/openvocab_adapter/out`'s
(Decisions). About 15–20 GPU minutes with the checks (8,184 answers). These questions are a subset of the full run,
so the fallback is not a replication of it. D and E take the same `--ids` (about 5 minutes more).

## Results

<!-- make_table:start -->
Not run yet.
<!-- make_table:end -->

### Examples

-

### What the results show

-

## Conclusion

- **Answer to the question:**
- **Hypothesis:** <!-- supported, rejected or inconclusive, and why -->
- **Caveats:** <!-- what limits the conclusion: subset size, one seed, one model -->
- **Next:** <!-- the decision this leads to, and follow-up experiments by id -->

## Log

<!-- Dated changes to the plan, reruns and bugs found, each with its reason. -->

- 2026-11-19: plan written. `--sheet-source gold_relations` was added to the renderer and the eval (with tests) so
  that the pilot's best condition runs through `python -m mvp.eval.vsibench`; its relations are identical to the
  pilot's for all 2,728 questions. `question_ids.txt`, `smoke_ids.txt` and `prompt_check_ids.txt` are the format
  experiment's. The prompt and smoke checks use the broad-exploration folder's `cluster_checks.py`, which checks the
  sheets' current `visible` field; the format experiment's copy still checks `first_seen`, which the sheets no longer
  have.
- 2026-11-24, before any run of this experiment: the plan is updated with what is now known. structured-cot ran the
  same three conditions on 200 of these questions (Background), and the pipeline tests showed that the model copies
  given relations. So the hypothesis now expects C − B to come from the distance questions, and the threats cover
  both. Which questions get relations is stated as `gold_relations_v2` does it, unchanged (every question with two
  named classes in the sheet, not only distance questions), so that the results compare with the pilot's and
  structured-cot's. An optional run on the full VSI-Bench is described, with what it needs (gold sheets for 213 more
  scenes and a gold-only index) and its cost. The checks now use the format experiment's `cluster_checks.py`, into
  which the broad-exploration copy was merged (`6011ba7`); it checks the sheets' `visible` field. No condition,
  setting or question of the 75-scene plan changed.
- 2026-11-24, before any run: the plan moves to the full VSI-Bench (5,130 questions, 288 scenes), with the 75 scenes
  kept as a fallback. Gold sheets need only VSI-Bench's labels, so every scene can have them:
  - `scene_sheet_report.py --prompts` takes the prompts from a file, and `sheet_index.py --gold-only` indexes gold
    reports without pipeline sheets or an export (both tested in `test_openvocab_adapter.py`).
  - `prompts_full.jsonl` holds every question's prompts, extracted once on Colab with spaCy 3.8 /
    `en_core_web_md` 3.8.0 (`extract_prompts.py`, `prompts_full.ipynb`). For the 2,728 questions of the 75 scenes
    they equal the prompts of the 75-scene sheets.
  - `build_gold_sheets.py` builds all 288 scenes' gold sheets and the index with numpy only. It ran locally twice with
    the same hashes (Runs), with no failed sheet, and `check_sheets.py` rendered B and C for all 5,130 questions.
  - Building the 75 scenes again and comparing with a local copy of their sheets (`data/scene_sheets_75`) showed
    that this copy predates `980cf00`, which fitted the gold boxes to their objects: the positions are the same, the
    extents differ in 2,643 of 2,728 sheets. The runs use the new build for every question, the fallback included.
    (The next entry: structured-cot's answers do not fit this copy.)
- 2026-11-24, a preview, not a result of this experiment: `preview_explore200.ipynb` (Colab, one L4) ran B and C on
  structured-cot's 200 explore200 questions with the gold sheets rebuilt at `87f3685` (the README's hashes), reusing
  structured-cot's video-only run after checking that its frames and input lengths are this code's.
  - The runs started from empty files (the eval reported 0 of 200 done; `.started` markers with the commit and the
    sheets' hash) and took 3.7 and 3.8 minutes. B scored 52.1 and C 62.7 overall.
  - Both equal structured-cot's `gold_csv.jsonl` and `gold_relations.jsonl` byte for byte: all 200 answers, input
    lengths and frames.
  - Which sheets structured-cot read is not recorded (its notebook takes a sheets zip by name; `run_info.json` does not
    list it). The zip of that name on Drive, `scene_sheets_75.zip` (18 Nov, 68.6 MB; compared through a download of the same
    date and size), predates `980cf00`: for these
    200 questions 28 of its gold `objects.csv` equal the rebuild and 172 differ in the sizes, and B's and C's prompts
    rendered from it differ in the same 172. structured-cot's answers fit the rebuilt sheets, not that zip: 25 of 25
    absolute-distance answers in C equal the rebuilt closest-point distance (1 of 25 the zip's; question 783: 1.174,
    the zip's 0.888 m), and in the 19 size questions where the two differ, 8 answers in B equal a rebuilt box size
    (0 the zip's). So structured-cot most likely read a later sheet set with fitted extents; its outputs equal a
    run on the rebuild, but the file it read is not identified. The earlier entry's statement that structured-cot's numbers
    came from older sheets is withdrawn. The 40-question pilot's sheets were not checked. No condition, setting or
    question changed.
- 2026-11-24, before any run: conditions D (gold sheet only) and E (gold sheet + relations only), without the video,
  are added, because the proposal's success criteria ask for frames only, sheet only and both, with their token cost.
  The eval's new `--no-video` sends the sheet and the question as text (the prompt of B or C less the video and its
  sentence; tested in `test_scene_sheet.NoVideo`). A–C are unchanged: their prompts for all 5,130 questions equal
  those at `f3fdf2f`. D and E's comparisons are a second Holm family, so the A–C analysis is as planned, and input
  tokens per condition are reported as the token cost.
- 2026-11-24, before any run: each prediction now also records `n_prompt_tokens`, the question's own input length
  (`n_input_tokens` is the padded length of its batch of up to 8 questions, so it overstates short prompts). Nothing
  sent to the model changes; it is for the token cost of A–E.
- 2026-11-25, before any run: `--no-video` stopped with a TypeError after its first video's questions (the progress
  line read the frame size, which a run without the video doesn't have); found by an exploratory sheet-only run of
  pipeline_v4 (`experiments/2026-11-18-scene-sheet-formats/v4_sheet_only.md`), and fixed in `1abeca4` with an
  end-to-end test (`test_scene_sheet.NoVideo.test_run_without_video`). D and E, and E's smoke test, need that commit or
  later. Nothing sent to the model changes.
- 2026-11-25, before any run: the revamped pipeline was merged into this branch (`6ead380`). Its eval checks each
  video's frames against the question's pipeline sheet, which an index of gold sheets only (`build_gold_sheets.py`'s)
  does not have, so B and C stopped on their first video; `b0ed3c1` runs that check only for sheets that refer to
  frames (pipeline sheets, cameras), with a test of B and C's setting. Rechecked at `b0ed3c1`: the gold build gives the
  hashes above, and the prompts of A–E for all 5,130 questions are those at `0f7c5f2`. The runs need `b0ed3c1` or
  later.
- 2026-11-25: the five full-benchmark runs (A–E, 5,130 questions each) were made at `0f7c5f2`, which has
  `1abeca4`'s `--no-video` fix and predates the merge (`6ead380`) whose frame check `b0ed3c1` fixes, so B and C ran
  without it. Qwen/Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc` (`model_revision.txt`). The
  predictions and summaries are in `results/zeroshot/vsibench/gold_sheet_relations_full/` (`c4d74df`).

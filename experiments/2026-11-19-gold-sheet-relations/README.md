---
id: 2026-11-19-gold-sheet-relations    # also the folder name: experiments/<id>/
title: Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench
owner: Tamara Grujičić
created: 2026-11-19
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, gold-sheet, relations]
varies: [scene_sheet.source]  # keys that differ between conditions
depends_on: [2026-11-18-scene-sheet-formats (gold sheets, the 40-question pilot)]
---

# Gold sheets and precomputed relations

**Outcome:** <!-- one line once the experiment is done: the answer and its headline number -->

## Question

Do correct scene sheets, and precomputed relations, help Qwen3.5-2B on VSI-Bench?

## Hypothesis and what would change our mind

- **Expected result:**
  - The gold sheet helps over video only (B − A > 0), almost all of it on counting, as in the format experiment; without
    counting it adds little.
  - The relations help well beyond the gold sheet (C − B > 0), on absolute distance, relative distance, relative
    direction, appearance order and route planning, the question types that get relations. The model uses the
    distances it is given but does not compute them from coordinates (pilot).
- **Results that would prove it wrong:**
  - C − B ≤ 0, or within the smallest difference that matters: the pilot's gain from the relations was noise.
  - C − B positive only on absolute and relative distance: the model copies the answer lines (the relations state the
    answer to most distance questions) but gets nothing else from the relations.
  - B − A clearly positive without counting: the gold sheet helps on its own, unlike in the format experiment.
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
- **Assumption:** that the gold sheet's axis-aligned boxes are close enough to the objects' shapes for closest-point
  distances; the pilot's 5 distance questions are the only check.

## Design

### Conditions

<!-- One row per condition, and one column per key in `varies`. Every other setting is in Configuration. -->

| ID | Condition | `scene_sheet.source` | Role | Existing run |
|---|---|---|---|---|
| A | Video only | none | baseline | |
| B | Video + gold sheet (csv) | gold | treatment | |
| C | Video + gold sheet + relations (csv) | gold_relations | treatment, upper bound for "uses given relations" | |

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run. -->

- **B − A: does a correct scene sheet help?** Also without counting, since the gold sheet contains the counting answer.
- **C − B: do precomputed relations help beyond the sheet?** The primary comparison. Also on the question types that
  get relations, and with absolute and relative distance (where the relations state the answer) apart from the rest.
- **C − A:** both together, descriptive.

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source. -->

| Decision | Choice | Why |
|---|---|---|
| Questions | All 2,728 of the 75 exported scenes (`question_ids.txt`) | The scenes with sheets; the pilot's 40 are among them, which a sign test on 2,728 questions barely feels (1.5%) |
| Format | csv | Among the best and the shortest in the format experiment; the pilot's format |
| Relations | Closest-point distances (`gold_relations`) | VSI-Bench's distance questions measure between closest points (Background) |
| Decoding | 2,048 new tokens, final answer scored (`--final-answer`) | The format experiment's and the pilot's settings: replies were bare answers, and the same rule scores every condition |
| Runs | All three at one commit, none reused | The renderer has changed since the format experiment (`980cf00`, `080acf4`, `6642c48`) |

## Configuration

<!-- The settings every condition shares. A key in `varies` has the value `varies` here, and its values in the
Conditions table. -->

```yaml
data:
  benchmark: vsibench          # vsibench | mmsibench
  split: full                  # full (5,130 questions) | debiased (2,362) | pruned (2,768)
  subset: experiments/2026-11-19-gold-sheet-relations/question_ids.txt  # the 2,728 questions of the 75 exported scenes
  scenes: tasks/vggt_data/scenes.csv (75 exported, 2,728 questions)
  question_types: all          # all 10
  n_questions: 2728

visual:                        # what the VLM sees of the video
  video: true
  model: Qwen/Qwen3.5-2B       # revision: the cluster's cached snapshot, recorded by the smoke test (model_revision.txt)
  protocol: video              # one Qwen video, frames merged in pairs
  nframes: 16
  sampling: uniform128         # 16 uniform frames among the 128 VGGT frames, as the sheets
  layout: sit                  # video, then question (lmms-eval)
  resolution: default

scene_sheet:
  source: varies               # none (A) | gold (B) | gold_relations (C)
  geometry: ground truth (source)  # gold sheets: VSI-Bench's labelled boxes
  frames: n/a                  # gold sheets are not built from frames
  sam3_prompts: ground truth   # the question's prompts matched to the dataset's classes
  sam3: n/a
  tracking: n/a
  content: [position, size]    # C adds closest-point distances between the named objects (relations)
  units: metres                # gold
  format: csv
  renderer: src/mvp/benchmarks/scene_sheet.py:render_sheet (relations: scene_sheet.relations)
  placement: between video and question

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

- **Primary metric:** VSI-Bench overall (8 tasks) on the 2,728 questions, for C − B.
- **Secondary metrics:** overall without counting; score per task and question type; parse rate
  (`compare_runs.parse_rate`, the final-answer rule); input tokens; the share of questions that get relations.
- **Pipeline metrics:** n/a (gold sheets only).
- **Breakdowns:** question type, always; questions with relations vs without; absolute and relative distance (the
  relations state the answer) vs the other types with relations; dataset (ScanNet, ScanNet++, ARKitScenes).
- **Statistics:** every condition on the same questions, compared question by question: how many get better and
  worse, with an exact two-sided sign test (`mvp.reporting.compare_runs.sign_test`), Holm-corrected over B − A and
  C − B.
- **Smallest difference that matters:** 2 points overall, 5 on a task, as in the format experiment.

## Threats to validity and checks

<!-- Keep the rows that apply and add this experiment's own. -->

| Risk | How we'd notice | What to do |
|---|---|---|
| The gold sheet contains the counting answer | Counting near 98% in B and C | Read B − A and C − A without counting too |
| The relations state the answer to most distance questions | C near 100 on absolute and relative distance | C − B there measures copying a given number; read the other types with relations apart |
| Gold boxes are axis-aligned, so closest-point distances can be off for rotated objects | C's absolute-distance errors concentrated on large or diagonal objects | Stated; the pilot's 5 checks are the only evidence |
| The relations lengthen the prompt (up to dozens of lines for a class with many instances) | Input tokens in C, and C − B lower where the relations are longest | Report tokens; break down by relation lines if needed |
| The pilot's conditions are among these questions | — | 40 of 2,728; the pilot motivates the plan, it is not a result here |
| The model's revision or the sheets differ from the pilot's | The smoke test's revision; the prompt check | Stated in the Log |

## Runs and cost

<!-- One row per run. Output: the folder with the predictions (.jsonl), the summaries and the run's freeze.txt or
environment.txt. -->

All commands run from the repository root on the cluster, on `scene_sheet_formats` at a commit with this folder, in
the project's environment. The sheets of `run_sheets.sh` are in `tasks/openvocab_adapter/out/`.

**Checklist, before the runs:**

1. **The prompts.** The full prompts of `prompt_check_ids.txt` for C, without loading the weights, checked against
   `build_sheet_prompt` (the video's labels, the sheet and the relations, the whole prompt) with the broad-exploration
   folder's `cluster_checks.check_prompts`:

   ```bash
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --sheet-source gold_relations \
       --sheet-format csv --sheets-root tasks/openvocab_adapter/out \
       --ids experiments/2026-11-19-gold-sheet-relations/prompt_check_ids.txt --print-prompt 4 > prompts_C.txt
   PYTHONPATH=src python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/2026-11-19-broad-exploration-of-approaches'); import cluster_checks as c; p = c.check_prompts(Path('prompts_C.txt').read_text(), Path('tasks/openvocab_adapter/out'), Path('data/vggt'), source='gold_relations'); print(*p, sep='\n') if p else print('OK'); sys.exit(bool(p))"
   ```

2. **20-question smoke test** of A and C, then the model revision:

   ```bash
   SMOKE="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer --ids experiments/2026-11-19-gold-sheet-relations/smoke_ids.txt"
   python -m mvp.eval.vsibench $SMOKE --out results/_scratch/gsr_smoke/A.jsonl
   python -m mvp.eval.vsibench $SMOKE --sheet-source gold_relations --sheet-format csv \
       --sheets-root tasks/openvocab_adapter/out --out results/_scratch/gsr_smoke/C.jsonl
   python experiments/2026-11-19-broad-exploration-of-approaches/cluster_checks.py smoke \
       results/_scratch/gsr_smoke/A.jsonl results/_scratch/gsr_smoke/C.jsonl --n 20
   python experiments/2026-11-19-broad-exploration-of-approaches/cluster_checks.py revision Qwen/Qwen3.5-2B \
       results/zeroshot/vsibench/gold_sheet_relations/model_revision.txt
   ```

**The 3 runs**, one command per condition. They are independent and each resumes from its `--out` file if stopped:

```bash
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer --ids experiments/2026-11-19-gold-sheet-relations/question_ids.txt"
OUT=results/zeroshot/vsibench/gold_sheet_relations
SHEETS=tasks/openvocab_adapter/out
python -m mvp.eval.vsibench $COMMON --out $OUT/qwen3_5_2B_A_video.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_B_gold_csv.jsonl
python -m mvp.eval.vsibench $COMMON --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_C_gold_relations_csv.jsonl
```

Record the commit each run was made at (`git rev-parse HEAD`) in the Log.

| Run | Conditions | Task or command | Output | Commit |
|---|---|---|---|---|
| A | A | the first command above | `results/zeroshot/vsibench/gold_sheet_relations/qwen3_5_2B_A_video.jsonl` | |
| B | B | the second | `.../qwen3_5_2B_B_gold_csv.jsonl` | |
| C | C | the third | `.../qwen3_5_2B_C_gold_relations_csv.jsonl` | |

**Cost:** about 10–15 GPU minutes in all, estimated from the format experiment, whose checks and eleven runs of the
same model, frames and questions took 31 minutes on one B200; C's prompts are longer than B's where there are many
objects.

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

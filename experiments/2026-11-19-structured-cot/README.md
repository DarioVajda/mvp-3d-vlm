---
id: 2026-11-19-structured-cot    # also the folder name: experiments/<id>/
title: Does structured chain-of-thought help Qwen3.5-2B use a gold scene sheet on VSI-Bench, and does it recover what precomputed relations give
owner: Tamara Grujičić
created: 2026-11-19
status: draft                # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, prompting, reasoning, chain-of-thought]
varies: [prompt.reasoning, scene_sheet.source, scene_sheet.content]  # keys that differ between conditions
depends_on: [2026-11-18-scene-sheet-formats (gold sheets, explore pilot), --sheet-source gold_relations]
---

# Structured chain-of-thought with scene sheets

**Outcome:** <!-- one line once the experiment is done: the answer and its headline number -->

## Question

Does step-by-step reasoning help Qwen3.5-2B use a scene sheet on VSI-Bench, and does it recover what precomputed
relations give it?

## Hypothesis and what would change our mind

- **Expected result:**
  - With the gold sheet, structured chain of thought helps (D − C > 0), because the sheet supplies the numbers to
    reason over: the model can copy two objects' coordinates and compute their distance in writing.
  - It recovers part of the gap between the gold sheet and the gold sheet with relations (C < D < E): part, not all,
    because a 2B model computes less reliably than the relations state.
  - Without a sheet it gives little or no gain (B − A ≈ 0 or below), as VSI-Bench's authors report for linguistic
    prompting (Background).
- **Results that would prove it wrong:**
  - D − C ≤ 0, overall and on the tasks that need computing from coordinates (absolute and relative distance, object
    size): reasoning in writing does not let the model use correct coordinates.
  - D ≥ E: computing from the sheet is as good as being given the relations, so precomputing them adds nothing.
  - B − A clearly positive (Holm p < 0.05 and above the smallest difference that matters): reasoning helps without a
    sheet, against VSI-Bench's finding, which would point to the prompt rather than the sheet.
- **What we do next in each case:**
  - CoT helps with the sheet and closes much of the gap to E: try it with the pipeline sheet, and keep relations as
    the upper bound to aim for.
  - CoT helps with the sheet but leaves most of the gap to E: precomputed relations are the better way to give the
    model spatial facts; CoT is secondary.
  - CoT doesn't help even with the sheet: precomputed relations are the way forward; stop prompting the 2B to compute
    from coordinates, and try a larger model only if relations alone are not enough.

## Background

<!-- What is already known: the results this builds on (numbers and paths, e.g. the README's tables), design notes in
docs/, earlier experiments, and papers in papers/ (file, page and verbatim quote). Mark anything that no result or
paper backs as an assumption. -->

- **Chain-of-thought prompting** (Wei et al. 2022, NeurIPS, "Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models"): writing out reasoning steps in the prompt improves arithmetic and multi-step reasoning, but the
  gains appear mainly in large models. **A risk for a 2B model** (Threats). TODO: add the PDF to `papers/` with page
  and verbatim quote.
- **Zero-shot chain of thought** (Kojima et al. 2022, NeurIPS, "Large Language Models are Zero-Shot Reasoners"):
  adding "Let's think step by step", without worked examples, also elicits reasoning. Our prompt is zero-shot in this
  sense, with a fixed structure. TODO: PDF, page and quote.
- **Linguistic prompting on VSI-Bench** (Yang et al. 2025, CVPR, "Thinking in Space", which introduced VSI-Bench;
  arXiv 2412.14171): chain of thought, self-consistency and tree-of-thoughts did not improve VSI-Bench, while asking
  the model to first generate a cognitive map did. This suggests the models lack spatial information rather than
  reasoning steps. A scene sheet gives the model that information, which is why this experiment tests reasoning with
  and without one. TODO: PDF, page and quote.
- **Our 40-question pilot** (`experiments/2026-11-18-scene-sheet-formats/explore.ipynb`; predictions on Google Drive,
  `MyDrive/mvp/ssf_explore_results/`): Qwen3.5-2B, 16 frames, direct answers, final answer scored. Video only 47.3;
  video + gold sheet (csv) 54.7; video + gold sheet + precomputed relations (closest-point distances between the
  objects the question names, `gold_relations_v2`) 67.3. The model uses relations it is given but does not compute
  them from the coordinates it is given. **40 questions is noisy**: one question moves a task's score by 20 points and
  the overall by about 2.5. TODO: copy the pilot's summary into the repository, so this is checkable without the
  Drive.
- **The 200-question CoT pilot** (`explore_cot.md`, from `explore_cot.ipynb`; predictions in
  `results/zeroshot/vsibench/structured_cot/explore200/`): all five conditions on explore200's questions, one commit.
  D − C −3.3 (Holm p = 0.27), B − A −13.3 (Holm p = 0.004), E − D +13.9 (Holm p = 0.0002). With the sheet, the model
  computes centre-to-centre distances, with correct arithmetic, where VSI-Bench measures between closest points. 12%
  of D's and 28% of B's replies loop until the 2,048-token limit. The gold sheet also contains the object-size answers.
- **Format experiment** (`experiments/2026-11-18-scene-sheet-formats/README.md`, Results): on 2,728 questions with
  direct answers, every reply was a bare letter or number, and the gold sheet's gain over video only came from
  counting alone. Gold coordinates in metres did not help on absolute or relative distance or relative direction.
- **The chain-of-thought prompt exists:** `--reasoning cot` in `src/mvp/eval/vsibench.py` and `cot_prompt` in
  `src/mvp/benchmarks/vsibench.py` (commit `50fe543`, on `scene_sheet_formats`). It asks for the objects, the facts
  (positions, distances, sizes), the computation one step at a time and an "Answer: X" line, with one hint per task;
  it implies `--final-answer` and 2,048 new tokens. The exact text is in `cot_prompt.md`.
- **Assumption:** that 2,048 new tokens are enough for the reasoning. The smoke test measures reply lengths and
  truncation (Runs).
- **Assumption:** that the gold sheet's axis-aligned boxes give closest-point distances close enough to VSI-Bench's
  for E to be an upper bound. On the pilot's 5 absolute-distance questions they were within 0.3 m of the answers
  (explore notebook, round with `gold_relations_v2`), which is too few to rely on.

## Design

### Conditions

<!-- One row per condition, and one column per key in `varies`. Every other setting is in Configuration. -->

All conditions: Qwen3.5-2B, 16 frames (`uniform128`), the sheet as csv between the video and the question, greedy
decoding, the final answer scored by one rule (`score_final`), the same questions, the same commit.

| ID | Condition | `prompt.reasoning` | `scene_sheet.source` | `scene_sheet.content` | Role | Existing run |
|---|---|---|---|---|---|---|
| A | Video only, direct | direct | none | n/a | baseline | |
| B | Video only, structured CoT | cot | none | n/a | treatment | |
| C | Video + gold sheet, direct | direct | gold | positions and sizes | control | |
| D | Video + gold sheet, structured CoT | cot | gold | positions and sizes | treatment | |
| E | Video + gold sheet + relations, direct | direct | gold_relations | + closest-point relations | upper bound | |

- **Structured CoT:** list the relevant objects → read their coordinates → compute → answer. It runs as the eval's
  `--reasoning cot` (`cot_prompt.md`), whose steps are these four: say which objects the question is about and find
  them in the video and the sheet; write down the facts needed, such as positions and distances; work out the answer
  one step at a time, showing the arithmetic; end on "Answer: X".
- **Relations (E):** after the gold sheet, the distances in metres between the closest points of the boxes of every
  two objects of different named classes, the two objects' distance for absolute-distance questions, and each
  candidate's distance to the target for relative-distance questions: `--sheet-source gold_relations`
  (`src/mvp/benchmarks/scene_sheet.py`, `relations`), which writes the same text as `gold_relations_v2` in the explore
  notebook for all 2,728 questions.
- **Qwen3.5's thinking mode** is not a condition for now: the code turns it off everywhere
  (`src/mvp/models/qwen_vl.py`: `enable_thinking: False`), and structured CoT through the prompt answers the question
  without it (Log, 2026-11-20).

### Comparisons

<!-- The differences between conditions that answer the question, fixed before the first run. -->

- **D − C: does CoT help when there is a sheet?** The primary comparison.
- **B − A: does CoT help without a sheet?**
- **E − D: do precomputed relations beat computing them in writing?**
- Descriptive: E − C (what relations give with direct answers; the pilot's gap), and (D − C) − (B − A) (does the sheet
  change what CoT does).
- Every comparison also without counting, since the gold sheet contains the counting answer, and on the computing
  tasks alone (absolute distance, relative distance, object size).

### Decisions

<!-- Choices the configuration alone doesn't explain (subset, frame count, model), each with its reason or source. -->

| Decision | Choice | Why |
|---|---|---|
| CoT prompt | The eval's `--reasoning cot` (`50fe543`), unchanged | It already has the four steps and an "Answer: X" line that `score_final` reads; a second prompt would duplicate it. An explicit "copy the objects' rows from the sheet" step is considered only if the reasoning check shows D ignoring the sheet, and then as an option of `cot_prompt` (TODO 4) |
| Sheet | Gold only | The pilot shows the direct model does not compute from correct coordinates; gold isolates the reasoning from detection errors. The pipeline sheet is the follow-up if CoT helps |
| Format | csv | Among the best and the shortest in the format experiment |
| Relations | Closest-point distances (`gold_relations_v2`) | VSI-Bench measures distance between closest points ("Measuring from the closest point of each object", all 428 absolute- and 379 relative-distance questions); centre distances were 0.5–1.6 m off on the pilot's 5 absolute-distance questions |
| Token limits | 2,048 new tokens in every condition | `--reasoning cot`'s default; the direct runs of the format experiment used the same limit with `--final-answer`, so the limit is not a difference between conditions. Raised for CoT if the smoke test shows more than 5% of replies cut off |
| Answer extraction | `score_final` in every condition (`--final-answer`) | One rule for all; a bare letter or number scores the same under it as under lmms-eval's first-word rule (format experiment: identical in every run) |
| Questions | All 2,728 (`question_ids.txt`), or the 933 of `subset_100_per_type.txt` if the CoT runs would take more than 4 GPU hours | CoT replies are hundreds of tokens against one; the subset keeps up to 100 per question type (all 75 room-size and 58 route-planning questions), drawn with seed 0. Neither overlaps the choice of anything in this experiment |
| Commit | All conditions at one commit, none reused | The eval and the sheets have changed since the format experiment (`980cf00`, `080acf4`, `6642c48`), so its runs are not the same configuration |

## Configuration

<!-- The settings every condition shares. A key in `varies` has the value `varies` here, and its values in the
Conditions table. -->

```yaml
data:
  benchmark: vsibench          # vsibench | mmsibench
  split: full                  # full (5,130 questions) | debiased (2,362) | pruned (2,768)
  subset: experiments/2026-11-19-structured-cot/question_ids.txt  # or subset_100_per_type.txt (Decisions)
  scenes: tasks/vggt_data/scenes.csv (75 exported, 2,728 questions)
  question_types: all          # all 10
  n_questions: 2728            # or 933

visual:                        # what the VLM sees of the video
  video: true
  model: Qwen/Qwen3.5-2B       # revision: TODO, recorded by the smoke test
  protocol: video              # one Qwen video, frames merged in pairs
  nframes: 16
  sampling: uniform128         # 16 uniform frames among the 128 VGGT frames, as the sheets
  layout: sit                  # video, then question (lmms-eval)
  resolution: default

scene_sheet:
  source: varies               # none (A, B) | gold (C, D) | gold_relations (E)
  geometry: ground truth (source)  # gold sheets: VSI-Bench's labelled boxes
  frames: n/a                  # gold sheets are not built from frames
  sam3_prompts: ground truth   # gold sheets: the question's prompts matched to the dataset's classes
  sam3: n/a
  tracking: n/a
  content: varies              # positions and sizes (C, D) | + closest-point relations (E)
  units: metres                # gold
  format: csv
  renderer: src/mvp/benchmarks/scene_sheet.py:render_scene (relations: scene_sheet.relations)
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval          # build_prompt in src/mvp/benchmarks/vsibench.py
  reasoning: varies            # direct (A, C, E) | structured, cot_prompt.md (B, D)
  thinking: off                # enable_thinking=False in every condition
  max_new_tokens: 2048         # every condition (Decisions); raised for CoT if the smoke test truncates
  decoding: greedy
  answer_extraction: score_final  # the last "Answer: X", else the last lone letter / the last number

training: none
```

## Metrics and analysis

- **Primary metric:** VSI-Bench overall (8 tasks) on the chosen questions, for D − C.
- **Secondary metrics:** overall without counting; the computing tasks' mean (absolute distance, relative distance,
  object size); score per task and question type; parse rate (`parsed(..., final=True)`, as `compare_runs.parse_rate`
  chooses for runs with `score_first`); reply length (median and 90th percentile, tokens) and the share of replies
  that hit the token limit, per condition and task.
- **Reasoning check:** in D, the share of replies that name a sheet row (an `id` such as `chair_2`) or copy one of its
  coordinates; and, on 20 wrong D answers drawn with seed 0 from the computing tasks, a hand-labelled reason: wrong
  row, coordinates misread, arithmetic wrong, sheet ignored, no answer line.
- **Pipeline metrics:** n/a (gold sheets only).
- **Breakdowns:** question type, always; computing tasks vs the rest; multiple choice vs numeric; dataset (ScanNet,
  ScanNet++, ARKitScenes).
- **Statistics:** every condition on the same questions, compared question by question: how many get better and
  worse, with an exact two-sided sign test (`mvp.reporting.compare_runs.sign_test`), Holm-corrected over the three
  main comparisons (D − C, B − A, E − D) as one family. The descriptive comparisons are reported without a test.
- **Smallest difference that matters:** 2 points overall, 5 on a task or on the computing tasks' mean, as in the
  format experiment.

## Threats to validity and checks

<!-- Keep the rows that apply and add this experiment's own. -->

| Risk | How we'd notice | What to do |
|---|---|---|
| CoT replies are long, so runs cost far more than direct ones | The smoke test's time per question | Use `subset_100_per_type.txt` if the CoT runs would take more than 4 GPU hours (Decisions) |
| CoT replies are cut off by the token limit, so CoT loses for length alone | More than 5% of replies hit the limit in B or D | Raise the limit and rerun that condition; report truncation per condition regardless |
| Parse failures: the final-answer rule reads a letter or number from the reasoning, or finds none | Parse rate under 95% in a condition, or "no answer line" common in the reasoning check | Read 20 replies by hand; fix the rule for all conditions alike and rescore |
| The 2B model is too small to benefit from CoT (Wei et al. 2022) | D − C and B − A both near 0, replies that copy rows but compute wrongly | Read as a result about this model size, not about CoT; the next step is relations (Hypothesis) |
| The gold sheet contains the counting answer | Counting near 98% in C, D, E (format experiment) | Read every comparison without counting too |
| The relations state the answer to most distance questions | E near 100 on absolute and relative distance | E is an upper bound for "uses the relations it is given", not a test of spatial reasoning |
| The CoT prompt's per-task hints differ from task to task | Large gains on one task only | Report per task; one wording is a caveat, not a condition |
| Structured CoT's last step asks to check against the video, which may pull the model away from the sheet | D's reasoning ignores the sheet (reasoning check) | Stated; if this is common, consider a "copy the rows" step (TODO 4) |
| The pilot's numbers come from an older commit and 40 questions | — | They motivate the design only; no run is reused (Decisions) |

## Runs and cost

<!-- One row per run. Output: the folder with the predictions (.jsonl), the summaries and the run's freeze.txt or
environment.txt. -->

All commands run from the repository root on the cluster, on `scene_sheet_formats` at a commit with this folder, in
the project's environment. The sheets of `run_sheets.sh` are in `tasks/openvocab_adapter/out/`.

**Checklist, before the runs:**

1. **The prompts** of `prompt_check_ids.txt` for D and E, without loading the weights, checked against
   `build_sheet_prompt` with the broad-exploration folder's `cluster_checks.check_prompts`:

   ```bash
   SHEETS=tasks/openvocab_adapter/out
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --sheet-source gold \
       --sheet-format csv --sheets-root $SHEETS --reasoning cot \
       --ids experiments/2026-11-19-structured-cot/prompt_check_ids.txt --print-prompt 4 > prompts_D.txt
   python -m mvp.eval.vsibench --model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --sheet-source gold_relations \
       --sheet-format csv --sheets-root $SHEETS \
       --ids experiments/2026-11-19-structured-cot/prompt_check_ids.txt --print-prompt 4 > prompts_E.txt
   for x in "D gold" "E gold_relations"; do set -- $x; PYTHONPATH=src python -c "import sys; from pathlib import Path; sys.path.insert(0, 'experiments/2026-11-19-broad-exploration-of-approaches'); import cluster_checks as c; p = c.check_prompts(Path('prompts_$1.txt').read_text(), Path('$SHEETS'), Path('data/vggt'), source='$2'); print(*p, sep='\n') if p else print('$1 OK'); sys.exit(bool(p))"; done
   ```

   The check rebuilds each prompt with direct reasoning, so for D it reports "the prompt is not build_sheet_prompt's"
   for every question, as the CoT instructions differ; its label and sheet checks are the ones that matter for D.

2. **20-question smoke test** of B and D, which records reply lengths, truncation, parse rate and the time per
   question, and from it the question set (Decisions); then the model revision:

   ```bash
   SMOKE="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer --reasoning cot --ids experiments/2026-11-19-structured-cot/smoke_ids.txt"
   python -m mvp.eval.vsibench $SMOKE --out results/_scratch/scot_smoke/B.jsonl
   python -m mvp.eval.vsibench $SMOKE --sheet-source gold --sheet-format csv --sheets-root tasks/openvocab_adapter/out \
       --out results/_scratch/scot_smoke/D.jsonl
   python experiments/2026-11-19-broad-exploration-of-approaches/cluster_checks.py smoke \
       results/_scratch/scot_smoke/B.jsonl results/_scratch/scot_smoke/D.jsonl --n 20
   python experiments/2026-11-19-broad-exploration-of-approaches/cluster_checks.py revision Qwen/Qwen3.5-2B \
       results/zeroshot/vsibench/structured_cot/model_revision.txt
   ```

   `cluster_checks.py smoke` also asks that the second run's prompts be longer than the first's, which holds here
   because D carries the sheet.

**The runs**, one command per condition. IDS is `question_ids.txt` (2,728) or, if the smoke test projects more than 4
GPU hours for B and D, `subset_100_per_type.txt` (933), the same for every condition. They are independent and each
resumes from its `--out` file if stopped:

```bash
IDS=experiments/2026-11-19-structured-cot/question_ids.txt
COMMON="--model Qwen/Qwen3.5-2B --nframes 16 --sampling uniform128 --max-new-tokens 2048 --final-answer --ids $IDS"
OUT=results/zeroshot/vsibench/structured_cot
SHEETS=tasks/openvocab_adapter/out
python -m mvp.eval.vsibench $COMMON                                                                                 --out $OUT/qwen3_5_2B_A_direct_video.jsonl
python -m mvp.eval.vsibench $COMMON --reasoning cot                                                                 --out $OUT/qwen3_5_2B_B_cot_video.jsonl
python -m mvp.eval.vsibench $COMMON                 --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_C_direct_gold.jsonl
python -m mvp.eval.vsibench $COMMON --reasoning cot --sheet-source gold           --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_D_cot_gold.jsonl
python -m mvp.eval.vsibench $COMMON                 --sheet-source gold_relations --sheet-format csv --sheets-root $SHEETS --out $OUT/qwen3_5_2B_E_direct_gold_relations.jsonl
```

Record the commit each run was made at (`git rev-parse HEAD`) in the Log.

| Run | Conditions | Task or command | Output | Commit |
|---|---|---|---|---|
| A | A | the first command above | `results/zeroshot/vsibench/structured_cot/qwen3_5_2B_A_direct_video.jsonl` | |
| B | B | the second | `.../qwen3_5_2B_B_cot_video.jsonl` | |
| C | C | the third | `.../qwen3_5_2B_C_direct_gold.jsonl` | |
| D | D | the fourth | `.../qwen3_5_2B_D_cot_gold.jsonl` | |
| E | E | the fifth | `.../qwen3_5_2B_E_direct_gold_relations.jsonl` | |

**Cost:** not known until the smoke test. Direct runs took about 3 minutes each on a B200 in the format experiment
(eleven runs in 31 minutes); CoT runs depend on reply length. The limit of 4 GPU hours for the CoT runs decides
between the full set and the subset.

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

- 2026-11-19: plan written (draft). It replaces an earlier, uncommitted chain-of-thought draft (a 3 × 3 design of
  reasoning mode by sheet), and with the explore pilot's results narrows the design to the gold sheet, uses structured
  CoT through the eval's existing `--reasoning cot` (`50fe543`) instead of a new prompt, and adds the relations
  condition (E) that the pilot showed to be the largest gain. `question_ids.txt`, `subset_100_per_type.txt`,
  `smoke_ids.txt` and `prompt_check_ids.txt` come from that draft (the subset: up to 100 per question type, seed 0);
  `cot_prompt.md` is generated from `cot_prompt` at `6642c48`.
- 2026-11-19: E now runs through the normal eval as `--sheet-source gold_relations`, whose relations are identical to
  the pilot's for all 2,728 questions, so the TODO to port the relations is closed and the Runs section gives the exact
  command for each condition. The prompt and smoke checks use the broad-exploration folder's `cluster_checks.py`,
  which checks the sheets' current `visible` field.
- 2026-11-20: condition F (Qwen3.5's thinking mode) is dropped for now. The eval turns thinking off everywhere, adding
  it would need code (an option that turns it on and scores the text after `</think>`), and the question is answered
  by structured CoT through the prompt (B, D) without it. The CoT prompt stays as `cot_prompt` writes it; a "copy the
  objects' rows from the sheet" step is decided only if the reasoning check shows D ignoring the sheet.
- 2026-11-20, the pilot: `explore_cot.ipynb` ran A–E on explore200's 200 questions on Colab (one L4). The code was
  commit `7d39234`: `abde60f` plus `metric_scale=None` in the notebook, which the eval's `--metric-scale` option
  needs. All five conditions ran at that commit; none of explore200's predictions was reused. The predictions are in
  `results/zeroshot/vsibench/structured_cot/explore200/` and the write-up is `explore_cot.md`. Its tables come from
  `make_table.py --explore`, which now also takes the reply lengths the eval records, adds a table of what fixed
  rules on the gold sheet answer, and counts the cut-off replies that loop. The reasoning check's 20 labels
  (`explore_cot_reasoning_labels.csv`) were assigned by Claude, reading each reply; no second labeller. For the
  plan:
  - Two Threats rows are triggered: replies cut off in B (28%) and D (12%), and parse rates (88.5%, 93.5%) under
    95%. Their planned responses don't fit. The cut-off replies are loops (frame-by-frame narration or repetition,
    51 of 56 in B and 20 of 23 in D), and they are the parse failures. A higher limit would not end them, and a lower
    one would cut off correct long replies.
  - The reasoning check does not show D ignoring the sheet (TODO 4). It shows D computing centre distances for
    closest-point questions (11 of 20 wrong answers) and the loops.
  - The prompt for the full run is an open decision (`explore_cot.md`, Conclusion).
  - At the pilot's decoding steps per batch, B and D would take at least about 17 hours on an L4 for all 2,728
    questions, and about 7 hours for `subset_100_per_type.txt`.

## TODO

1. Add Wei et al. 2022 (NeurIPS) to `papers/` with page and verbatim quote for "gains mainly in large models".
2. Add Kojima et al. 2022 (NeurIPS) to `papers/` with page and verbatim quote.
3. Add Yang et al. 2025 (CVPR, "Thinking in Space") to `papers/` with page and verbatim quotes for linguistic
   prompting not helping and the cognitive map helping.
4. After the runs: if the reasoning check shows D ignoring the sheet, decide on an explicit "copy the objects' rows
   from the sheet" step, added as an option of `cot_prompt` (not a second prompt), with `cot_prompt.md` regenerated.
   Until then `cot_prompt` stays unchanged.
5. Copy the pilot's summary (`MyDrive/mvp/ssf_explore_results/`) into the repository, and record the commit it ran at.
6. Smoke test: reply lengths, truncation, parse rate and time per question for B and D; then the question set and, if
   needed, a higher token limit. The pilot measured all four on 200 questions (`explore_cot.md`); what remains is the
   decision on the prompt and the question set.
7. Record the model revision (smoke test).
8. Fill in the Runs table's commits when the runs are made.

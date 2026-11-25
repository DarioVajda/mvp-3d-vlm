---
id: 2026-11-25-concise-math-cot
title: Concise, mathematical and few-shot structured CoT for Qwen3.5-2B on the computing tasks with the gold sheet
owner: Tamara Grujičić
created: 2026-11-25
status: planned              # draft | planned | running | analysing | done | abandoned
tags: [prompting, chain-of-thought, gold-sheet]
varies: [prompt.reasoning]
depends_on: [2026-11-19-structured-cot (its conditions A, C, D, E and their predictions), 2026-11-19-gold-sheet-relations (the rebuilt gold sheets)]
---

# Concise, mathematical and few-shot structured CoT

**Outcome:** <!-- once run: the answer and its headline number -->

## Question

Can a more constrained, mathematically explicit structured chain of thought, with or without one worked example per
task, make Qwen3.5-2B (no fine-tuning) answer the computing tasks better from the gold scene sheet than the pilot's
structured CoT (D), than direct answering with the sheet (C), and than the video alone (A)? How far is it from being
given the precomputed relations (E)?

## Hypothesis and what would change our mind

- **Expected result** (from the pilot's failure analysis, below):
  - F1 − D ≥ 0 with shorter replies and fewer cut off: the pilot's loops came from its check against the video.
  - F2 − F1 > 0, almost all of it on absolute distance: D computed centre distances (84% of its answers), and the
    closest-point formula is what VSI-Bench measures. Little on object size (D already took the longest side in 76%)
    and on relative distance in the development set (see Threats: there the two rules never disagree).
  - F3 − F2 ≥ 0, small: a worked example shows the formula once more.
  - E − F2 > 0 still: E is handed the closest-point distances; F2 has to compute them.
- **Results that would prove it wrong:** F2 ≤ F1 on absolute distance (the model does not apply a formula it is given),
  or F1 below D (brevity costs answers), or a rise in unread answers.
- **What we do next in each case:** if F2 or F3 improves on D and C on the development set, freeze it and evaluate it on
  held-out questions (Runs, Held-out evaluation), then on the pipeline's sheets. If none does, the model does not use
  a stated procedure, and precomputed relations (E) remain the way to give it distances.

## Background

### The pilot (`experiments/2026-11-19-structured-cot/explore_cot.md`, explore200, 200 questions)

On the computing tasks (absolute distance, relative distance, object size; 75 questions): A 61.5, C 48.5, D 58.0,
E 82.0. D − C = +9.5 (27 better, 19 worse, p = 0.30), almost all from object size. Read by hand and measured:

- **Wrong operation:** 84% of D's absolute-distance answers are the centre-to-centre distance and none the
  closest-point one; 11 of 20 labelled wrong answers used the wrong method, none got the arithmetic wrong.
- **Loops from the check against the video:** 11.5% of D's replies (all tasks) ran to 2,048 tokens, narrating
  invented mm:ss frames; cut-off replies score 8.7 against 54.0.
- **Rows:** 4 of 20 labelled errors used the wrong rows (relative distance mostly).
- **Units:** one object-size answer in metres (1.245 for 124.5 cm).

### Literature

- **Monjoree & Yan, "Agentic AI with Structured CoT for Enhancing AI's Spatial Intelligence: Visualization and
  Reasoning of Rotation", arXiv 2610.04188 (2026).** Mental rotation (Revised PSVT:R, 30 items), GPT-5.6 in its high
  reasoning mode, prompting only. A fixed per-task procedure (find reference features, infer the rotation, apply it to
  the options, check against criteria) raised accuracy from 13.3% to 40% without axes and to 100% with axes drawn on
  the items (Table I); adding 5 worked solutions of real test items gave 48% (Exp. 2, on the other 25). The prompts are
  in figures only; there is no plain-CoT or direct baseline, no small model, one item set and no repeated runs.
  **What we take:** the form, a fixed, task-specific sequence of steps ending in a verdict, and worked demonstrations
  in that same form. **What it does not show:** anything about a 2B model, distances, sizes or VSI-Bench. Our
  examples are synthetic, where theirs were real items of the same test.
- **Ji, Agrawal, Tang & Wu, "Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and
  Reinforcement Learning", arXiv 2507.13362 (2025).** Plain "think first" CoT is reported not to help and possibly to
  hurt spatial reasoning (abstract, introduction; no table row for it). A structured two-step "scene graph" CoT raised
  CV-Bench accuracy for Llama-4-Scout (53.8 → 75.7) and Qwen2.5-VL-7B (60.5 → 68.1), barely for the 3B PaLI-Gemma2
  (52.2 → 53.6) and lowered it after its SFT (66.8 → 57.1) (Table III). Program-, sketch- and visualization-of-thought
  variants were below direct answering (Table III, IV-D). Flat few-shot examples led to "the model repeating prior
  answers" (IV-E). **What we take:** a motivation to make the reasoning structured and short rather than free-form,
  and to mark worked examples as belonging to another scene. **What it does not show:** VSI-Bench, video, a 2B Qwen,
  or prompting gains for small models (its 3B results are mixed).

## Design

### Conditions

<!-- One row per condition. -->

| ID | Condition | `prompt.reasoning` | Role | Existing run |
|---|---|---|---|---|
| A | video only, direct | direct | baseline | pilot `A_video.jsonl` |
| C | video + gold sheet, direct | direct | baseline | pilot `gold_csv.jsonl` |
| D | video + gold sheet, structured CoT | cot | baseline (the pilot's CoT) | pilot `gold_cot.jsonl` |
| E | video + gold sheet + relations, direct | direct | upper reference | pilot `gold_relations.jsonl` |
| F1 | video + gold sheet, concise CoT | cot_f1 | treatment | |
| F2 | F1 + geometric formulas | cot_f2 | treatment | |
| F3 | F2 + one worked example | cot_f3 | treatment | |

### The prompts (`src/mvp/benchmarks/cot_variants.py`)

Each replaces the "answer directly" line, as D does; the video, the sentence before the sheet, the sheet and the
question are the same as in C, D and E. Pieces, in order:

| Piece | F1 | F2 | F3 | Source |
|---|:-:|:-:|:-:|---|
| `CONCISE`: reason briefly; use the sheet; the video only to tell which object is meant; no restating, describing, repeating | ✓ | ✓ | ✓ | the pilot (loops, verbosity); 2507.13362 (plain CoT does not help) |
| the task's hint, `vsibench.COT_HINTS`, unchanged from D | ✓ | ✓ | ✓ | D, kept so that F1 − D isolates the general instructions |
| `MATH[task]`: the operation and formula | | ✓ | ✓ | the pilot (wrong operation, units), checked against VSI-Bench (below) |
| `EXAMPLE_INTRO`, `EXAMPLES[task]`, `EXAMPLE_OUTRO`: one synthetic worked example | | | ✓ | 2610.04188 (worked solutions in the procedure's form); "for a different scene, do not use its numbers" after 2507.13362 IV-E |
| `FORMAT`: the lines `Objects:`, `Attributes:`, `Calculation:` | ✓ | ✓ | ✓ | 2610.04188 (a fixed procedure); the pilot (verbosity) |
| `vsibench.COT_ANSWER`: end on "Answer: X", as D | ✓ | ✓ | ✓ | unchanged, so the final-answer rule reads every condition the same way |

D's text (`experiments/2026-11-19-structured-cot/cot_prompt.md`) asks to describe, compute "one step at a time,
showing any arithmetic" and "check that the result agrees with what the video shows". F1 drops the check and the
showing, keeps the hint, and fixes the output lines. F2's formulas:

- **Absolute distance:** each row is a box, centre (x, y, z) and full extent (size_x, size_y, size_z), in metres; for
  each axis k, gap_k = max(0, |A_k − B_k| − (A_size_k + B_size_k) / 2); distance = √(gap_x² + gap_y² + gap_z²), 0 when
  the boxes overlap on all three axes; for objects with several rows, the smallest over pairs; metres, no conversion.
- **Relative distance:** the same distance, from each option's rows to the reference object's, smallest per option;
  the option with the smallest wins.
- **Object size:** the largest of size_x, size_y, size_z (not x, y, z, which are the centre); × 100 for centimetres; the
  largest row if there are several.

F3's examples (one per task, synthetic scenes and questions in the gold sheet's csv and the FORMAT lines): absolute
distance 1.059 m with one zero gap; relative distance where the nearest centre (shelf) and the nearest box (bed)
differ, so the example shows why centres are wrong; object size 1.830 m → 183 cm. `test_cot_variants.py` recomputes
their numbers with the code of the precomputed relations.

### How the conventions were checked

- **Against the code:** F2's distance is `scene_sheet.box_gap`, the function of E's relations
  (`scene_sheet.relations`); `test_cot_variants.Geometry` checks them equal on 200 random box pairs and on overlapping
  and touching boxes.
- **Against VSI-Bench's answers, on every question whose named objects are in the rebuilt gold sheets** (`out_full`,
  test.jsonl at `3a6a1cf`; computed 2026-11-25):

  | Rule applied to the gold sheet | Task | Questions | VSI-Bench score |
  |---|---|---:|---:|
  | closest-point gap (F2's formula), min over pairs | absolute distance | 830 | 91.4 |
  | centre-to-centre (what D did) | absolute distance | 830 | 34.3 |
  | nearest option by closest-point gap | relative distance | 688 | 98.3 |
  | nearest option by centres | relative distance | 688 | 83.7 |
  | largest side × 100 | object size | 952 | 100.0 |
  | largest side, no conversion | object size | 952 | 0.0 |

  All 710 relative-distance questions ask for the closest object (none for the farthest), so F2 says "smallest". 7
  object-size questions have more than one gold row of the class; F2 says to take the largest.
- **Not verified, and stated:** the boxes are treated as aligned with the sheet's axes. Gold boxes are fitted to each
  object and their sides matched to the nearest axes (`980cf00`), which is why the closest-point rule scores 91.4 and
  not 100; VSI-Bench's own distances come from its annotations, not from these boxes. The formula makes no use of an
  axis convention (it is symmetric in x, y, z).

### Comparisons

<!-- Fixed before the first run. -->

- **Primary, one family (Holm):** F1 − D (concise reasoning), F2 − F1 (formulas), F3 − F2 (worked example).
- **Secondary, each its own family (Holm):** F1, F2, F3 − C (against direct answering with the sheet); − E (against
  the relations); − A (against the video alone).
- On the computing score (the mean of the three tasks' VSI-Bench scores), per task, question by question.

### Controls

- **The same questions** in every condition compared (the analysis takes the questions all have answered and stops if
  a baseline lacks one), drawn by a fixed seed and nested.
- **The same model, frames, sheet text, decoding, 2,048-token limit, batch size and answer rule** as the pilot; only the
  instruction text after the question differs (`test_cot_variants.test_each_condition_has_only_its_intervention`
  checks F2 is F1 plus `MATH` and F3 is F2 plus the example, nothing else).
- **A–E unchanged:** their code paths are not touched (the prompts of A–E for all 5,130 questions are byte-identical to
  `9c29b23`, checked 2026-11-25), and their predictions are reused, not rerun (`test_followup.Preservation` pins the
  files' hashes).
- **No mixing:** each condition's folder holds a `config.json` with the prompt fingerprint and version, model,
  settings, gold sheets' index hash, test.jsonl's hash, seed and pool; a run under anything else stops.

**Unavoidable deviations:**
- F1 changes the general instructions and the output form together (both target the length); they are not separated.
- The baselines ran in the pilot's batches (up to 8 questions of a video); F1–F3 run on the subset's questions, so
  their batches are smaller. Greedy decoding in a batch can differ in the last digits from decoding alone; the
  pilot's D is not rerun to match.
- The pilot's sheets came from a zip that is not identified (2026-11-19-gold-sheet-relations, Log); a rerun of C and E
  on the rebuilt sheets (`out_full`) reproduced the pilot's predictions byte for byte, so D's sheet text was the same.

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: explore200's computing questions (development set), nested per task (followup.nested_subset, seed 0)
  scenes: explore200's 25 (the subset's)
  question_types: [object_abs_distance, object_rel_distance, object_size_estimation]
  n_questions: 3 / 15 / 30 / 75 by stage (followup.STAGES)

visual:
  video: true
  model: Qwen/Qwen3.5-2B
  protocol: video
  nframes: 16
  sampling: uniform128
  layout: sit
  resolution: default

scene_sheet:
  source: gold                 # tasks/openvocab_adapter/out_full (build_gold_sheets.py; index 22e0ee1e...)
  format: csv
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  reasoning: varies            # cot_f1 | cot_f2 | cot_f3 (cot_variants.PROMPT_VERSION 2026-11-25.1)
  thinking: off
  max_new_tokens: 2048
  decoding: greedy
  answer_extraction: score_final

training: none
```

## Metrics and analysis

`analysis.py` (exploratory below 25 questions per task, and says so in its output):
- the computing score and each task's (VSI-Bench's scoring, score_final), the parse rate (the final-answer rule), the
  median and 90th-percentile reply length, the share cut off at 2,048 tokens, seconds per question (`timing.jsonl`);
- per comparison: the difference, better / worse / same, an exact sign test, Holm within the family, and a paired
  bootstrap 95% interval (questions resampled within question type, 10,000 resamples, seed 0), as the pilot's analysis;
- `errors.csv`: per reply, what is measured automatically (cut off; no "Answer:" line; unread; the read answer equal to
  the centre rule or the closest-point rule from the gold sheet; an option picked by centres where closest points pick
  another; metres for centimetres; another side of the box; the gold sheet's own rule disagreeing with the ground
  truth; named objects missing from the sheet), and empty columns for what needs reading: object identification,
  attribute extraction, operation, arithmetic, unit, notes. A reply that quotes the sheet's ids or numbers is not
  counted as reasoning correctly; the flags say which number a reply ended on, not how it got there.
- Checked on the pilot's real D replies (with D's predictions standing in for F1–F3): the analysis reproduces D's
  computing score 58.0 and finds 84% of D's absolute-distance answers equal to the centre distance, the pilot's figure.

## Threats to validity and checks

| Risk | How we'd notice | What to do |
|---|---|---|
| Small subsets | the analysis prints "Exploratory" below 25 per task | read the smoke and pilot stages as checks of the prompts, not results |
| explore200 shaped these prompts | — | F results on it are development results; the claim needs the held-out evaluation |
| On explore200's relative-distance questions the centre and closest-point rules pick the same answer (all 23 with every option in the sheet) | — | F2's formula cannot show a gain on relative distance there; the held-out pool has questions where they differ (83.7 vs 98.3 on the full benchmark) |
| The gold sheet gives the object-size answer (largest side) | object size near 100 | read object size apart; the gains that matter are on distances |
| The model copies the example's numbers or answer (2507.13362 IV-E) | F3 answers equal to 1.059, 183 or the example's letter | the example is marked as another scene; check `errors.csv` and the replies |
| Axis-aligned boxes | absolute-distance errors on large or rotated objects | stated; the same approximation as E |
| Batches differ from the pilot's | — | stated (Controls) |

## Runs and cost

**CPU tests** (no GPU, no model):

```bash
PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_cot_variants   # prompts, formulas, examples, parser
PYTHONPATH=src python -m unittest discover -s experiments/2026-11-25-concise-math-cot -p "test_*.py"   # runs, analysis
```

**A dry run** builds the exact prompts and the configs and prints the eval commands, without a model:

```bash
PYTHONPATH=src python experiments/2026-11-25-concise-math-cot/followup.py --k 1 \
    --sheets-root tasks/openvocab_adapter/out_full --out /tmp/cot_dry --dry-run
```

**Colab: the smoke test** (`cot_followup.ipynb`): upload `mvp-3d-vlm-cot-followup.zip` (the code; made from the
repository with `git archive`, its comment the commit) to `MyDrive/mvp/`, open the notebook on an L4 or A100, keep
`STAGE = "smoke"` and run all cells: 3 questions (one per task), 9 replies, about 15–25 minutes in all. The notebook
rebuilds the gold sheets and checks their hashes, downloads only the subset's videos, checks the frames against the
pilot's, runs F1–F3, prints every prompt and reply, times each call, and writes the analysis.

**Expanding:** set `STAGE` to `small` (5 per task, 45 replies), then `extended` (10, 90) and `full` (25, 225; needs
`CONFIRM_LARGER = True`). The subsets are nested and each folder resumes, so a larger stage answers only the new
questions. Estimate a stage from the smoke stage's `timing.jsonl` (seconds per new question); the pilot's D needed a
median 547 tokens on these questions at about 53 ms per decoding step on an L4.

**Resuming:** run the notebook again with the same settings; answers are saved to Drive batch by batch.

**Changing a prompt or adding a variant:** edit the pieces in `cot_variants.py` (`CONCISE`, `FORMAT`, `MATH`,
`EXAMPLES`), or add a name to `VARIANTS` and a branch in `instructions()`; bump `PROMPT_VERSION`; use a new
`RUN_NAME`. The fingerprint changes with the text, so old answers are never mixed with new ones (the run stops).

**Comparing with the baselines:** `analysis.py --results <RESULTS_DIR> --sheets-root <out_full>` (the notebook's last
cell); A, C, D and E are read from `results/zeroshot/vsibench/structured_cot/explore200/` for the same question ids.

### Held-out evaluation (after the prompts are frozen)

1. Freeze the variant and `PROMPT_VERSION`.
2. Draw held-out questions: `followup.heldout_pool` gives the computing questions of scenes no explore200 question
   comes from (2,066 of the 5,130); `nested_subset` on them with a new seed.
3. Run the frozen variant and, since the pilot did not run these questions, the baselines on the same ids (A, C, D,
   E: the eval's `direct` and `cot` with the gold sheet and `gold_relations`; the gold-sheet experiment's full run
   gives A, C and E for all 5,130 questions once it is done).
4. Compare as above.
5. Then the pipeline's sheets: the variants are defined for the gold sheet only (`cot_variants.check`); the pipeline's
   sheet (pipeline_v4, metres by an estimated scale) needs its own wording of the units, a decision for then.

## Model size: Qwen3.5-9B (exploratory)

Added 2026-11-25, before any 9B answer existed. **Exploratory:** 30 development questions (explore200, on which the
prompts were designed), one run per condition; it can show a large difference, not establish a small one.

### Question

Does CoT over the gold scene sheet fail because the 2B model is too small? The same questions and prompts with
Qwen/Qwen3.5-9B: does concise CoT (F1) beat direct answering with the sheet (C) for the 9B, and is its gain over the
2B's gain larger? Ji et al. (2507.13362, Table III) found a structured CoT helped 7B and larger models and barely a 3B.

### What would answer it

- **Size is a reason:** 9B F1 − 9B C > 0 and (9B F1 − 9B C) − (2B F1 − 2B C) > 0, with the replies showing the 9B
  applying a procedure the 2B does not (from the replies and `errors.csv`, not from the scores alone).
- **Size is not the reason, at this scale:** 9B F1 − 9B C ≤ 0 as well. Then the next suspects are the prompt and the
  sheet's form, not the model's size.
- **Read alongside:** 9B C − 2B C (does the 9B read the sheet better when answering directly?) and 9B E − 2B E (with the
  distances given; E is near its ceiling for the 2B, 82.0 on the pilot's 75 computing questions).
- With 10 questions per task, only differences of about 15 points or more can clear the intervals. A result here
  decides whether a larger 9B run (25 per task, then held-out questions) is worth it, not the answer.

### Design

| Model | Conditions run | From the pilot |
|---|---|---|
| Qwen3.5-9B | C, F1, E, A; F2 last and only if time allows | — |
| Qwen3.5-2B | F1 on the 30 (the smoke stage's 3 are reused) | A, C, D, E |

- **Questions:** `model_size_ids.txt`, 30: the first 10 of each computing task in `followup.nested_subset`'s seed-0
  order over explore200's computing questions (the `extended` stage), so they include the smoke stage's 3 and every
  one has the pilot's A, C, D and E (`test_followup.ModelSize.test_question_ids`). 14 scenes, up to 4 questions per
  scene.
- **The same for both models:** thinking off, 16 frames `uniform128`, the gold sheet as csv, greedy decoding, 2,048 new
  tokens, batches of up to 8 questions of a video, the final-answer rule, the prompts (A, C, E: the eval's direct
  prompts, as the pilot; F1, F2: `cot_variants`, version 2026-11-25.1), parsing and scoring. The 9B's A, C and E use
  the pilot's flags (`followup.DIRECT`); each folder's `config.json` differs from the 2B's only in the model and the
  device map (`test_followup.ModelSize`).
- **Loading:** bf16, `device_map="auto"` (`--device-map auto`), no quantization; `QwenVL` stops if accelerate would
  put any weight on the CPU or disk. The 2B keeps `device_map="cuda"`; the default of every command is unchanged.
- **Folders:** the 9B's runs in `<results>/qwen3.5-9b/<condition>/`; the 2B's in `<results>/<condition>/`, as before.
- **Deviations, stated:** the pilot's 2B A, C, D and E ran in batches of its 200 questions, the 9B's in batches of
  these 30 (fewer questions per video), and on another GPU; batching and the GPU can change greedy decoding in the
  last digits. The 2B's F1 answers come from two sessions (the smoke stage's 3, the other 27 on the A100).

### Comparisons (one Holm family)

| Comparison | Question |
|---|---|
| 9B F1 − 9B C | does CoT help the 9B? |
| 2B F1 − 2B C | does CoT help the 2B? |
| (9B F1 − 9B C) − (2B F1 − 2B C) | does model size change the effect of CoT? |
| 9B C − 2B C | the sheet, direct |
| 9B E − 2B E | the sheet and relations, direct |

Each is a per-question contrast of scores on the questions every condition of both models answered, on the computing
score (the mean of the three tasks) and per task: better / worse / same, an exact sign test, Holm over the five, and a
paired bootstrap 95% interval (10,000 resamples within question type, seed 0), as the rest of the analysis. Per model
and condition: the scores, the parse rate, the share cut off at 2,048 tokens, the median and 90th-percentile reply
length and seconds per question (`analysis.py --model-size`, `model_size.md`).

### Runs and cost (Colab, A100)

`cot_9b.ipynb`, with the code zip `mvp-3d-vlm-cot-9b.zip` (`git archive`; its comment is the commit) in `MyDrive/mvp/`,
on an A100 runtime, `RUN_NAME = "dev_v1"` (the 2B smoke stage's folder):

1. **`STAGE = "smoke"`:** one question per task, the four 9B conditions (12 replies), about 25–35 minutes: installing
   5, the 9B's download (about 19 GB) 5, a load on its own 1–2, four runs each with a load. Before the runs it checks
   the 2B's F3 smoke replies for copying from the worked example. It prints the peak GPU memory, seconds per question
   per condition and an estimate for the main stage. The decision to go on is made there.
2. **`STAGE = "main"`:** the 30 questions in the order 9B C, 9B F1, 9B E, 2B F1, 9B A (then 9B F2 with `RUN_F2 =
   True`); resumable, answers written to Drive batch by batch. Rough estimate before measuring: 50–70 minutes, F2 about
   15 more; a reply that runs to 2,048 tokens takes 1.5–2 minutes on the 9B.
3. The notebook zips `dev_v1` (both models) to `MyDrive/mvp/cot_9b_results_<STAGE>.zip`.

## Results

<!-- make_table:start -->
Not run yet.
<!-- make_table:end -->

## Conclusion

- **Answer to the question:**
- **Hypothesis:**
- **Caveats:**
- **Next:**

## Log

- 2026-11-25: plan, prompts (`PROMPT_VERSION` 2026-11-25.1), runs and analysis written; CPU tests pass; nothing run on a
  GPU yet. The conventions were checked against VSI-Bench's answers on the full benchmark (Design).
- 2026-11-25: the model-size comparison planned (Model size): Qwen3.5-9B on 30 development questions
  (`model_size_ids.txt`), the 2B's F1 on the same; `followup.py` runs the pilot's direct conditions and other models
  (`--model`, `--device-map`), `analysis.py --model-size`, `cot_9b.ipynb`; F3 replies flagged when they take the worked
  example's answer or numbers. Nothing run with the 9B yet.

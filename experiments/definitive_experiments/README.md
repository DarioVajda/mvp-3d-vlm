# Definitive experiments: overview and common methodology

Written 2026-11-27. This folder will hold the experiment files sent to the cluster to gather the data for the final
report: one file per experiment, each a clean version (Results, Conclusion and Log empty). This file comes first:

1. what each experiment does and what it seeks to show;
2. the methodology they share;
3. what each experiment changes;
4. the runs they share;
5. what must exist before any run, and the open questions (TODO).

Where an experiment's own README (`experiments/<id>/README.md`) disagrees with this file, this file wins. The README
stays the record of what was run before. Experiment 2 is the exception: it is done, and `2_gold_sheet_relations.md`
describes it as registered and run.

## 1. The experiments

| # | Experiment | Source | Varies | What it seeks to show |
|---|---|---|---|---|
| 1 | Scene-sheet formats | `2026-11-18-scene-sheet-formats`, as redesigned on 2026-11-25 | sheet format | which way of writing the gold sheet helps most over the video alone |
| 2 | Gold sheet and relations (done) | `2026-11-19-gold-sheet-relations` | sheet source, video | whether a correct sheet helps, whether precomputed relations help beyond it, and how much is left without the video |
| 3 | Pipeline sheet vs video | `scene_sheet_vs_video.md`, as redesigned on 2026-11-25 | sheet source, video | the same as 2 with the sheet we can build, and what the pipeline's errors cost against gold |
| 4 | Camera path | `2026-11-20-camera`, as redesigned on 2026-11-25 | cameras, format | whether the camera's path adds anything to the pipeline's objects, and in which form |
| 5 | Structured CoT | `2026-11-19-structured-cot` | reasoning, sheet source | whether step-by-step reasoning lets the model use a correct sheet, and whether it recovers what the relations give |
| 6 | Concise, mathematical CoT | `2026-11-25-concise-math-cot` | reasoning | whether a shorter CoT with explicit formulas, then with a worked example, does better on the computing tasks |
| 7 | Model size (new) | none | model | whether a larger model uses the gold sheet better, answering directly |

The 2026-11-25 redesigns are in the Claude Doc "Scene-sheet final experiments: cluster run prompts" (prompts 2–4). The
`varies` keys are TEMPLATE.md's, as follows:

- sheet source: `scene_sheet.source`
- sheet format: `scene_sheet.format`
- video: `visual.video`
- cameras: `scene_sheet.cameras`
- reasoning: `prompt.reasoning`
- model: `visual.model`

How each one works:

1. **Formats.** Video only, and the video with the gold sheet written as csv, a markdown table, json or text; the
   sheet's content is identical in every format. With correct content, a difference between conditions comes from how
   the sheet is written, not from the pipeline's errors.
2. **Gold sheet and relations (done).** Five conditions:
   - video only;
   - video + gold sheet;
   - video + gold sheet + the closest-point distances between the objects the question names;
   - the two sheets without the video.

   It separates two effects: what a correct sheet gives (almost only counting) and what stated distances give (the
   distance questions, where they state the answer). It also measures the token cost of frames only, sheet only and
   both.
3. **Pipeline vs video.** Experiment 2's five conditions, with pipeline_v4's sheet and relations in place of gold's.
   Each condition has a gold twin on the same questions, so pipeline − gold is the cost of the pipeline's errors.
4. **Camera path.** Video only, then the pipeline's objects with and without the camera's path. The path comes in three
   forms: a camera table, a compass path in text, and one narrative of the walk. Each camera form has an objects-only
   control in the same format.
5. **Structured CoT.** Direct and structured-CoT answers, both without a sheet and with the gold sheet. The gold sheet
   with relations, answered directly, is the upper reference.
6. **Concise CoT.** Three CoT variants, each adding one piece to the one before:
   - F1, concise instructions;
   - F2, F1 plus the box-gap and size formulas;
   - F3, F2 plus one worked example.

   They run on the three computing tasks (absolute distance, relative distance, object size) and are compared with
   experiment 5's conditions on the same questions.
7. **Model size.** Qwen3.5-2B, 4B and 9B, each with the video and the gold sheet, answering directly.

## 2. Common methodology

Every experiment uses these settings unless section 3 says otherwise. They also replace the defaults in
`experiments/TEMPLATE.md`, several of which are stale: relative units, `subset:uniform16`, `first_seen`, 16 new
tokens, the first-word answer.

### 2.1 Model and decoding

- **Model:** `Qwen/Qwen3.5-2B` (experiment 7 adds `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`). Thinking is off: `QwenVL`
  turns it off for every Qwen3.5 id.
- **Prompt:** the system prompt "You are a helpful assistant.", lmms-eval's template (`build_prompt` in
  `src/mvp/benchmarks/vsibench.py`), and the layout `sit` (video, then question).
- **Decoding:** greedy, up to 2,048 new tokens in every condition, direct and CoT alike.
- **Answer rule:** `--final-answer` (`score_final`): the last "Answer: X", else the last lone letter or the last
  number. lmms-eval's first-word score stays in each record as `score_first`.
- **Batches:** the eval's defaults (`--batch-size 8 --token-budget 120000`, up to 8 questions of one video), the same
  for every run of a model.
- **One environment:** every run of every experiment at one commit, in one environment, with the linear-attention
  kernels. Qwen3.5's answers differ with and without them (repository README: 89.6% identical). Each model's revision is
  the cached snapshot, recorded by the smoke test in
  `results/zeroshot/vsibench/definitive/model_revision_<model>.txt` (e.g. `model_revision_qwen3_5_2B.txt`).

### 2.2 Video

- **Frames:** 16, with `--sampling uniform128`. These are frames `np.linspace(0, 127, 16, dtype=int)` of the 128 VGGT
  frames, the same 16 the pipeline's sheets are built on. Qwen merges the frames in pairs and shows one timestamp per
  pair.
- **Sheet-only conditions** use `--no-video`: the same prompt without the video and without its sentence ("These are
  frames of a video."), so it starts with the sheet.

### 2.3 Questions

- **Questions:** all 5,130 questions of VSI-Bench, on all 288 scenes, all 10 question types (8 tasks).
  - `data/raw/VSI-Bench/test.jsonl` must have SHA-256 `aa3ff8cc2010e555f4f865ff96483812ec0b275005b68562f53bb114ad4db9db`.
  - Questions with an empty sheet stay in.
  - Experiment 6 keeps its three computing tasks: 2,497 questions, on all scenes.
- **Datasets**, the only split of the scenes that the analyses report:

  | Dataset | Scenes | Questions |
  |---|---:|---:|
  | ScanNet | 88 | 2,071 |
  | ScanNet++ | 50 | 1,458 |
  | ARKitScenes | 150 | 1,601 |

### 2.4 Gold sheets

- **The build:** `tasks/openvocab_adapter/out_full`, built by
  `experiments/2026-11-19-gold-sheet-relations/build_gold_sheets.py` from `prompts_full.jsonl`.
  - Hashes: index.csv `22e0ee1e…`, objects.csv `bc0214c9…` (that README's Runs, step 1); `check_sheets.py` must end
    with OK.
  - The boxes are VSI-Bench's labelled boxes, each turned to fit its object (since `980cf00`).
- **Units and fields:** metres, in the dataset's frame, z up. Fields: `id, label, x, y, z, size_x, size_y, size_z`. Gold
  sheets have no per-frame field, so appearance order cannot be read from them.
- **Axes:** z points up, as in the pipeline's sheets. The horizontal axes and the origin are the dataset's, not the
  first camera's. No VSI-Bench answer depends on them:
  - distances, sizes and areas don't change under a horizontal rotation or a shift;
  - relative direction is posed from objects, not from the frame.

  So gold sheets stay in their own frame.
- **Relations (`--sheet-source gold_relations`):** the closest-point distances between boxes (`scene_sheet.box_gap`).
  - A distance table for every question with at least two named classes in the sheet: 3,210 of the 5,130.
  - An answer line for absolute- and relative-distance questions: their distance, or each option's distance to the
    reference object.

### 2.5 Pipeline sheets: pipeline_v4

The pipeline is pipeline_v4 as registered (`experiments/2026-11-18-scene-sheet-formats/prereg_batch2_A.md`), with its
own detection, tracking and post-processing, at 16 frames. Only how its sheet is written changes:

- upright axes, in place of the first camera's;
- `visible` ranges, in place of `first_seen` and `num_frames`;
- a preamble that says how the metres were estimated.

| Stage | pipeline_v4 here | The registered run, where different |
|---|---|---|
| Frames | the VLM's 16 (`subset:uniform16`) | |
| Geometry | the saved 128-frame VGGT run's cameras and **dense depth** (`depth.npy`, `depth_conf.npy`, `frame_indices.npy` in `data/vggt/<dataset>/<scene>/`), cut to the 16 frames; never the patch-point fallback (`--patch-depth-fallback`): a scene without dense depth stops the build | |
| Prompts | each question's extracted prompts (`prompts_full.jsonl`) under the `fixed_v2` set: multi-word names kept together, plural names, synonyms (`rerun_detection.SYNONYMS_V2`). A name's prompts are pooled (`rerun_detection.pool_masks`), and of two detections from different prompts at IoU ≥ 0.5 the lower is dropped. "room" is kept as it is | |
| SAM 3 | `facebook/sam3` at `3c879f3`, transformers backend, detections kept from score 0.2, tracked from 0.4 | |
| Tracking | the registered tracker (`rerun_detection.RUNS["pipeline_v3"]`, through `track_many.py`; `milestone4_video.py` unchanged): centroid association (within 5% of the first frame's diagonal), the global depth filter (the sequence's 55th percentile of confidence, at least 300 points), re-identification on (`reid_sim` 0.6, `reid_radius_frac` 0.15), a track kept if seen in 1 of the 16 frames | |
| Rows | rows of tracks seen in one frame dropped below score 0.5; the point merge at 0.5 m (`rerun_detection.FROZEN`) | |
| Sizes | full-extent boxes by v4's sizing rule (`scene_sheet_creator.py` at `50fe543`), along the upright axes | along the first camera's axes |
| Dedup | for distance and direction questions, one row per named object (`dedup_sheets.py`'s rule) | |
| Axes | upright (`src/mvp/openvocab_adapter/upright.py`): z up from the fitted floor, the floor at z = 0, the origin on the floor below the first camera, y the way that camera faces. A scene without a floor plane falls back to the cameras' up; the build lists those scenes | the first camera's: x right, y down, z forward |
| Scale | metres. Per scene, the median of three estimates (`relations_v4.py`): the priors scale (`scene_scale.py`), the room-height scale (2.5 m) and the door-height scale (2.0 m), with heights measured along z. Every sheet and camera of the scene is multiplied by it. Nothing is fitted to gold | heights along the cameras' mean down axis |
| Relations (`pipeline_relations` only) | `relations_v4.py`'s, appended after the sheet, only when the sheet has a row for every named object. Absolute and relative distance: the closest-point distances between the trimmed (10th–90th percentile) boxes, and for relative distance the closest option. Relative direction: the direction the rows imply, on the floor plane | direction in camera 0's x–z plane |
| Fields | `id, label, x, y, z, size_x, size_y, size_z, visible`. `visible` gives the stretches of the 16 frames the object is seen in, in seconds, with bridged visibility: one missed frame between two sightings counts as seen | `first_seen`, `num_frames` |
| Preamble | in metres, said to be estimated from the scene (object sizes, the room's height and doors), with the upright sentence (`scene_sheet.UPRIGHT_AXES`) | "estimated from typical object sizes", the first camera's axes |
| Empty sheet | `scene_sheet.EMPTY` ("The detector did not find the objects named in the question. …") | |

- **One build, one sheets root:** all 288 scenes and 5,130 questions. Experiments 3 and 4 read the same files. The
  build reports:
  - index.csv's SHA-256;
  - the number of empty sheets;
  - the scenes without a floor plane;
  - each scene's three scale estimates and their median.
- **Registered files stay as they are.** The files hash-registered in `prereg_batch2_A.md` are not edited; the upright
  axes, the `visible` field and the new preamble go in new files. To run v4's registered code at HEAD, pass a `--code`
  tree made from `git archive b4135e7 src` plus `50fe543`'s `scene_sheet_creator.py`. `pipeline_v3.py`'s default tree
  crashes since the revamp (`72b9a2d`).

### 2.6 Prompt layout

- **Placement:** the sheet goes after "These are frames of a video." and before the question; relations follow the
  sheet.
- **Format:** csv, unless the experiment compares formats. Numbers have 3 decimals, with no negative zero. Ids are the
  label with spaces as `_`, numbered (`trash_bin_1`).
- **Direct answers:** lmms-eval's last line, "Answer with the option's letter from the given choices directly." or
  "Please answer the question using a single word or phrase.".
- **CoT:** the CoT instructions replace that line. They keep the same 2,048-token limit and the same answer rule, with
  thinking off: the reasoning comes from the prompt only (structured-cot Log, 2026-11-20). Which CoT prompt to use is
  TODO C1 (section 5).

### 2.7 Analysis and statistics

- **Primary metric:** VSI-Bench's overall, the mean of the 8 task scores on the experiment's questions. Multiple-choice
  tasks score accuracy and numeric tasks mean relative accuracy; relative direction's three types form one task.
- **Paired tests:** every comparison is paired question by question.
  - Count the questions helped and hurt, and run an exact two-sided sign test with ties left out
    (`mvp.reporting.compare_runs.sign_test`).
  - Apply Holm at 0.05 within each family that section 3 names (`compare_runs.holm`).
  - Give every difference a paired bootstrap 95% interval: 10,000 resamples of the questions within each question
    type, seed 0.
- **Smallest difference that matters:** 2 points overall, 5 points on a task.
  - A condition helps or hurts only if it differs by at least that much and its Holm p is below 0.05.
  - Two conditions closer than that count as equal; between equal sheets, the one with fewer tokens is preferred.
- **Always reported:**
  - scores per task and question type, and per dataset (ScanNet, ScanNet++, ARKitScenes);
  - with counting left out, wherever a gold sheet is in the prompt, since it contains the counting answer;
  - the debiased overall, on the questions in `test_debiased.parquet` (2,362);
  - the parse rate under the final-answer rule (`compare_runs.parse_rate`);
  - input tokens per condition (mean, median) from `n_prompt_tokens`, not `n_input_tokens`, which is the batch's
    padded length;
  - reply length (median, 90th percentile) and the share of replies that reach 2,048 tokens;
  - on multiple choice, the share of each answer letter and the accuracy where the true answer is not B. The camera
    pilot's sheets pushed answers to B.
- **CoT conditions also report:**
  - the share of cut-off replies that loop: 10 or more mm:ss times, or a last 2,000 characters that zlib compresses
    below a quarter of their size;
  - the score with cut-off replies counted wrong;
  - the reasoning check: the share of replies that name a sheet row (an id such as `chair_2`) or copy one of its
    numbers.
- **Earlier runs.** No earlier run is reused for experiments 1 and 3–7. Every earlier run differs from these settings
  in at least one way: relative units or first-camera axes, `first_seen` fields, 256 tokens, a subset of the questions,
  or another commit and environment. Earlier results are background only. Experiment 2 has its own runs (2B, `0f7c5f2`,
  `results/zeroshot/vsibench/gold_sheet_relations_full/`); experiment 3 reads them as its gold twins.

### 2.8 Running

- **Runs sections:** each experiment file's Runs section follows the gold-sheet-relations README's:
  - the inputs, with their hashes and counts;
  - the tests;
  - a render check of the experiment's conditions for all its questions;
  - the prompt check (`--print-prompt` on `prompt_check_ids.txt`, then `cluster_checks.py prompts`);
  - a 20-question smoke test (`smoke_ids.txt`, 2 per question type): every question answered and at least 90% of
    answers parsed. It also records the model's revision.
  - then one command per condition, each resuming from its `--out` file and reporting how many questions it answered.
- **On one GPU, with no time or GPU-type estimates.** The final cluster is not the one used so far, so the person
  running reports measured times.
- **Data on the final cluster:** it needs its own copies of VSI-Bench's videos and `test.jsonl`, every scene's VGGT
  outputs with their dense depth, the gold and pipeline sheet roots (checked by hash), and the model snapshots.
- **Outputs:** every run goes to `results/zeroshot/vsibench/definitive/`, named by its run (section 4), so that a
  shared run exists once and every experiment's analysis reads it there. Commit only that folder, with the commit the
  runs were made at.
- **Stops:** stop and report on any failed hash, count, test or check. Don't edit code, settings or the experiment file
  during the runs.
- **Experiment files:** each keeps TEMPLATE.md's sections but holds only what is its own. Its Configuration lists the
  keys that differ from this section; everything shared is cited here, not repeated. Results, Examples, Conclusion and
  Log stay empty. Condition letters are each experiment's own; section 4 maps them to runs.

## 3. What each experiment modifies

Each subsection lists the conditions (with their runs from section 4), the comparisons and their Holm families, and the
hypothesis.

### 3.1 Scene-sheet formats

| ID | Condition | Sheet | Format | Run |
|---|---|---|---|---|
| A | Video only | none | – | R1 |
| B | Video + gold sheet | gold | csv | R3 |
| C | Video + gold sheet | gold | markdown_table | R6 |
| D | Video + gold sheet | gold | json | R7 |
| E | Video + gold sheet | gold | text | R8 |

- **Question:** which way of writing the gold sheet helps Qwen3.5-2B most over the video alone?
- **Comparisons:** B − A, C − A, D − A and E − A form one family (primary). The six pairs of formats form a second.
- **Expected** (the README's hypothesis for gold sheets, with what its 2026-11-19 gold runs found): every format beats
  A overall, almost all of it on counting. The differences between formats are smaller than the gain from adding the
  sheet.
- **Would prove it wrong:** the spread between the best and the worst format is at least the best format's gain over
  A. That would mean the format matters as much as the sheet itself.

### 3.2 Gold sheet and relations (done)

Registered in `experiments/2026-11-19-gold-sheet-relations/README.md` and run with Qwen3.5-2B at `0f7c5f2`, on all
5,130 questions. `2_gold_sheet_relations.md` holds its design, results and conclusion, and wins over this file where
they differ. Its five conditions (A video only, B + gold sheet, C + gold sheet + relations, D and E the two sheets
without the video) are the gold twins of experiment 3.

### 3.3 Pipeline sheet vs video

| ID | Condition | Sheet | Video | Run | Gold twin (experiment 2) |
|---|---|---|---|---|---|
| A | Video only | none | yes | R1 | A |
| B | Video + pipeline sheet | pipeline | yes | R9 | B |
| C | Video + pipeline sheet + relations | pipeline_relations | yes | R11 | C |
| D | Pipeline sheet only | pipeline | no | R12 | D |
| E | Pipeline sheet + relations only | pipeline_relations | no | R13 | E |

- **Question:** does pipeline_v4's sheet, and its relations, help Qwen3.5-2B over the video alone? How much is left
  without the video, and how far is each condition from its gold twin?
- **Comparisons:**
  - First family: B − A and C − B.
  - Second family: B − D, E − C and D − A.
  - Third family: pipeline − gold for each of B, C, D and E, against experiment 2's runs, broken down by question type.
- **The full method:** pipeline_v4's routing takes C's answer for absolute distance, relative distance, relative
  direction and route planning, and A's for the rest. It is computed from A's and C's answers and compared with A, with
  its interval, as a secondary result. It needs no extra run.
- **Expected** (the newest pipeline hypotheses, `prereg_batch2_A.md`'s H1–H3, carried over):
  - with relations (C), the distance questions score higher than with the video alone (H1), and so does relative
    direction (H2);
  - the full method raises the overall (H3).
- **Would prove it wrong:** C − A ≤ 0 on the distance questions. That is what registered pipeline_v4 gave on batch2_A,
  with first-camera axes: 53 questions helped against 89 hurt, p = 0.0032.
- **The gold twins' runs** come from another commit and environment (`0f7c5f2`); their prompts are the same as at later
  commits (experiment 2's Conclusion). Stated as a caveat on pipeline − gold.

### 3.4 Camera path

| ID | Condition | Objects | Cameras | Run |
|---|---|---|---|---|
| A | Video only | none | none | R1 |
| B | `obj_csv` | csv | none | R9 |
| C | `cam_csv` | csv | csv table | R14 |
| D | `obj_text` | sentences | none | R10 |
| E | `cam_text` | sentences | compass path, a separate block | R15 |
| F | `narrative` | in the walk | in the walk | R16 |

- **Question:** does the camera's path add to the pipeline's objects for Qwen3.5-2B, as a table, as a compass path in
  text, or merged with the objects into one narrative?
- **Comparisons:** C − B, E − D and F − D form the primary family. F − E and every condition − A form a secondary one.
- **Settings:**
  - the cameras in the objects' format (the code's rule since `080acf4`);
  - camera positions and move lengths in metres, on the sheet's own scale;
  - upright axes ("up is perpendicular to the floor");
  - timestamps in seconds, bridged visibility, move-then-turn steps, object directions along the floor;
  - video in every condition, direct answers, no relations. Relations are left out because pipeline_v4's direction
    relation would state the answer to the questions the cameras could help with.
- **Expected** (camera run 2, the newest): no camera form beats its objects-only control on direction and route by 5
  points, and the cameras push multiple-choice answers to B.
- **Would prove it wrong:** a camera form beats its control on direction and route (relative direction and route
  planning together) by at least 5 points, with Holm p < 0.05. The gain must hold on the questions whose answer is not
  B.

### 3.5 Structured CoT

| ID | Condition | Reasoning | Sheet | Run |
|---|---|---|---|---|
| A | Video only, direct | direct | none | R1 |
| B | Video only, CoT | CoT | none | R2 |
| C | Video + gold sheet, direct | direct | gold | R3 |
| D | Video + gold sheet, CoT | CoT | gold | R4 |
| E | Video + gold sheet + relations, direct | direct | gold_relations | R5 |

- **Question:** does step-by-step reasoning help Qwen3.5-2B use the gold sheet, and does it recover what precomputed
  relations give?
- **Comparisons:**
  - One family: D − C (the primary comparison), B − A and E − D.
  - Descriptive: E − C, and (D − C) − (B − A).
  - Each comparison is also read without counting, and on the computing tasks alone.
- **Reasoning check:** in D, the share of replies that name a sheet row or copy one of its numbers. In addition, 20 wrong
  D answers from the computing tasks, drawn with seed 0, are labelled by hand: wrong row, coordinates misread,
  arithmetic wrong, sheet ignored, or no answer line.
- **Expected** (the README's 2026-11-19 hypothesis, unchanged since):
  - D − C > 0;
  - C < D < E: CoT recovers part of the gap to the relations, not all;
  - B − A ≈ 0 or below.

  The 200-question pilot gave D − C −3.3, B − A −13.3 and E − D +13.9.
- **Would prove it wrong:**
  - D − C ≤ 0, overall and on the computing tasks;
  - D ≥ E;
  - B − A clearly positive.

### 3.6 Concise, mathematical CoT

| ID | Condition | Reasoning | Run |
|---|---|---|---|
| A | Video only, direct | direct | R1, computing questions |
| C | Video + gold sheet, direct | direct | R3, computing questions |
| D | Video + gold sheet, structured CoT | CoT | R4, computing questions |
| E | Video + gold sheet + relations, direct | direct | R5, computing questions |
| F1 | Video + gold sheet, concise CoT | `cot_f1` | R17 |
| F2 | F1 + geometric formulas | `cot_f2` | R18 |
| F3 | F2 + one worked example | `cot_f3` | R19 |

- **Question:** can a more constrained, mathematically explicit CoT, with or without one worked example per task, make
  Qwen3.5-2B answer the computing tasks better from the gold sheet than structured CoT (D), than direct answering (C)
  and than the video alone (A)? And how far does it stay from the given relations (E)?
- **Comparisons:**
  - Primary family: F1 − D, F2 − F1 and F3 − F2.
  - Secondary families, one each: every F against C; every F against E; every F against A.
  - All on the computing score (the mean of the three tasks' scores) and per task. Each reply also gets the
    README's automatic error flags in `errors.csv`.
- **Expected** (the README's 2026-11-25 hypothesis):
  - F1 − D ≥ 0, with shorter replies and fewer cut off;
  - F2 − F1 > 0, mostly on absolute distance;
  - F3 − F2 ≥ 0, small;
  - E − F2 > 0.

  On all scenes, relative distance can now gain too. The README expected nothing there because on its development
  questions the centre and closest-point rules never disagree, but on the full benchmark they score 83.7 against
  98.3.
- **Would prove it wrong:** F2 ≤ F1 on absolute distance, F1 below D, or more answers that can't be read.

### 3.7 Model size (new)

| ID | Model | Reasoning | Run |
|---|---|---|---|
| S2D | Qwen3.5-2B | direct | R3 |
| S4D | Qwen3.5-4B | direct | R20 |
| S9D | Qwen3.5-9B | direct | R22 |

All three have the video and the gold sheet (csv), answered directly, on all 5,130 questions.

- **Question:** does a larger Qwen3.5 use the gold sheet better, answering directly?
- **Comparisons:**
  - One family, between sizes: S4D − S2D, S9D − S4D and S9D − S2D.
  - Everything also without counting and on the computing tasks.
- **Expected:** direct answering with the sheet improves with size. The basis is concise-math-cot's model-size section.
- **Would prove it wrong:** S9D − S2D ≤ 0, so a larger model does not use the gold sheet better, at least at this
  scale.
- **Loading the 9B:** `--device-map auto`. `QwenVL` stops if any weight would go to the CPU or disk.
- TODO C5.

## 4. Shared runs

A run with the same flags serves every condition it matches, so it is made once. All on all 5,130 questions with
Qwen3.5-2B unless the row says otherwise. Every row also has the common flags: `--nframes 16 --sampling uniform128
--max-new-tokens 2048 --final-answer`. Experiment 2 is not here: it has its own runs (2.7).

| Run | Sheet source and format | Video | Reasoning | Used by |
|---|---|---|---|---|
| R1 | none | yes | direct | 1A, 3A, 4A, 5A, 6A |
| R2 | none | yes | CoT | 5B |
| R3 | gold, csv | yes | direct | 1B, 5C, 6C, S2D |
| R4 | gold, csv | yes | CoT | 5D, 6D (TODO C1) |
| R5 | gold_relations, csv | yes | direct | 5E, 6E |
| R6 | gold, markdown_table | yes | direct | 1C |
| R7 | gold, json | yes | direct | 1D |
| R8 | gold, text | yes | direct | 1E |
| R9 | pipeline, csv | yes | direct | 3B, 4B |
| R10 | pipeline, text | yes | direct | 4D |
| R11 | pipeline_relations, csv | yes | direct | 3C |
| R12 | pipeline, csv | no | direct | 3D |
| R13 | pipeline_relations, csv | no | direct | 3E |
| R14 | pipeline, csv, `--cameras` | yes | direct | 4C |
| R15 | pipeline, text, `--cameras` | yes | direct | 4E |
| R16 | pipeline, narrative | yes | direct | 4F |
| R17–R19 | gold, csv; the 2,497 computing questions | yes | `cot_f1`, `cot_f2`, `cot_f3` | 6F1–F3 |
| R20 | gold, csv; Qwen3.5-4B | yes | direct | S4D |
| R22 | gold, csv; Qwen3.5-9B | yes | direct | S9D |

That is 21 runs. Each writes `results/zeroshot/vsibench/definitive/<model>_<run>.jsonl`: `qwen3_5_2B_R1.jsonl` to
`qwen3_5_2B_R19.jsonl`, then `qwen3_5_4B_R20.jsonl` and `qwen3_5_9B_R22.jsonl`. A run another experiment has already
made resumes from its file and answers nothing new.
Experiments 3 and 4 also wait on the pipeline_v4 build; experiments 1, 5, 6 and 7 do not.

## 5. Needed before the runs, and open questions

### Needed (work to do; the methodology is settled)

1. **Exports for all 288 scenes:** frames, cameras, `meta.json` with `video_fps`, and adapter reports at
   `subset:uniform16`. Only 150 of the 288 scenes are exported. VGGT has run on all 288, but its outputs,
   dense depth included, must be copied to the final cluster or made again there.
2. **A build script for pipeline_v4** on any scene list:
   - SAM 3 and the registered tracking (`rerun_detection.py`, `track_many.py`) on one GPU, then the post-processing on
     CPU (`pipeline_v3.py`, `dedup_sheets.py`, `trimmed_sheets.py`, `relations_v4.py`), with the sheets written in
     upright axes and with `visible` ranges (2.5);
   - it writes one sheets root, with an index.csv of the 5,130 questions, the relations file and the scales file;
   - finished scenes are skipped when it is run again;
   - the registered v4 files are not edited (2.5).
3. **Eval support** for pipeline_v4's sheets, with tests:
   - a metres preamble without rescaling (a new `--metric-scale` value; `priors` would rescale sheets that are already
     in metres);
   - `--sheet-source pipeline_relations` with `--no-video` (as `NoVideo` tests `gold_relations`);
   - cameras, the compass path and the narrative in metres on the sheet's scale;
   - every format. Today pipeline_v4's sheets render through `explore_sheets.py`, in csv only.
4. **A render check** that writes every condition of section 4 for all its questions, as `check_sheets.py` does for
   gold, so that a rendering failure shows before any GPU time is spent.
5. **Analysis scripts** for each experiment that read the shared runs and write its Results table.

### TODO (experiments 5–7 only)

- **C1, the CoT prompt (experiments 5 and 6's D):**
  - Option 1: `--reasoning cot` as it stands (`experiments/2026-11-19-structured-cot/cot_prompt.md`, `6642c48`). Its
    step "check that the result agrees with what the video shows" made 12% of the pilot's replies with the gold sheet
    loop to the token limit, and 28% of those without a sheet.
  - Option 2: a revised version for all question types, for example without that step, as F1 does.
  - Open since 2026-11-20 (`explore_cot.md`, Conclusion). If it changes, experiment 6's D changes with it, and F1 − D
    no longer measures F1's pieces against the pilot's prompt.
  - **Evidence so far** (exploratory, on explore200's development questions;
    `experiments/2026-11-25-concise-math-cot/README.md`, Results): on the 30 computing questions of
    `model_size_ids.txt` (2B), D scores 66.7 with 3.3% of replies cut off, and F1, which drops the check against the
    video, 64.7 with none cut off and replies a quarter of D's length (F1 − D −2.0, interval −15.7 to +11.7). F1 is
    defined only for the three computing tasks, so option 2 needs new text for the other types.
- **C2, freezing F1–F3 (experiment 6):** should `cot_variants.PROMPT_VERSION` `2026-11-25.1` be frozen as it is, or
  revised first on explore200?
  - **Evidence so far** (exploratory, on explore200's development questions;
    `experiments/2026-11-25-concise-math-cot/README.md`, Results): the 2B's smoke stage (3 questions, `64e566e`) and
    F1 and F2 on the 30 questions (`e2cd67a`) ran; the predictions are in
    `results/zeroshot/vsibench/concise_math_cot/dev_v1/`.
    - F3 copied its worked example's numbers in 2 of 3 smoke replies (1.059 on question 838, 183 on question 2227)
      and was not run further, which bears on freezing F3 as written.
    - F2's formula raised absolute distance (2B 45.0 → 61.0, 9B 49.0 → 98.0) but lowered relative distance (2B
      70.0 → 30.0, 9B 80.0 → 40.0); 2B F2 − F1 −7.7 on the computing score.
    - No F beat the given relations: E 85.3 against at most 64.7 on the 2B, 98.7 against at most 79.3 on the 9B.
- **C3, batches (experiment 6):** A, C, D and E come from the shared runs on all 5,130 questions, so their batches mix
  question types. F1–F3 run on the computing questions only. Greedy decoding in a batch can differ in the last digits.
  Accept this as a stated deviation, or rerun A, C, D and E on the computing questions alone?
  - **So far:** the exploratory runs had the same deviation, stated there: the baselines in the pilot's batches of up
    to 8 questions of a video, F1 and F2 in the subset's smaller batches.
- **C5, the 9B's batch size (experiment 7):** the same batches as the 2B and 4B (8) if memory allows? Otherwise one
  smaller batch size for all three models, or for the 9B alone, as a stated deviation. The exploratory 9B plan answered
  one question at a time.
  - **Measured so far** (exploratory; `7_model_size.md`, Runs and cost): at `--batch-size 1` and `--device-map auto`
    on one L4 (23.7 GB), the 9B loaded entirely on the GPU (18.8 GB of weights) and peaked at 19.5 GB. 8 questions at
    a time has not been tried.

---
id: 2026-11-20-camera    # also the folder name: experiments/<id>/
title: Which way of giving Qwen3.5-2B the camera's path (camera tables, camera text, the narrative) is worth developing, in metres, with and without chain of thought
owner: Nicolás Ramos Fernandez
created: 2026-11-20
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, cameras, narrative, format, chain-of-thought, pilot]
varies: [scene_sheet.cameras, scene_sheet.format, scene_sheet.time_unit, scene_sheet.step_order,
         scene_sheet.object_directions, scene_sheet.visibility, prompt.reasoning]
depends_on: [2026-11-18-scene-sheet-formats (pipeline sheets, explore200's questions and video-only predictions, the metre scale),
             2026-11-19-structured-cot (the CoT prompt and its pilot)]
---

# Camera sheets and the narrative: a pilot

**Outcome:** no way of giving Qwen3.5-2B the camera's path is worth developing.
- **Run 2 (metres):** against the objects-only controls, the cameras change the overall by −1.6 to +2.8 points, every
  interval including 0. Chain of thought lowers every camera form by 10.5–11.1 points.
- **Run 1 (relative units, direct answers only):** −2.5 to +1.1.
- In both runs, the camera tables mostly shift multiple-choice answers to B.

**This is an exploratory pilot, not a result for the report.** It runs every way the code has of giving the model the
camera's path on 200 questions, to decide which one deserves a full experiment. Where this README departs from
`experiments/TEMPLATE.md`, that is on purpose:
- There is no single hypothesis that a test must reject. There are expectations, and each says what we do next.
- There is no sign test and no Holm correction. Every comparison gets a bootstrap 95% interval, and with about 40 of
  them, a few will exclude 0 by chance. A difference here is a lead to confirm, not a finding.
- The settings stay close to the earlier experiments' (model, frames, prompt, answer rule, questions; the CoT
  conditions follow the structured-CoT pilot). There is no gold sheet.

**Two runs.**
- **Run 1** (`explore_camera.ipynb`, done 2026-11-20) wrote every sheet in VGGT's relative units. It was meant to be in
  metres. The plan kept relative units because the code's metre conversion, `--metric-scale priors`, covered only
  tables of objects, not the cameras or the narrative (Log).
- **Run 2** (`explore_camera_metres.ipynb`, done 2026-11-21, the experiment as intended) writes every sheet in metres: the objects, the
  camera tables, the compass path and the narrative, with one scale per question. `--metric-scale priors` now covers
  all of them (Log, 2026-11-21). It reruns run 1's 15 direct conditions this way, and adds three chain-of-thought
  conditions. Run 1's results are kept below (Results, run 1) and are compared with run 2's condition by condition.

## Question

Does giving Qwen3.5-2B the camera's path, along with the pipeline's objects, help it on VSI-Bench? If so, in which
form:
- the camera poses as a table (csv or markdown table);
- the path as compass directions in text;
- one narrative of the walk that mixes the camera's moves with the objects it sees.

And which of the text forms' options matter (step order, how object directions are said, timestamps or frame numbers,
exact or bridged visibility)? All of it with the sheets in metres. Does reasoning step by step (chain of thought) let
the model use the camera's path when answering at once does not?

## What we expect, and where each result points

- **Metres against relative units (run 2's direct conditions against run 1's):** the metric tasks should recover.
  - Run 1's sheets lowered them because the model reads the sheet's numbers as metres. Its median absolute-distance
    answer was 1.0, against 1.9 m true.
  - In metres, the numbers are about right. The scale is within 30% of the true one in 82% of scenes, with a median
    error of 16% (`metric_scale.md` in the format experiment's folder).
  - Expected: absolute distance, object size and room size rise clearly over run 1, though not above video only. With
    correct gold metres, the format experiment's sheets still did not beat video on absolute distance.
  - The other tasks should barely change.
  - *If the metric tasks don't recover,* the model does not use the sheet's sizes and distances even when they are in
    the right unit; metres alone are not enough.
- **Camera tables, camera text and the narrative (direct):** as in run 1, little or no gain on direction and route.
  Metres don't change the directions or turns the camera forms give.
  - *If one of them beats its objects-only control on direction and route by 5 points or more* (interval excluding
    0), and not through the shift to answer B (Metrics), that form is worth a full run.
  - *If none does,* run 1's conclusion stands in metres too.
- **Chain of thought (`cot_<condition>` against `<condition>`):** the structured-CoT pilot gives the expectation.
  - On these 200 questions, CoT did not help the 2B with a gold sheet overall (−3.3, interval −10.5 to +4.0), and
    hurt without a sheet (−13.3).
  - It did make the model compute from the sheet, with correct arithmetic.
  - 12% of its replies with the gold sheet looped until the token limit.
  - Expected: no gain overall, possibly a gain on the tasks where the sheet gives the numbers to work with, and more
    looping with the narrative and the compass path, which are told frame by frame.
  - *If a CoT condition beats its direct one on direction and route by 5 points or more* (interval excluding 0, and
    not through letter bias), reasoning over the camera's path is worth pursuing, with a model that reasons (thinking
    on, or a larger one).
  - *If none does,* the camera's path is not what this model needs, answering at once or reasoning. Work goes to
    precomputed relations, including direction relations ("standing at the sofa facing the table, the tv is
    front-left").
- **Narrative against camera text:** whether one walk, from the camera's point of view, is better than the objects and
  the path kept apart. No expectation, directly or with CoT.
- **The options (step order, object directions, frame numbers, exact visibility):** small rewordings. We expect each
  to stay within a few points of its default, as in run 1.

## Background

- **Run 1** (this folder, Results, run 1; `analysis.md`): run 1's sheets were in relative units, and with them
  - every camera comparison's interval included 0;
  - the camera, text and narrative conditions shifted multiple-choice answers to B (39–76% of answers, against 31%
    with video only and 33% of true answers);
  - on the direction and route questions whose answer is not B they got 13–26% right, against 32% with video only
    and 35% with the objects-only tables;
  - the metric tasks fell 7–17 points overall below video, because the model read the units as metres.

  Every reply was a bare letter or number. The 14 runs took about 70 minutes on a Colab L4.
- **Format experiment** (`experiments/2026-11-18-scene-sheet-formats/README.md`): Qwen3.5-2B on 2,728 questions,
  direct answers, pipeline sheets in relative units. Every format lowered the overall (video only 43.2, csv 34.9); the
  loss was on the metric tasks. With gold sheets in metres, absolute distance still fell (37.2 → 26.4–32.7): the 2B did
  not compute distances from correct coordinates.
- **The metre scale** (`src/mvp/benchmarks/scene_scale.py`; `metric_scale.md`):
  - It is gold-free: typical object sizes times one calibration constant fitted on 24 scenes outside explore200's.
  - An object-size question leaves its own class out, so its answer does not come from the prior.
  - Over 34 scenes it puts the scale within 30% of the true one in 82% (median error 16%), and centre distances are
    20% off, against 57% with no conversion.
  - On the 200 questions here, the scale ranges from 1.26 to 6.01 metres per sheet unit (median 2.49). No question
    gets the fallback.
- **The structured-CoT pilot** (`experiments/2026-11-19-structured-cot/explore_cot.md`, `analysis.md`): Qwen3.5-2B on
  these 200 questions, `--reasoning cot`, up to 2,048 new tokens.
  - With the gold sheet, CoT was −3.3 overall against direct answers, +9.5 on the computing tasks (all from object
    size).
  - Without a sheet it was −13.3.
  - 12% (with the sheet) and 28% (without) of replies were cut off at the limit. Nearly all of those looped: they
    narrated the video frame by frame or repeated themselves.
  - The pilot recommended keeping the 2,048-token limit: the loops, not the limit, cut the replies off.
  - Two CoT runs took about 1.7 hours together on a Colab L4.
- **The CoT prompt** (`cot_prompt` in `src/mvp/benchmarks/vsibench.py`; its text in the structured-CoT folder's
  `cot_prompt.md`): say which objects or places the question is about and find them in the video and the scene
  information; write down the facts needed; work the answer out one step at a time; check it against the video; end on
  "Answer: X". Each task adds one hint. For a pipeline sheet in metres it adds: "The scene information was computed
  automatically from the video and can be wrong: an object can be missed, counted twice or misplaced. Where it
  disagrees with what you see, weigh the two." It does not ask the model to work out a scale, as it would for relative
  units.
- **What the camera forms are in metres**, for question 974 (2.820 m per unit; `python -m mvp.benchmarks.scene_sheet
  --question 974 --format narrative --metric-scale priors --prompt` prints one):
  - **Camera table:** "x, y, z is the camera's position, approximately in metres, estimated from typical object sizes",
    then one row per frame, `time,x,y,z,qw,qx,qy,qz,forward_x,forward_y,forward_z`, with the positions in metres.
  - **Camera text:** "At 13.6 seconds: Go 0.930 metres north and turn 37° to the right and 39° up. …".
  - **Narrative:** "At 13.6 seconds: Move 0.930 metres forward and turn 37° to the right and 39° up; you see tv_1 (a tv,
    0.040 × 0.523 × 0.955 metres) ahead, slightly to your right, 1.775 metres away; sofa_1 is no longer in view. …".
- **The questions:** `explore200_ids.txt` (in the format experiment's folder). There are 200 questions, 25 per task
  (relative direction 8 easy, 8 medium, 9 hard), from 25 scenes. 8 of the 200 have an empty pipeline sheet. They were
  not chosen for anything in this experiment.
- **Video only is already on Drive** (the structured-CoT pilot's `A_video`, which run 1 copied), and stands in for
  `video` if it passes the reuse check (Runs).

## Design

### Conditions (run 2)

Every sheet is the pipeline's (VGGT + SAM 3 + tracking), in metres (`--metric-scale priors`). The cameras are always
written in the objects' format (the code's rule since `080acf4`), so each camera table has its objects-only control in
the same format. The defaults are timestamps in seconds, bridged visibility, move then turn, and object directions
along the floor. A blank cell is the default.

| Name | Family | Objects | Cameras | Other options | Role |
|---|---|---|---|---|---|
| `video` | – | none | none | | baseline (reused if it checks out) |
| `obj_csv` | objects only | csv | – | | control |
| `obj_markdown_table` | objects only | markdown table | – | | control |
| `obj_text` | objects only | sentences | – | | control |
| `cam_csv` | camera table | csv | csv | | treatment |
| `cam_markdown_table` | camera table | markdown table | markdown table | | treatment |
| `cam_csv_frames` | camera table | csv | csv | frame numbers 1–16 | variant of `cam_csv` |
| `cam_text` | camera text | sentences | compass path | | treatment |
| `cam_text_turn_move` | camera text | sentences | compass path | turn, then move | variant of `cam_text` |
| `cam_text_frames` | camera text | sentences | compass path | frame numbers 1–16 | variant of `cam_text` |
| `narrative` | narrative | in the walk | in the walk | | treatment |
| `narrative_turn_move` | narrative | in the walk | in the walk | turn, then move | variant of `narrative` |
| `narrative_camera_dirs` | narrative | in the walk | in the walk | object directions in the camera's tilted axes | variant of `narrative` |
| `narrative_frames` | narrative | in the walk | in the walk | frame numbers 1–16 | variant of `narrative` |
| `narrative_exact` | narrative | in the walk | in the walk | exact visibility (no bridged frame) | variant of `narrative` |
| `cot_cam_csv` or `cot_cam_markdown_table` | camera table, CoT | csv or markdown table | the same | structured CoT | treatment (CoT) |
| `cot_cam_text` | camera text, CoT | sentences | compass path | structured CoT | treatment (CoT) |
| `cot_narrative` | narrative, CoT | in the walk | in the walk | structured CoT | treatment (CoT) |

That is 14 direct conditions to run, plus `video`, and 3 CoT ones. The CoT table format is the one, of csv and
markdown table, whose direct `cam_<format>` has the higher overall in run 2 (Decisions). Left out:
- json and markdown key-value (Decisions);
- frame numbers with the markdown table;
- exact visibility outside the narrative (it changes only the objects' `visible` field);
- the combinations of the variants with each other;
- CoT with the objects-only sheets or the variants.

### Comparisons

Each comparison is the first condition minus the second, with a bootstrap 95% interval:

1. **Do cameras help, and in which format?** `cam_<f>` − `obj_<f>` for csv, markdown table and text (camera
   text).
2. **Narrative:** `narrative` − `cam_text` (one walk or the two parts apart), `narrative` − `obj_text` (the walk or
   the objects alone, both text) and `narrative` − `obj_csv` (against the format experiment's best objects-only
   format).
3. **The options:**
   - `cam_text_turn_move` − `cam_text`
   - `cam_text_frames` − `cam_text`
   - `cam_csv_frames` − `cam_csv`
   - `narrative_turn_move` − `narrative`
   - `narrative_camera_dirs` − `narrative`
   - `narrative_frames` − `narrative`
   - `narrative_exact` − `narrative`
4. **Against video only:** every condition − `video`.
5. **Metres against relative units:** each direct sheet condition in run 2 − the same condition in run 1, on the same
   questions. The metric tasks are the row that matters.
6. **Chain of thought**, on the CoT questions:
   - each CoT condition − its direct one (`cot_cam_<f>` − `cam_<f>`, `cot_cam_text` − `cam_text`, `cot_narrative` −
     `narrative`);
   - `cot_cam_text` − `cot_cam_<f>`, `cot_narrative` − `cot_cam_text`, `cot_narrative` − `cot_cam_<f>`;
   - each CoT condition − `video`.

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | explore200's 200 (`explore200_ids.txt`) for the direct conditions. The CoT ones use all 200, or k per question type if the timing says the whole run would take more than 4 hours | The same questions as run 1 and the explore and CoT pilots, so `video` can be reused and run 1 compared question by question; 25 per task |
| Units | metres, `--metric-scale priors`, in every sheet: objects, camera tables, compass path, narrative; one scale per question | The sheets as intended. The scale is the repository's gold-free one (`scene_scale.py`), the only one a real system could use; the oracle scale fitted to gold is an upper bound, not a method |
| Sheet source | pipeline only | The aim is the sheet we can build; gold sheets have no cameras (`check_options`) |
| Reasoning | direct (lmms-eval's prompt) for run 1's 15 conditions; structured CoT (`--reasoning cot`, the eval's `cot_prompt`, unchanged) for three | Direct as the format experiment and run 1. CoT as the structured-CoT pilot, so its results compare with the pilot's. The pilot's open question about rewording the prompt (its loops) is left for later; a second prompt here would make the comparison with the pilot harder |
| CoT conditions | the better table format, the camera text, the narrative | The question's three camera forms, each with CoT. The table format is chosen by the direct runs so that CoT goes to the stronger one |
| CoT table format | the one whose direct `cam_<format>` has the higher overall (VSI-Bench's 8-task mean) on the 200 questions, csv on a tie | Fixed before the run. The overall rather than direction and route, which run 1 showed is moved by the shift to B. csv on a tie, being the shorter. Kept in `cot_table.json` |
| Token limit | 256 direct, 2,048 CoT | Direct replies are 1–10 tokens; 256 bounds any that starts to reason. 2,048 as the structured-CoT pilot, whose cut-off replies were loops that a higher limit would not end |
| Answer rule | `score_final` (`--final-answer`) in every condition | One rule for all, as the earlier pilots; parse rate with `parsed(..., final=True)` |
| Batches | up to 8 questions of one video (`--batch-size 8`, `--token-budget 120000`, the eval's defaults) | As run 1, explore200 and the CoT pilot, which the reuse check needs |
| Formats | csv, markdown table and text, plus the narrative; json and markdown key-value dropped | They were the worst object formats in the format experiment (31.3 and 25.5, against 34.9 and 35.1 for the tables), and their camera sheets are the longest (2,893 and 2,687 tokens on average), so they would cost the most and are the least likely to be used |
| Camera format | the objects' format | The code's rule (`080acf4`); one control per format separates the camera effect from the format effect |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt  # CoT: or k per question type (Runs)
  scenes: 25 of the 75 exported scenes (the questions')
  question_types: all          # all 10
  n_questions: 200             # the same questions in every direct condition; the CoT ones on the CoT questions

visual:
  video: true
  model: Qwen/Qwen3.5-2B       # thinking off (the eval turns it off for Qwen3.5)
  protocol: video
  nframes: 16
  sampling: uniform128         # the 16 frames the sheets were built on
  layout: sit
  resolution: default

scene_sheet:                   # video has none
  source: pipeline
  geometry: saved VGGT         # data/vggt, 128 frames
  frames: subset:uniform16
  sam3_prompts: full
  sam3: {backend: transformers, model: facebook/sam3@3c879f3, min_score: 0.5}
  tracking: {reid: true, conf_percentile: 55, min_points: 300}
  content: [position, size, visible]   # + cameras (cam_*), or the walk (narrative_*)
  cameras: varies              # off (obj_*) | in the objects' format (cam_*) | in the walk (narrative_*)
  units: metres                # --metric-scale priors (scene_scale: typical object sizes, no gold); run 1: relative
  format: varies               # csv | markdown_table | text | narrative
  time_unit: varies            # seconds (default) | frames
  visibility: varies           # bridged (default) | exact
  step_order: varies           # move-turn (default) | turn-move: camera text and narrative only
  object_directions: varies    # floor (default) | camera: narrative only
  renderer: src/mvp/benchmarks/scene_sheet.py:render_scene
  placement: between video and question

prompt:
  system: You are a helpful assistant.
  template: lmms-eval
  reasoning: varies            # direct (15 conditions) | structured CoT, cot_prompt (3)
  thinking: off
  max_new_tokens: varies       # 256 direct | 2,048 CoT
  decoding: greedy
  answer_extraction: score_final  # the last "Answer: X", else the last lone letter / the last number

training: none
```

## Metrics and analysis

- **Scores, for every condition:** VSI-Bench's overall (the mean of the 8 task scores), per question type, and three
  groupings:
  - **metric tasks:** the mean of absolute distance, object size and room size, which the metres are for;
  - **direction and route:** the mean of relative direction (its three types as one task) and route planning, the
    tasks the camera's path is most likely to help (50 questions);
  - **overall without the metric tasks:** the other 5 tasks.
- **Answer letters** (from run 1): on the multiple-choice questions, each condition's share of B answers, and its
  accuracy where the true answer is B and where it is not, with the score of always giving one letter. A gain on
  direction and route counts only if it holds where the answer is not B.
- **Numeric answers:** the median answer per numeric question type against the true median, to see whether the
  metres reach the model's answers.
- **Replies:** the parse rate (`parsed(..., final=True)`), the reply length (median and 90th percentile), and the
  replies that reach the limit. For CoT, as in the structured-CoT pilot:
  - the share with an "Answer:" line;
  - the cut-off replies with no answer line, and how many of those the rule still reads;
  - the cut-off replies that loop: 10 or more mm:ss times, or a last 2,000 characters that zlib compresses below a
    quarter of their size;
  - the score with cut-off replies counted wrong.
- **Reasoning check (CoT):** over replies whose sheet has objects, the share that name a sheet row (an id such as
  `chair_2`), copy an object number as the sheet writes it, or copy a number the sheet gives only for the camera
  (camera positions, or the move lengths of the compass path and the narrative). These are the structured-CoT pilot's
  checks, with the camera added. They are given for all replies, the computing tasks, direction and route, and the
  cut-off replies.
- **Input tokens** per question and condition, from the processor (the notebook's input-length table). Run 1's sheet
  token counts are below. In metres the numbers have about as many digits, and the CoT instructions add about 150
  tokens. The video adds about 2,400:

  | sheet | mean | median | max |
  |---|---:|---:|---:|
  | `obj_csv` | 604 | 571 | 2,485 |
  | `obj_markdown_table` | 689 | 655 | 2,818 |
  | `obj_text` | 730 | 686 | 3,251 |
  | `cam_csv` | 1,830 | 1,794 | 3,708 |
  | `cam_markdown_table` | 2,153 | 2,117 | 4,280 |
  | `cam_csv_frames` | 1,800 | 1,774 | 3,525 |
  | `cam_text` | 1,331 | 1,288 | 3,849 |
  | `cam_text_turn_move` | 1,342 | 1,298 | 3,860 |
  | `cam_text_frames` | 1,384 | 1,348 | 3,825 |
  | `narrative` | 1,229 | 1,182 | 3,211 |
  | `narrative_turn_move` | 1,241 | 1,194 | 3,232 |
  | `narrative_camera_dirs` | 1,223 | 1,174 | 3,170 |
  | `narrative_frames` | 1,281 | 1,233 | 3,264 |
  | `narrative_exact` | 1,204 | 1,158 | 3,158 |

- **Intervals:** for each comparison, the difference with a 95% interval from 10,000 paired resamples of the
  questions, drawn within each question type, the same draws for every condition (as in the earlier pilots). This is
  done for each grouping and each question type.
- **Reading rule:** a comparison is a *lead* if it differs by at least 5 points on direction and route, or 2 on the
  overall, and its interval excludes 0. A lead needs the full set of questions before anything is built on it.
  Everything else is "no signal at this size". With 25 questions a task (8–9 per relative-direction type), a single
  question type's interval spans about 30 points.
- **Examples:** for each family's direct comparison, one question the treatment gets right and its control wrong, and
  one the other way round. For each CoT condition, one right reply, one wrong and one cut off. Each comes with its
  sheet.

## Threats

| Risk | How we'd notice | What to do |
|---|---|---|
| 200 questions, about 40 comparisons: some intervals exclude 0 by chance | an isolated lead on one question type | only leads on direction and route or the overall count, and they need a full run |
| The shift to answer B passes for a gain (run 1) | a condition's gain on direction and route that vanishes where the answer is not B | read the answer-letter tables with every gain |
| The metre scale is wrong for some scenes (16% median error; 18% of scenes more than 30% off) | metric-task errors concentrated in a few scenes | stated; the scale is the one a real system would have. The oracle scale's upper bound is the format experiment's explore notebooks |
| Longer prompts hurt on their own, whatever they say | `cam_markdown_table` (the longest sheet) lowest, and losses that track length | report tokens next to scores; the camera text and the narrative are about 500 tokens shorter than `cam_csv`, so length and form can be told apart only roughly |
| The pipeline's objects are noisy (split objects, missed ones: `sheet_quality.md`), and the narrative repeats them frame by frame | the narrative worse on counting | per question type; the camera tasks are the ones that matter here |
| CoT replies loop until the limit (the CoT pilot: 12% with the gold sheet), more with the frame-by-frame narrative | cut-off and looping counts per condition | reported, and the score with cut-off replies wrong; the prompt is the pilot's, unchanged |
| The CoT runs cost hours, so they run on fewer questions | the estimate before the runs | `SUBSET` for the CoT conditions only; their comparisons are on their own questions |
| A reused `video` ran at another commit | the reuse check (same questions, frames, prompts and batches, replies under 256) | if it fails, `video` runs here |
| Colab has no linear-attention kernels, and Qwen3.5's outputs differ slightly with and without them (README: 89.6% identical) | n/a | every condition runs in the same runtime; run 1 and the reused `video` ran on Colab too |

## Runs and cost

`explore_camera_metres.ipynb` (in this folder) runs run 2 on Google Colab, built like run 1's notebook:
- the same Drive folder (`MyDrive/mvp/`), data from Hugging Face plus the scene-sheets zip, and the checks that the
  frames and timestamps match the sheets;
- every pipeline sheet of the questions' scenes extracted, which the metre scale reads;
- two prompt checks, for all 19 conditions:
  - the eval's own prompts from each condition's arguments (`--print-prompt`) against `build_sheet_prompt`;
  - question by question, that only the scene information (and for CoT the last line) differs between conditions,
    and that every sheet is in metres;
- then the input lengths;
- `video` reused if it checks out;
- timing: 5 questions per condition (the direct ones after a warm-up question), with the four CoT candidates (both
  table formats, the camera text, the narrative);
- an estimate of the rest on the current GPU; above 4 hours it stops and offers fewer CoT questions per question type;
- the direct runs on all 200 questions, the choice of the CoT table format, and then the CoT runs, all resuming after
  a crash;
- the summary.

**Files on Google Drive**, in `MyDrive/mvp/`:
- `mvp-3d-vlm-camera-metres.zip`, the code, made from the repository root with:

  ```bash
  REV=$(git stash create); git archive --format=zip -o mvp-3d-vlm-camera-metres.zip ${REV:-HEAD} src experiments/2026-11-18-scene-sheet-formats requirements.txt pyproject.toml
  ```

  It carries the uncommitted metre conversion in `src/` (`git stash create` includes changes to tracked files). The
  notebook checks that it does.
- `scene_sheets_75.zip`, `ssf_explore200_results/`, `scot_explore_results/` and `camera_explore_results/` (run 1),
  already there.

The notebook itself is opened in Colab from Drive or uploaded. Predictions go to
`MyDrive/mvp/camera_metres_explore_results/`.

**Cost**, on a Colab L4, estimated from what was measured; the notebook prints its own figure before the runs:
- **The direct conditions:** about 70 minutes, as run 1 (0.27–0.31 ms per input token, about 16 s per eval start).
- **Each CoT condition:** 45–70 minutes on all 200 questions. A batch runs until its longest reply ends, and the
  structured-CoT pilot's gold-sheet CoT run took 51,641 decoding steps at about 53 ms each (45 minutes). Batches whose
  longest reply runs to 2,048 tokens take about 2 minutes; there are 37.
- **The checks and the timing:** about 30 minutes, of which the four CoT timing batches are up to 2 minutes each.

In all, 4–5 hours with the CoT conditions on all 200 questions. That is over 4 hours, so the notebook will most likely
stop before the runs and ask for `SUBSET`. `SUBSET = 10` (95 CoT questions) brings it to about 3 hours.

## Results

### Run 2: metres, with chain of thought

The full tables are in [`analysis_metres.md`](analysis_metres.md), written by `analysis_metres.py` from
`results/zeroshot/vsibench/camera/explore200_metres/`. It has:
- scores per task and question type, and every comparison with its interval (metres − relative included);
- the CoT replies;
- the answer letters;
- the numeric answers, with two checks the results called for: answers copied from the sheet, and counts against the
  sheet's count;
- three CoT replies.

All 200 questions ran in every condition, the CoT ones too, on a Colab L4. The code was `4bd6b39` plus the
uncommitted metre conversion. `video` was again the structured-CoT pilot's run, which passed the reuse check. The CoT
table format is the markdown table: its direct `cam_markdown_table` scored 44.6 overall, against 38.4 for `cam_csv`
(`cot_table.json`).

| Condition | Overall | Metric tasks | Direction and route | Without metric tasks | Right on direction and route, answer not B (31) | Answers B, multiple choice | Sheet tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `video` | 48.6 | 49.2 | 35.6 | 48.2 | 32% | 31% | 0 |
| `obj_csv` | 40.0 | 23.5 | 49.9 | 50.0 | 32% | 49% | 602 |
| `cam_csv` | 38.4 | 23.2 | 46.0 | 47.5 | 19% | 70% | 1,826 |
| `obj_markdown_table` | 41.9 | 38.1 | 37.8 | 44.2 | 26% | 46% | 687 |
| `cam_markdown_table` | 44.6 | 37.1 | 47.9 | 49.2 | 19% | 68% | 2,149 |
| `obj_text` | 36.9 | 25.7 | 45.9 | 43.7 | 19% | 72% | 728 |
| `cam_text` | 39.8 | 31.3 | 46.1 | 44.8 | 19% | 60% | 1,326 |
| `narrative` | 37.8 | 28.3 | 44.2 | 43.6 | 19% | 64% | 1,227 |
| variants (7) | 35.6–41.1 | 19.5–30.0 | 42.2–52.0 | 43.7–49.0 | 16–29% | 55–75% | 1,202–1,796 |
| `cot_cam_markdown_table` | 34.1 | 33.5 | 36.2 | 34.4 | 35% | 22% | 2,325 |
| `cot_cam_text` | 28.7 | 26.9 | 26.7 | 29.7 | 16% | 25% | 1,501 |
| `cot_narrative` | 27.4 | 18.7 | 24.2 | 32.6 | 19% | 26% | 1,403 |

- **Metres help the metric tasks only a little, and the sheets still pull them down.**
  - Against run 1, the overall moves by −0.3 to +6.0 points. Three conditions are leads:
    - `cam_markdown_table`: +6.0 [+1.9, +10.0];
    - `narrative`: +3.9 [+0.6, +7.3];
    - `narrative_turn_move`: +3.8 [+0.1, +7.4].

    The metric tasks move by −3.1 to +12.1.
  - **Object size** rises with the tables and the narrative (+7.6 to +27.6; five intervals exclude 0). It does not
    rise with the sentences (−8.0 to +1.2).
  - **Absolute distance** rises by 1.6–13.2 points, no interval excluding 0. The median answer is 1.1–1.7 m, against
    about 1.0 in run 1 and 1.9 m true.
  - **Room size** falls with csv: `obj_csv` −22.8 (to 3.2), `cam_csv` −26.0, `cam_csv_frames` −12.8. Elsewhere it
    changes by −3.2 to +9.6.
  - Every room-size question's sheet has `room` rows: 6–12 pieces of the room, mostly 1.4–3.9 m across. `obj_csv`
    answers 10 of the 25 room-size questions with one of their numbers.
  - The narrative gives each object's distance from the camera ("… 1.775 metres away"). It answers 17 of the 25
    absolute-distance questions with one of those, not with a distance between the two objects.
  - Every sheet condition is still 11–30 points below video on the metric tasks, every interval excluding 0. Video
    only scores 49.2; the best sheet, `obj_markdown_table`, 38.1.
  - Outside the metric tasks, metres change almost nothing: 1–7 of the 50 direction and route answers.
- **No camera form helps with direct answers, as in run 1.**
  - Every interval includes 0, on the overall and on direction and route:
    - `cam_csv` − `obj_csv`: −1.6 and −3.9;
    - `cam_markdown_table` − `obj_markdown_table`: +2.7 and +10.1;
    - `cam_text` − `obj_text`: +2.8 and +0.1.
  - The markdown table's +10.1 on direction and route comes from the shift to B.
    - With the cameras, B answers rise from 46% to 68% of the multiple-choice questions.
    - 16 direction and route answers change, 13 of them to B.
    - Where the answer is not B, it gets 19% right, against 26% without the cameras.
  - `cam_markdown_table` is the best sheet condition overall (44.6; −3.9 against video, interval including 0).
  - The narrative is no better than the camera text (−1.9 and −1.9) or the sentences alone (+0.9 and −1.8).
- **The options: one lead,** `narrative_turn_move` − `narrative`, +7.9 on direction and route [+1.9, +15.7].
  - 7 answers change, 4 of them to right, 3 of those 4 not B.
  - It loses 8.8 points on the metric tasks (room size), and the overall is −2.3.
  - In run 1 the same comparison was +4.0 on direction and route.
  - It is one lead among about 50 comparisons, and needs the full set of questions before anything is built on it.
- **Chain of thought lowers every camera form by about 11 points.**
  - CoT − direct, every interval excluding 0:
    - `cam_markdown_table`: −10.6 overall [−18.5, −2.8];
    - `cam_text`: −11.1;
    - `narrative`: −10.5.

    On direction and route: −11.7 (interval including 0), −19.4 and −20.0. Against video: −14.5 to −21.2.
  - It loses most on counting (−25 to −27 points) and route planning (−20 to −36). With the table it also loses on
    relative distance (−28).
  - In the structured-CoT pilot, CoT cost 3.3 points with the gold sheet and 13.3 with video only. With the
    pipeline's sheets the loss (10.5–11.1) is closer to the video-only one.
  - **Cut-off replies:** 22, 47 and 57 of 200 were cut off at 2,048 tokens (table, compass path, narrative); the pilot
    had 12% with the gold sheet. 13, 23 and 45 of those loop. With the compass path and the narrative, 16 and 14 of
    the 25 route-planning replies are cut off.
  - **CoT trusts the pipeline's objects.** It answers 13–16 of the 25 counting questions with the sheet's count of
    the class, which is right for 6. It counts split pieces as separate objects (8 beds for 2).
  - **The letter shift goes away:** CoT answers B 22–26% of the time. With the table, it gets 35% right where the
    direction and route answer is not B. That is the best of any condition, but only 11 of 31 questions against
    video's 10.
  - **What the replies use:** with the table, CoT works from the numbers. 71% of replies copy an object number and
    31% a camera position. Replies with the compass path and the narrative name rows (86% and 71%) but rarely copy a
    move length (11–13%).
- **Replies, direct:** every reply was parsed. 3 of the 3,000 ran to 256 tokens, all room-size questions: a product
  of the sheet's sizes that never ended.
- **Cost on the L4** (the results' file times):
  - the timing: 14 minutes;
  - the 14 direct runs: 63 minutes;
  - the CoT runs: 54, 63 and 63 minutes.

  That is about 4.3 hours from the timing to the summary, within the planned 4–5.

### Run 1: relative units

The full tables are in [`analysis.md`](analysis.md), written by `analysis.py` from
`results/zeroshot/vsibench/camera/explore200/`: scores per task and question type, every comparison with its interval,
the answer letters, and the numeric answers. `video` was copied from the structured-CoT pilot's run, which passed the
reuse check. All 200 questions ran in every condition, at commit `8f027cb`, in about 70 minutes on a Colab L4.

| Condition | Overall | Direction and route | Without metric tasks | Right on direction and route, answer not B (31) | Answers B, multiple choice | Sheet tokens |
|---|---:|---:|---:|---:|---:|---:|
| `video` | 48.6 | 35.6 | 48.2 | 32% | 31% | 0 |
| `obj_csv` | 39.3 | 49.9 | 48.4 | 35% | 45% | 605 |
| `cam_csv` | 38.6 | 46.0 | 45.9 | 19% | 70% | 1,831 |
| `obj_markdown_table` | 41.2 | 44.0 | 47.0 | 35% | 39% | 690 |
| `cam_markdown_table` | 38.7 | 44.1 | 46.9 | 23% | 64% | 2,154 |
| `obj_text` | 37.3 | 45.9 | 44.9 | 19% | 70% | 731 |
| `cam_text` | 38.3 | 47.9 | 45.7 | 23% | 50% | 1,332 |
| `narrative` | 34.0 | 46.0 | 42.8 | 23% | 64% | 1,230 |
| variants (7) | 31.8–38.7 | 41.9–50.0 | 42.5–46.1 | 13–26% | 56–76% | 1,205–1,801 |

- **No camera form helps.**
  - On the overall and on direction and route, every camera comparison's interval includes 0:
    - `cam_csv` − `obj_csv`: −0.7 and −3.9;
    - `cam_markdown_table` − `obj_markdown_table`: −2.5 and +0.1;
    - `cam_text` − `obj_text`: +1.1 and +2.0.
  - The narrative is no better than the camera text (−4.4 and −1.9) or the sentences alone (−3.3 and +0.1).
  - None of the README's comparisons is a lead.
- **The options don't matter.** Each variant changes 1–6 of the 50 direction and route answers, and none is a lead.
- **Adding the cameras mostly moves multiple-choice answers to B, and the moved answers are no better.**
  - The share of B answers rises with the cameras: csv 45% → 70%, markdown table 39% → 64%. With the narrative and
    the variants it is 56–76%. Video only answers B 31% of the time, and 33% of the true answers are B.
  - Of the 50 direction and route answers, `cam_csv` changes 13 from `obj_csv`'s, all to B: 3 become right, 5 become
    wrong.
  - On the 31 direction and route questions whose answer is not B, the camera, text and narrative conditions get
    13–26% right, against 32% with video only and 35% with the objects-only tables.
- **The sheets' gain over video on direction and route is letter bias.**
  - Every sheet condition beats video there by 6–14 points. Only `obj_csv`'s +14.3 is a lead (interval +1.9 to
    +28.1).
  - But always answering B scores 38.2 on these questions, against video's 35.6.
  - Where the answer is not B, `obj_csv` gets 35% right against video's 32%, one question more.
  - Over all 100 multiple-choice questions, no condition beats video's 48%: they score 41–50%.
- **The metric tasks fail as in the format experiment.** Every sheet condition is 7–17 points below video overall,
  every interval excluding 0, and the loss is on the three metric tasks.
  - The model reads the sheets' relative units as metres. The median absolute-distance answer is 1.0–1.1 in every
    sheet condition, against 1.9 m true and 2.1 with video only.
  - Room size falls as far as a median of 1.3 m² (`narrative_turn_move`; true median 20.7).
  - Without the metric tasks, the objects-only tables equal video (48.4 and 47.0 against 48.2). The camera, text and
    narrative conditions are 1–6 points below it.
- **Replies:** every reply was parsed. None was longer than 5 tokens with a sheet (10 with video only), so the model
  never reasoned over the sheet.

## Conclusion

- **Answer to the question:** no. With the sheets in metres, giving Qwen3.5-2B the camera's path does not help, in any
  form or with any option:
  - not as a table (csv, markdown table);
  - not as a compass path in text;
  - not in the narrative.

  Reasoning step by step makes each form worse, by about 11 points overall. Metres help the metric tasks only a
  little; with any sheet they stay 11–30 points below video only.
- **Which form to develop, if any:** none, by the plan's rule.
  - No camera form beats its objects-only control on direction and route except through the shift to B.
  - No CoT condition beats its direct one.
  - If one form is kept for later tests, it should be the markdown table: the best sheet overall in metres (44.6),
    and the best with CoT (34.1).
- **Caveats:**
  - 200 questions, 50 of them direction and route, 19 of those with answer B;
  - the metre scale's error (median 16%);
  - the pipeline's objects: its count is right on 6 of the 25 counting questions, and its `room` rows invite
    copying;
  - the CoT prompt is the pilot's, unchanged, and loops and long replies cut off 11–29% of CoT replies;
  - one 2B model, with thinking off.
- **Next:**
  - As the plan says for this outcome: precomputed relations instead of the camera's path, including direction
    relations ("standing at the sofa facing the table, the tv is front-left").
  - On the metric tasks, the model answers with a sheet number that looks like the answer. So the sheet should give
    the numbers that answer those questions: distances between objects, the room's extent. It should leave out the
    ones that mislead: the `room` segments, and the narrative's distances from the camera.
  - Merge the pipeline's split objects, for counting.

Run 1's conclusion, in relative units: with direct answers, none of the ways of giving Qwen3.5-2B the camera's path
helps:
- not the camera tables (csv, markdown table);
- not the compass path in text;
- not the narrative of the walk;
- and none of their options.

The cameras shift multiple-choice answers to B, and where the true answer is not B the camera, text and narrative
conditions are worse than video only and the objects-only tables. Its caveats:
- the relative units, which sank the metric tasks in every condition alike (run 2 corrects this);
- direct answers only (run 2 adds CoT);
- 200 questions, 50 of them direction and route;
- 19 of those 50 answers are B.

## Log

- 2026-11-20: plan written (draft). Conditions: every camera format the code has, each with its objects-only control
  in the same format, the camera text and the narrative, and one variant per text option. The sheet token counts
  were measured on the 200 questions with the Qwen3.5-2B tokenizer.
- 2026-11-20: plan confirmed, with json and markdown key-value dropped (Decisions), leaving 14 conditions to run.
  Cameras stay in the objects' format. Status: planned.
- 2026-11-21: `explore_camera.ipynb` written, from explore200.ipynb and the CoT pilot's notebook. Tested locally (RTX
  3060 Laptop, a stand-in Drive folder, the same code zip), at commit `8f027cb`:
  - the data, fps and frame checks;
  - both prompt checks, in all 15 conditions;
  - the input lengths;
  - the reuse check: the CoT pilot's `A_video` passes with batches of 8 and is rejected with batches of 2;
  - timing, estimate and runs for `video`, `cam_csv` and `narrative` on 10 questions, with batches of 2;
  - the answers, summary, intervals and examples, on made-up predictions for every condition.

  Three changes came out of the tests:
  - The timing now starts with a warm-up question: the eval's first batch took 3–4 times as long as the others.
  - The input lengths now come from the tokenizer, checked against the processor on one video (4 minutes down to 1).
  - The reuse check now compares batches as well as prompts. With batches of 2, `video` answered 1 of the 10
    questions differently from the batch-8 predictions.
- 2026-11-21: run 1, on a Colab L4 on 2026-11-20, at commit `8f027cb`: all 200 questions in all 14 conditions, in
  about 70 minutes. `video` was copied from the structured-CoT pilot's `A_video` (reused.json), as the reuse check
  allowed. The notebook's results zip was unpacked into `results/zeroshot/vsibench/camera/explore200/`.
  `analysis.py` recomputes the scores and intervals from the predictions; they match the notebook's tables. It adds a
  check that was not in the plan, the answer letters, because several sheet conditions answered B on 21–24 of the 25
  relative-direction questions. Results, conclusion and outcome were written the same day. Status: done.
- 2026-11-21: run 2 planned.
  - **Why:** the experiment was meant to be in metres. Run 1's plan wrote the sheets in relative units because
    `--metric-scale priors` converted only tables of objects. That was a choice made in writing the plan, not asked
    of the owner.
  - **Code:** `--metric-scale priors` now covers the cameras and the narrative, with the objects' scale for the same
    question.
    - `scene_sheet.py`: camera positions in the tables; move lengths in the compass path
      (`camera_sheet_format_converter.frame_instructions(..., scale=)`); moves, distances and sizes in the narrative.
      Preambles that say "approximately in metres"; `check_options` now refuses priors only for gold sheets.
    - Directions, turns and the moves' thresholds don't depend on the scale. The relative-units text is unchanged.
    - Tested in `src/mvp/benchmarks/tests/test_scene_sheet.py` (`MetricScale`: the cameras, the compass path and the
      narrative in metres). The three suites (scene sheet, prompts, OpenVocab adapter) pass: 158 tests.
  - **Design:**
    - all 15 direct conditions again in metres;
    - three CoT conditions (the better table format, the camera text, the narrative) with the structured-CoT pilot's
      prompt and limit;
    - the answer letters and the metres − relative comparison, now planned;
    - the CoT diagnostics of the structured-CoT pilot.
  - **The notebook:** `explore_camera_metres.ipynb`. Run 1's notebook stays as it ran. Status: planned.
- 2026-11-21: run 2, on a Colab L4. The code zip was `4bd6b39` plus the uncommitted metre conversion (`git stash
  create` commit `ec85068`).
  - All 15 direct conditions and the 3 CoT ones ran on all 200 questions, in about 4.3 hours. `video` was copied again
    from the CoT pilot's `A_video`.
  - The rule chose the markdown table for CoT (`cot_table.json`).
  - The results zip was unpacked into `results/zeroshot/vsibench/camera/explore200_metres/`.
  - `analysis_metres.py` recomputes the scores, reply counts and intervals; they match the notebook's tables. It adds
    two checks that were not in the plan: numeric answers copied from the sheet (the `room` rows, the narrative's
    distances from the camera), and counting answers against the sheet's count.
  - Results, conclusion and outcome were written the same day. Status: done.

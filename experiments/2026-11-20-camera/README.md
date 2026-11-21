---
id: 2026-11-20-camera    # also the folder name: experiments/<id>/
title: Which way of giving Qwen3.5-2B the camera's path (camera tables, camera text, the narrative) is worth developing
owner: Nicolás Ramos Fernandez
created: 2026-11-20
status: done                 # draft | planned | running | analysing | done | abandoned
tags: [scene-sheet, cameras, narrative, format, pilot]
varies: [scene_sheet.cameras, scene_sheet.format, scene_sheet.time_unit, scene_sheet.step_order,
         scene_sheet.object_directions, scene_sheet.visibility]
depends_on: [2026-11-18-scene-sheet-formats (pipeline sheets, explore200's questions and video-only predictions)]
---

# Camera sheets and the narrative: a pilot

**Outcome:** no camera form is worth developing for Qwen3.5-2B with direct answers. Adding the cameras changes the
overall by −2.5 to +1.1 points and direction and route by −3.9 to +2.0, every interval including 0. Where the true
answer is not B, the camera forms are worse (19–23% against 35% for the objects-only tables), because the cameras
shift multiple-choice answers to B.

**This is an exploratory pilot, not a result for the report.** It runs every way the code has of giving the model the
camera's path on 200 questions, to decide which one deserves a full experiment. Where this README departs from
`experiments/TEMPLATE.md`, that is on purpose:
- There is no single hypothesis that a test must reject. There are expectations, and each says what we do next.
- There is no sign test and no Holm correction. Every comparison gets a bootstrap 95% interval, and with about 25 of
  them, one or two will exclude 0 by chance. A difference here is a lead to confirm, not a finding.
- The settings stay close to the earlier experiments' (model, frames, prompt, answer rule, questions). There is no gold
  sheet and no chain of thought.

## Question

Does giving Qwen3.5-2B the camera's path, along with the pipeline's objects, help it on VSI-Bench? If so, in which
form: the camera poses as a table (csv or markdown table), the path as compass directions in text, or one narrative of the walk that mixes the camera's moves with the objects it sees? And which of the text
forms' options matter (step order, how object directions are said, timestamps or frame numbers, exact or bridged
visibility)?

## What we expect, and where each result points

- **Camera tables (`cam_<format>` against `obj_<format>`):** little or no gain, and possibly a loss. The tables hold
  positions, quaternions and view directions in VGGT's relative units. The format experiment found that the 2B does
  not compute distances or directions from coordinates it is given, even correct ones in metres. Each table also adds
  1,200–1,500 tokens (Metrics). If a table helps all the same, on relative direction or route planning, the model can
  read camera poses, and pose tables are worth a full run.
- **Camera text and the narrative (`cam_text`, `narrative`):** these are the forms most likely to help, and mostly on
  relative direction, route planning and appearance order. They do the geometry in advance and say it in words, as
  the precomputed relations did:
  - In the structured-CoT pilot, giving the model distances in words (relations, direct answer) beat asking it to
    compute them from coordinates (CoT) by 13.9 points.
  - Both texts estimate which way is up from the camera's path, which the object sheet cannot say. Its y axis is
    "down" only as seen from the first camera, which is often tilted. So both can say "left" and "right" along the
    floor.
  - The narrative also says when each object comes into view, in order, and turns are said as left and right, as
    route planning's answers are.

  *If one of them beats its control on the direction and route tasks* (Metrics) by 5 points or more, with an interval
  that excludes 0, that form is worth a full run on the 2,728 questions, and its best variant is the one to take.
  *If neither does,* the camera's path in the prompt is not what the 2B is missing. Work goes to precomputed relations
  from the pipeline sheet, or to a model that reasons.
- **Narrative against camera text (`narrative` − `cam_text`):** this asks whether one walk, from the camera's point of
  view, is better than the objects and the path kept apart. No expectation; the result says which text form to
  develop.
- **The options (step order, object directions, frame numbers, exact visibility):** these are small rewordings, and
  we expect each to stay within a few points of its default. Any that moves the direction and route tasks by 5 points
  or more is worth keeping in mind. Otherwise the defaults stay.
- **Against video only:** the pipeline sheet in relative units lowers the metric tasks (format experiment: object
  size 59.4 → 18.2, absolute distance 37.2 → 20.3 with csv), and every condition here has it. So the overall can stay
  below video only even if the cameras help. Read `<condition> − video` on the direction and route tasks, and on the
  overall without the metric tasks.

## Background

- **Format experiment** (`experiments/2026-11-18-scene-sheet-formats/README.md`): Qwen3.5-2B on 2,728 questions,
  direct answers, pipeline sheets in relative units, run at `07bad18`:
  - Every format lowered the overall: video only 43.2, csv 34.9, markdown table 35.1, json 31.3, text 29.0, markdown
    key-value 25.5.
  - Relative direction did not improve: from 45.6 to 46.2–47.1 with the tables and json, and down to 38.2–40.7 with
    text and markdown key-value.
  - Route planning rose from 27.6 to 29.3–39.7 (58 questions).
  - Appearance order fell from 44.1 to 34.1–41.7.

  Those sheets had `first_seen` and `num_frames` and no cameras.
- **The camera options came later and have never been run.** Commits `50fe543` (cameras, narrative, CoT), `980cf00`
  (the `visible` intervals that replace `first_seen` and `num_frames`) and `080acf4` (cameras always in the objects'
  format) are all after the format runs. The explore notebooks ran at `b381205`, also before them. The options are
  `--cameras`, `--sheet-format narrative`, `--time-unit`, `--visibility`, `--step-order` and `--object-directions` in
  `python -m mvp.eval.vsibench`, rendered by `src/mvp/benchmarks/scene_sheet.py` (`render_cameras`,
  `render_narrative`).
- **What the camera forms are**, for question 974 (`python -m mvp.benchmarks.scene_sheet_prompts --question 974
  --format csv text narrative --cameras off on` prints them):
  - **Camera table:** "Camera sheet: where the camera is in each of the 16 video frames…", then one row per frame:
    `time,x,y,z,qw,qx,qy,qz,forward_x,forward_y,forward_z`.
  - **Camera text:** north is where the camera faces in the first frame, along the floor, and up is estimated from the
    path. Each line reads "At 13.6 seconds: Go 0.330 units north and turn 37° to the right and 39° up. Go 0.159 units
    north and …".
  - **Narrative:** egocentric, with the objects. Each line reads "At 13.6 seconds: Move 0.330 units forward and turn
    37° to the right and 39° up; you see tv_1 (a tv, 0.014 × 0.185 × 0.339 units) ahead, slightly to your right,
    0.629 units away; sofa_1 is no longer in view. …".
- **The questions:** `explore200_ids.txt` (in the format experiment's folder). There are 200 questions, 25 per task
  (relative direction 8 easy, 8 medium, 9 hard), from 25 scenes (ScanNet 71, ARKitScenes 66, ScanNet++ 63
  questions). 8 of the 200 have an empty pipeline sheet. They were not chosen for anything in this experiment.
- **Video only is already on Drive:** explore200's `A_video` and the structured-CoT pilot's `A_video` (the latter at
  `7d39234`, also in `results/zeroshot/vsibench/structured_cot/explore200/`). One of them stands in for `video` if it
  passes the reuse check (Runs). explore200's `pipeline_csv` cannot stand in for `obj_csv`, because it predates the
  `visible` intervals.

## Design

### Conditions

Every sheet is the pipeline's (VGGT + SAM 3 + tracking), in VGGT's relative units. `--metric-scale priors` does not
apply to cameras or the narrative (`scene_sheet.check_options`). The cameras are always written in the objects'
format (the code's rule since `080acf4`), so each camera table has its objects-only control in the same format; the
camera effect is `cam_<f>` − `obj_<f>`. The defaults are timestamps in seconds, bridged visibility, move then turn,
and object directions along the floor. A blank cell is the default.

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

That is 14 conditions to run, plus `video`. Left out:
- json and markdown key-value (Decisions);
- frame numbers with the markdown table;
- exact visibility outside the narrative (it changes only the objects' `visible` field);
- the combinations of the variants with each other.

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

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Questions | explore200's 200 (`explore200_ids.txt`), or a per-type subset if the timing says more than 4 hours | The same questions as the explore and CoT pilots, so `video` can be reused and the results set side by side; 25 per task |
| Sheet source | pipeline only | The aim is the sheet we can build; gold sheets have no cameras (`check_options`) |
| Reasoning | direct (lmms-eval's prompt), thinking off | As the format experiment; CoT is out of scope here |
| Token limit | 256 | Direct replies are 1–10 tokens (the CoT pilot's `A_video`: at most 10). 256 bounds any condition that starts to reason, and replies that reach it are counted. A reused reply under 256 tokens is what greedy decoding gives under either limit |
| Answer rule | `score_final` (`--final-answer`) in every condition | One rule for all, as the earlier pilots; parse rate with `parsed(..., final=True)` |
| Batches | up to 8 questions of one video (`--batch-size 8`, `--token-budget 120000`, the eval's defaults) | As explore200 and the CoT pilot, which the reuse check needs |
| Formats | csv, markdown table and text, plus the narrative; json and markdown key-value dropped | They were the worst object formats in the format experiment (31.3 and 25.5, against 34.9 and 35.1 for the tables), and their camera sheets are the longest (2,893 and 2,687 tokens on average), so they would cost the most and are the least likely to be used |
| Camera format | the objects' format | The code's rule (`080acf4`); one control per format separates the camera effect from the format effect |
| No metric scale | relative units everywhere | `--metric-scale priors` converts tables of objects only, not cameras or the narrative; one scale for all conditions keeps the comparisons about the cameras |

## Configuration

```yaml
data:
  benchmark: vsibench
  split: full
  subset: experiments/2026-11-18-scene-sheet-formats/explore200_ids.txt  # or k per question type (Runs)
  scenes: 25 of the 75 exported scenes (the questions')
  question_types: all          # all 10
  n_questions: 200             # the same questions in every condition

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
  units: relative              # VGGT's units, said in each preamble; no --metric-scale
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
  reasoning: direct
  thinking: off
  max_new_tokens: 256
  decoding: greedy
  answer_extraction: score_final  # the last "Answer: X", else the last lone letter / the last number

training: none
```

## Metrics and analysis

- **Scores, for every condition:** VSI-Bench's overall (the mean of the 8 task scores), per question type, and two
  groupings:
  - **direction and route:** the mean of relative direction (its three types as one task) and route planning, the
    tasks the camera's path is most likely to help (50 questions);
  - **overall without the metric tasks:** the 5 tasks other than absolute distance, object size and room size, which
    the relative units lower in every sheet condition alike.
- **Replies:** the parse rate (`parsed(..., final=True)`), median and longest reply length, and how many replies reach
  256 tokens.
- **Input tokens** per question and condition, from the processor (the notebook's input-length table). The sheet
  alone, with the Qwen3.5-2B tokenizer, on the 200 questions (measured 2026-11-20). The video adds about 2,400:

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
  done for the overall, direction and route, the overall without the metric tasks, and each question type.
- **Reading rule:** a comparison is a *lead* if it differs by at least 5 points on direction and route, or 2 on the
  overall, and its interval excludes 0. A lead needs the full set of questions before anything is built on it.
  Everything else is "no signal at this size". With 25 questions a task (8–9 per relative-direction type), a single
  question type's interval spans about 30 points.
- **Examples:** for each family, one question the treatment gets right and its control wrong, and one the other way
  round, each with its sheet and reply; also the longest replies.

## Threats

| Risk | How we'd notice | What to do |
|---|---|---|
| 200 questions, about 25 comparisons: some intervals exclude 0 by chance | an isolated lead on one question type | only leads on direction and route or the overall count, and they need a full run |
| Longer prompts hurt on their own, whatever they say | `cam_markdown_table` (the longest sheet) lowest, and losses that track length | report tokens next to scores; the camera text and the narrative are about 500 tokens shorter than `cam_csv`, so length and form can be told apart only roughly |
| The relative units lower the metric tasks in every condition | low absolute distance and object size everywhere | read direction and route, and the overall without the metric tasks |
| The pipeline's objects are noisy (split objects, missed ones: `sheet_quality.md`), and the narrative repeats them frame by frame | the narrative worse on counting | per question type; the camera tasks are the ones that matter here |
| Replies that start to reason and are cut off at 256 | replies at the limit, parse rate under 100% | counted per condition; the rule reads every reply the same way |
| A reused `video` ran at another commit | the reuse check (same questions, frames, input lengths batch by batch, replies under 256) | if it fails, `video` runs here |
| Colab has no linear-attention kernels, and Qwen3.5's outputs differ slightly with and without them (README: 89.6% identical) | n/a | every condition runs in the same runtime; the reused `video` ran on Colab too |

## Runs and cost

`explore_camera.ipynb` (in this folder) runs it on Google Colab, built like explore200.ipynb and the CoT pilot's
notebook:
- the same Drive folder (`MyDrive/mvp/`), data from Hugging Face plus the scene-sheets zip, and the checks that the
  frames and timestamps match the sheets;
- two prompt checks:
  - the eval's own prompts from each condition's arguments (`--print-prompt`) against `build_sheet_prompt`;
  - question by question, that only the scene information differs between conditions;
- then the input lengths;
- the `video` predictions reused if they check out;
- 5 questions timed per condition (one video, one batch, after a warm-up question) and an estimate of the rest on
  the current GPU. Above 4 hours it stops and offers fewer questions per question type;
- the runs, which resume after a crash;
- the summary.

**Files on Google Drive**, in `MyDrive/mvp/`:
- `mvp-3d-vlm-camera.zip`, the code, made from the repository root with:

  ```bash
  REV=$(git stash create); git archive --format=zip -o mvp-3d-vlm-camera.zip ${REV:-HEAD} src experiments/2026-11-18-scene-sheet-formats requirements.txt pyproject.toml
  ```

  The notebook needs nothing from this folder, so the zip works without it being committed.
- `scene_sheets_75.zip`, already there for explore200. It must have each pipeline sheet's `cameras.csv`, which the
  input check checks. The repository's copy does.
- `ssf_explore200_results/` and `scot_explore_results/`, already there. The video-only predictions are copied from
  them if they check out.

The notebook itself is opened in Colab from Drive or uploaded. Predictions go to `MyDrive/mvp/camera_explore_results/`.

**Cost:** about 2–3 hours on a Colab L4, an estimate; the notebook times it and prints its own figure before the
runs. These are direct answers (1–4 tokens), so the time is almost all prefill, and it grows with the input tokens.
In batches of 8, padded as the model gets them, the 14 conditions come to 12.1 million input tokens on the 200
questions. That ranges from 0.71 million (`obj_csv`) to 1.03 million (`cam_markdown_table`); `video`, if it runs,
adds 0.52 million.

The estimate's basis, measured locally on 2026-11-21 (RTX 3060 Laptop, with the linear-attention kernels, batches of
2): 0.38–0.42 ms per input token, and 12–17 s per run to start the eval and warm up. Colab has no kernels; on this
laptop they speed up prefill 3.3 times (memory notes). The L4 is about 1.5–2 times this laptop's speed without them,
which comes to roughly 0.5–0.9 ms per token. Add about 25 minutes for the data, the checks, the timing runs and the
start of each run.

If the notebook's estimate is over 4 hours, `SUBSET = 10` (95 questions) halves it.

**Measured, on a Colab L4 on 2026-11-20:** about 70 minutes for the timing and the 14 runs on all 200 questions, going
by the files' times. That is 0.27–0.31 ms per input token and about 16 s per eval start, so the estimate above was
about twice too high.

## Results

The full tables are in [`analysis.md`](analysis.md), written by `analysis.py` from
`results/zeroshot/vsibench/camera/explore200/`: scores per task and question type, every comparison with its interval,
the answer letters, and the numeric answers. `video` was copied from the structured-CoT pilot's run, which passed the
reuse check. All 200 questions ran in every condition, at commit `8f027cb`.

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

### What the results show

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

- **Answer to the question:** no. With direct answers, none of the ways of giving Qwen3.5-2B the camera's path helps
  on VSI-Bench:
  - not the camera tables (csv, markdown table);
  - not the compass path in text;
  - not the narrative of the walk;
  - and none of their options.

  The cameras do change how the model answers multiple-choice questions: they make it answer B more often. Where the
  true answer is not B, the camera, text and narrative conditions are worse than video only and the objects-only
  tables.
- **Which form to develop, if any:** none, for this model with direct answers.
  - The camera tables were expected not to help, and didn't.
  - The camera text and the narrative were the forms expected to help, on direction and route, and they don't. This
    is the README's case "If neither does": the camera's path in the prompt is not what the 2B is missing.
  - If one form had to be kept, the camera text, which changes the fewest answers and moves the fewest to B among the
    camera forms; nothing here favours it on accuracy.
- **Caveats:**
  - 200 questions, 50 of them direction and route, so a task's interval spans about 30 points.
  - One model, greedy decoding, Colab without the linear-attention kernels.
  - The model answered at once every time, so this says nothing about a model that reasons over a narrative or a
    camera path.
  - The pipeline sheets are in relative units, which sink the metric tasks in every condition alike.
  - 19 of the 50 direction and route answers in this sample are B, which is what lets the shift to B pass for a gain.
- **Next:**
  - Don't run the camera conditions on the full 2,728 questions.
  - For direction questions, give the answer's geometry in words, as the relations did for distances: where the target
    is from the question's standpoint and facing ("standing at the sofa facing the table, the tv is front-left"). This
    is the step the narrative leaves to the model.
  - Revisit the camera path only with a model that reasons before answering (thinking on, or a larger model).
  - In every sheet experiment with multiple choice, report the share of each answer letter and the accuracy where the
    answer is not the favoured letter. The format experiment's route-planning gain with pipeline sheets (27.6 → 39.7,
    58 questions) should be checked the same way.

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
- 2026-11-21: the run, on a Colab L4 on 2026-11-20, at commit `8f027cb`: all 200 questions in all 14 conditions, in
  about 70 minutes. `video` was copied from the structured-CoT pilot's `A_video` (reused.json), as the reuse check
  allowed. The notebook's results zip was unpacked into `results/zeroshot/vsibench/camera/explore200/`.
  `analysis.py` recomputes the scores and intervals from the predictions; they match the notebook's tables. It adds a
  check that was not in the plan, the answer letters, because several sheet conditions answered B on 21–24 of the 25
  relative-direction questions. Results, conclusion and outcome were written the same day. Status: done.

# Pre-registration: the full method against video only

Written on 2026-11-22, before any of the questions below was run in any condition. The method, the questions, the
tests and the decision rule are fixed here; nothing in them changes after results are seen. The notebook that runs it
is `final_test.ipynb`.

## Why this test

On the 200 dev questions (`dev200_ids.txt`; `ssf_dev200_relations_results`), adding precomputed relations to the
deduplicated pipeline_v3 sheet changed how Qwen3.5-2B uses the sheet: it repeated the precomputed closest-point distance
in 12 of 12 absolute-distance questions that had one, the precomputed closest object in 19 of 19 relative-distance
questions, and the stated direction in 9 of 11 relative-direction questions. Scores (VSI-Bench, dev):

| Task (25 questions each) | Video only | pipeline_v3_dedup | pipeline_v3_dedup_relations |
|---|---:|---:|---:|
| Absolute distance | 43.2 | 42.4 | 52.8 |
| Relative distance | 48.0 | 44.0 | 60.0 |
| Relative direction | 31.5 | 35.2 | 59.7 |

The full method below, simulated from those answers, scored 48.8 against video's 42.0 (+6.8, 95% interval
[+0.1, +13.4]). It was put together on those same questions, so that number is optimistic; this test uses questions no
condition has run.

## The full method (frozen)

- **Sheet**: pipeline_v3 (`rerun_detection.FROZEN["pipeline_v3"]`: SAM 3 with the fixed prompts, tracking from score
  0.4, rows of tracks seen in one frame dropped below score 0.5, the point merge, the scale from the unmerged rows),
  deduplicated by `dedup_sheets.py`: for relative-direction, relative-distance and absolute-distance questions, each
  object the question names keeps one row, the one seen in the most frames; ties are broken for distance questions by
  nearness to the other named objects, for direction questions by the track's detection score, then by the larger box.
- **Relations** (`explore_sheets.relation_text`, mode "v3"), appended after the sheet only when it has a row for every
  object the question names: for distance questions the closest-point distances between the named objects (and the
  closest candidate for relative distance); for relative-direction questions the direction their positions imply.
  Nothing for other questions.
- **Which answer is used**: the video-only answer for object-size and room-size questions; the sheet condition's answer
  (`pipeline_v3_dedup_relations`) for every other question.

## Questions

`final_ids.txt`: every question on the 25 scenes of `explore200_scenes.txt` that is not in `explore200_ids.txt`,
`explore_ids.txt`, `dev200_ids.txt` or `confirm_ids.txt`: 538 questions (122 absolute distance, 123 relative distance,
128 object size, 129 appearance order, 36 counting). These scenes were never used to choose the pipeline's settings or
the sheet rules; these questions were never run. There is no relative-direction, route-planning or room-size question
left on these scenes, so the method's handling of those is not tested here. 114 of the 245 distance questions have a
row for every named object and therefore get relations; the other 131 get the deduplicated sheet alone.

## Conditions and settings

- **`A_video`**: the video only.
- **`pipeline_v3_dedup_relations`**: the video, the deduplicated pipeline_v3 sheet (csv, approximately in metres with the
  scale from typical object sizes of its unmerged rows) and the relations above.

Both: Qwen/Qwen3.5-2B (the Hugging Face model as served when the notebook runs; no revision is pinned), thinking off, 16
frames sampled `uniform128`, up to 2,048 new tokens, the final answer scored as VSI-Bench scores it; the code at
`b381205` (`b3812055a0265528b738162847db8bb2e06642a2`) with `explore_sheets.py` and `scene_scale.py` as below. Both
conditions answer all 538 questions; the full method is assembled from their answers as above.

## Hypotheses

- **H1 (primary)**: on distance questions, the sheet with relations raises the score over the video alone.
- **H2 (secondary)**: the full method raises VSI-Bench's overall score over the video alone.

## Primary test and decision rule

The 245 distance questions (absolute and relative), paired by question. A question is *helped* if
pipeline_v3_dedup_relations' score is higher than A_video's and *hurt* if it is lower (absolute distance is scored by
MRA, relative distance 0 or 1). Exact two-sided sign test on helped against hurt (equal scores left out).

**H1 confirmed** if p < 0.05 and helped > hurt. **Not confirmed** otherwise. One primary test, no correction.

## Secondary test

The full method against A_video on all 538 questions: the difference in VSI-Bench's overall score (the mean over the
tasks present of each task's mean score), with a 95% paired bootstrap interval, 10,000 resamples of the questions within
each question type, seed 0. Reported with its interval; it does not change the primary verdict. H2 is described as
supported only if the interval excludes 0.

## Descriptive (no decision)

Per task: each condition's score, the full method's, and helped / hurt. Anything else is labelled exploratory.

## What may not change

The method, the questions, the conditions, the settings, the inputs below, the tests and the decision rule. A run that
stops before answering every question is continued or run again with the same inputs; its answers are kept as they
come. No question is dropped.

## Power

From the dev distance questions (21 helped, 12 hurt among 50 questions against video): 66% of scores change and 64% of
the changes are helps. At that rate the 245 questions give about 162 changed scores, and the sign test (two-sided, 0.05)
confirms H1 with probability 0.93 (exact binomial). Dev effects have shrunk on new questions before; with 60% of
changes helps the power is 0.70, with 57% it is 0.40. A "not confirmed" result does not show that the relations have no
effect.

## Files (SHA-256)

Checked by the notebook before it runs:

| File | SHA-256 |
|---|---|
| `final_ids.txt` | `e56aed4e41f9f157956af5481d87aa0de4851befddf29840cedd138684eabab7` |
| `scene_sheets_pipeline_v3_dedup2_explore200.zip` (the deduplicated pipeline_v3 sheets of the 25 scenes, with `index.csv` and `scale_rows.json`) | `97c9feb7d43ee8527665fe596817d30d3364594fbb4c8e1784877d6463653a65` |
| `explore_sheets.py` (rendering, `relation_text`) | `23b88303ba6d58e69af443a013af92e87dbbf85dd06e4756cd95afc18fd048cd` |
| `src/mvp/benchmarks/scene_scale.py` (the scale, with `SCALE_ROWS`) | `4ed52ca3cc1e489a762bc4c43a181c58ff19f48f8b64b6caa3771b5c8fcb4d53` |

The notebook and what the sheets were made from:

| File | SHA-256 |
|---|---|
| `final_test.ipynb` | `99c768114c099c3946bd4a59d3f0a8dff08698731dea996d434e664e29f0eb81` |
| `pipeline_v3-20261121T214213Z-1-001.zip` (the tracks of `rerun_detection.ipynb`'s pipeline_v3 run) | `2f2f800a210afcec439c6c32f7897db54397394e3c3600ca40712a3ddf257470` |
| `pipeline_v3.py` | `90b4d7e6f635dd4fabe8bb9011fbd42d428bfaf3004027cfcd52e2cf5cedb933` |
| `merge_points.py` | `8653826912a49c3f11f8f86409e48501d49a19a09674022f7f433635236f9386` |
| `dedup_sheets.py` | `10356327c29aa7ce2d04338a3d0a4dfc6846be483f8b8a34ca965048afc9a7a2` |
| `rerun_detection.py` | `9dbce506d6265ab5cd27b707c0b1fdfd3bedae94b990ff0a8fdbcc2e538093fb` |
| `track_many.py` | `aa57e0e906df8b16cd4b7e61dea661a160ad380121ab6ee1a43c083b899f738c` |

The sheets are reproducible from those files: `pipeline_v3.py --run <the pipeline_v3 run> --scenes
explore200_scenes.txt` (with the sizing of `50fe543e72dfe5c13b8e472ff1dc418a17f0677c`) and then `dedup_sheets.py`
rebuild every file of the registered sheets byte for byte, checked before this was written.

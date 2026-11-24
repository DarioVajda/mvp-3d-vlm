# Pre-registration: does pipeline_v3's sheet help relative-direction questions?

Written on 2026-11-22, before any of the questions below was run in any condition. The test and its decision rule
are fixed here; nothing in them changes after results are seen. The notebook that runs it is `confirm_v3.ipynb`.

## Why this test

On two question sets, pipeline_v3's sheet lowered the overall score but raised relative direction:

| Set | Relative-direction questions | Video only | pipeline_v3 | Helped / hurt |
|---|---:|---:|---:|---:|
| explore200 (`explore200_ids.txt`) | 25 | 48.0 | 64.0 | 7 / 3 |
| dev200 (`dev200_ids.txt`) | 25 | 31.5 | 43.5 | 4 / 1 |

Neither is significant on its own, and the idea of using the sheet for relative direction came from explore200's
results. This test uses questions that no condition has run.

## Hypothesis

On VSI-Bench relative-direction questions, giving Qwen3.5-2B pipeline_v3's scene sheet with the video raises the
score over the video alone.

## Questions

`confirm_ids.txt`: every relative-direction question (easy, medium and hard) on the 25 scenes of
`explore200_scenes.txt` that is not in `explore200_ids.txt`, `explore_ids.txt` or `dev200_ids.txt`: 177 questions
(41 easy, 69 medium, 67 hard). The scenes were used by explore200; these questions were not. The file also holds the 3
route-planning questions that no question set has used and that have pipeline_v3 sheets (5021, 5062, 5082, on dev
scenes); they are reported apart (below).

## Conditions and settings

- **`A_video`**: the video only.
- **`pipeline_v3`**: the video and pipeline_v3's sheet, csv, approximately in metres with the scale from typical object
  sizes of its unmerged rows; no "Not detected" line and no other fix.

Both: Qwen/Qwen3.5-2B, thinking off, 16 frames sampled `uniform128`, up to 2,048 new tokens, the final answer scored as
VSI-Bench scores it; the code at `b381205` with `explore_sheets.py` and `scene_scale.py` as below, as in
`explore200_v3.ipynb`.

## Inputs (SHA-256)

| File | SHA-256 |
|---|---|
| `confirm_ids.txt` | `bf4c774c2eeefa7cc99e34aa40cfc13ad7751e11b6728176a98dc48b6b86706f` |
| `scene_sheets_pipeline_v3_75.zip` (pipeline_v3's sheets of the 75 scenes; its explore200 scenes' sheets are the ones `explore200_v3.ipynb` ran on) | `207af4fed1d5a4335f768686f449c239907a126aaeea22ba7a57484b136ace74` |
| `explore_sheets.py` | `7c82e4ad337e0ad00c41c4efa43b517c644b79ac119a2e32f0328fe6742379dc` |
| `src/mvp/benchmarks/scene_scale.py` | `4ed52ca3cc1e489a762bc4c43a181c58ff19f48f8b64b6caa3771b5c8fcb4d53` |

## Primary test and decision rule

The 177 relative-direction questions, paired by question. A question is *helped* if pipeline_v3's score is higher than
A_video's and *hurt* if it is lower (relative direction is multiple choice, so each score is 0 or 1). Exact two-sided
sign test on helped against hurt (questions with equal scores left out).

**Confirmed** if p < 0.05 and helped > hurt. **Not confirmed** otherwise. There is one primary test, so no correction
for multiple tests.

## Secondary (reported, not part of the decision)

The difference in VSI-Bench's relative-direction score (the mean of the easy, medium and hard means), pipeline_v3
minus A_video, with a 95% paired bootstrap interval: 10,000 resamples of the questions within each of the three
sub-types, seed 0. Also each sub-type's scores and helped / hurt counts.

## Route planning: descriptive only

The 3 route-planning questions are reported as they come out, scores and helped / hurt. A sign test on at most 3
changed answers cannot reach p < 0.05 (its smallest two-sided p is 0.25), so they confirm or refute nothing.

## What may not change

The questions, the conditions, the settings, the inputs above, the test and the decision rule. A run that stops
before answering every question is continued or run again with the same inputs; its answers are kept as they come.
No question is dropped, and no other analysis replaces the primary one. Further analyses are labelled exploratory.

## Power

From the two sets above (11 helped, 4 hurt among 50 questions), 30% of answers change and 73% of the changes are
helps. At that rate the 177 questions give about 53 changed answers, and the sign test (two-sided, 0.05) detects the
effect with probability 0.92 (exact binomial calculation). The earlier sets' selection makes a smaller true effect
likely: with 65% of changes helps the power is 0.55, with 60% it is 0.26. A "not confirmed" result therefore does not
show that the sheet has no effect on relative direction, only that this test did not detect one.

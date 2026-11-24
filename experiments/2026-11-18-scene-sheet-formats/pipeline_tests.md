# Pipeline sheets against video only: the pre-registered tests

After the format experiment (README, Results), the pipeline sheet was improved on dev scenes and then tested on
questions that no condition had run, each test pre-registered before its run. This file gives the registered results
first, exactly as each pre-registration defines them, and then the exploratory analyses, which are not tests of any
hypothesis. The analyses are in `analyses/` and the predictions in `results/` (`analyses/README.md`).

Every run: Qwen/Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, thinking off, 16 frames sampled
`uniform128`, up to 2,048 new tokens, the final answer scored as VSI-Bench scores it, the code at `b381205` with the
files each pre-registration lists. A question is *helped* if the sheet condition scores higher than video only and
*hurt* if lower.

## Summary

| Test | Pre-registration | Questions | Registered result | Verdict |
|---|---|---:|---|---|
| pipeline_v3, relative direction | `prereg_v3_relative_direction.md` | 177 | 28 helped / 21 hurt, sign test p = 0.39 | not confirmed |
| full method (pipeline_v3_dedup_relations), H1 distance | `prereg_full_method.md` | 245 | 49 / 75, p = 0.024 | not confirmed (significant harm) |
| full method, H2 overall | `prereg_full_method.md` | 538 | −3.70 [−8.15, +0.71] | not supported |
| pipeline_v4, H1 distance | `prereg_batch2_A.md` | 309 | 53 / 89, p = 0.0032 | not confirmed (significant harm) |
| pipeline_v4, H2 relative direction | `prereg_batch2_A.md` | 110 | 20 / 14, p = 0.39, Holm 0.78 | not supported |
| full method (v4), H3 overall | `prereg_batch2_A.md` | 798 | +0.28 [−2.42, +2.97], p = 0.86, Holm 0.86 | not supported |

No version of the pipeline sheet has improved on video only on held-out questions. On distance questions it has made
the answers significantly worse in both tests that registered it as the primary hypothesis.

## Test 1: pipeline_v3 on relative direction (`confirm_v3`)

**Registered** (`prereg_v3_relative_direction.md`): 177 relative-direction questions on the 25 scenes of
`explore200_scenes.txt` that no question set had used (`confirm_ids.txt`). Primary: exact two-sided sign test,
confirmed if p < 0.05 and helped > hurt.

- 28 helped, 21 hurt, 128 unchanged; p = 0.392. **Not confirmed.**
- Secondary, reported only: relative direction 42.3 (video) against 46.8 (pipeline_v3), +4.44, 95% bootstrap
  [−4.04, +12.71] in the notebook's `preregistered_test.csv`.
- `analyses/confirm_analysis.md` recomputes the same counts and p from the predictions. Its own bootstrap gives
  [−3.97, +12.89]; the notebook's interval is the registered one.

**Deviation:** this pre-registration was written before the run, as its text says, but it is committed only now, with
this write-up, so git does not attest when it was written. Its hashes for `confirm_ids.txt` and
`scene_sheets_pipeline_v3_75.zip` match the files used; `explore_sheets.py` has changed since (`7c82e4ad…` then).

**Exploratory** (`analyses/confirm_analysis.md`): most of the helped questions are ones where video answered "A" and
was wrong (23 of 28). Video answers "A" in 87 of 177 questions against 59 in the truth. Part of the sheet's effect
may be undoing that bias rather than adding spatial information.

## Test 2: the full method with pipeline_v3_dedup_relations (`final`)

**Registered** (`prereg_full_method.md` and its addendum): 538 questions on the same 25 scenes not used before
(`final_ids.txt`): 122 absolute distance, 123 relative distance, 128 object size, 129 appearance order and 36
counting. The full method is the deduplicated pipeline_v3 sheet with precomputed relations, with video's answer for
object and room size.

- **H1 (primary), 245 distance questions:** 49 helped, 75 hurt, 121 unchanged; exact two-sided sign test
  p = 0.0244. **Not confirmed**, because the rule needs helped > hurt; the difference is significant in the direction
  of harm.
- **H2 (secondary), all 538 questions:** VSI-Bench overall 47.97 (video) against 44.27 (full method), −3.70, 95%
  stratified bootstrap [−8.15, +0.71]. **Not supported.**

`analyses/final_analysis.md` recomputes both from the predictions and matches the notebook's
`preregistered_test.csv`. No deviation beyond the addendum, which replaced the notebook before the run.

**Exploratory** (`analyses/final_analysis.md`, `analyses/relation_fixes.md`, `analyses/box_fixes.md`):

- Absolute distance with a precomputed distance (72): the precomputed distance was too short (median ratio to the
  truth 0.64, quartiles 0.33–0.82), the model repeated it in all 72, and MRA fell from 36.8 to 27.4.
- Relative distance with a precomputed closest object (42): it was right in 16, and the model answered it in all 42.
- Offline, on dev scenes only, a median of three scale estimates (typical object sizes, room height and door height)
  and boxes trimmed to the 10th–90th percentile reduced the bias. These became pipeline_v4. They were chosen on dev
  scenes; the final test's scenes were used only as a diagnostic, never to choose.

## Test 3: pipeline_v4 on batch2_A (`final2`)

**Registered** (`prereg_batch2_A.md` and addenda 1 and 2): 798 questions on the 38 scenes of `batch2_A_scenes.txt`,
from a VGGT export none of whose scenes had been used (`batch2_A_ids.txt`). pipeline_v4 is the deduplicated
pipeline_v3 sheet in metres with the median scale, plus relations: distances between trimmed boxes, and the direction
the rows imply. The full method uses pipeline_v4's answer for distance, direction and route planning and video's for
the rest. Holm correction across H2 and H3.

- **Completeness:** 798 answers per condition, exactly the questions of `batch2_A_ids.txt`, no missing scores, the
  same frames in both conditions, and one model snapshot before and after the runs.
- **H1 (primary), 309 distance questions:** 53 helped, 89 hurt, 167 unchanged; exact two-sided sign test p = 0.0032.
  **Not confirmed**; the difference is significant in the direction of harm.
- **H2, 110 relative-direction questions:** 20 helped, 14 hurt; p = 0.392, Holm 0.783. **Not supported.**
- **H3, all 798 questions:** overall 41.19 (video) against 41.46 (full method), +0.28, 95% stratified bootstrap
  [−2.42, +2.97], bootstrap p = 0.861, Holm 0.861. **Not supported.**

`analyses/final2_analysis.md` recomputes all three from the predictions and matches the notebook's
`preregistered_test.csv`.

**Deviations:**

- **Before any run (addendum 1):** the adapter's frame-0 check was loosened from 1e-2 to 2e-2 so that
  `arkitscenes/42897564` could be adapted.
- **Before any run (addendum 2):** the sheets zip and the notebook were pinned by hash.
- **During the run:** the tracking run disconnected once on Colab and resumed. Every scene was checked complete before
  the sheets were built.
- **Not recorded:** the notebook's check that its inputs match the registered hashes stops the run if they don't, but
  its printed line is not in the saved outputs.

**Exploratory** (`analyses/final2_analysis.md`), video against pipeline_v4 by question type:

| Question type | n | Video | pipeline_v4 | Helped / hurt |
|---|---:|---:|---:|---:|
| Absolute distance | 150 | 31.8 | 25.7 | 35 / 69 |
| Relative distance | 159 | 45.3 | 44.0 | 18 / 20 |
| Relative direction (easy / medium / hard) | 24 / 48 / 38 | 58.3 / 50.0 / 34.2 | 62.5 / 47.9 / 50.0 | 20 / 14 |
| Route planning | 28 | 32.1 | 35.7 | 3 / 2 |
| Object size | 157 | 55.5 | 31.3 | 26 / 90 |
| Room size | 38 | 43.2 | 34.7 | 14 / 17 |
| Counting | 88 | 30.0 | 30.2 | 21 / 23 |
| Appearance order | 68 | 44.1 | 38.2 | 8 / 12 |

With and without precomputed relations (a question gets them only when the sheet has every object it names):

| Questions | n | Video | pipeline_v4 | Helped / hurt |
|---|---:|---:|---:|---:|
| Absolute distance, with relations | 80 | 31.9 | 29.8 | 25 / 37 |
| Absolute distance, without | 70 | 31.7 | 21.1 | 10 / 32 |
| Relative distance, with relations | 25 | 52.0 | 32.0 | 2 / 7 |
| Relative distance, without | 134 | 44.0 | 46.3 | 16 / 13 |
| Relative direction, with relations | 45 | 48.9 | 57.8 | 11 / 7 |
| Relative direction, without | 65 | 44.6 | 47.7 | 9 / 7 |

The relations against the truth:

- **Absolute distance (80):**
  - The scale bias of test 2 is gone: the median precomputed/true distance is 0.99.
  - The spread is wide, though: quartiles 0.52–1.50, and only 21 of 80 within 25% of the truth.
  - Their MRA is 29.6, below video's 31.9 on the same questions.
  - The model copied all 80.
- **Relative distance (25):** the precomputed closest object was right in 8; the model copied all 25.
- **Relative direction (45):** the implied direction was right in 26; the model copied 39.

How this reads, without any test behind it:

- **The relations:** the model copies the precomputed relations, so they replace video's estimate whenever they exist.
  Unbiased but noisy distances are still worse than video's.
- **The sheet itself:** absolute distance also falls without relations (31.7 to 21.1), so seeing the coordinates
  seems to pull the model's own estimate away from the video's.
- **Dev results did not carry over** to new scenes: on 50 dev200 distance questions pipeline_v4 scored 50.8 against
  43.2 (absolute) and 56.0 against 48.0 (relative) (`results/dev200_v4`), but not on batch2_A.
- **Direction** is mildly positive in both tests that had direction questions (tests 1 and 3) and significant in
  neither.

## What is left

`batch2_B_scenes.txt` is reserved by `prereg_batch2_A.md` to replicate pipeline_v4 unchanged. Pooling batch2_A and
batch2_B is allowed only as a stated secondary analysis. Any changed method would need a new pre-registration and
questions no condition has run.

## Dev results (descriptive; chosen and analysed on the same questions)

| Run | Questions | Video overall | Sheet condition overall | Analysis |
|---|---|---:|---:|---|
| `explore200_v3` | 200 | 48.6 | pipeline_v3 45.2 | `v3_analysis.md` |
| `dev200` | 200 | 42.0 | pipeline_v3 36.1, v3_fixed 36.4 | `dev_analysis.md` |
| `dev200_dedup` | 200 | 42.0 | pipeline_v3_dedup 37.8 | `dedup_analysis.md` |
| `dev200_relations` | 200 | 42.0 | pipeline_v3_dedup_relations 44.2 | `relations_analysis.md` |
| `dev200_v4` | 50 distance | 45.6 | pipeline_v4 53.4 | (`results/dev200_v4`) |

# Pre-registration: pipeline_v4 against video only, on batch2_A

Written on 2026-11-22, before any question of batch 2 was run in any condition and before any of its answers was
read. The method, the questions, the tests and the decision rules are fixed here; nothing in them changes after results
are seen. The notebook that runs the test is `final_test2.ipynb` (made by the same generator as `final_test.ipynb`).

## Why this test

The test of `prereg_full_method.md` (pipeline_v3_dedup_relations against video only, 245 distance questions on
explore200's scenes) was not confirmed: 49 helped, 75 hurt, p = 0.024. In its exploratory analysis the model repeated
the precomputed relation in every question that had one (72 of 72 distances, 42 of 42 closest objects), and the
precomputed distances were short of the truth (median 0.64 of the true distance). That analysis motivated looking at
the relations' geometry; every choice below was then made on the 50 dev scenes only (`dev_scenes.txt`), scoring the
relations offline against the truth, with no model run, and nothing was fitted on explore200's or batch 2's scenes.
On all 512 distance questions of the dev scenes (167 absolute and 79 relative with relations):

| Relations | Absolute distance MRA | Median predicted / true | Relative distance: closest object right |
|---|---:|---:|---:|
| pipeline_v3's (full-extent boxes, priors scale) | 39.0 | 0.74 | 65.8% |
| median scale, full-extent boxes | 41.1 | 0.80 | 65.8% |
| median scale, 10th-90th percentile boxes (this method) | 47.6 | 0.99 | 63.3% |

Trimmed boxes made object and room sizes worse on the dev scenes (object-size MRA 47.6 to 24.0, room size 40.4 to
15.9), so they are used for the distance relations only, and the shown sheet keeps the full-extent boxes. On dev200
(25 questions per type), the sheet with relations scored below video on object size, room size and appearance order,
and no better on counting (35.6 against 35.2), so those four are answered by video alone.

A sanity check ran before this was written (`dev200_v4.ipynb`, the 50 distance questions of `dev200_ids.txt`): the
model repeated pipeline_v4's relation in all 12 absolute- and all 13 relative-distance questions that had one, and every
relation was rendered at the end of its sheet. It checked rendering and copying only; its scores were not used and
nothing was changed after it.

## The method (frozen)

- **Detection and tracking**: pipeline_v3's, unchanged (`rerun_detection.FROZEN["pipeline_v3"]`): SAM 3
  (`facebook/sam3` at `3c879f3`) with the fixed prompts, tracking from score 0.4, the point merge at 0.5 m, rows of
  tracks seen in one frame dropped below score 0.5, sizes as `scene_sheet_creator.py` at
  `50fe543e72dfe5c13b8e472ff1dc418a17f0677c`.
- **Sheet**: pipeline_v3's sheets (`pipeline_v3.py`) deduplicated by `dedup_sheets.py` (for direction and distance
  questions, one row per named object: the one seen in the most frames, ties broken for distance questions by nearness
  to the other named objects, then track score, then the larger box), as before.
- **Scale** (`relations_v4.py`): per scene, the median of the priors scale (`scene_scale.py`, typical object sizes of
  the unmerged rows), the room-height scale (2.5 m over the floor-to-ceiling extent of the 16 frames' confident depth
  along gravity, 1st to 99th percentile) and the door-height scale (2.0 m over the largest door row's extent along
  gravity, where the scene has a door row; with none, the median of the other two). Every pipeline sheet of the scene
  is multiplied by it. Nothing is fitted to gold.
- **Relations** (`relations_v4.py`, appended by `explore_sheets.relation_text` mode "v4"), only when the shown sheet
  has a row for every object the question names:
  - absolute and relative distance: the closest-point distances between the named objects' rows, with their boxes from
    `trimmed_sheets.py` (the same tracks, frames and centres; boxes from the 10th-90th percentile of each object's
    points), in the same scale; and, for relative distance, the closest candidate;
  - relative direction: the direction the shown rows' positions imply, as in pipeline_v3_dedup_relations.
- **Which answer is used (the full method)**: pipeline_v4's for absolute distance, relative distance, relative direction
  and route planning; video only for every other type (object size, room size, counting, appearance order).

**Two wording points, kept as they were in pipeline_v3_dedup_relations**: the sheet's preamble still says positions
are "approximately in metres, estimated from typical object sizes", although the scale now also uses room and door
height; and the relations' first line still says they are "from the sheet above", although the distance relations use
the trimmed boxes, not the sizes the sheet shows.

## Questions

Batch 2 is the 75 scenes of `vggt_vsibench_batch2.zip` (sha256 `330053dd…a22dda`; its depth add-on
`depth_vsibench_batch2.zip`, `6ef323ad…668ffc`), none of them among the 75 scenes used so far. It was split before any
of its questions was run: within each dataset, a random permutation of the sorted scene names
(`numpy.random.default_rng(0)`, the datasets in alphabetical order, one generator), the first half (rounded up) to A.

- **batch2_A** (`batch2_A_scenes.txt`, 38 scenes: 15 ARKitScenes, 13 ScanNet, 10 ScanNet++): every VSI-Bench question
  on them, sorted by id (`batch2_A_ids.txt`): 798 questions, of which 150 absolute distance, 159 relative distance,
  110 relative direction (24 easy, 48 medium, 38 hard), 28 route planning, 157 object size, 38 room size, 88 counting,
  68 appearance order.
- **batch2_B** (`batch2_B_scenes.txt`, 37 scenes: 15, 12, 10) is reserved for a replication with this method unchanged.
  None of its questions is run before this test's verdict.

**Portrait imbalance**: 4 of batch 2's 5 portrait ARKitScenes videos (518×392 frames) fell into A by the random split,
1 into B. The split was not stratified by orientation.

## Conditions and settings

- **`A_video`**: the video only.
- **`pipeline_v4`**: the video, the sheet (csv) and the relations above.

Both: Qwen/Qwen3.5-2B (the Hugging Face model as served when the notebook runs; the revision is recorded in
`model_revision.json`), thinking off, 16 frames sampled `uniform128`, up to 2,048 new tokens, the final answer scored as
VSI-Bench scores it; the eval code at `b381205` with `explore_sheets.py` and `scene_scale.py` as below. Both conditions
answer all 798 questions; the full method is assembled from their answers as above.

## Hypotheses

- **H1 (primary)**: on distance questions, pipeline_v4 raises the score over the video alone.
- **H2 (secondary)**: on relative-direction questions, pipeline_v4 raises the score over the video alone.
- **H3 (secondary)**: the full method raises VSI-Bench's overall score over the video alone.

## Primary test and decision rule

The 309 distance questions (absolute and relative), paired by question. A question is *helped* if pipeline_v4's score
is higher than A_video's and *hurt* if it is lower (absolute distance scored by MRA, relative distance 0 or 1). Exact
two-sided sign test on helped against hurt, equal scores left out.

**H1 confirmed** if p < 0.05 and helped > hurt; **not confirmed** otherwise. One primary test, no correction.

## Secondary tests (Holm across H2 and H3)

- **H2**: the 110 relative-direction questions, pipeline_v4 against A_video, exact two-sided sign test as above.
- **H3**: the full method against A_video on all 798 questions: the difference in VSI-Bench's overall score (the mean
  over the tasks present of each task's mean score; relative direction's three types form one task), with a 95%
  paired bootstrap interval (10,000 resamples of the questions within each question type, seed 0) and a two-sided
  bootstrap p = 2 × min(share of resampled differences ≤ 0, share ≥ 0), capped at 1.

Holm at 0.05 across the two: the smaller p is compared with 0.025 and, only if it is below, the larger with 0.05. A
secondary hypothesis is **supported** if Holm rejects it and its effect is positive (helped > hurt; difference > 0).
The secondary results do not change the primary verdict.

## Replication and pooling

batch2_B is run afterwards with everything above unchanged (its own ids file, the same pipeline, notebook and
analysis), as a replication of H1 to H3. A pooled analysis of A and B together (the same tests on all of batch 2's
questions) is a stated secondary analysis, reported as such; it does not change either verdict.

## Descriptive (no decision)

Per question type: each condition's score, the full method's, and helped / hurt. Anything else, including any
comparison with the earlier tests, is labelled exploratory.

## Power

The share of distance questions whose score changes is assumed to be about half (154 of 309), as with
pipeline_v3_dedup_relations on explore200 (124 of 245). The sign test (two-sided, 0.05) then confirms H1 with
probability 0.96 if 65% of the changes are helps, 0.68 at 60% and 0.22 at 55%. For H2 (about 55 changed of 110) the
power is 0.81 at 70%, 0.53 at 65% and 0.25 at 60%, before Holm. A "not confirmed" result does not show that the
method has no effect.

## How batch2_A's sheets are made

1. Per scene, the 16 frames, cameras and adapter report from Dario's export, as for the first 75 scenes
   (`adapt_vggt.py`, `subset:uniform16`, the dense depth), and `depth.npz` cut to those frames.
2. Per question, the SAM 3 prompts by `prompt_object_extractor.py` (spaCy 3.8 with `en_core_web_md` 3.8.0, NLTK
   WordNet), as for the first 75 scenes, run on Colab's CPU; `index.csv` and each sheet folder's `question.json`.
3. SAM 3 and tracking (`rerun_detection.ipynb`, the frozen `pipeline_v3` run) on Colab.
4. `pipeline_v3.py`, `dedup_sheets.py`, `trimmed_sheets.py` and `relations_v4.py`, locally (CPU), and the zip
   `scene_sheets_pipeline_v4_batch2_A.zip`.

No step reads an answer. Before any question is run, an addendum records the SHA-256 of that zip and of
`final_test2.ipynb`; nothing else may change.

## What may not change

The method, the questions, the conditions, the settings, the files below, the tests and the decision rules. A run that
stops before answering every question is continued or run again with the same inputs; its answers are kept as they
come. No question is dropped.

## Files (SHA-256)

| File | SHA-256 |
|---|---|
| `batch2_A_ids.txt` | `ed82abf7dc74179b57495e79097c6f015f720125bcacbf07d724302462409604` |
| `batch2_A_scenes.txt` | `fcd068b97c81eef07998216d2bba170834b4c130fcd0cb05b12838f156fd822c` |
| `batch2_B_scenes.txt` | `ab2b18ff286a3a5c18dc3fcd6727e759324b0015772e1d7fdb487401d34164f2` |
| `explore_sheets.py` (rendering, `relation_text` mode "v4") | `2ee6d17214abab0acea31ebd71160880e51e7981f7cded4ae90c3eaaef10b4eb` |
| `src/mvp/benchmarks/scene_scale.py` | `4ed52ca3cc1e489a762bc4c43a181c58ff19f48f8b64b6caa3771b5c8fcb4d53` |
| `relations_v4.py` | `5d31aea837af51a3fe7d99cc7f2999f42c079d198799a74a665ec3bc8466128c` |
| `trimmed_sheets.py` | `821a87c8e5ba4208a4e9749b7c0d1f7c0446557f53f6f5f49f2a34e16a9c01e9` |
| `pipeline_v3.py` | `90b4d7e6f635dd4fabe8bb9011fbd42d428bfaf3004027cfcd52e2cf5cedb933` |
| `merge_points.py` | `8653826912a49c3f11f8f86409e48501d49a19a09674022f7f433635236f9386` |
| `dedup_sheets.py` | `10356327c29aa7ce2d04338a3d0a4dfc6846be483f8b8a34ca965048afc9a7a2` |
| `rerun_detection.py` | `9dbce506d6265ab5cd27b707c0b1fdfd3bedae94b990ff0a8fdbcc2e538093fb` |
| `track_many.py` | `aa57e0e906df8b16cd4b7e61dea661a160ad380121ab6ee1a43c083b899f738c` |
| `vggt_vsibench_batch2.zip` | `330053dd20619f89cb4ccd905e6a983a3a2a37b09d414b39eafde52077a22dda` |
| `depth_vsibench_batch2.zip` | `6ef323addc2e2496bfbfdabadfa024bedf0d5fdce2dcbd65efd4dbb6f2668ffc` |

`batch2_A_ids.txt` is every VSI-Bench question whose `<dataset>/<scene>` is in `batch2_A_scenes.txt`, sorted by id.

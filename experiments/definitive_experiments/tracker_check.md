# Tracker check: the 2026-11-24 tracker against pipeline_v4's frozen tracker, at 16 frames

Instructions for a Claude session. Run this after the methodology in `README.md` (this folder) is settled, and after
two items of its section 5 exist:

- item 3, the pipeline_v4.1 build script;
- item 4, the eval's support for pipeline_v4.1's sheets.

It decides one thing: which tracker the 288-scene pipeline_v4.1 build uses. It is a choice made on development scenes,
not a test.

## Why

pipeline_v4.1 (`README.md`, 2.5) is pipeline_v4's design at 16 frames, with the tracker fixed on 2026-11-24:

- 3D voxel-overlap association in place of centroid association;
- the mask-relative depth rule in place of the global depth filter.

Those fixes came from a diagnosis on the first 75 scenes. At 16 frames, the same object's centre moves a median 0.65 m
between consecutive frames. The centroid radius catches 16% of same-object pairs, and a 10% voxel overlap at 10 cm
catches 56%. The global depth filter dropped 38% of SAM 3's detections of named classes.

But the fixed tracker has never run with v4's post-processing at 16 frames: the point merge, the dedup, the trimmed
boxes, the median scale and the relations. v5, the same tracker at 128 frames, failed its stage-1 go / no-go on box
sizes (12.3% of boxes over 1.5× gold, against a limit of 10.1%). This check compares the two trackers with everything
else equal.

## The two arms

| Setting | `fixed` (pipeline_v4.1 as planned) | `frozen` (pipeline_v4's registered tracker) |
|---|---|---|
| Association | 3D voxel overlap, `question_scene_sheet.OVERLAP_OPTIONS` as at the commit (provisional, set at 128 frames; used as they are) | centroid: within `near_frac` 0.05 of the first frame's diagonal |
| Depth rule | mask: a mask's pixels with confidence at least the lower of the sequence's 55th percentile and the mask's own median (`mask_conf_quantile` at its default) | global: the sequence's 55th percentile |
| `min_points` | 300 | 300 |
| Re-identification | on, the branch's settings for overlap association | on: `reid_sim` 0.6, `reid_radius_frac` 0.15 |
| Detection score for tracking | 0.4 (patched in as `v5_dev.patch_tracker` does) | 0.4 |
| Track kept if seen in | 1 of the 16 frames | 1 of the 16 frames |

Everything else is identical, and is `README.md` 2.5's table without its Tracking row:

- the 16 frames, and the geometry with dense depth;
- **one SAM 3 run per scene**: its masks are pooled once and both trackers read the same pooled masks;
- the rows, sizes, dedup, upright axes, scale, relations, fields and renderer.

Check that both arms read masks with the same SHA-256.

**The frozen arm can be made in two ways:**

1. The branch's tracker with `--association centroid --depth-rule global`, which `question_scene_sheet.py` documents
   as the original tracking.
2. `rerun_detection.py`'s registered `pipeline_v3` tracking: `milestone4_video.py` unchanged, run through
   `track_many.py`.

Way 1 is expected to equal way 2. That holds by reading the code and by the repository's no-re-id test, but the re-id
path itself has never been compared. So before using way 1, run both on 3 dev scenes and check that the tracks are
identical. If they differ, use way 2 for the frozen arm and say so.

Registered code runs from a `--code` tree made of `git archive b4135e7 src` plus `50fe543`'s
`src/mvp/openvocab_adapter/scene_sheet_creator.py`. `pipeline_v3.py`'s default tree crashes at HEAD.

## Scenes and questions

- **Scenes:** the 50 dev scenes, `experiments/2026-11-18-scene-sheet-formats/dev_scenes.txt`, with 1,813 questions.
  pipeline_v4 was designed on them.
- **Not used:** any other scene.
- **Smoke scene:** `scannet/scene0025_01`, one of the 50.

## Steps

All on one GPU, except the CPU steps.

1. **Code and tests.** Pull the commit `README.md` names and record `git rev-parse HEAD`. Run the build script's tests
   and the eval's (`PYTHONPATH=src python -m unittest mvp.benchmarks.tests.test_scene_sheet`, plus the new tests of
   `README.md` section 5, items 3 and 4). All must pass, with none skipped.
2. **Inputs.**
   - `data/raw/VSI-Bench/test.jsonl` has SHA-256 `aa3ff8cc2010e555f4f865ff96483812ec0b275005b68562f53bb114ad4db9db`.
   - Every dev scene has `depth.npy`, `depth_conf.npy` and `frame_indices.npy` in `data/vggt/<dataset>/<scene>/`.
   - The gold sheets are built and checked as `README.md` 2.4 says (`tasks/openvocab_adapter/out_full`, its two hashes).
3. **Smoke, both arms, on `scannet/scene0025_01`.** Each arm must:
   - finish;
   - read the same masks (same SHA-256);
   - record its association, depth rule and detection score in its tracks' `summary.json`;
   - give one sheet folder per question of the scene (38).
4. **Frozen-arm equivalence**, if way 1 is used: the 3-scene comparison above.
5. **Both arms on the 50 scenes,** into `tasks/openvocab_adapter/tracker_check/fixed` and `.../frozen`.
   - Each must end with 50 of 50 scenes, no failed scene, and an index.csv of the 1,813 questions.
   - If a run stops, run the same command again; finished scenes are skipped.
   - Record each arm's index.csv SHA-256, its empty sheets, its scenes without a floor plane, and each scene's three
     scale estimates with the one that is the median.
6. **Sheet measures (CPU).** Use the measures of
   `experiments/2026-11-18-scene-sheet-formats/v5_stage1_analysis.py` (`measures`), with `frozen` in pipeline_v4's
   place and `fixed` as the one configuration. If the script only takes `dev_scenes_15.txt`, add a scenes option
   without changing any measure.
   - **Sheet quality, against gold:**
     - named objects found;
     - rows per real object;
     - count equal to gold;
     - the box's longest side over gold's (median), and the share over 1.5×;
     - the closest-point gap error;
     - the position error after one similarity alignment per scene, and how many scenes it aligns.
   - **Relations, against VSI-Bench's answers:**
     - absolute distance: MRA, and the share of questions with a relation;
     - relative distance: the share whose stated closest object is the answer;
     - each also paired, on the questions both arms have a relation for.
   - **The simulated distance score:** the relation's answer where there is one, video only's otherwise. For video only,
     use Qwen3.5-4B's run R1 on these questions if it exists. Otherwise use the 2B's
     `results/zeroshot/vsibench/gold_sheet_relations_full/qwen3_5_2B_A_video.jsonl`, and say which.
7. **Model runs:** Qwen3.5-4B, with `README.md` 2.1–2.2's settings, on the 1,813 questions (`--ids` a file of them,
   sorted).
   - Four runs, two per arm:
     - B, video + pipeline sheet in csv (`--sheet-source pipeline`);
     - C, video + pipeline sheet + relations in csv (`--sheet-source pipeline_relations`).
   - Before them, the prompt check and the 20-question smoke test, as `README.md` 2.8 says. The smoke test records the
     model revision.
   - Outputs: `results/zeroshot/vsibench/tracker_check/qwen3_5_4B_<fixed|frozen>_<B|C>.jsonl`. Each run must report
     1,813 questions.
8. **The decision**, by the rule below. Then the report.

## Decision rule

Fixed on 2026-11-27, before any result. Do not change it after any result is seen.

`fixed` is used if all three hold on the 50 dev scenes:

1. **Model score:** its VSI-Bench overall (the 8-task mean, on the 1,813 questions) is at least `frozen`'s, in B and in
   C.
2. **Objects found:** it finds at most 2 points fewer of the named objects.
3. **Box sizes:** its median longest side / gold is at least as close to 1 (in log ratio), and it has at most 5 points
   more boxes over 1.5×.

The verdict is one of three:

- **better:** all three hold, and its overall is higher by at least 2 points in B or C;
- **not worse:** all three hold, without that gain. `fixed` is still used, since it removes the diagnosed causes;
- **worse:** any of the three fails. `frozen` is used.

For every model comparison (fixed − frozen, B and C), also report these. They are descriptive, since the dev scenes
are not a test:

- helped / hurt, with an exact two-sided sign test;
- a paired bootstrap 95% interval: 10,000 resamples within question type, seed 0;
- the per-task scores.

## What to write and report

- **The results file:** `experiments/definitive_experiments/tracker_check_results.md`, containing:
  - the commit;
  - each arm's settings as run;
  - which way the frozen arm was made, and the equivalence check's result;
  - measured times for SAM 3, tracking and each model run;
  - every measure of steps 5–7, in one table per step;
  - the decision, with each criterion's numbers.
- **Commit only:** the results file and `results/zeroshot/vsibench/tracker_check/` (the four `.jsonl` files, their
  `_summary.json` files, `model_revision.txt`).
- **Don't commit the sheets.** Zip each arm's sheets root (no masks, frames or depth), and report each zip's SHA-256 and
  size. Push only if asked.
- **Don't edit `README.md`.** Report the verdict. If it is "worse", `README.md` 2.5's Tracking row changes to the
  frozen tracker, and nothing else does; the user makes that change.

## Stop and report, without changing anything, if

- a hash, count, test, smoke check or the equivalence check fails, or any scene fails;
- running would need a change to a setting above, a tracker option, or a file hash-registered in
  `experiments/2026-11-18-scene-sheet-formats/prereg_batch2_A.md`;
- the measures would need changing to run on 50 scenes (anything beyond a scenes option).

Don't tune anything: no setting may change after a result is seen. Don't estimate times in anything you write; report
measured ones. If you run on the local WSL machine, check every output's SHA-256 after writing it and again before
the analysis. That machine has corrupted written files before (bit 0x20 flips). Redo any step whose output changed.

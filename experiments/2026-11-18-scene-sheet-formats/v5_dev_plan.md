# v5 on the 50 dev scenes: plan (draft)

**v5** = the revamped tracker (`revamped-pipeline`, `72b9a2d`: SAM 3 on all 128 frames, overlap association, the
mask depth rule, tracks kept from 4 of 128 frames, upright axes) with pipeline_v4's prompts (`fixed_v2`: synonyms
pooled, a mask dropped if it overlaps a higher-scoring one of another prompt at IoU ≥ 0.5) and v4's post-processing
ported by `v5_postprocess.py` (point merge, question-conditioned dedup, median scale, trimmed-box relations). Every
step here is on dev scenes, to choose a configuration; nothing is a test. A test of the frozen v5 against video only
comes after, pre-registered, on scenes no condition has used.

## What the smoke scene (scene0025_01) showed

- The port reproduces the branch's own sheets exactly before post-processing.
- The point merge joined nothing: the branch's tracker already merges overlapping tracks, and over 128 frames two
  tracks of one object are usually seen in a common frame, which the merge rule forbids.
- The priors scale fails on the new rows (0.57 m per unit against 2.53 from the room height and 1.91 from the door):
  the boxes are larger (a keyboard 2.35 m long), and `scene_scale.CALIBRATION` was fitted on 16-frame rows. The median
  hides it when the scene has a door; without one, the median of two is their mean.
- Positions after one shared alignment are closer to gold than v3 / v4's (median 0.37–0.86 m against 0.90–1.26 m,
  depending on the rows the alignment is fitted on); sizes are worse (0.36–0.51 m against 0.19–0.24 m).

## Configurations (tracking, from the same masks)

| | Detection threshold | `min_points` (mask rule) |
|---|---|---|
| v5-his | 0.55 | 50 |
| v5-det | 0.4 | 50 |
| v5-pts | 0.55 | 300 |
| v5-ours | 0.4 | 300 |

SAM 3 runs once per scene and saves masks from 0.2, so every configuration is tracking only.

## Step A: masks and tracks (GPU, the cluster)

Run it as the README's section "v5 on the dev scenes" says (`v5_dev.py`, with its checks and stop conditions).

- **Code (done, `v5_dev.py`):**
  - the branch's tracker: `--det-score` (two constants, 0.55) and `tracks/<prompt>.scores.json`, patched into a copy
    (`v5_dev.patch_tracker`), as `rerun_detection.patch_tracker` does for the old tracker;
  - SAM 3 with the `fixed_v2` prompts and synonyms over all 128 frames, pooled per object (`rerun_detection`'s
    pooling), saved from 0.2.
- **Cost:** 685 SAM 3 prompts over the 50 scenes (6–27 per scene, median 13), on 128 frames each. The smoke scene took
  234 s for 8 prompts on an RTX 3070; the dense-tracking task estimates 2 h for the 75 scenes on a B300. Estimate
  1.5–2.5 GPU hours on a B200/B300 for SAM 3, and under an hour for the four trackings (about 10 s a scene each).
  A Colab L4 would take about 5–6 hours, resumable by scene.
- **Back from the cluster** (no masks, frames or depth; the dense depth of all 50 scenes is already local): per scene
  and configuration `all/scene/{tracks/, cameras.npz, adapter_report.json, upright.json}` and every sheet's
  `question.json`. A few hundred MB in all.

## Step B: sheets and scorecard (CPU, local)

1. `v5_postprocess.py` per configuration (minutes), with `--dense-root data/vggt`: the run comes back without depth.
2. `pipeline_scorecard.py` on `dev_scenes_15.txt`: each v5 configuration against pipeline_v4's dev50 sheets and the
   16-frame original: rows per unique object, named objects found, count = gold, and position and size errors after
   alignment. Add a check that each configuration's relation lines pick the gold answer on distance questions.
3. Choose one configuration on `dev_scenes_15.txt`, check it unchanged on `dev_scenes_35.txt` (as pipeline_v3 was
   chosen), and freeze it, with the scale decision below.

Open decisions for Step B:
- **Scale:** keep v4's median of three, refit `CALIBRATION` on the new rows (dev scenes only), or drop the priors.
- **Sizes:** whether `min_points` 300 or a larger `size_trim` brings the boxes back to v4's size errors.
- **Point merge:** keep it (it does nothing here) or drop it.

## Step C: dev200 with the VLM (GPU, one run)

Qwen3.5-2B, thinking off, 16 frames `uniform128`, `--max-new-tokens 2048 --final-answer`, all at one commit on
`dev200_ids.txt`: video only, video + v4 sheet (`v4_eval.py`, as registered), video + v5 sheet (the branch's renderer,
upright axes, nearby visibility, v4's metres preamble, v5 relations), and the full method for each (v4's routing).
About 15 minutes on an L4 per sheet condition. Exploratory: per task, sign tests, input tokens. If v5 beats v4 on dev,
pre-register v5's full method against video only on fresh scenes.

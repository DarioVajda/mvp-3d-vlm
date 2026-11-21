# Pipeline fixes after tracking: merging split objects, and which box sizing to keep

The pipeline sheets split objects: 68% of rows are seen in a single frame, and a class's row count equals gold for
only 27% of (question, class) pairs (`pipeline_scorecard.md`). This compares fixes that run after tracking, on a CPU,
from what the sheet run saved (the tracks, the cameras) and the dense VGGT depth (`data/vggt/<dataset>/<scene>/depth.npy`
and `depth_conf.npy`). All numbers are on the 50 scenes of `dev_scenes.txt` (no question of `explore200_ids.txt`),
scored by `pipeline_scorecard.py` against gold.

## What was compared

- **Current:** `data/scene_sheets_75`, made before `980cf00`. Its boxes are what each object's points span along the
  world axes (the first camera's), between the 2nd and 98th percentiles (`size_trim` 0.02). Rebuilding the dev scenes
  from the saved tracks and the dense depth with `scene_sheet_creator.py` as of `50fe543` (`rebuild_scene_sheets.py`)
  gives all 1,665 sheets byte for byte, so the rebuild is exact.
- **Box merge:** split rows of one class joined on the finished sheets, from their boxes: rows never seen in the same
  frame, whose centres are within 0.5 m (the priors scale, `mvp.benchmarks.scene_scale`) or whose boxes overlap, the
  closest pair first. The joined box is the box around both. Two settings: every class at 0.5 m; and overlapping rows
  only, a joined box no longer than 3× the class's typical size (`scene_scale.PRIORS`), the larger box kept when the two
  overlap, classes without a typical size not merged.
- **Point merge** (`merge_points.py`): the same joining rule applied to the tracks before their rows are computed, so a
  merged object's centre and box are computed from the union of its tracks' 3D points. Every class at 0.5 m, and with
  the 3× cap.
- **Oriented sizing:** today's `scene_sheet_creator.py` (since `980cf00`), whose box is the smallest one around the
  points in any orientation (`fit_box`), its sides reported along the axes nearest x, y and z. Rows and centres are the
  same as the current sheets'; only sizes change. Alone, and with the point merge.

The scorecard's sizes, distances and fixed rules use the priors scale, whose calibration constant depends on how boxes
are sized. For oriented sizing it is refitted with `metric_scale.py`'s fit (the median of true / prior scale over the
24 reliable scenes outside explore200's scenes): **0.8756**, against 0.8417 for the current sizing (fallback 2.293
either way). Without the refit, oriented sizing looks 2–3 points worse on distances, from the scale alone. Both
constants are fitted on dev scenes and scored on them, so both are slightly optimistic in the same way.
`scene_scale.py` keeps 0.8417 until the sizing is changed.

## Results

| Measure | Current | Box merge 0.5 m | Box merge overlap, 3× | Point merge 0.5 m | Point merge 0.5 m, 3× | Oriented sizing | Oriented + point merge |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count = gold, per (question, class) | 27.1% | 36.3% | 32.3% | 34.7% | 31.0% | 27.1% | 34.7% |
| Unmatched rows (beyond the gold sheet's count) | 53.4% | 39.4% | 46.0% | 41.4% | 46.6% | 53.4% | 41.4% |
| Rows seen in only one frame | 67.6% | 49.4% | 56.6% | 51.3% | 57.1% | 67.6% | 51.3% |
| Size error, median relative | 30.0% | 33.1% | 29.5% | 31.6% | 31.9% | 29.1% | 29.2% |
| Distance error, median relative | 22.1% | 26.0% | 22.9% | 24.6% | 24.5% | 22.9% | 26.0% |
| Rule: counting | 23.9 | 36.4 | 29.9 | 34.0 | 27.5 | 23.9 | 34.0 |
| Rule: object size | 29.8 | 28.8 | 29.7 | 28.6 | 28.7 | 31.7 | 31.5 |
| Rule: absolute distance | 16.2 | 15.8 | 15.4 | 15.9 | 15.5 | 18.4 | 17.9 |
| Rule: relative distance | 34.2 | 33.8 | 32.9 | 33.8 | 34.2 | 34.2 | 34.6 |

The oriented columns use the refitted constant. Rebuild times for the 50 scenes on one 8-core laptop: current sizing
52 s, with the point merge 57 s; oriented sizing 437 s, with the point merge 347 s (the orientation search dominates).

## What it shows

- **Refitting from points is better than joining boxes.** At 0.5 m the point merge's sizes are 31.6% off against the
  box merge's 33.1%, and its distances 24.6% against 26.0%, for a slightly smaller count gain (34.7% against 36.3%:
  the decision uses tighter boxes, the points' 2nd–98th percentiles). Either merge still makes sizes worse than none
  with the current sizing.
- **A size cap does not help the point merge.** It stops merging for every class without a typical size, so counts
  fall to 31.0%, and sizes do not improve (31.9%).
- **Oriented sizing is better on everything about size**, at no cost to rows or counts: size error 29.1% against
  30.0%, the object-size rule 31.7 against 29.8, the absolute-distance rule 18.4 against 16.2; distances are about
  equal (22.9% against 22.1%), since centres do not change.
- **Oriented sizing with the point merge** has the largest count gain with sizes no worse than now: counts 34.7%
  against 27.1%, unmatched rows 41.4% against 53.4%, size error 29.2% against 30.0%. The cost is the distance between
  centres, 26.0% against 22.1%, because pooling separate objects moves centres; the distance rules do not suffer
  (17.9 and 34.6 against 16.2 and 34.2).

## Recommendation

- **Keep the oriented sizing** (today's `scene_sheet_creator.py`), with the priors scale's calibration constant set to
  0.8756 for it. It changes every pipeline sheet's sizes, so results made with the current sheets are not comparable
  with sheets made after the switch; the sheets should be rebuilt once and every later experiment run on them.
- **Add the point merge** on top if counts matter most (counting, the rows a question lists): the largest count gain
  with sizes no worse than now, at the cost of centre distances.
- **If centre distances matter more**, the box merge of overlapping rows with a 3× cap is the balanced choice (counts
  32.3%, size 29.5%, distance 22.9%).
- The merge threshold (0.5 m) comes from the earlier `pipeline_v2` tuning (`explore_sheets.MERGE_THRESHOLD`) and was
  not tuned again here.

## Reproducing

```bash
E=experiments/2026-11-18-scene-sheet-formats
# the sizing every experiment so far used: a src/ tree with scene_sheet_creator.py from 50fe543
mkdir -p old && git archive HEAD src | tar -x -C old
git show 50fe543:src/mvp/openvocab_adapter/scene_sheet_creator.py > old/src/mvp/openvocab_adapter/scene_sheet_creator.py
PYTHONPATH=src python $E/merge_points.py --sheets data/scene_sheets_75 --out sheets_point_merge \
    --scenes $E/dev_scenes.txt --code old/src                       # current sizing + point merge
PYTHONPATH=src python $E/merge_points.py --sheets data/scene_sheets_75 --out sheets_oriented_point_merge \
    --scenes $E/dev_scenes.txt --code src                           # oriented sizing + point merge
PYTHONPATH=src python $E/merge_points.py ... --no-merge             # a rebuild alone, with either sizing
PYTHONPATH=src python $E/pipeline_scorecard.py --old data/scene_sheets_75 --new sheets_point_merge
```

The box merges were made with `merge_sheets.py` (`--threshold 0.5`; `--cap 3 --threshold 0`). Scoring oriented
sheets as above needs the refitted constant, which `scene_scale.py` does not hold yet.

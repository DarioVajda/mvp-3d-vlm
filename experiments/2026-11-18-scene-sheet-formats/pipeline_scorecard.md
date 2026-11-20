# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `/Users/tamaragrujicic/Documents/GitHub/mvp-3d-vlm/data/scene_sheets_75`. New: `/Users/tamaragrujicic/Documents/GitHub/mvp-3d-vlm/data/scene_sheets_75`. Scenes: `dev_scenes.txt`, 50 scenes, 1,813 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 50 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 74.5% | 74.5% | +0.0 pts |
| questions with every named object | 49.8% | 49.8% | +0.0 pts |
| empty sheets where gold has objects | 138 | 138 | +0 |
| count = gold, per (question, class) | 27.1% | 27.1% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 9,966 | 9,966 | +0 |
| rows seen in only one frame | 67.6% | 67.6% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 53.4% | 53.4% | +0.0 pts |
| size error, median (m) | 0.28 | 0.28 | +0.00 |
| size error, median relative | 30.0% | 30.0% | +0.0 pts |
| distance error, median (m) | 0.50 | 0.50 | +0.00 |
| distance error, median relative | 22.1% | 22.1% | +0.0 pts |
| rule: counting | 23.9 | 23.9 | +0.0 |
| rule: object size | 29.8 | 29.8 | +0.0 |
| rule: absolute distance | 16.2 | 16.2 | +0.0 |
| rule: relative distance | 34.2 | 34.2 | +0.0 |

### ARKitScenes (19 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 71.0% | 71.0% | +0.0 pts |
| questions with every named object | 47.3% | 47.3% | +0.0 pts |
| empty sheets where gold has objects | 41 | 41 | +0 |
| count = gold, per (question, class) | 31.3% | 31.3% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 1,283 | 1,283 | +0 |
| rows seen in only one frame | 79.6% | 79.6% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 47.5% | 47.5% | +0.0 pts |
| size error, median (m) | 0.26 | 0.26 | +0.00 |
| size error, median relative | 25.1% | 25.1% | +0.0 pts |
| distance error, median (m) | 0.74 | 0.74 | +0.00 |
| distance error, median relative | 28.5% | 28.5% | +0.0 pts |
| rule: counting | 30.0 | 30.0 | +0.0 |
| rule: object size | 31.6 | 31.6 | +0.0 |
| rule: absolute distance | 19.0 | 19.0 | +0.0 |
| rule: relative distance | 44.2 | 44.2 | +0.0 |

### ScanNet (17 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 85.9% | 85.9% | +0.0 pts |
| questions with every named object | 65.6% | 65.6% | +0.0 pts |
| empty sheets where gold has objects | 28 | 28 | +0 |
| count = gold, per (question, class) | 27.4% | 27.4% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 5,921 | 5,921 | +0 |
| rows seen in only one frame | 64.1% | 64.1% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 54.7% | 54.7% | +0.0 pts |
| size error, median (m) | 0.32 | 0.32 | +0.00 |
| size error, median relative | 32.8% | 32.8% | +0.0 pts |
| distance error, median (m) | 0.50 | 0.50 | +0.00 |
| distance error, median relative | 22.1% | 22.1% | +0.0 pts |
| rule: counting | 21.1 | 21.1 | +0.0 |
| rule: object size | 31.6 | 31.6 | +0.0 |
| rule: absolute distance | 21.3 | 21.3 | +0.0 |
| rule: relative distance | 37.6 | 37.6 | +0.0 |

### ScanNet++ (14 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 61.0% | 61.0% | +0.0 pts |
| questions with every named object | 30.7% | 30.7% | +0.0 pts |
| empty sheets where gold has objects | 69 | 69 | +0 |
| count = gold, per (question, class) | 23.9% | 23.9% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 2,762 | 2,762 | +0 |
| rows seen in only one frame | 69.6% | 69.6% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 53.5% | 53.5% | +0.0 pts |
| size error, median (m) | 0.19 | 0.19 | +0.00 |
| size error, median relative | 27.0% | 27.0% | +0.0 pts |
| distance error, median (m) | 0.43 | 0.43 | +0.00 |
| distance error, median relative | 20.0% | 20.0% | +0.0 pts |
| rule: counting | 24.3 | 24.3 | +0.0 |
| rule: object size | 26.1 | 26.1 | +0.0 |
| rule: absolute distance | 7.2 | 7.2 | +0.0 |
| rule: relative distance | 26.3 | 26.3 | +0.0 |

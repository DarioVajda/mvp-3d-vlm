# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/original_current`. Scenes: `scenes.txt`, 3 scenes, 82 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 3 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 67.3% | 67.3% | +0.0 pts |
| questions with every named object | 29.1% | 29.1% | +0.0 pts |
| empty sheets where gold has objects | 7 | 7 | +0 |
| count = gold, per (question, class) | 17.5% | 17.5% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 293 | 293 | +0 |
| rows seen in only one frame | 58.4% | 58.4% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 63.5% | 63.5% | +0.0 pts |
| size error, median (m) | 0.42 | 0.42 | -0.00 ✓ |
| size error, median relative | 32.3% | 32.3% | -0.0 pts ✓ |
| distance error, median (m) | 0.43 | 0.43 | +0.00 |
| distance error, median relative | 28.9% | 28.9% | +0.0 pts |
| rule: counting | 0.0 | 0.0 | +0.0 |
| rule: object size | 13.1 | 13.1 | +0.0 |
| rule: absolute distance | 1.1 | 1.1 | +0.0 |
| rule: relative distance | 25.0 | 25.0 | +0.0 |

### ARKitScenes (1 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 56.5% | 56.5% | +0.0 pts |
| questions with every named object | 12.0% | 12.0% | +0.0 pts |
| empty sheets where gold has objects | 3 | 3 | +0 |
| count = gold, per (question, class) | 31.1% | 31.1% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 47 | 47 | +0 |
| rows seen in only one frame | 100.0% | 100.0% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 46.8% | 46.8% | +0.0 pts |
| size error, median (m) | 0.43 | 0.43 | +0.00 |
| size error, median relative | 34.6% | 34.6% | +0.0 pts |
| distance error, median (m) | 1.38 | 1.38 | +0.00 |
| distance error, median relative | 30.3% | 30.3% | +0.0 pts |
| rule: counting | – | – | – |
| rule: object size | 0.0 | 0.0 | +0.0 |
| rule: absolute distance | 0.0 | 0.0 | +0.0 |
| rule: relative distance | – | – | – |

### ScanNet (1 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 78.9% | 78.9% | +0.0 pts |
| questions with every named object | 44.8% | 44.8% | +0.0 pts |
| empty sheets where gold has objects | 1 | 1 | +0 |
| count = gold, per (question, class) | 0.0% | 0.0% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 180 | 180 | +0 |
| rows seen in only one frame | 68.9% | 68.9% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 77.2% | 77.2% | +0.0 pts |
| size error, median (m) | 0.45 | 0.45 | -0.00 ✓ |
| size error, median relative | 33.8% | 33.8% | -0.0 pts ✓ |
| distance error, median (m) | 0.58 | 0.58 | -0.00 ✓ |
| distance error, median relative | 31.7% | 31.7% | -0.0 pts ✓ |
| rule: counting | 0.0 | 0.0 | +0.0 |
| rule: object size | 20.0 | 20.0 | +0.0 |
| rule: absolute distance | 3.3 | 3.3 | +0.0 |
| rule: relative distance | 0.0 | 0.0 | +0.0 |

### ScanNet++ (1 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 64.1% | 64.1% | +0.0 pts |
| questions with every named object | 28.0% | 28.0% | +0.0 pts |
| empty sheets where gold has objects | 3 | 3 | +0 |
| count = gold, per (question, class) | 25.0% | 25.0% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 66 | 66 | +0 |
| rows seen in only one frame | 0.0% | 0.0% | +0.0 pts |
| unmatched rows (beyond the gold sheet's count) | 37.9% | 37.9% | +0.0 pts |
| size error, median (m) | 0.16 | 0.16 | +0.00 |
| size error, median relative | 21.0% | 21.0% | +0.0 pts |
| distance error, median (m) | 0.37 | 0.37 | +0.00 |
| distance error, median relative | 20.0% | 20.0% | +0.0 pts |
| rule: counting | – | – | – |
| rule: object size | 18.0 | 18.0 | +0.0 |
| rule: absolute distance | 0.0 | 0.0 | +0.0 |
| rule: relative distance | 50.0 | 50.0 | +0.0 |

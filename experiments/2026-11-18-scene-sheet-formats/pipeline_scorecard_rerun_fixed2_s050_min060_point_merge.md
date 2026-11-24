# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed2_s050_min060_point_merge`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 76.0% | +3.9 pts ✓ |
| questions with every named object | 46.6% | 54.7% | +8.2 pts ✓ |
| empty sheets where gold has objects | 41 | 36 | -5 ✓ |
| count = gold, per (question, class) | 23.7% | 36.8% | +13.1 pts ✓ |
| rows over the sheet folders (room fragments left out) | 2,276 | 1,854 | -422 |
| rows seen in only one frame | 67.3% | 44.0% | -23.3 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 38.6% | -12.9 pts ✓ |
| size error, median (m) | 0.31 | 0.23 | -0.08 ✓ |
| size error, median relative | 29.3% | 29.0% | -0.3 pts ✓ |
| distance error, median (m) | 0.40 | 0.44 | +0.04 ✗ |
| distance error, median relative | 22.6% | 27.0% | +4.4 pts ✗ |
| rule: counting | 23.9 | 36.1 | +12.1 ✓ |
| rule: object size | 29.7 | 32.1 | +2.4 ✓ |
| rule: absolute distance | 13.8 | 15.4 | +1.5 ✓ |
| rule: relative distance | 35.2 | 29.6 | -5.6 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 66.4% | +0.3 pts ✓ |
| questions with every named object | 45.6% | 46.5% | +0.9 pts ✓ |
| empty sheets where gold has objects | 15 | 15 | +0 |
| count = gold, per (question, class) | 24.1% | 24.1% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 365 | 315 | -50 |
| rows seen in only one frame | 69.0% | 52.7% | -16.3 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 37.8% | -8.5 pts ✓ |
| size error, median (m) | 0.30 | 0.35 | +0.06 ✗ |
| size error, median relative | 26.0% | 30.0% | +4.0 pts ✗ |
| distance error, median (m) | 0.57 | 0.83 | +0.26 ✗ |
| distance error, median relative | 27.8% | 32.7% | +5.0 pts ✗ |
| rule: counting | 32.2 | 32.2 | +0.0 |
| rule: object size | 25.0 | 28.3 | +3.3 ✓ |
| rule: absolute distance | 13.3 | 11.0 | -2.4 ✗ |
| rule: relative distance | 20.0 | 20.0 | +0.0 |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 89.3% | +5.8 pts ✓ |
| questions with every named object | 58.4% | 72.1% | +13.7 pts ✓ |
| empty sheets where gold has objects | 8 | 5 | -3 ✓ |
| count = gold, per (question, class) | 26.1% | 45.0% | +18.9 pts ✓ |
| rows over the sheet folders (room fragments left out) | 1,548 | 1,203 | -345 |
| rows seen in only one frame | 71.5% | 43.3% | -28.2 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 39.2% | -15.3 pts ✓ |
| size error, median (m) | 0.32 | 0.22 | -0.10 ✓ |
| size error, median relative | 32.1% | 29.8% | -2.2 pts ✓ |
| distance error, median (m) | 0.39 | 0.51 | +0.12 ✗ |
| distance error, median relative | 19.4% | 24.6% | +5.2 pts ✗ |
| rule: counting | 11.3 | 41.3 | +30.0 ✓ |
| rule: object size | 36.0 | 38.7 | +2.7 ✓ |
| rule: absolute distance | 22.7 | 22.7 | -0.0 ✗ |
| rule: relative distance | 59.1 | 54.5 | -4.5 ✗ |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 56.9% | +3.4 pts ✓ |
| questions with every named object | 25.8% | 30.6% | +4.8 pts ✓ |
| empty sheets where gold has objects | 18 | 16 | -2 ✓ |
| count = gold, per (question, class) | 18.2% | 31.1% | +12.9 pts ✓ |
| rows over the sheet folders (room fragments left out) | 363 | 336 | -27 |
| rows seen in only one frame | 47.7% | 38.4% | -9.3 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 37.5% | -6.9 pts ✓ |
| size error, median (m) | 0.16 | 0.29 | +0.12 ✗ |
| size error, median relative | 23.6% | 22.7% | -0.9 pts ✓ |
| distance error, median (m) | 0.43 | 0.35 | -0.08 ✓ |
| distance error, median relative | 29.8% | 28.0% | -1.8 pts ✓ |
| rule: counting | 32.1 | 32.9 | +0.7 ✓ |
| rule: object size | 25.2 | 26.5 | +1.3 ✓ |
| rule: absolute distance | 2.1 | 9.2 | +7.1 ✓ |
| rule: relative distance | 17.6 | 5.9 | -11.8 ✗ |

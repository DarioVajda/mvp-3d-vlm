# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed_s020_point_merge`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 83.8% | +11.8 pts ✓ |
| questions with every named object | 46.6% | 66.8% | +20.3 pts ✓ |
| empty sheets where gold has objects | 41 | 20 | -21 ✓ |
| count = gold, per (question, class) | 23.7% | 25.4% | +1.7 pts ✓ |
| rows over the sheet folders (room fragments left out) | 2,276 | 3,836 | +1,560 |
| rows seen in only one frame | 67.3% | 55.9% | -11.4 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 67.0% | +15.5 pts ✗ |
| size error, median (m) | 0.31 | 0.29 | -0.03 ✓ |
| size error, median relative | 29.3% | 28.1% | -1.1 pts ✓ |
| distance error, median (m) | 0.40 | 0.73 | +0.33 ✗ |
| distance error, median relative | 22.6% | 40.2% | +17.6 pts ✗ |
| rule: counting | 23.9 | 20.0 | -3.9 ✗ |
| rule: object size | 29.7 | 33.8 | +4.1 ✓ |
| rule: absolute distance | 13.8 | 14.6 | +0.8 ✓ |
| rule: relative distance | 35.2 | 24.1 | -11.1 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 81.9% | +15.8 pts ✓ |
| questions with every named object | 45.6% | 57.9% | +12.3 pts ✓ |
| empty sheets where gold has objects | 15 | 3 | -12 ✓ |
| count = gold, per (question, class) | 24.1% | 28.2% | +4.1 pts ✓ |
| rows over the sheet folders (room fragments left out) | 365 | 525 | +160 |
| rows seen in only one frame | 69.0% | 61.0% | -8.1 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 53.5% | +7.2 pts ✗ |
| size error, median (m) | 0.30 | 0.25 | -0.05 ✓ |
| size error, median relative | 26.0% | 26.9% | +0.9 pts ✗ |
| distance error, median (m) | 0.57 | 0.90 | +0.33 ✗ |
| distance error, median relative | 27.8% | 37.9% | +10.1 pts ✗ |
| rule: counting | 32.2 | 5.6 | -26.7 ✗ |
| rule: object size | 25.0 | 39.4 | +14.4 ✓ |
| rule: absolute distance | 13.3 | 15.7 | +2.4 ✓ |
| rule: relative distance | 20.0 | 13.3 | -6.7 ✗ |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 95.3% | +11.7 pts ✓ |
| questions with every named object | 58.4% | 89.8% | +31.4 pts ✓ |
| empty sheets where gold has objects | 8 | 3 | -5 ✓ |
| count = gold, per (question, class) | 26.1% | 27.6% | +1.5 pts ✓ |
| rows over the sheet folders (room fragments left out) | 1,548 | 2,801 | +1,253 |
| rows seen in only one frame | 71.5% | 56.3% | -15.2 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 71.5% | +17.1 pts ✗ |
| size error, median (m) | 0.32 | 0.36 | +0.04 ✗ |
| size error, median relative | 32.1% | 29.5% | -2.5 pts ✓ |
| distance error, median (m) | 0.39 | 0.84 | +0.45 ✗ |
| distance error, median relative | 19.4% | 42.4% | +23.0 pts ✗ |
| rule: counting | 11.3 | 20.0 | +8.7 ✓ |
| rule: object size | 36.0 | 38.0 | +2.0 ✓ |
| rule: absolute distance | 22.7 | 16.4 | -6.4 ✗ |
| rule: relative distance | 59.1 | 45.5 | -13.6 ✗ |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 61.6% | +8.1 pts ✓ |
| questions with every named object | 25.8% | 33.1% | +7.3 pts ✓ |
| empty sheets where gold has objects | 18 | 14 | -4 ✓ |
| count = gold, per (question, class) | 18.2% | 18.2% | +0.0 pts |
| rows over the sheet folders (room fragments left out) | 363 | 510 | +147 |
| rows seen in only one frame | 47.7% | 49.0% | +1.4 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 56.5% | +12.1 pts ✗ |
| size error, median (m) | 0.16 | 0.16 | +0.00 |
| size error, median relative | 23.6% | 30.9% | +7.3 pts ✗ |
| distance error, median (m) | 0.43 | 0.41 | -0.03 ✓ |
| distance error, median relative | 29.8% | 37.9% | +8.0 pts ✗ |
| rule: counting | 32.1 | 29.3 | -2.9 ✗ |
| rule: object size | 25.2 | 23.9 | -1.3 ✗ |
| rule: absolute distance | 2.1 | 11.2 | +9.2 ✓ |
| rule: relative distance | 17.6 | 5.9 | -11.8 ✗ |

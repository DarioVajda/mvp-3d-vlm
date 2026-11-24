# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed_s050`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 76.0% | +3.9 pts ✓ |
| questions with every named object | 46.6% | 54.7% | +8.2 pts ✓ |
| empty sheets where gold has objects | 41 | 36 | -5 ✓ |
| count = gold, per (question, class) | 23.7% | 22.5% | -1.2 pts ✗ |
| rows over the sheet folders (room fragments left out) | 2,276 | 2,578 | +302 |
| rows seen in only one frame | 67.3% | 70.6% | +3.3 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 55.4% | +3.9 pts ✗ |
| size error, median (m) | 0.31 | 0.24 | -0.07 ✓ |
| size error, median relative | 29.3% | 26.6% | -2.7 pts ✓ |
| distance error, median (m) | 0.40 | 0.47 | +0.07 ✗ |
| distance error, median relative | 22.6% | 27.5% | +4.9 pts ✗ |
| rule: counting | 23.9 | 23.9 | +0.0 |
| rule: object size | 29.7 | 30.8 | +1.1 ✓ |
| rule: absolute distance | 13.8 | 16.0 | +2.2 ✓ |
| rule: relative distance | 35.2 | 31.5 | -3.7 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 66.4% | +0.3 pts ✓ |
| questions with every named object | 45.6% | 46.5% | +0.9 pts ✓ |
| empty sheets where gold has objects | 15 | 15 | +0 |
| count = gold, per (question, class) | 24.1% | 22.3% | -1.7 pts ✗ |
| rows over the sheet folders (room fragments left out) | 365 | 375 | +10 |
| rows seen in only one frame | 69.0% | 69.6% | +0.6 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 47.7% | +1.4 pts ✗ |
| size error, median (m) | 0.30 | 0.32 | +0.02 ✗ |
| size error, median relative | 26.0% | 25.6% | -0.3 pts ✓ |
| distance error, median (m) | 0.57 | 0.54 | -0.03 ✓ |
| distance error, median relative | 27.8% | 27.0% | -0.8 pts ✓ |
| rule: counting | 32.2 | 32.2 | +0.0 |
| rule: object size | 25.0 | 25.0 | +0.0 |
| rule: absolute distance | 13.3 | 12.4 | -1.0 ✗ |
| rule: relative distance | 20.0 | 20.0 | +0.0 |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 89.3% | +5.8 pts ✓ |
| questions with every named object | 58.4% | 72.1% | +13.7 pts ✓ |
| empty sheets where gold has objects | 8 | 5 | -3 ✓ |
| count = gold, per (question, class) | 26.1% | 22.9% | -3.1 pts ✗ |
| rows over the sheet folders (room fragments left out) | 1,548 | 1,808 | +260 |
| rows seen in only one frame | 71.5% | 74.9% | +3.4 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 59.0% | +4.6 pts ✗ |
| size error, median (m) | 0.32 | 0.21 | -0.12 ✓ |
| size error, median relative | 32.1% | 28.4% | -3.7 pts ✓ |
| distance error, median (m) | 0.39 | 0.57 | +0.18 ✗ |
| distance error, median relative | 19.4% | 27.1% | +7.7 pts ✗ |
| rule: counting | 11.3 | 11.3 | +0.0 |
| rule: object size | 36.0 | 37.7 | +1.7 ✓ |
| rule: absolute distance | 22.7 | 23.3 | +0.6 ✓ |
| rule: relative distance | 59.1 | 50.0 | -9.1 ✗ |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 56.9% | +3.4 pts ✓ |
| questions with every named object | 25.8% | 30.6% | +4.8 pts ✓ |
| empty sheets where gold has objects | 18 | 16 | -2 ✓ |
| count = gold, per (question, class) | 18.2% | 21.7% | +3.5 pts ✓ |
| rows over the sheet folders (room fragments left out) | 363 | 395 | +32 |
| rows seen in only one frame | 47.7% | 51.9% | +4.2 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 46.1% | +1.7 pts ✗ |
| size error, median (m) | 0.16 | 0.28 | +0.12 ✗ |
| size error, median relative | 23.6% | 22.6% | -1.0 pts ✓ |
| distance error, median (m) | 0.43 | 0.35 | -0.08 ✓ |
| distance error, median relative | 29.8% | 28.4% | -1.4 pts ✓ |
| rule: counting | 32.1 | 32.1 | +0.0 |
| rule: object size | 25.2 | 26.5 | +1.3 ✓ |
| rule: absolute distance | 2.1 | 9.2 | +7.1 ✓ |
| rule: relative distance | 17.6 | 17.6 | +0.0 |

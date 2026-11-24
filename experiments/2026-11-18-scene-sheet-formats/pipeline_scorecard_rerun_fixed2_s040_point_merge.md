# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed2_s040_point_merge`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 78.9% | +6.9 pts ✓ |
| questions with every named object | 46.6% | 62.9% | +16.4 pts ✓ |
| empty sheets where gold has objects | 41 | 34 | -7 ✓ |
| count = gold, per (question, class) | 23.7% | 32.3% | +8.6 pts ✓ |
| rows over the sheet folders (room fragments left out) | 2,276 | 2,191 | -85 |
| rows seen in only one frame | 67.3% | 48.2% | -19.1 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 46.6% | -4.9 pts ✓ |
| size error, median (m) | 0.31 | 0.28 | -0.04 ✓ |
| size error, median relative | 29.3% | 27.5% | -1.8 pts ✓ |
| distance error, median (m) | 0.40 | 0.54 | +0.14 ✗ |
| distance error, median relative | 22.6% | 27.9% | +5.2 pts ✗ |
| rule: counting | 23.9 | 31.8 | +7.9 ✓ |
| rule: object size | 29.7 | 31.8 | +2.1 ✓ |
| rule: absolute distance | 13.8 | 18.1 | +4.2 ✓ |
| rule: relative distance | 35.2 | 27.8 | -7.4 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 66.4% | +0.3 pts ✓ |
| questions with every named object | 45.6% | 46.5% | +0.9 pts ✓ |
| empty sheets where gold has objects | 15 | 15 | +0 |
| count = gold, per (question, class) | 24.1% | 21.6% | -2.4 pts ✗ |
| rows over the sheet folders (room fragments left out) | 365 | 356 | -9 |
| rows seen in only one frame | 69.0% | 53.4% | -15.7 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 44.9% | -1.4 pts ✓ |
| size error, median (m) | 0.30 | 0.32 | +0.02 ✗ |
| size error, median relative | 26.0% | 25.6% | -0.4 pts ✓ |
| distance error, median (m) | 0.57 | 0.71 | +0.14 ✗ |
| distance error, median relative | 27.8% | 27.7% | -0.0 pts ✓ |
| rule: counting | 32.2 | 21.1 | -11.1 ✗ |
| rule: object size | 25.0 | 25.6 | +0.6 ✓ |
| rule: absolute distance | 13.3 | 10.5 | -2.9 ✗ |
| rule: relative distance | 20.0 | 20.0 | +0.0 |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 95.0% | +11.4 pts ✓ |
| questions with every named object | 58.4% | 88.9% | +30.5 pts ✓ |
| empty sheets where gold has objects | 8 | 3 | -5 ✓ |
| count = gold, per (question, class) | 26.1% | 41.5% | +15.4 pts ✓ |
| rows over the sheet folders (room fragments left out) | 1,548 | 1,444 | -104 |
| rows seen in only one frame | 71.5% | 47.4% | -24.1 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 47.1% | -7.4 pts ✓ |
| size error, median (m) | 0.32 | 0.26 | -0.06 ✓ |
| size error, median relative | 32.1% | 28.4% | -3.6 pts ✓ |
| distance error, median (m) | 0.39 | 0.61 | +0.22 ✗ |
| distance error, median relative | 19.4% | 28.0% | +8.6 pts ✗ |
| rule: counting | 11.3 | 37.3 | +26.0 ✓ |
| rule: object size | 36.0 | 41.3 | +5.3 ✓ |
| rule: absolute distance | 22.7 | 28.8 | +6.1 ✓ |
| rule: relative distance | 59.1 | 50.0 | -9.1 ✗ |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 56.9% | +3.4 pts ✓ |
| questions with every named object | 25.8% | 30.6% | +4.8 pts ✓ |
| empty sheets where gold has objects | 18 | 16 | -2 ✓ |
| count = gold, per (question, class) | 18.2% | 22.6% | +4.4 pts ✓ |
| rows over the sheet folders (room fragments left out) | 363 | 391 | +28 |
| rows seen in only one frame | 47.7% | 46.8% | -0.9 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 46.3% | +1.9 pts ✗ |
| size error, median (m) | 0.16 | 0.22 | +0.06 ✗ |
| size error, median relative | 23.6% | 27.1% | +3.5 pts ✗ |
| distance error, median (m) | 0.43 | 0.32 | -0.12 ✓ |
| distance error, median relative | 29.8% | 26.5% | -3.4 pts ✓ |
| rule: counting | 32.1 | 32.9 | +0.7 ✓ |
| rule: object size | 25.2 | 24.3 | -0.9 ✗ |
| rule: absolute distance | 2.1 | 10.0 | +7.9 ✓ |
| rule: relative distance | 17.6 | 5.9 | -11.8 ✗ |

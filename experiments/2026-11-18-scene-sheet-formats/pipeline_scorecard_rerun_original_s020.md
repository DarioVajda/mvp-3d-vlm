# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/original_s020`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 83.0% | +10.9 pts ✓ |
| questions with every named object | 46.6% | 65.5% | +19.0 pts ✓ |
| empty sheets where gold has objects | 41 | 22 | -19 ✓ |
| count = gold, per (question, class) | 23.7% | 12.9% | -10.7 pts ✗ |
| rows over the sheet folders (room fragments left out) | 2,276 | 4,565 | +2,289 |
| rows seen in only one frame | 67.3% | 73.9% | +6.6 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 72.5% | +20.9 pts ✗ |
| size error, median (m) | 0.31 | 0.33 | +0.01 ✗ |
| size error, median relative | 29.3% | 28.3% | -1.0 pts ✓ |
| distance error, median (m) | 0.40 | 0.72 | +0.32 ✗ |
| distance error, median relative | 22.6% | 37.9% | +15.2 pts ✗ |
| rule: counting | 23.9 | 17.4 | -6.6 ✗ |
| rule: object size | 29.7 | 33.1 | +3.4 ✓ |
| rule: absolute distance | 13.8 | 12.4 | -1.4 ✗ |
| rule: relative distance | 35.2 | 29.6 | -5.6 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 81.9% | +15.8 pts ✓ |
| questions with every named object | 45.6% | 57.9% | +12.3 pts ✓ |
| empty sheets where gold has objects | 15 | 3 | -12 ✓ |
| count = gold, per (question, class) | 24.1% | 28.2% | +4.1 pts ✓ |
| rows over the sheet folders (room fragments left out) | 365 | 587 | +222 |
| rows seen in only one frame | 69.0% | 72.9% | +3.9 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 58.4% | +12.1 pts ✗ |
| size error, median (m) | 0.30 | 0.33 | +0.03 ✗ |
| size error, median relative | 26.0% | 21.6% | -4.4 pts ✓ |
| distance error, median (m) | 0.57 | 0.72 | +0.15 ✗ |
| distance error, median relative | 27.8% | 37.2% | +9.4 pts ✗ |
| rule: counting | 32.2 | 5.6 | -26.7 ✗ |
| rule: object size | 25.0 | 39.4 | +14.4 ✓ |
| rule: absolute distance | 13.3 | 13.8 | +0.5 ✓ |
| rule: relative distance | 20.0 | 13.3 | -6.7 ✗ |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 95.3% | +11.7 pts ✓ |
| questions with every named object | 58.4% | 89.8% | +31.4 pts ✓ |
| empty sheets where gold has objects | 8 | 3 | -5 ✓ |
| count = gold, per (question, class) | 26.1% | 4.8% | -21.3 pts ✗ |
| rows over the sheet folders (room fragments left out) | 1,548 | 3,421 | +1,873 |
| rows seen in only one frame | 71.5% | 76.4% | +4.9 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 76.7% | +22.2 pts ✗ |
| size error, median (m) | 0.32 | 0.34 | +0.02 ✗ |
| size error, median relative | 32.1% | 33.2% | +1.2 pts ✗ |
| distance error, median (m) | 0.39 | 0.74 | +0.35 ✗ |
| distance error, median relative | 19.4% | 37.2% | +17.9 pts ✗ |
| rule: counting | 11.3 | 10.0 | -1.3 ✗ |
| rule: object size | 36.0 | 37.3 | +1.3 ✓ |
| rule: absolute distance | 22.7 | 17.6 | -5.2 ✗ |
| rule: relative distance | 59.1 | 59.1 | +0.0 |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 58.1% | +4.7 pts ✓ |
| questions with every named object | 25.8% | 28.2% | +2.4 pts ✓ |
| empty sheets where gold has objects | 18 | 16 | -2 ✓ |
| count = gold, per (question, class) | 18.2% | 16.0% | -2.2 pts ✗ |
| rows over the sheet folders (room fragments left out) | 363 | 557 | +194 |
| rows seen in only one frame | 47.7% | 59.4% | +11.8 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 61.6% | +17.2 pts ✗ |
| size error, median (m) | 0.16 | 0.15 | -0.01 ✓ |
| size error, median relative | 23.6% | 29.6% | +6.0 pts ✗ |
| distance error, median (m) | 0.43 | 0.47 | +0.04 ✗ |
| distance error, median relative | 29.8% | 41.8% | +12.0 pts ✗ |
| rule: counting | 32.1 | 32.9 | +0.7 ✓ |
| rule: object size | 25.2 | 22.6 | -2.6 ✗ |
| rule: absolute distance | 2.1 | 4.2 | +2.1 ✓ |
| rule: relative distance | 17.6 | 5.9 | -11.8 ✗ |

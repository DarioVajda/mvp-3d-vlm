# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed_s030`. Scenes: `dev_scenes_15_without_d755b3d9d8.txt`, 14 scenes, 478 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 14 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 72.1% | 80.4% | +8.4 pts ✓ |
| questions with every named object | 46.6% | 63.4% | +16.8 pts ✓ |
| empty sheets where gold has objects | 41 | 26 | -15 ✓ |
| count = gold, per (question, class) | 23.7% | 14.2% | -9.5 pts ✗ |
| rows over the sheet folders (room fragments left out) | 2,276 | 3,687 | +1,411 |
| rows seen in only one frame | 67.3% | 72.6% | +5.2 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 67.2% | +15.6 pts ✗ |
| size error, median (m) | 0.31 | 0.26 | -0.06 ✓ |
| size error, median relative | 29.3% | 26.9% | -2.4 pts ✓ |
| distance error, median (m) | 0.40 | 0.56 | +0.15 ✗ |
| distance error, median relative | 22.6% | 29.9% | +7.2 pts ✗ |
| rule: counting | 23.9 | 18.9 | -5.0 ✗ |
| rule: object size | 29.7 | 32.8 | +3.1 ✓ |
| rule: absolute distance | 13.8 | 14.9 | +1.0 ✓ |
| rule: relative distance | 35.2 | 20.4 | -14.8 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 73.0% | +6.9 pts ✓ |
| questions with every named object | 45.6% | 48.2% | +2.6 pts ✓ |
| empty sheets where gold has objects | 15 | 7 | -8 ✓ |
| count = gold, per (question, class) | 24.1% | 28.5% | +4.5 pts ✓ |
| rows over the sheet folders (room fragments left out) | 365 | 455 | +90 |
| rows seen in only one frame | 69.0% | 69.9% | +0.8 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 51.9% | +5.6 pts ✗ |
| size error, median (m) | 0.30 | 0.25 | -0.04 ✓ |
| size error, median relative | 26.0% | 26.4% | +0.4 pts ✗ |
| distance error, median (m) | 0.57 | 0.73 | +0.16 ✗ |
| distance error, median relative | 27.8% | 29.6% | +1.8 pts ✗ |
| rule: counting | 32.2 | 26.7 | -5.6 ✗ |
| rule: object size | 25.0 | 34.4 | +9.4 ✓ |
| rule: absolute distance | 13.3 | 9.0 | -4.3 ✗ |
| rule: relative distance | 20.0 | 13.3 | -6.7 ✗ |

### ScanNet (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 83.6% | 95.0% | +11.4 pts ✓ |
| questions with every named object | 58.4% | 88.9% | +30.5 pts ✓ |
| empty sheets where gold has objects | 8 | 3 | -5 ✓ |
| count = gold, per (question, class) | 26.1% | 7.2% | -18.9 pts ✗ |
| rows over the sheet folders (room fragments left out) | 1,548 | 2,738 | +1,190 |
| rows seen in only one frame | 71.5% | 76.3% | +4.7 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 71.6% | +17.1 pts ✗ |
| size error, median (m) | 0.32 | 0.27 | -0.06 ✓ |
| size error, median relative | 32.1% | 28.4% | -3.7 pts ✓ |
| distance error, median (m) | 0.39 | 0.59 | +0.20 ✗ |
| distance error, median relative | 19.4% | 33.7% | +14.3 pts ✗ |
| rule: counting | 11.3 | 4.7 | -6.7 ✗ |
| rule: object size | 36.0 | 38.3 | +2.3 ✓ |
| rule: absolute distance | 22.7 | 22.1 | -0.6 ✗ |
| rule: relative distance | 59.1 | 36.4 | -22.7 ✗ |

### ScanNet++ (4 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 53.4% | 56.9% | +3.4 pts ✓ |
| questions with every named object | 25.8% | 30.6% | +4.8 pts ✓ |
| empty sheets where gold has objects | 18 | 16 | -2 ✓ |
| count = gold, per (question, class) | 18.2% | 15.7% | -2.5 pts ✗ |
| rows over the sheet folders (room fragments left out) | 363 | 494 | +131 |
| rows seen in only one frame | 47.7% | 54.5% | +6.8 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 44.4% | 56.9% | +12.5 pts ✗ |
| size error, median (m) | 0.16 | 0.22 | +0.05 ✗ |
| size error, median relative | 23.6% | 27.4% | +3.8 pts ✗ |
| distance error, median (m) | 0.43 | 0.38 | -0.06 ✓ |
| distance error, median relative | 29.8% | 25.6% | -4.2 pts ✓ |
| rule: counting | 32.1 | 29.3 | -2.9 ✗ |
| rule: object size | 25.2 | 24.3 | -0.9 ✗ |
| rule: absolute distance | 2.1 | 10.0 | +7.9 ✓ |
| rule: relative distance | 17.6 | 5.9 | -11.8 ✗ |

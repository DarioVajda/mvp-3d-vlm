# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/fixed_s020`. Scenes: `dev_scenes_15.txt`, 15 scenes, 555 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 15 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 70.1% | 82.5% | +12.4 pts ✓ |
| questions with every named object | 43.1% | 63.0% | +19.8 pts ✓ |
| empty sheets where gold has objects | 44 | 23 | -21 ✓ |
| count = gold, per (question, class) | 23.3% | 15.2% | -8.1 pts ✗ |
| rows over the sheet folders (room fragments left out) | 2,827 | 5,708 | +2,881 |
| rows seen in only one frame | 71.2% | 75.8% | +4.5 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 55.3% | 74.3% | +19.0 pts ✗ |
| size error, median (m) | 0.27 | 0.25 | -0.02 ✓ |
| size error, median relative | 29.3% | 27.6% | -1.6 pts ✓ |
| distance error, median (m) | 0.40 | 0.72 | +0.32 ✗ |
| distance error, median relative | 22.6% | 36.9% | +14.3 pts ✗ |
| rule: counting | 20.7 | 15.2 | -5.4 ✗ |
| rule: object size | 30.4 | 32.6 | +2.2 ✓ |
| rule: absolute distance | 12.7 | 13.1 | +0.4 ✓ |
| rule: relative distance | 33.8 | 24.6 | -9.2 ✗ |

### ARKitScenes (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 66.1% | 81.9% | +15.8 pts ✓ |
| questions with every named object | 45.6% | 57.9% | +12.3 pts ✓ |
| empty sheets where gold has objects | 15 | 3 | -12 ✓ |
| count = gold, per (question, class) | 24.1% | 28.2% | +4.1 pts ✓ |
| rows over the sheet folders (room fragments left out) | 365 | 598 | +233 |
| rows seen in only one frame | 69.0% | 73.4% | +4.4 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 46.3% | 59.2% | +12.9 pts ✗ |
| size error, median (m) | 0.30 | 0.25 | -0.05 ✓ |
| size error, median relative | 26.0% | 26.9% | +1.0 pts ✗ |
| distance error, median (m) | 0.57 | 0.72 | +0.15 ✗ |
| distance error, median relative | 27.8% | 32.1% | +4.4 pts ✗ |
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
| rows over the sheet folders (room fragments left out) | 1,548 | 3,690 | +2,142 |
| rows seen in only one frame | 71.5% | 76.5% | +5.0 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 54.5% | 78.4% | +23.9 pts ✗ |
| size error, median (m) | 0.32 | 0.31 | -0.02 ✓ |
| size error, median relative | 32.1% | 27.6% | -4.4 pts ✓ |
| distance error, median (m) | 0.39 | 0.79 | +0.40 ✗ |
| distance error, median relative | 19.4% | 38.3% | +18.9 pts ✗ |
| rule: counting | 11.3 | 10.0 | -1.3 ✗ |
| rule: object size | 36.0 | 36.7 | +0.7 ✓ |
| rule: absolute distance | 22.7 | 16.7 | -6.1 ✗ |
| rule: relative distance | 59.1 | 40.9 | -18.2 ✗ |

### ScanNet++ (5 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 56.2% | 67.6% | +11.3 pts ✓ |
| questions with every named object | 24.5% | 35.5% | +11.0 pts ✓ |
| empty sheets where gold has objects | 21 | 17 | -4 ✓ |
| count = gold, per (question, class) | 19.6% | 20.9% | +1.4 pts ✓ |
| rows over the sheet folders (room fragments left out) | 914 | 1,420 | +506 |
| rows seen in only one frame | 71.6% | 74.8% | +3.2 pts ✗ |
| unmatched rows (beyond the gold sheet's count) | 60.3% | 70.1% | +9.9 pts ✗ |
| size error, median (m) | 0.12 | 0.19 | +0.06 ✗ |
| size error, median relative | 23.6% | 32.1% | +8.5 pts ✗ |
| distance error, median (m) | 0.43 | 0.48 | +0.04 ✗ |
| distance error, median relative | 28.9% | 28.1% | -0.8 pts ✓ |
| rule: counting | 22.3 | 22.7 | +0.5 ✓ |
| rule: object size | 28.0 | 24.3 | -3.7 ✗ |
| rule: absolute distance | 1.6 | 8.7 | +7.1 ✓ |
| rule: relative distance | 21.4 | 17.9 | -3.6 ✗ |

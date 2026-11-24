# Pipeline scorecard

<!-- Written by pipeline_scorecard.py in this folder; edit that, not this file. -->

Old: `data/scene_sheets_75`. New: `data/rerun_sheets/pipeline_v3_dev35`. Scenes: `dev_scenes_35.txt`, 35 scenes, 1,258 questions. Each folder against its own gold sheets. ✓ / ✗: the new sheets are better / worse on that measure. Definitions are in `pipeline_scorecard.py`.

The size and distance errors and the fixed rules use the priors scale (`--metric-scale priors`, `mvp.benchmarks.scene_scale`; no gold). Its calibration constant was fitted on the reliable scenes outside explore200's scenes, which are among the dev scenes, so on dev scenes it is slightly optimistic.

### All 35 scenes

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 76.5% | 79.9% | +3.5 pts ✓ |
| questions with every named object | 52.7% | 58.2% | +5.5 pts ✓ |
| empty sheets where gold has objects | 94 | 82 | -12 ✓ |
| count = gold, per (question, class) | 28.7% | 31.0% | +2.3 pts ✓ |
| rows over the sheet folders (room fragments left out) | 7,139 | 6,890 | -249 |
| rows seen in only one frame | 66.1% | 54.2% | -11.9 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 52.7% | 49.4% | -3.3 pts ✓ |
| size error, median (m) | 0.29 | 0.27 | -0.02 ✓ |
| size error, median relative | 30.8% | 28.7% | -2.1 pts ✓ |
| distance error, median (m) | 0.56 | 0.61 | +0.04 ✗ |
| distance error, median relative | 21.9% | 22.9% | +1.1 pts ✗ |
| rule: counting | 25.2 | 27.7 | +2.5 ✓ |
| rule: object size | 29.5 | 30.7 | +1.2 ✓ |
| rule: absolute distance | 17.8 | 17.2 | -0.6 ✗ |
| rule: relative distance | 34.3 | 38.0 | +3.6 ✓ |

### ARKitScenes (14 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 73.1% | 73.2% | +0.1 pts ✓ |
| questions with every named object | 47.9% | 47.9% | +0.0 pts |
| empty sheets where gold has objects | 26 | 26 | +0 |
| count = gold, per (question, class) | 34.3% | 32.1% | -2.2 pts ✗ |
| rows over the sheet folders (room fragments left out) | 918 | 905 | -13 |
| rows seen in only one frame | 83.8% | 62.1% | -21.7 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 48.0% | 48.1% | +0.0 pts ✗ |
| size error, median (m) | 0.24 | 0.24 | -0.00 ✓ |
| size error, median relative | 24.9% | 24.7% | -0.3 pts ✓ |
| distance error, median (m) | 0.77 | 0.69 | -0.08 ✓ |
| distance error, median relative | 28.9% | 30.4% | +1.6 pts ✗ |
| rule: counting | 28.7 | 25.6 | -3.1 ✗ |
| rule: object size | 34.0 | 33.2 | -0.8 ✗ |
| rule: absolute distance | 21.0 | 18.1 | -2.9 ✗ |
| rule: relative distance | 57.1 | 57.1 | +0.0 |

### ScanNet (12 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 86.8% | 87.9% | +1.0 pts ✓ |
| questions with every named object | 68.6% | 71.7% | +3.1 pts ✓ |
| empty sheets where gold has objects | 20 | 19 | -1 ✓ |
| count = gold, per (question, class) | 28.0% | 30.3% | +2.3 pts ✓ |
| rows over the sheet folders (room fragments left out) | 4,373 | 4,266 | -107 |
| rows seen in only one frame | 61.4% | 51.7% | -9.7 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 54.7% | 53.0% | -1.7 pts ✓ |
| size error, median (m) | 0.32 | 0.28 | -0.04 ✓ |
| size error, median relative | 33.2% | 33.5% | +0.3 pts ✗ |
| distance error, median (m) | 0.64 | 0.69 | +0.04 ✗ |
| distance error, median relative | 22.6% | 25.2% | +2.6 pts ✗ |
| rule: counting | 24.0 | 30.2 | +6.2 ✓ |
| rule: object size | 29.7 | 31.0 | +1.3 ✓ |
| rule: absolute distance | 20.8 | 18.6 | -2.1 ✗ |
| rule: relative distance | 31.0 | 28.2 | -2.8 ✗ |

### ScanNet++ (9 scenes)

| Measure | Old | New | New − old |
|---|---:|---:|---:|
| named objects found | 63.4% | 72.7% | +9.3 pts ✓ |
| questions with every named object | 33.9% | 46.9% | +13.0 pts ✓ |
| empty sheets where gold has objects | 48 | 37 | -11 ✓ |
| count = gold, per (question, class) | 26.1% | 31.4% | +5.3 pts ✓ |
| rows over the sheet folders (room fragments left out) | 1,848 | 1,719 | -129 |
| rows seen in only one frame | 68.6% | 56.2% | -12.4 pts ✓ |
| unmatched rows (beyond the gold sheet's count) | 50.2% | 41.0% | -9.2 pts ✓ |
| size error, median (m) | 0.24 | 0.23 | -0.00 ✓ |
| size error, median relative | 29.7% | 24.2% | -5.5 pts ✓ |
| distance error, median (m) | 0.40 | 0.52 | +0.12 ✗ |
| distance error, median relative | 15.7% | 19.7% | +4.0 pts ✗ |
| rule: counting | 25.2 | 25.8 | +0.6 ✓ |
| rule: object size | 25.0 | 27.9 | +2.9 ✓ |
| rule: absolute distance | 10.2 | 14.2 | +4.0 ✓ |
| rule: relative distance | 28.4 | 40.3 | +11.9 ✓ |

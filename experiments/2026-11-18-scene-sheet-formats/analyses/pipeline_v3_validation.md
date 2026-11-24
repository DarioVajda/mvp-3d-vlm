| Measure | tuning 14 (14 scenes, 478 q): current | pipeline_v3 | change | validation 35 (35 scenes, 1258 q): current | pipeline_v3 | change |
|---|---:|---:|---:|---:|---:|---:|
| named objects found | 72.1% | 78.9% | +6.9 pts | 76.5% | 79.9% | +3.5 pts |
| questions with every named object | 46.6% | 62.9% | +16.4 pts | 52.7% | 58.2% | +5.5 pts |
| empty sheets where gold has objects | 41 | 34 | -7 | 94 | 82 | -12 |
| count = gold, per (question, class) | 23.7% | 32.3% | +8.6 pts | 28.7% | 31.0% | +2.3 pts |
| rows over the sheet folders (room fragments left out) | 2,276 | 2,191 | -85 | 7,139 | 6,890 | -249 |
| unmatched rows (beyond the gold sheet's count) | 51.5% | 46.6% | -4.9 pts | 52.7% | 49.4% | -3.3 pts |
| size error, median relative | 29.3% | 27.7% | -1.5 pts | 30.8% | 28.7% | -2.1 pts |
| distance error, median relative | 22.6% | 27.2% | +4.6 pts | 21.9% | 22.9% | +1.1 pts |
| rule: counting | 23.9 | 31.8 | +7.9 | 25.2 | 27.7 | +2.5 |
| rule: object size | 29.7 | 32.1 | +2.4 | 29.5 | 30.7 | +1.2 |
| rule: absolute distance | 13.8 | 19.1 | +5.3 | 17.8 | 17.2 | -0.6 |
| rule: relative distance | 35.2 | 27.8 | -7.4 | 34.3 | 38.0 | +3.6 |
| size error, shared pairs (49 / 125) | 29.3% | 24.8% | -4.5 pts | 30.8% | 28.1% | -2.7 pts |
| distance error, shared pairs (86 / 219) | 22.6% | 23.6% | +1.0 pts | 21.9% | 22.1% | +0.3 pts |
| relative distance lost / gained |  | 4 / 0 of 54 |  |  | 6 / 12 of 166 |  |
| scenes better/same/worse: found |  | 5/9/0 |  |  | 11/24/0 |  |
| scenes better/same/worse: unmatched rows |  | 8/2/4 |  |  | 21/2/12 |  |

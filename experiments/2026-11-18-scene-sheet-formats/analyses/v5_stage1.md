# v5 stage 1: pipeline_v4 and the four configurations on dev_scenes_15

<!-- Written by v5_stage1_analysis.py. -->

**Exploratory: 15 tuning scenes (dev_scenes_15.txt), no test.** The configurations were made to be chosen on these scenes; the differences are not significance-tested, and the simulated score rests on the few distance questions with a video-only answer.

| Measure | v4 | v5-his | v5-det | v5-pts | v5-ours |
|---|---:|---:|---:|---:|---:|
| named objects found | 77.6 | 95.3 | 96.2 | 94.7 | 95.6 |
| rows per real object | 1.53 | 1.30 | 1.37 | 1.30 | 1.36 |
| count = gold | 49.6 | 67.2 | 65.5 | 67.3 | 65.3 |
| box side / gold, median | 0.79 | 1.02 | 1.06 | 1.02 | 1.03 |
| boxes over 1.5x | 5.08 | 16.2 | 20.3 | 12.3 | 16.4 |
| gap error, median (m) | 0.60 | 0.48 | 0.55 | 0.47 | 0.51 |
| aligned position error, median (m) | 0.38 | 0.19 | 0.17 | 0.19 | 0.17 |
| scenes aligned | 10/15 | 14/15 | 14/15 | 14/15 | 14/15 |
| abs: questions with a relation | 46/85 | 75/85 | 75/85 | 73/85 | 73/85 |
| abs: MRA of the relation | 30.0 | 56.3 | 51.2 | 56.2 | 50.7 |
| abs: relation / truth, median | 1.03 | 0.89 | 0.84 | 0.90 | 0.84 |
| rel: questions with a relation | 24/65 | 58/65 | 61/65 | 58/65 | 61/65 |
| rel: closest object right | 70.8 | 60.3 | 60.7 | 62.1 | 63.9 |
| simulated distance score | 39.3 | 44.7 | 39.3 | 44.7 | 39.3 |
| video-only distance score | 44.0 | 44.0 | 44.0 | 44.0 | 44.0 |
| simulated: questions | 15 | 15 | 15 | 15 | 15 |
| abs: MRA, paired with v4 (v5 / v4, n) | - | 50.9 / 30.0, 46 | 46.3 / 30.0, 46 | 50.9 / 30.0, 46 | 45.9 / 30.0, 46 |
| rel: closest right, paired with v4 (v5 / v4, n) | - | 54.2 / 70.8, 24 | 58.3 / 70.8, 24 | 54.2 / 70.8, 24 | 58.3 / 70.8, 24 |

## Stage 2: NO-GO (README's criteria, best v5 configuration: v5-pts)

- pass: simulated distance score >= v4's
- pass: abs: MRA of the relation >= v4's, on the questions both have a relation for
- FAIL: rel: closest object right >= v4's, on the questions both have a relation for
- pass: box side / gold at least as close to 1 as v4's
- FAIL: boxes over 1.5x at most v4's + 5 points
- pass: named objects found at least v4's - 2 points

Video-only answers: /Users/tamaragrujicic/Documents/GitHub/mvp-3d-vlm-ssf/experiments/2026-11-18-scene-sheet-formats/results/dev200/A_video.jsonl. Percentages are of questions or (question, object) pairs; sizes and errors in metres.

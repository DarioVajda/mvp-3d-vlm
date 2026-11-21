# Camera pilot, run 2 (metres, with chain of thought): analysis

<!-- Written by analysis_metres.py in this folder; edit that, not this file. -->

The predictions of `explore_camera_metres.ipynb` (`results/zeroshot/vsibench/camera/explore200_metres/`), on one NVIDIA L4 at commit `ec85068` (the code zip, uncommitted `src/` changes included):
- 200 questions (`explore200_ids.txt`, 25 per task), Qwen3.5-2B, 16 frames (`uniform128`), final answer scored, batches of 8;
- every pipeline sheet in metres (`--metric-scale priors`): the objects, the camera tables, the compass path, the narrative;
- 15 direct conditions (up to 256 new tokens). `video` was copied from `scot_explore_results/A_video.jsonl` after the reuse check, as in run 1;
- 3 CoT conditions on all 200 questions (`--reasoning cot`, up to 2,048 new tokens). The table format was chosen by the rule fixed before the run: the direct `cam_markdown_table` scored 44.6 overall against `cam_csv`'s 38.4, so the CoT table is `cot_cam_markdown_table`. `cot_cam_csv.jsonl` holds only its 5 timing questions.

Bootstrap intervals: questions resampled within each question type, the same draws for every condition (10,000 resamples, seed 0). The scores, the reply counts and every interval are recomputed here from the predictions and match the notebook's tables. Run 1's predictions (relative units) come from `results/zeroshot/vsibench/camera/explore200/`; both runs share `video`.

## Replies

**Direct:** every reply was parsed. Median 1 token, 90th percentile 3–5. 3 replies ran to the 256-token limit: `narrative_camera_dirs` on 620, `narrative_frames` on 3782, `narrative_exact` on 2697, all room-size questions, each a product of the sheet's sizes that never ends ("1.874*2.190*4.504*…").

**CoT:** 22, 47 and 57 of 200 replies were cut off at 2,048 tokens (the table, the compass path, the narrative); 20, 47 and 57 of them before an "Answer:" line. 13, 23 and 45 loop by the structured-CoT pilot's rule (10 or more mm:ss times, or a tail that compresses below a quarter); the others run out while still working: distances over many pairs of rows, the route options tried one by one, or lists of frames and numbers that the rule does not count as a loop. The pilot had 12% of replies cut off with the gold sheet and 28% with video only.

| Condition | parsed % | 'Answer:' line % | median tokens | 90th pct tokens | hit the limit | of them before answering | of those, read anyway | cut off and looping | score | score, cut-off replies wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cot_cam_markdown_table` | 96.0 | 90.0 | 644.5 | 2048 | 22 | 20 | 12 | 13 | 34.0 | 32.4 |
| `cot_cam_text` | 87.5 | 76.5 | 929.0 | 2048 | 47 | 47 | 22 | 23 | 28.5 | 27.4 |
| `cot_narrative` | 88.5 | 71.5 | 785.0 | 2048 | 57 | 57 | 34 | 45 | 27.3 | 25.9 |

Per question type (n = 25; relative direction 8, 8 and 9): median reply tokens, replies cut off, parse rate:

| Question type | markdown_table: median tokens | markdown_table: cut off | markdown_table: parsed % | text: median tokens | text: cut off | text: parsed % | narrative: median tokens | narrative: cut off | narrative: parsed % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| counting | 432 | 4 | 100 | 549 | 1 | 100 | 425 | 4 | 100 |
| abs_distance | 914 | 7 | 100 | 2048 | 13 | 100 | 2048 | 19 | 100 |
| size_estimation | 510 | 2 | 100 | 439 | 2 | 100 | 548 | 8 | 100 |
| room_size_estimation | 587 | 0 | 100 | 1389 | 4 | 100 | 980 | 2 | 100 |
| rel_distance | 1064 | 3 | 88 | 1317 | 6 | 76 | 676 | 3 | 88 |
| rel_direction_easy | 682 | 0 | 100 | 682 | 0 | 100 | 528 | 1 | 88 |
| rel_direction_medium | 786 | 1 | 100 | 792 | 1 | 88 | 752 | 1 | 88 |
| rel_direction_hard | 629 | 0 | 100 | 716 | 2 | 78 | 520 | 1 | 89 |
| route_planning | 741 | 3 | 88 | 2048 | 16 | 36 | 2048 | 14 | 44 |
| appearance_order | 435 | 2 | 92 | 559 | 2 | 100 | 419 | 4 | 88 |

## Scores

×100, all 18 conditions on the same 200 questions. **Metric tasks**: the mean of absolute distance, object size and room size. **Direction and route**: the mean of relative direction (its three types as one task) and route planning, 50 questions. **Without metric tasks**: the other 5 tasks. **Sheet tokens**: what the prompt adds to the video-only one, on average (for CoT, the CoT instructions too).

| Condition | Overall | Metric tasks | Direction and route | Without metric tasks | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Sheet tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `video` | 48.6 | 49.2 | 35.6 | 48.2 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 | 0.0 |
| `obj_csv` | 40.0 | 23.5 | 49.9 | 50.0 | 46.0 | 30.8 | 36.4 | 3.2 | 64.0 | 63.9 | 36.0 | 40.0 | 602.2 |
| `cam_csv` | 38.4 | 23.2 | 46.0 | 47.5 | 45.6 | 25.6 | 36.4 | 7.6 | 56.0 | 56.0 | 36.0 | 44.0 | 1825.9 |
| `obj_text` | 36.9 | 25.7 | 45.9 | 43.7 | 50.4 | 23.2 | 22.8 | 31.2 | 32.0 | 51.9 | 40.0 | 44.0 | 727.7 |
| `cam_text` | 39.8 | 31.3 | 46.1 | 44.8 | 52.0 | 32.4 | 25.6 | 36.0 | 48.0 | 48.1 | 44.0 | 32.0 | 1325.6 |
| `narrative` | 37.8 | 28.3 | 44.2 | 43.6 | 49.6 | 26.0 | 34.4 | 24.4 | 40.0 | 52.3 | 36.0 | 40.0 | 1227.3 |
| `obj_markdown_table` | 41.9 | 38.1 | 37.8 | 44.2 | 49.2 | 35.2 | 39.2 | 40.0 | 56.0 | 47.7 | 28.0 | 40.0 | 686.7 |
| `cam_markdown_table` | 44.6 | 37.1 | 47.9 | 49.2 | 54.0 | 24.8 | 45.2 | 41.2 | 60.0 | 51.9 | 44.0 | 36.0 | 2149.4 |
| `cam_csv_frames` | 41.1 | 28.0 | 45.9 | 49.0 | 57.2 | 26.8 | 38.0 | 19.2 | 52.0 | 51.9 | 40.0 | 44.0 | 1796.5 |
| `cam_text_turn_move` | 39.2 | 30.0 | 47.9 | 44.8 | 52.0 | 31.6 | 25.6 | 32.8 | 48.0 | 51.9 | 44.0 | 28.0 | 1337.2 |
| `cam_text_frames` | 38.4 | 28.7 | 44.1 | 44.3 | 53.2 | 25.2 | 28.0 | 32.8 | 40.0 | 48.1 | 40.0 | 40.0 | 1378.6 |
| `narrative_turn_move` | 35.6 | 19.5 | 52.0 | 45.2 | 46.0 | 19.6 | 32.8 | 6.0 | 36.0 | 56.0 | 48.0 | 40.0 | 1239.3 |
| `narrative_camera_dirs` | 36.6 | 24.8 | 44.2 | 43.7 | 50.0 | 26.0 | 30.8 | 17.6 | 44.0 | 52.3 | 36.0 | 36.0 | 1221.2 |
| `narrative_frames` | 37.3 | 24.1 | 44.0 | 45.2 | 50.0 | 20.8 | 34.8 | 16.8 | 48.0 | 56.0 | 32.0 | 40.0 | 1278.9 |
| `narrative_exact` | 36.9 | 25.7 | 42.2 | 43.7 | 50.0 | 21.6 | 37.2 | 18.4 | 40.0 | 52.3 | 32.0 | 44.0 | 1201.9 |
| `cot_cam_markdown_table` | 34.1 | 33.5 | 36.2 | 34.4 | 27.6 | 27.6 | 40.4 | 32.4 | 32.0 | 56.5 | 16.0 | 40.0 | 2325.2 |
| `cot_cam_text` | 28.7 | 26.9 | 26.7 | 29.7 | 27.2 | 24.8 | 36.0 | 20.0 | 36.0 | 45.4 | 8.0 | 32.0 | 1501.3 |
| `cot_narrative` | 27.4 | 18.7 | 24.2 | 32.6 | 22.4 | 15.2 | 28.8 | 12.0 | 44.0 | 32.4 | 16.0 | 48.0 | 1403.0 |

## The comparisons

`*`: the interval excludes 0. **Lead**: the README's reading rule (at least 5 points on direction and route or 2 on the overall, interval excluding 0).

### 1–3. Cameras, the narrative, the options (direct)

| Comparison | overall | metric tasks | direction and route | without metric tasks | lead |
|---|---:|---:|---:|---:|---:|
| `cam_csv − obj_csv` | -1.6 [-5.9, +2.6] | -0.3 [-6.5, +5.9] | -3.9 [-12.3, +4.3] | -2.5 [-8.1, +3.2] |  |
| `cam_markdown_table − obj_markdown_table` | +2.7 [-2.0, +7.4] | -1.1 [-6.8, +4.8] | +10.1 [-2.2, +22.3] | +5.0 [-1.7, +11.8] |  |
| `cam_text − obj_text` | +2.8 [-1.1, +6.8] | +5.6 [+1.7, +9.5] * | +0.1 [-5.4, +6.0] | +1.2 [-4.7, +7.1] |  |
| `narrative − cam_text` | -1.9 [-7.2, +3.1] | -3.1 [-10.9, +4.7] | -1.9 [-10.0, +6.2] | -1.2 [-8.1, +5.4] |  |
| `narrative − obj_text` | +0.9 [-4.2, +6.1] | +2.5 [-5.1, +10.0] | -1.8 [-9.9, +6.2] | -0.1 [-7.1, +6.9] |  |
| `narrative − obj_csv` | -2.2 [-8.1, +3.6] | +4.8 [-3.2, +12.8] | -5.8 [-18.0, +6.3] | -6.4 [-14.5, +1.7] |  |
| `cam_csv_frames − cam_csv` | +2.7 [-0.7, +6.2] | +4.8 [+0.3, +9.7] * | -0.1 [-6.2, +6.0] | +1.5 [-3.3, +6.2] |  |
| `cam_text_turn_move − cam_text` | -0.5 [-2.1, +0.9] | -1.3 [-3.3, +0.8] | +1.9 [+0.0, +5.6] | -0.1 [-2.4, +2.2] |  |
| `cam_text_frames − cam_text` | -1.3 [-5.6, +2.9] | -2.7 [-7.5, +1.9] | -2.0 [-6.0, +0.0] | -0.6 [-6.7, +5.6] |  |
| `narrative_turn_move − narrative` | -2.3 [-6.1, +1.6] | -8.8 [-15.5, -2.7] * | +7.9 [+1.9, +15.7] * | +1.6 [-3.1, +6.4] | direction and route + |
| `narrative_camera_dirs − narrative` | -1.3 [-3.3, +0.6] | -3.5 [-7.6, -0.3] * | +0.0 [+0.0, +0.0] | +0.1 [-2.3, +2.4] |  |
| `narrative_frames − narrative` | -0.5 [-4.2, +3.2] | -4.1 [-10.0, +1.5] | -0.1 [-8.0, +7.7] | +1.6 [-3.2, +6.4] |  |
| `narrative_exact − narrative` | -0.9 [-3.0, +1.1] | -2.5 [-6.9, +1.2] | -2.0 [-6.0, +0.0] | +0.1 [-2.3, +2.4] |  |

Per question type:

| Comparison | object_counting | object_abs_distance | object_size_estimation | room_size_estimation | object_rel_distance | object_rel_direction_easy | object_rel_direction_medium | object_rel_direction_hard | route_planning | obj_appearance_order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cam_csv − obj_csv` | -0.4 [-12.0, +10.8] | -5.2 [-18.4, +7.2] | +0.0 [-10.4, +12.0] | +4.4 [-3.2, +13.2] | -8.0 [-20.0, +0.0] | -25.0 [-62.5, +0.0] | +12.5 [+0.0, +37.5] | -11.1 [-33.3, +0.0] | +0.0 [-12.0, +12.0] | +4.0 [-12.0, +20.0] |
| `cam_markdown_table − obj_markdown_table` | +4.8 [-8.0, +18.0] | -10.4 [-21.2, +0.4] | +6.0 [-1.6, +15.2] | +1.2 [-9.6, +12.4] | +4.0 [+0.0, +12.0] | +12.5 [+0.0, +37.5] | +0.0 [-37.5, +37.5] | +0.0 [+0.0, +0.0] | +16.0 [-4.0, +36.0] | -4.0 [-20.0, +12.0] |
| `cam_text − obj_text` | +1.6 [-6.0, +9.6] | +9.2 [+2.4, +17.2] * | +2.8 [-2.0, +9.6] | +4.8 [-2.4, +10.4] | +16.0 [+0.0, +36.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -11.1 [-33.3, +0.0] | +4.0 [+0.0, +12.0] | -12.0 [-32.0, +8.0] |
| `narrative − cam_text` | -2.4 [-19.6, +14.0] | -6.4 [-23.6, +10.0] | +8.8 [-2.8, +21.2] | -11.6 [-22.8, -0.8] * | -8.0 [-24.0, +8.0] | +0.0 [+0.0, +0.0] | +12.5 [+0.0, +37.5] | +0.0 [+0.0, +0.0] | -8.0 [-24.0, +8.0] | +8.0 [-12.0, +28.0] |
| `narrative − obj_text` | -0.8 [-18.0, +15.6] | +2.8 [-14.4, +19.6] | +11.6 [-0.4, +24.8] | -6.8 [-16.4, +0.0] | +8.0 [-8.0, +24.0] | +0.0 [+0.0, +0.0] | +12.5 [+0.0, +37.5] | -11.1 [-33.3, +0.0] | -4.0 [-16.0, +8.0] | -4.0 [-24.0, +16.0] |
| `narrative − obj_csv` | +3.6 [-14.8, +22.4] | -4.8 [-22.8, +12.8] | -2.0 [-9.6, +4.4] | +21.2 [+8.0, +36.4] * | -24.0 [-40.0, -8.0] * | -25.0 [-62.5, +0.0] | +12.5 [-25.0, +50.0] | -22.2 [-55.6, +0.0] | +0.0 [-16.0, +16.0] | +0.0 [-20.0, +20.0] |
| `cam_csv_frames − cam_csv` | +11.6 [-4.0, +28.4] | +1.2 [-7.2, +8.8] | +1.6 [-3.6, +9.6] | +11.6 [+3.2, +22.0] * | -4.0 [-12.0, +0.0] | +0.0 [+0.0, +0.0] | -12.5 [-37.5, +0.0] | +0.0 [+0.0, +0.0] | +4.0 [+0.0, +12.0] | +0.0 [-12.0, +12.0] |
| `cam_text_turn_move − cam_text` | +0.0 [+0.0, +0.0] | -0.8 [-3.6, +1.6] | +0.0 [-2.4, +2.4] | -3.2 [-8.0, +2.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +11.1 [+0.0, +33.3] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] |
| `cam_text_frames − cam_text` | +1.2 [-12.0, +15.2] | -7.2 [-19.2, +3.2] | +2.4 [-1.6, +8.8] | -3.2 [-11.2, +2.8] | -8.0 [-28.0, +12.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] | +8.0 [-12.0, +28.0] |
| `narrative_turn_move − narrative` | -3.6 [-10.8, +0.0] | -6.4 [-16.8, +0.8] | -1.6 [-8.4, +7.2] | -18.4 [-34.0, -5.2] * | -4.0 [-20.0, +8.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +11.1 [+0.0, +33.3] | +12.0 [+0.0, +24.0] | +0.0 [-12.0, +12.0] |
| `narrative_camera_dirs − narrative` | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | -3.6 [-7.2, -0.8] * | -6.8 [-18.8, +2.0] | +4.0 [+0.0, +12.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] |
| `narrative_frames − narrative` | +0.4 [+0.0, +1.2] | -5.2 [-14.0, +2.4] | +0.4 [-10.4, +12.0] | -7.6 [-18.8, +0.8] | +8.0 [+0.0, +20.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +11.1 [+0.0, +33.3] | -4.0 [-16.0, +8.0] | +0.0 [-16.0, +16.0] |
| `narrative_exact − narrative` | +0.4 [+0.0, +1.2] | -4.4 [-11.2, +0.0] | +2.8 [+0.0, +8.4] | -6.0 [-16.8, +2.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] | +4.0 [+0.0, +12.0] |

### 4. Against video only

| Comparison | overall | metric tasks | direction and route | without metric tasks | lead |
|---|---:|---:|---:|---:|---:|
| `obj_csv − video` | -8.5 [-15.2, -1.6] * | -25.7 [-34.7, -16.8] * | +14.3 [+0.2, +28.6] * | +1.8 [-7.4, +11.3] | overall −, direction and route + |
| `cam_csv − video` | -10.2 [-17.2, -2.9] * | -26.0 [-35.1, -16.9] * | +10.4 [-5.5, +26.4] | -0.6 [-10.5, +9.4] | overall − |
| `obj_text − video` | -11.6 [-18.8, -4.4] * | -23.5 [-33.3, -13.6] * | +10.3 [-7.3, +27.0] | -4.5 [-14.3, +5.5] | overall − |
| `cam_text − video` | -8.8 [-15.7, -1.7] * | -17.9 [-27.6, -8.0] * | +10.5 [-6.0, +26.9] | -3.3 [-12.8, +6.3] | overall − |
| `narrative − video` | -10.7 [-18.2, -3.3] * | -20.9 [-31.3, -10.7] * | +8.5 [-7.8, +24.8] | -4.6 [-14.6, +5.8] | overall − |
| `obj_markdown_table − video` | -6.6 [-13.2, +0.1] | -11.1 [-20.5, -1.9] * | +2.2 [-13.4, +18.1] | -4.0 [-12.9, +5.2] |  |
| `cam_markdown_table − video` | -3.9 [-10.8, +3.0] | -12.1 [-21.5, -2.8] * | +12.3 [-4.0, +28.6] | +1.0 [-8.4, +10.4] |  |
| `cam_csv_frames − video` | -7.4 [-14.5, -0.2] * | -21.2 [-30.9, -11.2] * | +10.3 [-6.0, +26.5] | +0.8 [-9.0, +10.6] | overall − |
| `cam_text_turn_move − video` | -9.3 [-16.3, -2.2] * | -19.2 [-28.9, -9.3] * | +12.3 [-5.3, +30.0] | -3.4 [-12.9, +6.2] | overall − |
| `cam_text_frames − video` | -10.1 [-17.3, -2.8] * | -20.5 [-30.9, -10.0] * | +8.5 [-9.4, +26.2] | -3.9 [-13.8, +6.0] | overall − |
| `narrative_turn_move − video` | -13.0 [-20.5, -5.3] * | -29.7 [-38.5, -20.9] * | +16.4 [-1.5, +34.4] | -3.0 [-13.5, +8.0] | overall − |
| `narrative_camera_dirs − video` | -12.0 [-19.3, -4.5] * | -24.4 [-34.3, -14.5] * | +8.5 [-7.8, +24.8] | -4.5 [-14.6, +5.8] | overall − |
| `narrative_frames − video` | -11.3 [-18.8, -3.5] * | -25.1 [-35.2, -15.2] * | +8.4 [-9.3, +26.1] | -3.0 [-13.6, +8.0] | overall − |
| `narrative_exact − video` | -11.6 [-19.2, -4.0] * | -23.5 [-33.7, -13.5] * | +6.5 [-10.1, +23.0] | -4.5 [-14.9, +6.1] | overall − |

### 5. Metres against relative units

Each run-2 condition minus the same condition in run 1, on the same 200 questions. The metric tasks are the row that matters.

| Comparison | overall | metric tasks | direction and route | without metric tasks | object_abs_distance | object_size_estimation | room_size_estimation | lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `obj_csv: metres − relative` | +0.8 [-3.3, +4.9] | -0.7 [-8.7, +7.2] | +0.0 [-6.2, +6.2] | +1.6 [-2.5, +6.4] | +10.8 [-6.0, +27.2] | +10.0 [-3.2, +22.8] | -22.8 [-34.0, -12.0] * |  |
| `cam_csv: metres − relative` | -0.2 [-3.3, +3.1] | -3.1 [-11.1, +4.8] | +0.0 [+0.0, +0.0] | +1.6 [+0.0, +4.0] | +9.2 [-6.0, +24.4] | +7.6 [-5.6, +20.8] | -26.0 [-39.2, -14.0] * |  |
| `obj_text: metres − relative` | -0.3 [-3.7, +2.9] | +1.1 [-5.3, +7.1] | +0.0 [+0.0, +0.0] | -1.2 [-5.2, +2.6] | +1.6 [-9.6, +11.6] | -8.0 [-21.6, +5.6] | +9.6 [+2.4, +17.2] * |  |
| `cam_text: metres − relative` | +1.4 [-2.1, +5.0] | +5.2 [-1.9, +12.1] | -1.9 [-5.6, +0.0] | -0.8 [-4.7, +3.1] | +13.2 [-0.0, +25.6] | -5.6 [-18.8, +8.4] | +8.0 [-1.2, +18.0] |  |
| `narrative: metres − relative` | +3.9 [+0.6, +7.3] * | +9.1 [+2.0, +16.3] * | -1.9 [-5.6, +0.0] | +0.8 [-2.3, +4.0] | +8.8 [-6.8, +23.6] | +13.2 [+1.6, +26.0] * | +5.2 [-2.4, +14.4] | overall + |
| `obj_markdown_table: metres − relative` | +0.8 [-3.3, +4.8] | +6.7 [-0.3, +13.9] | -6.2 [-14.4, +2.1] | -2.8 [-7.9, +2.1] | +4.0 [-11.2, +18.8] | +14.0 [+2.0, +26.8] * | +2.0 [-6.4, +10.8] |  |
| `cam_markdown_table: metres − relative` | +6.0 [+1.9, +10.0] * | +12.1 [+5.5, +18.7] * | +3.9 [-6.0, +13.7] | +2.3 [-2.5, +7.2] | +4.8 [-8.4, +17.2] | +27.6 [+14.4, +41.2] * | +4.0 [-2.8, +10.8] | overall + |
| `cam_csv_frames: metres − relative` | +2.4 [-1.3, +6.3] | +1.7 [-6.3, +9.7] | +0.0 [+0.0, +0.0] | +2.9 [-0.6, +6.7] | +7.2 [-10.0, +24.0] | +10.8 [-2.8, +24.8] | -12.8 [-23.2, -3.2] * |  |
| `cam_text_turn_move: metres − relative` | +2.9 [-0.3, +6.3] | +5.2 [-1.9, +12.4] | +0.0 [+0.0, +0.0] | +1.6 [-1.6, +4.8] | +12.0 [-1.2, +24.4] | -5.6 [-19.6, +8.8] | +9.2 [+0.8, +19.6] * |  |
| `cam_text_frames: metres − relative` | +2.1 [-1.5, +5.7] | +2.7 [-3.7, +9.1] | -1.9 [-5.6, +0.0] | +1.7 [-2.3, +5.9] | +3.6 [-6.4, +13.2] | +1.2 [-12.0, +15.6] | +3.2 [-5.6, +12.8] |  |
| `narrative_turn_move: metres − relative` | +3.8 [+0.1, +7.4] * | +6.0 [-2.0, +13.9] | +2.0 [+0.0, +6.0] | +2.4 [-0.8, +6.4] | +2.4 [-13.6, +18.0] | +16.0 [+2.8, +29.6] * | -0.4 [-12.0, +9.6] | overall + |
| `narrative_camera_dirs: metres − relative` | +1.9 [-2.1, +5.7] | +4.9 [-3.6, +13.3] | -3.9 [-9.7, +0.0] | +0.1 [-3.7, +3.8] | +8.4 [-8.4, +24.0] | +7.2 [-7.2, +21.6] | -0.8 [-15.2, +12.0] |  |
| `narrative_frames: metres − relative` | +3.0 [-0.6, +6.5] | +4.1 [-4.4, +12.3] | +2.1 [+0.0, +6.2] | +2.4 [+0.0, +5.3] | +2.4 [-13.6, +17.6] | +13.2 [+1.2, +26.0] * | -3.2 [-19.6, +12.0] |  |
| `narrative_exact: metres − relative` | +3.5 [-0.0, +7.0] | +8.0 [+0.1, +15.6] * | -1.9 [-5.6, +0.0] | +0.8 [-2.3, +4.0] | +8.4 [-5.6, +22.0] | +15.6 [+0.0, +31.2] | -0.0 [-11.6, +9.2] |  |

How many answers the metres change, out of the 75 metric-task questions, the 50 direction and route ones and the 75 others:

| Condition | answers changed, metric tasks (75) | answers changed, direction and route (50) | answers changed, other (75) |
|---|---:|---:|---:|
| `obj_csv` | 68 | 2 | 8 |
| `cam_csv` | 67 | 1 | 4 |
| `obj_text` | 62 | 3 | 11 |
| `cam_text` | 59 | 4 | 10 |
| `narrative` | 69 | 3 | 5 |
| `obj_markdown_table` | 65 | 7 | 10 |
| `cam_markdown_table` | 67 | 6 | 7 |
| `cam_csv_frames` | 66 | 1 | 9 |
| `cam_text_turn_move` | 54 | 2 | 6 |
| `cam_text_frames` | 63 | 2 | 7 |
| `narrative_turn_move` | 68 | 2 | 4 |
| `narrative_camera_dirs` | 69 | 5 | 7 |
| `narrative_frames` | 67 | 1 | 2 |
| `narrative_exact` | 67 | 1 | 5 |

### 6. Chain of thought

| Comparison | overall | metric tasks | direction and route | without metric tasks | lead |
|---|---:|---:|---:|---:|---:|
| `cot_cam_markdown_table − cam_markdown_table` | -10.6 [-18.5, -2.8] * | -3.6 [-13.6, +6.4] | -11.7 [-30.3, +6.9] | -14.8 [-25.9, -3.6] * | overall − |
| `cot_cam_text − cam_text` | -11.1 [-18.4, -3.8] * | -4.4 [-15.5, +6.8] | -19.4 [-33.9, -5.1] * | -15.1 [-24.9, -5.7] * | overall −, direction and route − |
| `cot_narrative − narrative` | -10.5 [-17.9, -3.2] * | -9.6 [-19.3, +0.3] | -20.0 [-36.0, -3.8] * | -11.0 [-21.3, -0.9] * | overall −, direction and route − |
| `cot_cam_text − cot_cam_markdown_table` | -5.4 [-11.9, +1.0] | -6.5 [-15.1, +2.1] | -9.6 [-23.5, +4.4] | -4.7 [-14.0, +4.3] |  |
| `cot_narrative − cot_cam_text` | -1.3 [-7.4, +4.9] | -8.3 [-16.3, -0.1] * | -2.5 [-14.4, +9.6] | +2.8 [-5.5, +11.5] |  |
| `cot_narrative − cot_cam_markdown_table` | -6.7 [-13.8, +0.4] | -14.8 [-23.7, -5.9] * | -12.0 [-28.3, +4.4] | -1.9 [-11.8, +7.9] |  |
| `cot_cam_markdown_table − video` | -14.5 [-22.2, -6.8] * | -15.7 [-25.6, -6.0] * | +0.6 [-17.6, +18.6] | -13.7 [-24.6, -3.0] * | overall − |
| `cot_cam_text − video` | -19.9 [-27.1, -12.8] * | -22.3 [-31.9, -12.5] * | -8.9 [-23.4, +5.3] | -18.5 [-28.4, -8.5] * | overall − |
| `cot_narrative − video` | -21.2 [-28.7, -13.6] * | -30.5 [-40.5, -19.7] * | -11.4 [-26.2, +2.8] | -15.6 [-25.9, -5.2] * | overall − |

Per question type:

| Comparison | object_counting | object_abs_distance | object_size_estimation | room_size_estimation | object_rel_distance | object_rel_direction_easy | object_rel_direction_medium | object_rel_direction_hard | route_planning | obj_appearance_order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cot_cam_markdown_table − cam_markdown_table` | -26.4 [-45.6, -7.2] * | +2.8 [-16.8, +22.8] | -4.8 [-22.8, +12.8] | -8.8 [-22.8, +6.0] | -28.0 [-52.0, -4.0] * | +25.0 [-25.0, +62.5] | +0.0 [-50.0, +50.0] | -11.1 [-55.6, +33.3] | -28.0 [-52.0, +0.0] | +4.0 [-24.0, +32.0] |
| `cot_cam_text − cam_text` | -24.8 [-44.8, -4.8] * | -7.6 [-26.0, +10.8] | +10.4 [-9.6, +30.0] | -16.0 [-36.0, +4.0] | -12.0 [-36.0, +12.0] | +12.5 [-25.0, +50.0] | +12.5 [-25.0, +50.0] | -33.3 [-66.7, +0.0] | -36.0 [-56.0, -20.0] * | +0.0 [-24.0, +24.0] |
| `cot_narrative − narrative` | -27.2 [-46.8, -8.0] * | -10.8 [-24.4, +2.8] | -5.6 [-24.4, +14.0] | -12.4 [-30.0, +5.2] | +4.0 [-20.0, +28.0] | -37.5 [-87.5, +12.5] | +0.0 [-37.5, +37.5] | -22.2 [-66.7, +22.2] | -20.0 [-44.0, +4.0] | +8.0 [-16.0, +32.0] |
| `cot_cam_text − cot_cam_markdown_table` | -0.4 [-15.6, +14.4] | -2.8 [-18.4, +13.2] | -4.4 [-15.2, +6.8] | -12.4 [-29.6, +4.0] | +4.0 [-20.0, +28.0] | -12.5 [-37.5, +0.0] | +12.5 [-25.0, +50.0] | -33.3 [-77.8, +11.1] | -8.0 [-28.0, +12.0] | -8.0 [-32.0, +12.0] |
| `cot_narrative − cot_cam_text` | -4.8 [-23.2, +13.2] | -9.6 [-26.8, +8.8] | -7.2 [-17.6, +2.8] | -8.0 [-20.8, +4.4] | +8.0 [-16.0, +28.0] | -50.0 [-87.5, -12.5] * | +0.0 [-37.5, +37.5] | +11.1 [+0.0, +33.3] | +8.0 [-8.0, +24.0] | +16.0 [-4.0, +36.0] |
| `cot_narrative − cot_cam_markdown_table` | -5.2 [-20.0, +9.6] | -12.4 [-31.6, +7.6] | -11.6 [-21.6, -2.8] * | -20.4 [-36.0, -5.2] * | +12.0 [-12.0, +36.0] | -62.5 [-87.5, -25.0] * | +12.5 [-25.0, +50.0] | -22.2 [-66.7, +33.3] | +0.0 [-20.0, +20.0] | +8.0 [-16.0, +32.0] |

## The answer letters

B is the true answer of 33 of the 100 multiple-choice questions, and of 19 of the 50 direction and route ones. The score of always giving one letter:

| Answer | Rel. dist. | Rel. dir. | Route | Order | direction and route |
|---|---:|---:|---:|---:|---:|
| always A | 40.0 | 28.7 | 24.0 | 28.0 | 26.4 |
| always B | 20.0 | 44.4 | 32.0 | 36.0 | 38.2 |
| always C | 20.0 | 19.4 | 24.0 | 20.0 | 21.7 |
| always D | 20.0 | 7.4 | 20.0 | 16.0 | 13.7 |

Each condition on the 100 multiple-choice questions (**no letter**: the rule reads none, as in cut-off CoT replies):

| Condition | answers B % | no letter % | right, truth B % | right, truth not B % | right, all % |
|---|---:|---:|---:|---:|---:|
| `video` | 31.0 | 0.0 | 45.5 | 49.3 | 48.0 |
| `obj_csv` | 49.0 | 0.0 | 72.7 | 40.3 | 51.0 |
| `cam_csv` | 70.0 | 0.0 | 87.9 | 28.4 | 48.0 |
| `obj_text` | 72.0 | 0.0 | 84.8 | 20.9 | 42.0 |
| `cam_text` | 60.0 | 0.0 | 72.7 | 28.4 | 43.0 |
| `narrative` | 64.0 | 0.0 | 78.8 | 23.9 | 42.0 |
| `obj_markdown_table` | 46.0 | 0.0 | 60.6 | 34.3 | 43.0 |
| `cam_markdown_table` | 68.0 | 0.0 | 87.9 | 28.4 | 48.0 |
| `cam_csv_frames` | 75.0 | 0.0 | 90.9 | 25.4 | 47.0 |
| `cam_text_turn_move` | 55.0 | 0.0 | 69.7 | 29.9 | 43.0 |
| `cam_text_frames` | 70.0 | 0.0 | 81.8 | 22.4 | 42.0 |
| `narrative_turn_move` | 64.0 | 0.0 | 78.8 | 28.4 | 45.0 |
| `narrative_camera_dirs` | 62.0 | 0.0 | 75.8 | 25.4 | 42.0 |
| `narrative_frames` | 72.0 | 0.0 | 87.9 | 22.4 | 44.0 |
| `narrative_exact` | 64.0 | 0.0 | 81.8 | 22.4 | 42.0 |
| `cot_cam_markdown_table` | 22.0 | 8.0 | 30.3 | 38.8 | 36.0 |
| `cot_cam_text` | 25.0 | 25.0 | 39.4 | 25.4 | 30.0 |
| `cot_narrative` | 26.0 | 23.0 | 33.3 | 35.8 | 35.0 |

On the 50 direction and route questions (31 whose answer is not B):

| Condition | answers B % | no letter % | right, truth B % | right, truth not B % | right, all % |
|---|---:|---:|---:|---:|---:|
| `video` | 32.0 | 0.0 | 42.1 | 32.3 | 36.0 |
| `obj_csv` | 56.0 | 0.0 | 78.9 | 32.3 | 50.0 |
| `cam_csv` | 80.0 | 0.0 | 89.5 | 19.4 | 46.0 |
| `obj_text` | 76.0 | 0.0 | 89.5 | 19.4 | 46.0 |
| `cam_text` | 72.0 | 0.0 | 89.5 | 19.4 | 46.0 |
| `narrative` | 70.0 | 0.0 | 84.2 | 19.4 | 44.0 |
| `obj_markdown_table` | 50.0 | 0.0 | 57.9 | 25.8 | 38.0 |
| `cam_markdown_table` | 72.0 | 0.0 | 94.7 | 19.4 | 48.0 |
| `cam_csv_frames` | 84.0 | 0.0 | 94.7 | 16.1 | 46.0 |
| `cam_text_turn_move` | 68.0 | 0.0 | 89.5 | 22.6 | 48.0 |
| `cam_text_frames` | 76.0 | 0.0 | 89.5 | 16.1 | 44.0 |
| `narrative_turn_move` | 68.0 | 0.0 | 89.5 | 29.0 | 52.0 |
| `narrative_camera_dirs` | 70.0 | 0.0 | 84.2 | 19.4 | 44.0 |
| `narrative_frames` | 76.0 | 0.0 | 89.5 | 16.1 | 44.0 |
| `narrative_exact` | 68.0 | 0.0 | 84.2 | 16.1 | 42.0 |
| `cot_cam_markdown_table` | 26.0 | 6.0 | 36.8 | 35.5 | 36.0 |
| `cot_cam_text` | 26.0 | 38.0 | 42.1 | 16.1 | 26.0 |
| `cot_narrative` | 28.0 | 34.0 | 31.6 | 19.4 | 24.0 |

Run 1's main conditions on the same 50, for comparison:

| Condition | answers B % | no letter % | right, truth B % | right, truth not B % | right, all % |
|---|---:|---:|---:|---:|---:|
| obj_csv (relative) | 52.0 | 0.0 | 73.7 | 35.5 | 50.0 |
| cam_csv (relative) | 78.0 | 0.0 | 89.5 | 19.4 | 46.0 |
| obj_markdown_table (relative) | 40.0 | 0.0 | 57.9 | 35.5 | 44.0 |
| cam_markdown_table (relative) | 66.0 | 0.0 | 78.9 | 22.6 | 44.0 |
| obj_text (relative) | 70.0 | 0.0 | 89.5 | 19.4 | 46.0 |
| cam_text (relative) | 66.0 | 0.0 | 89.5 | 22.6 | 48.0 |
| narrative (relative) | 66.0 | 0.0 | 84.2 | 22.6 | 46.0 |

How many of the 50 direction and route answers each comparison changes, and which way:

| Comparison | answers changed | to right | from right | to B | from B |
|---|---:|---:|---:|---:|---:|
| `cam_csv − obj_csv` | 12 | 2 | 4 | 12 | 0 |
| `cam_markdown_table − obj_markdown_table` | 16 | 8 | 3 | 13 | 2 |
| `cam_text − obj_text` | 4 | 1 | 1 | 0 | 2 |
| `narrative − cam_text` | 9 | 2 | 3 | 3 | 4 |
| `narrative_turn_move − narrative` | 7 | 4 | 0 | 2 | 3 |
| `cot_cam_markdown_table − cam_markdown_table` | 35 | 10 | 16 | 2 | 25 |
| `cot_cam_text − cam_text` | 32 | 4 | 14 | 0 | 23 |
| `cot_narrative − narrative` | 36 | 5 | 15 | 2 | 23 |

## The numeric answers

Median answer per numeric question type (25 questions each), with the true median in the column's name; run 1's in brackets. Counts are whole objects, distances metres, object sizes centimetres, room sizes square metres.

| Condition | Count (truth 2) | Abs. dist. (truth 1.9) | Obj. size (truth 94) | Room size (truth 20.7) |
|---|---:|---:|---:|---:|
| `video` | 2 | 2.1 | 100 | 20 |
| `obj_csv` | 2 (2) | 1.23 (1.02) | 100 (100) | 1.884 (14.2) |
| `cam_csv` | 2 (2) | 1.23 (1.01) | 100 (100) | 10.2 (13.6) |
| `obj_text` | 2 (3) | 1.13 (1.13) | 116 (82) | 11.2 (11.2) |
| `cam_text` | 2 (2) | 1.3 (1.03) | 158 (96) | 11.6 (12.5) |
| `narrative` | 3 (3) | 1.728 (1.011) | 100 (82) | 10.5 (11.3) |
| `obj_markdown_table` | 2 (3) | 1.37 (1.13) | 84.4 (40) | 14.7 (14.4) |
| `cam_markdown_table` | 2 (2) | 1.3 (1.03) | 84 (40) | 14.8 (14.4) |
| `cam_csv_frames` | 2 (2) | 1.23 (1.01) | 100 (90) | 11.2 (13.5) |
| `cam_text_turn_move` | 2 (2) | 1.2 (1.01) | 186 (96) | 11.5 (12.5) |
| `cam_text_frames` | 2 (2) | 1.23 (1.01) | 116 (47) | 11.5 (13.5) |
| `narrative_turn_move` | 3 (3) | 1.7 (1.04) | 100 (64) | 1.936 (1.3) |
| `narrative_camera_dirs` | 3 (3) | 1.728 (1.03) | 100 (74) | 10.5 (11.2) |
| `narrative_frames` | 3 (3) | 1.478 (1.03) | 100 (66.4) | 10.01 (1.8) |
| `narrative_exact` | 3 (3) | 1.534 (1.03) | 90 (64) | 10.5 (11.3) |
| `cot_cam_markdown_table` | 3 | 1.8 | 84.4 | 12 |
| `cot_cam_text` | 2 | 2.05 | 99.2 | 11.25 |
| `cot_narrative` | 4 | 1.834 | 58.2 | 14.134 |

**Sheet numbers copied as the answer.** A bare answer with 2 or more decimals that equals a number in the condition's sheet, out of 25 questions. Object sizes are asked in centimetres, so a copied size would not show; none has 2 decimals. All 25 room-size sheets have `room` rows, the pipeline's segments for "room": 6–12 per sheet, each a piece of the room 1.4–3.9 m across at its longest (10th–90th percentile). The last column counts the room-size answers that are one of those rows' numbers.

| Condition | Abs. dist., metres | Abs. dist., relative | Room size, metres | Room size, relative | room size: a room row's number, metres |
|---|---:|---:|---:|---:|---:|
| `obj_csv` | 5 | 4 | 10 | 1 | 10 |
| `cam_csv` | 3 | 12 | 3 | 0 | 3 |
| `obj_markdown_table` | 7 | 4 | 0 | 0 | 0 |
| `cam_markdown_table` | 2 | 4 | 0 | 0 | 0 |
| `obj_text` | 4 | 4 | 2 | 0 | 2 |
| `cam_text` | 3 | 3 | 0 | 0 | 0 |
| `narrative` | 18 | 15 | 4 | 0 | 4 |

The narrative gives each object's distance from the camera ("… 1.775 metres away"), not between objects: 17 of its 18 copied absolute-distance answers are one of those.

**Counting and the sheet's count.** The pipeline sheet's count of the asked class equals the true count on 6 of the 25 counting questions; it is higher on 11 and lower on 8 (split, merged or missed objects). Each condition's answers against it (MRA score ×100 in the last column):

| Condition | answer = the sheet's count | answer = truth | right where the sheet is right | right where the sheet is wrong | score |
|---|---:|---:|---:|---:|---:|
| `video` | 6 | 10 | 2 | 8 | 49.6 |
| `obj_csv` | 17 | 10 | 5 | 5 | 46.0 |
| `cam_csv` | 16 | 10 | 5 | 5 | 45.6 |
| `obj_text` | 12 | 11 | 4 | 7 | 50.4 |
| `cam_text` | 12 | 11 | 4 | 7 | 52.0 |
| `narrative` | 15 | 11 | 6 | 5 | 49.6 |
| `obj_markdown_table` | 14 | 10 | 4 | 6 | 49.2 |
| `cam_markdown_table` | 13 | 12 | 5 | 7 | 54.0 |
| `cam_csv_frames` | 11 | 12 | 4 | 8 | 57.2 |
| `cam_text_turn_move` | 11 | 11 | 4 | 7 | 52.0 |
| `cam_text_frames` | 10 | 11 | 3 | 8 | 53.2 |
| `narrative_turn_move` | 16 | 10 | 6 | 4 | 46.0 |
| `narrative_camera_dirs` | 14 | 11 | 6 | 5 | 50.0 |
| `narrative_frames` | 14 | 11 | 6 | 5 | 50.0 |
| `narrative_exact` | 14 | 11 | 6 | 5 | 50.0 |
| `cot_cam_markdown_table` | 16 | 5 | 3 | 2 | 27.6 |
| `cot_cam_text` | 13 | 5 | 3 | 2 | 27.2 |
| `cot_narrative` | 16 | 4 | 3 | 1 | 22.4 |

## Chain of thought: what the replies do

The notebook's reasoning check (`reasoning_check.csv`), over the replies whose sheet has objects (192 of 200): the share that name a sheet row (an id such as `chair_2`), copy an object number as the sheet writes it, or copy a number the sheet gives only for the camera (camera positions in the table; the move lengths in the compass path and the narrative). **Right**: multiple choice right, or a numeric score of at least 0.5.

| Condition, replies | n | names a row | copies an object number | copies a camera number | right |
|---|---:|---:|---:|---:|---:|
| `cot_cam_markdown_table`, all | 192 | 43.8 | 70.8 | 31.2 | 36.5 |
| `cot_cam_markdown_table`, computing tasks | 67 | 61.2 | 91.0 | 40.3 | 38.8 |
| `cot_cam_markdown_table`, direction and route | 50 | 30.0 | 78.0 | 40.0 | 36.0 |
| `cot_cam_markdown_table`, cut off | 19 | 57.9 | 63.2 | 31.6 | 21.1 |
| `cot_cam_text`, all | 192 | 85.9 | 76.6 | 11.5 | 29.7 |
| `cot_cam_text`, computing tasks | 67 | 94.0 | 89.6 | 10.4 | 35.8 |
| `cot_cam_text`, direction and route | 50 | 88.0 | 90.0 | 12.0 | 26.0 |
| `cot_cam_text`, cut off | 44 | 86.4 | 79.5 | 18.2 | 2.3 |
| `cot_narrative`, all | 192 | 70.8 | 38.0 | 13.0 | 28.1 |
| `cot_narrative`, computing tasks | 67 | 83.6 | 46.3 | 13.4 | 34.3 |
| `cot_narrative`, direction and route | 50 | 68.0 | 14.0 | 16.0 | 24.0 |
| `cot_narrative`, cut off | 51 | 66.7 | 29.4 | 25.5 | 3.9 |

Three replies, the head and the tail:

**`cot_cam_markdown_table`, question 974** (object_rel_direction_hard, truth A, 629 tokens): right: it takes the three objects' coordinates from the sheet and compares them.

```text
To determine the position of the TV relative to the sofa when standing by the sofa and facing the table, let's break down the scene and the positions of the objects.

1.  **Identify the objects and their positions:**
    *   **Sofa:** The sofa is located in the corner of the room, near the window. Its center is approximately at coordinates (-1.053, 0.111, 2.591).
    *   **Table:** The table is a small side table next to the sofa, holding a lamp and some items. Its center is approximately at (-0.425, -0.440, 3.853).
    *   **TV:** The large flat-screen TV is on a white stand. Its center is approximately at (1.202, -0.474, 2.726).

2.  **Determine the orientation:**
    *   The question states we are standing by the sofa and facing the table.
    *   The sofa is at (-1.053, 0.111, 2.591) and the table is at (-0.425, -0.440, 3.853).
    *   To go from the sofa to the table, we move in the positive X direction (from -1.053 to -0.425) and in the negative Y direction (from 0.111 to -0.440).
    *   Therefore, if we are at the sofa and face the table, our "forward" direction is roughly towards the positive X and negative Y axis.

3.  **Determine the position of the TV relative to this orientation:**
    *   The TV is at (1.202, -0.474, 2.726).
    *   Let's compare the TV's position to our forward direction (towards the table).
    *   The TV's X-coordinate (1.202) is greater than t
[… 328 characters …]
 negative Y and a less negative X), the TV is located to the right of our forward direction.

4.  **Conclusion:**
    *   The TV is to the right of the table.
    *   Since we are facing the table, the TV is in front of us and to our right.
    *   Therefore, the TV is to our front-right.

Answer: A
```

**`cot_cam_markdown_table`, question 4474** (object_counting, truth 2, 725 tokens): wrong: it counts the pipeline's bed rows, split pieces of 2 beds, as 8 beds.

```text
The question asks for the number of beds in the room. Let's analyze the video and scene information to identify the beds.

1.  **Identify Bed Objects:**
    -   **bed_1:** This object is seen in frames at 4.0, 20.8, 38.0, 55.2, 72.4, 89.7, 106.9, and 124.1. It is consistently present in the room.
    -   **bed_4:** This object is seen in frames at 4.0, 20.8, 38.0, 55.2, 72.4, 89.7, 106.9, and 124.1. It is also consistently present in the room.
    -   **bed_5:** This object is seen in frames at 20.8 and 20.8. It appears to be a different bed, possibly a child's bed, located near the dollhouse.
    -   **bed_6:** This object is seen in frames at 20.8 and 20.8. It is another bed, also appearing to be a child's bed, located near the dollhouse.
    -   **bed_7:** This object is seen in frames at 55.2 and 55.2. It is a bed located near the dresser.
    -   **bed_8:** This object is seen in frames at 72.4 and 72.4. It is a bed located near the dresser.
    -   **bed_10:** This object is seen in frames at 89.7 and 89.7. It is a bed located near the dresser.
    -   **bed_12:** This object is seen in frames at 106.9 and 106.9. It is a bed located near the dresser.

2.  **Analyze Scene Information:**
    -   The scene sheet lists multiple bed objects with different positions and sizes.
    -   The scene sheet also lists multiple room objects (room_1, room_2, room_3, room_4, room_5, room
[… 477 characters …]
*
    -   From the video frames, we can see at least 7 different bed objects (bed_1, bed_4, bed_5, bed_6, bed_7, bed_8, bed_10, bed_12).
    -   The scene sheet lists 8 bed objects (bed_1, bed_4, bed_5, bed_6, bed_7, bed_8, bed_10, bed_12).
    -   Therefore, there are 8 beds in the room.

Answer: 8
```

**`cot_cam_markdown_table`, question 2659** (object_counting, truth 2, 2048 tokens): cut off: a run of mm:ss times to the limit; the rule reads the last number.

```text
The question asks for the number of monitors in the room. I will identify the monitors in the video and scene information.

1.  **Identify the objects:** I need to find objects that are monitors.
    *   In the scene information, there is an entry for `monitor_1` with position `(-1.803, 0.660, 1.426)`.
    *   There is also an entry for `monitor_2` with position `(-2.088, 0.649, 1.452)`.
    *   In the video frames, a computer monitor is visible on the desk in several shots (e.g., at 00:19, 00:20, 00:21, 00:22, 00:23, 00:24, 00:25, 00:26, 00:27, 00:28, 00:29, 00:30, 00:31, 00:32, 00:33, 00:34, 00:35, 00:36, 00:37, 00:38, 00:39, 00:40, 00:41, 00:42, 00:43, 00:44, 00:45, 00:46, 00:47, 00:48, 00:49, 00:50, 00:51, 00:52, 00:53, 00:54, 00:55, 00:56, 00:57, 00:58, 00:59, 01:00, 01:01, 01:02, 01:03, 01:04, 01:05, 01:06, 01:07, 01:08, 01:09, 01:10, 01:11, 01:12, 01:13, 01:14, 01:15, 01:16, 01:17, 01:18, 01:19, 01:20, 01:21, 01:22, 01:23, 01:24, 01:25, 01:26, 01:27, 01:28, 01:29, 01:30, 01:31, 01:32, 01:33, 01:34, 01:35, 01:36, 01:37, 01:38, 01:39, 01:40, 01:41, 01:42, 01:43, 01:44, 01:45, 01:46, 01:47, 01:48, 01:49, 01:50, 01:51, 01:52, 01:53, 01:54, 01:55, 01:56, 01:57, 01:58, 01:59, 02:00, 02:01, 02:02, 02:03, 02:04, 02:05, 02:06, 02:07, 02:08, 02:09, 02:10, 02:11, 02:12, 02:13, 02:14, 02:15, 02:16, 02:17, 02:18, 02:19, 02:20, 02:21, 02:22, 02:23, 02:24, 02:25, 02:26, 02:27, 02:28, 0
[… 684 characters …]
 04:07, 04:08, 04:09, 04:10, 04:11, 04:12, 04:13, 04:14, 04:15, 04:16, 04:17, 04:18, 04:19, 04:20, 04:21, 04:22, 04:23, 04:24, 04:25, 04:26, 04:27, 04:28, 04:29, 04:30, 04:31, 04:32, 04:33, 04:34, 04:35, 04:36, 04:37, 04:38, 04:39, 04:40, 04:41, 04:42, 04:43, 04:44, 04:45, 04:46, 04:47, 04:48, 04:49
```

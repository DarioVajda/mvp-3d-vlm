# Camera pilot: analysis

<!-- Written by analysis.py in this folder; edit that, not this file. -->

The predictions of `explore_camera.ipynb` (`results/zeroshot/vsibench/camera/explore200/`): 200 questions (`explore200_ids.txt`, 25 per task), Qwen3.5-2B, 16 frames (`uniform128`), direct answers, up to 256 new tokens, final answer scored, batches of 8, pipeline sheets in VGGT's relative units, on one NVIDIA L4 at commit `8f027cb`. `video` was copied from `scot_explore_results/A_video.jsonl` (the structured-CoT pilot's video-only run), which passed the reuse check: the same questions, frames, prompts and batches. Bootstrap intervals: questions resampled within each question type, the same draws for every condition (10,000 resamples, seed 0); they and the scores are recomputed here and match the notebook's tables.

Every reply in every condition was read by the answer rule (parse rate 100%), and none was longer than 10 tokens: the model answered with a letter or a number every time, so the 256-token limit never mattered. The 14 runs and the timing took about 70 minutes on the L4, at 0.27–0.31 ms per input token.

## Scores

×100. **Direction and route**: the mean of relative direction (its three types as one task) and route planning, 50 questions. **Without metric tasks**: the 5 tasks other than absolute distance, object size and room size. Sheet tokens: what the scene information adds to the video-only prompt, on average.

| Condition | Overall | Direction and route | Without metric tasks | Count | Abs. dist. | Obj. size | Room size | Rel. dist. | Rel. dir. | Route | Order | Sheet tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `video` | 48.6 | 35.6 | 48.2 | 49.6 | 39.6 | 68.8 | 39.2 | 76.0 | 47.2 | 24.0 | 44.0 | 0.0 |
| `obj_csv` | 39.3 | 49.9 | 48.4 | 42.0 | 20.0 | 26.4 | 26.0 | 64.0 | 63.9 | 36.0 | 36.0 | 605.2 |
| `cam_csv` | 38.6 | 46.0 | 45.9 | 45.6 | 16.4 | 28.8 | 33.6 | 56.0 | 56.0 | 36.0 | 36.0 | 1830.9 |
| `obj_text` | 37.3 | 45.9 | 44.9 | 48.4 | 21.6 | 30.8 | 21.6 | 36.0 | 51.9 | 40.0 | 48.0 | 730.7 |
| `cam_text` | 38.3 | 47.9 | 45.7 | 48.4 | 19.2 | 31.2 | 28.0 | 60.0 | 51.9 | 44.0 | 24.0 | 1331.5 |
| `narrative` | 34.0 | 46.0 | 42.8 | 46.0 | 17.2 | 21.2 | 19.2 | 36.0 | 56.0 | 36.0 | 40.0 | 1230.3 |
| `obj_markdown_table` | 41.2 | 44.0 | 47.0 | 38.8 | 31.2 | 25.2 | 38.0 | 60.0 | 56.0 | 32.0 | 48.0 | 689.7 |
| `cam_markdown_table` | 38.7 | 44.1 | 46.9 | 54.4 | 20.0 | 17.6 | 37.2 | 60.0 | 48.1 | 40.0 | 32.0 | 2154.4 |
| `cam_csv_frames` | 38.7 | 45.9 | 46.1 | 54.8 | 19.6 | 27.2 | 32.0 | 44.0 | 51.9 | 40.0 | 40.0 | 1801.4 |
| `cam_text_turn_move` | 36.3 | 47.9 | 43.2 | 48.0 | 19.6 | 31.2 | 23.6 | 48.0 | 51.9 | 44.0 | 24.0 | 1343.1 |
| `cam_text_frames` | 36.3 | 45.9 | 42.5 | 48.8 | 21.6 | 26.8 | 29.6 | 36.0 | 51.9 | 40.0 | 36.0 | 1384.6 |
| `narrative_turn_move` | 31.8 | 50.0 | 42.8 | 46.0 | 17.2 | 16.8 | 6.4 | 32.0 | 56.0 | 44.0 | 36.0 | 1242.3 |
| `narrative_camera_dirs` | 34.7 | 48.0 | 43.6 | 46.0 | 17.6 | 23.6 | 18.4 | 36.0 | 56.0 | 40.0 | 40.0 | 1224.2 |
| `narrative_frames` | 34.3 | 41.9 | 42.9 | 46.4 | 18.4 | 21.6 | 20.0 | 44.0 | 51.9 | 32.0 | 40.0 | 1281.9 |
| `narrative_exact` | 33.5 | 44.0 | 42.9 | 46.4 | 13.2 | 21.6 | 18.4 | 36.0 | 56.0 | 32.0 | 44.0 | 1204.9 |

## The comparisons

The README's comparisons, then every condition against `video`. `*`: the interval excludes 0. Lead: the README's reading rule (at least 5 points on direction and route or 2 on the overall, interval excluding 0).

| Comparison | overall | direction and route | without metric tasks | lead |
|---|---:|---:|---:|---:|
| `cam_csv − obj_csv` | -0.7 [-5.2, +3.8] | -3.9 [-14.2, +6.5] | -2.5 [-8.9, +4.1] |  |
| `cam_markdown_table − obj_markdown_table` | -2.5 [-7.0, +2.1] | +0.1 [-12.2, +12.2] | -0.1 [-6.8, +6.6] |  |
| `cam_text − obj_text` | +1.1 [-4.0, +6.3] | +2.0 [+0.0, +6.0] | +0.8 [-6.3, +8.2] |  |
| `narrative − cam_text` | -4.4 [-9.5, +0.7] | -1.9 [-10.0, +6.2] | -2.8 [-10.1, +4.3] |  |
| `narrative − obj_text` | -3.3 [-7.8, +1.1] | +0.1 [-8.0, +8.1] | -2.0 [-7.6, +3.5] |  |
| `narrative − obj_csv` | -5.3 [-11.3, +0.8] | -3.9 [-16.5, +8.6] | -5.6 [-14.1, +2.9] |  |
| `cam_csv_frames − cam_csv` | +0.1 [-3.1, +3.4] | -0.1 [-6.2, +6.0] | +0.2 [-4.6, +4.8] |  |
| `cam_text_turn_move − cam_text` | -2.0 [-4.3, +0.0] | +0.0 [+0.0, +0.0] | -2.5 [-5.7, +0.6] |  |
| `cam_text_frames − cam_text` | -2.0 [-6.1, +2.1] | -2.0 [-6.0, +0.0] | -3.1 [-9.1, +3.0] |  |
| `narrative_turn_move − narrative` | -2.2 [-4.8, +0.4] | +4.0 [+0.0, +10.0] | +0.0 [-3.2, +3.2] |  |
| `narrative_camera_dirs − narrative` | +0.8 [-0.3, +2.1] | +2.0 [+0.0, +6.0] | +0.8 [+0.0, +2.4] |  |
| `narrative_frames − narrative` | +0.3 [-3.7, +4.3] | -4.1 [-12.2, +3.9] | +0.0 [-5.3, +5.4] |  |
| `narrative_exact − narrative` | -0.5 [-2.8, +1.7] | -2.0 [-6.0, +0.0] | +0.1 [-3.1, +3.2] |  |

| Against video | overall | direction and route | without metric tasks | lead |
|---|---:|---:|---:|---:|
| `obj_csv − video` | -9.3 [-15.7, -2.9] * | +14.3 [+1.9, +28.1] * | +0.2 [-7.9, +8.4] | overall −, direction and route + |
| `cam_csv − video` | -10.0 [-17.3, -2.6] * | +10.4 [-5.5, +26.4] | -2.2 [-12.0, +7.7] | overall − |
| `obj_text − video` | -11.3 [-18.9, -3.7] * | +10.3 [-7.3, +27.0] | -3.3 [-13.6, +7.2] | overall − |
| `cam_text − video` | -10.2 [-17.2, -3.0] * | +12.3 [-5.3, +30.0] | -2.5 [-12.1, +6.9] | overall − |
| `narrative − video` | -14.6 [-22.2, -6.9] * | +10.4 [-6.1, +26.8] | -5.4 [-15.7, +5.5] | overall − |
| `obj_markdown_table − video` | -7.4 [-13.9, -0.8] * | +8.4 [-5.5, +22.4] | -1.2 [-9.6, +7.5] | overall − |
| `cam_markdown_table − video` | -9.9 [-16.9, -2.7] * | +8.5 [-7.8, +24.8] | -1.3 [-10.9, +8.5] | overall − |
| `cam_csv_frames − video` | -9.9 [-17.2, -2.4] * | +10.3 [-6.0, +26.5] | -2.0 [-11.9, +8.0] | overall − |
| `cam_text_turn_move − video` | -12.3 [-19.3, -5.1] * | +12.3 [-5.3, +30.0] | -5.0 [-14.4, +4.6] | overall − |
| `cam_text_frames − video` | -12.2 [-19.2, -5.3] * | +10.3 [-7.6, +28.3] | -5.6 [-15.2, +4.1] | overall − |
| `narrative_turn_move − video` | -16.8 [-24.3, -9.1] * | +14.4 [-3.5, +32.0] | -5.4 [-15.9, +5.5] | overall − |
| `narrative_camera_dirs − video` | -13.9 [-21.6, -6.0] * | +12.4 [-5.3, +29.1] | -4.6 [-14.9, +6.4] | overall − |
| `narrative_frames − video` | -14.3 [-21.9, -6.4] * | +6.3 [-11.0, +24.0] | -5.3 [-15.8, +5.4] | overall − |
| `narrative_exact − video` | -15.1 [-22.7, -7.3] * | +8.4 [-9.5, +26.0] | -5.3 [-15.8, +5.6] | overall − |

Per question type (25 questions each; relative direction 8, 8 and 9):

| Comparison | object_counting | object_abs_distance | object_size_estimation | room_size_estimation | object_rel_distance | object_rel_direction_easy | object_rel_direction_medium | object_rel_direction_hard | route_planning | obj_appearance_order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cam_csv − obj_csv` | +3.6 [+0.0, +10.8] | -3.6 [-10.8, +4.0] | +2.4 [-7.6, +13.2] | +7.6 [+1.6, +14.4] * | -8.0 [-20.0, +0.0] | -25.0 [-75.0, +25.0] | +12.5 [+0.0, +37.5] | -11.1 [-33.3, +0.0] | +0.0 [-12.0, +12.0] | +0.0 [-20.0, +20.0] |
| `cam_markdown_table − obj_markdown_table` | +15.6 [+3.6, +30.0] * | -11.2 [-20.8, -2.8] * | -7.6 [-18.4, +0.8] | -0.8 [-5.2, +3.6] | +0.0 [+0.0, +0.0] | -12.5 [-50.0, +25.0] | +0.0 [-37.5, +37.5] | -11.1 [-33.3, +0.0] | +8.0 [-8.0, +24.0] | -16.0 [-36.0, +0.0] |
| `cam_text − obj_text` | +0.0 [-16.8, +16.8] | -2.4 [-16.0, +10.8] | +0.4 [-10.8, +9.6] | +6.4 [+0.4, +13.6] * | +24.0 [+4.0, +44.0] * | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +4.0 [+0.0, +12.0] | -24.0 [-48.0, +0.0] |
| `narrative − cam_text` | -2.4 [-19.2, +14.0] | -2.0 [-13.6, +9.2] | -10.0 [-21.6, +0.8] | -8.8 [-20.4, +1.2] | -24.0 [-44.0, -4.0] * | +0.0 [+0.0, +0.0] | +12.5 [+0.0, +37.5] | +0.0 [+0.0, +0.0] | -8.0 [-24.0, +8.0] | +16.0 [+0.0, +36.0] |
| `narrative − obj_text` | -2.4 [-12.0, +5.6] | -4.4 [-18.8, +10.0] | -9.6 [-22.4, +2.0] | -2.4 [-15.6, +8.4] | +0.0 [-12.0, +12.0] | +0.0 [+0.0, +0.0] | +12.5 [+0.0, +37.5] | +0.0 [+0.0, +0.0] | -4.0 [-16.0, +8.0] | -8.0 [-28.0, +12.0] |
| `narrative − obj_csv` | +4.0 [-15.2, +22.8] | -2.8 [-14.4, +10.4] | -5.2 [-18.4, +7.6] | -6.8 [-22.0, +8.0] | -28.0 [-48.0, -12.0] * | -25.0 [-75.0, +25.0] | +12.5 [-25.0, +50.0] | -11.1 [-33.3, +0.0] | +0.0 [-16.0, +16.0] | +4.0 [-16.0, +24.0] |
| `cam_csv_frames − cam_csv` | +9.2 [-4.4, +24.0] | +3.2 [-2.8, +11.6] | -1.6 [-10.0, +6.8] | -1.6 [-4.4, +1.6] | -12.0 [-24.0, +0.0] | +0.0 [+0.0, +0.0] | -12.5 [-37.5, +0.0] | +0.0 [+0.0, +0.0] | +4.0 [+0.0, +12.0] | +4.0 [+0.0, +12.0] |
| `cam_text_turn_move − cam_text` | -0.4 [-12.0, +10.8] | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | -4.4 [-10.8, +0.4] | -12.0 [-24.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| `cam_text_frames − cam_text` | +0.4 [-13.6, +14.4] | +2.4 [-5.6, +12.0] | -4.4 [-11.6, +0.0] | +1.6 [-5.6, +9.6] | -24.0 [-44.0, -4.0] * | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] | +12.0 [-4.0, +28.0] |
| `narrative_turn_move − narrative` | +0.0 [+0.0, +0.0] | +0.0 [-2.0, +1.6] | -4.4 [-13.2, +4.8] | -12.8 [-25.6, -2.8] * | -4.0 [-12.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +8.0 [+0.0, +20.0] | -4.0 [-12.0, +0.0] |
| `narrative_camera_dirs − narrative` | +0.0 [+0.0, +0.0] | +0.4 [-1.6, +2.4] | +2.4 [+0.0, +7.2] | -0.8 [-4.4, +2.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +4.0 [+0.0, +12.0] | +0.0 [+0.0, +0.0] |
| `narrative_frames − narrative` | +0.4 [-10.4, +10.8] | +1.2 [-10.4, +12.8] | +0.4 [-8.0, +9.6] | +0.8 [-10.4, +12.8] | +8.0 [+0.0, +20.0] | +0.0 [+0.0, +0.0] | -12.5 [-37.5, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-16.0, +8.0] | +0.0 [-16.0, +16.0] |
| `narrative_exact − narrative` | +0.4 [+0.0, +1.2] | -4.0 [-12.0, +2.0] | +0.4 [-6.0, +7.2] | -0.8 [-2.8, +0.8] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -4.0 [-12.0, +0.0] | +4.0 [-8.0, +16.0] |

## The answer letters

On the 100 multiple-choice questions, B is the true answer of 33, and on the 50 direction and route questions of 19. The sheets shift the model's answers to B: the objects-only tables least, the camera tables, the sentences and the narrative most. A condition can gain on a task from that alone; the constant answers' scores show how much:

| Answer | Rel. dist. | Rel. dir. | Route | Order | direction and route |
|---|---:|---:|---:|---:|---:|
| always A | 40.0 | 28.7 | 24.0 | 28.0 | 26.4 |
| always B | 20.0 | 44.4 | 32.0 | 36.0 | 38.2 |
| always C | 20.0 | 19.4 | 24.0 | 20.0 | 21.7 |
| always D | 20.0 | 7.4 | 20.0 | 16.0 | 13.7 |

Each condition's answers on the 100 multiple-choice questions:

| Condition | answers B % | right, truth B % | right, truth not B % | right, all % |
|---|---:|---:|---:|---:|
| `video` | 31.0 | 45.5 | 49.3 | 48.0 |
| `obj_csv` | 45.0 | 63.6 | 43.3 | 50.0 |
| `cam_csv` | 70.0 | 84.8 | 26.9 | 46.0 |
| `obj_text` | 70.0 | 87.9 | 22.4 | 44.0 |
| `cam_text` | 50.0 | 66.7 | 34.3 | 45.0 |
| `narrative` | 64.0 | 78.8 | 23.9 | 42.0 |
| `obj_markdown_table` | 39.0 | 60.6 | 43.3 | 49.0 |
| `cam_markdown_table` | 64.0 | 75.8 | 29.9 | 45.0 |
| `cam_csv_frames` | 76.0 | 90.9 | 20.9 | 44.0 |
| `cam_text_turn_move` | 56.0 | 66.7 | 29.9 | 42.0 |
| `cam_text_frames` | 69.0 | 78.8 | 22.4 | 41.0 |
| `narrative_turn_move` | 66.0 | 78.8 | 23.9 | 42.0 |
| `narrative_camera_dirs` | 61.0 | 78.8 | 25.4 | 43.0 |
| `narrative_frames` | 74.0 | 87.9 | 19.4 | 42.0 |
| `narrative_exact` | 68.0 | 84.8 | 20.9 | 42.0 |

And on the 50 direction and route questions (31 whose answer is not B):

| Condition | answers B % | right, truth B % | right, truth not B % | right, all % |
|---|---:|---:|---:|---:|
| `video` | 32.0 | 42.1 | 32.3 | 36.0 |
| `obj_csv` | 52.0 | 73.7 | 35.5 | 50.0 |
| `cam_csv` | 78.0 | 89.5 | 19.4 | 46.0 |
| `obj_text` | 70.0 | 89.5 | 19.4 | 46.0 |
| `cam_text` | 66.0 | 89.5 | 22.6 | 48.0 |
| `narrative` | 66.0 | 84.2 | 22.6 | 46.0 |
| `obj_markdown_table` | 40.0 | 57.9 | 35.5 | 44.0 |
| `cam_markdown_table` | 66.0 | 78.9 | 22.6 | 44.0 |
| `cam_csv_frames` | 82.0 | 94.7 | 16.1 | 46.0 |
| `cam_text_turn_move` | 68.0 | 89.5 | 22.6 | 48.0 |
| `cam_text_frames` | 74.0 | 89.5 | 19.4 | 46.0 |
| `narrative_turn_move` | 68.0 | 89.5 | 25.8 | 50.0 |
| `narrative_camera_dirs` | 62.0 | 84.2 | 25.8 | 48.0 |
| `narrative_frames` | 78.0 | 89.5 | 12.9 | 42.0 |
| `narrative_exact` | 68.0 | 84.2 | 19.4 | 44.0 |

How many of the 50 direction and route answers each comparison changes: to the right answer, away from it, and to B:

| Comparison | answers changed | to right | from right | to B |
|---|---:|---:|---:|---:|
| `cam_csv − obj_csv` | 13 | 3 | 5 | 13 |
| `cam_markdown_table − obj_markdown_table` | 16 | 5 | 5 | 14 |
| `cam_text − obj_text` | 7 | 1 | 0 | 2 |
| `narrative − cam_text` | 8 | 2 | 3 | 2 |
| `narrative − obj_text` | 9 | 2 | 2 | 3 |
| `narrative − obj_csv` | 17 | 5 | 7 | 11 |
| `cam_csv_frames − cam_csv` | 6 | 1 | 1 | 4 |
| `cam_text_turn_move − cam_text` | 1 | 0 | 0 | 1 |
| `cam_text_frames − cam_text` | 4 | 0 | 1 | 4 |
| `narrative_turn_move − narrative` | 3 | 2 | 0 | 1 |
| `narrative_camera_dirs − narrative` | 2 | 1 | 0 | 0 |
| `narrative_frames − narrative` | 6 | 1 | 3 | 6 |
| `narrative_exact − narrative` | 3 | 0 | 1 | 1 |
| `obj_csv − video` | 20 | 10 | 3 | 14 |
| `narrative − video` | 26 | 13 | 8 | 19 |

## The numeric answers

Median answer per numeric question type (25 questions each), with the true median in the column's name. The sheets' relative units are read as metres: one sheet unit is about 2.3 m (`metric_scale.md` in the format experiment's folder).

| Condition | Count (truth 2) | Abs. dist. (truth 1.9) | Obj. size (truth 94) | Room size (truth 20.7) |
|---|---:|---:|---:|---:|
| `video` | 2.0 | 2.1 | 100.0 | 20.0 |
| `obj_csv` | 2.0 | 1.02 | 100.0 | 14.2 |
| `cam_csv` | 2.0 | 1.01 | 100.0 | 13.6 |
| `obj_text` | 3.0 | 1.13 | 82.0 | 11.2 |
| `cam_text` | 2.0 | 1.03 | 96.0 | 12.5 |
| `narrative` | 3.0 | 1.01 | 82.0 | 11.3 |
| `obj_markdown_table` | 3.0 | 1.13 | 40.0 | 14.4 |
| `cam_markdown_table` | 2.0 | 1.03 | 40.0 | 14.4 |
| `cam_csv_frames` | 2.0 | 1.01 | 90.0 | 13.5 |
| `cam_text_turn_move` | 2.0 | 1.01 | 96.0 | 12.5 |
| `cam_text_frames` | 2.0 | 1.01 | 47.0 | 13.5 |
| `narrative_turn_move` | 3.0 | 1.04 | 64.0 | 1.3 |
| `narrative_camera_dirs` | 3.0 | 1.03 | 74.0 | 11.2 |
| `narrative_frames` | 3.0 | 1.03 | 66.4 | 1.8 |
| `narrative_exact` | 3.0 | 1.03 | 64.0 | 11.3 |

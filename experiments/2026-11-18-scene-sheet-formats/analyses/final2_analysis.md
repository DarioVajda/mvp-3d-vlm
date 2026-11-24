## COMPLETENESS

A_video: 798 lines, 798 distinct, = batch2_A_ids: True, missing scores 0; settings ['Qwen/Qwen3.5-2B', 16, 'uniform128', 2048, True, 798]; sheet None
pipeline_v4: 798 lines, 798 distinct, = batch2_A_ids: True, missing scores 0; settings ['Qwen/Qwen3.5-2B', 16, 'uniform128', 2048, True, 798]; sheet {'source': 'pipeline', 'format': 'csv', 'root': '/content/mvp-3d-vlm/data/scene_sheets_pipeline_v4'}
same frames in both conditions: True
model: 15852e8c16360a2fea060d615a32b45270f8a8fc | snapshots after: ['15852e8c16360a2fea060d615a32b45270f8a8fc']

## REGISTERED RESULTS (recomputed)

H1 primary, 309 distance: helped 53, hurt 89, unchanged 167; sign test p = 0.003172 -> NOT CONFIRMED
H2, 110 direction: helped 20, hurt 14, unchanged 76; p = 0.391528, Holm 0.783057 -> NOT SUPPORTED
H3, full method overall: video 41.19, full 41.46, diff +0.28 [-2.42, +2.97], bootstrap p 0.8612, Holm 0.8612 -> NOT SUPPORTED

notebook's csv:
                          test   n  helped  hurt        p   holm p       verdict  difference   95% low  95% high
             primary: distance 309    53.0  89.0 0.003172      NaN NOT CONFIRMED         NaN       NaN       NaN
 secondary: relative direction 110    20.0  14.0 0.391528 0.783057 NOT SUPPORTED         NaN       NaN       NaN
secondary: full method overall 798     NaN   NaN 0.861200 0.861200 NOT SUPPORTED    0.275563 -2.419419  2.966832

## EXPLORATORY (not registered)

                       type   n  A_video  pipeline_v4  full method answer used v4 vs video h/h
       obj_appearance_order  68     44.1         38.2         44.1       video            8/12
        object_abs_distance 150     31.8         25.7         25.7          v4           35/69
            object_counting  88     30.0         30.2         30.0       video           21/23
  object_rel_direction_easy  24     58.3         62.5         62.5          v4             4/3
  object_rel_direction_hard  38     34.2         50.0         50.0          v4             9/3
object_rel_direction_medium  48     50.0         47.9         47.9          v4             7/8
        object_rel_distance 159     45.3         44.0         44.0          v4           18/20
     object_size_estimation 157     55.5         31.3         55.5       video           26/90
       room_size_estimation  38     43.2         34.7         43.2       video           14/17
             route_planning  28     32.1         35.7         35.7          v4             3/2

object_abs_distance, with relations (80): video 31.9, v4 29.8, helped/hurt 25/37
object_abs_distance, without relations (70): video 31.7, v4 21.1, helped/hurt 10/32
object_rel_distance, with relations (25): video 52.0, v4 32.0, helped/hurt 2/7
object_rel_distance, without relations (134): video 44.0, v4 46.3, helped/hurt 16/13
relative direction, with relations (45): video 48.9, v4 57.8, helped/hurt 11/7
relative direction, without relations (65): video 44.6, v4 47.7, helped/hurt 9/7

absolute distance relations (80): MRA vs truth 29.6; pred/truth median 0.99 (quartiles 0.52-1.50); within 25%: 21; copied 80/80
relative distance relations (25): closest right 8/25; copied 25/25
direction relations (45): mapped to an option 45; right 26/45; copied 39/45
scale components (median over scenes): median 2.71 | priors/median 1.0 | room/median 1.0

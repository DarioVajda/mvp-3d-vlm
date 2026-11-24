## (1) Per task (200 questions)

| task | n | video | pipeline_v3 | pipeline_v3 helped/hurt | pipeline_v3 sign p (Holm) |
|---|---|---|---|---|---|
| obj_appearance_order | 25 | 44.0 | 36.0 | 3/5 | 0.727 (1.00) |
| object_abs_distance | 25 | 39.6 | 29.6 | 7/12 | 0.359 (1.00) |
| object_counting | 25 | 49.6 | 40.4 | 5/10 | 0.302 (1.00) |
| object_rel_direction | 25 | 48.0 | 64.0 | 7/3 | 0.344 (1.00) |
| object_rel_distance | 25 | 76.0 | 68.0 | 2/4 | 0.688 (1.00) |
| object_size_estimation | 25 | 68.8 | 53.2 | 4/16 | 0.012 (0.09) |
| room_size_estimation | 25 | 39.2 | 34.4 | 6/13 | 0.167 (1.00) |
| route_planning | 25 | 24.0 | 36.0 | 5/2 | 0.453 (1.00) |
| overall (VSI-Bench) | 200 | 48.6 | 45.2 | 39/65 | 0.014 |

## (2) By whether the v3 sheet has every object the question names (question text and sheet only)

| task | group | n | video | pipeline_v3 | v3 - video | helped/hurt | sign p |
|---|---|---|---|---|---|---|---|
| obj_appearance_order | all present | 11 | 45.5 | 54.5 | 9.1 | 2/1 | 1.0 |
| obj_appearance_order | some missing | 14 | 42.9 | 21.4 | -21.4 | 1/4 | 0.4 |
| object_abs_distance | all present | 13 | 36.9 | 29.2 | -7.7 | 5/6 | 1.0 |
| object_abs_distance | some missing | 12 | 42.5 | 30.0 | -12.5 | 2/6 | 0.3 |
| object_counting | all present | 23 | 49.1 | 43.0 | -6.1 | 5/9 | 0.4 |
| object_counting | some missing | 2 | 55.0 | 10.0 | -45.0 | 0/1 | 1.0 |
| object_rel_direction | all present | 13 | 53.8 | 76.9 | 23.1 | 4/1 | 0.4 |
| object_rel_direction | some missing | 12 | 41.7 | 50.0 | 8.3 | 3/2 | 1.0 |
| object_rel_distance | all present | 7 | 57.1 | 71.4 | 14.3 | 2/1 | 1.0 |
| object_rel_distance | some missing | 18 | 83.3 | 66.7 | -16.7 | 0/3 | 0.2 |
| object_size_estimation | all present | 19 | 73.2 | 53.7 | -19.5 | 4/15 | 0.0 |
| object_size_estimation | some missing | 6 | 55.0 | 51.7 | -3.3 | 0/1 | 1.0 |
| room_size_estimation | no named objects | 25 | 39.2 | 34.4 | -4.8 | 6/13 | 0.2 |
| route_planning | all present | 14 | 14.3 | 35.7 | 21.4 | 3/0 | 0.2 |
| route_planning | some missing | 11 | 36.4 | 36.4 | 0.0 | 2/2 | 1.0 |
| all tasks | all present | 100 | 48.0 | 49.9 | 1.9 | 25/33 | 0.4 |
| all tasks | some missing | 75 | 52.7 | 42.5 | -10.1 | 8/19 | 0.1 |
| all tasks | no named objects | 25 | 39.2 | 34.4 | -4.8 | 6/13 | 0.2 |

## (3) Selective rules, from the existing predictions (IN-SAMPLE: chosen on these 200)

types where v3 beats video on these 200: ['object_rel_direction', 'route_planning'] 

| rule | uses v3 on | overall | minus video | helped/hurt vs video | note |
|---|---|---|---|---|---|
| video only | 0 | 48.6 | 0.0 | 0/0 |  |
| pipeline_v3 always | 200 | 45.2 | -3.4 | 39/65 |  |
| v3 for the types where it helps | 50 | 52.1 | 3.6 | 12/5 | types chosen on these 200 |
| v3 when every named object is in its sheet | 100 | 49.6 | 1.1 | 25/33 | fixed rule; room size -> video |
|   same, room size -> v3 | 125 | 49.0 | 0.5 | 31/46 |  |
| v3 for helping types AND all present | 27 | 51.7 | 3.1 | 7/1 | types chosen on these 200 |
| best of the two per question | 39 | 63.6 | 15.0 | 39/0 | UPPER BOUND, needs the answer |

split-half estimate (rule chosen on a random half, scored on the other half, 2,000 splits): minus video, median [2.5%, 97.5%]
  types                  -0.3 [-7.3, +4.4]
  present                +1.2 [-4.1, +6.4]
  types and present      +1.2 [-3.5, +4.7]

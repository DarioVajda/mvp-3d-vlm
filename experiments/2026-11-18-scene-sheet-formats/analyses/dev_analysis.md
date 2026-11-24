## (1) Per task, dev200

| task | n | video | v3 | v3_fixed | v3 helped/hurt | v3 sign p (Holm) | v3_fixed helped/hurt | v3_fixed sign p (Holm) | fixed vs v3 helped/hurt |
|---|---|---|---|---|---|---|---|---|---|
| obj_appearance_order | 25 | 48.0 | 44.0 | 44.0 | 3/4 | 1.000 (1.00) | 4/5 | 1.000 (1.00) | 1/1 |
| object_abs_distance | 25 | 43.2 | 31.2 | 39.6 | 9/12 | 0.664 (1.00) | 11/9 | 0.824 (1.00) | 7/0 |
| object_counting | 25 | 35.2 | 35.6 | 33.2 | 7/5 | 0.774 (1.00) | 7/8 | 1.000 (1.00) | 0/3 |
| object_rel_direction | 25 | 31.5 | 43.5 | 31.9 | 4/1 | 0.375 (1.00) | 4/4 | 1.000 (1.00) | 1/4 |
| object_rel_distance | 25 | 48.0 | 36.0 | 36.0 | 3/6 | 0.508 (1.00) | 4/7 | 0.549 (1.00) | 1/1 |
| object_size_estimation | 25 | 56.8 | 37.2 | 41.2 | 5/13 | 0.096 (0.67) | 4/10 | 0.180 (1.00) | 11/7 |
| room_size_estimation | 25 | 45.2 | 25.6 | 29.6 | 6/15 | 0.078 (0.63) | 6/15 | 0.078 (0.63) | 3/0 |
| route_planning | 25 | 28.0 | 36.0 | 36.0 | 4/2 | 0.688 (1.00) | 4/2 | 0.688 (1.00) | 0/0 |
| overall (VSI-Bench) | 200 | 42.0 | 36.1 | 36.4 | 41/58 | 0.107 | 44/60 | 0.141 | 24/16 |

## (2) The rule fixed on explore200, tested unchanged on dev200 (stratified paired bootstrap, 10,000 resamples)

| condition | uses sheet on | overall | minus video | 95% CI of minus video | P(minus video > 0) | helped/hurt | sign p |
|---|---|---|---|---|---|---|---|
| video only | 0 | 42.0 | 0.0 |  |  | 0/0 |  |
| pipeline_v3 always | 200 | 36.1 | -5.8 | [-12.5, +1.0] | 0.05 | 41/58 | 0.107 |
| pipeline_v3_fixed always | 200 | 36.4 | -5.5 | [-12.5, +1.5] | 0.06 | 44/60 | 0.141 |
| RULE (frozen on explore200): v3 when all named objects present, else video | 109 | 41.0 | -1.0 | [-6.1, +4.2] | 0.35 | 25/28 | 0.784 |
| same rule with v3_fixed's answers (not the frozen rule) | 109 | 41.7 | -0.3 | [-5.0, +4.7] | 0.45 | 24/24 | 1.000 |

questions where the rule uses v3: 109 of 200 (all named objects present); by task: obj_appearance_order 16, object_abs_distance 12, object_counting 21, object_rel_direction 11, object_rel_distance 13, object_size_estimation 21, room_size_estimation 0, route_planning 15

by coverage (as on explore200):

| group | n | video | v3 | v3_fixed | v3 helped/hurt |
|---|---|---|---|---|---|
| all present | 109 | 41.2 | 39.3 | 40.6 | 25/28 |
| some missing | 66 | 42.3 | 35.2 | 32.1 | 10/15 |
| no named objects | 25 | 45.2 | 25.6 | 29.6 | 6/15 |

## (3) The Not detected line: v3_fixed's answers against v3's, measured against video's answer

- questions with the Not detected line, object-size questions left out (only this fix applies): 62 questions; v3_fixed's answer differs from v3's in 22: now video's answer 1, was video's answer before 12, numeric and nearer video's 3; answers equal to video's: v3 32, v3_fixed 21; score v3 34.2 -> v3_fixed 31.0 (video 40.0); fixed vs v3 helped/hurt 8/9
- object-size questions (no sizes; and the line where an object is missing): 25 questions; v3_fixed's answer differs from v3's in 21: now video's answer 8, was video's answer before 0, numeric and nearer video's 17; answers equal to video's: v3 2, v3_fixed 10; score v3 37.2 -> v3_fixed 41.2 (video 56.8); fixed vs v3 helped/hurt 11/7
- questions with every named object present (neither fix applies, apart from object size): 88 questions; v3_fixed's answer differs from v3's in 2: now video's answer 0, was video's answer before 0, numeric and nearer video's 1; answers equal to video's: v3 42, v3_fixed 42; score v3 40.3 -> v3_fixed 40.9 (video 38.4); fixed vs v3 helped/hurt 2/0

## (4) The task rule from explore200, tested unchanged on dev200

| rule | uses sheet on | overall | minus video | 95% CI | P(> 0) | helped/hurt | sign p |
|---|---|---|---|---|---|---|---|
| v3 for relative direction and route planning (chosen on explore200), else video | 50 | 44.5 | 2.5 | [-0.5, +5.5] | 0.95 | 8/3 | 0.227 |
| v3_fixed for relative direction and route planning (chosen on explore200), else video | 50 | 43.0 | 1.1 | [-2.5, +4.6] | 0.73 | 8/6 | 0.791 |
